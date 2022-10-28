# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Long-context model pretraining with fast blocksparse and extrapolation attentions
"""
from typing import Optional

import logging
import math

import torch
import torch.nn as nn
from fairseq import utils, modules
from fairseq.utils import safe_getattr
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel, TransformerConfig
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.transformer.transformer_config import TransformerConfig
import copy

from fairseq.models.roberta import RobertaEncoder

logger = logging.getLogger(__name__)


@register_model("loco")
class LOCOModel(TransformerModel):
    __jit_unused_properties__ = ["supported_targets"]


    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        generator_architecture(args)
        self.generator = RobertaEncoder(args, encoder.dictionary)

        if not self.args.train_generator:
            for p in self.generator.parameters():
                p.requires_grad = False

        self.pad_idx = self.encoder.dictionary.pad()
        self.bos = self.encoder.dictionary.bos()

        self.sentinel_start_idx = self.encoder.dictionary.index("<sentinel_0>")
        self.sentinel_end_idx = len(self.encoder.dictionary) - 1

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()
        if hasattr(self.encoder, "dictionary"):
            self.eos: int = self.encoder.dictionary.eos()

    @staticmethod
    def add_args(parser):
        super(LOCOModel, LOCOModel).add_args(parser)
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--finetune",
            action="store_true",
            help="different forwards used for pretraining and finetuning"
        )
        parser.add_argument(
            "--train-generator",
            action="store_true",
        )
        parser.add_argument(
            "--generator-xformer-config",
            type=str,
            metavar="D",
        )
        parser.add_argument(
            "--generator-layers",
            type=int,
        )
        
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        # use vanilla attention for now
        args.use_xformers = False # HACK, disable efficient attentions for cross-attention & decoder-side attention
        return super().build_decoder(
            TransformerConfig.from_namespace(args), tgt_dict, embed_tokens
        )


    @classmethod
    def build_generator(cls, args, src_dict, embed_tokens):

        return super().build_encoder(
            TransformerConfig.from_namespace(args), src_dict, embed_tokens
        )

    @property
    def supported_targets(self):
        return {"self"}

    def _sum_spans(self, input, starts, ends):
        """
        sum the span values to the start of each span;
        zero out all other positions
        """
        starts_before = torch.roll(starts, -1, -1)
        input_cumsum = torch.cumsum(input, dim=-1)

        input_cumsum[starts.bool()] = input_cumsum[ends.bool()] - input_cumsum[starts_before.bool()]
        sumed = input_cumsum * starts
        return sumed

    def _avg_spans(self, span_sum, span_lens, starts):
        span_sum[starts.bool()] = span_sum[starts.bool()] / span_lens[starts.bool()]
        return span_sum

    """
    utils from T5's objective
    """
    def _create_sentinels(self, mask_indices):
        """
        mask_indices: binary mask
        start spans as sentinel ids and other masked positions as -1
        """
        start_indices = mask_indices - torch.roll(mask_indices, 1, -1) * mask_indices
        start_indices[:,0] = mask_indices[:,0]

        sentinel_ids = torch.where(start_indices != 0, torch.cumsum(start_indices, dim=-1), start_indices)
        assert sentinel_ids.max() + self.sentinel_start_idx - 1 <= self.sentinel_end_idx, (sentinel_ids.max() + self.sentinel_start_idx - 1, self.sentinel_end_idx, sentinel_ids)

        sentinel_ids = torch.where(sentinel_ids != 0, (sentinel_ids + self.sentinel_start_idx - 1), 0)
        sentinel_ids -= mask_indices - start_indices
        return sentinel_ids

    def _build_inputs(self, masked_input, span_mask):
        sentinel_ids = self._create_sentinels(span_mask)
        masked_input = torch.where(sentinel_ids != 0, sentinel_ids, masked_input)
        src_lens = (masked_input >= 0).sum(-1)

        # src_tokens padded to max_source_positions, useful for blocksparse attention
        src_tokens = masked_input.new_full((masked_input.size(0), self.cfg.max_source_positions), self.pad_idx)
        fill_indices = torch.arange(masked_input.size(-1)).to(masked_input)

        fill_indices = fill_indices < src_lens.unsqueeze(-1)
        assert fill_indices.sum() == (masked_input >= 0).sum() # = 0 for sequence starts
        src_tokens[:,:masked_input.size(-1)][fill_indices] = masked_input[masked_input >= 0]

        return src_tokens, src_lens
    
    def _build_targets(self, masked_target, span_mask, pad_mask, eos_mask):
        """
        masked_targets: masked positions as their original token ids and 
        other positions as pad index
        eos_mask: end of sequence as 0
        pad_mask: padding positions as 0
        """

        unmasked_positions = ~span_mask.bool()
        unmasked_positions[:,0] = 0

        sentinel_ids = self._create_sentinels(unmasked_positions.to(masked_target))
        sentinel_ids *= pad_mask
        sentinel_ids *= eos_mask

        # target: masked positions with sentinel ids or -1;
        # bos, eos and padding positions with value 1
        target = torch.where(sentinel_ids != 0, sentinel_ids, masked_target)
        target[~eos_mask] = self.eos
        target[:,0] = self.bos
        tgt_lens = (target.abs() != 1).sum(-1)

        tgt_tokens = target.new_full(target.size(), self.pad_idx)
        fill_indices = torch.arange(tgt_tokens.size(-1)).to(tgt_tokens)

        fill_indices = fill_indices < tgt_lens.unsqueeze(-1)
        tgt_tokens[fill_indices] = target[target.abs() != 1]
        tgt_tokens = tgt_tokens[:,:tgt_lens.max()]
        
        # truncating if needed
        if tgt_tokens.size(-1) > self.args.max_target_positions:
            end_positions = (tgt_tokens == self.eos).nonzero(as_tuple=True)[1]

            sample_exceeds = end_positions >= (self.args.max_target_positions - 1)
            tgt_tokens = torch.cat(
                [tgt_tokens[:,:self.args.max_target_positions-1], tgt_tokens[:,-1:]], dim=-1
            )
            tgt_tokens[:,-1] = torch.where(sample_exceeds, self.eos, tgt_tokens[:,-1])
        
        decoder_input = tgt_tokens.clone()
        decoder_input[:,0] = self.eos
        decoder_input[:,1:] = tgt_tokens[:,:-1]

        return tgt_tokens, decoder_input

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens: Optional[torch.Tensor] = None,
        features_only: bool = False,
        classification_head_name: Optional[str] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = True,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        masked_unfiltered: Optional[torch.Tensor] = None,
    ):
        if classification_head_name is not None:
            features_only = True

        if not self.cfg.finetune:
            """
            use an encoder-only model to build long-range objectives
            """
            masked_tokens_unfiltered = masked_unfiltered.ne(self.pad_idx).to(src_tokens) # 1: the masked tokens (before hard sampling)
            src_tokens_for_mlm = copy.deepcopy(src_tokens)

            if self.cfg.train_generator:
                masked_logits = self.generator(
                    src_tokens_for_mlm,
                    src_lengths=src_lengths,
                    return_all_hiddens=return_all_hiddens,
                )[0]

                token_loss = modules.cross_entropy(
                        masked_logits.view(-1, masked_logits.size(-1)),
                        masked_unfiltered.view(-1),
                        reduction='none',
                        ignore_index=self.pad_idx,
                    ).view(masked_unfiltered.size())

                masked_cnt = masked_tokens_unfiltered.sum()
                mlm_loss = token_loss.sum() / masked_cnt

            else:
                with torch.no_grad():
                    masked_logits = self.generator(
                        src_tokens,
                        src_lengths=src_lengths,
                        return_all_hiddens=return_all_hiddens,
                    )[0]

                    token_loss = modules.cross_entropy(
                        masked_logits.view(-1, masked_logits.size(-1)),
                        masked_unfiltered.view(-1),
                        reduction='none',
                        ignore_index=self.pad_idx).view(masked_unfiltered.size())
            

            # 1 marking the span starts
            span_starts = masked_tokens_unfiltered - torch.roll(masked_tokens_unfiltered, 1, -1) * masked_tokens_unfiltered
            span_starts[:,0] = masked_tokens_unfiltered[:,0]

            span_ends = masked_tokens_unfiltered - torch.roll(masked_tokens_unfiltered, -1, -1) * masked_tokens_unfiltered
            span_ends[:,-1] = masked_tokens_unfiltered[:,-1]

            # span_starts: binary mask marking the start of each span
            # span_ends: binary mask marking the end of each span
            # span_lens: span lens calculated put at the starts of each span
            span_loss = self._sum_spans(token_loss, span_starts, span_ends)
            span_lens = self._sum_spans(masked_tokens_unfiltered, span_starts, span_ends)
            span_loss_avg = self._avg_spans(span_loss, span_lens, span_starts)

            # find the hard spans, i.e, topk largest loss
            span_counts = span_starts.sum(-1).min()
            hard_span_starts = span_loss_avg.topk(k=math.floor(span_counts*self.cfg.sample_ratio), dim=-1)[1] # bsz * num_hard_spans
            hard_span_ends = span_lens.gather(1, index=hard_span_starts) + hard_span_starts

            # masking source with only the hard spans
            row_idx = torch.arange(hard_span_starts.size(0)).unsqueeze(1).repeat(1, hard_span_starts.size(1)).to(hard_span_starts)
            hard_mask = span_starts.new_zeros(span_starts.size())
            hard_mask[row_idx.view(-1), hard_span_starts.view(-1)] = 1
            hard_mask[row_idx.view(-1), hard_span_ends.view(-1)] = 1

            hard_mask = (hard_mask.cumsum(dim=-1) % 2) == 1
            hard_mask = hard_mask.type_as(masked_tokens_unfiltered)

            # filter our easy span masks
            mask_off = torch.logical_xor(hard_mask, masked_tokens_unfiltered)
            src_tokens[mask_off] = masked_unfiltered[mask_off]
            
            # build the denoising targets
            src_pad_mask = src_tokens.ne(self.pad_idx)
            src_eos_mask = src_tokens.ne(self.eos)
            src_tokens, src_lengths = self._build_inputs(src_tokens, hard_mask)

            target, prev_output_tokens = self._build_targets(masked_unfiltered, hard_mask, src_pad_mask, src_eos_mask)


        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            token_embeddings=token_embeddings,
            return_all_hiddens=return_all_hiddens
        )

        x, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        eos: int = self.eos
        if classification_head_name is not None:
            sentence_representation = x[
                src_tokens.eq(eos), :
            ].view(x.size(0), -1, x.size(-1))[:, -1, :]
            for k, head in self.classification_heads.items():
                # for torch script only supports iteration
                if k == classification_head_name:
                    x = head(sentence_representation)
                    break

        if not self.cfg.finetune:

            if self.cfg.train_generator:
                return x, target, mlm_loss, extra
            
            return x, target, extra
        return x, extra


    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + "." if name != "" else ""
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        def truncate_emb(key):
            if key in state_dict:
                state_dict[key] = state_dict[key][:-1, :]

        # When finetuning on translation task, remove last row of
        # embedding matrix that corresponds to mask_idx token.
        loaded_dict_size = state_dict["encoder.embed_tokens.weight"].size(0)
        if (
            loaded_dict_size == len(self.encoder.dictionary) + 1
            and "<mask>" not in self.encoder.dictionary
        ):
            truncate_emb("encoder.embed_tokens.weight")
            truncate_emb("decoder.embed_tokens.weight")
            truncate_emb("encoder.output_projection.weight")
            truncate_emb("decoder.output_projection.weight")

        # When continued pretraining on new set of languages for mbart,
        # add extra lang embeddings at the end of embed_tokens.
        # Note: newly added languages are assumed to have been added at the end.
        if self.args.task == "multilingual_denoising" and loaded_dict_size < len(
            self.encoder.dictionary
        ):
            logger.info(
                "Adding extra language embeddings not found in pretrained model for "
                "continued pretraining of MBART on new set of languages."
            )
            loaded_mask_token_embedding = state_dict["encoder.embed_tokens.weight"][
                -1, :
            ]

            num_langids_to_add = len(self.encoder.dictionary) - loaded_dict_size
            embed_dim = state_dict["encoder.embed_tokens.weight"].size(1)

            new_lang_embed_to_add = torch.zeros(num_langids_to_add, embed_dim)
            nn.init.normal_(new_lang_embed_to_add, mean=0, std=embed_dim ** -0.5)
            new_lang_embed_to_add = new_lang_embed_to_add.to(
                dtype=state_dict["encoder.embed_tokens.weight"].dtype,
            )

            state_dict["encoder.embed_tokens.weight"] = torch.cat(
                [
                    state_dict["encoder.embed_tokens.weight"][
                        : loaded_dict_size - 1, :
                    ],
                    new_lang_embed_to_add,
                    loaded_mask_token_embedding.unsqueeze(0),
                ]
            )
            state_dict["decoder.embed_tokens.weight"] = torch.cat(
                [
                    state_dict["decoder.embed_tokens.weight"][
                        : loaded_dict_size - 1, :
                    ],
                    new_lang_embed_to_add,
                    loaded_mask_token_embedding.unsqueeze(0),
                ]
            )

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v


def generator_architecture(args):

    # options to use different sizes of generator models
    args.encoder_layers = safe_getattr(args, "generator_layers", 6)
    args.encoder_embed_dim = safe_getattr(args, "generator_embed_dim", 768)
    args.encoder_ffn_embed_dim = safe_getattr(args, "generator_ffn_embed_dim", 3072)
    args.encoder_attention_heads = safe_getattr(args, "generator_attention_heads", 12)

    args.max_positions = safe_getattr(args, "max_source_positions", 8192)
    
    args.encoder_learned_pos = safe_getattr(args, "generator_learned_pos", True)
    args.encoder_normalize_before = safe_getattr(args, "generator_normalize_before", False)
    args.untie_weights_roberta = safe_getattr(args, "untie_weights_roberta", False)

    # xformers config
    args.use_xformers = safe_getattr(args, "generator_use_xformers", True)

    args.attention_name = safe_getattr(args, "generator_attention_name", 'block_noglobal')
    args.xformer_config = safe_getattr(args, 'generator_xformer_config', '{"block_size": 512}')
    args.pooling_layers = safe_getattr(args, "generator_pooling_layers", 0)

@register_model_architecture("loco", "loco_large")
def loco_large_architecture(args):
    
    args.finetune = safe_getattr(args, "finetune", False)
    args.train_generator = safe_getattr(args, "train_generator", False)

    # # @xwhan is it necessary to put it here? had issues
    # def getattr(args, key, value):
    #     return value

    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.relu_dropout = getattr(args, "relu_dropout", 0.0)
    args.dropout = getattr(args, "dropout", 0.1)

    args.max_target_positions = safe_getattr(args, "max_target_positions", 1024) #hack
    args.max_source_positions = safe_getattr(args, "max_source_positions", 1024)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", True)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)

@register_model_architecture("loco", "loco_base")
def loco_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4 * 768)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    loco_large_architecture(args)

@register_model_architecture("loco", "loco_xlarge")
def loco_xlarge_architecture(args):
    loco_large_architecture(args)
    args.encoder_layers = 24
    args.decoder_layers = 24



