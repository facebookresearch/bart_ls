# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from fairseq.models import register_model, register_model_architecture
from fairseq.models.roberta import (
    init_bert_params,
    roberta_base_architecture,
    RobertaEncoder,
    RobertaModel,
)

import math
from typing import Dict, Optional
from torch import Tensor, nn

from fairseq.modules import (
LayerDropModuleList,
TransformerEncoderLayer,
MultiheadAttention,
PositionalEmbedding,
)

from fairseq.modules.quant_noise import quant_noise

import torch.nn.functional as F

from fairseq.models.transformer import TransformerEncoder
from .utils import sliding_chunks_matmul_pv, sliding_chunks_matmul_qk


logger = logging.getLogger(__name__)

@register_model("sliding_window_roberta")
class SlidingWindownModel(RobertaModel):
    @staticmethod
    def add_args(parser):
        RobertaModel.add_args(parser)

        parser.add_argument(
            "--attention-window", type=int,
        )
        parser.add_argument(
            "--train-global", 
            action="store_true",
            help="Whether to set CLS as global token during pre-training"
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        from omegaconf import OmegaConf

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, False)

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = SlidingWindowEncoder(args, task.source_dictionary)

        if OmegaConf.is_config(args):
            OmegaConf.set_struct(args, True)

        return cls(args, encoder)

def safe_getattr(obj, k, default=None):
    from omegaconf import OmegaConf

    if OmegaConf.is_config(obj):
        return obj[k] if k in obj and obj[k] is not None else default

    return getattr(obj, k, default)

@register_model_architecture("sliding_window_roberta", "sliding_window_base")
def base_architecture(args):
    args.attention_window = safe_getattr(args, "attention_window", 1024) # equavalent to 512 in longformer
    args.train_global = safe_getattr(args, "train_global", False)
    roberta_base_architecture(args)

@register_model_architecture("sliding_window_roberta", "sliding_window_large")
def large_architecture(args):
    args.encoder_layers = safe_getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 16)
    base_architecture(args)

class SlidingWindowEncoder(RobertaEncoder):

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = SWTransformerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder
    
    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, masked_tokens=None, **unused):

        # pad src_tokens to multiplier of attention window size
        _, seqlen = src_tokens.size()
        w = max(self.sentence_encoder.window_per_layer) * 2
        padding_len = (w - seqlen % w) % w

        x, extra = self.extract_features(
            src_tokens, return_all_hiddens=return_all_hiddens, key_padding_mask=unused.get("key_padding_mask", None)
        )

        if masked_tokens is not None:
            masked_tokens = F.pad(masked_tokens, (0, padding_len), value=False)

        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens)
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **kwargs):
        encoder_out = self.sentence_encoder(
            src_tokens,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=kwargs.get("token_embeddings", None), key_padding_mask=kwargs.get("key_padding_mask", None),
        )
        # T x B x C -> B x T x C
        features = encoder_out["encoder_out"][0].transpose(0, 1)
        inner_states = encoder_out["encoder_states"] if return_all_hiddens else None
        return features, {"inner_states": inner_states}

class SWTransformerEncoder(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = torch.nn.ModuleList([])

        self.window_per_layer = [int(args.attention_window[i // (self.num_layers // len(args.attention_window))]) for i in range(self.num_layers)]
        
        self.layers.extend(
            [self.build_sw_encoder_layer(args, self.window_per_layer[i], self.padding_idx) for i in range(args.encoder_layers)])
        
        self.num_layers = len(self.layers)

    def build_sw_encoder_layer(self, args, window_size, padding_idx):
        return SWTransformerEncoderLayer(args, window_size, padding_idx)

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None, # @xwhan in order to add global mask
    ):
        # sliding-window attention sequence length requirements
        _, seqlen = src_tokens.size()
        w = max(self.window_per_layer) * 2
        padding_len = (w - seqlen % w) % w
        src_tokens = F.pad(src_tokens, (0, padding_len), value=self.padding_idx)

        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, padding_len), value=1)

        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings, key_padding_mask
        )


class SWTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args, window_size, padding_idx):
        super().__init__(args)

        # replace self-attn 
        self.window_size = window_size
        self.padding_idx = padding_idx
        self.self_attn  = self.build_sw_self_attention(self.embed_dim, window_size, padding_idx, args)

    def forward(
        self, 
        x, 
        encoder_padding_mask: Optional[Tensor], 
        attn_mask: Optional[Tensor] = None
    ):
        if attn_mask is not None:
            attn_mask = (attn_mask * -1e8).type_as(attn_mask) # -1 in attn_mask means global attention
        return super().forward(x, encoder_padding_mask, attn_mask=attn_mask)

    def build_sw_self_attention(self, embed_dim, window_size, padding_idx, args):
        return SWSelfAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            window_size=window_size,
            padding_idx=padding_idx,
            max_source_positions=args.max_source_positions,
            train_global=args.train_global,
        )     

class SWSelfAttention(MultiheadAttention):

    def __init__(self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        window_size=256,
        padding_idx=1,
        max_source_positions=1024,
        train_global=False
        ):
        super().__init__(embed_dim, num_heads, kdim, vdim, dropout,
                         bias, add_bias_kv, add_zero_attn, self_attention,
                         encoder_decoder_attention, q_noise, qn_block_size)
        self.attention_window = window_size
        self.train_global = train_global
        self.padding_idx = padding_idx

        self.k_proj_global = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj_global = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj_global = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

    def forward(self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False
        ):

        if need_head_weights:
            need_weights = True

        # attn_mark None
        # key padding mask 0,1 bool tensor 1 means masked position (bsz x seqlen), -1 means global attention

        if key_padding_mask is None:
            seqlen, bsz, _= query.size()
            key_padding_mask = query.new_zeros(bsz, seqlen).bool()

        if attn_mask is None:
            # bos token as global attention
            attn_mask = key_padding_mask.type_as(query)
            num_global_masks = attn_mask.eq(-1).sum()
            #TODO whether to use first token as global token at pretraining time
            if self.train_global:
                if num_global_masks == 0:
                    attn_mask[:,0] = -1

            attn_mask = (attn_mask * -1e8).type_as(query)

        if len(attn_mask.size()) > 2:
            attention_mask = attn_mask.squeeze(dim=2).squeeze(dim=1)
        else:
            attention_mask = attn_mask
        key_padding_mask = attention_mask < 0
        extra_attention_mask = attention_mask > 0
        remove_from_windowed_attention_mask = attention_mask != 0

        # num of global tokens
        num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
        max_num_extra_indices_per_batch = num_extra_indices_per_batch.max()

        if max_num_extra_indices_per_batch <= 0:
            extra_attention_mask = None
        else:
            extra_attention_mask_nonzeros = extra_attention_mask.nonzero(as_tuple=True)
            zero_to_max_range = torch.arange(0, max_num_extra_indices_per_batch, device=extra_attention_mask.device)
            # mask indicating which values are actually going to be padding
            num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
            selection_padding_mask = zero_to_max_range < num_extra_indices_per_batch.unsqueeze(dim=-1)
            # 2) location of the non-padding values in the selected global attention
            selection_padding_mask_nonzeros = selection_padding_mask.nonzero(as_tuple=True)
            # 3) location of the padding values in the selected global attention
            selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True)

        seq_len, bsz, embed_dim = query.size()

        assert self.self_attention
        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = sliding_chunks_matmul_qk(q, k, self.attention_window, padding_value=0) # bsz, seq_len, num_heads, 2 * w + 1

        if remove_from_windowed_attention_mask is not None:
            # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
            # from (bsz x seq_len) to (bsz x seq_len x num_heads x hidden_size)
            remove_from_windowed_attention_mask = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            # cast to float/half then replace 1's with -inf
            float_mask = remove_from_windowed_attention_mask.type_as(q).masked_fill(remove_from_windowed_attention_mask, -10000.0)
            # repeat_size = 1 if isinstance(self.attention_dilation, int) else len(self.attention_dilation)
            repeat_size = 1
            float_mask = float_mask.repeat(1, 1, repeat_size, 1)
            ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
            # diagonal mask with zeros everywhere and -inf inplace of padding
            d_mask = sliding_chunks_matmul_qk(ones, float_mask, self.attention_window, padding_value=0)

            attn_weights += d_mask
        assert list(attn_weights.size()) == [bsz, seq_len, self.num_heads, self.attention_window * 2 + 1]

        # the extra attention
        if extra_attention_mask is not None:
            selected_k = k.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_k[selection_padding_mask_nonzeros] = k[extra_attention_mask_nonzeros]
            # (bsz, seq_len, num_heads, max_num_extra_indices_per_batch)
            selected_attn_weights = torch.einsum('blhd,bshd->blhs', (q, selected_k))
            selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = -10000
            # concat to attn_weights
            # (bsz, seq_len, num_heads, extra attention count + 2*window+1)
            attn_weights = torch.cat((selected_attn_weights, attn_weights), dim=-1)

        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # use fp32 for numerical stability

        if key_padding_mask is not None:
            # softmax sometimes inserts NaN if all positions are masked, replace them with 0
            attn_weights_float = torch.masked_fill(attn_weights_float, key_padding_mask.unsqueeze(-1).unsqueeze(-1),
                                                   0.0)
        attn_weights = attn_weights_float.type_as(attn_weights)
        #attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
        attn_probs = self.dropout_module(attn_weights)
        v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        attn = 0

        if extra_attention_mask is not None:
            selected_attn_probs = attn_probs.narrow(-1, 0, max_num_extra_indices_per_batch)
            selected_v = v.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_v[selection_padding_mask_nonzeros] = v[extra_attention_mask_nonzeros]
            # use `matmul` because `einsum` crashes sometimes with fp16
            # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
            attn += torch.matmul(selected_attn_probs.transpose(1, 2), selected_v.transpose(1, 2).type_as(selected_attn_probs)).transpose(1, 2)
            attn_probs = attn_probs.narrow(-1, max_num_extra_indices_per_batch, attn_probs.size(-1) - max_num_extra_indices_per_batch).contiguous()

        attn += sliding_chunks_matmul_pv(attn_probs, v, self.attention_window)

        attn = attn.type_as(query)
        assert list(attn.size()) == [bsz, seq_len, self.num_heads, self.head_dim]
        attn = attn.transpose(0, 1).reshape(seq_len, bsz, embed_dim).contiguous()

        if extra_attention_mask is not None:
            selected_hidden_states = query.new_zeros(max_num_extra_indices_per_batch, bsz, embed_dim)
            selected_hidden_states[selection_padding_mask_nonzeros[::-1]] = query[extra_attention_mask_nonzeros[::-1]]

            q = self.q_proj_global(selected_hidden_states)
            k = self.k_proj_global(query)
            v = self.v_proj_global(query)
            q /= math.sqrt(self.head_dim)

            q = q.contiguous().view(max_num_extra_indices_per_batch, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # (bsz*self.num_heads, max_num_extra_indices_per_batch, head_dim)
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # bsz * self.num_heads, seq_len, head_dim)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            assert list(attn_weights.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len]

            attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            attn_weights[selection_padding_mask_zeros[0], :, selection_padding_mask_zeros[1], :] = -10000.0
            if key_padding_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    -10000.0,
                )
            attn_weights = attn_weights.view(bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len)
            attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_weights = attn_weights_float.type_as(attn_weights)  # use fp32 for numerical stability
            # attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
            attn_probs = self.dropout_module(attn_weights)
            selected_attn = torch.bmm(attn_probs, v)
            assert list(selected_attn.size()) == [bsz * self.num_heads, max_num_extra_indices_per_batch, self.head_dim]

            selected_attn_4d = selected_attn.view(bsz, self.num_heads, max_num_extra_indices_per_batch, self.head_dim)
            nonzero_selected_attn = selected_attn_4d[selection_padding_mask_nonzeros[0], :, selection_padding_mask_nonzeros[1]]
            attn[extra_attention_mask_nonzeros[::-1]] = nonzero_selected_attn.view(len(selection_padding_mask_nonzeros[0]), -1).type_as(query)

        # context_layer = attn # seqlen x bsz x embed_dim
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            if extra_attention_mask is not None:
                attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            else:
                attn_weights = attn_weights.permute(0, 2, 1, 3) #bsz x head x seqlen x head_dim

        return attn, attn_weights