# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from audioop import cross
import logging

from typing import Dict, Optional, Tuple
import torch

from typing import Dict, Optional
from torch import Tensor, nn

from fairseq.modules import (
LayerNorm,
TransformerEncoderLayer,
MultiheadAttention,
)


import torch.nn.functional as F

from fairseq.models.transformer import TransformerEncoder
from functools import partial, reduce
from fairseq.distributed import fsdp_wrap
from inspect import isfunction
from operator import mul
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from .block import BlockTransformerEncoderLayer, BlockSelfAttention

"""
A hacky implementation of simple block attention transformer
"""

class TopDownTransformerEncoder(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.args = args
        
        del self.layers
        layers = [self.build_sw_encoder_layer(args, args.window_size, self.padding_idx) for i in range(args.encoder_n1)]
        layers += [self.build_td_encoder_layer(args, args.window_size, self.padding_idx) for i in range(args.encoder_n3)]
        self.layers = nn.ModuleList(layers)
        self.top_pool = nn.AvgPool1d(32, stride=24)
        self.top_layers = nn.ModuleList([self.build_encoder_layer(args) for i in range(args.encoder_n2)])
        self.n1 = args.encoder_n1
        self.n2 = args.encoder_n2
        self.n3 = args.encoder_n3
        self.num_layers = len(self.layers)

    def build_sw_encoder_layer(self, args, window_size, padding_idx):
        layer = BlockTransformerEncoderLayer(args, window_size, padding_idx)
        checkpoint = args.checkpoint_activations
        if checkpoint:
            offload_to_cpu = self.cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)

        min_params_to_wrap = self.cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def build_td_encoder_layer(self, args, window_size, padding_idx):
        layer = TopDownEncoderLayer(args, window_size, padding_idx)
        checkpoint = args.checkpoint_activations
        if checkpoint:
            offload_to_cpu = self.cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)

        min_params_to_wrap = self.cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
    
    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None, # @xwhan in order to add global mask
    ):
        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings, key_padding_mask
        )

    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None, # @xwhan in order to add global mask
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        if key_padding_mask is None:
            encoder_padding_mask = src_tokens.eq(self.padding_idx)
            key_padding_mask = encoder_padding_mask
        else:
            encoder_padding_mask = key_padding_mask.eq(1) # key_padding_mask might -1 elements

        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # bottom up
        for layer in self.layers[:self.n1]:
            x = layer(
                # x, encoder_padding_mask=encoder_padding_mask if has_pads else None
                x, encoder_padding_mask=key_padding_mask # always pass key_padding_mask
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        # Pool and compute top level
        top_x = x.transpose(0, 2) # T x B x C -> C x B x T
        # Can multiple with weights here later for weighted pooling
        top_x = self.top_pool(top_x)
        top_x = top_x.transpose(0, 2) # T_pool x B x C
        for layer in self.top_layers:
            top_x = layer(top_x,encoder_padding_mask=None)

        # Top Down layers with cross attention
        for layer in self.layers[self.n1:]:
            x = layer(
                # x, encoder_padding_mask=encoder_padding_mask if has_pads else None
                x, top_x, encoder_padding_mask=key_padding_mask # always pass key_padding_mask
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = src_tokens.ne(self.padding_idx).sum(dim=1, dtype=torch.int32).reshape(-1, 1).contiguous()
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }

class TopDownEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args, window_size, padding_idx):
        super().__init__(args)

        # replace self-attn 
        self.window_size = window_size
        self.padding_idx = padding_idx
        self.self_attn  = self.build_sw_self_attention(self.embed_dim, window_size, padding_idx, args)
        # init cross attention 
        self.cross_attn = self.build_self_attention(self.embed_dim, args)
        # init cross layernorm
        self.cross_attn_layer_norm = LayerNorm(self.embed_dim, export=args.export)
    def forward(
        self, 
        x, 
        top_x,
        encoder_padding_mask: Optional[Tensor], 
        attn_mask: Optional[Tensor] = None
    ):
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(
                attn_mask.to(torch.bool),
                -1e8 if x.dtype == torch.float32 else -1e4
            )
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        cross_x, _ = self.cross_attn(
            query=x,
            key=top_x,
            value=top_x,
            need_weights=False,
        )
        x = (cross_x + x)/2  # divide by 2 to maintain similar scale for subsequent pre-trained layers
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x

    def build_sw_self_attention(self, embed_dim, window_size, padding_idx, args):
        return BlockSelfAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            window_size=window_size,
            padding_idx=padding_idx,
        )
    def build_cross_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=False,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            use_xformers=cfg.use_xformers,
            attention_name=cfg.attention_name,
            xformer_config=None if not cfg.use_xformers else cfg.xformer_config,
        )    