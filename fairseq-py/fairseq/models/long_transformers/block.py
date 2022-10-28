# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from typing import Dict, Optional, Tuple
import torch

from typing import Dict, Optional
from torch import Tensor, nn

from fairseq.modules import (
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

"""
A hacky implementation of simple block attention transformer
"""

class BlockTransformerEncoder(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.args = args

        del self.layers
        self.layers = nn.ModuleList([self.build_sw_encoder_layer(args, args.window_size, self.padding_idx) for i in range(args.encoder_layers)])
        
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

class BlockTransformerEncoderLayer(TransformerEncoderLayer):
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
        attn_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
    ):
        if attn_mask is not None:
            attn_mask = (attn_mask * -1e8).type_as(attn_mask) # -1 in attn_mask means global attention
        return super().forward(x, encoder_padding_mask, attn_mask=attn_mask)

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


class BlockSelfAttention(MultiheadAttention):


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
        window_size=1024,
        padding_idx=1,
        ):
        super().__init__(embed_dim, num_heads, kdim, vdim, dropout,
                         bias, add_bias_kv, add_zero_attn, self_attention,
                         encoder_decoder_attention, q_noise, qn_block_size)
        self.window_size = window_size
        self.padding_idx = padding_idx
        self.drop_attn = self.dropout_module

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
        need_head_weights: bool = False,
        attn_bias: Optional[Tensor] = None,
        ):

        seq_len, bsz, embed_dim = query.size()

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == seq_len

        assert self.self_attention
        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        q = (
            q.contiguous()
            .view(seq_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )


        y = self.block_attn_forward(q, k, v, key_padding_mask=key_padding_mask)
        assert list(y.size()) == [bsz * self.num_heads, seq_len, self.head_dim], (y.size(), query.size(), q.size())
        y = y.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim)
        y = self.out_proj(y)

        return y, None

    def block_attn_forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        att_mask: Optional[torch.Tensor] = None, 
        key_padding_mask: Optional[Tensor] = None,
        *args, **kwargs
    ):

        bh = q.size(0)
        orig_seq_len = q.size(1)
        bsz = bh // self.num_heads
        head_dim = q.size(-1)

        assert key_padding_mask is not None
        key_padding_mask = key_padding_mask.to(q)
        key_padding_mask[:,0] = -1


        # pad the input length to factors of bucket size
        def _pad_to_window_size(x, window_size):
            seq_len = x.size(-2)
            pad_len = (window_size - seq_len % window_size) % window_size
            return F.pad(x, (0,0,0,pad_len), value=0), pad_len
        q, _ = _pad_to_window_size(q, self.window_size)
        k, _ = _pad_to_window_size(k, self.window_size)
        v, _ = _pad_to_window_size(v, self.window_size)

        if key_padding_mask.shape[1] % self.window_size != 0:
            pad_len = (self.window_size - key_padding_mask.shape[1] % self.window_size) % self.window_size
            # key padding mask: 1 means padding tokens
            key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_ones(key_padding_mask.size(0), pad_len).to(key_padding_mask)], dim=1)
            
        # global attention tokens
        extra_attention_mask = key_padding_mask < 0
        num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
        max_num_extra_indices_per_batch = num_extra_indices_per_batch.max()

        hard_mask = key_padding_mask == 1 

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

        # every token attend to global tokens
        if extra_attention_mask is not None:
            selected_k = k.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, head_dim)
            k_splited = k.view(bsz, self.num_heads, -1, head_dim).transpose(1,2)
            q_splited = q.view(bsz, self.num_heads, -1, head_dim).transpose(1,2)
            v_splited = v.view(bsz, self.num_heads, -1, head_dim).transpose(1,2)

            selected_k[selection_padding_mask_nonzeros] = k_splited[extra_attention_mask_nonzeros]
            # (bsz, seq_len, num_heads, max_num_extra_indices_per_batch)
            selected_attn_weights = torch.einsum('blhd,bshd->blhs', (q_splited, selected_k)) * (head_dim ** -0.5)
            selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = -10000
            attn_weights_over_g_tokens = selected_attn_weights.transpose(1,2)

            selected_v = v.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, head_dim)
            selected_v[selection_padding_mask_nonzeros] = v_splited[extra_attention_mask_nonzeros]
            selected_v = selected_v.transpose(1,2).contiguous().view(bsz*self.num_heads, max_num_extra_indices_per_batch, head_dim)

        tgt_len = k.size(1)
        buckets = q.shape[1] // self.window_size
        b_q = bucket(buckets, q)
        b_k, b_v = map(partial(bucket, buckets), (k, v)) # BH * bct * n_b * D

        dots = torch.einsum('buie,buje->buij', b_q, b_k) * (head_dim ** -0.5)
        mask_value = -10000

        # # mask
        # if key_padding_mask is not None:
        q_mask = default(key_padding_mask.eq(0), lambda: torch.ones((bsz, tgt_len), device=q.device).bool())

        # 1 means not masking
        kv_mask = q_mask
        mq, mk = bucket(buckets, q_mask), bucket(buckets, kv_mask) # B * bkt * n_b
        expand_head_and_merge_into_batch = lambda x: merge_dims(0, 1, expand_dim(x.unsqueeze(1), 1, self.num_heads))
        mq, mk = map(expand_head_and_merge_into_batch, (mq, mk)) # BH * bkt * n_b
        mask = mq[:, :, :, None] * mk[:, :, None, :]

        dots.masked_fill_(~mask, mask_value)
        del mask
        
        block_attn_weights = dots.view(bsz*self.num_heads, -1, self.window_size)
        
        if extra_attention_mask is not None:
            attn_weights_over_g_tokens  = attn_weights_over_g_tokens.view(bsz*self.num_heads, -1, max_num_extra_indices_per_batch)
            all_attn = torch.cat([block_attn_weights, attn_weights_over_g_tokens], dim=-1)
        else:
            all_attn = block_attn_weights

        all_attn_probs = all_attn.softmax(dim=-1)
        all_attn_probs = self.drop_attn(all_attn_probs)


        C = 0
        # calculate block attention
        block_attn_probs = all_attn_probs[:, :, :block_attn_weights.shape[-1]]
        block_attn_probs = block_attn_probs.view(bsz*self.num_heads, -1, self.window_size, self.window_size)
        C += block_attn_probs.matmul(b_v).view(bsz*self.num_heads, -1, head_dim)
        
        if extra_attention_mask is not None:
            attn_probs_over_g_tokens = all_attn_probs[:,:,-attn_weights_over_g_tokens.shape[-1]:]
            C += attn_probs_over_g_tokens.matmul(selected_v)

            # global tokens to attend all other tokens
            selected_q = q_splited.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, head_dim)
            selected_q[selection_padding_mask_nonzeros] = q_splited[extra_attention_mask_nonzeros]

            g2all_attn_weights = selected_q.transpose(1,2).matmul(k_splited.permute(0,2,3,1)) * (head_dim ** -0.5)
            g2all_attn_weights[selection_padding_mask_zeros[0], :, selection_padding_mask_zeros[1], :] = -10000.0

            if hard_mask is not None:
                g2all_attn_weights = g2all_attn_weights.masked_fill(
                    hard_mask.unsqueeze(1).unsqueeze(2),
                    -10000.0,
                )

            g2all_attn_probs_float = F.softmax(g2all_attn_weights, dim=-1, dtype=torch.float32)
            g2all_attn_probs = self.drop_attn(g2all_attn_probs_float.type_as(g2all_attn_weights))
            g2all_attn = g2all_attn_probs.matmul(v.view(bsz, self.num_heads, -1, head_dim)) # (batch_size, self.num_head, max_num_extra_indices_per_batch, head_dim)

            nonzero_global_attn = g2all_attn[selection_padding_mask_nonzeros[0], :, selection_padding_mask_nonzeros[1]]

            C = C.view(bsz, self.num_heads, -1, head_dim)             
            C[extra_attention_mask_nonzeros[0],:,extra_attention_mask_nonzeros[1]] = nonzero_global_attn
            C = C.view(bsz*self.num_heads, -1, head_dim)

        C = C[:,:orig_seq_len]
        return C


import math

class BlockSWSelfAttention(BlockSelfAttention):

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
        need_head_weights: bool = False,
        attn_bias: Optional[Tensor] = None,
        ):

        seq_len, bsz, embed_dim = query.size()

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == seq_len

        assert self.self_attention
        q = self.q_proj(query)
        k = self.proj_pool(self.k_proj, key)
        v = self.proj_pool(self.v_proj, value)

        q = (
            q.contiguous()
            .view(seq_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )


        y = self.block_attn_forward(q, k, v, key_padding_mask=key_padding_mask)
        assert list(y.size()) == [bsz * self.num_heads, seq_len, self.head_dim], (y.size(), query.size(), q.size())
        y = y.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim)
        y = self.out_proj(y)

        return y, None

    def proj_pool(self,proj,input):
        input = proj(input)
        # Pool
        input = input.permute(1, 2 ,0) # T x B x C ->  B x C x T
        input = F.avg_pool1d(input, kernel_size=18, stride=12, padding=9)
        input = input.permute(2, 0 ,1) # T_pool x B x C
        return input

    def block_attn_forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        att_mask: Optional[torch.Tensor] = None, 
        key_padding_mask: Optional[Tensor] = None,
        *args, **kwargs
    ):

        batch_size = q.shape[0] // self.num_heads
        sequence_length = q.shape[1]

        key_padding_mask = key_padding_mask.to(q)

        Q = q.view(batch_size, self.num_heads, -1, self.head_dim).mul(1./math.sqrt(self.head_dim))
        K = k.view(batch_size, self.num_heads, -1, self.head_dim).transpose(1,2).reshape(batch_size, -1, self.embed_dim)
        V = v.view(batch_size, self.num_heads, -1, self.head_dim).transpose(1,2).reshape(batch_size, -1, self.embed_dim)

        # needs centain sequence length to make the block wise local attention work
        def _pad_to_window_size(x, window_size):
            seq_len = x.size(-2)
            pad_len = (window_size - seq_len % window_size) % window_size
            return F.pad(x, (0,0,0,pad_len), value=0), pad_len
        
        Q, _ = _pad_to_window_size(Q, self.window_size)
        K, _ = _pad_to_window_size(K, self.window_size)
        V, _ = _pad_to_window_size(V, self.window_size)
        
        if key_padding_mask.shape[1] % self.window_size != 0:
            pad_len = (self.window_size - key_padding_mask.shape[1] % self.window_size) % self.window_size
            # key padding mask: 1 means padding tokens
            key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_ones(key_padding_mask.size(0), pad_len).to(key_padding_mask)], dim=1)


        K = self.split_heads(K) # (B, H, seq_len, head_dim)
        V = self.split_heads(V)

        padding_mask = key_padding_mask != 0 # True means masked position
        win_attn_weights = self.sliding_chunks_matmul_qk_v2(Q, K, padding_mask) # bsz x num_heads x seqlen x 2winsize

        all_attn = win_attn_weights.float().softmax(dim=-1).to(win_attn_weights)

        hard_mask = key_padding_mask == 1
        all_attn = all_attn.masked_fill(hard_mask[:,None,:,None], 0)

        all_attn = all_attn.to(q)
        all_attn = self.drop_attn(all_attn)

        win_attn_probs = all_attn[:,:,:,:win_attn_weights.shape[-1]]
        seq_len = win_attn_probs.shape[2]
        win_attn_probs = win_attn_probs.view(batch_size, self.num_heads, seq_len // self.window_size, self.window_size,-1)
        V_tiles = self.get_tiles_v2(V, transpose=False)
        outputs = win_attn_probs.matmul(V_tiles).view(batch_size, self.num_heads, seq_len, self.head_dim)

        # get rid of the padding positions
        outputs = outputs[:,:,:sequence_length].view(-1, sequence_length, self.head_dim)

        return outputs

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_heads, self.head_dim)
        X = X.transpose(1, 2)
        return X

    def sliding_chunks_matmul_qk_v2(self, Q, K, padding_mask):
        bsz, num_heads, seqlen, d_h = Q.shape
        if self.window_size > 0:
            # Q, K: bsz x num_heads x seqlen x d_head
            # padding_mask: bsz x seqlen

            mask_tiles = self.get_tiled_mask_v2(padding_mask)
            K_tiles = self.get_tiles_v2(K, transpose=True)
            Q_tiles = Q.view(bsz, num_heads, seqlen // self.window_size, self.window_size, d_h)
            # bsz x num_heads x seqlen//winsize x winsize x 2winsize
            qk_scores = Q_tiles.matmul(K_tiles)
            qk_scores = qk_scores.masked_fill(mask_tiles, -10000)
            return qk_scores.view(bsz, num_heads, seqlen, -1)
        else:
            qk_scores = torch.sum(Q*K, dim=-1, keepdim=True)
            return qk_scores

    def get_tiled_mask_v2(self, mask):
        # only mask along the key dimension
        bsz, seqlen = mask.shape
        ext_len = max(self.window_size//2, 1)
        mask = F.pad(mask, (ext_len, ext_len), value=True) # (bsz, seq_len + 2*ext_len)
        out_shape = (bsz, seqlen//self.window_size, 2*ext_len + self.window_size)
        in_stride = mask.stride()
        out_stride = (in_stride[0], in_stride[1]*self.window_size, in_stride[1])

        return mask.as_strided(size=out_shape, stride=out_stride)[:, None, :, None, :]

    def get_tiles_v2(self, x, transpose=False):
        if self.window_size <= 0:
            return x

        bsz, n_heads, seqlen, d_h = x.shape
        n_groups = seqlen // self.window_size
        ext_len = max(self.window_size//2, 1)
        x = F.pad(x, (0, 0, ext_len, ext_len), value=0)
        strides = x.stride()
        if transpose:
            out_shape = (bsz, n_heads, n_groups, d_h, 2 * ext_len + self.window_size)
            out_stride = (strides[0], strides[1], self.window_size * strides[2], strides[3], strides[2])
        else:
            out_shape = (bsz, n_heads, n_groups, 2 * ext_len + self.window_size, d_h)
            out_stride = (strides[0], strides[1], self.window_size * strides[2], strides[2], strides[3])
        return torch.as_strided(x, size=out_shape, stride=out_stride)

def expand_dim(t, dim, k):
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)

def default(x, d):
    if x is None:
        return d if not isfunction(d) else d()
    return x

def bucket(buckets, t, dim=1):
    shape = list(t.shape)
    shape[dim:dim+1] = [buckets, -1]
    return t.reshape(*shape)

def unbucket(t, dim=1):
    shape = list(t.shape)
    shape[dim:dim+2] = [-1]
    return t.reshape(*shape)