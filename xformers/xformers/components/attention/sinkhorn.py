# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Optional
from functools import partial, reduce
from inspect import isfunction
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xformers.components.attention import Attention, AttentionConfig, register_attention

@dataclass
class SinkhornSelfAttentionConfig(AttentionConfig):
    block_size: int
    temperature: float
    sinkhorm_iter: int
    num_heads: int
    require_key_mask: bool

@register_attention("sinkhorn", SinkhornSelfAttentionConfig)
class SinkhornAttention(Attention):
    def __init__(
        self, 
        dropout: float, 
        num_heads: int,
        block_size: int = 128, 
        temperature: float = 0.7, 
        sinkhorm_iter: int = 7, 
        *args, **kwargs
    ):
        """
        Sparse Sinkhorn Attention
        https://arxiv.org/abs/2002.11296

        Code largely based on https://github.com/lucidrains/sinkhorn-transformer

        The paper's notation are kept wherever possible

        #TODO only support encoding only settings
        """
        super().__init__()
        self.bucket_size = block_size
        self.sinkhorn_iter = sinkhorm_iter
        self.temperature = temperature
        self.drop_attn = nn.Dropout(dropout)
        self.num_head = num_heads
        self.sort_net = AttentionSortNet(block_size, temperature, sinkhorm_iter)

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        att_mask: Optional[torch.Tensor] = None, 
        key_padding_mask: Optional[Tensor] = None,
        *args, **kwargs
    ):
        # q, k, v: (B * nh, S, hs)
        bh = q.size(0)
        orig_seq_len = q.size(1)
        bsz = bh // self.num_head
        head_dim = q.size(-1)


        assert key_padding_mask is not None
        key_padding_mask = key_padding_mask.to(q)
        key_padding_mask[:,0] = -1

        # pad the input length to factors of bucket size
        def _pad_to_window_size(x, window_size):
            seq_len = x.size(-2)
            pad_len = (window_size - seq_len % window_size) % window_size
            return F.pad(x, (0,0,0,pad_len), value=0), pad_len
        q, _ = _pad_to_window_size(q, self.bucket_size)
        k, _ = _pad_to_window_size(k, self.bucket_size)
        v, _ = _pad_to_window_size(v, self.bucket_size)

        if key_padding_mask.shape[1] % self.bucket_size != 0:
            pad_len = (self.bucket_size - key_padding_mask.shape[1] % self.bucket_size) % self.bucket_size
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
            selected_k = k.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_head, head_dim)
            k_splited = k.view(bsz, self.num_head, -1, head_dim).transpose(1,2)
            q_splited = q.view(bsz, self.num_head, -1, head_dim).transpose(1,2)
            v_splited = v.view(bsz, self.num_head, -1, head_dim).transpose(1,2)

            selected_k[selection_padding_mask_nonzeros] = k_splited[extra_attention_mask_nonzeros]
            # (bsz, seq_len, num_heads, max_num_extra_indices_per_batch)
            selected_attn_weights = torch.einsum('blhd,bshd->blhs', (q_splited, selected_k)) * (head_dim ** -0.5)
            selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = -10000
            attn_weights_over_g_tokens = selected_attn_weights.transpose(1,2)

            selected_v = v.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_head, head_dim)
            selected_v[selection_padding_mask_nonzeros] = v_splited[extra_attention_mask_nonzeros]
            selected_v = selected_v.transpose(1,2).contiguous().view(bsz*self.num_head, max_num_extra_indices_per_batch, head_dim)

        tgt_len = k.size(1)
        buckets = q.shape[1] // self.bucket_size
        b_q = bucket(buckets, q)
        b_k, b_v = map(partial(bucket, buckets), (k, v)) # BH * bct * n_b * D

        R = self.sort_net(q, k)
        R = R.type_as(q).to(q)

        b_k_r = reorder_buckets(b_k, R).reshape(bh, buckets, -1, head_dim) 
        b_v_r = reorder_buckets(b_v, R).reshape(bh, buckets, -1, head_dim) 

        b_k = torch.cat((b_k_r, b_k), dim=2)
        b_v = torch.cat((b_v_r, b_v), dim=2)

        dots = torch.einsum('buie,buje->buij', b_q, b_k) * (head_dim ** -0.5)
        mask_value = -10000

        # # mask
        # if key_padding_mask is not None:
        q_mask = default(key_padding_mask.eq(0), lambda: torch.ones((bsz, tgt_len), device=q.device).bool())

        # 1 means not masking
        kv_mask = q_mask
        mq, mk = bucket(buckets, q_mask), bucket(buckets, kv_mask) # B * bkt * n_b
        expand_head_and_merge_into_batch = lambda x: merge_dims(0, 1, expand_dim(x.unsqueeze(1), 1, self.num_head))
        mq, mk = map(expand_head_and_merge_into_batch, (mq, mk)) # BH * bkt * n_b
        mk_r = batched_index_select(mk, R.abs().argmax(dim=-1))
        mk_r = mk_r.reshape(bh, buckets, -1)
        mk = torch.cat((mk_r, mk), dim=2)
        mask = mq[:, :, :, None] * mk[:, :, None, :]

        dots.masked_fill_(~mask, mask_value)
        del mask
        
        sinkhorn_attn_weights = dots.view(bsz*self.num_head, -1, self.bucket_size*2)

        attn_weights_over_g_tokens  = attn_weights_over_g_tokens.view(bsz*self.num_head, -1, max_num_extra_indices_per_batch)
        all_attn = torch.cat([sinkhorn_attn_weights, attn_weights_over_g_tokens], dim=-1)

        all_attn_probs = all_attn.softmax(dim=-1)
        all_attn_probs = self.drop_attn(all_attn_probs)

        C = 0
        # calculate block attention
        block_attn_probs = all_attn_probs[:, :, :sinkhorn_attn_weights.shape[-1]]
        block_attn_probs = block_attn_probs.view(bsz*self.num_head, -1, self.bucket_size, 2*self.bucket_size)
        C += block_attn_probs.matmul(b_v).view(bsz*self.num_head, -1, head_dim)
        
        if extra_attention_mask is not None:
            attn_probs_over_g_tokens = all_attn_probs[:,:,-attn_weights_over_g_tokens.shape[-1]:]
            C += attn_probs_over_g_tokens.matmul(selected_v)

            # global tokens to attend all other tokens
            selected_q = q_splited.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_head, head_dim)
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
            g2all_attn = g2all_attn_probs.matmul(v.view(bsz, self.num_head, -1, head_dim)) # (batch_size, self.num_head, max_num_extra_indices_per_batch, head_dim)

            nonzero_global_attn = g2all_attn[selection_padding_mask_nonzeros[0], :, selection_padding_mask_nonzeros[1]]

            C = C.view(bsz, self.num_head, -1, head_dim)             
            C[extra_attention_mask_nonzeros[0],:,extra_attention_mask_nonzeros[1]] = nonzero_global_attn
            C = C.view(bsz*self.num_head, -1, head_dim)

        C = C[:,:orig_seq_len]
        return C

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

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

def reorder_buckets(t, r):
    return torch.einsum('buv,bvtd->butd', r, t)

def unbucket(t, dim=1):
    shape = list(t.shape)
    shape[dim:dim+2] = [-1]
    return t.reshape(*shape)

class AttentionSortNet(nn.Module):
    def __init__(self, bucket_size, temperature, sinkhorn_iter):
        super().__init__()
        self.bucket_size = bucket_size
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter

    def forward(self, q, k, topk=1):
        dim = q.size(-1)

        buckets = q.shape[1] // self.bucket_size
        kv_buckets = k.shape[1] // self.bucket_size

        b_q = bucket(buckets, q)
        b_k = bucket(kv_buckets, k)

        sq = b_q.mean(dim=2) # TODO original paper uses sum
        sk = b_k.mean(dim=2) 

        R = torch.einsum('bie,bje->bij', sq, sk).to(q) * (dim ** -0.5)

        return gumbel_sinkhorn(F.relu(R), self.sinkhorn_iter, self.temperature)

def log(t, eps = 1e-6):
    return torch.log(t + eps)

def gumbel_sinkhorn(r, n_iters=8, temperature=0.7):
    r = log(r)
    gumbel = sample_gumbel(r.shape, r.device, r.dtype)
    r = (r + gumbel) / temperature
    return sinkhorn_sorting_operator(r, n_iters)

def sample_gumbel(shape, device, dtype, eps=1e-6):
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(0, 1)
    return -log(-log(u, eps), eps)

def sinkhorn_sorting_operator(r, n_iters=8):
    n = r.shape[1]
    for _ in range(n_iters):
        r = r - torch.logsumexp(r, dim=2, keepdim=True)
        r = r - torch.logsumexp(r, dim=1, keepdim=True)
    return torch.exp(r)
