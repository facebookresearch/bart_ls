# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple
from fairseq.modules.multihead_attention import MultiheadAttention
import torch

from typing import Dict, Optional
from torch import Tensor, nn

from fairseq.modules import (
TransformerEncoderLayer,
)

import torch.nn.functional as F


class PoolEncoderLayer(TransformerEncoderLayer):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.top_pool = nn.AvgPool1d(18, stride=12, padding=9)
        self.top_pool_mask = nn.AvgPool1d(18, stride=12, padding=9,count_include_pad=False)
        # init top level attention 
        self.pool_attn = self.build_cross_attention(self.embed_dim, cfg)
    
    def forward(
        self, 
        x, 
        encoder_padding_mask: Optional[Tensor], 
        attn_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None # relative position encoding
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
        # Project x to get K, Q 
        k = self.pool_attn.k_proj(x)
        v = self.pool_attn.v_proj(x)
        # account for padding before pooling

        if (encoder_padding_mask is not None) and encoder_padding_mask.any():
            k = k * (1 - encoder_padding_mask.unsqueeze(-1).permute(1, 0 ,2).type_as(k))
            v = v * (1 - encoder_padding_mask.unsqueeze(-1).permute(1, 0 ,2).type_as(v))
        
        # Pool K, Q
        pool_k = self.apply_pool(k)
        pool_v = self.apply_pool(v)

        # breakpoint()
        # Do not attend to pooled padding tokens
        if encoder_padding_mask is not None:
            # pool_mask = self.top_pool(encoder_padding_mask.to(k)).type_as(encoder_padding_mask)
            pool_mask = self.top_pool_mask(encoder_padding_mask.to(k)).floor().type_as(encoder_padding_mask)
        else:
            pool_mask = None

        cross_x, _ = self.pool_attn(
            query=x,
            key=pool_k,
            value=pool_v,
            key_padding_mask=pool_mask,
            need_weights=False,
        )
        # cross_x = torch.nan_to_num(cross_x)

        x = cross_x + x
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

    def apply_pool(self, input):
        out = input.permute(1, 2 ,0) # T x B x C ->  B x C x T
        out = self.top_pool(out)
        return out.permute(2, 0 ,1) # T_pool x B x C


    def build_cross_attention(self, embed_dim, cfg):
        return MultiheadAttentionNoProj(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=False,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )


class TwoLevelEncoderLayer(TransformerEncoderLayer):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.top_pool = nn.AvgPool1d(18, stride=12, padding=9)
        # init top level attention 
        self.pool_attn = self.build_cross_attention(self.embed_dim, cfg)
    
    def forward(
        self, 
        x, 
        encoder_padding_mask: Optional[Tensor], 
        attn_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None # relative position encoding
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
        # Project x to get K, Q 
        k = self.pool_attn.k_proj(x)
        v = self.pool_attn.v_proj(x)
        # account for padding before pooling

        # breakpoint()
        if (encoder_padding_mask is not None) and encoder_padding_mask.any():
            k = k * (1 - encoder_padding_mask.unsqueeze(-1).permute(1, 0 ,2).type_as(k))
            v = v * (1 - encoder_padding_mask.unsqueeze(-1).permute(1, 0 ,2).type_as(v))
        # Pool K, Q
        pool_k = self.apply_pool(k)
        pool_v = self.apply_pool(v)
        # Do not attend to pooled padding tokens
        if encoder_padding_mask is not None:
            pool_mask = self.top_pool(encoder_padding_mask.float()).type_as(encoder_padding_mask)
        else:
            pool_mask = None

        cross_x, _ = self.pool_attn(
            query=x,
            key=pool_k,
            value=pool_v,
            key_padding_mask=pool_mask,
            need_weights=False,
        )
        x = cross_x + x
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

    def apply_pool(self, input):
        out = input.permute(1, 2 ,0) # T x B x C ->  B x C x T
        out = self.top_pool(out)
        return out.permute(2, 0 ,1) # T_pool x B x C


    def build_cross_attention(self, embed_dim, cfg):
        return MultiheadAttentionNoProj(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=False,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

class MultiheadAttentionNoProj(MultiheadAttention):
    """Multi-headed attention where the key, value are assumed to be pre-projected 
        this is done to support pooling between projection and attention calculation
    """
    def forward(
        self,
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
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel
        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, torch.zeros_like(self.k_proj.bias), torch.zeros_like(self.v_proj.bias))),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout_module.p,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training or self.dropout_module.apply_during_inference,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=torch.eye(*self.k_proj.weight.size(), out=torch.empty_like(self.k_proj.weight)),
                v_proj_weight=torch.eye(*self.v_proj.weight.size(), out=torch.empty_like(self.v_proj.weight)),
            )
