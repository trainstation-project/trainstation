# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan.models.common.attention import AttentionMasksType, GQAttention
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.deepseek_v3.model import Attention as MLAttention
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)


class GptTrainstationTransformerBlock(TransformerBlock):
    """Transformer block whose attention and FFN types are chosen per layer.

    ``attention`` may be GQA (``GQAttention``) or multi-head latent attention
    (DeepSeek V3 ``Attention``). Exactly one of ``feed_forward`` (dense MLP)
    and ``moe`` is set.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        if (config.feed_forward is None) == (config.moe is None):
            raise ValueError(
                "GptTrainstationTransformerBlock needs exactly one of feed_forward and moe."
            )

        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        *,
        padding_mask: torch.Tensor | None = None,
    ):
        x = x + self.attention(self.attention_norm(x), attention_masks, positions)
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x), padding_mask_T=padding_mask)
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x


class GptTrainstationModel(Decoder):
    """Decoder that mixes every shared layer type in one model.

    It has no released checkpoint. Its purpose is test coverage: a single
    model whose layers interleave GQA and MLA attention with dense MLP and MoE
    (with and without shared experts, softmax and sigmoid routers), so every
    parallelism combination can be exercised against all of them at once.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)

            from torchtitan.models.gpt_trainstation.sharding import (
                set_gpt_trainstation_sharding_config,
            )

            parallelism = config.parallelism
            set_gpt_trainstation_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            nparams, active_nparams = get_nparams_and_active_nparams(model)
            attention_op_flops = 0
            for layer in self.layers:
                attention = layer.attention
                if isinstance(attention, MLAttention.Config):
                    qk_head_dim = (
                        attention.qk_nope_head_dim + attention.qk_rope_head_dim
                    )
                    v_head_dim = attention.v_head_dim
                else:
                    assert isinstance(attention, GQAttention.Config)
                    qk_head_dim = v_head_dim = (
                        attention.head_dim
                        if attention.head_dim is not None
                        else attention.dim // attention.n_heads
                    )
                attention_op_flops += quadratic_attention_flops_per_token(
                    num_heads=attention.n_heads,
                    qk_head_dim=qk_head_dim,
                    v_head_dim=v_head_dim,
                    seq_len=seq_len,
                )
            return nparams, 6 * active_nparams + attention_op_flops
