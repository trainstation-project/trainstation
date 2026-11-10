# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.decoder_sharding import set_decoder_sharding_config
from torchtitan.models.deepseek_v3.model import Attention as MLAttention
from torchtitan.models.deepseek_v3.sharding import set_deepseek_v3_layer_sharding
from torchtitan.models.qwen3.sharding import set_qwen3_layer_sharding

if TYPE_CHECKING:
    from torchtitan.models.gpt_base.model import GptBaseModel


def set_gpt_base_sharding_config(
    config: "GptBaseModel.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on all GptBase sub-configs.

    Each layer reuses the sharding of the model its attention type comes from:
    GQA layers use Qwen3's, MLA layers use DeepSeek V3's. Both handle dense FFN
    and MoE (with or without shared experts) the same way.
    """
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        attention = layer_cfg.attention
        if isinstance(attention, MLAttention.Config):
            set_deepseek_v3_layer_sharding(
                layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep
            )
        elif isinstance(attention, GQAttention.Config):
            set_qwen3_layer_sharding(
                layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep
            )
        else:
            raise TypeError(
                f"Unsupported attention type {type(attention).__name__} in GptBase."
            )
