# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.deepseek_v3.model import Attention as MLAttention
from torchtitan.models.deepseek_v3.moe import DeepSeekV3Router
from torchtitan.models.gpt_trainstation import gpt_trainstation_configs
from torchtitan.models.gpt_trainstation.config_registry import (
    dist_muon_optimizer,
    gpt_trainstation_debugmodel,
)

_SEQ_LEN = 64


class _CausalAttention(nn.Module):
    """Eager causal attention on (T, H, K) inputs.

    FlexAttention has no CPU backward, so the CPU test swaps it out.
    """

    def forward(self, q_THK, k_THK, v_THV, *, scale=None, **kwargs):
        q_HTK, k_HTK, v_HTV = (t.transpose(0, 1) for t in (q_THK, k_THK, v_THV))
        out_HTV = F.scaled_dot_product_attention(
            q_HTK,
            k_HTK,
            v_HTV,
            is_causal=True,
            scale=scale,
            enable_gqa=q_HTK.shape[0] != k_HTK.shape[0],
        )
        return out_HTV.transpose(0, 1)


def _debug_config():
    build_config, _ = gpt_trainstation_configs["debugmodel"]
    return build_config("flex", seq_len=_SEQ_LEN)


def test_debugmodel_mixes_every_layer_type() -> None:
    layers = _debug_config().layers
    assert len(layers) % 4 == 0
    for layer_id, layer in enumerate(layers):
        kind = layer_id % 4
        attention = layer.attention
        if kind in (0, 2):
            assert isinstance(attention, GQAttention.Config)
        else:
            assert isinstance(attention, MLAttention.Config)
            assert (attention.q_lora_rank > 0) == (kind == 3)
        assert (layer.feed_forward is not None) == (kind in (0, 3))
        assert (layer.moe is not None) == (kind in (1, 2))
        if kind == 1:
            assert layer.moe.shared_experts is None
            assert not isinstance(layer.moe.router, DeepSeekV3Router.Config)
        if kind == 2:
            assert layer.moe.shared_experts is not None
            assert isinstance(layer.moe.router, DeepSeekV3Router.Config)
            assert layer.moe.router.aux_loss is not None


@pytest.fixture(autouse=True)
def _clean_aux_loss_state():
    """Reset AuxLoss class-level state so it does not leak into other tests."""
    yield
    AuxLoss._group_counts.clear()
    AuxLoss.group_acc.clear()
    AuxLoss._step_denominator = None


def test_forward_backward() -> None:
    torch.manual_seed(0)
    config = _debug_config()
    model = config.build()
    model.init_states()
    for layer in model.layers.values():
        layer.attention.inner_attention = _CausalAttention()

    tokens = torch.randint(0, config.vocab_size, (_SEQ_LEN,))
    AuxLoss.set_step_denominator(torch.tensor(float(_SEQ_LEN), dtype=torch.float64))
    logits = model(tokens, torch.arange(_SEQ_LEN), None)
    assert logits.shape == (_SEQ_LEN, config.vocab_size)

    logits.float().logsumexp(-1).mean().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name

    _, flops = config.get_nparams_and_flops(model, _SEQ_LEN)
    assert flops > 0


@pytest.mark.parametrize(
    "tp,ep,enable_sp", [(1, 1, False), (2, 2, True), (2, 4, False)]
)
def test_sharding_config_covers_every_layer(tp: int, ep: int, enable_sp: bool) -> None:
    trainer_config = gpt_trainstation_debugmodel(seq_len=_SEQ_LEN)
    trainer_config.parallelism.tensor_parallel_degree = tp
    trainer_config.parallelism.expert_parallel_degree = ep
    trainer_config.parallelism.enable_sequence_parallel = enable_sp
    model_config = trainer_config.model_spec.model
    model_config.update_from_config(config=trainer_config)
    for layer in model_config.layers:
        assert layer.attention_norm.sharding_config is not None
        assert layer.attention.wo.sharding_config is not None
        if layer.moe is not None:
            assert layer.moe.router.gate.sharding_config is not None


def test_dist_muon_covers_every_matrix() -> None:
    trainer_config = gpt_trainstation_debugmodel(seq_len=_SEQ_LEN)
    trainer_config.parallelism.expert_parallel_degree = 2
    optimizer = dist_muon_optimizer(
        trainer_config.model_spec, lr=1e-3, parallelism=trainer_config.parallelism
    )
    compute_sharding_by_fqn = optimizer.optimizer_factory_kwargs_by_name["DistMuon"][
        "compute_sharding_by_fqn"
    ]
    model = trainer_config.model_spec.model.build()
    # Every DistMuon FQN is a real parameter, and every non-norm matrix is
    # claimed. Embedding and LM head stay on AdamW.
    matrices = {
        name
        for name, param in model.named_parameters()
        if param.ndim >= 2 and not name.startswith(("tok_embeddings", "lm_head"))
    }
    assert set(compute_sharding_by_fqn) == matrices

    trainer_config.parallelism.tensor_parallel_degree = 2
    with pytest.raises(ValueError, match="tensor_parallel_degree"):
        dist_muon_optimizer(
            trainer_config.model_spec, lr=1e-3, parallelism=trainer_config.parallelism
        )
