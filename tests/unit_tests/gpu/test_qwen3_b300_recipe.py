# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check the B300 recipe's expert gradients when routing leaves experts empty."""

from dataclasses import replace

import pytest
import torch


pytest.importorskip("torchao.prototype.moe_training")

from torchtitan.models.common.moe import GroupedExperts  # noqa: E402
from torchtitan_recipes.qwen3 import qwen3_30b_a3b_mxfp8_fsdp8_ep8_b300  # noqa: E402


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() < (10, 0),
        reason="MXFP8 requires SM100 or later",
    ),
]


@pytest.mark.parametrize("extra_capacity", [0, 256])
def test_empty_experts_have_zero_weight_gradients(extra_capacity):
    torch.manual_seed(42)
    config = qwen3_30b_a3b_mxfp8_fsdp8_ep8_b300()
    _, expert_config, _, _ = next(
        config.model_spec.model.traverse(GroupedExperts.Config)
    )
    experts = (
        replace(expert_config, num_experts=4, dim=128, hidden_dim=256)
        .build()
        .cuda()
        .bfloat16()
    )
    for param in experts.parameters():
        torch.nn.init.normal_(param, std=0.02)
    counts = torch.tensor([0, 128, 0, 256], device="cuda", dtype=torch.int64)
    inputs = torch.randn(
        384 + extra_capacity,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    output = experts(inputs, counts)
    output[:384].float().square().mean().backward()

    assert torch.isfinite(output[:384]).all()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad[:384]).all()
    for param in experts.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
        torch.testing.assert_close(
            param.grad[[0, 2]], torch.zeros_like(param.grad[[0, 2]]), rtol=0, atol=0
        )
        assert param.grad[[1, 3]].abs().sum() > 0
