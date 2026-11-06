# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for ``CastLinear`` and ``LMHeadCastConverter``."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torchtitan.models.common.linear import CastLinear


def test_forward_matmul_runs_in_fp32():
    lm_head = CastLinear.Config(in_features=8, out_features=16).build()
    lm_head = lm_head.to(torch.bfloat16)
    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)

    out = lm_head(x)

    # Output is fp32 while the stored weight stays bf16 -- the cast lives in
    # forward, so a tied embedding weight keeps its storage dtype.
    assert out.dtype == torch.float32
    assert lm_head.weight.dtype == torch.bfloat16

    # Bitwise-matches an explicit fp32 matmul of the same bf16 operands...
    ref_fp32 = F.linear(x.float(), lm_head.weight.float())
    assert torch.equal(out, ref_fp32)
    # ...and differs from a bf16-accumulated matmul, proving the cast matters.
    ref_bf16 = F.linear(x, lm_head.weight)
    assert not torch.equal(out, ref_bf16.float())


def test_weight_tying_is_dtype_safe():
    # Emulate Decoder weight tying: the embedding shares the lm_head weight.
    lm_head = CastLinear.Config(in_features=8, out_features=16).build()
    lm_head = lm_head.to(torch.bfloat16)
    embedding = torch.nn.Embedding(16, 8).to(torch.bfloat16)
    embedding.weight = lm_head.weight

    x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    out = lm_head(x)

    # The shared parameter is untouched (still bf16) even though the lm_head
    # matmul ran in fp32, so the embedding lookup is unaffected.
    assert out.dtype == torch.float32
    assert embedding.weight is lm_head.weight
    assert embedding.weight.dtype == torch.bfloat16
