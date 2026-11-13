# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for topk_permute_gather (MoE expert-sorted token gather)."""

import unittest

import torch

from torchtitan.ops.topk_gather import topk_permute_gather


def _inputs(num_tokens: int, top_k: int, dim: int, num_experts: int, dtype):
    torch.manual_seed(0)
    x_TD = torch.randn(num_tokens, dim, dtype=dtype, requires_grad=True)
    expert_ids_TK = torch.randint(0, num_experts, (num_tokens, top_k))
    slot_perm_N = torch.argsort(expert_ids_TK.view(-1), stable=True)
    return x_TD, slot_perm_N


class TestTopkPermuteGather(unittest.TestCase):
    def test_matches_indexing(self):
        for dtype in (torch.float32, torch.bfloat16):
            x_TD, slot_perm_N = _inputs(64, 8, 32, 16, dtype)
            grad_ND = torch.randn(slot_perm_N.numel(), 32, dtype=dtype)

            # fp64 reference: the op reduces in fp32 and rounds once, so it
            # can differ from a low-precision index_put_ accumulation.
            ref_x = x_TD.detach().double().requires_grad_()
            ref = ref_x[slot_perm_N // 8]
            ref.backward(grad_ND.double())

            out = topk_permute_gather(x_TD, slot_perm_N, 8)
            out.backward(grad_ND)

            torch.testing.assert_close(out, ref.to(dtype), rtol=0, atol=0)
            torch.testing.assert_close(x_TD.grad, ref_x.grad.to(dtype))

    def test_gradcheck(self):
        x_TD, slot_perm_N = _inputs(6, 2, 3, 4, torch.float64)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda x: topk_permute_gather(x, slot_perm_N, 2), (x_TD,)
            )
        )


if __name__ == "__main__":
    unittest.main()
