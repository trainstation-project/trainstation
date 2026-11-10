# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shape suffixes: T = num tokens, K = top-k, N = T * K routed slots,
# D = hidden dim.

import torch


@torch.library.custom_op("torchtitan::topk_permute_gather", mutates_args=())
def topk_permute_gather(
    x_TD: torch.Tensor, slot_perm_N: torch.Tensor, top_k: int
) -> torch.Tensor:
    """Return ``x_TD[slot_perm_N // top_k]``.

    ``slot_perm_N`` must be a permutation of ``range(T * top_k)`` (e.g. the
    argsort of the flattened ``(T, K)`` expert ids), so every token is gathered
    exactly ``top_k`` times.

    The default autograd of the indexed gather is an accumulating
    ``index_put_`` that sorts all N indices before reducing duplicates, which
    is slow (notably on ROCm). Because ``slot_perm_N`` is a permutation, the
    backward here is a gather through the inverse permutation followed by a
    fixed-order sum over ``top_k``: no sort, no atomics, deterministic.
    """
    return x_TD[torch.div(slot_perm_N, top_k, rounding_mode="floor")]


@topk_permute_gather.register_fake
def _(x_TD: torch.Tensor, slot_perm_N: torch.Tensor, top_k: int) -> torch.Tensor:
    return x_TD.new_empty((slot_perm_N.shape[0], x_TD.shape[1]))


def _backward(
    ctx: torch.autograd.function.FunctionCtx, grad_ND: torch.Tensor
) -> tuple[torch.Tensor, None, None]:
    (slot_perm_N,) = ctx.saved_tensors  # pyrefly: ignore[missing-attribute]
    top_k = ctx.top_k  # pyrefly: ignore[missing-attribute]
    num_slots = slot_perm_N.shape[0]
    inv_perm_N = torch.empty_like(slot_perm_N)
    inv_perm_N[slot_perm_N] = torch.arange(
        num_slots, device=slot_perm_N.device, dtype=slot_perm_N.dtype
    )
    # Row n of grad_TKD is the gradient of slot n = t * K + k.
    grad_TKD = grad_ND[inv_perm_N].view(num_slots // top_k, top_k, -1)
    # Accumulate in fp32 to match the precision of index_put_'s reduction.
    grad_TD = grad_TKD.sum(dim=1, dtype=torch.float32).to(grad_ND.dtype)
    return grad_TD, None, None


def _setup_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[torch.Tensor, torch.Tensor, int],
    output: torch.Tensor,
) -> None:
    _x, slot_perm_N, top_k = inputs
    ctx.save_for_backward(slot_perm_N)
    ctx.top_k = top_k  # pyrefly: ignore[missing-attribute]


topk_permute_gather.register_autograd(_backward, setup_context=_setup_context)
