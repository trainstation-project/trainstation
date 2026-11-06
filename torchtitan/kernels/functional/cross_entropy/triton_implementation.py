# Copyright (c) 2026, trainstation team
# The following code is adapted from https://github.com/open-lm-engine/lm-engine
# Changes from lm-engine:
# - ignore_index support (lm-engine read out of bounds for label -100 and
#   trained on those rows); "mean" divides by the number of non-ignored labels
#   instead of B
# - vocab-parallel (TP) support: dx needs the global M and Z, so the fused
#   forward-backward kernel is split into a forward kernel that writes the
#   per-row M, Z and target logit and a backward kernel that computes dx from
#   the (TP-combined) M and Z; the loss is reduced from those statistics
#   instead of with atomic_add

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import triton
import triton.language as tl

from ...custom_op import ctx_save_for_backward, trainstation_op
from ...math import ceil_divide, get_next_power_of_2, get_powers_of_2


def _get_autotune_configs() -> list[triton.Config]:
    configs = []
    for BLOCK_SIZE_B in get_powers_of_2(1, 8):
        for num_warps in get_powers_of_2(4, 8):
            configs.append(
                triton.Config({"BLOCK_SIZE_B": BLOCK_SIZE_B}, num_warps=num_warps)
            )

    return configs


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_V"])
@triton.jit
def _cross_entropy_forward_triton_kernel(
    x_ptr,
    x_stride,
    y_ptr,
    y_stride,
    M_ptr,
    Z_ptr,
    xy_ptr,
    logits_multiplier,
    vocab_start,
    ignore_index,
    B,
    V,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
):
    BLOCK_ID = tl.program_id(0)

    BLOCK_B = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    MASK_B = BLOCK_B < B

    Z = tl.zeros((BLOCK_SIZE_B, 1), dtype=tl.float32)
    M = tl.full((BLOCK_SIZE_B, 1), -float("inf"), dtype=tl.float32)

    NUM_BLOCKS_V = tl.cdiv(V, BLOCK_SIZE_V)
    BLOCK_V = tl.arange(0, BLOCK_SIZE_V)
    x_ptrs = x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_V[None, :] * x_stride[1]

    for _ in range(NUM_BLOCKS_V):
        MASK_V = BLOCK_V < V
        # pyrefly: ignore [bad-index]
        MASK_BV = MASK_B[:, None] & MASK_V[None, :]

        x = tl.load(x_ptrs, mask=MASK_BV, other=-float("inf")).to(tl.float32)
        x_ptrs += BLOCK_SIZE_V * x_stride[1]

        if logits_multiplier is not None:
            x *= logits_multiplier

        prev_m = M
        m = tl.max(x, axis=1, keep_dims=True)
        M = max(M, m)

        x -= M
        x = tl.exp(x)
        Z = Z * tl.exp(prev_m - M) + tl.sum(x, axis=1, keep_dims=True)

        BLOCK_V += BLOCK_SIZE_V

    y = tl.load(y_ptr + BLOCK_B * y_stride[0], mask=MASK_B, other=ignore_index)
    MASK_Y = MASK_B & (y != ignore_index) & (y >= vocab_start) & (y < vocab_start + V)

    xy = tl.load(
        x_ptr + BLOCK_B * x_stride[0] + (y - vocab_start) * x_stride[1],
        mask=MASK_Y,
        other=0,
    ).to(tl.float32)
    if logits_multiplier is not None:
        xy *= logits_multiplier

    # pyrefly: ignore [bad-index]
    tl.store(M_ptr + BLOCK_B[:, None], M, mask=MASK_B[:, None])
    # pyrefly: ignore [bad-index]
    tl.store(Z_ptr + BLOCK_B[:, None], Z, mask=MASK_B[:, None])
    tl.store(xy_ptr + BLOCK_B, xy, mask=MASK_B)


@triton.autotune(configs=_get_autotune_configs(), key=["BLOCK_SIZE_V"])
@triton.jit
def _cross_entropy_backward_triton_kernel(
    x_ptr,
    x_stride,
    y_ptr,
    y_stride,
    M_ptr,
    Z_ptr,
    g_ptr,
    dx_ptr,
    dx_stride,
    logits_multiplier,
    vocab_start,
    B,
    V,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_V: tl.constexpr,
):
    BLOCK_ID = tl.program_id(0)

    BLOCK_B = BLOCK_ID * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    MASK_B = BLOCK_B < B

    # pyrefly: ignore [bad-index]
    M = tl.load(M_ptr + BLOCK_B[:, None], mask=MASK_B[:, None])
    # pyrefly: ignore [bad-index]
    Z = tl.load(Z_ptr + BLOCK_B[:, None], mask=MASK_B[:, None])
    # pyrefly: ignore [bad-index]
    g = tl.load(g_ptr + BLOCK_B[:, None], mask=MASK_B[:, None])
    y = tl.load(y_ptr + BLOCK_B * y_stride[0], mask=MASK_B) - vocab_start

    NUM_BLOCKS_V = tl.cdiv(V, BLOCK_SIZE_V)
    BLOCK_V = tl.arange(0, BLOCK_SIZE_V)
    x_ptrs = x_ptr + BLOCK_B[:, None] * x_stride[0] + BLOCK_V[None, :] * x_stride[1]
    dx_ptrs = dx_ptr + BLOCK_B[:, None] * dx_stride[0] + BLOCK_V[None, :] * dx_stride[1]

    for _ in range(NUM_BLOCKS_V):
        MASK_V = BLOCK_V < V
        # pyrefly: ignore [bad-index]
        MASK_BV = MASK_B[:, None] & MASK_V[None, :]

        x = tl.load(x_ptrs, mask=MASK_BV).to(tl.float32)
        x_ptrs += BLOCK_SIZE_V * x_stride[1]

        if logits_multiplier is not None:
            x *= logits_multiplier

        x -= M
        x = tl.exp(x)
        x /= Z

        x -= tl.where(BLOCK_V[None, :] == y[:, None], 1, 0)

        if logits_multiplier is not None:
            x *= logits_multiplier
        x *= g

        tl.store(dx_ptrs, x, mask=MASK_BV)
        dx_ptrs += BLOCK_SIZE_V * dx_stride[1]

        BLOCK_V += BLOCK_SIZE_V


@trainstation_op(mutates_args={"M", "Z", "xy"})
def _cross_entropy_forward_triton(
    x: torch.Tensor,
    labels: torch.Tensor,
    M: torch.Tensor,
    Z: torch.Tensor,
    xy: torch.Tensor,
    logits_multiplier: float | None,
    vocab_start: int,
    ignore_index: int,
) -> None:
    B, V = x.size()

    BLOCK_SIZE_V = min(
        get_next_power_of_2(V), 4096 if x.dtype == torch.float32 else 8192
    )
    GRID = lambda kwargs: (ceil_divide(B, kwargs["BLOCK_SIZE_B"]),)

    _cross_entropy_forward_triton_kernel[GRID](
        x_ptr=x,
        x_stride=x.stride(),
        y_ptr=labels,
        y_stride=labels.stride(),
        M_ptr=M,
        Z_ptr=Z,
        xy_ptr=xy,
        logits_multiplier=logits_multiplier,
        vocab_start=vocab_start,
        ignore_index=ignore_index,
        B=B,
        V=V,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )


@trainstation_op(mutates_args={"dx"})
def _cross_entropy_backward_triton(
    x: torch.Tensor,
    labels: torch.Tensor,
    M: torch.Tensor,
    Z: torch.Tensor,
    g: torch.Tensor,
    dx: torch.Tensor,
    logits_multiplier: float | None,
    vocab_start: int,
) -> None:
    B, V = x.size()

    BLOCK_SIZE_V = min(
        get_next_power_of_2(V), 4096 if x.dtype == torch.float32 else 8192
    )
    GRID = lambda kwargs: (ceil_divide(B, kwargs["BLOCK_SIZE_B"]),)

    _cross_entropy_backward_triton_kernel[GRID](
        x_ptr=x,
        x_stride=x.stride(),
        y_ptr=labels,
        y_stride=labels.stride(),
        M_ptr=M,
        Z_ptr=Z,
        g_ptr=g,
        dx_ptr=dx,
        dx_stride=dx.stride(),
        logits_multiplier=logits_multiplier,
        vocab_start=vocab_start,
        B=B,
        V=V,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
    )


class _CrossEntropyTriton(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        x: torch.Tensor,
        labels: torch.Tensor,
        reduction: str,
        logits_multiplier: float | None,
        ignore_index: int,
        tp_group: dist.ProcessGroup | None,
        vocab_start: int,
    ) -> torch.Tensor:
        B = x.size(0)
        M = torch.empty(B, device=x.device, dtype=torch.float32)
        Z = torch.empty(B, device=x.device, dtype=torch.float32)
        xy = torch.empty(B, device=x.device, dtype=torch.float32)

        _cross_entropy_forward_triton(
            x=x,
            labels=labels,
            M=M,
            Z=Z,
            xy=xy,
            logits_multiplier=logits_multiplier,
            vocab_start=vocab_start,
            ignore_index=ignore_index,
        )

        if tp_group is not None:
            # global M = max_r M_r, Z = sum_r Z_r * exp(M_r - M); xy is nonzero
            # only on the rank holding the label
            stats = funcol.all_gather_tensor(
                torch.stack([M, Z, xy]), gather_dim=0, group=tp_group
            ).view(-1, 3, B)
            M_r, Z_r = stats[:, 0], stats[:, 1]
            M = M_r.max(dim=0).values
            Z = (Z_r * torch.exp(M_r - M)).sum(dim=0)
            xy = stats[:, 2].sum(dim=0)

        is_valid = labels != ignore_index
        l = torch.where(is_valid, M + torch.log(Z) - xy, 0).sum()

        num_valid = None
        if reduction == "mean":
            num_valid = is_valid.sum()
            l = l / num_valid

        ctx_save_for_backward(ctx, x, labels, M, Z, num_valid)
        ctx.reduction = reduction
        ctx.logits_multiplier = logits_multiplier
        ctx.ignore_index = ignore_index
        ctx.vocab_start = vocab_start

        return l

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(
        ctx, dy: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        x, labels, M, Z, num_valid = ctx.saved_tensors

        g = dy.float()
        if ctx.reduction == "mean":
            g = g / num_valid
        g = torch.where(labels != ctx.ignore_index, g, 0)

        dx = torch.empty_like(x, memory_format=torch.contiguous_format)
        _cross_entropy_backward_triton(
            x=x,
            labels=labels,
            M=M,
            Z=Z,
            g=g,
            dx=dx,
            logits_multiplier=ctx.logits_multiplier,
            vocab_start=ctx.vocab_start,
        )

        return dx, None, None, None, None, None, None
