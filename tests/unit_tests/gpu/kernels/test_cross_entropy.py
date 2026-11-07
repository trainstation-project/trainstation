# Copyright (c) 2026, trainstation team
# The following code is adapted from https://github.com/open-lm-engine/lm-engine
#
# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from collections.abc import Callable

import pytest
import torch

from torchtitan.kernels import KernelBackend
from torchtitan.kernels.functional import cross_entropy

from .utils import (
    assert_equal_tensors,
    get_duplicated_tensors,
    skip_if_incompatible_kernel_backend,
)


# (num_tokens, vocab_size): tiny / non-power-of-2 shapes, vocab sizes that need multiple
# BLOCK_SIZE_V iterations (BLOCK_SIZE_V caps at 4096 for fp32 and 8192 for 16-bit dtypes), and
# real-model vocab sizes (llama3 128256, qwen3 151936).
_SHAPES = [
    (1, 1),
    (7, 15),
    (33, 1000),
    (64, 4096),
    (19, 8193),
    (128, 32000),
    (17, 128256),
    (9, 151936),
]
_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
_SEED = 42

# Kernel and torch both compute in fp32 from the same inputs, so gradients should agree to
# about one ulp of the output dtype. atol is near zero (fp16: its subnormal step) so that the
# many tiny non-target gradients are checked relatively rather than absorbed by atol.
_GRAD_TOLERANCES = dict(
    atol_float32=1e-9,
    rtol_float32=1e-5,
    atol_float16=6e-8,
    rtol_float16=2e-3,
    atol_bfloat16=1e-9,
    rtol_bfloat16=1.6e-2,
)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("logits_multiplier", [None, 0.7])
@pytest.mark.parametrize(
    "function",
    [cross_entropy, torch.compile(cross_entropy, fullgraph=True)],
    ids=["eager", "compiled"],
)
@torch._dynamo.config.patch(recompile_limit=1024)
def test_cross_entropy_triton_vs_torch(
    shape: tuple[int, int],
    dtype: torch.dtype,
    reduction: str,
    logits_multiplier: float | None,
    function: Callable,
) -> None:
    device = skip_if_incompatible_kernel_backend(KernelBackend.triton)
    torch.manual_seed(_SEED)

    x_kernel, x_torch = get_duplicated_tensors(
        shape, device=device, dtype=dtype, std=None
    )
    labels = torch.randint(0, shape[-1], (shape[0],), device=device)

    loss_kernel = function(
        x=x_kernel,
        labels=labels,
        reduction=reduction,
        logits_multiplier=logits_multiplier,
        kernel_backend=KernelBackend.triton,
    )
    loss_torch = cross_entropy(
        x=x_torch,
        labels=labels,
        reduction=reduction,
        logits_multiplier=logits_multiplier,
        kernel_backend=KernelBackend.torch,
    )

    # the per-row losses are summed in a different order than in torch; scale atol with the
    # number of summed rows for "sum"
    atol = 7.2e-5 * (shape[0] if reduction == "sum" else 1)
    assert_equal_tensors(
        loss_kernel, loss_torch, False, atol_float32=atol, rtol_float32=1e-6
    )

    loss_kernel.backward()
    loss_torch.backward()

    assert x_kernel.grad is not None and x_torch.grad is not None
    assert_equal_tensors(x_kernel.grad, x_torch.grad, False, **_GRAD_TOLERANCES)


@pytest.mark.parametrize("ignore_fraction", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cross_entropy_triton_ignore_index(
    dtype: torch.dtype, reduction: str, ignore_fraction: float
) -> None:
    device = skip_if_incompatible_kernel_backend(KernelBackend.triton)
    torch.manual_seed(_SEED)

    shape = (257, 8193)
    x_kernel, x_torch = get_duplicated_tensors(
        shape, device=device, dtype=dtype, std=None
    )
    labels = torch.randint(0, shape[-1], (shape[0],), device=device)
    labels[torch.rand(shape[0], device=device) < ignore_fraction] = -100

    loss_kernel = cross_entropy(
        x=x_kernel,
        labels=labels,
        reduction=reduction,
        kernel_backend=KernelBackend.triton,
    )
    loss_torch = cross_entropy(
        x=x_torch,
        labels=labels,
        reduction=reduction,
        kernel_backend=KernelBackend.torch,
    )

    # mean over zero valid labels is nan for both
    torch.testing.assert_close(
        loss_kernel, loss_torch, atol=1e-3, rtol=1e-5, equal_nan=True
    )

    upstream = torch.randn_like(loss_torch)
    loss_kernel.backward(upstream)
    loss_torch.backward(upstream)

    assert x_kernel.grad is not None and x_torch.grad is not None
    assert_equal_tensors(x_kernel.grad, x_torch.grad, False, **_GRAD_TOLERANCES)
    if ignore_fraction > 0:
        ignored = labels == -100
        assert torch.all(x_kernel.grad[ignored] == 0)


def test_cross_entropy_triton_deterministic() -> None:
    """Repeated calls in one process (same autotuned config) are bitwise equal."""
    device = skip_if_incompatible_kernel_backend(KernelBackend.triton)
    torch.manual_seed(_SEED)

    x = torch.randn(
        4096, 32000, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    labels = torch.randint(0, 32000, (4096,), device=device)

    results = []
    for _ in range(3):
        x.grad = None
        loss = cross_entropy(
            x=x, labels=labels, reduction="sum", kernel_backend=KernelBackend.triton
        )
        loss.backward()
        assert x.grad is not None
        results.append((loss.detach().clone(), x.grad.clone()))

    for loss, grad in results[1:]:
        assert_equal_tensors(loss, results[0][0], True)
        assert_equal_tensors(grad, results[0][1], True)


def test_cross_entropy_triton_large_logits_precision() -> None:
    """For large |logits| the kernel must keep torch's log_softmax precision."""
    device = skip_if_incompatible_kernel_backend(KernelBackend.triton)
    torch.manual_seed(_SEED)

    B, V = 256, 32000
    x = torch.randn(B, V, device=device) * 300
    labels = torch.randint(0, V, (B,), device=device)
    x_kernel = x.clone().requires_grad_()
    x_golden = x.double().requires_grad_()
    x_torch = x.clone().requires_grad_()

    cross_entropy(
        x=x_kernel, labels=labels, reduction="sum", kernel_backend=KernelBackend.triton
    ).backward()
    torch.nn.functional.cross_entropy(x_golden, labels, reduction="sum").backward()
    torch.nn.functional.cross_entropy(x_torch, labels, reduction="sum").backward()

    assert x_kernel.grad is not None and x_golden.grad is not None
    assert x_torch.grad is not None
    kernel_error = (x_kernel.grad.double() - x_golden.grad).abs().max()
    torch_error = (x_torch.grad.double() - x_golden.grad).abs().max()
    assert kernel_error <= 2 * torch_error


@pytest.mark.parametrize(
    "shape", [(4096, 128256), (2048, 151936), (1024, 201088), (8, 8193)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cross_entropy_triton_vs_fp64_golden(
    shape: tuple[int, int], dtype: torch.dtype
) -> None:
    """Error against an fp64 golden must be within 2x the error of torchtitan's default
    fp32 loss plus the output rounding floor, on peaked (LM-like) logits."""
    device = skip_if_incompatible_kernel_backend(KernelBackend.triton)
    torch.manual_seed(_SEED)

    B, V = shape
    labels = torch.randint(0, V, (B,), device=device)
    labels[::11] = -100
    x = torch.randn(B, V, device=device) * 3
    # make most rows confident in their label, as a trained LM is
    x[torch.arange(B), labels.clamp(min=0)] += 15
    x = x.to(dtype)

    upstream = torch.randn((), device=device, dtype=torch.float64)
    results = {}
    for name in ("kernel", "default", "golden"):
        xx = (x.double() if name == "golden" else x.clone()).requires_grad_()
        if name == "golden":
            loss = torch.nn.functional.cross_entropy(xx, labels, reduction="sum")
        else:
            backend = KernelBackend.triton if name == "kernel" else KernelBackend.torch
            loss = cross_entropy(
                x=xx, labels=labels, reduction="sum", kernel_backend=backend
            )
        loss.backward(upstream.to(loss.dtype))
        assert xx.grad is not None
        results[name] = (loss.detach().double(), xx.grad.double())

    for i, out_dtype in enumerate((torch.float32, dtype)):
        golden = results["golden"][i]
        floor = (golden.to(out_dtype).double() - golden).abs().max()
        default_error = (results["default"][i] - golden).abs().max()
        kernel_error = (results["kernel"][i] - golden).abs().max()
        assert kernel_error <= 2 * default_error + floor, (
            i,
            kernel_error,
            default_error,
        )
