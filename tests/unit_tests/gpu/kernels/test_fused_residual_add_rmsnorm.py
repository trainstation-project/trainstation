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
from torchtitan.kernels.functional import fused_residual_add_rmsnorm
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.overrides.triton_rmsnorm import TritonRMSNorm

from .utils import (
    assert_equal_tensors,
    get_duplicated_tensors,
    skip_if_incompatible_kernel_backend,
)


_EPSILON = 1e-5
_SEED = 42

# (num_tokens, hidden_size): tiny sizes, powers of 2 and their neighbors (the kernel pads H
# to the next power of 2 and masks the rest), and real-model hidden sizes.
_SIZES = [
    (400, h)
    for h in (1, 7, 64, 100, 1000, 2880, 4095, 4096, 4097, 5120, 7168, 8192, 16384)
]


@pytest.mark.parametrize("size", _SIZES)
@pytest.mark.parametrize("memory_efficient", [False, True])
@pytest.mark.parametrize("has_weight", [False, True])
@pytest.mark.parametrize("multiplier", [None, 0.9])
@pytest.mark.parametrize(
    "function",
    [
        fused_residual_add_rmsnorm,
        torch.compile(fused_residual_add_rmsnorm, fullgraph=True),
    ],
    ids=["eager", "compiled"],
)
@torch._dynamo.config.patch(recompile_limit=1024)
def test_fused_residual_add_rmsnorm(
    size: tuple[int, int],
    memory_efficient: bool,
    has_weight: bool,
    multiplier: float | None,
    function: Callable,
) -> None:
    device = skip_if_incompatible_kernel_backend(KernelBackend.triton)
    torch.manual_seed(_SEED)
    dtype = torch.float32

    x_kernel, x_expected = get_duplicated_tensors(
        size, device=device, dtype=dtype, std=None
    )
    residual_kernel, residual_expected = get_duplicated_tensors(
        size, device=device, dtype=dtype, std=None
    )

    if has_weight:
        weight_kernel, weight_expected = get_duplicated_tensors(
            (size[-1],), device=device, dtype=dtype, std=None
        )
    else:
        weight_kernel = None
        weight_expected = None

    z_kernel, r_kernel = function(
        x=x_kernel,
        residual=residual_kernel,
        weight=weight_kernel,
        eps=_EPSILON,
        multiplier=multiplier,
        memory_efficient=memory_efficient,
        kernel_backend=KernelBackend.triton,
    )

    z_expected, r_expected = fused_residual_add_rmsnorm(
        x=x_expected,
        residual=residual_expected,
        weight=weight_expected,
        eps=_EPSILON,
        multiplier=multiplier,
        kernel_backend=KernelBackend.torch,
    )

    assert r_kernel is not None and r_expected is not None
    z_kernel = z_kernel * 2 + r_kernel * 3
    z_expected = z_expected * 2 + r_expected * 3

    assert_equal_tensors(z_kernel, z_expected, False)

    z_kernel.sum().backward()
    z_expected.sum().backward()

    assert x_kernel.grad is not None and x_expected.grad is not None
    assert residual_kernel.grad is not None and residual_expected.grad is not None
    assert_equal_tensors(x_kernel.grad, x_expected.grad, False)
    assert_equal_tensors(residual_kernel.grad, residual_expected.grad, False)

    if has_weight:
        assert weight_kernel is not None and weight_expected is not None
        assert weight_kernel.grad is not None and weight_expected.grad is not None
        assert_equal_tensors(
            weight_kernel.grad,
            weight_expected.grad,
            False,
            atol_float32=7.7e-5,
            rtol_float32=0,
        )


@pytest.mark.parametrize("hidden_size", [1000, 4096, 5120])
@pytest.mark.parametrize("memory_efficient", [False, True])
@pytest.mark.parametrize("multiplier", [None, 0.5])
def test_rmsnorm_without_residual(
    hidden_size: int, memory_efficient: bool, multiplier: float | None
) -> None:
    """Without a residual the kernel normalizes ``multiplier * x``; its backward must use the
    scaled input too (lm-engine used the unscaled input there)."""
    device = skip_if_incompatible_kernel_backend(KernelBackend.triton)
    torch.manual_seed(_SEED)

    x_kernel, x_expected = get_duplicated_tensors(
        (400, hidden_size), device=device, dtype=torch.float32, std=None
    )
    weight_kernel, weight_expected = get_duplicated_tensors(
        (hidden_size,), device=device, dtype=torch.float32, std=None
    )

    y_kernel, r_kernel = fused_residual_add_rmsnorm(
        x=x_kernel,
        residual=None,
        weight=weight_kernel,
        eps=_EPSILON,
        multiplier=multiplier,
        memory_efficient=memory_efficient,
        kernel_backend=KernelBackend.triton,
    )
    y_expected, _ = fused_residual_add_rmsnorm(
        x=x_expected,
        residual=None,
        weight=weight_expected,
        eps=_EPSILON,
        multiplier=multiplier,
        kernel_backend=KernelBackend.torch,
    )
    assert r_kernel is None
    assert_equal_tensors(y_kernel, y_expected, False)

    upstream = torch.randn_like(y_expected)
    y_kernel.backward(upstream)
    y_expected.backward(upstream)

    assert x_kernel.grad is not None and x_expected.grad is not None
    assert weight_kernel.grad is not None and weight_expected.grad is not None
    assert_equal_tensors(x_kernel.grad, x_expected.grad, False)
    assert_equal_tensors(
        weight_kernel.grad,
        weight_expected.grad,
        False,
        atol_float32=7.7e-5,
        rtol_float32=0,
    )


def _golden(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    multiplier: float | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    h = x if multiplier is None else x * multiplier
    if residual is not None:
        h = h + residual
    y = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + _EPSILON) * weight
    return y, (h if residual is not None else None)


@pytest.mark.parametrize(
    "shape", [(4096, 4096), (4096, 5120), (2048, 7168), (2048, 8192), (8, 2880)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("multiplier", [None, 0.5])
def test_fused_residual_add_rmsnorm_vs_fp64_golden(
    shape: tuple[int, int],
    dtype: torch.dtype,
    has_residual: bool,
    multiplier: float | None,
) -> None:
    """Every output and gradient must be within 2x the torch implementation's error against an
    fp64 golden, plus the output rounding floor."""
    device = skip_if_incompatible_kernel_backend(KernelBackend.triton)
    torch.manual_seed(_SEED)

    x = torch.randn(shape, device=device).to(dtype)
    residual = torch.randn(shape, device=device).to(dtype) if has_residual else None
    weight = (1 + 0.1 * torch.randn(shape[-1], device=device)).to(dtype)
    grad_y = torch.randn(shape, device=device).to(dtype)
    grad_h = torch.randn(shape, device=device).to(dtype)

    results: dict[str, list[torch.Tensor]] = {}
    for name in ("kernel", "torch", "golden"):
        cast = (lambda t: t.double()) if name == "golden" else (lambda t: t.clone())
        xx = cast(x).requires_grad_()
        rr = None if residual is None else cast(residual).requires_grad_()
        ww = cast(weight).requires_grad_()
        if name == "golden":
            y, h = _golden(xx, rr, ww, multiplier)
        else:
            backend = KernelBackend.triton if name == "kernel" else KernelBackend.torch
            y, h = fused_residual_add_rmsnorm(
                x=xx,
                residual=rr,
                weight=ww,
                eps=_EPSILON,
                multiplier=multiplier,
                kernel_backend=backend,
            )
        outputs, grads = [y], [cast(grad_y)]
        if h is not None:
            outputs.append(h)
            grads.append(cast(grad_h))
        torch.autograd.backward(outputs, grads)
        tensors = outputs + [xx.grad, ww.grad] + ([] if rr is None else [rr.grad])
        assert all(t is not None for t in tensors)
        results[name] = [t.detach().double() for t in tensors]

    for i, golden in enumerate(results["golden"]):
        floor = (golden.to(dtype).double() - golden).abs().max()
        torch_error = (results["torch"][i] - golden).abs().max()
        kernel_error = (results["kernel"][i] - golden).abs().max()
        assert kernel_error <= 2 * torch_error + floor, (i, kernel_error, torch_error)


def test_fused_residual_add_rmsnorm_deterministic() -> None:
    """Repeated calls in one process (same autotuned config) are bitwise equal."""
    device = skip_if_incompatible_kernel_backend(KernelBackend.triton)
    torch.manual_seed(_SEED)

    x = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = (1 + 0.1 * torch.randn(4096, device=device)).bfloat16()

    results = []
    for _ in range(3):
        xx, rr, ww = (t.clone().requires_grad_() for t in (x, residual, weight))
        y, h = fused_residual_add_rmsnorm(
            x=xx,
            residual=rr,
            weight=ww,
            eps=_EPSILON,
            kernel_backend=KernelBackend.triton,
        )
        assert h is not None
        torch.autograd.backward([y, h], [torch.ones_like(y), torch.ones_like(h)])
        results.append([y, h, xx.grad, rr.grad, ww.grad])

    for result in results[1:]:
        for a, b in zip(result, results[0]):
            assert a is not None and b is not None
            assert_equal_tensors(a, b, True)


@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_triton_rmsnorm_module_matches_rmsnorm(
    has_residual: bool, compiled: bool
) -> None:
    """``TritonRMSNorm`` on 3D activations matches ``RMSNorm``, with and without a residual."""
    device = skip_if_incompatible_kernel_backend(KernelBackend.triton)
    torch.manual_seed(_SEED)

    H = 4096
    weight = 1 + 0.1 * torch.randn(H, device=device)
    x = torch.randn(2, 512, H, device=device, dtype=torch.bfloat16)
    residual = torch.randn_like(x)

    results = []
    for cls in (RMSNorm, TritonRMSNorm):
        norm = cls(cls.Config(normalized_shape=H, eps=_EPSILON)).to(device).bfloat16()
        with torch.no_grad():
            norm.weight.copy_(weight)
        xx, rr = x.clone().requires_grad_(), residual.clone().requires_grad_()

        def fn(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, ...]:
            out = norm(a, residual=b) if has_residual else norm(a)
            return out if isinstance(out, tuple) else (out,)

        outputs: tuple[torch.Tensor, ...] = (
            torch.compile(fn, fullgraph=True) if compiled else fn
        )(xx, rr)
        torch.autograd.backward(
            outputs, [torch.ones_like(o) * (i + 1) for i, o in enumerate(outputs)]
        )
        grads = [xx.grad] + ([rr.grad] if has_residual else [])
        results.append(([o.detach() for o in outputs] + grads, norm.weight.grad))

    (ref_tensors, ref_dW), (tri_tensors, tri_dW) = results
    assert len(ref_tensors) == len(tri_tensors)
    for a, b in zip(ref_tensors, tri_tensors):
        assert a is not None and b is not None
        assert a.shape == b.shape
        # both compute in fp32 from the same bf16 inputs: about one bf16 ulp apart
        torch.testing.assert_close(b, a, atol=3.2e-2, rtol=1.6e-2)
    assert ref_dW is not None and tri_dW is not None
    # dW sums 1024 rows of O(1) values, so its bf16 ulp is ~0.25
    torch.testing.assert_close(tri_dW, ref_dW, atol=6e-1, rtol=1.6e-2)
