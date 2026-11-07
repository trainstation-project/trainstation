# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run ``RMSNorm`` (and ``norm(x, residual=r)`` residual adds) with the
lm-engine fused residual add RMSNorm Triton kernel. ``RMSNorm`` subclasses
are left unchanged, with a warning.

Enable with::

    --override.imports torchtitan.overrides.triton_rmsnorm.triton_rmsnorm
"""

import logging
from dataclasses import dataclass
from typing import overload

import spmd_types as spmd
import torch

from torchtitan.config import derive, override
from torchtitan.kernels.functional.fused_residual_add_rmsnorm.torch_implementation import (
    _fused_residual_add_rmsnorm_torch,
)
from torchtitan.kernels.functional.fused_residual_add_rmsnorm.triton_implementation import (
    _FusedResidualAddRMSNormTriton,
)
from torchtitan.models.common.nn_modules import RMSNorm

logger = logging.getLogger(__name__)

__all__ = ["TritonRMSNorm", "triton_rmsnorm"]


class _TritonFusedResidualAddRMSNorm(_FusedResidualAddRMSNormTriton):
    """SPMD types are inferred from the torch implementation (registered below)."""


spmd.register_decomposition(
    _TritonFusedResidualAddRMSNorm, _fused_residual_add_rmsnorm_torch
)


class TritonRMSNorm(RMSNorm):
    """``RMSNorm`` computed by the fused residual add RMSNorm Triton kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        pass

    @overload
    def forward(self, x: torch.Tensor, *, residual: None = None) -> torch.Tensor:
        ...

    @overload
    def forward(
        self, x: torch.Tensor, *, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def forward(
        self, x: torch.Tensor, *, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # the kernel normalizes the last dim of a 2D [rows, H] tensor
        shape = x.shape
        y, h = _TritonFusedResidualAddRMSNorm.apply(
            x.reshape(-1, shape[-1]),
            None if residual is None else residual.reshape(-1, shape[-1]),
            self.weight,
            self.eps,
            None,
            False,
        )
        y = y.view(shape)
        if residual is None:
            return y
        return y, h.view(shape)


@override(
    target=RMSNorm.Config,
    description="Compute RMSNorm (and fused residual adds) with a Triton kernel.",
)
def triton_rmsnorm(cfg: RMSNorm.Config) -> RMSNorm.Config:
    if type(cfg) is TritonRMSNorm.Config:
        return cfg
    if type(cfg) is not RMSNorm.Config:
        logger.warning(
            f"triton_rmsnorm does not support {type(cfg).__qualname__}; "
            "it keeps the default implementation."
        )
        return cfg
    return derive(cfg, TritonRMSNorm.Config)
