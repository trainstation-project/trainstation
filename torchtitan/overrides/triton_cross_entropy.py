# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Triton cross-entropy loss override.

``triton_cross_entropy`` makes ``CrossEntropyLoss`` (including the inner loss
of ``ChunkedLossWrapper``) use the lm-engine Triton kernel from ``torchtitan.kernels.functional.cross_entropy``. Compared
with the default loss, it reads bf16 logits directly instead of materializing
an fp32 copy and an fp32 log-softmax for backward, and it computes the
vocab-parallel (TP) loss with a single small all-gather instead of three
all-reduces. Other ``CrossEntropyLoss`` subclasses are left unchanged, with a
warning.

Enable with::

    --override.imports torchtitan.overrides.triton_cross_entropy.triton_cross_entropy
"""

import logging
from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.distributed as dist

from torchtitan.components.loss import (
    CrossEntropyLoss,
    get_local_vocab_shard,
    IGNORE_INDEX,
)
from torchtitan.config import CompileConfig, derive, override
from torchtitan.distributed.spmd_types import current_spmd_mesh, spmd_mesh_size
from torchtitan.kernels.functional.cross_entropy.torch_implementation import (
    _cross_entropy_torch,
)
from torchtitan.kernels.functional.cross_entropy.triton_implementation import (
    _CrossEntropyTriton,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TritonCrossEntropyLoss",
    "triton_cross_entropy",
    "triton_cross_entropy_loss",
]


class _TritonLossParallelCrossEntropy(_CrossEntropyTriton):
    """``_CrossEntropyTriton`` with the vocab-parallel SPMD types declared.

    The kernel and its all-gather are opaque to the SPMD checker, so declare
    the same contract as ``_LossParallelCrossEntropy``: logits S(-1)@TP,
    labels I@TP -> loss I@TP, with non-TP axes passed through from logits.
    """

    @staticmethod
    def spmd_typecheck(
        result: torch.Tensor,
        *,
        x: torch.Tensor,
        labels: torch.Tensor,
        tp_group: dist.ProcessGroup,
    ) -> None:
        spmd.assert_type(x, {tp_group: spmd.S(x.dim() - 1)})
        spmd.assert_type(labels, {tp_group: spmd.I})
        spmd.assert_local_type_like(
            result,
            x,
            {tp_group: spmd.I},  # pyrefly: ignore [bad-argument-type]
        )


class _TritonCrossEntropy(_CrossEntropyTriton):
    """``_CrossEntropyTriton`` without vocab sharding. Its SPMD types are
    inferred from the equivalent torch implementation (registered below), so
    they match the default ``cross_entropy_loss`` exactly."""


spmd.register_decomposition(_TritonCrossEntropy, _cross_entropy_torch)


def triton_cross_entropy_loss(
    pred: torch.Tensor,
    labels: torch.Tensor,
    *,
    global_vocab_size: int | None = None,
) -> torch.Tensor:
    """Triton cross entropy over ``pred[T, V]`` and ``labels[T]`` with sum
    reduction; drop-in for ``torchtitan.components.loss.cross_entropy_loss``."""
    tp_sharded = spmd_mesh_size("tp") > 1
    if tp_sharded:
        if global_vocab_size is None:
            raise ValueError(
                "triton_cross_entropy_loss requires global_vocab_size when the "
                "vocab is sharded over TP."
            )
        vocab_size = global_vocab_size
    else:
        vocab_size = pred.size(-1)
    # The kernel treats a label outside the vocab as a target logit of 0, which
    # would silently train on it; F.cross_entropy and _LossParallelCrossEntropy
    # both reject such labels, so do the same (asynchronously, without a sync).
    torch._assert_async(
        torch.all((labels == IGNORE_INDEX) | ((labels >= 0) & (labels < vocab_size))),
        f"labels must be {IGNORE_INDEX} or in [0, {vocab_size})",
    )

    if tp_sharded:
        mesh = current_spmd_mesh()
        assert mesh is not None  # spmd_mesh_size("tp") > 1 implies a mesh
        tp_group = mesh.get_group("tp")
        vocab_start, _ = get_local_vocab_shard(pred, tp_group, vocab_size)
        return _TritonLossParallelCrossEntropy.apply(
            pred, labels, "sum", None, IGNORE_INDEX, tp_group, vocab_start
        )

    return _TritonCrossEntropy.apply(pred, labels, "sum", None, IGNORE_INDEX, None, 0)


class TritonCrossEntropyLoss(CrossEntropyLoss):
    """``CrossEntropyLoss`` computed by the Triton cross-entropy kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(CrossEntropyLoss.Config):
        pass

    def __init__(self, config: Config, *, compile_config: CompileConfig | None = None):
        self.fn = triton_cross_entropy_loss
        self._maybe_compile(compile_config)
        self.global_vocab_size = config.global_vocab_size


@override(
    target=CrossEntropyLoss.Config,
    description="Compute cross-entropy loss with the Triton cross-entropy kernel.",
)
def triton_cross_entropy(
    cfg: CrossEntropyLoss.Config,
) -> CrossEntropyLoss.Config:
    if type(cfg) is TritonCrossEntropyLoss.Config:
        return cfg
    if type(cfg) is not CrossEntropyLoss.Config:
        logger.warning(
            f"triton_cross_entropy does not support {type(cfg).__qualname__}; "
            "it keeps the default cross-entropy implementation."
        )
        return cfg
    return derive(cfg, TritonCrossEntropyLoss.Config)
