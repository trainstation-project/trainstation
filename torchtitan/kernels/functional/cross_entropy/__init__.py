# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.distributed as dist

from ...accelerator import KernelBackend
from ...custom_op import CustomOp

from ...utils import is_triton_available
from .torch_implementation import _cross_entropy_torch


class _CrossEntropy(CustomOp):
    ...


_CrossEntropy[KernelBackend.torch] = _cross_entropy_torch


if is_triton_available():
    from .triton_implementation import _CrossEntropyTriton

    _CrossEntropy[KernelBackend.cuda] = _CrossEntropyTriton
    _CrossEntropy[KernelBackend.triton] = _CrossEntropyTriton


def cross_entropy(
    x: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
    logits_multiplier: float | None = None,
    *,
    ignore_index: int = -100,
    tp_group: dist.ProcessGroup | None = None,
    vocab_start: int = 0,
    kernel_backend: KernelBackend | None = None,
) -> torch.Tensor:
    """
    cross entropy loss

    :param x: logits
    :type x: torch.Tensor
    :param labels: labels
    :type labels: torch.Tensor
    :param reduction: reduction method: "sum" or "mean"
    :type reduction: str
    :param logits_multiplier: logits multiplier pre-multiplies logits, None implies 1. Defaults to None.
    :type logits_multiplier: float | None
    :param ignore_index: label to ignore
    :type ignore_index: int
    :param tp_group: TP group the vocab is sharded over, triton only. Defaults to None.
    :type tp_group: dist.ProcessGroup | None
    :param vocab_start: global vocab id of the first column of x. Defaults to 0.
    :type vocab_start: int
    :param kernel_backend: KernelBackend
    :type kernel_backend: KernelBackend | None
    :return: loss
    :rtype: Tensor
    """

    assert reduction in ["sum", "mean"]
    assert x.dim() == 2, "x should be 2 dimensional"
    assert labels.dim() == 1, "labels should be 1 dimensional"
    assert labels.size(0) == x.size(
        0
    ), "x and labels have different number of elements along batch dimension"

    x = _CrossEntropy.run(
        x=x,
        labels=labels,
        reduction=reduction,
        logits_multiplier=logits_multiplier,
        ignore_index=ignore_index,
        tp_group=tp_group,
        vocab_start=vocab_start,
        kernel_backend=kernel_backend,
    )

    return x
