# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _cross_entropy_torch(
    x: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
    logits_multiplier: float | None = None,
    ignore_index: int = -100,
    tp_group: dist.ProcessGroup | None = None,
    vocab_start: int = 0,
) -> torch.Tensor:
    assert tp_group is None, "tp_group is only supported by the triton backend"

    x = x.float()

    if logits_multiplier not in [None, 1]:
        x = x * logits_multiplier

    return F.cross_entropy(x, labels, reduction=reduction, ignore_index=ignore_index)
