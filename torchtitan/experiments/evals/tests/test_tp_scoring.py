# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Vocab-parallel scoring on 2 ranks must match scoring the full logits."""

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("lm_eval")

from torchtitan.experiments.evals.lm import TrainstationLM

NUM_ROWS = 64
# Odd, so the two vocab slices have different sizes.
VOCAB_SIZE = 1001
TP_DEGREE = 2


def _reference(logits: torch.Tensor, ids: torch.Tensor):
    logprob = logits.gather(-1, ids.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)
    return logprob, logits.argmax(dim=-1) == ids


def _inputs():
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(NUM_ROWS, VOCAB_SIZE, generator=generator) * 3
    ids = torch.randint(0, VOCAB_SIZE, (NUM_ROWS,), generator=generator)
    # Rows 0-7: the target is the argmax, half of them in the second slice.
    for row in range(8):
        target = 10 if row % 2 == 0 else VOCAB_SIZE - 10
        ids[row] = target
        logits[row, target] = logits[row].max() + 1.0
    # Rows 8-11: the max is tied between one entry in each slice; argmax must
    # pick the smaller index, as torch.argmax does on the full logits.
    for row in range(8, 12):
        top = logits[row].max() + 1.0
        logits[row, 5] = top
        logits[row, VOCAB_SIZE - 5] = top
        ids[row] = 5 if row % 2 == 0 else VOCAB_SIZE - 5
    return logits, ids


def _worker(rank: int, init_file: str, result_file: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=TP_DEGREE
    )
    logits, ids = _inputs()
    # Same slicing as the lm_head sharding: ceil-sized slices, last one shorter.
    slice_size = (VOCAB_SIZE + TP_DEGREE - 1) // TP_DEGREE
    local = logits[:, rank * slice_size : (rank + 1) * slice_size].contiguous()

    lm = object.__new__(TrainstationLM)
    lm._tp_group = dist.group.WORLD
    lm._vocab_start = None
    logprob, greedy = lm._score_logits(local, ids)
    if rank == 0:
        torch.save({"logprob": logprob, "greedy": greedy}, result_file)
    dist.destroy_process_group()


def test_vocab_parallel_scoring_matches_full_vocab():
    with tempfile.TemporaryDirectory() as tmp:
        result_file = os.path.join(tmp, "result.pt")
        mp.spawn(
            _worker,
            args=(os.path.join(tmp, "rdzv"), result_file),
            nprocs=TP_DEGREE,
            join=True,
        )
        result = torch.load(result_file, weights_only=True)

    logits, ids = _inputs()
    expected_logprob, expected_greedy = _reference(logits, ids)
    torch.testing.assert_close(result["logprob"], expected_logprob, rtol=0, atol=1e-5)
    assert torch.equal(result["greedy"], expected_greedy)
    # The planted rows cover both outcomes of the greedy check.
    assert expected_greedy[:8].all() and expected_greedy[8:12].tolist() == [
        True,
        False,
        True,
        False,
    ]
