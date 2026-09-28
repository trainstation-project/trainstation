# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("lm_eval")

from lm_eval.tasks import TaskManager

from torchtitan.experiments.evals.evaluate import (
    expand_tasks,
    missing_tasks,
    results_path,
    TASKS_DIR,
    write_on_rank0,
)

QUICK = ["wikitext", "lambada_openai", "hellaswag", "arc_easy", "piqa"]


@pytest.fixture(scope="module")
def task_manager():
    return TaskManager(include_path=TASKS_DIR)


def test_expand_tasks(task_manager):
    assert expand_tasks(["trainstation_quick"], task_manager) == QUICK
    # Suites expand recursively; trainstation_mmlu aggregates, so it stays whole.
    assert expand_tasks(["trainstation_full"], task_manager) == QUICK + [
        "arc_challenge",
        "winogrande",
        "boolq",
        "openbookqa",
        "sciq",
        "trainstation_mmlu",
    ]
    # A task listed twice, directly and through a suite, is kept once.
    assert expand_tasks(["hellaswag", "trainstation_quick"], task_manager) == [
        "hellaswag",
        "wikitext",
        "lambada_openai",
        "arc_easy",
        "piqa",
    ]
    assert expand_tasks(["mmlu"], task_manager) == ["mmlu"]


def test_missing_tasks(tmp_path):
    output = str(tmp_path)
    path = results_path(output, 100, "hellaswag")
    os.makedirs(os.path.dirname(path))
    open(path, "w").close()
    assert missing_tasks(output, 100, ["hellaswag", "piqa"]) == ["piqa"]
    assert missing_tasks(output, 200, ["hellaswag"]) == ["hellaswag"]


def _write_worker(rank: int, init_file: str, path: str, errors_dir: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        write_on_rank0(path, {"step": 1})
    except RuntimeError as e:
        with open(os.path.join(errors_dir, f"rank{rank}"), "w") as f:
            f.write(str(e))
    dist.destroy_process_group()


def test_failed_write_raises_on_every_rank():
    with tempfile.TemporaryDirectory() as tmp:
        # The parent of the results folder is a file, so makedirs fails on rank 0.
        blocker = os.path.join(tmp, "not_a_dir")
        open(blocker, "w").close()
        path = os.path.join(blocker, "step-1", "hellaswag.json")
        mp.spawn(
            _write_worker,
            args=(os.path.join(tmp, "rdzv"), path, tmp),
            nprocs=2,
            join=True,
        )
        for rank in range(2):
            with open(os.path.join(tmp, f"rank{rank}")) as f:
                assert "Rank 0 failed to write" in f.read()
