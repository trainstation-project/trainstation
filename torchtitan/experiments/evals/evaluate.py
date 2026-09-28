# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluate the DCP checkpoints of a training run with lm-evaluation-harness.

Runs as its own job next to (never inside) the training job. It reads the
training config with the same CLI as ``torchtitan.train``, loads each
``step-N`` checkpoint natively (no HF conversion), and writes lm-eval results
to ``<dump_folder>/evals/step-N/<task>.json``, one file per task.

Usage::

    torchrun --nproc_per_node=8 -m torchtitan.experiments.evals.evaluate \\
        --tasks trainstation_quick [--steps 1000 2000 | --watch] \\
        -- --module llama3 --config llama3_8b [training overrides]

Without ``--steps``, every completed checkpoint is evaluated on the tasks it
has no results for yet. ``--watch`` keeps polling for new checkpoints until
the final training step has been evaluated, or until no new checkpoint has
appeared for ``--max-idle-hours``.
"""

import dataclasses
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import lm_eval
import torch
import torch.distributed as dist
import tyro
from lm_eval.evaluator import simple_evaluate
from lm_eval.tasks import TaskManager

from torchtitan.config import ConfigManager
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.evals.checkpoint import (
    build_model,
    checkpoint_folder,
    eval_parallelism,
    list_checkpoint_steps,
    load_weights,
)
from torchtitan.experiments.evals.lm import TrainstationLM
from torchtitan.experiments.evals.wandb_logger import EvalWandBLogger
from torchtitan.observability.logging import init_logger
from torchtitan.tools import utils
from torchtitan.trainer import Trainer

logger = logging.getLogger(__name__)

T = TypeVar("T")

TASKS_DIR = os.path.join(os.path.dirname(__file__), "tasks")


@dataclass(kw_only=True, slots=True)
class EvalConfig:
    """Options of the eval job; the training config is passed after '--'."""

    tasks: list[str]
    """lm-eval task or group names, e.g. trainstation_quick."""

    steps: list[int] | None = None
    """Checkpoint steps to evaluate. Default: every completed checkpoint that
    has no results for these tasks yet."""

    watch: bool = False
    """Keep polling for new checkpoints until the final training step
    (training.steps) has been evaluated."""

    poll_interval: int = 300
    """Seconds between checks for new checkpoints in watch mode."""

    max_idle_hours: float = 24.0
    """In watch mode, stop after this long without a new checkpoint to
    evaluate, e.g. when the training run died before its final step."""

    limit: float | None = None
    """Examples per task (a fraction if < 1). For quick checks only."""

    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    """Dtype the model is evaluated in."""

    tensor_parallel_degree: int = 1
    """Shard each model replica over this many GPUs, for models that do not
    fit on one. The remaining GPUs form data-parallel replicas."""

    wandb: bool = False
    """Also log results to W&B, next to the training run (see wandb_logger.py)."""

    num_tokens_per_batch: int | None = None
    """Tokens per packed forward pass. Default: max(16384, context length)."""

    output_folder: str | None = None
    """Where to write results. Default: <dump_folder>/evals."""

    def __post_init__(self) -> None:
        if self.watch and self.steps:
            raise ValueError("watch and steps are mutually exclusive.")


def parse_args(argv: list[str]) -> tuple[EvalConfig, list[str]]:
    if "--" not in argv:
        raise ValueError(
            "Pass the training config after '--', e.g. "
            "'-- --module llama3 --config llama3_8b'."
        )
    split = argv.index("--")
    return tyro.cli(EvalConfig, args=argv[:split]), argv[split + 1 :]


def init_distributed() -> torch.device:
    device = torch.device(utils.device_type, int(os.environ.get("LOCAL_RANK", 0)))
    utils.device_module.set_device(device)
    # gloo carries lm-eval's object gathers; nccl carries tensor collectives.
    dist.init_process_group(backend=f"cpu:gloo,{device.type}:nccl")
    return device


def broadcast_from_rank0(fn: Callable[[], T]) -> T:
    """Evaluate ``fn`` on rank 0 only and return its result on every rank."""
    value = [fn() if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(value, src=0)
    return value[0]  # pyrefly: ignore [bad-return]


def suite_name(tasks: list[str]) -> str:
    return "+".join(tasks)


def expand_tasks(names: list[str], task_manager: TaskManager) -> list[str]:
    """The units results are stored under: tasks, plus groups that aggregate.

    Suite groups such as trainstation_full only list other tasks, so they are
    expanded into their members; each member then gets its own results file,
    and a task shared by two suites is evaluated once per step. Groups with
    aggregate metrics (e.g. MMLU) stay whole, because the aggregate needs all
    their subtasks in one evaluation.
    """
    units: list[str] = []
    for name in names:
        entry = task_manager.task_index.get(name)
        cfg = entry.cfg if entry is not None and isinstance(entry.cfg, dict) else {}
        members = cfg.get("task")
        is_suite = (
            entry is not None
            and entry.kind.name == "GROUP"
            and not cfg.get("aggregate_metric_list")
            and isinstance(members, list)
            # Members with per-task overrides (e.g. num_fewshot) only apply
            # inside the group, so such groups stay whole.
            and all(isinstance(member, str) for member in members)
        )
        units.extend(expand_tasks(members, task_manager) if is_suite else [name])
    return list(dict.fromkeys(units))


def results_path(output_folder: str, step: int, task: str) -> str:
    return os.path.join(output_folder, f"step-{step}", f"{task}.json")


def missing_tasks(output_folder: str, step: int, tasks: list[str]) -> list[str]:
    return [
        task
        for task in tasks
        if not os.path.exists(results_path(output_folder, step, task))
    ]


def pending_work(
    args: EvalConfig, tasks: list[str], ckpt_folder: str, output_folder: str
) -> list[tuple[int, list[str]]]:
    """(step, tasks without results) for each checkpoint that has work left."""
    # Rank 0 lists and broadcasts, so a checkpoint or results file that appears
    # mid-listing cannot give ranks different work lists.
    return broadcast_from_rank0(
        lambda: [
            (step, missing)
            for step in (args.steps or list_checkpoint_steps(ckpt_folder))
            if (missing := missing_tasks(output_folder, step, tasks))
        ]
    )


def write_on_rank0(path: str, record: dict[str, Any]) -> None:
    """Write ``record`` on rank 0; every rank raises if the write fails.

    Other ranks would otherwise block on the next collective while rank 0
    exits, and the job would hang instead of failing.
    """
    error = None
    if dist.get_rank() == 0:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Write then rename so pending_work() never sees a partial file.
            with open(f"{path}.tmp", "w") as f:
                json.dump(record, f, indent=2, default=str)
            os.replace(f"{path}.tmp", path)
        except OSError as e:
            error = f"{type(e).__name__}: {e}"
    error = broadcast_from_rank0(lambda: error)
    if error is not None:
        raise RuntimeError(f"Rank 0 failed to write {path}: {error}")


def evaluate_step(
    step: int,
    tasks: list[str],
    *,
    args: EvalConfig,
    config: Trainer.Config,
    lm: TrainstationLM,
    task_manager: TaskManager,
    ckpt_folder: str,
    output_folder: str,
) -> dict[str, Any]:
    """Evaluate one checkpoint on ``tasks``, writing one results file per task.

    Returns the lm-eval results of all tasks, keyed by task and subtask.
    """
    checkpoint = os.path.join(ckpt_folder, f"step-{step}")
    logger.info(f"Evaluating {checkpoint} on {tasks}")
    load_weights(lm.model, checkpoint)

    all_results: dict[str, Any] = {}
    for task in tasks:
        start = time.perf_counter()
        results = simple_evaluate(
            model=lm,
            tasks=[task],
            limit=args.limit,
            task_manager=task_manager,
            log_samples=False,
        )
        record = None
        # Every TP rank of data-parallel rank 0 gets results; rank 0 writes.
        if dist.get_rank() == 0:
            assert results is not None
            record = {
                "step": step,
                "task": task,
                "checkpoint": checkpoint,
                "module": config.model_spec.name,
                "flavor": config.model_spec.flavor,
                "dtype": args.dtype,
                "tensor_parallel_degree": args.tensor_parallel_degree,
                "limit": args.limit,
                "lm_eval_version": lm_eval.__version__,
                "eval_seconds": time.perf_counter() - start,
                "results": results["results"],
                "groups": results.get("groups", {}),
                "n-shot": results.get("n-shot", {}),
                "versions": results.get("versions", {}),
            }
            all_results.update(results["results"])
        path = results_path(output_folder, step, task)
        write_on_rank0(path, record or {})
        logger.info(f"Wrote {path}")
    return all_results


def main() -> None:
    init_logger()
    args, train_args = parse_args(sys.argv[1:])
    config = ConfigManager().parse_args(train_args)
    device = init_distributed()

    ckpt_folder = checkpoint_folder(config)
    output_folder = args.output_folder or os.path.join(config.dump_folder, "evals")
    max_context_length = config.training.max_context_length

    config.parallelism = eval_parallelism(
        config.parallelism,
        world_size=dist.get_world_size(),
        tensor_parallel_degree=args.tensor_parallel_degree,
    )
    parallel_dims = ParallelDims.from_config(config.parallelism, dist.get_world_size())
    lm = TrainstationLM(
        # Checkpoint values are cast into these parameters on every load.
        build_model(
            config,
            parallel_dims=parallel_dims,
            device=device,
            dtype=args.dtype,
        ),
        config.tokenizer.build(tokenizer_path=config.hf_assets_path),
        parallel_dims=parallel_dims,
        parallelism=config.parallelism,
        max_context_length=max_context_length,
        num_tokens_per_batch=args.num_tokens_per_batch
        or max(16384, max_context_length),
        max_num_documents=getattr(config.dataloader, "max_num_documents", None),
        device=device,
    )
    task_manager = TaskManager(include_path=TASKS_DIR)
    tasks = expand_tasks(args.tasks, task_manager)
    logger.info(f"Results are stored per task: {tasks}")
    wandb_logger = None
    if args.wandb and dist.get_rank() == 0:
        wandb_logger = EvalWandBLogger(
            output_folder=output_folder,
            suite=suite_name(args.tasks),
            config={"eval": dataclasses.asdict(args), "train_args": train_args},
        )

    try:
        last_work = time.monotonic()
        while True:
            work = pending_work(args, tasks, ckpt_folder, output_folder)
            for step, step_tasks in work:
                results = evaluate_step(
                    step,
                    step_tasks,
                    args=args,
                    config=config,
                    lm=lm,
                    task_manager=task_manager,
                    ckpt_folder=ckpt_folder,
                    output_folder=output_folder,
                )
                if wandb_logger is not None:
                    wandb_logger.log(step, {"results": results})
            if work:
                last_work = time.monotonic()
            if not args.watch:
                break
            final_done = broadcast_from_rank0(
                lambda: not missing_tasks(output_folder, config.training.steps, tasks)
            )
            if final_done:
                break
            # Decided on rank 0 so all ranks stop together.
            idle = broadcast_from_rank0(
                lambda: time.monotonic() - last_work > args.max_idle_hours * 3600
            )
            if idle:
                logger.warning(
                    f"No new checkpoint for {args.max_idle_hours} hours and step "
                    f"{config.training.steps} has no results; stopping."
                )
                break
            if not work:
                time.sleep(args.poll_interval)
    finally:
        if wandb_logger is not None:
            wandb_logger.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
