# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Find eval questions that also appear in a run's training data.

An eval question counts as contaminated when one of its 13-word sequences
(13-grams) also occurs in a training document, the test used for GPT-3 and by
lm-eval's decontamination tools. Text is lowercased and stripped of
punctuation before comparing, with lm-eval's Janitor normalization.

The training data is read through the training config's own dataloader and
decoded back to text, so the check covers exactly what the run trains on,
whatever the data source. By default it scans the run's whole token budget
(training.steps x training.num_tokens_per_train_step).

Usage (CPU only; each process scans its own share of the data)::

    torchrun --nproc_per_node=32 -m torchtitan.experiments.evals.contamination \\
        --tasks trainstation_full [--max-tokens 1000000000] \\
        -- --module llama3 --config llama3_8b [training overrides]

The report goes to ``<dump_folder>/evals/contamination.json``. The eval job's
``--decontaminate`` option then also scores each task on its clean questions.
"""

import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist
import tyro
from lm_eval.decontamination.janitor import Janitor
from lm_eval.tasks import TaskManager
from lm_eval.utils import apply_template

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import ConfigManager
from torchtitan.experiments.evals.evaluate import (
    broadcast_from_rank0,
    expand_tasks,
    TASKS_DIR,
)
from torchtitan.observability.logging import init_logger

logger = logging.getLogger(__name__)

_JANITOR = Janitor()


@dataclass(kw_only=True, slots=True)
class ContaminationConfig:
    """Options of the contamination scan; the training config follows '--'."""

    tasks: list[str]
    """lm-eval task or group names, as passed to the eval job."""

    ngram_size: int = 13
    """Words per n-gram. Questions shorter than this cannot be checked."""

    max_tokens: int | None = None
    """Training tokens to scan across all processes. Default: the run's token
    budget, training.steps x training.num_tokens_per_train_step."""

    output_folder: str | None = None
    """Where to write contamination.json. Default: <dump_folder>/evals."""


def ngrams(text: str, n: int) -> Iterator[int]:
    """Hashes of the normalized word n-grams of ``text``."""
    words = _JANITOR.normalize_string(text).split()
    for i in range(len(words) - n + 1):
        yield hash(tuple(words[i : i + n]))


def eval_doc_text(task: Any, doc: dict[str, Any]) -> str:
    """The text of an eval question that training data could contain."""
    if task.config.should_decontaminate:
        query = task.config.doc_to_decontamination_query
        if isinstance(query, str) and query not in task.features:
            # lm-eval's doc_to_decontamination_query passes a rendered template
            # through ast.literal_eval, which fails on plain text (e.g. the
            # WikiText pages), so render string templates here.
            return apply_template(query, doc)
        return task.doc_to_decontamination_query(doc)
    # Question, answer choices and target: whichever are strings. Multiple-input
    # tasks (e.g. WinoGrande) put the text in the choices instead.
    parts = [task.doc_to_text(doc), task.doc_to_target(doc)]
    if task.OUTPUT_TYPE == "multiple_choice":
        parts.extend(task.doc_to_choice(doc))
    return " ".join(part for part in parts if isinstance(part, str))


class EvalIndex:
    """Maps n-gram hashes to the eval questions that contain them."""

    def __init__(self, ngram_size: int) -> None:
        self.ngram_size = ngram_size
        self.index: dict[int, list[tuple[str, int]]] = {}
        # Per task: number of questions, and of questions long enough to check.
        self.num_docs: dict[str, int] = {}
        self.num_checked: dict[str, int] = {}

    def add(self, task_name: str, doc_id: int, text: str) -> None:
        self.num_docs[task_name] = self.num_docs.get(task_name, 0) + 1
        hashes = set(ngrams(text, self.ngram_size))
        if hashes:
            self.num_checked[task_name] = self.num_checked.get(task_name, 0) + 1
        for h in hashes:
            self.index.setdefault(h, []).append((task_name, doc_id))

    def match(self, text: str) -> set[tuple[str, int]]:
        """Eval questions sharing an n-gram with ``text``."""
        found: set[tuple[str, int]] = set()
        for h in ngrams(text, self.ngram_size):
            found.update(self.index.get(h, ()))
        return found


def collect_eval_docs(
    tasks: list[str], task_manager: TaskManager
) -> tuple[list[tuple[str, int, str]], dict[str, list[str]]]:
    """(task, doc id, text) of every evaluated question of ``tasks``.

    Also returns, per results unit (see evaluate.expand_tasks), the lm-eval
    tasks it contains, which is how clean subsets are selected later.
    """
    docs: list[tuple[str, int, str]] = []
    units: dict[str, list[str]] = {}
    for unit in expand_tasks(tasks, task_manager):
        loaded = task_manager.load([unit])["tasks"]
        units[unit] = sorted(loaded)
        for name, task in loaded.items():
            # Same order as lm-eval's doc ids, which --decontaminate selects by.
            for doc_id, doc in enumerate(task.eval_docs):
                docs.append((name, doc_id, eval_doc_text(task, doc)))
    return docs, units


def batch_documents(batch: dict[str, Any], tokenizer: BaseTokenizer) -> list[str]:
    """Decode a training batch into the text of each packed document.

    Positions restart at 0 at each document start, and padding is dropped, so
    n-grams never span two documents.
    """
    tokens = batch["input"].reshape(-1)
    keep = ~batch["padding_mask"].reshape(-1) if "padding_mask" in batch else None
    if "positions" in batch:
        starts = (batch["positions"].reshape(-1) == 0).nonzero().flatten().tolist()
    else:
        starts = [0]
    bounds = sorted(set(starts) | {0}) + [tokens.numel()]
    documents = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        segment = tokens[start:end]
        if keep is not None:
            segment = segment[keep[start:end]]
        if segment.numel():
            documents.append(tokenizer.decode(segment.tolist()))
    return documents


def token_budget(args: ContaminationConfig, config: Any) -> int:
    if args.max_tokens is not None:
        return args.max_tokens
    per_step = config.training.num_tokens_per_train_step
    if per_step <= 0:
        raise ValueError(
            "The training config does not set training.num_tokens_per_train_step, "
            "so its token budget is unknown; pass --max-tokens."
        )
    return config.training.steps * per_step


def main() -> None:
    init_logger()
    argv = sys.argv[1:]
    if "--" not in argv:
        raise ValueError("Pass the training config after '--'.")
    split = argv.index("--")
    args = tyro.cli(ContaminationConfig, args=argv[:split])
    train_args = argv[split + 1 :]
    config = ConfigManager().parse_args(train_args)

    dist.init_process_group("gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    output_folder = args.output_folder or os.path.join(config.dump_folder, "evals")

    start = time.perf_counter()
    # Only rank 0 loads the benchmarks: every rank loading ~70 datasets from
    # the Hugging Face Hub at once exceeds its request rate limit (see
    # evaluate.cache_datasets_then_go_offline).
    docs, units = broadcast_from_rank0(
        lambda: collect_eval_docs(args.tasks, TaskManager(include_path=TASKS_DIR))
    )
    index = EvalIndex(args.ngram_size)
    for task_name, doc_id, text in docs:
        index.add(task_name, doc_id, text)
    logger.info(
        f"Indexed {sum(index.num_docs.values())} eval questions "
        f"({len(index.index)} {args.ngram_size}-grams) in "
        f"{time.perf_counter() - start:.0f}s"
    )

    tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
    num_tokens_per_batch = config.training.num_tokens_per_microbatch_per_dp_rank
    dataloader = config.dataloader.build(
        dp_world_size=world_size,
        dp_rank=rank,
        tokenizer=tokenizer,
        max_context_length=config.training.max_context_length,
        num_tokens_per_batch=num_tokens_per_batch,
    )
    budget = token_budget(args, config)
    num_batches = -(-budget // (num_tokens_per_batch * world_size))

    found: set[tuple[str, int]] = set()
    num_documents = 0
    start = time.perf_counter()
    for step, batch in enumerate(dataloader):
        if step >= num_batches:
            break
        for document in batch_documents(batch, tokenizer):
            found |= index.match(document)
            num_documents += 1
        if rank == 0 and (step + 1) % 100 == 0:
            logger.info(
                f"Scanned {step + 1}/{num_batches} batches per process, "
                f"{(step + 1) * num_tokens_per_batch / (time.perf_counter() - start):.0f} "
                f"tokens/s per process"
            )
    dataloader.close()

    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, (sorted(found), num_documents))
    if rank == 0:
        contaminated: dict[str, set[int]] = {}
        for rank_found, _ in gathered:
            for task_name, doc_id in rank_found:
                contaminated.setdefault(task_name, set()).add(doc_id)
        report = {
            "ngram_size": args.ngram_size,
            "scanned_tokens": num_batches * num_tokens_per_batch * world_size,
            "scanned_documents": sum(n for _, n in gathered),
            "train_args": train_args,
            "units": units,
            "tasks": {
                name: {
                    "num_docs": index.num_docs[name],
                    # Questions shorter than ngram_size words cannot be checked
                    # and count as clean.
                    "num_checked": index.num_checked.get(name, 0),
                    "num_contaminated": len(contaminated.get(name, ())),
                    "contaminated": sorted(contaminated.get(name, ())),
                }
                for name in sorted(index.num_docs)
            },
        }
        os.makedirs(output_folder, exist_ok=True)
        path = os.path.join(output_folder, "contamination.json")
        with open(f"{path}.tmp", "w") as f:
            json.dump(report, f, indent=1)
        os.replace(f"{path}.tmp", path)
        for name, entry in report["tasks"].items():
            if entry["num_contaminated"]:
                logger.info(
                    f"{name}: {entry['num_contaminated']} of {entry['num_docs']} "
                    "questions overlap the training data"
                )
        logger.info(
            f"Scanned {report['scanned_tokens']} tokens "
            f"({report['scanned_documents']} documents); wrote {path}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
