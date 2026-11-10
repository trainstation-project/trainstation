# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Log eval results to W&B in a run of their own, next to the training run.

Reads the training logger's environment variables; see README.md.
"""

import hashlib
import math
import os
from typing import Any


def flatten_results(results: dict[str, Any], prefix: str = "eval") -> dict[str, float]:
    """``{"<prefix>/<task>/<metric>": value}`` for every numeric lm-eval metric.

    ``results`` is lm-eval's ``results`` dict. Metrics are keyed as
    ``"<metric>,<filter>"``; the filter is dropped when it is ``none``.
    Standard errors are kept as ``<metric>_stderr``.
    """
    metrics = {}
    for task, task_metrics in results.items():
        for key, value in task_metrics.items():
            numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
            if not numeric or "," not in key or math.isnan(value):
                continue
            metric, filter_name = key.split(",", 1)
            name = metric if filter_name == "none" else f"{metric}_{filter_name}"
            metrics[f"{prefix}/{task}/{name}"] = value
    return metrics


class EvalWandBLogger:
    """One W&B run per (results folder, suite), resumed across restarts.

    Eval results arrive out of order (backfills, restarts), so they are
    plotted against a ``train_step`` metric rather than W&B's own step.
    """

    def __init__(self, *, output_folder: str, suite: str, config: dict[str, Any]):
        import wandb

        base_name = os.getenv("WANDB_RUN_NAME") or os.path.basename(
            os.path.dirname(os.path.abspath(output_folder))
        )
        # Deterministic id: a restarted eval job continues the same run.
        run_id = hashlib.sha1(
            f"{os.path.abspath(output_folder)}:{suite}".encode()
        ).hexdigest()[:16]
        self.run = wandb.init(
            entity=os.getenv("WANDB_TEAM", None),
            project=os.getenv("WANDB_PROJECT", "torchtitan"),
            name=f"{base_name}-eval-{suite}",
            id=run_id,
            resume="allow",
            group=os.getenv("WANDB_RUN_GROUP", base_name),
            job_type="eval",
            dir=output_folder,
            config=config,
        )
        self.run.define_metric("train_step")
        self.run.define_metric("eval/*", step_metric="train_step")
        self.run.define_metric("eval_clean/*", step_metric="train_step")

    def log(
        self, step: int, results: dict[str, Any], clean_results: dict[str, Any]
    ) -> None:
        """Log ``results`` under eval/ and clean-subset results under eval_clean/."""
        metrics = flatten_results(results)
        metrics.update(flatten_results(clean_results, prefix="eval_clean"))
        self.run.log({"train_step": step, **metrics})

    def close(self) -> None:
        self.run.finish()
