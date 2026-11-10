# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from unittest.mock import MagicMock

from torchtitan.experiments.evals.wandb_logger import EvalWandBLogger, flatten_results

RESULTS = {
    "hellaswag": {
        "alias": "hellaswag",
        "acc,none": 0.25,
        "acc_stderr,none": 0.01,
        "acc_norm,none": 0.3,
    },
    "gsm8k": {"exact_match,strict-match": 0.1},
    "trainstation_mmlu": {"acc,none": 0.4, "acc_stderr,none": "N/A"},
}


def test_flatten_results():
    assert flatten_results(RESULTS) == {
        "eval/hellaswag/acc": 0.25,
        "eval/hellaswag/acc_stderr": 0.01,
        "eval/hellaswag/acc_norm": 0.3,
        "eval/gsm8k/exact_match_strict-match": 0.1,
        "eval/trainstation_mmlu/acc": 0.4,
    }


def _init_kwargs(monkeypatch, output_folder: str, suite: str) -> dict:
    wandb = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    EvalWandBLogger(output_folder=output_folder, suite=suite, config={})
    return wandb.init.call_args.kwargs


def test_logger_plots_against_train_step(monkeypatch, tmp_path):
    monkeypatch.setenv("WANDB_RUN_NAME", "run7")
    monkeypatch.delenv("WANDB_RUN_GROUP", raising=False)
    wandb = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    output_folder = str(tmp_path / "run7" / "evals")
    logger = EvalWandBLogger(output_folder=output_folder, suite="quick", config={})
    logger.log(300, RESULTS, {})

    kwargs = wandb.init.call_args.kwargs
    assert kwargs["name"] == "run7-eval-quick"
    assert kwargs["group"] == "run7"
    run = wandb.init.return_value
    run.define_metric.assert_any_call("eval/*", step_metric="train_step")
    run.log.assert_called_once_with({"train_step": 300, **flatten_results(RESULTS)})

    # A restarted job resumes the same run; another suite gets its own.
    assert _init_kwargs(monkeypatch, output_folder, "quick")["id"] == kwargs["id"]
    assert _init_kwargs(monkeypatch, output_folder, "full")["id"] != kwargs["id"]
