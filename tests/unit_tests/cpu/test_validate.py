# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from torchtitan.components import validate as validate_module
from torchtitan.components.validate import Validator


class _ClosableLoader:
    def __init__(self, rows):
        self.rows = rows
        self.closed = False

    def __iter__(self):
        return iter(self.rows)

    def close(self):
        self.closed = True


class _EchoModel(nn.Module):
    def preprocess_inputs(self, input_dict, *, parallel_dims, parallelism):
        del parallel_dims, parallelism
        return input_dict["input"], input_dict["labels"], {}

    def forward(self, inputs, **kwargs):
        del kwargs
        return inputs


class _FailingModel(_EchoModel):
    def forward(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("validation failed")


class _FluxModel(nn.Module):
    def forward(self, **kwargs):
        return torch.zeros_like(kwargs["img"])


def _generic_validator(loader):
    validator = object.__new__(Validator)
    validator.config = SimpleNamespace(steps=1)
    validator.parallel_dims = SimpleNamespace(
        dp_enabled=False,
        pp_enabled=False,
        dp_cp_enabled=False,
    )
    validator.dl_config = SimpleNamespace(build=mock.Mock(return_value=loader))
    validator.dp_world_size = 1
    validator.dp_rank = 0
    validator.tokenizer = mock.Mock()
    validator.seq_len = 4
    validator.num_tokens_per_batch = 4
    validator.metrics_processor = SimpleNamespace(
        ntokens_since_last_log=0,
        log_validation=mock.Mock(),
    )
    validator.validation_context = nullcontext
    validator.loss_fn = lambda predictions, labels: (predictions.sum(), None)
    validator.parallelism = SimpleNamespace()
    return validator


@pytest.mark.parametrize("raises", [False, True])
def test_generic_validator_closes_temporary_loader(monkeypatch, raises):
    # A fresh dict per row: the validator pops num_valid_tokens.
    def row():
        return {
            "input": torch.ones(1, 1),
            "labels": torch.ones(1, 1, dtype=torch.long),
            "num_valid_tokens": 1,
        }

    loader = _ClosableLoader([row(), row()])
    validator = _generic_validator(loader)
    model = _FailingModel() if raises else _EchoModel()
    monkeypatch.setattr(validate_module.utils, "device_type", "cpu")

    if raises:
        with pytest.raises(RuntimeError, match="validation failed"):
            validator.validate([model], step=1)
    else:
        validator.validate([model], step=1)

    assert loader.closed
