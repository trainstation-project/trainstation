# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validate the MXFP8 and CUDA graph contracts without executing a model."""

from dataclasses import replace

import pytest

pytest.importorskip("torchao.prototype.moe_training")

from torchtitan.config.transform import quantization  # noqa: E402
from torchtitan.distributed import ParallelDims  # noqa: E402
from torchtitan.models.common.linear import RouterGateLinear  # noqa: E402
from torchtitan.models.common.moe import GroupedExperts  # noqa: E402
from torchtitan.models.common.token_dispatcher import (  # noqa: E402
    HybridEPTokenDispatcher,
)
from torchtitan.quantization.mxfp8 import MXFP8Linear  # noqa: E402
from torchtitan_recipes.qwen3 import qwen3_30b_a3b_mxfp8_fsdp8_ep8_b300  # noqa: E402


def test_qwen3_b300_recipe_preserves_graph_and_precision_contracts(monkeypatch):
    if MXFP8Linear is None:
        pytest.skip("The recipe requires TorchAO nightly MXFP8 kernels")
    monkeypatch.setattr(quantization, "has_cuda_capability", lambda *_args: True)
    config = qwen3_30b_a3b_mxfp8_fsdp8_ep8_b300()
    mesh = ParallelDims.from_config(config.parallelism, world_size=8)
    assert mesh.dp_shard * mesh.cp * mesh.tp // mesh.ep == 1
    assert config.checkpoint.enable and config.checkpoint.initial_load_in_hf
    assert not config.training.disable_cuda_graphs
    assert config.training.num_tokens_per_train_step == (
        config.training.num_tokens_per_microbatch_per_dp_rank * mesh.dp_shard
    )
    linears = list(config.model_spec.model.traverse(MXFP8Linear.Config))
    assert linears
    for fqn, _, _, _ in linears:
        assert "attention" in fqn
    experts = list(config.model_spec.model.traverse(GroupedExperts.Config))
    assert len(experts) == 48
    assert all(expert.recipe_name == "mxfp8_rceil" for _, expert, _, _ in experts)
    assert len(list(config.model_spec.model.traverse(RouterGateLinear.Config))) == 48
    dispatchers = [
        dispatcher
        for _, dispatcher, _, _ in config.model_spec.model.traverse(
            HybridEPTokenDispatcher.Config
        )
    ]
    assert len(dispatchers) == 48
    for dispatcher in dispatchers:
        assert dispatcher.pad_multiple == 128
        assert dispatcher.non_blocking_capacity_factor == 1.0
        assert dispatcher.num_max_tokens_per_rank == (
            config.training.num_tokens_per_microbatch_per_dp_rank
        )
    dispatchers[0].non_blocking_capacity_factor = None
    with pytest.raises(ValueError, match="without CPU synchronization"):
        replace(config)
