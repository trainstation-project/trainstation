# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch

from torchtitan.config import TrainingConfig
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    validate_autoparallel_config,
)


class _FakeMesh:
    def __init__(self, mesh_axis_names, size=2):
        self.mesh_dim_names = tuple(mesh_axis_names)
        self.ndim = len(self.mesh_dim_names)
        self._size = size

    def size(self):
        return self._size


class _FakeParallelDims:
    dp_replicate_enabled = False
    cp_enabled = False
    pp_enabled = False
    tp_enabled = True
    dp_replicate = 1
    dp_shard = 2

    def __init__(self, *, sparse: bool = False):
        self.sparse = sparse
        self.tp_enabled = not sparse

    def get_optional_mesh(self, name):
        enabled = {"dp_shard", "tp"} if not self.sparse else {"efsdp", "ep"}
        return _FakeMesh((name,)) if name in enabled else None

    def get_mesh(self, names):
        if isinstance(names, str):
            return _FakeMesh((names,))
        return _FakeMesh(tuple(names))


class _FakeAutoParallelGraph:
    instances = []

    def __init__(self, model, input_fn, mesh, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.used_fx_path = False
        _FakeAutoParallelGraph.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_parameter_memory_constraint(self, *, low, high):
        pass

    def add_input_constraints(self, constraints):
        pass

    def add_output_constraints(self, constraints):
        pass

    def optimize_placement(self, verbose=False):
        return object()

    def apply_placement_for_fx_module(self, *args, **kwargs):
        self.used_fx_path = True
        self.apply_kwargs = kwargs
        return torch.nn.Linear(1, 1)


def _training_config():
    return TrainingConfig(
        num_tokens_per_microbatch_per_dp_rank=2 * 8,
        max_context_length=8,
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
    )


def test_autoparallel_config_validation():
    with pytest.raises(ValueError, match="only supports --compile.mode aot_fx_trace"):
        validate_autoparallel_config(
            GraphTrainerCompileConfig(
                mode="jit",
                enable_autoparallel=True,
            )
        )

    compile_config = GraphTrainerCompileConfig(
        inductor_compilation="regional",
        enable_autoparallel=True,
    )
    validate_autoparallel_config(compile_config)


def test_autoparallel_graph_pass_selection_uses_regular_memory_policy():
    from torchtitan.experiments.graph_trainer import passes

    traced_result = SimpleNamespace(
        gm=torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph()),
        state_fqns=[],
    )
    config = SimpleNamespace(
        compile=GraphTrainerCompileConfig(
            enable_autoparallel=True,
            enable_async_tensor_parallel=False,
            disable_passes=["cudagraph_pass"],
        ),
        model_spec=SimpleNamespace(model=SimpleNamespace(layers=[object()])),
        parallelism=SimpleNamespace(
            fsdp_reshard_after_forward="always",
            pipeline_parallel_degree=1,
        ),
    )

    graph_passes = passes.construct_default_graph_passes(traced_result, config)
    pass_fns = [getattr(pass_fn, "func", pass_fn) for pass_fn in graph_passes]

    assert passes.tag_with_memory_policy_pass in pass_fns
    assert passes.selective_activation_remat_pass in pass_fns
    assert passes.apply_cpu_offload_pass in pass_fns
    assert passes.joint_transformer_block_bucketing_reordering_pass in pass_fns
