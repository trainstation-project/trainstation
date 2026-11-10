# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the ``models`` integration test suite."""

from torchtitan.config.transform import apply_transforms, ContextParallelTransform
from torchtitan.distributed.activation_checkpoint import RegionAC, SelectiveAC

from torchtitan.models.common.cp_attention import KVAllGatherCPFlexInnerAttention
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_debugmodel,
    deepseek_v3_debugmodel_mtp,
)
from torchtitan.models.llama3.config_registry import llama3_debugmodel
from torchtitan.models.qwen3.config_registry import (
    qwen3_debugmodel,
    qwen3_debugmodel_moe_param_groups,
)
from torchtitan.trainer import Trainer

from . import _set_spmd_typechecking


def _configure_fsdp_numerics(
    config: Trainer.Config, *, expert_parallel_degree: int = 1
) -> Trainer.Config:
    """Use a logical-world-eight FSDP topology for numerical comparison."""
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.context_parallel_degree = 1
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.parallelism.expert_parallel_degree = expert_parallel_degree
    config.training.max_context_length = 512
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return config


def llama3_debugmodel_fsdp2_tp2_cp2() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.training.max_context_length = 512
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return apply_transforms(
        config,
        [ContextParallelTransform(inner_attention=KVAllGatherCPFlexInnerAttention)],
    )


def llama3_debugmodel_region_ac_fsdp2_tp2_cp2() -> Trainer.Config:
    config = llama3_debugmodel_fsdp2_tp2_cp2()
    config.activation_checkpoint = RegionAC.Config(
        save_regions=[
            "attention.qkv",
            "attention.inner_attention",
            "attention.wo",
        ]
    )
    return config


def llama3_debugmodel_fsdp2_tp2_pp2() -> Trainer.Config:
    config = llama3_debugmodel(seq_len=2048)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 2
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.training.max_context_length = 512
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return config


def deepseek_v3_debugmodel_mtp_fsdp4_ep2_compile() -> Trainer.Config:
    config = deepseek_v3_debugmodel_mtp(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 2
    config.compile.enable = True
    config.override.imports = [
        "torchtitan.overrides.helion_rope.helion_cos_sin_rope",
        "torchtitan.overrides.helion_rope.helion_complex_rope",
    ]
    config.training.disable_cuda_graphs = True
    return config


def deepseek_v3_debugmodel_mtp_cp2() -> Trainer.Config:
    config = deepseek_v3_debugmodel_mtp(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.context_parallel_degree = 2
    config.training.max_context_length = 512
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return apply_transforms(
        config,
        [ContextParallelTransform(inner_attention=KVAllGatherCPFlexInnerAttention)],
    )


def deepseek_v3_debugmodel_mtp_tp2_cp2() -> Trainer.Config:
    config = deepseek_v3_debugmodel_mtp_cp2()
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.enable_sequence_parallel = True
    return config


def deepseek_v3_debugmodel_fsdp8_ep8() -> Trainer.Config:
    return _configure_fsdp_numerics(
        deepseek_v3_debugmodel(seq_len=2048), expert_parallel_degree=8
    )


def deepseek_v3_debugmodel_fsdp2_tp2_cp2_ep8() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 8
    config.training.max_context_length = 512
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return apply_transforms(
        config,
        [ContextParallelTransform(inner_attention=KVAllGatherCPFlexInnerAttention)],
    )


def deepseek_v3_debugmodel_fsdp2_cp2_pp2_ep4() -> Trainer.Config:
    config = deepseek_v3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    config.training.disable_cuda_graphs = True
    return apply_transforms(
        config,
        [ContextParallelTransform(inner_attention=KVAllGatherCPFlexInnerAttention)],
    )


def deepseek_v3_debugmodel_hsdp2x2_ep2() -> Trainer.Config:
    config = deepseek_v3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_replicate_degree = 2
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.training.disable_cuda_graphs = True
    return config


def deepseek_v4_debugmodel_fsdp2_tp2_ep2() -> Trainer.Config:
    from torchtitan.models.deepseek_v4.config_registry import deepseek_v4_debugmodel

    config = deepseek_v4_debugmodel(seq_len=512)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.parallelism.context_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.training.max_context_length = 512
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return config


def deepseek_v3_debugmodel_fused_mla_swiglu_fsdp4_ep2() -> Trainer.Config:
    config = deepseek_v3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=True)
    config.override.imports = [
        "torchtitan.overrides.fused_mla.fused_mla",
        "torchtitan.overrides.fused_swiglu.fused_swiglu",
    ]
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.expert_parallel_degree = 2
    config.training.disable_cuda_graphs = True
    return config


def qwen3_debugmodel_moe_param_groups_fsdp2_tp2_ep4() -> Trainer.Config:
    config = qwen3_debugmodel_moe_param_groups(seq_len=4096)
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    config.training.disable_cuda_graphs = True
    return config


def qwen3_debugmodel_moe_param_groups_fsdp2_tp2_cp2_ep8() -> Trainer.Config:
    config = qwen3_debugmodel_moe_param_groups(seq_len=4096)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 8
    config.training.max_context_length = 512
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return apply_transforms(
        config,
        [ContextParallelTransform(inner_attention=KVAllGatherCPFlexInnerAttention)],
    )


def qwen3_debugmodel_fsdp2_tp2_cp2() -> Trainer.Config:
    config = qwen3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    return apply_transforms(
        config,
        [ContextParallelTransform(inner_attention=KVAllGatherCPFlexInnerAttention)],
    )


def qwen3_debugmodel_fsdp2_tp2_cp2_no_sp() -> Trainer.Config:
    config = qwen3_debugmodel_fsdp2_tp2_cp2()
    config.parallelism.enable_sequence_parallel = False
    return config


def qwen3_debugmodel_fsdp2_tp2_cp2_compile_helion_rope() -> Trainer.Config:
    config = qwen3_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.compile.enable = True
    config.override.imports = ["torchtitan.overrides.helion_rope.helion_cos_sin_rope"]
    return apply_transforms(
        config,
        [ContextParallelTransform(inner_attention=KVAllGatherCPFlexInnerAttention)],
    )


def qwen35_debugmodel_moe_fsdp2_tp2_pp2_ep4() -> Trainer.Config:
    from torchtitan.models.qwen3_5.config_registry import qwen35_debugmodel_moe

    config = qwen35_debugmodel_moe(seq_len=512)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    _set_spmd_typechecking(config, typechecking=False)
    config.training.disable_cuda_graphs = True
    return config


def qwen35_debugmodel_moe_fsdp4_tp2_ep4() -> Trainer.Config:
    from torchtitan.models.qwen3_5.config_registry import qwen35_debugmodel_moe

    config = qwen35_debugmodel_moe(seq_len=512)
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    config.parallelism.pipeline_parallel_degree = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = (
        config.training.max_context_length
    )
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return config


def qwen35_debugmodel_varlen_attn_fsdp2_tp2_sac() -> Trainer.Config:
    from torchtitan.models.qwen3_5.config_registry import qwen35_debugmodel_varlen_attn

    config = qwen35_debugmodel_varlen_attn(seq_len=512)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.activation_checkpoint = SelectiveAC.Config()
    _set_spmd_typechecking(config, typechecking=False)
    config.training.disable_cuda_graphs = True
    return config


# GptBase mixes GQA/MLA attention with dense MLP and MoE (with and without
# shared experts) in one model, so these layouts cover every layer type at once.


def gpt_base_debugmodel_fsdp2_tp2_ep2_spmd_typecheck() -> Trainer.Config:
    from torchtitan.models.gpt_base.config_registry import gpt_base_debugmodel

    config = gpt_base_debugmodel(seq_len=512)
    _set_spmd_typechecking(config, typechecking=True)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.enable_sequence_parallel = True
    config.parallelism.expert_parallel_degree = 2
    config.training.disable_cuda_graphs = True
    return config


def gpt_base_debugmodel_fsdp4_tp2_ep4_compile() -> Trainer.Config:
    from torchtitan.models.gpt_base.config_registry import gpt_base_debugmodel

    config = gpt_base_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 4
    config.compile.enable = True
    config.training.disable_cuda_graphs = True
    return config


def gpt_base_debugmodel_fsdp2_cp2_pp2_ep4_ptrr_sac() -> Trainer.Config:
    from torchtitan.models.gpt_base.config_registry import gpt_base_debugmodel

    config = gpt_base_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = "ptrr"
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.expert_parallel_degree = 4
    config.activation_checkpoint = SelectiveAC.Config()
    config.training.max_context_length = 512
    config.training.num_tokens_per_microbatch_per_dp_rank = 1024
    config.training.steps = 10
    config.training.disable_cuda_graphs = True
    return apply_transforms(
        config,
        [ContextParallelTransform(inner_attention=KVAllGatherCPFlexInnerAttention)],
    )


def gpt_base_debugmodel_varlen_fsdp4_pp2_ep4_sac() -> Trainer.Config:
    from torchtitan.models.gpt_base.config_registry import gpt_base_debugmodel_varlen

    config = gpt_base_debugmodel_varlen(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 8
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.expert_parallel_degree = 4
    config.activation_checkpoint = SelectiveAC.Config()
    config.training.num_tokens_per_microbatch_per_dp_rank = 1024
    config.training.num_tokens_per_train_step = 131072
    config.training.disable_cuda_graphs = True
    return config


def gpt_base_debugmodel_tp2_cp2_pp2_sp() -> Trainer.Config:
    from torchtitan.models.gpt_base.config_registry import gpt_base_debugmodel

    config = gpt_base_debugmodel(seq_len=2048)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.enable_sequence_parallel = True
    config.parallelism.context_parallel_degree = 2
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.num_pp_microbatches = 2
    config.parallelism.pipeline_parallel_schedule = "1F1B"
    config.training.max_context_length = 512
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    config.training.disable_cuda_graphs = True
    return apply_transforms(
        config,
        [ContextParallelTransform(inner_attention=KVAllGatherCPFlexInnerAttention)],
    )


def _gpt_base_dist_muon(
    *, data_parallel_shard_degree: int, expert_parallel_degree: int
) -> Trainer.Config:
    from torchtitan.models.gpt_base.config_registry import (
        dist_muon_optimizer,
        gpt_base_debugmodel,
    )

    config = gpt_base_debugmodel(seq_len=512)
    config.parallelism.data_parallel_shard_degree = data_parallel_shard_degree
    config.parallelism.expert_parallel_degree = expert_parallel_degree
    config.training.disable_cuda_graphs = True
    # Built after the parallelism above: routed-expert compute layouts depend
    # on whether EP is enabled.
    config.optimizer = dist_muon_optimizer(
        config.model_spec, lr=8e-4, parallelism=config.parallelism
    )
    return config


def gpt_base_debugmodel_dist_muon_fsdp2_pp2_ep2() -> Trainer.Config:
    """DistMuon with PP: each stage consumes its local subset of the global
    compute-sharding map. Interleaved1F1B with two stages per rank gives each
    of the four virtual stages two layers."""
    config = _gpt_base_dist_muon(data_parallel_shard_degree=2, expert_parallel_degree=2)
    _set_spmd_typechecking(config, typechecking=False)
    config.parallelism.pipeline_parallel_degree = 2
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    config.parallelism.num_pp_microbatches = 4
    return config


def gpt_base_debugmodel_dist_muon_fsdp8_ep8() -> Trainer.Config:
    return _gpt_base_dist_muon(data_parallel_shard_degree=8, expert_parallel_degree=8)
