# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

from torch.distributed.tensor import Shard

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optimizer import (
    default_adamw,
    LRSchedulersContainer,
    OptimizersContainer,
    ParamGroupConfig,
)
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.flex_shard import BucketConfig, ComputeLayout, Owned
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.config_utils import (
    decoder_vocab_size,
    DEFAULT_DEBUG_MODEL_SEQ_LEN,
)
from torchtitan.models.deepseek_v3.model import Attention as MLAttention
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from . import model_registry
from .model import GptTrainstationModel


def gpt_trainstation_debugmodel(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    model_spec = model_registry("debugmodel", seq_len=seq_len)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            shuffle=False,
        ),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8 * model_spec.max_context_length,
            max_context_length=model_spec.max_context_length,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )


def gpt_trainstation_debugmodel_varlen(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    config = gpt_trainstation_debugmodel(seq_len=seq_len)
    config.model_spec = model_registry(
        "debugmodel", seq_len=seq_len, attn_backend="varlen"
    )
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    config.dataloader.max_num_documents = 64
    return config


def _routed_expert_compute_layout(parallelism: ParallelismConfig) -> ComputeLayout:
    """Compute layout for stacked routed-expert weights.

    Without EP, experts are FSDP-sharded on ``dp_shard`` along the expert
    dimension. With EP, EP splits the expert dimension first and EFSDP
    repartitions each EP-local expert domain.
    """
    if parallelism.expert_parallel_degree == 1:
        return ComputeLayout(
            shardings_by_mesh_axis={MeshAxisName.DP_SHARD.value: Shard(0)},
        )
    return ComputeLayout(
        shardings_by_mesh_axis={
            MeshAxisName.EFSDP.value: Shard(0),
            MeshAxisName.EP.value: Shard(0),
        },
        shard_order_by_tensor_dim={
            0: (MeshAxisName.EP.value, MeshAxisName.EFSDP.value),
        },
    )


def dist_muon_optimizer(
    model_spec: ModelSpec,
    *,
    lr: float,
    parallelism: ParallelismConfig,
) -> OptimizersContainer.Config:
    """DistMuon on every matrix weight, AdamW on everything else.

    Dense matrices use ``Owned`` (one rank runs Newton-Schulz on the full
    matrix); routed experts use the expert-dimension layout. One bucket per
    layer, with routed experts in their own bucket since their layout differs.

    DistMuon does not support tensor parallel (TP produces ``_StridedShard``
    storage), and every PP stage must own at least one layer.
    """
    if parallelism.tensor_parallel_degree > 1:
        raise ValueError("DistMuon requires tensor_parallel_degree=1.")

    model_config = cast(GptTrainstationModel.Config, model_spec.model)
    owned = ComputeLayout(
        shardings_by_mesh_axis={MeshAxisName.DP_SHARD.value: Owned()},
    )
    per_expert = _routed_expert_compute_layout(parallelism)

    bucket_configs = []
    compute_sharding_by_fqn: dict[str, ComputeLayout] = {}
    for layer_id, layer in enumerate(model_config.layers):
        prefix = f"layers.{layer_id}"
        attention = layer.attention
        if isinstance(attention, MLAttention.Config):
            projections = ["wkv_a", "wkv_b", "wo"]
            projections += ["wq_a", "wq_b"] if attention.q_lora_rank else ["wq"]
        else:
            assert isinstance(attention, GQAttention.Config)
            projections = ["qkv_linear.wqkv", "wo"]
        dense_fqns = [f"{prefix}.attention.{p}.weight" for p in projections]

        if layer.feed_forward is not None:
            dense_fqns += [f"{prefix}.feed_forward.{p}.weight" for p in ("w13", "w2")]
        routed_fqns = []
        if layer.moe is not None:
            dense_fqns.append(f"{prefix}.moe.router.gate.weight")
            if layer.moe.shared_experts is not None:
                dense_fqns += [
                    f"{prefix}.moe.shared_experts.{p}.weight" for p in ("w13", "w2")
                ]
            routed_fqns = [
                f"{prefix}.moe.routed_experts.inner_experts.{p}"
                for p in ("w1_EFD", "w2_EDF", "w3_EFD")
            ]

        compute_sharding_by_fqn.update({fqn: owned for fqn in dense_fqns})
        compute_sharding_by_fqn.update({fqn: per_expert for fqn in routed_fqns})
        bucket_configs.append(BucketConfig(name=prefix, patterns=tuple(dense_fqns)))
        if routed_fqns:
            bucket_configs.append(
                BucketConfig(
                    name=f"{prefix}.routed-experts", patterns=tuple(routed_fqns)
                )
            )

    # Muon is designed for matrix parameters; norms, embeddings and the LM head
    # stay on AdamW (Moonlight, Sec. 2.2: https://arxiv.org/abs/2502.16982).
    muon_pattern = (
        "(?:" + "|".join(fqn.replace(".", r"\.") for fqn in compute_sharding_by_fqn)
    ) + ")$"
    return OptimizersContainer.Config(
        implementation="foreach",
        param_groups=[
            ParamGroupConfig(
                pattern=muon_pattern,
                optimizer_name="DistMuon",
                optimizer_kwargs={
                    "lr": lr,
                    "weight_decay": 0.1,
                    "foreach": False,
                    "adjust_lr_fn": "match_rms_adamw",
                },
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={
                    "lr": lr,
                    "betas": (0.9, 0.95),
                    "eps": 1e-8,
                    "weight_decay": 0.1,
                },
            ),
        ],
        optimizer_factory_kwargs_by_name={
            "DistMuon": {
                "bucket_configs": tuple(bucket_configs),
                "compute_sharding_by_fqn": compute_sharding_by_fqn,
            }
        },
    )
