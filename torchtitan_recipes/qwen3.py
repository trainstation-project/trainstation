# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 recipes for single-node B300 training."""

from dataclasses import replace
from pathlib import Path
from typing import cast

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    IndexedJsonlSource,
    SingleDatasetConfig,
)
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.config.transform import (
    MXFP8GroupedExpertsConverter,
    MXFP8LinearConverter,
)
from torchtitan.hf_datasets.text_datasets import TextProcessor
from torchtitan.models.common.token_dispatcher import HybridEPTokenDispatcher
from torchtitan.models.qwen3 import model_registry
from torchtitan.models.qwen3.config_registry import qwen3_30b_a3b
from torchtitan.trainer import Trainer


def qwen3_30b_a3b_mxfp8_fsdp8_ep8_b300() -> Trainer.Config:
    """Prepare a 100-step continued-pretraining sample with CUDA graph replay.

    Requires the model and C4 sample from scripts/prepare_qwen3_30b_b300.py.
    This recipe is not a measured B300 performance or convergence result.
    """
    root = Path(__file__).resolve().parents[1]
    assets = root / "assets/hf/Qwen3-30B-A3B"
    config = qwen3_30b_a3b(seq_len=8192)
    model_spec = model_registry(
        "30B-A3B",
        seq_len=8192,
        moe_comm_backend="hybridep",
        converters=[
            MXFP8LinearConverter.Config(fqns=["attention"], model_compile_enabled=True),
            MXFP8GroupedExpertsConverter.Config(
                recipe_name="mxfp8_rceil",
                pad_multiple=128,
                model_compile_enabled=True,
            ),
        ],
    )
    for _, dispatcher, _, _ in model_spec.model.traverse(
        HybridEPTokenDispatcher.Config
    ):
        dispatcher = cast(HybridEPTokenDispatcher.Config, dispatcher)
        dispatcher.non_blocking_capacity_factor = 1.0
        dispatcher.num_max_tokens_per_rank = 8192
    return replace(
        config,
        model_spec=model_spec,
        hf_assets_path=str(assets),
        dump_folder=str(root / "outputs/qwen3_30b_mxfp8_b300"),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=IndexedJsonlSource.Config(
                        patterns=(str(root / "data/qwen3_30b_c4_sample/train.jsonl"),),
                    ),
                    processor=TextProcessor.Config(),
                ),
            ),
            seed=42,
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8192,
            num_tokens_per_train_step=65536,
            max_context_length=8192,
            steps=100,
            disable_cuda_graphs=False,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=8,
            expert_parallel_degree=8,
            fsdp_reshard_after_forward="never",
        ),
        compile=CompileConfig(enable=True, components=["model"]),
        optimizer=default_adamw(lr=1e-5),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=10),
        checkpoint=CheckpointManager.Config(
            enable=True,
            initial_load_path=str(assets),
            initial_load_in_hf=True,
            initial_load_model_only=True,
            interval=100,
            last_save_model_only=True,
            export_dtype="bfloat16",
        ),
    )
