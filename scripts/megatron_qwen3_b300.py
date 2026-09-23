# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prepare or explicitly launch the matched Qwen3 Megatron experiment.

The default action validates preparation only. Training requires --train.
Training runs eagerly with full, per-layer activation recomputation.
"""

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformer_engine.pytorch as transformer_engine

from megatron.bridge import AutoBridge
from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider
from megatron.bridge.models.conversion.param_mapping import AutoMapping
from megatron.bridge.models.gpt_provider import default_layer_spec
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.config import (
    OptimizerConfigOverrideProvider,
    OptimizerConfigOverrideProviderContext,
)
from megatron.bridge.training.mixed_precision import bf16_mixed
from megatron.core.enums import Fp8Recipe
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import TopKRouter
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/qwen3_30b_c4_megatron"
MODEL = ROOT / "assets/hf/Qwen3-30B-A3B"
OUTPUT = ROOT / "outputs/megatron_qwen3_30b_eager_full_recompute_100"


def quantizer_factory(role):
    """Keep expert WGRAD operands in BF16, with MXFP8 forward and DGRAD."""
    if role is None or role.tensor_type not in ("input", "weight", "grad_output"):
        return transformer_engine.IdentityQuantizer()
    is_expert = role.module_type == "grouped_linear"
    quantizer = transformer_engine.MXFP8Quantizer(
        fp8_dtype=transformer_engine.DType.kFloat8E4M3,
        with_2d_quantization=role.tensor_type == "weight" and not is_expert,
    )
    if is_expert and role.tensor_type in ("input", "grad_output"):
        return transformer_engine.HybridQuantizer(
            rowwise_quantizer=quantizer,
            columnwise_quantizer=transformer_engine.IdentityQuantizer(),
            columnwise_source="original",
        )
    return quantizer


def biased_softmax_routing(logits_TE, bias_E, *, top_k):
    """T: tokens; E: experts; K: experts selected per token."""
    scores_TE = logits_TE.float().softmax(dim=-1)
    selected_TK = (scores_TE + bias_E).topk(top_k, sorted=False).indices
    weights_TK = scores_TE.gather(-1, selected_TK)
    weights_TK = weights_TK / (weights_TK.sum(-1, keepdim=True) + 1e-20)
    probs_TE = torch.zeros_like(scores_TE).scatter(-1, selected_TK, weights_TK)
    routing_TE = torch.zeros_like(scores_TE, dtype=torch.bool).scatter(
        -1, selected_TK, True
    )
    return probs_TE, routing_TE


class MatchedRouter(TopKRouter, torch.nn.Module):
    """Preserve TorchTitan's softmax-score bias and normalized top-k weights."""

    config: Any
    weight: torch.Tensor

    def __init__(self, config, pg_collection=None, is_mtp_layer=False):
        super().__init__(config, pg_collection, is_mtp_layer)
        self._keep_in_float32_parameter_names = ("weight",)
        # MCore's built-in bias requires sigmoid scores. This router owns the
        # softmax bias and its centered update independently of that option.
        self.enable_expert_bias = True
        del self.expert_bias, self.local_tokens_per_expert
        self.register_buffer(
            "expert_bias",
            torch.zeros(
                config.num_moe_experts, dtype=torch.float32, device=self.weight.device
            ),
            persistent=False,
        )
        self.register_buffer(
            "local_tokens_per_expert",
            torch.zeros(
                config.num_moe_experts, dtype=torch.int64, device=self.weight.device
            ),
            persistent=False,
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.weight.data = self.weight.data.float()

    def gating(self, input_TBH):
        # T: tokens; B: batch; H: hidden size. The stock gate passes BF16
        # activations with FP32 weights to cuBLASLt, which rejects mixed operands.
        return torch.nn.functional.linear(input_TBH.float(), self.weight)

    def routing(self, logits_TBE, padding_mask=None):
        # T: tokens; B: batch; E: experts. The exported batches contain no padding.
        logits_TE = logits_TBE.reshape(-1, self.config.num_moe_experts)
        probs_TE, routing_TE = biased_softmax_routing(
            logits_TE, self.expert_bias, top_k=self.topk
        )
        if padding_mask is not None:
            valid_T1 = ~padding_mask.reshape(-1, 1)
            probs_TE = probs_TE * valid_T1
            routing_TE = routing_TE & valid_T1
        super()._apply_expert_bias(routing_TE)
        return probs_TE, routing_TE


def layer_spec(config):
    spec = default_layer_spec(config)
    spec.submodules.mlp.keywords["submodules"].router = MatchedRouter
    return spec


class PackedBatches(Dataset):
    """Expose the exact exported per-rank stream to Megatron's single sampler."""

    def __init__(self, path: Path):
        self.manifest = json.loads((path / "manifest.json").read_text())
        self.ranks = [
            {
                name: np.load(path / f"rank{rank}" / f"{name}.npy", mmap_mode="r")
                for name in ("input", "labels", "positions")
            }
            for rank in range(8)
        ]

    def __len__(self):
        return self.manifest["steps"] * 8

    def __getitem__(self, index):
        step, rank = divmod(index, 8)
        row = self.ranks[rank]
        positions = np.array(row["positions"][step], dtype=np.int64)
        boundaries = np.flatnonzero(positions == 0)
        # Fixed-size metadata lets the collator stack document boundaries.
        cu_seqlens = np.full(self.manifest["max_documents"] + 1, 8192, np.int32)
        cu_seqlens[: len(boundaries)] = boundaries
        return {
            "tokens": torch.tensor(row["input"][step], dtype=torch.int64),
            "labels": torch.tensor(row["labels"][step], dtype=torch.int64),
            "position_ids": torch.from_numpy(positions),
            "loss_mask": torch.ones(8192, dtype=torch.float32),
            "cu_seqlens_q": torch.from_numpy(cu_seqlens),
            "cu_seqlens_kv": torch.from_numpy(cu_seqlens.copy()),
            "max_seqlen_q": self.manifest["max_document_length"],
            "max_seqlen_kv": self.manifest["max_document_length"],
        }


@dataclass
class MatchedData(DatasetProvider):
    seq_length: int = 8192
    skip_getting_attention_mask_from_dataset: bool = True

    def build_datasets(self, context: DatasetBuildContext):
        dataset = PackedBatches(DATA)
        if context.train_samples > len(dataset):
            raise ValueError("Requested training exceeds the exported token stream")
        return dataset, None, None


@dataclass
class AllParametersWeightDecay(OptimizerConfigOverrideProvider):
    def build_config_overrides(self, context: OptimizerConfigOverrideProviderContext):
        del context
        # TorchTitan's catch-all AdamW group includes norms and biases.
        return {}


def learning_rate(step: int) -> float:
    """Return the LR used for a one-based optimizer step in the 1000-step run."""
    if step <= 10:
        return 1e-5 * step / 10
    if step == 11:
        return 1e-5
    return 1e-5 * (1 - (step - 11) / 990)


class MatchedUpdates(Callback):
    @staticmethod
    def routers(context):
        return [
            module
            for chunk in context.model
            for module in chunk.modules()
            if isinstance(module, MatchedRouter)
        ]

    def on_train_start(self, context: CallbackContext):
        # The auxiliary bias starts at zero and is absent from the pretrained HF
        # checkpoint. Persist it in our checkpoints after strict HF import finishes.
        for router in self.routers(context):
            router.register_buffer("expert_bias", router.expert_bias, persistent=True)
        for chunk in context.model:
            for module in chunk.modules():
                if isinstance(module, MoELayer):
                    manager = module.token_dispatcher._comm_manager
                    # The dispatcher uses the recipe only to select alignment.
                    # Custom recipes default to 16 rows, but MXFP8 requires 32.
                    # Copy this config so expert kernels retain the custom recipe.
                    manager.config = replace(manager.config, fp8_recipe=Fp8Recipe.mxfp8)

    def on_train_step_start(self, context: CallbackContext):
        lr = learning_rate(context.state.train_state.step + 1)
        for group in context.optimizer.param_groups:
            group["lr"] = lr
        for router in self.routers(context):
            router.local_tokens_per_expert.zero_()

    def on_train_step_end(self, context: CallbackContext):
        # With TP=PP=CP=1, the global group is the eight-rank data-parallel group.
        with torch.no_grad():
            routers = self.routers(context)
            counts_LE = torch.stack(
                [router.local_tokens_per_expert for router in routers]
            )
            torch.distributed.all_reduce(counts_LE)
            counts_LE = counts_LE.float()
            delta_LE = 1e-3 * (counts_LE.mean(-1, keepdim=True) - counts_LE).sign()
            delta_LE = delta_LE - delta_LE.mean(-1, keepdim=True)
            for router, delta_E in zip(routers, delta_LE):
                router.expert_bias.add_(delta_E)


def build_config():
    AutoMapping.register_module_type("MatchedRouter", "replicated")
    cfg = _pretrain_common()
    cfg.model = AutoBridge.from_hf_pretrained(str(MODEL)).to_megatron_provider(
        load_weights=True
    )
    model = cfg.model
    model.transformer_layer_spec = layer_spec
    model.tensor_model_parallel_size = 1
    model.pipeline_model_parallel_size = 1
    model.context_parallel_size = 1
    model.expert_tensor_parallel_size = 1
    model.expert_model_parallel_size = 8
    model.sequence_parallel = False
    model.seq_length = 8192
    model.moe_grouped_gemm = True
    model.moe_use_grouped_tensor = False
    model.moe_single_grouped_weight = False
    model.use_transformer_engine_op_fuser = False
    model.moe_token_dispatcher_type = "flex"
    model.moe_flex_dispatcher_backend = "hybridep"
    model.moe_router_padding_for_quantization = True
    model.moe_router_force_load_balancing = False
    model.moe_expert_capacity_factor = None
    model.moe_router_dtype = "fp32"
    model.moe_router_enable_expert_bias = False
    model.moe_router_bias_update_rate = 1e-3
    model.moe_router_load_balancing_type = "none"
    model.moe_aux_loss_coeff = 0.0
    model.moe_router_fusion = False
    model.moe_router_score_function = "softmax"
    model.moe_router_pre_softmax = True
    model.hidden_dropout = 0.0
    model.attention_dropout = 0.0
    model.cuda_graph_impl = "none"
    model.cuda_graph_modules = []
    model.use_te_rng_tracker = True
    # Uniform chunks of one checkpoint every transformer layer, matching FullAC.
    model.recompute_granularity = "full"
    model.recompute_modules = []
    model.recompute_method = "uniform"
    model.recompute_num_layers = 1
    model.overlap_moe_expert_parallel_comm = False
    model.gradient_accumulation_fusion = True
    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.fp8 = "e4m3"
    cfg.mixed_precision.fp8_recipe = "custom"
    cfg.mixed_precision.fp8_quantizer_factory = "megatron_qwen3_b300.quantizer_factory"
    cfg.mixed_precision.fp8_param_gather = False
    cfg.mixed_precision.fp8_dot_product_attention = False
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.dist.use_megatron_fsdp = True
    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.megatron_fsdp_version = 1
    # Keep compute weights through forward/backward, as in the baseline's
    # reshard_after_forward="never". FP32 master weights and gradients stay sharded.
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads"
    # Expert DP has size one, so these buffers remain local. Using the same
    # strategy also supplies the buffers required by FSDP's prefetch bookkeeping.
    cfg.ddp.expert_data_parallel_sharding_strategy = "optim_grads"
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.megatron_fsdp_main_grads_dtype = torch.float32
    cfg.ddp.megatron_fsdp_grad_comm_dtype = torch.float32
    cfg.ddp.average_in_collective = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.optimizer.lr = 1e-5
    cfg.optimizer.min_lr = 0.0
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.clip_grad = 1.0
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer_config_override_provider = AllParametersWeightDecay()
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 1000
    cfg.scheduler.lr_decay_style = "linear"
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.weight_decay_incr_style = "constant"
    # Measure the first 100 steps of the reference schedule and token stream.
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_iters = 0
    cfg.validation.eval_interval = 100
    cfg.dataset = MatchedData()
    cfg.dataset.dataloader_type = "single"
    cfg.dataset.num_workers = 0
    cfg.tokenizer.tokenizer_model = str(MODEL)
    cfg.tokenizer.use_tokenizer_vocab_size = False
    # Import before FSDP wraps parameters; the pinned post-wrap HF loader expects
    # legacy DTensor slice metadata that the current FSDP adapter no longer sets.
    cfg.checkpoint.pretrained_checkpoint = None
    cfg.checkpoint.hf_source_path = str(MODEL)
    cfg.checkpoint.load = None
    cfg.checkpoint.save = str(OUTPUT / "checkpoints")
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.ckpt_format = "fsdp_dtensor"
    cfg.checkpoint.save_optim = False
    cfg.checkpoint.save_rng = False
    cfg.logger.tensorboard_dir = str(OUTPUT / "tb")
    cfg.logger.log_interval = 1
    cfg.logger.tensorboard_log_interval = 1
    cfg.logger.wandb_project = None
    cfg.rng.seed = 42
    cfg.rng.te_rng_tracker = True
    cfg.comm_overlap = None
    return cfg


def check_preparation():
    """Check artifacts and metadata on CPU, without constructing a model."""
    from safetensors import safe_open
    from transformer_engine.pytorch.quantization import QuantizerRole

    dataset = PackedBatches(DATA)
    manifest = dataset.manifest
    if (manifest["steps"], manifest["total_tokens"]) != (1000, 65536000):
        raise ValueError("Expected the full 1000-step token stream")
    for rank in manifest["ranks"]:
        for name, expected in rank["sha256"].items():
            path = DATA / f"rank{rank['rank']}" / f"{name}.npy"
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Export checksum mismatch: {path}")
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())
    tensors = {}
    for filename in sorted(set(index["weight_map"].values())):
        with safe_open(MODEL / filename, framework="pt", device="cpu") as shard:
            for key in shard.keys():
                tensors[key] = shard.get_slice(key).get_shape()
    if set(tensors) != set(index["weight_map"]):
        raise ValueError("Checkpoint shard keys differ from the index")
    num_parameters = sum(int(np.prod(shape)) for shape in tensors.values())
    if num_parameters != 30532122624:
        raise ValueError(f"Unexpected checkpoint size: {num_parameters}")
    expert_input = quantizer_factory(
        QuantizerRole(module_type="grouped_linear", tensor_type="input")
    )
    expert_grad = quantizer_factory(
        QuantizerRole(module_type="grouped_linear", tensor_type="grad_output")
    )
    for quantizer in (expert_input, expert_grad):
        assert isinstance(
            quantizer.rowwise_quantizer, transformer_engine.MXFP8Quantizer
        )
        assert isinstance(
            quantizer.columnwise_quantizer, transformer_engine.IdentityQuantizer
        )
    cfg = build_config()
    # Full Bridge validation probes CUDA device properties for HybridEP.
    # Finalize the model and precision contracts here without GPU execution.
    cfg.mixed_precision.finalize()
    cfg.mixed_precision.setup(cfg.model, cfg.optimizer, cfg.ddp)
    cfg.model.finalize()
    assert (
        layer_spec(cfg.model).submodules.mlp.keywords["submodules"].router
        is MatchedRouter
    )
    cfg.ddp.__post_init__()
    cfg.optimizer.__post_init__()
    cfg.dataset.finalize()
    assert not torch.cuda.is_initialized()
    report = {
        "status": "prepared_without_gpu_execution",
        "training_started": False,
        "num_parameters": num_parameters,
        "total_tokens": manifest["total_tokens"],
        "parallelism": {"fsdp": 8, "ep": 8, "tp": 1, "pp": 1, "cp": 1},
        "expert_wgrad": "BF16 original operands",
        "cuda_graph_modules": [],
        "execution": "eager; torch.compile disabled by force_eager stance",
        "activation_checkpointing": "full, uniform, one transformer layer per checkpoint",
        "parameter_sharding": "optim_grads: sharded FP32 master/gradients, retained compute weights",
        "runtime_validation": "deferred: hardware validation, model loading, kernels, numerics, memory",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("%s", json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    # Megatron decorates some helpers with torch.compile even without model graphs.
    torch.compiler.set_stance("force_eager")
    if not args.train:
        check_preparation()
        return
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        parser.error("Training requires an explicit eight-rank launch")
    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain

    torch.backends.cuda.matmul.allow_tf32 = False
    pretrain(build_config(), forward_step, callbacks=[MatchedUpdates()])


if __name__ == "__main__":
    main()
