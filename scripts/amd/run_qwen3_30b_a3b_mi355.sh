#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Basic Qwen3-30B-A3B (MoE, ~3B active) pretraining example on a single
# 8x AMD Instinct MI355X node.
#
# Prerequisites:
#   - torch nightly for ROCm (same index as the ROCm CI), e.g.
#       pip install --pre torch torchao --index-url https://download.pytorch.org/whl/nightly/rocm10.0
#       pip install -r requirements.txt && pip install -e . --no-deps
#   - tokenizer:
#       python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-30B-A3B --assets tokenizer config
#
# Usage (extra args are forwarded to torchtitan.train and override defaults):
#   ./scripts/amd/run_qwen3_30b_a3b_mi355.sh --training.steps 300
#
# Notes:
#   - CUDA graphs are disabled: the default all-to-all EP token dispatcher needs
#     CPU sync, and on 8 GPUs plain FSDP (EP=1) was fastest in our sweep
#     (EP=1: ~8.3k tok/s/GPU, EP=8: ~7.9k, EP=2: ~6.4k, EP=4: ~5.7k at
#     seq 4096, 16k tokens/microbatch, compile on, no activation checkpointing).
#   - Global batch size is the main MFU lever: 16k tok/GPU, no AC -> ~9% MFU;
#     128k tok/GPU, selective AC (default) -> ~22% MFU, 221 GiB peak memory.

set -ex

cd "$(dirname "$0")/../.."

NGPU=${NGPU:-8}
EP=${EP:-1}
SEQ_LEN=${SEQ_LEN:-4096}
TOKENS_PER_MICROBATCH=${TOKENS_PER_MICROBATCH:-131072}
STEPS=${STEPS:-300}
AC=${AC:-selective}  # none | selective | full

# ROCm / RCCL settings carried over from AMD-AGI/torchtitan-amd launch scripts.
export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-2}
export PYTORCH_TUNABLEOP_ENABLED=${PYTORCH_TUNABLEOP_ENABLED:-0}
# Multinode only: enable PXN to speed up cross-node all-to-all.
#   export NCCL_PXN_DISABLE=0 NCCL_P2P_NET_CHUNKSIZE=262144

if [[ "${AC}" == "none" ]]; then
    AC_ARG="activation-checkpoint:none"
else
    AC_ARG="activation-checkpoint:${AC}"
fi

NGPU=${NGPU} MODULE=qwen3 CONFIG=qwen3_30b_a3b ./run_train.sh \
    --training.steps "${STEPS}" \
    --training.max-context-length "${SEQ_LEN}" \
    --training.num-tokens-per-microbatch-per-dp-rank "${TOKENS_PER_MICROBATCH}" \
    --parallelism.expert-parallel-degree "${EP}" \
    --training.disable-cuda-graphs \
    --compile.enable \
    --metrics.log-freq 1 \
    --metrics.enable-tensorboard \
    "$@" \
    "${AC_ARG}"
