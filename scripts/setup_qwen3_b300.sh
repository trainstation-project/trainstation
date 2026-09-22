#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install dependencies only. This script does not launch training.
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
prep_dir="$repo_dir/outputs/qwen3_30b_mxfp8_prep"
env_dir="$prep_dir/venv"
mkdir -p "$prep_dir"
if [[ ! -x "$env_dir/bin/python" ]]; then
    uv venv --python 3.12 "$env_dir"
fi

uv pip install --python "$env_dir/bin/python" --pre \
    'torch==2.15.0.dev20260921+cu130' 'torchao==0.19.0.dev20260922+cu130' \
    --index-url https://download.pytorch.org/whl/nightly/cu130
uv pip install --python "$env_dir/bin/python" \
    -r requirements.txt -r requirements-dev.txt \
    'torch_checkpointing @ git+https://github.com/meta-pytorch/torch_checkpointing.git@1391408e294286acfa30358ba0c665ba1cad2c76' \
    'nvidia-cutlass-dsl==4.5.2' apache-tvm-ffi \
    'nvidia-cuda-nvcc==13.0.88' 'nvidia-nvvm==13.0.88' 'nvidia-cuda-crt==13.0.88' \
    'nvidia-cuda-cccl==13.0.85' 'nvidia-cuda-profiler-api==13.0.85' \
    pynvml cmake ninja wheel matplotlib

export CUDA_HOME="$env_dir/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$env_dir/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include/cccl:${CPLUS_INCLUDE_PATH:-}"
export NVCC_APPEND_FLAGS="-I$CUDA_HOME/include/cccl"
export TORCH_CUDA_ARCH_LIST=10.3a
export HYBRID_EP_MULTINODE=0
export MAX_JOBS=16

# NVIDIA's runtime wheels contain versioned libraries without linker aliases.
python - <<'PY'
import os
from pathlib import Path

cuda = Path(os.environ["CUDA_HOME"])
lib = cuda / "lib"
if not (cuda / "lib64").exists():
    (cuda / "lib64").symlink_to("lib")
for name, target in (
    ("libcudart.so", "libcudart.so.13"),
    ("libnvtx3interop.so", "libnvtx3interop.so.1"),
):
    alias = lib / name
    if not alias.exists():
        alias.symlink_to(target)
PY

if [[ ! -d "$prep_dir/DeepEP/.git" ]]; then
    git clone --branch hybrid-ep --single-branch \
        https://github.com/deepseek-ai/DeepEP.git "$prep_dir/DeepEP"
fi
git -C "$prep_dir/DeepEP" checkout f725d29699f5bda9ba789456bb9579af69844685
uv pip install --python "$env_dir/bin/python" --no-build-isolation --no-deps \
    "$prep_dir/DeepEP"
python -c 'import torch; from deep_ep import HybridEPBuffer; print(torch.__version__, HybridEPBuffer.__name__)'
