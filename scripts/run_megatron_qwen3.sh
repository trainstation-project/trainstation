#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the BSD-style license in the repository LICENSE file.

set -euo pipefail

TASK_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TASK_OUTPUT="$TASK_ROOT/outputs/megatron_comparison"
TASK_IMAGE=trainstation-megatron:qwen3-b300
TASK_BASE=nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf
BRIDGE_REV=7630ad82864f9e64ce957b323a2779856def2f62
TE_REV=5e52befd5262c06289106338c308079d6adb391f

# Explicit driver mounts also work on hosts without nvidia-container-runtime.
# GPU devices are added only by the train action.
container_args=(
    --entrypoint bash
    -e NVIDIA_VISIBLE_DEVICES=void
    -v "$TASK_ROOT:/workspace/trainstation"
    -v /lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro
    -v /lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so:ro
    -v /lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro
    -v /lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:ro
    -w /workspace/trainstation
    -e PYTHONPATH=/workspace/trainstation/scripts
    -e TOKENIZERS_PARALLELISM=false
    -e CUDA_DEVICE_MAX_CONNECTIONS=32
    -e OMP_NUM_THREADS=4
    -e TORCHINDUCTOR_CACHE_DIR=/workspace/trainstation/outputs/megatron_qwen3_30b_100/inductor-cache
    -e TRITON_CACHE_DIR=/workspace/trainstation/outputs/megatron_qwen3_30b_100/triton-cache
    -e TORCH_EXTENSIONS_DIR=/workspace/trainstation/outputs/megatron_qwen3_30b_100/extension-cache
    -e TASK_UID="$(id -u)" -e TASK_GID="$(id -g)"
)

checkout() {
    local url=$1 destination=$2 revision=$3
    if [[ ! -d "$destination/.git" ]]; then
        git clone "$url" "$destination"
    fi
    if [[ -n $(git -C "$destination" status --porcelain) ]]; then
        echo "Source checkout has local changes: $destination" >&2
        exit 1
    fi
    git -C "$destination" checkout "$revision"
}

case "${1:-check}" in
    prepare)
        mkdir -p "$TASK_OUTPUT"
        checkout https://github.com/NVIDIA-NeMo/Megatron-Bridge.git \
            "$TASK_OUTPUT/Megatron-Bridge" "$BRIDGE_REV"
        git -C "$TASK_OUTPUT/Megatron-Bridge" submodule update --init 3rdparty/Megatron-LM
        checkout https://github.com/NVIDIA/TransformerEngine.git \
            "$TASK_OUTPUT/TransformerEngine" "$TE_REV"
        git -C "$TASK_OUTPUT/TransformerEngine" submodule update --init --recursive
        docker pull "$TASK_BASE"
        docker run -d --name trainstation-megatron-build \
            "${container_args[@]}" -e CUDA_VISIBLE_DEVICES='' "$TASK_BASE" \
            -lc 'exec sleep infinity'
        docker exec -e NVTE_FRAMEWORK=pytorch -e NVTE_CUDA_ARCHS=100 \
            -e NVTE_WITH_NCCL_EP=0 -e NVTE_BUILD_MAX_JOBS=16 -e MAX_JOBS=16 \
            -e UV_CACHE_DIR=/workspace/trainstation/outputs/megatron_comparison/uv-cache \
            trainstation-megatron-build bash -lc '
                set -euo pipefail
                uv pip install --python /opt/venv/bin/python --no-build-isolation --no-deps \
                    /workspace/trainstation/outputs/megatron_comparison/TransformerEngine
                uv pip install --python /opt/venv/bin/python --no-build-isolation --no-deps \
                    -e /workspace/trainstation/outputs/megatron_comparison/Megatron-Bridge/3rdparty/Megatron-LM \
                    -e /workspace/trainstation/outputs/megatron_comparison/Megatron-Bridge
            '
        docker commit trainstation-megatron-build "$TASK_IMAGE"
        docker exec trainstation-megatron-build bash -lc \
            'chown -R "$TASK_UID:$TASK_GID" outputs/megatron_comparison'
        docker rm -f trainstation-megatron-build
        ;;
    check)
        docker run --rm "${container_args[@]}" -e CUDA_VISIBLE_DEVICES='' \
            -e HF_HUB_OFFLINE=1 "$TASK_IMAGE" -lc \
            'trap '\''chown -R "$TASK_UID:$TASK_GID" outputs/megatron_qwen3_30b_100'\'' EXIT
             uv run --no-project python scripts/megatron_qwen3_b300.py'
        ;;
    train)
        for device in /dev/nvidia{0..7} /dev/nvidiactl /dev/nvidia-uvm; do
            container_args+=(--device "$device")
        done
        docker run --rm --name trainstation-megatron-qwen3 \
            --ipc=host --network=host --ulimit memlock=-1 --ulimit stack=67108864 \
            "${container_args[@]}" -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
            -e HF_HUB_OFFLINE=1 "$TASK_IMAGE" -lc \
            'trap '\''chown -R "$TASK_UID:$TASK_GID" outputs/megatron_qwen3_30b_100'\'' EXIT
             ldconfig
             uv run --no-project python -m torch.distributed.run --standalone \
                --nproc_per_node=8 scripts/megatron_qwen3_b300.py --train'
        ;;
    *)
        echo "Usage: $0 [prepare|check|train]" >&2
        exit 2
        ;;
esac
