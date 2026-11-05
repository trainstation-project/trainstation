#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

required_env_vars=(
  VALIDATION_SUITE
  TORCH_VERSION
  TORCHVISION_VERSION
  TORCHAO_VERSION
  PYTORCH_INDEX_URL
)
for var in "${required_env_vars[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "Missing required environment variable: ${var}"
    exit 1
  fi
done

case "${VALIDATION_SUITE}" in
  core|graph-trainer|transformers-modeling-backend) ;;
  *)
    echo "Unknown validation suite: ${VALIDATION_SUITE}" >&2
    exit 1
    ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
RC_VERSION="$(tr -d '[:space:]' < "${REPO_ROOT}/assets/version.txt")"
if [[ "${RC_VERSION}" != *rc* ]]; then
  echo "Expected an RC version, got: ${RC_VERSION}"
  exit 1
fi
export RC_VERSION VALIDATION_SUITE TORCH_VERSION TORCHVISION_VERSION TORCHAO_VERSION

eval "$(conda shell.bash hook)"
conda activate "$(conda env list --json | jq -r '.envs[-1]')"

export HF_HOME="${RUNNER_TEMP}/hf_home"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONUNBUFFERED=1

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 || true)
echo "CUDA driver version: ${DRIVER_VERSION}"
nvidia-smi

python -m pip uninstall -y torch torchvision torchao torchtitan

USE_CPP=0 PIP_EXTRA_INDEX_URL='' python -m pip install --pre \
  --index-url "${PYTORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchao==${TORCHAO_VERSION}"

install_release_candidate() {
  for attempt in {1..5}; do
    echo "Installing torchtitan==${RC_VERSION} from TestPyPI (attempt ${attempt}/5)"
    if python -m pip install \
      --index-url https://test.pypi.org/simple/ \
      --extra-index-url https://pypi.org/simple/ \
      "torchtitan==${RC_VERSION}"; then
      return 0
    fi
    if [[ "${attempt}" -lt 5 ]]; then
      sleep 15
    fi
  done

  echo "Failed to install torchtitan==${RC_VERSION} after 5 attempts" >&2
  return 1
}
install_release_candidate

ARTIFACTS="${RUNNER_TEMP}/artifacts-to-be-uploaded"
VALIDATION_ROOT="${RUNNER_TEMP}/validate-release-gpu"
mkdir -p \
  "${ARTIFACTS}" \
  "${VALIDATION_ROOT}/scripts" \
  "${VALIDATION_ROOT}/torchtitan/experiments/graph_trainer" \
  "${VALIDATION_ROOT}/torchtitan/models/flux/inference"

# Copy test support files without copying TorchTitan Python packages. Training
# subprocesses must import the installed TestPyPI RC rather than the checkout.
cp "${REPO_ROOT}/run_train.sh" "${VALIDATION_ROOT}/run_train.sh"
cp "${REPO_ROOT}/scripts/loss_compare.py" "${VALIDATION_ROOT}/scripts/loss_compare.py"
cp -R "${REPO_ROOT}/tests" "${VALIDATION_ROOT}/tests"
cp \
  "${REPO_ROOT}/torchtitan/models/flux/run_infer.sh" \
  "${VALIDATION_ROOT}/torchtitan/models/flux/run_infer.sh"
cp \
  "${REPO_ROOT}/torchtitan/models/flux/inference/prompts.txt" \
  "${VALIDATION_ROOT}/torchtitan/models/flux/inference/prompts.txt"
cp \
  "${REPO_ROOT}/torchtitan/experiments/graph_trainer/run_train_precompile.sh" \
  "${VALIDATION_ROOT}/torchtitan/experiments/graph_trainer/run_train_precompile.sh"

cd "${VALIDATION_ROOT}"
unset PYTHONPATH

python - <<'PY'
import os

import torch
import torchao
import torchtitan
import torchvision
import triton

assert torchtitan.__version__ == os.environ["RC_VERSION"]
assert "site-packages" in torchtitan.__file__
assert torch.__version__.split("+")[0] == os.environ["TORCH_VERSION"]
assert torchvision.__version__.split("+")[0] == os.environ["TORCHVISION_VERSION"]
assert torchao.__version__.split("+")[0] == os.environ["TORCHAO_VERSION"]
assert torch.version.cuda == "13.0"
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 8
print(f"torchtitan={torchtitan.__version__} ({torchtitan.__file__})")
print(f"torch={torch.__version__} ({torch.version.git_version})")
print(f"torchvision={torchvision.__version__}")
print(f"torchao={torchao.__version__}")
print(f"triton={triton.__version__}")
print(f"num_cuda_devices={torch.cuda.device_count()}")
PY

TORCHTITAN_PACKAGE_ROOT="$(python - <<'PY'
from pathlib import Path

import torchtitan

print(Path(torchtitan.__file__).parent)
PY
)"

run_core_tests() {
  CUDA_VISIBLE_DEVICES=0 python -m pytest tests/unit_tests/gpu \
    -m "not multi_gpu" \
    --strict-markers \
    --durations=20 \
    -vv

  python -m pytest tests/unit_tests/gpu \
    -m multi_gpu \
    --strict-markers \
    --durations=20 \
    -vv
}

run_transformers_modeling_backend_tests() {
  python -m pip install transformers==5.9.0
  python -m torchtitan.experiments.transformers_modeling_backend.tests.integration_tests \
    "${ARTIFACTS}/transformers_modeling_backend" \
    --ngpu 8
}

run_graph_trainer_tests() {
  local tests_root="${TORCHTITAN_PACKAGE_ROOT}/experiments/graph_trainer/tests"

  python -m pytest "${tests_root}/test_passes.py" -v
  python -m pytest "${tests_root}/test_profiler.py" -v
  python -m pytest "${tests_root}/test_trace_module.py" -v
  python -m pytest "${tests_root}/test_precompile.py" -v
}

case "${VALIDATION_SUITE}" in
  core) run_core_tests ;;
  graph-trainer) run_graph_trainer_tests ;;
  transformers-modeling-backend) run_transformers_modeling_backend_tests ;;
esac

find "${ARTIFACTS}" -type d -name checkpoint -prune -exec rm -rf {} +
