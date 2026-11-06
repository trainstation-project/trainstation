# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torchtitan_recipes.tests.models as recipes

from torchtitan.models.deepseek_v3.config_registry import deepseek_v3_debugmodel
from torchtitan.models.llama3.config_registry import llama3_debugmodel

from tests.integration_tests import OverrideDefinitions


def build_model_tests_list() -> list[OverrideDefinitions]:
    """
    Build the list of model parallelism test configurations.
    This test suite is aimed at testing the model parallelism of torchtitan, and will
    broadly cover all the supported model parallelism patterns on all the supported
    models.
    """
    return [
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_fsdp2_tp2_cp2],
            test_descr="Llama 3 FSDP+TP+CP",
            test_name="llama3_fsdp+tp+cp",
            ngpu=8,
            golden_numerics_path=(
                "tests/assets/losses/{execution_mode}/{gpu_arch}/llama3.txt"
            ),
            loss_compare_seed_config=llama3_debugmodel,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_region_ac_fsdp2_tp2_cp2],
            test_descr="Llama 3 FSDP+TP+CP+RegionAC",
            test_name="llama3_fsdp+tp+cp+region_ac",
            ngpu=8,
            golden_numerics_path=(
                "tests/assets/losses/{execution_mode}/{gpu_arch}/llama3.txt"
            ),
            loss_compare_seed_config=llama3_debugmodel,
        ),
        OverrideDefinitions(
            configs=[recipes.llama3_debugmodel_fsdp2_tp2_pp2],
            test_descr="Llama 3 FSDP+TP+PP",
            test_name="llama3_fsdp+tp+pp",
            ngpu=8,
            golden_numerics_path="tests/assets/losses/real_pg/{gpu_arch}/llama3_pp.txt",
            use_real_pg=True,
        ),
        # Integration Test Cases for DeepSeek V3
        OverrideDefinitions(
            configs=[recipes.deepseek_v3_debugmodel_mtp_fsdp4_ep2_compile],
            test_descr="DeepSeek V3 MTP FSDP+EP+compile",
            test_name="deepseek_v3_mtp_fsdp+ep+compile",
            ngpu=4,
            # The Helion fused RoPE kernels are CUDA-only and tuned for NVIDIA
            # H100/GB200; skip on ROCm where they are unvalidated.
            skip_rocm_test=True,
        ),
        OverrideDefinitions(
            configs=[recipes.deepseek_v3_debugmodel_mtp_tp2_cp2],
            test_descr="DeepSeek V3 MTP TP+CP with SP",
            test_name="deepseek_v3_mtp_tp+cp",
            ngpu=4,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.deepseek_v3_debugmodel_fsdp8_ep8],
            test_descr="DeepSeek V3 FSDP+EP",
            test_name="deepseek_v3_fsdp+ep",
            ngpu=8,
            golden_numerics_path=(
                "tests/assets/losses/{execution_mode}/{gpu_arch}/deepseek_v3.txt"
            ),
        ),
        OverrideDefinitions(
            configs=[recipes.deepseek_v3_debugmodel_fsdp2_cp2_pp2_ep4],
            test_descr="DeepSeek V3 FSDP+CP+PP+EP",
            test_name="deepseek_v3_fsdp+cp+pp+ep",
            ngpu=8,
            golden_numerics_path=(
                "tests/assets/losses/real_pg/{gpu_arch}/deepseek_v3_cp_pp.txt"
            ),
            loss_compare_seed_config=deepseek_v3_debugmodel,
            use_real_pg=True,
        ),
        OverrideDefinitions(
            configs=[recipes.deepseek_v3_debugmodel_hsdp2x2_ep2],
            test_descr="DeepSeek V3 HSDP+EP",
            test_name="deepseek_v3_hsdp+ep",
            ngpu=4,
        ),
    ]
