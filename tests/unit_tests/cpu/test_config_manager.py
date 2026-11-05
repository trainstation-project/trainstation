# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import unittest
from unittest import mock

import pytest
from torchtitan.config import ConfigManager, ParallelismConfig, TrainingConfig


class TestConfigManager(unittest.TestCase):
    def test_parse_args_uses_current_sys_argv(self):
        """parse_args() without args reads sys.argv at call time."""
        config_manager = ConfigManager()
        argv = ["train.py", "--module", "nonexistent", "--config", "foo"]
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(ImportError, match="Cannot import module 'nonexistent'"):
                config_manager.parse_args()

    def test_model_without_config_errors(self):
        """--module alone raises ValueError."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="--config is required"):
            config_manager.parse_args(["--module", "llama3"])

    def test_config_without_model_errors(self):
        """--config alone raises ValueError."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="--module is required"):
            config_manager.parse_args(["--config", "llama3_debugmodel"])

    def test_missing_both_errors(self):
        """No --module or --config raises ValueError."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="--module is required"):
            config_manager.parse_args([])

    def test_invalid_model_errors(self):
        """--module with unknown module name raises ImportError."""
        config_manager = ConfigManager()
        with pytest.raises(ImportError, match="Cannot import module"):
            config_manager.parse_args(["--module", "nonexistent", "--config", "foo"])

    def test_num_tokens_per_train_step_must_be_positive_or_unset(self):
        with pytest.raises(ValueError, match="must be -1 or greater than 0"):
            TrainingConfig(num_tokens_per_train_step=0)

    def test_max_context_length_must_be_positive(self):
        for max_context_length in (0, -1):
            with pytest.raises(ValueError, match="must be greater than 0"):
                TrainingConfig(max_context_length=max_context_length)

    # TODO: remove this test when we remove the merge functionality

    def test_default_context_parallel_load_balancer(self):
        assert ParallelismConfig().context_parallel_load_balancer == "headtail"

    def test_suppressed_model_spec_is_opaque_to_tyro(self):
        config = ConfigManager().parse_args(
            [
                "--module",
                "torchtitan_recipes.tests.transformers_modeling_backend",
                "--config",
                "transformers_backend_dense_cp_pp",
            ]
        )

        assert config.model_spec.name == "transformers_modeling_backend"

    def test_fqn_module_invalid_errors(self):
        """--module with invalid FQN raises ImportError."""
        config_manager = ConfigManager()
        with pytest.raises(ImportError, match="Cannot import module"):
            config_manager.parse_args(
                [
                    "--module",
                    "torchtitan.models.nonexistent",
                    "--config",
                    "foo",
                ]
            )


if __name__ == "__main__":
    unittest.main()
