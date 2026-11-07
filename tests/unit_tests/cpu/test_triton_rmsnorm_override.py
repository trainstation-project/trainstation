# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass, field

from torchtitan.config import apply_overrides, Configurable, OverrideConfig
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.overrides.triton_rmsnorm import TritonRMSNorm

_IMPORTS = OverrideConfig(
    imports=["torchtitan.overrides.triton_rmsnorm.triton_rmsnorm"]
)
_HIDDEN = 256


class _Root(Configurable):
    """Stand-in for a model config holding two norms."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        attention_norm: RMSNorm.Config = field(
            default_factory=lambda: RMSNorm.Config(normalized_shape=_HIDDEN, eps=1e-6)
        )
        ffn_norm: RMSNorm.Config = field(
            default_factory=lambda: RMSNorm.Config(normalized_shape=_HIDDEN)
        )


class _OffsetRMSNorm(RMSNorm):
    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        pass


class TestTritonRMSNormOverride(unittest.TestCase):
    def test_replaces_rmsnorm_and_keeps_fields(self):
        config = _Root.Config()

        replacements = apply_overrides(_IMPORTS, config)

        self.assertEqual(len(replacements), 2)
        for norm in (config.attention_norm, config.ffn_norm):
            self.assertIs(type(norm), TritonRMSNorm.Config)
            self.assertEqual(norm.normalized_shape, _HIDDEN)
            self.assertIsInstance(norm.build(), TritonRMSNorm)
        self.assertEqual(config.attention_norm.eps, 1e-6)
        self.assertEqual(config.ffn_norm.eps, RMSNorm.Config(normalized_shape=1).eps)

    def test_unsupported_subclass_is_kept_with_warning(self):
        config = _Root.Config(ffn_norm=_OffsetRMSNorm.Config(normalized_shape=_HIDDEN))

        with self.assertLogs(level="WARNING") as logs:
            apply_overrides(_IMPORTS, config)

        self.assertIs(type(config.attention_norm), TritonRMSNorm.Config)
        self.assertIs(type(config.ffn_norm), _OffsetRMSNorm.Config)
        self.assertTrue(any("does not support" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
