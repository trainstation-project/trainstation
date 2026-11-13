# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass, field

from torchtitan.components.loss import BaseLoss, ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.config import apply_overrides, Configurable, OverrideConfig
from torchtitan.overrides.triton_cross_entropy import (
    triton_cross_entropy_loss,
    TritonCrossEntropyLoss,
)

_IMPORTS = OverrideConfig(
    imports=["torchtitan.overrides.triton_cross_entropy.triton_cross_entropy"]
)
_VOCAB_SIZE = 2048


class _Root(Configurable):
    """Stand-in for the trainer config: a root holding a ``loss`` config."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        loss: BaseLoss.Config = field(default_factory=CrossEntropyLoss.Config)


class _OtherCrossEntropyLoss(CrossEntropyLoss):
    @dataclass(kw_only=True, slots=True)
    class Config(CrossEntropyLoss.Config):
        pass


class TestTritonCrossEntropyOverride(unittest.TestCase):
    def test_replaces_chunked_inner_loss(self):
        config = _Root.Config(
            loss=ChunkedLossWrapper.Config(
                loss_fn=CrossEntropyLoss.Config(global_vocab_size=_VOCAB_SIZE)
            )
        )

        replacements = apply_overrides(_IMPORTS, config)

        self.assertEqual(len(replacements), 1)
        assert isinstance(config.loss, ChunkedLossWrapper.Config)
        inner = config.loss.loss_fn
        self.assertIs(type(inner), TritonCrossEntropyLoss.Config)
        self.assertEqual(inner.global_vocab_size, _VOCAB_SIZE)
        self.assertIs(inner.build().fn, triton_cross_entropy_loss)

    def test_replaces_top_level_loss(self):
        config = _Root.Config(
            loss=CrossEntropyLoss.Config(global_vocab_size=_VOCAB_SIZE)
        )

        apply_overrides(_IMPORTS, config)

        self.assertIs(type(config.loss), TritonCrossEntropyLoss.Config)
        self.assertEqual(config.loss.global_vocab_size, _VOCAB_SIZE)

    def test_unsupported_subclass_is_kept_with_warning(self):
        config = _Root.Config(
            loss=_OtherCrossEntropyLoss.Config(global_vocab_size=_VOCAB_SIZE)
        )

        with self.assertLogs(level="WARNING") as logs:
            apply_overrides(_IMPORTS, config)

        self.assertIs(type(config.loss), _OtherCrossEntropyLoss.Config)
        self.assertTrue(any("does not support" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
