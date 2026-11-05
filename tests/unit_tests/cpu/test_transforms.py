# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model transform base class, ordering, and the context-parallel transform."""

import unittest

import torchtitan.config.transform as transform_api
from torchtitan.config.transform import (
    ContextParallelTransform,
    convert_config_type,
    ModelConfigTransform,
)

from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.cp_attention import KVAllGatherCPFlexInnerAttention


class _Record(ModelConfigTransform):
    order: list[str] = []

    def transform(self, model):
        _Record.order.append(type(self).__qualname__)
        return model


class _First(_Record):
    pass


class _Second(_Record):
    run_after = (_First,)


class _Third(_Record):
    run_after = (_Second,)


class _Rival(_Record):
    conflicts_with = (_First,)


class _SelfConflicting(_Record):
    pass


_SelfConflicting.conflicts_with = (_SelfConflicting,)


class _Loose(_Record):
    pass


class _Boom(ModelConfigTransform):
    def transform(self, model):
        model.layers[0].attention.inner_attention.block_size = (1, 1)
        raise ValueError("boom")


class TestConvertConfigType(unittest.TestCase):
    def test_keeps_the_fields_of_the_config_it_replaces(self):
        existing = FlexInnerAttention.Config()
        existing.block_size = (256, 128)
        existing.kernel_options = {"BACKEND": "FLASH"}

        swapped = convert_config_type(existing, KVAllGatherCPFlexInnerAttention)

        self.assertIsInstance(swapped, KVAllGatherCPFlexInnerAttention.Config)
        self.assertEqual(swapped.block_size, (256, 128))
        self.assertEqual(swapped.kernel_options, {"BACKEND": "FLASH"})

    def test_rejects_a_replacement_that_does_not_inherit_the_current_type(self):
        # A non-subclass would drop fields added by an earlier transform.
        existing = KVAllGatherCPFlexInnerAttention.Config()
        with self.assertRaisesRegex(ValueError, "must inherit"):
            convert_config_type(existing, FlexInnerAttention)


class TestOrdering(unittest.TestCase):
    def setUp(self):
        _Record.order = []


class TestTransformModel(unittest.TestCase):
    """The primitive runs on a model config alone, with no trainer config."""


class TestContextParallelTransform(unittest.TestCase):
    def test_linear_lora_handler_is_exported(self):
        handler_cls = getattr(transform_api, "LinearLoRAHandler", None)
        self.assertIsNotNone(handler_cls, "LinearLoRAHandler is not exported")

    def test_rejects_a_kernel_that_is_not_context_parallel(self):
        with self.assertRaisesRegex(ValueError, "must inherit CPInnerAttention"):
            ContextParallelTransform(inner_attention=FlexInnerAttention)


if __name__ == "__main__":
    unittest.main()
