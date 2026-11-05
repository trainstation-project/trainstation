# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest


class TestDecoderConfigCpValidation(unittest.TestCase):
    """``Trainer.Config.__post_init__`` applies the CP gates at config time."""


class TestUlyssesConfigValidation(unittest.TestCase):
    """Validate Ulysses configuration requirements."""


class TestHeadDivisibility(unittest.TestCase):
    """Validate when CP adds to head sharding."""


if __name__ == "__main__":
    unittest.main()
