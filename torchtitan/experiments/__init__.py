# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

_supported_experiments = frozenset(
    [
        "graph_trainer.llama3",
        "graph_trainer.deepseek_v3",
        "transformers_modeling_backend",
        "torchft.llama3",
        "rl",
    ]
)
