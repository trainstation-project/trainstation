# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import Mock

import torch

import torchtitan.experiments.torchft.trainer as ft
from torchtitan.components.loss import ChunkedLossWrapper


def test_ft_chunked_loss_without_pp_binds_lm_head_and_skips_model_projection():
    model = SimpleNamespace(lm_head=torch.nn.Linear(4, 8), _skip_lm_head=False)
    trainer = object.__new__(ft.FaultTolerantTrainer)
    trainer.parallel_dims = SimpleNamespace(pp_enabled=False)
    trainer.model_parts = [model]
    trainer.loss_fn = ChunkedLossWrapper.Config(num_chunks=2).build()

    trainer._configure_chunked_loss()

    assert trainer.loss_fn.lm_head is model.lm_head
    assert model._skip_lm_head is True


def test_ft_averages_logged_loss_by_active_replica_count(monkeypatch):
    trainer = Mock(
        spec=ft.FaultTolerantTrainer,
        config=Mock(training=Mock(disable_cuda_graphs=True, max_norm=1.0)),
        device=torch.device("cpu"),
        parallel_dims=Mock(
            dp_enabled=False, dp_cp_enabled=True, pp_enabled=False, ep_enabled=False
        ),
        ft_manager=Mock(loss_sync_pg=Mock(size=lambda: 2), group_size=4),
        optimizers=Mock(),
        lr_schedulers=Mock(schedulers=[Mock(get_last_lr=lambda: [0.1])]),
        metrics_processor=Mock(should_log=Mock(return_value=True)),
        checkpointer=Mock(),
        model_parts=[],
        gradient_accumulation_steps=1,
        num_pp_microbatches=1,
        sdc_replayer=None,
        step=1,
        ntokens_seen=4,
        forward_backward_step=Mock(return_value=torch.tensor(2.0)),
    )
    # Two active replicas contribute a loss sum of 4.0, despite group_size=4.
    monkeypatch.setattr(ft.dist_utils, "dist_sum", Mock(side_effect=[4.0, 8]))
    monkeypatch.setattr(ft.dist_utils, "dist_max", Mock(return_value=2.0))
    monkeypatch.setattr(
        ft.dist_utils, "clip_grad_norm_", Mock(return_value=torch.tensor(0.0))
    )
    monkeypatch.setattr(ft.AuxLoss, "set_step_denominator", Mock())
    monkeypatch.setattr(ft, "collect_aux_loss_metrics", Mock(return_value={}))

    ft.FaultTolerantTrainer.train_step(trainer, iter([{"num_valid_tokens": 4}]))

    trainer.metrics_processor.log.assert_called_once()
    _, logged_loss, *_ = trainer.metrics_processor.log.call_args.args
    assert logged_loss == 2.0
