# Copyright (c) 2026, trainstation team

import unittest

import spmd_types as spmd
import torch
from spmd_types._checker import typecheck
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.components.loss import CrossEntropyLoss, IGNORE_INDEX
from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.overrides.triton_cross_entropy import TritonCrossEntropyLoss


@unittest.skipIf(torch.cuda.device_count() < 4, "needs 4 GPUs")
class TestTritonCrossEntropyLossSPMD(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @property
    def device_type(self):
        return "cuda"

    def _run_loss(self, loss_fn, logits, labels, logits_type, labels_type):
        logits = logits.detach().clone().requires_grad_(True)
        spmd.assert_type(logits, logits_type)
        spmd.assert_type(labels, labels_type)
        with typecheck(strict_mode="strict"):
            loss, _ = loss_fn(logits, labels)
        loss.backward()
        return loss, logits.grad

    @with_comms
    def test_matches_default_loss_and_types(self):
        """The Triton loss must match the default CrossEntropyLoss (torch, or
        _LossParallelCrossEntropy under TP) in loss and local-shard grads, pass
        strict SPMD typechecking, and be bitwise equal across repeated calls.
        Covers TP-only, DP+TP and DP-only meshes, even/uneven vocab sharding,
        and IGNORE_INDEX labels."""
        torch.use_deterministic_algorithms(True)

        T = 256
        mesh_shapes = ((1, 1, 4), (2, 1, 2), (4, 1, 1))
        cases = ((32000, 109, False), (32003, 211, False), (32000, 307, True))

        for mesh_shape in mesh_shapes:
            mesh = init_device_mesh(
                self.device_type, mesh_shape, mesh_dim_names=("dp", "cp", "tp")
            )
            coordinate = mesh.get_coordinate()
            assert coordinate is not None
            dp_rank, _, tp_rank = coordinate
            dp_degree, _, tp_degree = mesh_shape
            tp_group = mesh.get_group("tp")
            dp_group = mesh.get_group("dp")

            for vocab_size, seed, ignore in cases:
                with self.subTest(
                    mesh_shape=mesh_shape, vocab_size=vocab_size, ignore=ignore
                ):
                    generator = torch.Generator(device=self.device_type).manual_seed(
                        seed
                    )
                    global_logits = torch.randn(
                        T, vocab_size, device=self.device_type, generator=generator
                    ).bfloat16()
                    global_labels = torch.randint(
                        0,
                        vocab_size,
                        (T,),
                        device=self.device_type,
                        generator=generator,
                    )
                    if ignore:
                        mask = torch.rand(
                            T, device=self.device_type, generator=generator
                        )
                        global_labels[mask < 0.3] = IGNORE_INDEX

                    # local shard: rows split over dp, vocab split over tp
                    rows = T // dp_degree
                    chunk = (vocab_size + tp_degree - 1) // tp_degree
                    vocab_start = min(vocab_size, chunk * tp_rank)
                    vocab_end = min(vocab_size, vocab_start + chunk)
                    row_slice = slice(dp_rank * rows, (dp_rank + 1) * rows)
                    local_logits = global_logits[row_slice, vocab_start:vocab_end]
                    local_labels = global_labels[row_slice]

                    logits_type: dict = {dp_group: spmd.S(0)}
                    labels_type: dict = {dp_group: spmd.S(0)}
                    if tp_degree > 1:
                        logits_type[tp_group] = spmd.S(1)
                        labels_type[tp_group] = spmd.I

                    config = CrossEntropyLoss.Config(global_vocab_size=vocab_size)
                    with set_current_spmd_mesh(mesh):
                        loss_ref, grad_ref = self._run_loss(
                            CrossEntropyLoss(config),
                            local_logits,
                            local_labels,
                            logits_type,
                            labels_type,
                        )
                        triton_loss_fn = TritonCrossEntropyLoss(
                            TritonCrossEntropyLoss.Config(global_vocab_size=vocab_size)
                        )
                        loss, grad = self._run_loss(
                            triton_loss_fn,
                            local_logits,
                            local_labels,
                            logits_type,
                            labels_type,
                        )
                        loss_again, grad_again = self._run_loss(
                            triton_loss_fn,
                            local_logits,
                            local_labels,
                            logits_type,
                            labels_type,
                        )

                    if tp_degree > 1:
                        self.assertIs(spmd.get_axis_local_type(loss, tp_group), spmd.I)
                    torch.testing.assert_close(loss, loss_ref, atol=1e-3, rtol=1e-5)
                    # both compute in fp32 from the same bf16 logits: ~1 bf16 ulp
                    torch.testing.assert_close(grad, grad_ref, atol=1e-9, rtol=1.6e-2)
                    self.assertTrue(torch.equal(loss, loss_again))
                    self.assertTrue(torch.equal(grad, grad_again))


if __name__ == "__main__":
    run_tests()
