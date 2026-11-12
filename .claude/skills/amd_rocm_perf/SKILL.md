---
name: amd_rocm_perf
description: Make torchtitan pretraining faster on AMD Instinct GPUs (MI300X / MI325X / MI355X, ROCm). Covers the ROCm launch env, the throughput levers (tokens-per-microbatch, EP, activation checkpointing, compile), reading a ROCm profile to tell compute-bound from RCCL-bound steps, and the AMD-specific MFU and bit-identical gotchas. Use when speeding up or reviewing a torchtitan run on AMD/ROCm. For the hardware-agnostic method, defer to AGENTS.md and the numerics_debugging, torch_bisect, and cuda_graph_trace_compaction skills.
---

# Optimizing torchtitan on AMD (ROCm)

AMD-specific notes for making a run faster on MI300X / MI325X / MI355X. The
method is the same as on NVIDIA -- freeze a config, profile, change one thing,
confirm on the real metric, keep only a bit-identical (or converged) win -- so
this skill records only what is different on ROCm. For the shared machinery:

- numerics / bit-identical loss: `AGENTS.md` ("Validating Numerics"),
  `scripts/loss_compare.py`, and the `numerics_debugging` skill.
- regressions: the `torch_bisect` skill.
- CUDA-graph trace reading: the `cuda_graph_trace_compaction` skill.
- per-op kernel numerics: the `kernel-numerics-verifier` skill.

Gate every AMD-only env var or kernel path behind `torch.version.hip` so the
NVIDIA path is untouched, and record which GPU confirmed each number.

## Launch

Example: single 8x MI355X node, Qwen3-30B-A3B (see
`scripts/amd/run_qwen3_30b_a3b_mi355.sh`). Install the ROCm nightly build:

```bash
pip install --pre torch torchao --index-url https://download.pytorch.org/whl/nightly/rocm10.0
pip install -r requirements.txt && pip install -e . --no-deps
```

ROCm / RCCL settings that matter:

- `GPU_MAX_HW_QUEUES=2` -- more hardware queues did not help and added contention.
- `PYTORCH_TUNABLEOP_ENABLED=0` -- TunableOp GEMM autotuning is off by default.
  Turning it on is a separate, large first-step cost, so measure it as its own
  A/B and keep its cache out of the run's working copy.
- Multinode only: `NCCL_PXN_DISABLE=0` and `NCCL_P2P_NET_CHUNKSIZE=262144` speed
  up cross-node all-to-all (RCCL honors the NCCL_* names).
- CUDA graphs are disabled for the default all-to-all EP dispatcher (it needs a
  CPU sync): `--training.disable-cuda-graphs`.

Pin each run's GPUs with BOTH `ROCR_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES`
(HIP's device order is not amd-smi's; never map ids yourself), and give each
concurrent run its own rendezvous port.

## Measure (AMD gotchas on top of AGENTS.md)

Follow AGENTS.md "Performance Testing": at least 10 steps, read the rank-0
`tps / tflops / mfu` line, average a steady-state window (MFU peaks early and
settles by ~step 100). Two AMD-specific MFU traps:

1. **The peak-FLOPs lookup matches by device-name substring.** `get_peak_flops`
   (`torchtitan/tools/utils.py`) matches `"MI355X"` in
   `torch.cuda.get_device_name()`. If a ROCm / PyTorch build reports a gfx name
   (e.g. `gfx950`) instead, the match fails and the function **falls back to
   A100 (312 TFLOPS)**, so MFU reads ~8x too high and is silently wrong. Print
   the device name once and confirm it contains the expected `MI3..X` token
   before trusting any MFU number.
2. **The denominator is BF16 dense peak** (MI355X: 2.5 PFLOPS, matching the
   file's dense convention for NVIDIA parts). If you enable float8 / FP8 matmuls
   (torchao), the achieved FLOPs run against a much higher FP8 peak, so MFU
   reported against the BF16 peak understates utilization -- state the dtype with
   the number, and confirm 2.5 PFLOPS is the dense (not sparsity-inflated) figure
   for your part.

tokens/day = `tps * num_gpus * 86400`.

## The levers (priors from the MI355X Qwen3-30B-A3B study; re-measure on your config)

- **Tokens per microbatch is the main MFU lever.** The per-step fixed costs (FSDP
  all-gather / reduce-scatter of the params, the optimizer update) do not scale
  with tokens; a bigger microbatch amortizes them. 16k tok/GPU gave ~9-12% MFU;
  128k with selective AC gave ~22-26%.
- **Gradient accumulation does not substitute.** Plain FSDP still all-gathers and
  reduce-scatters every microbatch; only a larger microbatch (or HSDP, which
  skips the inter-replica sync) amortizes the communication.
- **Activation checkpointing buys the bigger microbatch.** `full` frees the most
  memory but recomputes most; `selective` was the sweet spot. The MFU win is
  indirect: AC -> larger microbatch -> higher MFU.
- **Expert parallelism is not free.** On one 8-GPU node EP=1 (plain FSDP) beat
  EP=2 / 4 / 8 because the MoE all-to-all dominated. Sweep EP; do not assume more
  is faster.

## Reading a ROCm profile

Capture with the repo profiler (`torchtitan/observability/profiler.py`); it emits
a kineto trace you open in Perfetto or `chrome://tracing`. For kernel and
collective detail beyond that, use ROCm's own tools: `rocprofv3` (per-kernel),
`rocprof-sys` (timeline, formerly omnitrace), `rocprof-compute` (per-kernel
counters, formerly omniperf). Prefer a single-rank trace.

Classify the step before optimizing:

- Group GPU time by family (GEMM, attention, norm / pointwise, scatter / gather,
  and **RCCL collectives**). RCCL kernels carry `ncclDevKernel_*` / all-to-all
  names.
- If roughly half the GPU time is RCCL (the MoE all-to-all) and GEMMs are ~10%,
  the step is **communication-bound**: a faster grouped GEMM buys almost nothing.
  The lever is compute / communication overlap or a fused EP dispatcher
  (DeepEP-style), not the GEMM.
- For collective-level logging use `NCCL_DEBUG=INFO` (RCCL honors the NCCL_*
  names).

## Bit-identical / correctness gate

"Optimize bit-identical" is the repo's standard for a non-computation change. Use
the repo's mechanism, not a hand-rolled tolerance:

- Run both sides with `--debug.seed=42 --debug.deterministic` (NEVER
  `--debug.deterministic_warn_only`) and compare **loss and grad_norm** with
  `scripts/loss_compare.py --assert-equal`. stdout prints only 5 significant
  digits, so compare from TensorBoard, as that script does.
- A change that must not alter math (a custom op replacing an indexed gather, a
  layout change, an AC / refactor) must produce **bitwise identical** loss and
  grad_norm. A change that does alter math (a new kernel) needs loss convergence
  on a representative dataset (C4), not bit-identity.
- For a custom op under `torchtitan/ops/` (e.g. `topk_permute_gather`): run
  `torch.library.opcheck` and `torch.autograd.gradcheck` (fp64 inputs) and match
  a high-precision reference for forward and backward, as
  `tests/unit_tests/cpu/test_topk_gather.py` does (forward `rtol=0, atol=0`).
- **ROCm determinism caveat.** Some ROCm reductions and atomic accumulations are
  nondeterministic; a kernel that relies on them can fail bit-identity even when
  the math is "right." Prefer a deterministic formulation. The
  `topk_permute_gather` backward is the model: it replaces `index_put_`'s sorted
  atomic scatter with an inverse-permutation gather plus a fixed-order fp32 sum --
  no sort, no atomics, deterministic -- which is both faster on ROCm and
  bit-stable.
