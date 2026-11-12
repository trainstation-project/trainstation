---
name: gpu_crash_debug
description: Debug a torchtitan run that crashes with a CUDA/HIP error (illegal memory access, device-side assert, misaligned address) or produces NaN/Inf loss. Capture the inputs that reached the failing op, localize the bad kernel, and separate a real memory fault from a numerics blow-up. Use when training aborts with a device error or the loss goes NaN/Inf.
---

# Debugging GPU crashes and NaN/Inf loss

Two different failures that often look alike:

- A **device error** (illegal memory access, device-side assert, misaligned
  address) -- usually a shape/indexing/layout bug in a kernel or custom op.
- A **numerics blow-up** (loss or grad is NaN/Inf) -- usually bad values flowing
  in, not a memory fault.

For per-op activation divergence between two runs that should agree, use
`numerics_debugging`. For custom-op correctness, use `kernel-numerics-verifier`.

## Make the error point at the real site

CUDA/HIP kernels are async, so the Python traceback usually points at a later,
innocent line. Force synchronous reporting first:

```bash
CUDA_LAUNCH_BLOCKING=1 ./run_train.sh ...     # NVIDIA and ROCm both honor this
```

Now the traceback lands on the launching op. Re-run and read the shapes/dtypes of
its inputs.

## Device error (illegal access / device assert / misaligned)

- A **device-side assert** is usually an out-of-range index (embedding id past
  vocab, a gather index past its dim, a routing id past num_experts). Check the
  index tensor's min/max against the indexed dim.
- A **misaligned address** is usually a vectorized kernel meeting a padded /
  non-contiguous row; check `.is_contiguous()` and strides of the inputs.
- Localize the faulting kernel:
  - NVIDIA: `compute-sanitizer --tool memcheck ./run_train.sh ...`
  - ROCm: run under `rocgdb`, and set `AMD_LOG_LEVEL=3` for HSA/HIP fault detail;
    `PYTORCH_NO_HIP_MEMORY_CACHING=1` makes a use-after-free fault at the real
    allocation instead of inside the caching allocator.
- For a kernel/op you own, `printf` the suspect index and launch geometry, then
  `torch.cuda.synchronize()` to flush. In a warp-specialized kernel print one
  representative lane per warp, not only `threadIdx.x == 0`.

## NaN/Inf loss

- Find the first op that produces the non-finite value:

  ```python
  with torch.autograd.detect_anomaly():
      loss.backward()   # raises at the backward op that first makes NaN/Inf
  ```

  (Slow; use it only to localize, then turn it off.)
- Check the usual sources: a too-high LR or a bad warmup, an unclipped grad
  spike (`--training.max-norm`), `log`/`sqrt`/`div` of a near-zero or negative,
  softmax over a fully-masked row, or fp16/fp8 overflow (prefer bf16).
- Make it reproducible with `--debug.seed=42 --debug.deterministic` so the step
  that first goes non-finite is stable, then bisect inputs into that op.
- If a change (not an LR change) introduced the NaN, confirm with
  `scripts/loss_compare.py` against the pre-change commit; `torch_bisect` finds
  the offending commit.

## When it is distributed

A crash on one rank stalls the rest (they wait in a collective). Read every
rank's log, not just rank 0; see `distributed_hang_debug`.
