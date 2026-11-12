---
name: distributed_hang_debug
description: Debug a hanging or timed-out torchtitan training run (FSDP / HSDP / TP / PP / CP / EP). Locate the stuck collective with the PyTorch Flight Recorder and py-spy, diff per-rank state to find where ranks diverge, and apply the common fixes. Use when a multi-GPU/multi-node run freezes, stalls, or raises a collective watchdog timeout.
---

# Debugging distributed hangs

A hang is almost always ranks disagreeing on what collective to run next, so one
rank waits forever (all-gather / reduce-scatter / all-to-all / broadcast /
barrier, or a PP send/recv). Common causes: a tensor-size mismatch into a
collective, a branch one rank takes and another skips, cascading state drift from
nondeterminism, or one rank OOMing while the others wait.

For AMD/ROCm launch specifics see `amd_rocm_perf`; the distributed-code
invariants are in `.claude/rules/distributed.md`. Always record the full
parallelism config (mesh axes and degrees) with any hang report.

## Step 1: locate the stuck collective

**Flight Recorder (preferred, PyTorch-native).** Launch with the NCCL/RCCL trace
buffer on; on timeout each rank dumps its in-flight and recent collectives:

```bash
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000   # ring of recent collectives per rank
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/tmp/fr_trace_   # dump path prefix
```

Analyze the dumps with `torch/distributed/flight_recorder/fr_trace.py`: the rank
whose last collective has no matching peers, or whose collective count is one
behind, is where divergence surfaced.

**py-spy.** Dump every rank's Python stack and compare; the hung rank sits in a
`c10d`/NCCL collective or a CUDA sync:

```bash
for pid in $(pgrep -f torchtitan.train); do py-spy dump --pid $pid > /tmp/stack_$pid.txt; done
```

**More signal.** `TORCH_DISTRIBUTED_DEBUG=DETAIL`, `TORCH_CPP_LOG_LEVEL=INFO`,
and `NCCL_DEBUG=INFO` (RCCL honors the NCCL_* names). `NCCL_DEBUG_SUBSYS=COLL`
prints each collective's shape -- a size mismatch shows one rank entering with a
different count.

## Step 2: find the first diverging rank

Have each rank append structured events to its own file, gated behind an env var
so it is free in normal runs. Log things already on CPU (step, shapes, counts,
routing sizes); hashing a GPU tensor forces a sync that can move the hang, so do
it only between steps when the GPU is idle:

```python
f.write(f"STEP step={step} tokens={n} route_counts={counts}\n"); f.flush()
```

Then diff the per-rank logs; the first differing line is at or before the root
cause. `grep -c` the event to spot a rank that ran one extra iteration.

## Step 3: binary-search the cause

For the diverging op, hash each input per rank and diff; recurse into whichever
input differs until you reach the source.

## Common causes and fixes

- **Data-dependent shape into a collective** (MoE routing counts, variable
  seqlen, dropped tokens): the all-to-all / all-gather sizes differ per rank.
  Synchronize the size, or pad to a fixed capacity.
- **Nondeterminism cascading into different batches.** Reproduce under
  `--debug.seed=42 --debug.deterministic`; generate randomness on one rank and
  broadcast, or use a shared seed. See `numerics_debugging` for per-op drift.
- **A per-rank branch** (an OOM check, a queue length, an `if rank == 0`) that
  enters a collective on some ranks only: make the predicate identical across
  ranks before branching.
- **PP send/recv mismatch**: stages disagree on microbatch count or send/recv
  order. Unlike TP collective hangs these are point-to-point; line up the
  schedule.
- **One rank OOMs or crashes** and the rest wait: check every rank's log, not
  just rank 0.

## Verify the fix

Intermittent hangs need many clean passes. A run that hung ~30% of the time needs
10+ clean repeats at the same parallelism config before you trust the fix.
