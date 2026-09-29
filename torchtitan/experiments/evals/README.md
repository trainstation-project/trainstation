# Pretraining evals

Evaluate the checkpoints of a training run with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
while it trains, without touching the training job and without converting
checkpoints to Hugging Face format.

- **Out of band.** The eval job runs next to the training job on its own GPUs
  and only reads the `step-N` checkpoints training writes. Training is never
  paused or slowed.
- **Native checkpoints.** Each DCP checkpoint is loaded straight into the
  torchtitan model built from the training config, so what is evaluated is
  exactly what was trained. Convert to HF once, for the final model only
  (`scripts/checkpoint_conversion/convert_to_hf.py`).
- **Same numerics as training.** Requests are packed into token buffers in the
  training collator's layout and scored through the model's own
  `preprocess_inputs` and attention masking. On training documents the eval
  path reproduces the trainer's cross-entropy exactly.

## Setup

```bash
pip install -r torchtitan/experiments/evals/requirements.txt
```

This adds lm-eval to the training env; it does not touch the installed torch.

## Usage

Pass the eval options, then `--`, then the exact training arguments (the same
`--module`, `--config` and overrides as the training job, so the eval finds
its `dump_folder` and builds the same model):

```bash
TRAIN_ARGS="--module llama3 --config llama3_8b --dump_folder ./outputs/run7"

# Watch mode: evaluate each checkpoint as it lands; exits once the final
# training step (training.steps) has been evaluated, or after
# --max-idle-hours (default 24) without a new checkpoint.
torchrun --nproc_per_node=8 -m torchtitan.experiments.evals.evaluate \
    --tasks trainstation_quick --watch -- $TRAIN_ARGS

# Backfill: every completed checkpoint, on the tasks it has no results for.
torchrun --nproc_per_node=8 -m torchtitan.experiments.evals.evaluate \
    --tasks trainstation_full -- $TRAIN_ARGS

# Specific steps.
torchrun --nproc_per_node=8 -m torchtitan.experiments.evals.evaluate \
    --tasks trainstation_full --steps 20000 40000 -- $TRAIN_ARGS
```

Results go to `<dump_folder>/evals/step-N/<task>.json`, one file per task
(lm-eval's `results`, plus step, dtype and lm-eval version). Suites are
expanded into their tasks, while groups with an aggregate score, such as
MMLU, keep one file. A task with results is skipped on later runs, so
re-running is cheap, a crashed job can simply be restarted, a task added to a
suite is backfilled on old steps, and a task in two suites runs once per step.

The step-0 checkpoint written by `checkpoint.create_seed_checkpoint` is
evaluated too. It is the model training starts from, so its scores are each
task's baseline.

### Models that do not fit on one GPU

`--tensor-parallel-degree N` shards each model replica over N GPUs with the
model's own tensor-parallel plan; the remaining GPUs are data-parallel
replicas that lm-eval splits the examples across. For example, 8 GPUs with
`--tensor-parallel-degree 4` run two replicas of four GPUs each. The model is
parallelized the same way as for training, so the checkpoint loads through
DCP resharding whatever layout it was trained with.

Other axes of the training layout (FSDP sharding, EP, CP, PP) are not used by
the eval job: lm-eval gives each data-parallel rank different requests, and
those axes communicate across ranks in the forward pass.

### W&B

`--wandb` logs every result as `eval/<task>/<metric>` to a W&B run named
`<run name>-eval-<suite>`. It reads the same environment variables as the
training job's W&B logger (`WANDB_TEAM`, `WANDB_PROJECT`, `WANDB_RUN_NAME`,
`WANDB_RUN_GROUP`). To group the eval run with the training run, set the same
`WANDB_RUN_GROUP` for both jobs. Metrics are plotted
against `train_step`, so backfilled or out-of-order results land in the right
place, and a restarted eval job resumes the same run.

### Other options

`--limit` (examples per task, for smoke tests), `--dtype`,
`--num-tokens-per-batch`, `--poll-interval`, `--max-idle-hours`,
`--output-folder`, `--decontaminate` (see [Contamination](#contamination));
see `--help`.

Rank 0 downloads the eval datasets once at startup; after that every rank
reads them from the local cache. Loading a task makes about ten Hugging Face
Hub requests even when it is cached, and the Hub allows 1000 requests per 5
minutes, so every rank loading every task at each step would be throttled.

Models with multi-token-prediction layers (the DeepSeek `_mtp` configs) are
evaluated on their main output layer; the MTP layers are not built.

Make sure `checkpoint.keep_latest_k` does not delete checkpoints before they
are evaluated; `checkpoint.purge_exempt` can keep selected steps.

## Contamination

Benchmark questions that also appear in the training data raise scores
through memorization. `contamination.py` checks for this with the 13-gram
overlap test used for GPT-3: an eval question counts as contaminated when any
13 consecutive words of it (lowercased, punctuation removed) also occur in a
training document.

It reads the training data through the run's own dataloader and decodes it
back to text, so it checks exactly what the run trains on, whatever the data
source. By default it scans the run's whole token budget
(`training.steps` x `training.num_tokens_per_train_step`). It needs no GPUs,
and each process scans its own share of the data:

```bash
torchrun --nproc_per_node=32 -m torchtitan.experiments.evals.contamination \
    --tasks trainstation_full [--max-tokens 1000000000] -- $TRAIN_ARGS
```

The report, `<dump_folder>/evals/contamination.json`, lists for each task
how many questions overlap the training data, and which. Questions shorter
than 13 words cannot be checked and count as clean.

With `--decontaminate`, the eval job also scores each task on its clean
questions only, written as `step-N/<task>.clean.json` and logged to W&B under
`eval_clean/`. A gap between the full and clean scores shows how much a task
is inflated by contamination. For reading-comprehension tasks such as BoolQ,
an overlap is usually the passage (often from Wikipedia) rather than the
question and answer.

## Suites

Suites are lm-eval groups defined in [`tasks/`](./tasks); any lm-eval task or
group name also works with `--tasks`.

| Suite | Tasks |
|---|---|
| `trainstation_quick` | WikiText (bits per byte), LAMBADA, HellaSwag, ARC-Easy, PIQA |
| `trainstation_full` | `trainstation_quick` + ARC-Challenge, WinoGrande, BoolQ, OpenBookQA, SciQ, `trainstation_mmlu` |
| `trainstation_mmlu` | MMLU, 5-shot, accuracy weighted by subject size |

Paloma is a good addition to `trainstation_quick` once your Hugging Face
account has accepted its terms (`allenai/paloma` is gated).

## Limitations

- Log-likelihood tasks only. Generative tasks (GSM8K, HumanEval, ...) need a
  generation loop and are not supported yet.
- No FSDP, EP, CP or PP in the eval job (see above), so MoE models too large
  for tensor parallelism alone are not supported yet.
- Scoring in the training loop is not implemented yet; `TrainstationLM` is
  written so a validator can reuse it.
