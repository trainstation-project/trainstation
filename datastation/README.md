# datastation

Data curation for trainstation. `framework/` is our fork of datatrove (see
[framework/README.md](framework/README.md)), and `pipeline/` holds the curation
pipelines built on it.

## Environment

datastation is a uv workspace with its own environment (`datastation/.venv`) and
lockfile (`uv.lock`). It is separate from trainstation's training environment
because the two have conflicting dependencies:

- datatrove's `inference` extra pins torch 2.10 (through vllm), while training uses a torch nightly.
- datatrove locks `datasets` 4.8.x, while training requires `datasets<4.8`.

Data moves between the two environments as files on disk (for example tokenized
Megatron `.bin`/`.idx` shards), not through Python imports.

```bash
cd datastation
uv sync                                  # core dependencies only
uv sync --extra io --extra processing    # add datatrove extras as needed
uv run python pipeline/<script>.py
```

`uv sync` is exact: it removes packages that the requested extras don't need, so
pass every extra you need on each call. The extras are datatrove's: `io`, `processing`, `s3`,
`cli`, `ray`, `inference`, `decont`, `multilingual`, `testing` and `quality`
(see [framework/pyproject.toml](framework/pyproject.toml)).

`uv.lock` started as upstream datatrove's lock, with the same package versions.
After changing dependencies in `framework/pyproject.toml`, run `uv lock`.

## Tests

`uv sync --extra testing` also installs vllm, sglang and CUDA torch, several GB in all.
On CPU machines, use a pruned environment instead, the same way upstream CI does:

```bash
cd datastation
uv export --locked --package datatrove --extra testing --no-hashes \
  --prune vllm --prune sglang --prune bitsandbytes --prune torch -o /tmp/requirements-ci.txt
uv venv /tmp/datastation-test
VIRTUAL_ENV=/tmp/datastation-test uv pip install --no-deps -r /tmp/requirements-ci.txt
VIRTUAL_ENV=/tmp/datastation-test uv pip install --no-deps torch==2.10.0 --torch-backend cpu
/tmp/datastation-test/bin/python -m nltk.downloader punkt

cd framework
# Call python directly instead of `uv run`: Ray workers would try to rebuild the env.
# --dist loadfile keeps each test file on a single worker (Ray clusters, stanza downloads).
/tmp/datastation-test/bin/python -m pytest -n 4 --dist loadfile tests/
```

## Linting

trainstation's pre-commit config runs ruff on `framework/`, using the settings in
`framework/pyproject.toml`. trainstation's own formatting, lint and license-header
hooks skip `framework/`. They do apply to `pipeline/`.
