# datastation framework (datatrove fork)

This directory is a fork of [datatrove](https://github.com/huggingface/datatrove), the
Hugging Face library for processing, filtering and deduplicating text data at scale.
datastation pipelines build on it. The package and import name stay `datatrove`
(`from datatrove.pipeline.readers import JsonlReader`).

## Upstream base

- Repository: https://github.com/huggingface/datatrove
- Base commit: `384d81c011fda5cf54be3ba1d58967c21fafe25d` (v0.10.1)
- Copied: `src/`, `tests/`, `pyproject.toml`, `LICENSE`. Not copied: examples, CI, docs.
- Usage documentation for the base version:
  https://github.com/huggingface/datatrove/blob/384d81c011fda5cf54be3ba1d58967c21fafe25d/README.md

The files were copied without git history and are byte-identical to the base commit.
Any changes made after that are tracked in trainstation's git history.

## Pulling in upstream changes

From a datatrove checkout, apply the upstream diff between the base and the new
commit, then update the base commit above:

```bash
git -C /path/to/datatrove diff 384d81c011fda5cf54be3ba1d58967c21fafe25d <new_commit> -- src tests pyproject.toml \
  | git -C /path/to/trainstation apply -3 --directory=datastation/framework
```

## Development

The framework is part of the datastation uv workspace and has its own environment,
separate from trainstation's. See [../README.md](../README.md) for setup.

Style follows upstream: ruff, configured in `pyproject.toml` (line length 119).
trainstation's black/usort/flake8/license-header pre-commit hooks skip this directory,
and ruff runs on it instead.

## License

datatrove is licensed under Apache-2.0 (see [LICENSE](LICENSE)), not the BSD license at
the trainstation root. The license and upstream copyright notices must be kept.
