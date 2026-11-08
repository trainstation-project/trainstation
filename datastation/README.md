# datastation

Data curation for trainstation. `framework/` is our fork of datatrove (see
[framework/README.md](framework/README.md)), and `pipeline/` holds the curation
pipelines built on it.

## Environment

```bash
cd datastation
uv sync
uv run python pipeline/<script>.py
```
