# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checks for eval selection and tokenization run boundaries."""

import json
import sys

import pytest

from rough.tokenizer_experiments import tokenize


def test_prepare_evals_keeps_splits_and_skips_known_empty_shards(tmp_path):
    for dataset, paths, empty in [
        (
            "paloma",
            [
                "source/val/a.jsonl.gz",
                "source/test/b.jsonl.gz",
                "source/test/empty.jsonl.gz",
            ],
            ["source/test/empty.jsonl.gz"],
        ),
        ("proof-pile-2", ["open-web-math/validation/val.jsonl.zst"], []),
    ]:
        root = tmp_path / dataset
        for path in paths:
            source = root / path
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"fixture")
        (root / "download-manifest.json").write_text(
            json.dumps({"files": [{"path": path} for path in paths]})
        )
        (root / "download-verification.json").write_text(
            json.dumps({"empty_jsonl_files": empty})
        )

    output = tmp_path / "selected"
    tokenize.prepare_evals(tmp_path, output)
    selected = sorted(
        str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
    )
    assert selected == [
        "open-web-math/validation/val.jsonl.zst",
        "paloma/source/test/b.jsonl.gz",
        "paloma/source/val/a.jsonl.gz",
    ]
    assert all(path.is_symlink() for path in output.rglob("*") if path.is_file())


def test_train_requires_explicit_input_when_cleaned_data_is_missing(
    tmp_path, monkeypatch
):
    root = tmp_path / "tokenizers/example"
    root.mkdir(parents=True)
    (root / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tokenize.py",
            "train",
            "--tokenizer-root",
            str(root.parent),
            "--data-root",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "output"),
        ],
    )
    with pytest.raises(SystemExit):
        tokenize.main()
    assert not (tmp_path / "output").exists()


def test_failed_job_has_no_success_marker(tmp_path, monkeypatch):
    root = tmp_path / "tokenizers/example"
    root.mkdir(parents=True)
    (root / "tokenizer.json").write_text("{}")
    training = tmp_path / "training"
    training.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tokenize.py",
            "train",
            "--tokenizer-root",
            str(root.parent),
            "--train-input",
            str(training),
            "--output-root",
            str(tmp_path / "output"),
        ],
    )

    def fail(command, **kwargs):
        raise tokenize.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(tokenize.subprocess, "run", fail)
    with pytest.raises(tokenize.subprocess.CalledProcessError):
        tokenize.main()
    assert not (tmp_path / "output/example/train/_SUCCESS").exists()
