# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tokenize the experiment data with each local tokenizer export."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[2]


def prepare_evals(data_dir: Path, destination: Path) -> None:
    """Select downloaded eval shards, preserving sources and splits."""
    for dataset, output_name in [
        ("paloma", "paloma"),
        ("proof-pile-2", "open-web-math"),
    ]:
        source = data_dir / dataset
        manifest = json.loads((source / "download-manifest.json").read_text())
        verification = json.loads((source / "download-verification.json").read_text())
        empty_files = set(verification.get("empty_jsonl_files", []))
        for entry in manifest["files"]:
            relative = Path(entry["path"])
            if str(relative) in empty_files:
                print(f"Skip empty eval shard: {dataset}/{relative}", flush=True)
                continue
            input_file = (source / relative).resolve(strict=True)
            if dataset == "proof-pile-2":
                if relative != Path("open-web-math/validation/val.jsonl.zst"):
                    continue
                relative = relative.relative_to("open-web-math")
            link = destination / output_name / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(input_file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=["train", "eval", "all"])
    parser.add_argument(
        "--tokenizers", nargs="+", help="Folder names; defaults to all exports"
    )
    parser.add_argument(
        "--tokenizer-root", type=Path, default=Path.home() / "tokenizers/tokenizers"
    )
    parser.add_argument("--data-root", type=Path, default=REPO_DIR / "data")
    parser.add_argument(
        "--train-input",
        type=Path,
        help="Defaults to data/dolmino-decontaminated/cleaned",
    )
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/shared/data"))
    parser.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    tokenizer_root = args.tokenizer_root.resolve()
    tokenizers = sorted(tokenizer_root.glob("*/tokenizer.json"))
    if args.tokenizers:
        tokenizers = [
            tokenizer_root / name / "tokenizer.json" for name in args.tokenizers
        ]
    if not tokenizers:
        parser.error(f"No tokenizer exports found in {tokenizer_root}")
    for tokenizer in tokenizers:
        if not tokenizer.is_file():
            parser.error(f"Tokenizer file does not exist: {tokenizer}")

    train_input = args.train_input or args.data_root / "dolmino-decontaminated/cleaned"
    if args.split in {"train", "all"} and not train_input.is_dir():
        parser.error(
            f"Cleaned training data not found: {train_input}. "
            "Create the cleaned copy with decon.sh --purify, or set --train-input explicitly."
        )

    # Each subprocess uses one CPU; the preprocessor controls file-level parallelism.
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(REPO_DIR), os.environ.get("PYTHONPATH")])
        ),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    with tempfile.TemporaryDirectory(prefix="tokenizer-evals-") as temporary:
        jobs = []
        if args.split in {"train", "all"}:
            jobs.append((train_input.resolve(), "train"))
        if args.split in {"eval", "all"}:
            eval_input = Path(temporary)
            prepare_evals(args.data_root.resolve(), eval_input)
            jobs.append((eval_input, "test"))

        outputs = [
            args.output_root / path.parent.name / split
            for path in tokenizers
            for _, split in jobs
        ]
        if len(set(outputs)) != len(outputs):
            parser.error("Each tokenizer must be selected once")
        for output in outputs:
            if output.exists():
                parser.error(
                    f"Output already exists: {output}. Select a fresh --output-root."
                )

        for tokenizer in tokenizers:
            for input_dir, split in jobs:
                output = args.output_root / tokenizer.parent.name / split
                output.mkdir(parents=True)
                command = [
                    sys.executable,
                    "-m",
                    "scripts.data.megatron_preprocess",
                    "--input",
                    str(input_dir),
                    "--tokenizer",
                    str(tokenizer.parent),
                    "--output-prefix",
                    str(output.resolve()),
                    "--json-keys",
                    "text",
                    "--append-eod",
                    "--max-local-processes",
                    str(args.workers),
                ]
                print(
                    f"Tokenizing {tokenizer.parent.name}: {split} -> {output}",
                    flush=True,
                )
                with (output / "tokenize.log").open("w") as log:
                    subprocess.run(
                        command,
                        cwd=REPO_DIR,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                (output / "_SUCCESS").touch()
                print(f"Completed {output}", flush=True)


if __name__ == "__main__":
    main()
