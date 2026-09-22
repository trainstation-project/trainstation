# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Download pinned Qwen3 assets and a C4 sample without starting training."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import snapshot_download
from tokenizers import Tokenizer


MODEL_ID = "Qwen/Qwen3-30B-A3B"
MODEL_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
DATASET_ID = "allenai/c4"
DATASET_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
ROOT = Path(__file__).resolve().parents[1]


def download_sample(
    split: str, *, num_documents: int, data_dir: Path, tokenizer: Tokenizer
) -> dict:
    """Stream a bounded prefix of one pinned C4 shard into local JSONL."""
    num_shards = 1024 if split == "train" else 8
    source = f"en/c4-{split}.00000-of-{num_shards:05d}.json.gz"
    url = (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
        f"{DATASET_REVISION}/{source}"
    )
    destination = data_dir / f"{split}.jsonl"
    temporary = destination.with_suffix(".jsonl.partial")
    num_tokens = 0
    checksum = hashlib.sha256()
    with (
        urlopen(url, timeout=120) as response,
        gzip.GzipFile(fileobj=response) as documents,
        temporary.open("wb") as output,
    ):
        for index in range(num_documents):
            line = documents.readline()
            if not line:
                raise ValueError(f"{source} exhausted after {index} documents")
            row = json.loads(line)
            num_tokens += len(
                tokenizer.encode(row["text"], add_special_tokens=False).ids
            )
            encoded = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            output.write(encoded)
            checksum.update(encoded)
            if (index + 1) % 2048 == 0:
                print(
                    f"{split}: {index + 1} documents, {num_tokens} tokens", flush=True
                )
    temporary.replace(destination)
    return {
        "file": destination.name,
        "source": source,
        "selection": f"first {num_documents} documents",
        "num_documents": num_documents,
        "content_tokens": num_tokens,
        "sha256": checksum.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets-dir", type=Path, default=ROOT / "assets/hf/Qwen3-30B-A3B"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data/qwen3_30b_c4_sample"
    )
    parser.add_argument("--train-documents", type=int, default=16384)
    parser.add_argument("--validation-documents", type=int, default=512)
    parser.add_argument(
        "--data-only", action="store_true", help="Use an already downloaded tokenizer"
    )
    args = parser.parse_args()
    if min(args.train_documents, args.validation_documents) <= 0:
        parser.error("document counts must be positive")
    if not args.data_only:
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=args.assets_dir,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "LICENSE", "README.md"],
            max_workers=4,
        )
    tokenizer = Tokenizer.from_file(str(args.assets_dir / "tokenizer.json"))
    args.data_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        split: download_sample(
            split, num_documents=count, data_dir=args.data_dir, tokenizer=tokenizer
        )
        for split, count in (
            ("train", args.train_documents),
            ("validation", args.validation_documents),
        )
    }
    manifest = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_license": "odc-by",
        "token_count_note": "Content tokens only; excludes added BOS/EOS tokens.",
        "purpose": "Small preparation sample, not a representative convergence corpus.",
        "splits": splits,
    }
    (args.data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("Preparation complete. No model execution or training was started.")


if __name__ == "__main__":
    main()
