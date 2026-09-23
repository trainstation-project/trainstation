# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Download pinned Qwen3 assets and a bounded C4 corpus without training."""

import argparse
import gzip
import hashlib
import json
import random
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
    split: str,
    *,
    num_documents: int,
    num_source_shards: int,
    seed: int,
    data_dir: Path,
    tokenizer: Tokenizer,
    seen_texts: set[bytes],
) -> dict:
    """Sample shard prefixes, excluding exact text duplicates across splits."""
    num_shards = 1024 if split == "train" else 8
    shard_ids = (
        [0]
        if num_source_shards == 1
        else random.Random(seed).sample(range(num_shards), num_source_shards)
    )
    destination = data_dir / f"{split}.jsonl"
    temporary = destination.with_suffix(".jsonl.partial")
    num_tokens = 0
    num_duplicates = 0
    sources = []
    checksum = hashlib.sha256()
    with temporary.open("wb") as output:
        for shard_index, shard_id in enumerate(shard_ids):
            source = f"en/c4-{split}.{shard_id:05d}-of-{num_shards:05d}.json.gz"
            url = (
                f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
                f"{DATASET_REVISION}/{source}"
            )
            target = num_documents // num_source_shards + (
                shard_index < num_documents % num_source_shards
            )
            accepted = 0
            with (
                urlopen(url, timeout=120) as response,
                gzip.GzipFile(fileobj=response) as documents,
            ):
                while accepted < target:
                    line = documents.readline()
                    if not line:
                        raise ValueError(
                            f"{source} exhausted after {accepted} unique documents"
                        )
                    row = json.loads(line)
                    digest = hashlib.sha256(row["text"].encode("utf-8")).digest()
                    if digest in seen_texts:
                        num_duplicates += 1
                        continue
                    seen_texts.add(digest)
                    row["content_tokens"] = len(
                        tokenizer.encode(row["text"], add_special_tokens=False).ids
                    )
                    num_tokens += row["content_tokens"]
                    encoded = (json.dumps(row, ensure_ascii=False) + "\n").encode(
                        "utf-8"
                    )
                    output.write(encoded)
                    checksum.update(encoded)
                    accepted += 1
                    if accepted % 2048 == 0:
                        print(
                            f"{split} shard {shard_id}: {accepted}/{target} documents, "
                            f"{num_tokens} total tokens",
                            flush=True,
                        )
            sources.append({"source": source, "num_documents": accepted})
    temporary.replace(destination)
    return {
        "file": destination.name,
        "sources": sources,
        "selection": "Equal document quotas from seeded shard prefixes",
        "seed": seed,
        "num_documents": num_documents,
        "num_duplicates_skipped": num_duplicates,
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
    parser.add_argument("--train-shards", type=int, default=1)
    parser.add_argument("--validation-shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-only", action="store_true", help="Use an already downloaded tokenizer"
    )
    args = parser.parse_args()
    if min(args.train_documents, args.validation_documents) <= 0:
        parser.error("document counts must be positive")
    for split, num_shards, maximum, num_documents in (
        ("train", args.train_shards, 1024, args.train_documents),
        ("validation", args.validation_shards, 8, args.validation_documents),
    ):
        if not 1 <= num_shards <= min(maximum, num_documents):
            parser.error(f"invalid {split} shard count: {num_shards}")
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
    seen_texts: set[bytes] = set()
    splits = {
        split: download_sample(
            split,
            num_documents=count,
            num_source_shards=num_shards,
            seed=args.seed,
            data_dir=args.data_dir,
            tokenizer=tokenizer,
            seen_texts=seen_texts,
        )
        for split, count, num_shards in (
            ("train", args.train_documents, args.train_shards),
            ("validation", args.validation_documents, args.validation_shards),
        )
    }
    manifest = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_license": "odc-by",
        "token_count_note": "Content tokens only; excludes added BOS/EOS tokens.",
        "purpose": "Bounded continued-pretraining corpus; not a full C4 evaluation.",
        "splits": splits,
    }
    (args.data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print("Preparation complete. No model execution or training was started.")


if __name__ == "__main__":
    main()
