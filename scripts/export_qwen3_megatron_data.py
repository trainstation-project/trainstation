# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export the existing TorchTitan batches for a matched Megatron workload."""

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_rank(args: tuple[int, int, str]) -> dict:
    import torch

    from torchtitan.components.data import (
        ConcatThenSplitPackingConfig,
        GrainDataLoader,
        IndexedJsonlSource,
        SingleDatasetConfig,
    )
    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan.hf_datasets.text_datasets import TextProcessor

    rank, steps, destination = args
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    tokenizer = HuggingFaceTokenizer(
        tokenizer_path=str(root / "assets/hf/Qwen3-30B-A3B")
    )
    # Match the existing recipe without constructing its GPU-specific converters.
    config = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(
            dataset=SingleDatasetConfig(
                source=IndexedJsonlSource.Config(
                    patterns=(str(root / "data/qwen3_30b_c4_1k/train.jsonl"),),
                ),
                processor=TextProcessor.Config(),
            ),
        ),
        seed=42,
    )
    loader = config.build(
        dp_world_size=8,
        dp_rank=rank,
        tokenizer=tokenizer,
        max_context_length=8192,
        num_tokens_per_batch=8192,
    )
    output = Path(destination) / f"rank{rank}"
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: np.lib.format.open_memmap(
            output / f"{name}.npy", mode="w+", dtype=np.int32, shape=(steps, 8192)
        )
        for name in ("input", "labels", "positions")
    }
    max_documents = 0
    max_document_length = 0
    iterator = iter(loader)
    try:
        for step in range(steps):
            batch = next(iterator)
            if batch["num_valid_tokens"] != 8192 or batch["padding_mask"].any():
                raise ValueError(f"Unexpected padding in rank {rank}, step {step}")
            for name, array in arrays.items():
                array[step] = batch[name].numpy()
            positions = arrays["positions"][step]
            if positions[0] != 0:
                raise ValueError("Packed row must start at position zero")
            max_documents = max(max_documents, int((positions == 0).sum()))
            max_document_length = max(max_document_length, int(positions.max()) + 1)
    finally:
        loader.close()
    for array in arrays.values():
        array.flush()
    return {
        "rank": rank,
        "steps": steps,
        "max_documents": max_documents,
        "max_document_length": max_document_length,
        "sha256": {name: sha256(output / f"{name}.npy") for name in arrays},
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/qwen3_30b_c4_megatron",
    )
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error("steps must be between 1 and the audited 1000-step horizon")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        parser.error(f"Export already exists: {manifest_path}")
    source = root / "data/qwen3_30b_c4_1k/train.jsonl"
    source_manifest = json.loads((source.parent / "manifest.json").read_text())
    if sha256(source) != source_manifest["splits"]["train"]["sha256"]:
        raise ValueError("C4 source does not match its audited manifest")
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        ranks = list(
            pool.map(
                export_rank, [(rank, args.steps, str(args.output)) for rank in range(8)]
            )
        )
    manifest = {
        "steps": args.steps,
        "dp_size": 8,
        "sequence_length": 8192,
        "tokens_per_step": 65536,
        "total_tokens": args.steps * 65536,
        "seed": 42,
        "source_sha256": source_manifest["splits"]["train"]["sha256"],
        "packing": "TorchTitan Grain ConcatThenSplitPackingConfig",
        "attention": "causal within each document; positions reset at boundaries",
        "max_documents": max(rank["max_documents"] for rank in ranks),
        "max_document_length": max(rank["max_document_length"] for rank in ranks),
        "ranks": ranks,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
