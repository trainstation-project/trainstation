"""Download the code (Stack-Edu) slice of Dolma 3 150B and save it as a normalized Parquet file.

Each row has `id` (xxh3-128 hex of `text`), `text`, `source_id` (the original Dolma
id), plus the other Dolma columns, written to <output-dir>/outputs/main/part-*.parquet.

    uv run --extra io --extra processing python pipeline/normalize_dolma.py \
        --output-dir <output_dir> --tasks 2 --limit 1000

Note: eventually we will want to write this straight to object storage, but for now, just save to local disk
"""

import argparse
import re

import pyarrow as pa

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import ParquetWriter


DOLMA3_150B = "hf://datasets/allenai/dolma3_mix-150B-1025/data"
CODE_FILES = "stack_edu-*/*.jsonl.zst"
MAX_WHITESPACE_RUN = 128  

# Explicit schema: `detected_licenses` is often empty, which breaks schema inference.
SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.string()),
        ("source_id", pa.string()),
        ("source", pa.string()),
        ("created", pa.string()),
        ("added", pa.string()),
        (
            "metadata",
            pa.struct(
                [
                    ("language", pa.string()),
                    ("detected_licenses", pa.list_(pa.string())),
                    ("path", pa.string()),
                    ("score", pa.float64()),
                    ("int_score", pa.int64()),
                    ("license_type", pa.string()),
                    ("repo_name", pa.string()),
                    ("length_bytes", pa.int64()),
                    ("src_encoding", pa.string()),
                    ("uri", pa.string()),
                ]
            ),
        ),
    ]
)


def read_dolma(self, data: dict, path: str, id_in_file: int) -> dict:
    """Keep every Dolma column: `text` and `id` as-is, the rest in metadata."""
    return {"text": data.pop("text"), "id": data.pop("id"), "metadata": data}


def to_row(self, doc) -> dict:
    return {"id": doc.id, "text": doc.text, **doc.metadata}


class Normalize(PipelineStep):
    """Drop blank docs, cap whitespace runs, and set `id` to the xxh3-128 hash of the text"""

    name = "Normalize"
    _requires_dependencies = ["xxhash"]

    def run(self, data, rank: int = 0, world_size: int = 1):
        import xxhash

        long_whitespace = re.compile(r"\s{%d,}" % (MAX_WHITESPACE_RUN + 1))
        for doc in data:
            if not doc.text.strip():
                self.stat_update("dropped_blank")
                continue
            doc.metadata["source_id"] = doc.id
            doc.text = long_whitespace.sub(lambda m: m.group(0)[:MAX_WHITESPACE_RUN], doc.text)
            doc.id = xxhash.xxh3_128_hexdigest(doc.text.encode("utf-8"))
            yield doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="Local path (or any fsspec URL, e.g. s3://...).")
    parser.add_argument("--tasks", type=int, default=64, help="Number of tasks, one output file each.")
    parser.add_argument("--limit", type=int, default=-1, help="Max documents per task, for test runs.")
    args = parser.parse_args()

    LocalPipelineExecutor(
        pipeline=[
            JsonlReader(DOLMA3_150B, glob_pattern=CODE_FILES, limit=args.limit, adapter=read_dolma, add_file_path=False),
            Normalize(),
            ParquetWriter(
                f"{args.output_dir}/outputs/main",
                output_filename=f"part-${{rank}}-of-{args.tasks:05d}.parquet",
                adapter=to_row,
                schema=SCHEMA,
                max_file_size=-1,  # one file per task, part-NNNNN-of-MMMMM names
            ),
        ],
        tasks=args.tasks,
        logging_dir=f"{args.output_dir}/logs",
    ).run()


if __name__ == "__main__":
    main()
