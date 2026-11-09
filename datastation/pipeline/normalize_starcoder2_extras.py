"""Download one StarCoder2-Extras config and save it as a normalized Parquet dataset.

Source is the Hugging Face dataset ``bigcode/starcoder2data-extras`` at revision
``1ba0d4f``, the same pin as ``starcoder2_extras_normalize_steps``. Each run is
one dataset config. Marin keeps six: documentation, ir_cpp, ir_python, ir_rust,
ir_low_resource, and kaggle. Shards are ``<subset>/*.parquet``. Text lives in
``content``; ``id`` is kept as ``source_id`` when the config has that column.

Each row has ``id`` (xxh3-128 hex of ``text``), ``text``, ``source_id``, plus
the config's other columns, written to <output-dir>/outputs/main/part-*.parquet.

    uv run python pipeline/normalize_starcoder2_extras.py --subset ir_python --output-dir <output_dir> --tasks 2 --limit 1000

Note: eventually we will want to write this straight to object storage, but for now, just save to local disk
"""

import argparse
import re

import pyarrow as pa

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter


# Pinned to the same revision as marin's starcoder2_extras download.
HF_DATASET = "hf://datasets/bigcode/starcoder2data-extras@1ba0d4f"
SUBSETS = ("documentation", "ir_cpp", "ir_python", "ir_rust", "ir_low_resource", "kaggle")
MAX_WHITESPACE_RUN = 128

# Columns other than `content` (renamed to `text`) and `id` (renamed to `source_id`).
# Schemas differ per config, so each run is one dataset config.
_EXTRA_FIELDS: dict[str, list[pa.Field]] = {
    "documentation": [
        pa.field("project", pa.string()),
        pa.field("source", pa.string()),
        pa.field("language", pa.string()),
    ],
    "ir_cpp": [pa.field("__index_level_0__", pa.string())],
    "ir_python": [],
    "ir_rust": [pa.field("__index_level_0__", pa.string())],
    "ir_low_resource": [
        pa.field("__index_level_0__", pa.string()),
        pa.field("size", pa.int64()),
    ],
    "kaggle": [pa.field("file_id", pa.string())],
}


def schema_for(subset: str) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("source_id", pa.string()),
            *_EXTRA_FIELDS[subset],
        ]
    )


def _as_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def read_row(self, data: dict, path: str, id_in_file: int | str) -> dict:
    """Take text from `content`. Keep `id` as `source_id` only when the config has it."""
    text = _as_text(data.pop("content", ""))
    source_id = data.pop("id", None)
    if source_id is not None:
        data["source_id"] = source_id
    return {
        "text": text,
        "id": source_id if source_id is not None else f"{path}/{id_in_file}",
        "metadata": data,
    }


def make_to_row(schema: pa.Schema):
    names = schema.names

    def to_row(self, doc) -> dict:
        row = {"id": doc.id, "text": doc.text, **doc.metadata}
        return {name: row.get(name) for name in names}

    return to_row


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
            doc.text = long_whitespace.sub(lambda m: m.group(0)[:MAX_WHITESPACE_RUN], doc.text)
            doc.id = xxhash.xxh3_128_hexdigest(doc.text.encode("utf-8"))
            yield doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", required=True, choices=SUBSETS, help="Dataset config to normalize.")
    parser.add_argument("--output-dir", required=True, help="Local path (or any fsspec URL, e.g. s3://...).")
    parser.add_argument("--tasks", type=int, default=64, help="Number of tasks, one output file each.")
    parser.add_argument("--limit", type=int, default=-1, help="Max documents per task, for test runs.")
    args = parser.parse_args()

    schema = schema_for(args.subset)
    LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                HF_DATASET,
                glob_pattern=f"{args.subset}/*.parquet",
                limit=args.limit,
                adapter=read_row,
                text_key="content",
                id_key="id",
            ),
            Normalize(),
            ParquetWriter(
                f"{args.output_dir}/outputs/main",
                output_filename=f"part-${{rank}}-of-{args.tasks:05d}.parquet",
                adapter=make_to_row(schema),
                schema=schema,
                max_file_size=-1,  # one file per task, part-NNNNN-of-MMMMM names
            ),
        ],
        tasks=args.tasks,
        logging_dir=f"{args.output_dir}/logs",
    ).run()


if __name__ == "__main__":
    main()
