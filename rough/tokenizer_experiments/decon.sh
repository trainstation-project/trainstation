#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_dir="${DATA_DIR:-$repo_dir/data}"
output_dir="${OUTPUT_DIR:-$data_dir/dolmino-decon-report}"
decon_bin="${DECON_BIN:-$HOME/decon/target/release/decon}"

# Build ~/decon with `cargo build --release --locked` before running this script.
test -x "$decon_bin"

python3 - "$data_dir" "$output_dir" <<'PY'
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import zstandard

data_dir, output_dir = map(Path, sys.argv[1:])
# A new directory prevents reruns from mixing old and new references or results.
output_dir.mkdir(parents=True)
doc_id = 0
indices = Counter()
words_per_chunk = 1000
stride = 500
skipped_short = 0

for eval_key, dataset in [("paloma", "paloma"), ("open-web-math", "proof-pile-2")]:
    dataset_dir = data_dir / dataset
    manifest = json.loads((dataset_dir / "download-manifest.json").read_text())
    # Use the manifest so a missing shard fails instead of silently being omitted.
    for entry in sorted(manifest["files"], key=lambda entry: entry["path"]):
        relative_path = Path(entry["path"])
        split = relative_path.parts[1]
        source_path = dataset_dir / relative_path
        reference_path = output_dir / "references" / eval_key / relative_path.with_suffix("")
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if source_path.suffix == ".gz" else zstandard.open
        with opener(source_path, "rt", encoding="utf-8") as source:
            with reference_path.open("w", encoding="utf-8") as destination:
                for line_index, line in enumerate(source):
                    text = json.loads(line)["text"]
                    if not isinstance(text, str):
                        raise ValueError(f"Non-string text in {source_path}:{line_index + 1}")
                    words = text.split()
                    if len(words) < words_per_chunk:
                        skipped_short += 1
                        continue
                    last_start = len(words) - words_per_chunk
                    starts = list(range(0, last_start + 1, stride))
                    # Cover the tail with a full block, never a short reference.
                    if starts[-1] != last_start:
                        starts.append(last_start)
                    for chunk_index, start in enumerate(starts):
                        chunk = " ".join(words[start : start + words_per_chunk])
                        # Decon calls the reference text 'question'.
                        record = {
                            "question": chunk,
                            "doc_id": doc_id,
                            "eval_key": eval_key,
                            "split": split,
                            "eval_instance_index": indices[eval_key, split],
                            "fingerprint": hashlib.sha256(
                                chunk.encode("utf-8")
                            ).hexdigest()[:16],
                            "source_file": str(relative_path),
                            "source_line": line_index,
                            "chunk_index": chunk_index,
                            "word_start": start,
                            "word_end": start + words_per_chunk,
                        }
                        destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                        doc_id += 1
                        indices[eval_key, split] += 1
    print(f"Prepared {eval_key} references", flush=True)

print(f"Prepared {doc_id} total reference chunks", flush=True)
print(f"Skipped {skipped_short} documents shorter than {words_per_chunk} words", flush=True)
if doc_id == 0:
    raise ValueError("No evaluation documents contain a full 1000-word reference block")
PY

"$decon_bin" detect \
    --training-dir "$data_dir/dolma3_dolmino_mix-10B-1025/data" \
    --evals-dir "$output_dir/references" \
    --content-key text \
    --contamination-score-threshold "${THRESHOLD:-0.95}" \
    --cleaned-output-dir "$output_dir/cleaned" \
    --report-output-dir "$output_dir/reports" \
    --coverage-output-dir "$output_dir/coverage" \
    "$@"

printf 'Contamination reports: %s/reports\n' "$output_dir"
if [[ -d "$output_dir/cleaned" ]]; then
    printf 'Cleaned training data: %s/cleaned\n' "$output_dir"
fi
