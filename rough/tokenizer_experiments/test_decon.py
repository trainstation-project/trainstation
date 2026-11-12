# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU integration check; requires zstandard and a built ~/decon binary."""

import gzip
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import zstandard


class TestDeconScript(unittest.TestCase):
    def test_word_blocks_and_purification(self):
        binary = Path(
            os.environ.get("DECON_BIN", str(Path.home() / "decon/target/release/decon"))
        )
        if not binary.is_file():
            self.skipTest("Build ~/decon or set DECON_BIN before running this check")

        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            paragraph = "\t  ".join(f"observatory{index}" for index in range(1000))
            long_document = " ".join(f"botanical{index}" for index in range(2500))
            short_document = "\n\n".join(f"culinary{index}" for index in range(999))
            math_words = [f"algebraic{index}" for index in range(1001)]

            def write_records(path, records):
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = "".join(json.dumps(record) + "\n" for record in records).encode()
                if path.suffix == ".gz":
                    payload = gzip.compress(payload)
                elif path.suffix == ".zst":
                    payload = zstandard.ZstdCompressor().compress(payload)
                path.write_bytes(payload)

            for dataset, files in [
                (
                    "paloma",
                    {
                        "source/val/part.jsonl.gz": [
                            {"text": paragraph},
                            {"text": long_document},
                            {"text": short_document},
                        ],
                        "source/test/empty.jsonl.gz": [],
                    },
                ),
                (
                    "proof-pile-2",
                    {
                        "open-web-math/validation/val.jsonl.zst": [
                            {"text": "\r\n".join(math_words)}
                        ]
                    },
                ),
            ]:
                for name, records in files.items():
                    write_records(data / dataset / name, records)
                (data / dataset / "download-manifest.json").write_text(
                    json.dumps({"files": [{"path": name} for name in files]})
                )

            training = data / "dolma3_dolmino_mix-10B-1025/data"
            clean = {"id": "clean", "text": short_document}
            write_records(
                training / "a/part.jsonl.zst",
                [{"id": "paloma-match", "text": paragraph}, clean],
            )
            write_records(
                training / "b/part.jsonl",
                [{"id": "math-match", "text": "\r\n".join(math_words[1:])}, clean],
            )
            original_hashes = {
                path: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in data.rglob("*")
                if path.is_file()
            }

            def run(output, *arguments):
                return subprocess.run(
                    [
                        "bash",
                        str(Path(__file__).with_name("decon.sh")),
                        "--worker-threads",
                        "2",
                        *arguments,
                    ],
                    env={
                        **os.environ,
                        "DATA_DIR": str(data),
                        "OUTPUT_DIR": str(output),
                        "DECON_BIN": str(binary),
                        "THRESHOLD": "0.95",
                    },
                    capture_output=True,
                    text=True,
                )

            report_output = data / "report-only"
            result = run(report_output)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse((report_output / "cleaned").exists())
            references = [
                json.loads(line)
                for path in (report_output / "references").rglob("*.jsonl")
                for line in path.read_text().splitlines()
            ]
            self.assertEqual(len(references), 7)
            self.assertEqual(len({row["doc_id"] for row in references}), 7)
            self.assertTrue(
                all(len(row["question"].split()) == 1000 for row in references)
            )
            self.assertEqual(
                sorted(
                    row["word_start"]
                    for row in references
                    if row["eval_key"] == "open-web-math"
                ),
                [0, 1],
            )
            self.assertEqual(
                sorted(
                    row["word_start"]
                    for row in references
                    if row["eval_key"] == "paloma" and row["source_line"] == 1
                ),
                [0, 500, 1000, 1500],
            )
            self.assertTrue(
                any(
                    row["question"] == " ".join(paragraph.split())
                    for row in references
                )
            )
            self.assertTrue(
                all(row["word_end"] - row["word_start"] == 1000 for row in references)
            )
            self.assertTrue(
                all(
                    row["question"] != " ".join(short_document.split())
                    for row in references
                )
            )

            self.assertNotEqual(run(report_output).returncode, 0)
            purified_output = data / "purified"
            result = run(purified_output, "--purify")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for source in ["a", "b"]:
                with gzip.open(
                    purified_output / "cleaned" / source / "part.jsonl.gz", "rt"
                ) as stream:
                    self.assertEqual([json.loads(line) for line in stream], [clean])
            self.assertEqual(
                len(list((purified_output / "coverage").rglob("*.gz"))), 2
            )
            for path, digest in original_hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

            for dataset in ["paloma", "proof-pile-2"]:
                manifest = json.loads(
                    (data / dataset / "download-manifest.json").read_text()
                )
                for entry in manifest["files"]:
                    write_records(
                        data / dataset / entry["path"], [{"text": short_document}]
                    )
            empty_output = data / "no-eligible-references"
            result = run(empty_output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No evaluation documents contain", result.stderr)
            self.assertFalse((empty_output / "reports").exists())


if __name__ == "__main__":
    unittest.main()
