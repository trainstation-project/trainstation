# Tokenizer Experiments

This folder contains some scratch code for tokenizer experiments.
The main experiment we run is on training 100M parameter models on [the Dolma 10B mix](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-10B-1025) and measuring BPB loss on Paloma and Math datasets.

Steps:
1. Download the data from [here](https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-10B-1025)
2. Download the eval sets : [Paloma](https://huggingface.co/datasets/allenai/paloma) and the `open-web-math/validation/val.jsonl.zst` from [Proof-Pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2)
3. Decontaminate the data using the `decon.sh` script from the Paloma and Math datasets
4. Create a folder per tokenizer on `/mnt/shared/data`
5. Tokenize the training data per tokenizer using the `~/trainstation/scripts/data/megatron_preprocess.py` tokenizer script and store it in `/mnt/shared/data/<tokenizer>/train`
6. Tokenize the evaluation data per tokenizer using the same script, place the data in `/mnt/shared/data/<tokenizer>/test`

Prepare step 3 from the repository root after the downloads finish:

```bash
cargo build --release --locked --manifest-path "$HOME/decon/Cargo.toml"
python3 -m pip install zstandard
bash rough/tokenizer_experiments/decon.sh
```

The script reads the downloaded manifests in `data/paloma` and `data/proof-pile-2`,
creates references of exactly 1,000 whitespace-separated words. Blocks advance
by 500 words; a final full block is anchored at the end of each document to cover
the tail without producing any reference shorter than 1,000 words. Documents below
that minimum are skipped and counted. Blank lines do not count as words. Blocks never
span documents. Each block becomes a decon `question`, with its source file, JSONL
record index, chunk index, and zero-based word range (end exclusive) retained.
Both Paloma splits and the OpenWebMath validation set are included. Original
downloads stay unchanged.

By default, the script only reports matches in `data/dolmino-decon-report/reports`.
It uses a score threshold of 0.95; for these text-only references, the score is the
IDF-weighted n-gram overlap with the reference chunk. These are conservative starting
settings, not calibrated cutoffs. Review match counts and examples before removing
data: one matching chunk causes `--purify` to remove the entire training document.
This 1,000-word minimum excludes shorter eval documents entirely, so their
contamination is not checked.

To write cleaned `.jsonl.gz` shards after reviewing the reports:

```bash
OUTPUT_DIR=data/dolmino-decontaminated bash rough/tokenizer_experiments/decon.sh --purify
```

Cleaned shards go to `data/dolmino-decontaminated/cleaned`, preserving source
directories. Use `cleaned` as the training input for step 5, only after a successful
exit. Every run also saves references and per-line coverage receipts. Decon may skip
short or malformed training records; coverage distinguishes skipped records from
examined ones.

`DATA_DIR`, `OUTPUT_DIR`, and `DECON_BIN` override the default paths, and `THRESHOLD`
overrides the score threshold. The output directory must be new. Extra arguments
are passed to `decon detect`, for example `--worker-threads 16` or `--purify`.

Run the small CPU integration check with
`python3 rough/tokenizer_experiments/test_decon.py`. It uses temporary fixtures to
check 999/1,000/1,001-word boundaries, mixed whitespace, overlapping windows, report-only
behavior, and preservation of clean records and original files during purification.

For steps 4-6, `tokenize.py` uses the exports in
`/home/gpu/tokenizers/tokenizers`: `deepseek-v4-d1`, `glm-5-d1`, and `marin-d1`.
It calls `scripts/data/megatron_preprocess.py` with `--append-eod`. These exports
add no BOS token; one EOS token is appended to each nonempty document.

The local CPU preprocessing environment is `data/tokenization-venv`. Run eval
tokenization independently, or tokenize training data after creating the cleaned copy:

```bash
data/tokenization-venv/bin/python rough/tokenizer_experiments/tokenize.py eval
data/tokenization-venv/bin/python rough/tokenizer_experiments/tokenize.py train
```

Use `all` to run both. The training input defaults to
`data/dolmino-decontaminated/cleaned`; a missing cleaned copy causes an error.
Use `--train-input PATH` to select another input explicitly. The report-only
decontamination scan does not create a cleaned copy.

Outputs go to `/mnt/shared/data/<tokenizer>/train` and
`/mnt/shared/data/<tokenizer>/test`. Training source directories are preserved.
Paloma remains under `test/paloma/<source>/<val-or-test>`, and OpenWebMath validation
goes under `test/open-web-math/validation`. All eval documents are tokenized,
including those below the decontamination reference minimum. The verified empty
Paloma shard is omitted because it contains no documents.

Each shard produces `<shard>_text.bin` and `<shard>_text.idx`. Each completed split
also has `_SUCCESS` and `tokenize.log`. A failed job returns an error and leaves
no success marker. Existing output split directories are rejected to prevent
overwriting a previous or partial run; select a fresh `--output-root` for a rerun.

The script uses all CPUs available to the process by default, with one thread per
worker. Tokenizers run sequentially. To select a tokenizer or limit parallelism:

```bash
data/tokenization-venv/bin/python rough/tokenizer_experiments/tokenize.py eval \
    --tokenizers marin-d1 --workers 32
```

`--tokenizer-root`, `--data-root`, and `--output-root` override the default paths.
The tokenization environment is CPU-only; use the training environment for GPU jobs.
