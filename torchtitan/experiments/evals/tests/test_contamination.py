# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("lm_eval")

from torchtitan.experiments.evals.contamination import (
    batch_documents,
    eval_doc_text,
    EvalIndex,
)
from torchtitan.experiments.evals.evaluate import (
    clean_samples,
    load_contamination_report,
)

QUESTION = (
    "Blade Runner 2049 is a 2017 American neo-noir science fiction film "
    "directed by Denis Villeneuve and written by Hampton Fancher."
)


def test_index_matches_normalized_text():
    index = EvalIndex(ngram_size=13)
    index.add("task_a", 7, QUESTION)
    # Case and punctuation differ, and the question sits inside other text.
    document = (
        "Poker night. BLADE RUNNER 2049 is a 2017 american neonoir science "
        "fiction film directed by denis villeneuve, and more."
    )
    assert index.match(document) == {("task_a", 7)}
    assert (
        index.match("An unrelated document about bread and how to store it.") == set()
    )


def test_short_questions_are_counted_but_not_checked():
    index = EvalIndex(ngram_size=13)
    index.add("task_a", 0, QUESTION)
    index.add("task_a", 1, "Is New York in New England?")
    assert index.num_docs == {"task_a": 2}
    assert index.num_checked == {"task_a": 1}
    assert index.match("Is New York in New England? " * 5) == set()


class _WordTokenizer:
    """Token id i decodes to the word w<i>."""

    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids)


def test_batch_documents_splits_at_document_starts_and_drops_padding():
    batch = {
        "input": torch.tensor([1, 2, 3, 4, 5, 0, 0]),
        "positions": torch.tensor([0, 1, 2, 0, 1, 0, 1]),
        "padding_mask": torch.tensor([False, False, False, False, False, True, True]),
    }
    assert batch_documents(batch, _WordTokenizer()) == ["w1 w2 w3", "w4 w5"]


def test_documents_do_not_match_across_boundaries():
    index = EvalIndex(ngram_size=4)
    index.add("task_a", 0, "w3 w4 w5 w6")
    batch = {
        "input": torch.tensor([1, 2, 3, 4, 5, 6]),
        "positions": torch.tensor([0, 1, 2, 3, 0, 1]),
    }
    documents = batch_documents(batch, _WordTokenizer())
    assert documents == ["w1 w2 w3 w4", "w5 w6"]
    assert all(not index.match(document) for document in documents)


def test_eval_doc_text_renders_plain_text_templates():
    # lm-eval's own doc_to_decontamination_query runs ast.literal_eval on this
    # and raises on the en dash.
    task = SimpleNamespace(
        config=SimpleNamespace(
            should_decontaminate=True, doc_to_decontamination_query="{{page}}"
        ),
        features=["page"],
    )
    assert eval_doc_text(task, {"page": "= 2000 – 2005 ="}) == "= 2000 – 2005 ="


REPORT = {
    "units": {"boolq": ["boolq"], "trainstation_mmlu": ["mmlu_a", "mmlu_b"]},
    "tasks": {
        "boolq": {"num_docs": 5, "contaminated": [1, 3]},
        "mmlu_a": {"num_docs": 3, "contaminated": []},
        "mmlu_b": {"num_docs": 2, "contaminated": [0]},
    },
}


def test_clean_samples():
    assert clean_samples(REPORT, "boolq") == {"boolq": [0, 2, 4]}
    assert clean_samples(REPORT, "trainstation_mmlu") == {
        "mmlu_a": [0, 1, 2],
        "mmlu_b": [1],
    }


def test_load_contamination_report_checks_coverage(tmp_path):
    path = tmp_path / "contamination.json"
    with pytest.raises(ValueError, match="No contamination report"):
        load_contamination_report(str(path), ["boolq"])
    path.write_text(json.dumps(REPORT))
    assert load_contamination_report(str(path), ["boolq"]) == REPORT
    with pytest.raises(ValueError, match="does not cover"):
        load_contamination_report(str(path), ["boolq", "hellaswag"])
