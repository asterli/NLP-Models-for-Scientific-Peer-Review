# Tests for retrieval/f11_retriever.py (F11 callable retriever). Run with:  pytest tests/test_f11_retriever.py
# Uses a small fake corpus written to a temp folder.

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "retrieval"))
from f11_retriever import PriorWorkRetriever  # noqa: E402

SCHEMA = ("paper_id", "title", "score", "date", "url", "evidence")  # docs/data-schema.md, section 3
CLAIM = "Housing First improved housing stability"
ABS = "Housing First improved housing stability for homeless adults. Participants were followed for two years."

CORPUS = [
    {"paper_id": "doi:10.1/a", "title": "Housing First housing stability trial", "date": "2015-06-01", "year": 2015,
     "abstract": ABS, "url": "https://doi.org/10.1/a", "doi": "10.1/a"},
    {"paper_id": "doi:10.1/titleonly", "title": "Housing First and housing stability", "date": "2014-02-01",
     "year": 2014, "abstract": None, "url": "https://doi.org/10.1/titleonly", "doi": "10.1/titleonly"},
    {"paper_id": "doi:10.1/self", "title": "Housing First trial for veterans", "date": "2020-06-15", "year": 2020,
     "abstract": ABS, "url": "https://doi.org/10.1/self", "doi": "10.1/self"},
    {"paper_id": "doi:10.1/new", "title": "Housing First stability follow-up", "date": "2022-03-01", "year": 2022,
     "abstract": ABS, "url": "https://doi.org/10.1/new", "doi": "10.1/new"},
]
SUBMITTED = {"doi": "10.1/self", "title": "Housing First trial for veterans", "date": "2020-06-15"}
SPLIT_ROW = {"newer_paper_id": "doi:10.1/self", "newer_link": "https://doi.org/10.1/self", "newer_pmid": None,
             "newer_pmcid": None, "newer_title": "Housing First trial for veterans", "newer_pub_date": "2020-06-15"}


@pytest.fixture
def retriever(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps(d) + "\n" for d in CORPUS), encoding="utf-8")
    return PriorWorkRetriever(str(path))


def test_returns_schema_fields_best_first(retriever):
    candidates = retriever.retrieve(CLAIM, SUBMITTED, k=10)
    assert candidates, "expected at least one candidate"
    for c in candidates:
        assert all(field in c for field in SCHEMA)
        assert c["url"] and c["date"]
    assert [c["rank"] for c in candidates] == list(range(1, len(candidates) + 1))
    assert candidates == sorted(candidates, key=lambda c: -c["score"])


def test_drops_the_submitted_paper_and_newer_work(retriever):
    ids = {c["paper_id"] for c in retriever.retrieve(CLAIM, SUBMITTED, k=10)}
    assert ids == {"doi:10.1/a", "doi:10.1/titleonly"}


def test_evidence_is_a_real_abstract_sentence_or_none(retriever):
    by_id = {c["paper_id"]: c for c in retriever.retrieve(CLAIM, SUBMITTED, k=10)}
    assert by_id["doi:10.1/a"]["evidence"] in ABS
    assert by_id["doi:10.1/a"]["evidence_type"] == "abstract_sentence"
    # Title-only records never get a passage, and the title is not used as one.
    assert by_id["doi:10.1/titleonly"]["evidence"] is None
    assert by_id["doi:10.1/titleonly"]["evidence_type"] == "title_only"


def test_split_row_and_plain_dict_give_the_same_results(retriever):
    assert retriever.retrieve(CLAIM, SPLIT_ROW, k=10) == retriever.retrieve(CLAIM, SUBMITTED, k=10)


def test_k_limits_the_number_of_candidates(retriever):
    assert len(retriever.retrieve(CLAIM, SUBMITTED, k=1)) == 1


def test_no_match_returns_empty_list(retriever):
    assert retriever.retrieve("foot conditions survey", SUBMITTED, k=10) == []


def test_explain_also_returns_exclusion_counts(retriever):
    candidates, excluded = retriever.retrieve(CLAIM, SUBMITTED, k=10, explain=True)
    assert len(candidates) == 2
    assert excluded == {"submitted_paper": 1, "after_cutoff": 1}


@pytest.mark.parametrize("paper", [None, {}, {"doi": "10.1/self"}])
def test_missing_submitted_date_is_an_error(retriever, paper):
    # Without a date every result would be silently dropped, so fail loudly instead.
    with pytest.raises(ValueError, match="needs a 'date'"):
        retriever.retrieve(CLAIM, paper, k=10)