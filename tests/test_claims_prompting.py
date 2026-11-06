# No GPU, no internet: checks the shared claim-extractor prompt/data helpers (src/claims/prompting.py).
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claims.prompting import (claim_rows, load_split, messages, parse_all_labels, parse_claim_ids,  # noqa: E402
                              target_text, user_text)

PAPER = {"paper_id": "doi:10.1/x", "title": "T",
         "sentences": [{"sentence_id": "s001", "section": "abstract", "text": "Prior work shows X."},
                       {"sentence_id": "s002", "section": "abstract:results", "text": "We find Y."},
                       {"sentence_id": "s003", "section": "introduction", "text": "Data were collected."}],
         "labels": {"s001": "BACKGROUND", "s002": "CLAIM", "s003": "BACKGROUND"}}


def test_target_text_one_line_per_sentence_in_order():
    assert target_text(PAPER) == "s001: BACKGROUND\ns002: CLAIM\ns003: BACKGROUND"


def test_user_text_numbers_every_sentence_with_section():
    text = user_text(PAPER)
    assert text.startswith("Paper title: T")
    assert "s002 [abstract:results]: We find Y." in text
    assert messages(PAPER)[0]["role"] == "system" and messages(PAPER)[1]["content"] == text


def test_parse_claim_ids_handles_noise_duplicates_and_invalid_ids():
    raw = "Sure!\ns001: BACKGROUND\ns002 - CLAIM\ns002: CLAIM\ns009: CLAIM\ns003: background\n"
    assert parse_claim_ids(raw, {"s001", "s002", "s003"}) == ["s002"]
    assert parse_all_labels(raw, {"s001", "s002", "s003"}) == {"s001": "BACKGROUND", "s002": "CLAIM", "s003": "BACKGROUND"}


def test_claim_rows_match_shared_schema():
    rows = claim_rows(PAPER, ["s002"])
    assert rows == [{"claim_id": "doi:10.1/x#s002", "paper_id": "doi:10.1/x", "sentence_id": "s002", "text": "We find Y."}]


@pytest.mark.parametrize("split,n_papers,n_sent", [("train", 14, 445), ("dev", 3, 80), ("test", 3, 120)])
def test_load_split_matches_frozen_m08_counts(split, n_papers, n_sent):
    root = Path(__file__).resolve().parents[1]
    papers = load_split(split, root / "data/papers/papers.jsonl", root / "data/claims/splits")
    assert len(papers) == n_papers
    assert sum(len(p["sentences"]) for p in papers) == n_sent
    for p in papers:  # every kept sentence is labeled, and the target covers all of them
        assert set(p["labels"]) == {s["sentence_id"] for s in p["sentences"]}
        assert target_text(p).count("\n") + 1 == len(p["sentences"])


def test_splits_do_not_share_papers():
    root = Path(__file__).resolve().parents[1]
    ids = {s: {p["paper_id"] for p in load_split(s, root / "data/papers/papers.jsonl", root / "data/claims/splits")}
           for s in ("train", "dev", "test")}
    assert not (ids["train"] & ids["dev"]) and not (ids["train"] & ids["test"]) and not (ids["dev"] & ids["test"])
