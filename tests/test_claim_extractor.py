# No GPU, no model: checks the callable claim extractor (src/claims/extractor.py) with a stand-in generator.
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from claims.extractor import ClaimExtractor  # noqa: E402

PAPER = {"paper_id": "doi:10.1/x", "title": "T",
         "sentences": [{"sentence_id": "s001", "section": "abstract", "text": "Prior work shows X."},
                       {"sentence_id": "s002", "section": "abstract:results", "text": "We find Y."},
                       {"sentence_id": "s003", "section": "introduction", "text": "We aimed to test Z."}]}


def fake(raw):
    calls = []
    def generate(msgs, max_new_tokens):
        calls.append((msgs, max_new_tokens))
        return raw
    return ClaimExtractor("unused", generate=generate), calls


def test_returns_claim_records_with_original_text_and_ids():
    extractor, calls = fake("s001: BACKGROUND\ns002: CLAIM\ns003: CLAIM")
    claims = extractor.extract(PAPER)
    assert claims == [
        {"claim_id": "doi:10.1/x#s002", "paper_id": "doi:10.1/x", "sentence_id": "s002", "text": "We find Y."},
        {"claim_id": "doi:10.1/x#s003", "paper_id": "doi:10.1/x", "sentence_id": "s003", "text": "We aimed to test Z."}]
    assert extractor.last_unlabeled == []
    msgs, budget = calls[0]
    assert "s002 [abstract:results]: We find Y." in msgs[1]["content"] and budget == 700


def test_no_claims_and_no_sentences_return_empty_list():
    extractor, calls = fake("s001: BACKGROUND\ns002: BACKGROUND\ns003: BACKGROUND")
    assert extractor.extract(PAPER) == []
    assert extractor.extract({"paper_id": "doi:10.1/y", "title": "T", "sentences": []}) == []
    assert len(calls) == 1  # the model is not called for a paper with no sentences


def test_off_format_labels_are_not_claims_and_are_reported():
    extractor, _ = fake("s001: BACKGROUND\ns002: RESULTS\ns003: OBJECTIVE\ns009: CLAIM")
    assert extractor.extract(PAPER) == []          # invented labels and unknown ids never become claims
    assert extractor.last_unlabeled == ["s002", "s003"]


def test_long_papers_get_a_larger_generation_budget():
    extractor, calls = fake("")
    long_paper = {"paper_id": "doi:10.1/z", "title": "T", "sentences": [
        {"sentence_id": f"s{i:03d}", "section": "introduction", "text": "x"} for i in range(1, 121)]}
    assert extractor.extract(long_paper) == []
    assert calls[0][1] == 1220 and len(extractor.last_unlabeled) == 120


@pytest.mark.parametrize("bad", [
    {"title": "T", "sentences": []},
    {"paper_id": "doi:10.1/x", "title": "T", "sentences": [{"sentence_id": "s001", "text": "no section"}]},
    {"paper_id": "doi:10.1/x", "title": "T", "sentences": [
        {"sentence_id": "s001", "section": "abstract", "text": "a"}, {"sentence_id": "s001", "section": "abstract", "text": "b"}]},
])
def test_malformed_papers_raise(bad):
    extractor, _ = fake("s001: CLAIM")
    with pytest.raises(ValueError):
        extractor.extract(bad)
