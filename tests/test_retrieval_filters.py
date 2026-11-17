# Tests for retrieval/m23_filters.py (F10). Run with:  pytest tests/test_retrieval_filters.py
# Uses a small fake corpus written to a temp folder, plus one check on the real corpus (skipped if it's missing).

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "retrieval"))
from m23_filters import FilteredRetriever, date_range, date_status, submitted_from_row  # noqa: E402

ABS = "Housing first improved housing stability. Participants had fewer emergency visits."


def doc(pid, title="Housing first housing stability trial", date="2015-06-01", abstract=ABS, **kw):
    d = {"paper_id": pid, "title": title, "date": date, "year": int(date[:4]) if date else None,
         "abstract": abstract, "url": f"https://example.org/{pid}", "doi": None, "pmid": None,
         "openalex_id": None, "type": "article", "venue": "J Test", "cited_by": []}
    if pid.startswith("doi:"): d["doi"] = pid[4:]
    d.update(kw)
    return d


SUBMITTED = {"paper_id": "doi:10.1000/sub", "doi": "10.1000/sub", "pmid": "999", "pmcid": "PMC9",
             "title": "Housing First for veterans: a randomized trial", "date": "2020-06-15"}

CORPUS = [
    # the submitted paper and other versions of it
    doc("doi:10.1000/sub", title="Housing First for veterans: a randomized trial", date="2020-06-15"),
    doc("doi:10.1101/preprint", title="Housing first for Veterans - a randomized trial", date="2019-11-01", type="preprint"),
    doc("openalex:W77", title="Something else entirely about housing first", date="2018-03-02", pmid="999"),
    # dates relative to the cutoff (2020-06-15)
    doc("doi:10.1000/before", date="2020-06-14"),
    doc("doi:10.1000/sameday", date="2020-06-15"),
    doc("doi:10.1000/after", date="2021-02-01"),
    doc("doi:10.1000/yr_prev", date="2019-01-01"),        # year-only, previous year -> eligible
    doc("doi:10.1000/yr_same", date="2020-01-01"),        # year-only, same year -> uncertain
    doc("doi:10.1000/yr_next", date="2021-01-01"),        # year-only, later year -> excluded
    doc("doi:10.1000/nodate", date=None),                 # missing -> uncertain
    # duplicates
    doc("doi:10.1000/dupdoi", title="Housing first housing stability cohort", date="2012-05-05"),
    doc("openalex:W10", title="Housing first stability cohort copy", date="2012-05-05", doi="10.1000/DUPDOI"),
    doc("doi:10.1000/pm1", title="Housing first mental health outcomes", date="2011-05-05", pmid="555"),
    doc("doi:10.1000/pm2", title="Housing first mental health outcomes (reprint)", date="2011-07-07", pmid="555"),
    doc("doi:10.1000/t1", title="Housing First and Health", date="2010-03-01"),
    doc("openalex:W11", title="Housing first and health.", date="2010-09-09"),           # same title, same year
    doc("doi:10.1000/t3", title="Housing First and Health", date="2008-04-04"),             # same title, other year
    # title-only and non-scholarly records
    doc("doi:10.1000/titleonly", title="Housing first housing stability title only", date="2014-02-02", abstract=None),
    doc("doi:10.1000/paratext", title="Housing first issue information", date="2014-02-02", type="paratext"),
    doc("doi:10.1000/notitle", title=None, date="2014-02-02"),
]

Q = "housing first housing stability emergency visits health trial cohort outcomes"


@pytest.fixture
def corpus_path(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(d) for d in CORPUS))
    return str(p)


def run(corpus_path, submitted=SUBMITTED, uncertain="exclude"):
    fr = FilteredRetriever(corpus_path, uncertain=uncertain)
    hits, excluded = fr.search(Q, submitted, k=50, explain=True)
    return {h["paper_id"]: h for h in hits}, excluded


# ---------- self matches ----------
def test_submitted_paper_excluded_by_doi_pmid_and_title(corpus_path):
    hits, ex = run(corpus_path)
    assert "doi:10.1000/sub" not in hits            # same DOI
    assert "doi:10.1101/preprint" not in hits    # other version: same title, different DOI
    assert "openalex:W77" not in hits            # same PMID
    assert ex["submitted_paper"] == 3


def test_submitted_doi_given_as_url(corpus_path):
    hits, ex = run(corpus_path, {"doi": "https://doi.org/10.1000/SUB", "date": "2020-06-15"})
    assert "doi:10.1000/sub" not in hits
    assert ex["submitted_paper"] == 1            # matched as the submitted paper, not just dropped by date


# ---------- dates ----------
def test_full_dates_before_same_day_after(corpus_path):
    hits, ex = run(corpus_path)
    assert hits["doi:10.1000/before"]["eligibility"] == "eligible"
    assert "doi:10.1000/sameday" not in hits and ex["same_date"] == 1
    assert "doi:10.1000/after" not in hits


def test_year_only_dates(corpus_path):
    hits, _ = run(corpus_path)
    assert hits["doi:10.1000/yr_prev"]["eligibility"] == "eligible"
    assert "date_year_only" in hits["doi:10.1000/yr_prev"]["flags"]
    assert "doi:10.1000/yr_same" not in hits        # uncertain -> dropped by default
    assert "doi:10.1000/yr_next" not in hits


def test_missing_date_not_treated_as_eligible(corpus_path):
    hits, ex = run(corpus_path)
    assert "doi:10.1000/nodate" not in hits
    assert ex["date_uncertain"] == 2             # yr_same + nodate


def test_uncertain_flag_policy_keeps_them_marked(corpus_path):
    hits, _ = run(corpus_path, uncertain="flag")
    for pid, flag in (("doi:10.1000/yr_same", "date_year_only"), ("doi:10.1000/nodate", "date_missing")):
        assert hits[pid]["eligibility"] == "uncertain"
        assert "date_uncertain" in hits[pid]["flags"] and flag in hits[pid]["flags"]
    assert "doi:10.1000/after" not in hits and "doi:10.1000/yr_next" not in hits   # still excluded


def test_no_submitted_date_means_nothing_is_certain(corpus_path):
    hits, _ = run(corpus_path, {"doi": "10.1000/sub"})
    assert hits == {}
    hits, _ = run(corpus_path, {"doi": "10.1000/sub"}, uncertain="flag")
    assert hits and all(h["eligibility"] == "uncertain" for h in hits.values())


def test_date_rules_directly():
    assert date_range("2020-03-04")[2] == "day"
    assert date_range("2020-01-01")[2] == "year"
    assert date_range("2020")[2] == "year"
    assert date_range(None, 2020)[2] == "year"
    assert date_range(None)[2] == "missing"
    assert date_range("2020-01-01", jan1_is_year_only=False)[2] == "day"
    cut = ("2020-06-15", None)
    assert date_status(("2020-06-14", None), cut)[0] == "eligible"
    assert date_status(("2020-06-15", None), cut)[0] == "same_date"
    assert date_status(("2020-06-16", None), cut)[0] == "after_cutoff"
    assert date_status(("2020-01-01", None), cut)[0] == "uncertain"
    assert date_status(("2019-01-01", None), cut)[0] == "eligible"
    assert date_status((None, None), cut)[0] == "uncertain"
    assert date_status(("2019-05-05", None), ("2020", None))[0] == "eligible"   # year-only submitted date
    assert date_status(("2020-05-05", None), ("2020", None))[0] == "uncertain"


# ---------- duplicates ----------
def test_duplicate_doi_case_insensitive(corpus_path):
    hits, _ = run(corpus_path)
    kept = [p for p in ("doi:10.1000/dupdoi", "openalex:W10") if p in hits]
    assert len(kept) == 1
    assert hits[kept[0]]["duplicate_ids"]        # the merged copy is recorded


def test_duplicate_pmid(corpus_path):
    hits, _ = run(corpus_path)
    assert len([p for p in ("doi:10.1000/pm1", "doi:10.1000/pm2") if p in hits]) == 1


def test_duplicate_title_same_year_merged_other_year_kept(corpus_path):
    hits, _ = run(corpus_path)
    assert len([p for p in ("doi:10.1000/t1", "openalex:W11") if p in hits]) == 1
    assert "doi:10.1000/t3" in hits                 # same title, different year = different paper


# ---------- evidence ----------
def test_title_only_gets_no_passage(corpus_path):
    hits, _ = run(corpus_path)
    h = hits["doi:10.1000/titleonly"]
    assert h["evidence"] is None and h["evidence_type"] == "title_only" and "no_abstract" in h["flags"]


def test_abstract_evidence_is_a_real_sentence(corpus_path):
    hits, _ = run(corpus_path)
    h = hits["doi:10.1000/before"]
    assert h["evidence_type"] == "abstract_sentence" and h["evidence"] in ABS


def test_non_scholarly_records_excluded(corpus_path):
    hits, ex = run(corpus_path)
    assert "doi:10.1000/paratext" not in hits and "doi:10.1000/notitle" not in hits
    assert ex["not_scholarly_work"] == 2


def test_candidates_are_traceable(corpus_path):
    hits, _ = run(corpus_path)
    for h in hits.values():
        assert h["paper_id"] and (h["url"] or h["doi"])
        for key in ("date", "date_precision", "eligibility", "flags", "evidence", "evidence_type", "cited_by", "duplicate_ids"):
            assert key in h


def test_submitted_from_papers_row():
    s = submitted_from_row({"paper_id": "doi:10.1000/x", "doi": "10.1000/x", "pmcid": "PMC1", "title": "T", "date": "2020-01-02"})
    assert s["doi"] == "10.1000/x" and s["pmcid"] == "PMC1" and s["date"] == "2020-01-02"


# ---------- real corpus ----------
ROOT = os.path.join(os.path.dirname(__file__), "..")
REAL_CORPUS = os.path.join(ROOT, "data", "corpus", "corpus.jsonl")
REAL_PAPERS = os.path.join(ROOT, "data", "papers", "papers.jsonl")


@pytest.mark.skipif(not (os.path.exists(REAL_CORPUS) and os.path.exists(REAL_PAPERS)), reason="real data not present")
def test_real_corpus_no_self_or_later_papers():
    fr = FilteredRetriever(REAL_CORPUS)
    papers = [json.loads(line) for line in open(REAL_PAPERS, encoding="utf-8") if line.strip()]
    for p in papers:
        sub = submitted_from_row(p)
        query = p["title"] + " " + " ".join(s["text"] for s in p.get("sentences", [])[:3])
        for h in fr.search(query, sub, k=10):
            assert h["paper_id"] != p["paper_id"]
            assert h["eligibility"] == "eligible"
            assert h["date"] and h["date"][:4] <= sub["date"][:4]
            if h["date_precision"] == "day":
                assert h["date"] < sub["date"]
            else:
                assert h["date"][:4] < sub["date"][:4]
            if h["evidence_type"] == "title_only":
                assert h["evidence"] is None
