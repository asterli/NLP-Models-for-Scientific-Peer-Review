# Tests for scripts/f11_eval_retrieval.py (F11). Run with:  pytest tests/test_f11_eval_retrieval.py
# Uses a small fake corpus written to a temp folder, plus checks on the real frozen files (skipped if missing).

import json
from pathlib import Path

import pytest

from f11_eval_retrieval import main
from score_predictions import main as m02_main

ROOT = Path(__file__).resolve().parents[1]
CLAIM = "Housing First improved housing stability"
ABS = "Housing First improved housing stability for homeless adults."


def doc(pid, title, date, abstract=ABS):
    return {"paper_id": pid, "title": title, "date": date, "year": int(date[:4]), "abstract": abstract}


# One claim from a paper published 2020-06-15, and one corpus paper for every way a relevant paper can be missed.
CORPUS = [
    doc("doi:10.1/hit", "Housing First housing stability trial", "2015-06-01"),
    doc("doi:10.1/below", "Housing outcomes cohort", "2014-03-01", abstract="Housing was measured once."),
    doc("doi:10.1/mid", "Housing First stability review", "2016-03-01"),
    doc("doi:10.1/new", "Housing First stability follow-up", "2022-03-01"),          # published after the claim
    doc("doi:10.1/sameyear", "Housing First stability in 2020", "2020-01-01"),       # Jan 1 = year only, same year
    doc("doi:10.1/nowords", "Foot conditions survey", "2012-01-02", abstract="Foot problems were common."),
    doc("doi:10.1/self", "Housing First trial for veterans", "2020-06-15"),          # the submitted paper itself
    doc("doi:10.1/dupA", "Housing First stability pilot", "2013-05-01"),
    doc("doi:10.1/dupB", "Housing First stability pilot", "2013-07-01", abstract="A pilot."),  # weaker copy of dupA
]

QUERY = {"query_id": "Q1", "claim_text": CLAIM, "newer_paper_id": "doi:10.1/self", "newer_link": None,
         "newer_pmid": None, "newer_pmcid": None, "newer_title": "Housing First trial for veterans",
         "newer_pub_date": "2020-06-15"}

EXPECTED_REASONS = {
    "doi:10.1/below": "ranked_below_k",
    "doi:10.1/hit": "ranked_below_k",
    "doi:10.1/new": "after_cutoff",
    "doi:10.1/sameyear": "date_uncertain",
    "doi:10.1/nowords": "no_term_overlap",
    "doi:10.1/self": "submitted_paper",
    "doi:10.1/dupB": "duplicate_version",
    "doi:10.1/ghost": "not_in_corpus",
}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def run(tmp_path, relevant, k=2, queries=None, extra=()):
    # Write the fake corpus, queries and qrels, run F11, and return (results, out folder).
    corpus = write_jsonl(tmp_path / "corpus.jsonl", CORPUS)
    qfile = write_jsonl(tmp_path / "queries.jsonl", queries or [QUERY])
    qrels = write_jsonl(tmp_path / "qrels.jsonl", [{"query_id": "Q1", "relevant": relevant}])
    out = tmp_path / "out"
    results = main(["--corpus", corpus, "--queries", qfile, "--qrels", qrels, "--out", str(out),
                    "--k", str(k), *extra])
    return results, out


def test_relevant_paper_at_rank_1_scores_full_recall(tmp_path):
    results, out = run(tmp_path, ["doi:10.1/dupA"], k=1)
    assert results["recall@1"] == 1.0
    assert results["misses_total"] == 0
    assert read_jsonl(out / "run.jsonl") == [{"query_id": "Q1", "ranked": ["doi:10.1/dupA"]}]


def test_every_miss_gets_the_right_reason(tmp_path):
    results, out = run(tmp_path, sorted(EXPECTED_REASONS), k=2)
    got = {m["paper_id"]: m["reason"] for m in read_jsonl(out / "misses.jsonl")}
    assert got == EXPECTED_REASONS
    assert results["misses_by_group"] == {"ranking": 3, "eligibility": 4, "data": 1}


def test_ranked_below_k_records_the_filtered_rank(tmp_path):
    _, out = run(tmp_path, ["doi:10.1/hit", "doi:10.1/below"], k=2)
    ranks = {m["paper_id"]: m["filtered_rank"] for m in read_jsonl(out / "misses.jsonl")}
    assert ranks == {"doi:10.1/hit": 3, "doi:10.1/below": 4}


def test_eligible_only_recall_drops_papers_f10_removed(tmp_path):
    # Top 4 finds hit and below: 2 of 8 overall, 2 of 4 once the 4 F10 removals are taken out.
    results, _ = run(tmp_path, sorted(EXPECTED_REASONS), k=4)
    assert results["recall@4"] == 0.25
    assert results["recall@4_eligible_only"] == 0.5


def test_uncertain_flag_keeps_year_only_papers(tmp_path):
    # With --uncertain flag, the same-year paper stays in the ranking, so it is no longer an eligibility miss.
    _, out = run(tmp_path, ["doi:10.1/sameyear"], k=1, extra=["--uncertain", "flag"])
    reasons = {m["reason"] for m in read_jsonl(out / "misses.jsonl")}
    assert "date_uncertain" not in reasons


def test_run_file_scores_the_same_through_m02(tmp_path):
    results, out = run(tmp_path, sorted(EXPECTED_REASONS), k=4)
    m02_out = tmp_path / "m02.json"
    m02_main(["retrieval", "--run", str(out / "run.jsonl"), "--qrels", str(tmp_path / "qrels.jsonl"),
              "--k", "4", "--out", str(m02_out)])
    assert json.loads(m02_out.read_text(encoding="utf-8"))["recall@4"] == results["recall@4"]


def test_writes_all_output_files_and_settings(tmp_path):
    results, out = run(tmp_path, ["doi:10.1/hit"], k=2)
    for name in ("run.jsonl", "retrieval_details.jsonl", "misses.jsonl", "results.json"):
        assert (out / name).exists()
    assert (results["task"], results["k"], results["uncertain"]) == ("f11_retrieval", 2, "exclude")
    detail = read_jsonl(out / "retrieval_details.jsonl")[0]
    assert detail["submitted"]["date"] == "2020-06-15"


def test_same_inputs_give_the_same_results(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first, _ = run(tmp_path / "a", sorted(EXPECTED_REASONS), k=3)
    second, _ = run(tmp_path / "b", sorted(EXPECTED_REASONS), k=3)
    keep = ("recall@3", "per_query", "misses_by_reason", "recall@3_eligible_only")
    assert {k: first[k] for k in keep} == {k: second[k] for k in keep}


def test_query_ids_must_match_between_files(tmp_path):
    other = dict(QUERY, query_id="Q2")
    with pytest.raises(SystemExit, match="query ids differ"):
        run(tmp_path, ["doi:10.1/hit"], queries=[other])


def test_duplicate_query_id_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="appears twice"):
        run(tmp_path, ["doi:10.1/hit"], queries=[QUERY, QUERY])


# ---------- checks on the real frozen files ----------
REAL = {"queries": ROOT / "evaluation/f11/queries.jsonl", "qrels": ROOT / "evaluation/f11/qrels.jsonl",
        "corpus": ROOT / "data/corpus/corpus.jsonl"}


@pytest.mark.skipif(not all(p.exists() for p in REAL.values()), reason="frozen F11 files or corpus not present")
def test_frozen_files_match_and_point_into_the_corpus():
    queries = read_jsonl(REAL["queries"])
    qrels = read_jsonl(REAL["qrels"])
    corpus_ids = {row["paper_id"] for row in read_jsonl(REAL["corpus"])}
    assert [q["query_id"] for q in queries] == [r["query_id"] for r in qrels]
    assert all(r["relevant"] for r in qrels), "every scored query needs at least one relevant paper"
    assert all(pid in corpus_ids for r in qrels for pid in r["relevant"])
    assert all(q["claim_text"] and q["newer_pub_date"] for q in queries)