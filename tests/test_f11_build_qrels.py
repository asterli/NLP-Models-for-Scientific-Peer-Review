# Tests for scripts/f11_build_qrels.py (F11). Run with:  pytest tests/test_f11_build_qrels.py
# Uses a small fake review CSV and fake splits written to a temp folder, plus one check on the real files.

import csv
import json
from pathlib import Path

import pytest

from f11_build_qrels import main

ROOT = Path(__file__).resolve().parents[1]

# Same layout as the exported sheet: extra columns, a repeated "Reason" header, a multi-line cell.
HEADER = ["Comp ID", "Query ID", "Newer claim", "Older DOI / link", "Reason", "Reason", "Final"]
ROWS = [
    ["C1", "Q1", "claim one", "https://doi.org/10.1/AAA", "a", "b", "Yes"],
    ["C2", "Q1", "claim one", "https://doi.org/10.1/bbb", "line 1\nline 2", "", "No"],
    ["C3", "Q1", "claim one", "https://doi.org/10.1/aaa", "", "", "Yes"],  # same paper again
    ["C4", "Q2", "claim two", "https://doi.org/10.1/ccc", "", "", "No"],    # Q2 has no Yes
    ["C5", "Q3", "claim three", "https://doi.org/10.1/ddd", "", "", "Yes"],
]
SPLIT_ROW = {"newer_paper_id": "doi:10.9/new", "newer_link": "https://doi.org/10.9/new",
             "newer_pmcid": None, "newer_title": "T", "newer_pub_date": "2020-01-01"}  # no newer_pmid


def write_review(path, header=HEADER, rows=ROWS):
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows([header, *rows])
    return path


def write_splits(folder, ids=("Q1", "Q2", "Q3")):
    folder.mkdir()
    dev = [{"comp_id": "Q1", "claim_text": "Lower ED use (\u221250%)", **SPLIT_ROW}]
    test = [{"comp_id": q, "claim_text": f"claim {q}", **SPLIT_ROW} for q in ids if q != "Q1"]
    for name, rows in (("dev.jsonl", dev), ("test.jsonl", test)):
        (folder / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return folder


def run(tmp_path, review=None, splits=None):
    review = review or write_review(tmp_path / "review.csv")
    splits = splits or write_splits(tmp_path / "splits")
    out = tmp_path / "out"
    summary = main(["--review", str(review), "--splits-dir", str(splits), "--out-dir", str(out)])
    read = lambda name: [json.loads(l) for l in (out / name).read_text(encoding="ascii").splitlines()]
    return summary, read("qrels.jsonl"), read("queries.jsonl"), out


def test_yes_rows_become_qrels_and_empty_queries_are_dropped(tmp_path):
    summary, qrels, queries, _ = run(tmp_path)
    assert qrels == [{"query_id": "Q1", "relevant": ["doi:10.1/aaa"]},
                     {"query_id": "Q3", "relevant": ["doi:10.1/ddd"]}]
    assert summary == {"queries": 2, "pairs": 2, "dropped": ["Q2"]}
    assert [q["query_id"] for q in queries] == ["Q1", "Q3"]


def test_queries_copy_split_fields_and_stay_ascii(tmp_path):
    _, _, queries, out = run(tmp_path)
    q1 = queries[0]
    assert q1["claim_text"] == "Lower ED use (\u221250%)"  # escaped in the file, same text when read back
    assert q1["newer_pmid"] is None                      # missing in the split row -> null
    assert q1["newer_pub_date"] == "2020-01-01"
    (out / "queries.jsonl").read_bytes().decode("ascii")  # raises if any byte is not ASCII


@pytest.mark.parametrize("final", ["", "yes", "Unsure"])
def test_final_must_be_yes_or_no(tmp_path, final):
    rows = [ROWS[0][:-1] + [final]]
    with pytest.raises(SystemExit):
        run(tmp_path, review=write_review(tmp_path / "review.csv", rows=rows))


def test_query_missing_from_splits_stops(tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path, splits=write_splits(tmp_path / "splits", ids=("Q1", "Q2")))


@pytest.mark.parametrize("header", [HEADER[:-1] + ["Judgment"], HEADER + ["Final"]])
def test_final_column_must_appear_once(tmp_path, header):
    rows = [r + ["Yes"] * (len(header) - len(r)) for r in ROWS]
    with pytest.raises(SystemExit):
        run(tmp_path, review=write_review(tmp_path / "review.csv", header=header, rows=rows))


def test_real_queries_match_the_splits():
    # Every frozen query must equal what this script would copy from the splits.
    frozen = ROOT / "evaluation" / "f11" / "queries.jsonl"
    if not frozen.exists():
        pytest.skip("frozen queries not present")
    splits = {}
    for name in ("dev.jsonl", "test.jsonl"):
        for line in (ROOT / "data" / "novelty" / "splits" / name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            splits[row["comp_id"]] = row
    for line in frozen.read_text(encoding="utf-8").splitlines():
        q = json.loads(line)
        split = splits[q["query_id"]]
        assert all(q[k] == split.get(k) for k in q if k != "query_id"), q["query_id"]