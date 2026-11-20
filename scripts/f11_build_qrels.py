"""F11 - Build the frozen queries and qrels from the final relevance review.

  python scripts/f11_build_qrels.py --review evaluation/f11/F11_relevance_review.csv \
      --splits-dir data/novelty/splits --out-dir evaluation/f11

Input:
  review CSV   the review sheet exported as CSV, one row per claim-paper pair. Columns used:
               "Query ID", "Older DOI / link" (https://doi.org/...), "Final" (Yes or No)
  splits dir   dev.jsonl and test.jsonl from F03; each query ID is the comp_id of a split row

Writes to --out-dir:
  qrels.jsonl    {"query_id", "relevant": [corpus paper ids]}, only queries with at least one Yes
  queries.jsonl  the same queries, with the claim and submitted-paper fields copied from the splits
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

QUERY_FIELDS = ("claim_text", "newer_paper_id", "newer_link", "newer_pmid",
                "newer_pmcid", "newer_title", "newer_pub_date")
NEEDED = ("Query ID", "Older DOI / link", "Final")


def read_review(path: Path) -> list[dict]:
    # utf-8-sig also reads CSVs that Excel saved with a BOM.
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header = [h.strip() for h in rows[0]]
    for name in NEEDED:
        if header.count(name) != 1:
            sys.exit(f"{path}: need exactly one column named {name!r}, found {header.count(name)}")
    col = {name: header.index(name) for name in NEEDED}
    out = []
    for line, row in enumerate(rows[1:], start=2):
        if not any(cell.strip() for cell in row):
            continue  # blank line at the end of an export
        out.append({"line": line, **{name: row[col[name]].strip() for name in NEEDED}})
    return out


def doi_id(link: str) -> str:
    # "https://doi.org/10.1/ABC" -> "doi:10.1/abc", the corpus paper_id format.
    if "doi.org/" not in link:
        raise ValueError(f"not a doi.org link: {link!r}")
    return "doi:" + link.split("doi.org/", 1)[1].lower()


def build_qrels(review: list[dict]) -> tuple[dict[str, list[str]], list[str]]:
    # Returns {query_id: relevant ids} for queries with a Yes, and the queries with no Yes.
    relevant: dict[str, list[str]] = {}
    seen = []  # every query, in sheet order
    for row in review:
        qid, final = row["Query ID"], row["Final"]
        if final not in ("Yes", "No"):
            sys.exit(f"line {row['line']}: Final must be Yes or No, got {final!r}")
        if qid not in seen:
            seen.append(qid)
        if final == "Yes":
            pid = doi_id(row["Older DOI / link"])
            ids = relevant.setdefault(qid, [])
            if pid not in ids:
                ids.append(pid)
    # Keep sheet order so the files diff cleanly against earlier versions.
    ordered = {qid: relevant[qid] for qid in seen if qid in relevant}
    dropped = [qid for qid in seen if qid not in relevant]
    return ordered, dropped


def load_splits(splits_dir: Path) -> dict[str, dict]:
    rows = {}
    for name in ("dev.jsonl", "test.jsonl"):
        with (splits_dir / name).open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows[row["comp_id"]] = row
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    # json.dumps escapes non-ASCII by default, so the files stay ASCII-only.
    with path.open("w", encoding="ascii", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    qrels, dropped = build_qrels(read_review(args.review))
    splits = load_splits(args.splits_dir)
    missing = [qid for qid in qrels if qid not in splits]
    if missing:
        sys.exit(f"query ids not found in the dev/test splits: {missing}")
    # Fields a split row doesn't have (e.g. newer_pmid) are written as null, as before.
    queries = [{"query_id": qid, **{k: splits[qid].get(k) for k in QUERY_FIELDS}} for qid in qrels]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "qrels.jsonl", [{"query_id": q, "relevant": ids} for q, ids in qrels.items()])
    write_jsonl(args.out_dir / "queries.jsonl", queries)

    pairs = sum(len(ids) for ids in qrels.values())
    print(f"{len(qrels)} queries, {pairs} relevant pairs -> {args.out_dir}")
    print(f"dropped (no Yes): {', '.join(dropped) or 'none'}")
    return {"queries": len(qrels), "pairs": pairs, "dropped": dropped}


if __name__ == "__main__":
    main()