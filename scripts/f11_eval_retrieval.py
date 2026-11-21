"""F11 - Retrieval evaluation: Recall@k for the filtered BM25 retriever (F10).

  python scripts/f11_eval_retrieval.py --corpus data/corpus/corpus.jsonl \
      --queries evaluation/f11/queries.jsonl --qrels evaluation/f11/qrels.jsonl \
      --out results/f11 [--k 10] [--uncertain exclude]

Inputs (one JSON object per line):
  queries  {"query_id", "claim_text", "newer_paper_id", "newer_link", "newer_pmid",
            "newer_pmcid", "newer_title", "newer_pub_date"}
  qrels    {"query_id", "relevant": [corpus paper ids]}

Writes to --out:
  run.jsonl                {"query_id", "ranked": [paper ids, best first]}, the format score_predictions.py reads
  retrieval_details.jsonl  per query: submitted paper, F10 candidates with evidence, partial exclusion counts
  misses.jsonl             one line per relevant paper not in the top k, with the reason it was missed
  results.json             Recall@k from evaluation/metrics.py, the miss breakdown, settings and git commit

Miss reasons, grouped by what they say about the system:
  ranking      ranked_below_k (passed F10, BM25 put it past k), no_term_overlap (BM25 score 0)
  eligibility  submitted_paper, same_date, after_cutoff, date_uncertain, not_scholarly_work,
               duplicate_version (F10 removed it before ranking counted)
  data         not_in_corpus (the qrels id is not in the corpus file)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Let this script import evaluation/ and retrieval/ from the repo root without installing anything.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "retrieval"))

from evaluation.metrics import recall_at_k  # noqa: E402
from m23_filters import FilteredRetriever, date_status, ids_of, norm_title, submitted_from_row  # noqa: E402
from f11_retriever import PriorWorkRetriever  # noqa: E402

RANKING = ("ranked_below_k", "no_term_overlap")
ELIGIBILITY = ("submitted_paper", "same_date", "after_cutoff", "date_uncertain",
               "not_scholarly_work", "duplicate_version")


def read_jsonl(path: Path) -> list[dict]:
    # utf-8-sig also reads files PowerShell saved with a BOM.
    with path.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def by_query_id(rows: list[dict], path: Path) -> dict[str, dict]:
    out = {}
    for row in rows:
        qid = row["query_id"]
        if qid in out:
            sys.exit(f"{path}: query {qid} appears twice")
        out[qid] = row
    return out


def load_inputs(queries_path: Path, qrels_path: Path) -> tuple[dict, dict]:
    queries = by_query_id(read_jsonl(queries_path), queries_path)
    qrels = by_query_id(read_jsonl(qrels_path), qrels_path)
    # The two files are frozen as a pair, so their query ids must match exactly.
    if queries.keys() != qrels.keys():
        only_q = sorted(queries.keys() - qrels.keys())
        only_r = sorted(qrels.keys() - queries.keys())
        sys.exit(f"query ids differ: only in queries {only_q}, only in qrels {only_r}")
    relevant = {qid: set(row["relevant"]) for qid, row in qrels.items()}
    return queries, relevant


def run_queries(retriever: FilteredRetriever, queries: dict, k: int) -> tuple[dict[str, list[str]], list[dict]]:
    # One search per claim through the same callable retriever the pipeline uses (F12).
    # F10 inside it drops the submitted paper and anything not older than it.
    ranked = {}
    details = []
    for qid, row in queries.items():
        submitted = submitted_from_row(row)
        hits, excluded = retriever.retrieve(row["claim_text"], submitted, k=k, explain=True)
        ranked[qid] = [hit["paper_id"] for hit in hits]
        details.append({
            "query_id": qid,
            "claim_text": row["claim_text"],
            "submitted": submitted,
            "excluded_counts": excluded,  # only exclusions seen before reaching k, not the whole corpus
            "candidates": hits,
        })
    return ranked, details


def eligibility_reason(retriever: FilteredRetriever, doc: dict, submitted: dict) -> str | None:
    # Re-check one paper against F10's rules, in F10's order, using F10's own helpers.
    # Returns the rule that removes it, or None if it is eligible.
    if not doc.get("title") or doc.get("type") == "paratext":
        return "not_scholarly_work"
    sub_title = norm_title(submitted.get("title"))
    if (ids_of(doc) & ids_of(submitted)) or (sub_title and norm_title(doc["title"]) == sub_title):
        return "submitted_paper"
    status, _ = date_status((doc.get("date"), doc.get("year")),
                            (submitted.get("date"), submitted.get("year")), retriever.jan1)
    if status in ("same_date", "after_cutoff"):
        return status
    if status == "uncertain" and retriever.uncertain == "exclude":
        return "date_uncertain"
    return None


def classify_misses(retriever: FilteredRetriever, queries: dict, relevant: dict,
                    ranked: dict, k: int) -> list[dict]:
    # For each relevant paper outside the top k, work out why it was missed.
    everything = len(retriever.docs)
    misses = []
    for qid, row in queries.items():
        lost = sorted(relevant[qid] - set(ranked[qid]))
        if not lost:
            continue
        claim, submitted = row["claim_text"], submitted_from_row(row)
        # Full filtered ranking: where every eligible paper with a BM25 score > 0 lands.
        full = retriever.search(claim, submitted, k=everything)
        filtered_rank = {hit["paper_id"]: hit["rank"] for hit in full}
        # Raw BM25 hits before F10. M22 only returns papers with score > 0.
        scored = {hit["paper_id"] for hit in retriever.R.search(claim, k=everything)}
        for pid in lost:
            rank = filtered_rank.get(pid)
            doc = retriever.by_pid.get(pid)
            if rank is not None:
                reason = "ranked_below_k"
            elif doc is None:
                reason = "not_in_corpus"
            else:
                # Eligibility comes first: a paper F10 removes could not count, whatever its score.
                reason = eligibility_reason(retriever, doc, submitted)
                if reason is None:
                    # Eligible but not in the filtered list: either BM25 never scored it,
                    # or F10 merged it into another record as a duplicate version.
                    reason = "no_term_overlap" if pid not in scored else "duplicate_version"
            misses.append({"query_id": qid, "paper_id": pid, "reason": reason,
                           "filtered_rank": rank, "date": doc.get("date") if doc else None,
                           "title": doc.get("title") if doc else None})
    return misses


def summarize_misses(misses: list[dict], relevant: dict, ranked: dict, k: int) -> dict:
    by_reason = Counter(m["reason"] for m in misses)
    # Recall over eligible papers only: drop relevant papers F10 removed, then score again.
    removed = {(m["query_id"], m["paper_id"]) for m in misses if m["reason"] in ELIGIBILITY}
    eligible = {q: {p for p in rel if (q, p) not in removed} for q, rel in relevant.items()}
    eligible_scores = recall_at_k(ranked, eligible, k)
    return {
        "misses_total": len(misses),
        "misses_by_reason": dict(sorted(by_reason.items())),
        "misses_by_group": {
            "ranking": sum(by_reason[r] for r in RANKING),
            "eligibility": sum(by_reason[r] for r in ELIGIBILITY),
            "data": by_reason["not_in_corpus"],
        },
        f"recall@{k}_eligible_only": eligible_scores[f"recall@{k}"],
        "queries_eligible_only": eligible_scores["queries"],
    }


def git_commit() -> str | None:
    # Record the exact code version so the number can be reproduced. None if git isn't available.
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--uncertain", choices=["exclude", "flag"], default="exclude")
    args = parser.parse_args(argv)

    queries, relevant = load_inputs(args.queries, args.qrels)
    retriever = PriorWorkRetriever(str(args.corpus), uncertain=args.uncertain)
    ranked, details = run_queries(retriever, queries, args.k)
    # Miss analysis needs F10's internals (full-depth ranking, raw BM25), so it uses the wrapped retriever.
    misses = classify_misses(retriever.filtered, queries, relevant, ranked, args.k)

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "run.jsonl", [{"query_id": qid, "ranked": ranked[qid]} for qid in queries])
    write_jsonl(args.out / "retrieval_details.jsonl", details)
    write_jsonl(args.out / "misses.jsonl", misses)

    results = {
        "task": "f11_retrieval",
        "k": args.k,
        "uncertain": args.uncertain,
        "commit": git_commit(),
        "inputs": {"corpus": str(args.corpus), "queries": str(args.queries), "qrels": str(args.qrels)},
        **recall_at_k(ranked, relevant, args.k),
        **summarize_misses(misses, relevant, ranked, args.k),
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"recall@{args.k} = {results[f'recall@{args.k}']:.3f} over {results['queries']} queries, "
          f"misses by group {results['misses_by_group']} -> {args.out}")
    return results


if __name__ == "__main__":
    main()