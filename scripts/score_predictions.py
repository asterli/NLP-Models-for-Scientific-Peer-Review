"""M02 - Score predictions against gold labels and write a results file.

  python scripts/score_predictions.py claims    --pred P.jsonl --gold G.jsonl --out results/claims.json
  python scripts/score_predictions.py novelty   --pred P.jsonl --gold G.jsonl --out results/novelty.json
  python scripts/score_predictions.py retrieval --run R.jsonl --qrels Q.jsonl --out results/retrieval.json [--k 10]

One JSON object per line in every input:
  claims     pred   {"paper_id", "sentence_id"}, one line per sentence the model called CLAIM
             gold   {"paper_id", "sentence_id", "label": "CLAIM" or "BACKGROUND"}, every labeled sentence
  novelty    pred   {"comparison_id", "predicted"}
             gold   {"comparison_id", "label"}
  retrieval  run    {"query_id", "ranked": [work ids, best first]}
             qrels  {"query_id", "relevant": [work ids]}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Let this script import evaluation/ from the repo root without installing anything.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import claim_scores, classification_scores, recall_at_k  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    # utf-8-sig also reads files PowerShell saved with a BOM.
    with path.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_claims(args) -> dict:
    gold = {}
    for row in read_jsonl(args.gold):
        key = (row["paper_id"], row["sentence_id"])
        label = row["label"].strip().upper()  # "Claim" from a spreadsheet still counts
        if label not in ("CLAIM", "BACKGROUND"):
            sys.exit(f"{args.gold}: {key} has label {row['label']!r}, expected CLAIM or BACKGROUND")
        if key in gold:
            sys.exit(f"{args.gold}: {key} is labeled twice")
        gold[key] = label
    predicted = {(row["paper_id"], row["sentence_id"]) for row in read_jsonl(args.pred)}
    return claim_scores(predicted, gold)


def score_novelty(args) -> dict:
    predicted = {row["comparison_id"]: row["predicted"] for row in read_jsonl(args.pred)}
    gold = read_jsonl(args.gold)
    missing = [row["comparison_id"] for row in gold if row["comparison_id"] not in predicted]
    if missing:
        sys.exit(f"no prediction for {len(missing)} gold comparison(s), e.g. {missing[0]}")
    scores = classification_scores([row["label"] for row in gold],
                                   [predicted[row["comparison_id"]] for row in gold])
    scores["ignored_predictions"] = len(predicted.keys() - {row["comparison_id"] for row in gold})
    return scores


def score_retrieval(args) -> dict:
    ranked = {row["query_id"]: row["ranked"] for row in read_jsonl(args.run)}
    relevant = {row["query_id"]: set(row["relevant"]) for row in read_jsonl(args.qrels)}
    return recall_at_k(ranked, relevant, args.k)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tasks = parser.add_subparsers(dest="task", required=True)

    claims = tasks.add_parser("claims")
    claims.add_argument("--pred", type=Path, required=True)
    claims.add_argument("--gold", type=Path, required=True)
    claims.set_defaults(score=score_claims)

    novelty = tasks.add_parser("novelty")
    novelty.add_argument("--pred", type=Path, required=True)
    novelty.add_argument("--gold", type=Path, required=True)
    novelty.set_defaults(score=score_novelty)

    retrieval = tasks.add_parser("retrieval")
    retrieval.add_argument("--run", type=Path, required=True)
    retrieval.add_argument("--qrels", type=Path, required=True)
    retrieval.add_argument("--k", type=int, default=10)
    retrieval.set_defaults(score=score_retrieval)

    for sub in (claims, novelty, retrieval):
        sub.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        scores = args.score(args)
    except ValueError as e:  # bad labels or mismatched lengths from metrics.py
        sys.exit(str(e))

    inputs = {name: str(value) for name, value in vars(args).items() if isinstance(value, Path) and name != "out"}
    results = {"task": args.task, "inputs": inputs} | scores
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    # The full breakdowns are in the file. Print just the headline numbers and counts.
    print(json.dumps({k: v for k, v in results.items() if k not in ("per_class", "confusion", "per_query")}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()