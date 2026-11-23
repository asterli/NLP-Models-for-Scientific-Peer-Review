"""F12: run paper text through the real components and save results plus an evidence-linked report.

  paper -> F06 claim extractor -> F11 retrieval (F10 filters) -> F09 comparison classifier -> report

The extractor needs a CUDA GPU and the comparison classifier needs Apple Silicon (MLX), so the usual
route is two steps (full instructions in docs/pipeline.md):

  # on the GPU cluster: extract claims, keeping zero-claim papers
  python scripts/run_pipeline.py extract --split test \
      --claims-adapter /project2/xiangren_1987/group_40/runs/m11/r16_lr2e-4_s544/best --out outputs/claims_test.json

  # on an Apple Silicon Mac: retrieve, compare, report
  python scripts/run_pipeline.py run --split test --claims-file outputs/claims_test.json \
      --local-model-dir /path/to/qwen-4bit --out-dir outputs/pipeline_test

`run` also accepts --claims-adapter (live extraction, on a machine that has both), and --no-comparison
(extraction + retrieval only). Writes results.json, predictions.jsonl and report.md to a new --out-dir.
Exits 1 if any paper, claim or pair hit an error (outputs are still written).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline import core, inputs, report  # noqa: E402
from pipeline.components import LiveClaims, MLXComparison, Retrieval, SavedClaims, save_claims  # noqa: E402
from pipeline.provenance import file_sha256, rel, run_info  # noqa: E402

DEFAULT_PAPERS = ROOT / "data/papers/papers.jsonl"
DEFAULT_CORPUS = ROOT / "data/corpus/corpus.jsonl"
DEFAULT_COMPARISON_ADAPTER = ROOT / "artifacts/comparison/local_20261114/adapter"
SPLITS_DIR = ROOT / "data/claims/splits"


def add_paper_args(p):
    g = p.add_argument_group("papers (choose one source)")
    g.add_argument("--paper-id", action="append", default=[], help="a paper_id in --papers; repeatable")
    g.add_argument("--split", choices=("train", "dev", "test"), help="every paper in this frozen claim split (M08)")
    g.add_argument("--paper-file", type=Path, help="JSON/JSONL Paper record(s), docs/data-schema.md section 1")
    g.add_argument("--text-file", type=Path, help="plain text; needs --title and --date")
    g.add_argument("--title")
    g.add_argument("--date", help="publication/submission date YYYY-MM-DD (retrieval keeps only older work)")
    g.add_argument("--doi")
    g.add_argument("--papers", type=Path, default=DEFAULT_PAPERS)


def load_papers(args, parser):
    sources = [bool(args.paper_id), bool(args.split), bool(args.paper_file), bool(args.text_file)]
    if sum(sources) != 1:
        parser.error("give exactly one of --paper-id, --split, --paper-file, --text-file")
    if args.text_file:
        if not (args.title and args.date):
            parser.error("--text-file needs --title and --date")
        return [inputs.paper_from_text(args.text_file, args.title, args.date, args.doi)], str(args.text_file)
    if args.paper_file:
        return inputs.papers_from_file(args.paper_file), str(args.paper_file)
    ids = args.paper_id
    if args.split:
        rows = [json.loads(line) for line in (SPLITS_DIR / f"{args.split}.jsonl").read_text().splitlines() if line.strip()]
        ids = sorted({r["paper_id"] for r in rows})
    return inputs.papers_by_id(args.papers, ids), str(args.papers)


def data_info(source, papers):
    path = Path(source)
    return {"papers_source": rel(source) if path.exists() else source, "papers_source_sha256": file_sha256(path) if path.is_file() else None,
            "paper_ids": [p["paper_id"] for p in papers]}


def fresh(path: Path, parser, what):
    if path.exists() and (path.is_file() or any(path.iterdir())):
        parser.error(f"{what} {path} already exists; choose a new one so earlier results are not overwritten")


def cmd_extract(args, parser):
    fresh(args.out, parser, "--out")
    papers, source = load_papers(args, parser)
    info = run_info() | {"data": data_info(source, papers)}
    live = LiveClaims(args.claims_adapter, load_4bit=args.load_4bit)
    out = save_claims(args.out, papers, live, info)
    for pid, entry in out["papers"].items():
        print(f"{pid}: {entry['status']}, {len(entry['claims'])} claims")
    print(f"claims -> {args.out}")
    return 1 if any(e["status"] == "error" for e in out["papers"].values()) else 0


def cmd_run(args, parser):
    if args.claims_file is None and args.claims_adapter is None:
        parser.error("give --claims-file or --claims-adapter")
    if not args.no_comparison and args.local_model_dir is None:
        parser.error("give --local-model-dir for the comparison classifier, or --no-comparison")
    if args.compare_top is not None and args.compare_top < 1:
        parser.error("--compare-top must be at least 1")
    fresh(args.out_dir, parser, "--out-dir")
    papers, source = load_papers(args, parser)
    info = run_info()

    extract = (SavedClaims(args.claims_file) if args.claims_file
               else LiveClaims(args.claims_adapter, load_4bit=args.load_4bit))
    retrieve = Retrieval(args.corpus, uncertain=args.uncertain)
    compare = None if args.no_comparison else MLXComparison(args.comparison_adapter, args.local_model_dir)

    records, t0 = [], time.time()
    for paper in papers:
        rec = core.run_paper(paper, extract, retrieve, compare, k=args.k, compare_top=args.compare_top)
        records.append(rec)
        print(f"{paper['paper_id']}: {rec['status']}, {len(rec['claims'])} claims", flush=True)

    result = {
        "run": info | {"finished_seconds": round(time.time() - t0, 1)},
        "settings": {"k": args.k, "compare_top": args.compare_top, "uncertain_dates": args.uncertain,
                     "evidence_for_comparison": "F10 best-matching abstract sentence (evidence_kind=passage)"},
        "data": data_info(source, papers),
        "components": {"claims": extract.provenance(), "retrieval": retrieve.provenance(),
                       "comparison": compare.provenance() if compare else None},
        "summary": core.summarize(records),
        "papers": records,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in core.predictions(records):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "report.md").write_text(report.render(result), encoding="utf-8")

    s = result["summary"]
    print(json.dumps(s))
    print(f"results -> {args.out_dir}/results.json, predictions.jsonl, report.md")
    failed = (s["paper_status"].get("extraction_error", 0) + s["claim_status"].get("retrieval_error", 0)
              + s["pair_status"].get("error", 0))
    if failed:
        print(f"WARNING: {failed} paper/claim/pair errors; see results.json", file=sys.stderr)
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    ex = sub.add_parser("extract", help="F06 only: write a claims file (CUDA GPU)")
    add_paper_args(ex)
    ex.add_argument("--claims-adapter", type=Path, required=True, help="M11 LoRA adapter directory")
    ex.add_argument("--load-4bit", action="store_true", help="must match how the adapter was trained")
    ex.add_argument("--out", type=Path, required=True)

    run = sub.add_parser("run", help="full pipeline: claims -> retrieval -> comparison -> report")
    add_paper_args(run)
    claims = run.add_mutually_exclusive_group()
    claims.add_argument("--claims-file", type=Path, help="claims file from `extract`, or F06 predictions JSONL")
    claims.add_argument("--claims-adapter", type=Path, help="run the F06 extractor live (CUDA GPU)")
    run.add_argument("--load-4bit", action="store_true")
    run.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    run.add_argument("--k", type=int, default=10, help="candidates retrieved per claim (F11 evaluated 10)")
    run.add_argument("--uncertain", choices=("exclude", "flag"), default="exclude",
                     help="works whose date cannot confirm they are older (F10)")
    run.add_argument("--compare-top", type=int, help="compare only the top N candidates per claim (default all)")
    run.add_argument("--comparison-adapter", type=Path, default=DEFAULT_COMPARISON_ADAPTER)
    run.add_argument("--local-model-dir", type=Path, help="pinned mlx-community/Qwen2.5-7B-Instruct-4bit download")
    run.add_argument("--no-comparison", action="store_true", help="skip the classifier; pairs are recorded not_run")
    run.add_argument("--out-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    return cmd_extract(args, parser) if args.command == "extract" else cmd_run(args, parser)


if __name__ == "__main__":
    sys.exit(main())
