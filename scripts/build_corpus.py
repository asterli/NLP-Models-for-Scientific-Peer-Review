"""M21 - Build the background corpus of older papers from OpenAlex.

Run:  python scripts/build_corpus.py
Reads data/papers/papers.jsonl and writes to data/corpus/:
  corpus.jsonl          one older work per line (docs/data-schema.md, "Corpus document")
  citation_links.jsonl  which corpus work each source-paper reference (ref_id) points to
  coverage.json         how many references were found, and which fields are missing

For each source paper, its references are looked up by DOI (or PMID), then the paper's own
OpenAlex reference list adds the ones our XML had no DOI for. Every lookup is cached in
data/raw/openalex/, so if a run stops on an error, running it again continues where it left off.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Let this script import from src/ without installing anything.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus import openalex  # noqa: E402
from corpus.records import corpus_record, field_coverage, lookup_key, openalex_id  # noqa: E402

OUTPUT_FILES = ("corpus.jsonl", "citation_links.jsonl", "coverage.json")


def build(papers: list[dict], cache: Path) -> tuple[list[dict], list[dict], dict]:
    # The corpus, the citation links, and the coverage report for a list of source papers.
    corpus: dict[str, dict] = {}       # paper_id -> record, so each work appears once
    by_openalex_id: dict[str, str] = {}  # openalex_id -> paper_id, to skip works we already have
    links, per_paper = [], []

    def add(work: dict, citing_paper: str) -> dict:
        # Add a work (or find it if we already have it) and note which source paper cites it.
        new = corpus_record(work, "reference")
        by_openalex_id[new["openalex_id"]] = new["paper_id"]  # OpenAlex can have 2 IDs for one DOI
        record = corpus.setdefault(new["paper_id"], new)
        cite(record, citing_paper)
        return record

    def cite(record: dict, citing_paper: str) -> None:
        if citing_paper not in record["cited_by"]:
            record["cited_by"].append(citing_paper)

    for paper in papers:
        stats = {"paper_id": paper["paper_id"], "pmcid": paper["pmcid"],
                 "references": len(paper["references"]), "no_id": 0, "not_in_openalex": 0,
                 "linked": 0, "added_from_openalex_list": 0}

        # 1) Each reference in our XML, by DOI or PMID.
        for ref in paper["references"]:
            key = lookup_key(ref)
            if key is None:
                stats["no_id"] += 1  # e.g. a report or website with no DOI or PMID
                continue
            work = openalex.get_work(key, cache)
            if work is None:
                stats["not_in_openalex"] += 1
                continue
            record = add(work, paper["paper_id"])
            links.append({"paper_id": paper["paper_id"], "ref_id": ref["ref_id"],
                          "corpus_id": record["paper_id"], "matched_by": key.split(":")[0]})
            stats["linked"] += 1

        # 2) The paper's own reference list in OpenAlex, for references our XML had no DOI for.
        source = openalex.get_work(f"doi:{paper['doi']}", cache) if paper.get("doi") else None
        for url in (source or {}).get("referenced_works", []):
            work_id = openalex_id(url)
            if work_id in by_openalex_id:  # already have it: just note who cites it, no request
                cite(corpus[by_openalex_id[work_id]], paper["paper_id"])
                continue
            work = openalex.get_work(work_id, cache)
            if work is not None:
                before = len(corpus)
                add(work, paper["paper_id"])
                stats["added_from_openalex_list"] += len(corpus) - before
        stats["source_in_openalex"] = source is not None

        per_paper.append(stats)
        print(f"ok  {paper['pmcid']:<12} {stats['linked']:>3}/{stats['references']:<3} refs linked, "
              f"+{stats['added_from_openalex_list']} from OpenAlex list")

    records = list(corpus.values())
    coverage = {"summary": {**field_coverage(records), "links": len(links)}, "papers": per_paper}
    return records, links, coverage


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--papers", type=Path, default=Path("data/papers/papers.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/corpus"), help="output folder")
    parser.add_argument("--cache", type=Path, default=Path("data/raw/openalex"))
    parser.add_argument("--overwrite", action="store_true", help="replace existing corpus files")
    args = parser.parse_args(argv)

    existing = [name for name in OUTPUT_FILES if (args.out / name).exists()]
    if existing and not args.overwrite:
        sys.exit(f"{args.out} already has {', '.join(existing)}. Rebuilding can change corpus IDs.\n"
                 f"If you're sure, run again with --overwrite.")

    papers = [json.loads(line) for line in args.papers.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        records, links, coverage = build(papers, args.cache)
    except openalex.OpenAlexError as error:
        sys.exit(f"stopped: {error}\nEverything fetched so far is cached; run again to continue.")

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "corpus.jsonl", records)
    write_jsonl(args.out / "citation_links.jsonl", links)
    (args.out / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")

    s = coverage["summary"]
    print(f"\nwrote {s['works']} works and {s['links']} citation links to {args.out}")
    print(f"with abstract: {s['abstract']}, with date: {s['date']}, with DOI: {s['doi']}, with venue: {s['venue']}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    from dotenv import load_dotenv  # only when run as a command, so tests never read .env
    load_dotenv()
    main()