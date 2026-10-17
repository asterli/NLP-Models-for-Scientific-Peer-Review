# M05 - Collecting source papers from Europe PMC.

# Steps: 
# 1) List open-access candidates: python scripts/fetch_papers.py search
#    saves to data/raw/candidates.csv 
# 2) Win picks papers and lists their PMCIDs in data/raw/selected_pmcids.txt,
#    with one per line. 
#    In there, anything after a # in there is a comment. e.g. PMC 123 # RCT, Housing First, Canada 
# 3) Download them, build paper records: python scripts/fetch_papers.py fetch
#    saves to data/papers/papers.jsonl

# When labeling starts, the sentence IDs in papers.jsonl MUST NOT change.
# fetch refuses to overwrite it unless --overwrite is passed.

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path

# Let this script import from src/ without installing anything.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from papers import europepmc  # noqa: E402
from papers.jats import parse_article  # noqa: E402
from papers.records import build_record, check_record  # noqa: E402

# Our scope (T1): open-access papers on interventions for people experiencing homelessness, 2019 to 2026.
DEFAULT_QUERY = (
    '("housing first" OR "supportive housing" OR "permanent supportive housing" OR '
    '"homeless outreach" OR "people experiencing homelessness") '
    "AND OPEN_ACCESS:y AND IN_EPMC:y AND PUB_YEAR:[2019 TO 2026]"
)


def search(args: argparse.Namespace) -> None:
    candidates = europepmc.search(args.query, args.max)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[field.name for field in fields(europepmc.Candidate)])
        writer.writeheader()
        writer.writerows(asdict(c) for c in candidates)
    print(f"wrote {len(candidates)} candidates to {args.out}")


def fetch(args: argparse.Namespace) -> None:
    if args.out.exists() and not args.overwrite:
        sys.exit(f"{args.out} already exists. Re-fetching can renumber sentences and break labels.\n"
                 f"If you're sure, run again with --overwrite.")

    records, warnings = [], []
    for pmcid in read_pmcids(args.ids):
        try:
            xml = europepmc.download_full_text(pmcid, args.cache / "xml")
            meta = europepmc.fetch_metadata(pmcid, args.cache / "meta")
        except europepmc.FullTextUnavailable as error:
            warnings.append(f"{pmcid}: SKIPPED, {error}")
            continue
        record = build_record(pmcid, parse_article(xml), meta)
        records.append(record)
        warnings += [f"{pmcid}: {w}" for w in check_record(record)]
        print(f"ok  {pmcid}  {len(record['sentences']):>3} sentences  {(record['title'] or '')[:60]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = sum(len(r["sentences"]) for r in records)
    print(f"\nwrote {len(records)} papers ({total} sentences) to {args.out}")
    if warnings:
        print("\ncheck these before labeling:\n  " + "\n  ".join(warnings))


def read_pmcids(path: Path) -> list[str]:
    # PMCIDs from a text file: one per line, '#' starts a comment, duplicates dropped.
    pmcids = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = line.split("#", 1)[0].strip()
        if not value:
            continue
        try:
            pmcid = europepmc.normalize_pmcid(value)
        except ValueError as error:
            sys.exit(f"{path}, line {line_number}: {error}")
        if pmcid not in pmcids:
            pmcids.append(pmcid)
    return pmcids


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    s = commands.add_parser("search", help="list open-access candidates to pick from")
    s.add_argument("--query", default=DEFAULT_QUERY)
    s.add_argument("--max", type=int, default=200)
    s.add_argument("--out", type=Path, default=Path("data/raw/candidates.csv"))
    s.set_defaults(run=search)

    f = commands.add_parser("fetch", help="download the picked papers and build papers.jsonl")
    f.add_argument("--ids", type=Path, default=Path("data/raw/selected_pmcids.txt"))
    f.add_argument("--out", type=Path, default=Path("data/papers/papers.jsonl"))
    f.add_argument("--cache", type=Path, default=Path("data/raw"))
    f.add_argument("--overwrite", action="store_true", help="replace an existing papers.jsonl")
    f.set_defaults(run=fetch)

    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
