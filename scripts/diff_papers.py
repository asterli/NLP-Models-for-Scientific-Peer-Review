"""Compare two papers.jsonl files sentence by sentence (M05).

Use this before replacing the locked data/papers/papers.jsonl with a rebuilt copy:
  python scripts/fetch_papers.py fetch --out data/papers/papers.rebuilt.jsonl
  python scripts/diff_papers.py data/papers/papers.jsonl data/papers/papers.rebuilt.jsonl
(fetch reads the cached XML in data/raw/, so the rebuild doesn't download anything.)

Papers are matched by PMCID. Labels point to sentence IDs, so the two kinds of change matter:
  "edited"            same sentence, new text or cites. The ID still points to the same sentence.
  "sentences change"  sentences added or removed. Every later ID now means a different sentence.
"""
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

METADATA_FIELDS = ("paper_id", "title", "date", "doi", "url")


def load(path: Path) -> dict[str, dict]:
    # {pmcid: record} for every line in a papers.jsonl file.
    with path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {r["pmcid"]: r for r in records}


def diff_paper(old: dict, new: dict) -> tuple[list[str], bool]:
    # Lines describing what changed, and whether any sentence IDs shift.
    lines = [f"  {key}: {old.get(key)!r} -> {new.get(key)!r}"
             for key in METADATA_FIELDS if old.get(key) != new.get(key)]
    old_s, new_s = old["sentences"], new["sentences"]
    ids_shift = False

    matcher = difflib.SequenceMatcher(a=[s["text"] for s in old_s], b=[s["text"] for s in new_s], autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            # Same number of sentences: each old sentence lines up with one new one.
            for a, b in zip(old_s[i1:i2], new_s[j1:j2]):
                sid = a["sentence_id"] if a["sentence_id"] == b["sentence_id"] else f"{a['sentence_id']} (now {b['sentence_id']})"
                if a["text"] != b["text"]:
                    lines += [f"  {sid} edited:", f"    - {a['text']}", f"    + {b['text']}"]
                if a["cites"] != b["cites"]:
                    lines.append(f"  {sid} cites: {a['cites']} -> {b['cites']}")
            continue
        later = len(old_s) - i2  # sentences after this change, whose IDs move
        where = old_s[i1]["sentence_id"] if i1 < len(old_s) else "the end"
        lines.append(f"  sentences change at {where}: {i2 - i1} became {j2 - j1}; "
                     f"{later} later sentence(s) get new IDs")
        lines += [f"    - {s['sentence_id']} {s['text']}" for s in old_s[i1:i2]]
        lines += [f"    + {s['sentence_id']} {s['text']}" for s in new_s[j1:j2]]
        ids_shift = ids_shift or later > 0 or i2 > i1  # a removed or replaced sentence loses its ID too

    old_dois = [r["doi"] for r in old["references"]]
    new_dois = [r["doi"] for r in new["references"]]
    if old_dois != new_dois:
        lines.append(f"  references: {len(old_dois)} -> {len(new_dois)}, "
                     f"with a DOI: {sum(map(bool, old_dois))} -> {sum(map(bool, new_dois))}")
    return lines, ids_shift


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    args = parser.parse_args(argv)

    old, new = load(args.old), load(args.new)
    unchanged, edited_only, shifted = [], [], []
    for pmcid in old.keys() - new.keys():
        print(f"{pmcid}: only in {args.old}")
    for pmcid in new.keys() - old.keys():
        print(f"{pmcid}: only in {args.new}")
    for pmcid in [p for p in old if p in new]:
        lines, ids_shift = diff_paper(old[pmcid], new[pmcid])
        if not lines:
            unchanged.append(pmcid)
            continue
        (shifted if ids_shift else edited_only).append(pmcid)
        print(f"{pmcid}:\n" + "\n".join(lines))

    print(f"\n{len(unchanged)} unchanged, {len(edited_only)} changed with the same IDs, "
          f"{len(shifted)} with shifted IDs" + (f": {', '.join(shifted)}" if shifted else ""))


if __name__ == "__main__":
    main()