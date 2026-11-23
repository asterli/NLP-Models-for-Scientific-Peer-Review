"""Load papers in the shared Paper schema (docs/data-schema.md, section 1).

Three sources:
  papers_by_id   records from data/papers/papers.jsonl (sentence IDs are the frozen M05 ones)
  papers_from_file  a JSON Paper record, a JSON list of them, or a JSONL file of them
  paper_from_text   plain text: paragraphs separated by blank lines, split with the M05 sentence splitter.
                    A paragraph that is just "Abstract" or "Introduction" (optionally after #) sets the section
                    of the paragraphs after it; text before any heading is "abstract".
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REQUIRED = ("paper_id", "title", "date", "sentences")
_HEADING = re.compile(r"#*\s*(abstract|introduction)\s*:?", re.I)


def check_paper(paper: dict, where: str = "") -> dict:
    for field in REQUIRED:
        if not paper.get(field):
            # date is required: without it, retrieval cannot tell which works are older (F10)
            raise ValueError(f"{where}paper {paper.get('paper_id', '?')} is missing `{field}`")
    return paper


def _read_records(path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        doc = json.loads(text)
        return doc if isinstance(doc, list) else [doc]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def papers_by_id(papers_path, ids: list[str]) -> list[dict]:
    available = {p["paper_id"]: p for p in _read_records(papers_path)}
    missing = [i for i in ids if i not in available]
    if missing:
        raise KeyError(f"not in {papers_path}: {', '.join(missing)}")
    return [check_paper(available[i]) for i in ids]


def papers_from_file(path) -> list[dict]:
    return [check_paper(p, f"{path}: ") for p in _read_records(path)]


def paper_from_text(path, title: str, date: str, doi: str | None = None, url: str | None = None) -> dict:
    from papers.jats import Paragraph
    from papers.sentences import split_sentences

    raw = Path(path).read_text(encoding="utf-8-sig")
    section, sentences = "abstract", []
    for block in re.split(r"\n\s*\n", raw):
        block = " ".join(block.split())
        if not block:
            continue
        heading = _HEADING.fullmatch(block)
        if heading:
            section = heading.group(1).lower()
            continue
        for s in split_sentences(Paragraph(section=section, text=block)):
            sentences.append({"sentence_id": f"s{len(sentences) + 1:03d}", "section": section, "text": s.text,
                              "cites": []})
    doi = doi.lower().removeprefix("https://doi.org/") if doi else None
    paper_id = f"doi:{doi}" if doi else "text:" + hashlib.sha256(raw.encode()).hexdigest()[:12]
    paper = {"paper_id": paper_id, "title": title, "date": date, "doi": doi,
             "url": url or (f"https://doi.org/{doi}" if doi else None), "sentences": sentences,
             "source": f"text file {Path(path).name}"}
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date or ""):
        raise ValueError("--date must be YYYY-MM-DD")
    return check_paper(paper)
