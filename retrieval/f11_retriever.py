"""F11 - Callable prior-work retriever for the pipeline (F12).

Thin wrapper around the F10 FilteredRetriever (m23_filters.py). Loads the corpus once and returns
candidates in the shared Retrieval Candidate schema (docs/data-schema.md, section 3). This is the
same function scripts/f11_eval_retrieval.py scores, so the F11 Recall@10 describes it directly.

  from f11_retriever import PriorWorkRetriever
  R = PriorWorkRetriever("data/corpus/corpus.jsonl")
  candidates = R.retrieve(claim_text, submitted_paper, k=10)

submitted_paper: a dict with any of doi, pmid, pmcid, openalex_id, paper_id, title, and a
date (YYYY-MM-DD), or an F03 split row (newer_paper_id, newer_pub_date, ...).

Each candidate, best first:
  schema fields  paper_id, title, score, date, url, evidence
  extras         rank, doi, pmid, evidence_type, eligibility, flags, duplicate_ids
Title-only papers get evidence None and evidence_type "title_only"; no passage is made up.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m23_filters import FilteredRetriever, submitted_from_row  # noqa: E402

# Order matters only for readability: shared schema fields first, then extras.
FIELDS = ("rank", "paper_id", "title", "score", "date", "url", "evidence",
          "doi", "pmid", "evidence_type", "eligibility", "flags", "duplicate_ids")


def as_submitted(paper: dict | None) -> dict:
    # Accept a plain paper dict or an F03 split row, and return what F10 expects.
    paper = paper or {}
    if any(key.startswith("newer_") for key in paper):
        paper = submitted_from_row(paper)
    if not paper.get("date"):
        # Without a date F10 can't tell which works are older, so every result would be dropped.
        raise ValueError("submitted paper needs a 'date' (YYYY-MM-DD) to filter out newer work")
    return paper


class PriorWorkRetriever:
    def __init__(self, corpus_path: str, uncertain: str = "exclude", **kw):
        # uncertain: "exclude" (F10 default) drops works whose date can't confirm they are older.
        self.filtered = FilteredRetriever(corpus_path, uncertain=uncertain, **kw)

    def retrieve(self, claim_text: str, submitted_paper: dict | None, k: int = 10, explain: bool = False):
        """Top k eligible prior works for one claim. With explain=True, also return F10's exclusion counts."""
        hits, excluded = self.filtered.search(claim_text, as_submitted(submitted_paper), k=k, explain=True)
        candidates = [{field: hit.get(field) for field in FIELDS} for hit in hits]
        return (candidates, excluded) if explain else candidates