# Converts OpenAlex works into corpus records (docs/data-schema.md, "Corpus document"),
# plus the small helpers the corpus builder needs to go between papers.jsonl and OpenAlex.

# Everything works on data that openalex.py already downloaded.

from __future__ import annotations

import re

from papers.jats import normalize_doi

# Formatting tags that some publishers leave inside abstracts. Only these are removed, so text
# like "p<0.05 ... >" is never mistaken for a tag.
_ABSTRACT_TAGS = re.compile(r"</?(?:p|jats:p|i|b|em|strong|sup|sub|italic|bold|br)\b[^>]*>", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_OPENALEX_WORK = re.compile(r"(?:https://openalex\.org/)?(W\d+)")


def corpus_record(work: dict, source: str) -> dict:
    # One OpenAlex work is one line of corpus.jsonl. `source` is "reference" or "search".
    doi = normalize_doi(work.get("doi"))
    work_id = openalex_id(work["id"])
    location = work.get("primary_location") or {}
    pmid = (work.get("ids") or {}).get("pmid")  # a URL like "https://pubmed.ncbi.nlm.nih.gov/15054020"
    return {
        "paper_id": f"doi:{doi}" if doi else f"openalex:{work_id}",  # same format as papers.jsonl
        "title": work.get("title") or work.get("display_name"),
        "date": work.get("publication_date"),
        "url": f"https://doi.org/{doi}" if doi else (location.get("landing_page_url") or work["id"]),
        "abstract": rebuild_abstract(work.get("abstract_inverted_index")),
        "venue": (location.get("source") or {}).get("display_name"),
        "year": work.get("publication_year"),
        "type": work.get("type"),
        "doi": doi,
        "pmid": pmid.rstrip("/").rsplit("/", 1)[-1] if pmid else None,
        "openalex_id": work_id,
        "source": source,
        "cited_by": [],  # paper_ids of the source papers that cite this work; filled in by the builder
    }


def rebuild_abstract(inverted_index: dict | None) -> str | None:
    # OpenAlex stores abstracts as {word: [positions]}. Put the words back in reading order.
    if not inverted_index:
        return None
    positions = {pos: word for word, places in inverted_index.items() for pos in places}
    text = " ".join(positions[pos] for pos in sorted(positions))
    text = _WHITESPACE.sub(" ", _ABSTRACT_TAGS.sub(" ", text)).strip()
    return text or None


def lookup_key(reference: dict) -> str | None:
    # A papers.jsonl reference -> the key openalex.get_work() expects, or None if it has no usable ID.
    doi, pmid = reference.get("doi"), reference.get("pmid")
    if doi and re.fullmatch(r"10\.\S+", doi):
        return f"doi:{doi}"
    if pmid and pmid.isdigit():
        return f"pmid:{pmid}"
    return None


def openalex_id(url: str) -> str:
    # 'https://openalex.org/W2132295262' -> 'W2132295262'
    match = _OPENALEX_WORK.fullmatch(url.strip())
    if not match:
        raise ValueError(f"not an OpenAlex work ID: {url!r}")
    return match.group(1)


def field_coverage(records: list[dict]) -> dict[str, int]:
    # How many records have each field that retrieval and date filtering depend on.
    fields = ("abstract", "date", "doi", "venue")
    return {"works": len(records), **{f: sum(1 for r in records if r.get(f)) for f in fields}}