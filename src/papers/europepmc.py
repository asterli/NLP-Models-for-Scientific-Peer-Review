# This file is looks online at Europe PMC to search for papers, and downloadsd full-text XMLs, and fetches metadata.

# Should be only file in src/papers that uses the internet. Downloads are cached on the disk
# so rerunning this will give same data even if the website updates a paper later.

# Referenced: API docs: https://europepmc.org/RestfulWebService 

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
HEADERS = {"User-Agent": "csci544-claim-novelty-project (course project)"}
PAUSE_SECONDS = 0.5  # be nice boi and wait between requests
RETRIES = 3          # for "slow down" (429) and server errors (5xx)


@dataclass
class Candidate:
    # One search result, shown in candidates.csv so a person can pick papers.
    pmcid: str | None
    doi: str | None
    date: str | None
    journal: str | None
    pub_type: str | None
    cited_by: int | None
    title: str | None


@dataclass
class PaperMeta:
    date: str | None  # first publication date, YYYY-MM-DD
    authors: list[str]
    title: str | None
    doi: str | None
    journal: str | None


class FullTextUnavailable(Exception):
    """ IF Europe PMC has no full-text XML for this paper (usually: not open access)."""


def search(query: str, max_results: int = 200) -> list[Candidate]:
    # Run a Europe PMC search, following pages until max_results or no more results.
    candidates, cursor = [], "*"
    while len(candidates) < max_results:
        page = _get_json("/search", {"query": query, "format": "json", "resultType": "core",
                                     "pageSize": min(100, max_results - len(candidates)),
                                     "cursorMark": cursor})
        results = page.get("resultList", {}).get("result", [])
        candidates += [_to_candidate(r) for r in results]
        next_cursor = page.get("nextCursorMark")
        if not results or not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
    return candidates[:max_results]


def download_full_text(pmcid: str, cache_dir: Path) -> bytes:
    # A paper's full-text XML, from the cache if we've downloaded it before.
    pmcid = normalize_pmcid(pmcid)
    path = Path(cache_dir) / f"{pmcid}.xml"
    if path.exists():
        return path.read_bytes()
    response = _get(f"/{pmcid}/fullTextXML", {})
    if response.status_code == 429 or response.status_code >= 500:
        response.raise_for_status()  # server still failing after retries: stop, don't skip the paper
    if response.status_code != 200 or not response.content.lstrip().startswith(b"<"):
        raise FullTextUnavailable(f"{pmcid}: no full text (HTTP {response.status_code})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return response.content


def fetch_metadata(pmcid: str, cache_dir: Path) -> PaperMeta:
    # Publication date, authors, and other details, from the cache if available.
    pmcid = normalize_pmcid(pmcid)
    path = Path(cache_dir) / f"{pmcid}.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
    else:
        page = _get_json("/search", {"query": f"PMCID:{pmcid}", "format": "json", "resultType": "core"})
        results = page.get("resultList", {}).get("result", [])
        record = results[0] if results else {}
        if record:  # don't cache an empty result, so the next run asks again
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return PaperMeta(
        date=record.get("firstPublicationDate"),
        authors=[a["fullName"] for a in record.get("authorList", {}).get("author", []) if a.get("fullName")],
        title=record.get("title"),
        doi=record.get("doi"),
        journal=_journal(record),
    )


def normalize_pmcid(pmcid: str) -> str:
    # 'pmc123' or ' PMC123 ' -> 'PMC123'. Bare numbers are rejected because they are usually a PMID (learned the hard way this is different from PMC), 
    # a different ID that Europe PMC also shows, and it won't work for downloading full text.
    cleaned = pmcid.strip().upper()
    if re.fullmatch(r"\d+", cleaned):
        raise ValueError(f"{pmcid!r} looks like a PMID. Use the PMCID instead (it starts with 'PMC').")
    if not re.fullmatch(r"PMC\d+", cleaned):
        raise ValueError(f"not a PMCID: {pmcid!r}")
    return cleaned


# --- helpers -----------------------------------------------------------------

def _get(path: str, params: dict) -> requests.Response:
    # GET with a pause after each request and retries (waiting 1s, 2s, 4s) on 429 or 5xx.
    for attempt in range(RETRIES + 1):
        response = requests.get(API + path, params=params, headers=HEADERS, timeout=60)
        time.sleep(PAUSE_SECONDS)
        if response.status_code != 429 and response.status_code < 500:
            return response
        if attempt < RETRIES:
            time.sleep(2 ** attempt)
    return response


def _get_json(path: str, params: dict) -> dict:
    response = _get(path, params)
    response.raise_for_status()
    return response.json()


def _to_candidate(r: dict) -> Candidate:
    types = r.get("pubTypeList", {}).get("pubType", [])
    return Candidate(
        pmcid=r.get("pmcid"),
        doi=r.get("doi"),
        date=r.get("firstPublicationDate"),
        journal=_journal(r),
        pub_type="; ".join(types) if isinstance(types, list) else types,
        cited_by=r.get("citedByCount"),
        title=r.get("title"),
    )


def _journal(r: dict) -> str | None:
    return r.get("journalInfo", {}).get("journal", {}).get("title") or r.get("journalTitle")
