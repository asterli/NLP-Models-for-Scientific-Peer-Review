# Looks up older papers in OpenAlex for the background corpus (ticket M21).

# Should be the only file in src/corpus that uses the internet. Every answer is cached on disk
# as one JSON file, so rebuilding the corpus later gives the same data and makes no requests.

# Looking up one work by DOI, PMID, or OpenAlex ID is free and unlimited (as of 2026).
# An API key is optional for these lookups; set OPENALEX_API_KEY in .env if you have one.

# Referenced: API docs: https://developers.openalex.org

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests

API = "https://api.openalex.org"
HEADERS = {"User-Agent": "csci544-claim-novelty-project (course project)"}
PAUSE_SECONDS = 0.1  # the API allows up to 100 requests per second; we stay far below that
RETRIES = 3          # for "slow down" (429) and server errors (5xx)

# A lookup key: "doi:10.1234/abc", "pmid:12345", or an OpenAlex work ID like "W2741809807".
_KEY = re.compile(r"doi:10\.\S+|pmid:\d+|W\d+")


class OpenAlexError(Exception):
    """OpenAlex kept failing (after retries) or refused the request. The message never includes
    the request URL, because that URL would contain the API key."""


def get_work(key: str, cache_dir: Path) -> dict | None:
    # One OpenAlex work, or None if OpenAlex doesn't have it. Uses the cache when it can.
    if not _KEY.fullmatch(key):
        raise ValueError(f"not a lookup key: {key!r} (expected doi:..., pmid:..., or W...)")
    path = Path(cache_dir) / f"{quote(key, safe='')}.json"  # "doi:10.1/x" -> "doi%3A10.1%2Fx.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))  # null in the file means "not found"

    response = _get(f"/works/{quote(key, safe=':/')}")
    if response.status_code == 400:
        return None  # malformed key (e.g. a broken DOI); not cached, in case the bug is ours
    if response.status_code not in (200, 404):
        # 429/5xx after all retries, or 401/403 (bad API key): stop instead of guessing
        raise OpenAlexError(f"{key}: OpenAlex returned HTTP {response.status_code}")
    work = response.json() if response.status_code == 200 else None  # 404: OpenAlex doesn't have it

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(work, ensure_ascii=False), encoding="utf-8")
    return work


# --- helpers -----------------------------------------------------------------

def _get(path: str) -> requests.Response:
    # GET with a pause after each request and retries (waiting 1s, 2s, 4s) on 429 or 5xx.
    params = {"api_key": os.environ["OPENALEX_API_KEY"]} if os.environ.get("OPENALEX_API_KEY") else {}
    for attempt in range(RETRIES + 1):
        try:
            response = requests.get(API + path, params=params, headers=HEADERS, timeout=60)
        except requests.RequestException as error:
            # requests' own message includes the URL (and so the key): report only the error type
            raise OpenAlexError(f"network error on {path}: {type(error).__name__}") from None
        time.sleep(PAUSE_SECONDS)
        if response.status_code != 429 and response.status_code < 500:
            return response
        if attempt < RETRIES:
            time.sleep(2 ** attempt)
    return response