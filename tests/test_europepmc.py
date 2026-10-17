# Tests for src/papers/europepmc.py. Run with:  pytest tests/test_europepmc.py
# unlike europepmc.py, this test does not need internet.
# each test will swap requests.get for a fake that returns canned responses and records what was asked

import pytest

from papers import europepmc


class FakeResponse:
    def __init__(self, status=200, json_data=None, content=b""):
        self.status_code, self._json, self.content = status, json_data, content

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture
def fake_api(monkeypatch):
    # Replace requests.get and time.sleep. Tests fill `responses`; `calls` records URLs hit."""
    state = {"responses": [], "calls": []}

    def fake_get(url, params=None, **kwargs):
        state["calls"].append((url, params))
        return state["responses"].pop(0)

    monkeypatch.setattr(europepmc.requests, "get", fake_get)
    monkeypatch.setattr(europepmc.time, "sleep", lambda seconds: None)
    return state


def result(pmcid, **extra):
    return {"pmcid": pmcid, "doi": f"10.1/{pmcid}", "title": f"Paper {pmcid}", **extra}


# --- search ------------------------------------------------------------------

def test_search_follows_pages_until_results_run_out(fake_api):
    fake_api["responses"] = [
        FakeResponse(json_data={"resultList": {"result": [result("PMC1"), result("PMC2")]}, "nextCursorMark": "abc"}),
        FakeResponse(json_data={"resultList": {"result": [result("PMC3")]}, "nextCursorMark": "abc"}),  # same cursor: last page
    ]
    found = europepmc.search("housing", max_results=10)
    assert [c.pmcid for c in found] == ["PMC1", "PMC2", "PMC3"]
    assert fake_api["calls"][1][1]["cursorMark"] == "abc"


def test_search_stops_at_max_results(fake_api):
    fake_api["responses"] = [FakeResponse(json_data={"resultList": {"result": [result("PMC1"), result("PMC2")]},
                                                     "nextCursorMark": "next"})]
    assert len(europepmc.search("housing", max_results=2)) == 2
    assert len(fake_api["calls"]) == 1  # didn't ask for a second page


def test_candidate_fields(fake_api):
    fake_api["responses"] = [FakeResponse(json_data={"resultList": {"result": [result(
        "PMC1", firstPublicationDate="2024-03-05", citedByCount=7,
        journalInfo={"journal": {"title": "BMC Public Health"}},
        pubTypeList={"pubType": ["research-article", "Journal Article"]})]}})]
    c = europepmc.search("housing", max_results=5)[0]
    assert (c.date, c.journal, c.cited_by, c.pub_type) == ("2024-03-05", "BMC Public Health", 7,
                                                           "research-article; Journal Article")


# --- full text ---------------------------------------------------------------

def test_download_saves_to_cache_and_reuses_it(fake_api, tmp_path):
    fake_api["responses"] = [FakeResponse(content=b"<article/>")]
    assert europepmc.download_full_text("pmc123", tmp_path) == b"<article/>"
    assert (tmp_path / "PMC123.xml").exists()
    assert europepmc.download_full_text("PMC123", tmp_path) == b"<article/>"  # second time: from disk
    assert len(fake_api["calls"]) == 1


def test_missing_full_text_raises_clear_error(fake_api, tmp_path):
    fake_api["responses"] = [FakeResponse(status=404)]
    with pytest.raises(europepmc.FullTextUnavailable, match="PMC999"):
        europepmc.download_full_text("PMC999", tmp_path)
    assert not (tmp_path / "PMC999.xml").exists()  # nothing bad gets cached


def test_server_error_after_all_retries_stops_instead_of_skipping(fake_api, tmp_path):
    fake_api["responses"] = [FakeResponse(status=503) for _ in range(europepmc.RETRIES + 1)]
    with pytest.raises(RuntimeError, match="503"):  # not FullTextUnavailable, which would skip it
        europepmc.download_full_text("PMC1", tmp_path)


def test_retries_when_told_to_slow_down(fake_api, tmp_path):
    fake_api["responses"] = [FakeResponse(status=429), FakeResponse(status=503), FakeResponse(content=b"<article/>")]
    assert europepmc.download_full_text("PMC1", tmp_path) == b"<article/>"
    assert len(fake_api["calls"]) == 3


# --- metadata ----------------------------------------------------------------

def test_metadata_is_parsed_and_cached(fake_api, tmp_path):
    fake_api["responses"] = [FakeResponse(json_data={"resultList": {"result": [result(
        "PMC1", firstPublicationDate="2024-03-05",
        authorList={"author": [{"fullName": "Doe J"}, {"lastName": "NoFullName"}]})]}})]
    meta = europepmc.fetch_metadata("PMC1", tmp_path)
    assert (meta.date, meta.authors, meta.doi) == ("2024-03-05", ["Doe J"], "10.1/PMC1")
    assert europepmc.fetch_metadata("PMC1", tmp_path) == meta  # second time: from disk
    assert len(fake_api["calls"]) == 1


def test_empty_metadata_result_is_not_cached(fake_api, tmp_path):
    fake_api["responses"] = [FakeResponse(json_data={"resultList": {"result": []}})]
    assert europepmc.fetch_metadata("PMC1", tmp_path).date is None
    assert not (tmp_path / "PMC1.json").exists()

    
# --- PMCIDs ------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [("PMC123", "PMC123"), (" pmc123 ", "PMC123")])
def test_normalize_pmcid(raw, expected):
    assert europepmc.normalize_pmcid(raw) == expected


def test_bare_number_is_rejected_as_a_likely_pmid():
    with pytest.raises(ValueError, match="looks like a PMID"):
        europepmc.normalize_pmcid("42827296")


def test_junk_is_rejected():
    with pytest.raises(ValueError, match="not a PMCID"):
        europepmc.normalize_pmcid("MED/42827296")
