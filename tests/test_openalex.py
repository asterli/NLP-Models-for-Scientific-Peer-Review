# Tests for src/corpus/openalex.py. Run with:  pytest tests/test_openalex.py
# No internet needed: requests.get is swapped for a fake that returns canned responses
# and records what was asked (same setup as test_europepmc.py).

import pytest
import requests

from corpus import openalex


class FakeResponse:
    def __init__(self, status=200, json_data=None):
        self.status_code, self._json = status, json_data

    def json(self):
        return self._json


@pytest.fixture
def fake_api(monkeypatch):
    # Replace requests.get and time.sleep. Tests fill `responses`; `calls` records (url, params).
    state = {"responses": [], "calls": []}

    def fake_get(url, params=None, **kwargs):
        state["calls"].append((url, params))
        response = state["responses"].pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(openalex.requests, "get", fake_get)
    monkeypatch.setattr(openalex.time, "sleep", lambda seconds: None)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    return state


WORK = {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/x", "title": "A paper"}


def test_lookup_by_doi_is_cached(fake_api, tmp_path):
    fake_api["responses"] = [FakeResponse(json_data=WORK)]
    assert openalex.get_work("doi:10.1/x", tmp_path) == WORK
    assert fake_api["calls"][0][0] == "https://api.openalex.org/works/doi:10.1/x"
    assert openalex.get_work("doi:10.1/x", tmp_path) == WORK  # second time: from disk
    assert len(fake_api["calls"]) == 1


def test_pmid_and_openalex_id_keys(fake_api, tmp_path):
    fake_api["responses"] = [FakeResponse(json_data=WORK), FakeResponse(json_data=WORK)]
    openalex.get_work("pmid:123", tmp_path)
    openalex.get_work("W2741809807", tmp_path)
    assert [url for url, _ in fake_api["calls"]] == ["https://api.openalex.org/works/pmid:123",
                                                     "https://api.openalex.org/works/W2741809807"]


def test_odd_doi_characters_are_safe_in_the_url_and_file_name(fake_api, tmp_path):
    fake_api["responses"] = [FakeResponse(json_data=WORK)]
    openalex.get_work("doi:10.1002/(sici)1097#x?y", tmp_path)
    url = fake_api["calls"][0][0]
    assert "#" not in url and "?" not in url  # would cut the URL short
    assert [p.name for p in tmp_path.iterdir()] == ["doi%3A10.1002%2F%28sici%291097%23x%3Fy.json"]


def test_not_found_is_cached_as_none(fake_api, tmp_path):
    fake_api["responses"] = [FakeResponse(status=404)]
    assert openalex.get_work("doi:10.1/missing", tmp_path) is None
    assert openalex.get_work("doi:10.1/missing", tmp_path) is None  # from disk, no new request
    assert len(fake_api["calls"]) == 1


def test_bad_request_is_not_cached(fake_api, tmp_path):
    fake_api["responses"] = [FakeResponse(status=400)]
    assert openalex.get_work("doi:10.1/broken", tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_retries_when_told_to_slow_down(fake_api, tmp_path):
    fake_api["responses"] = [FakeResponse(status=429), FakeResponse(status=503), FakeResponse(json_data=WORK)]
    assert openalex.get_work("W1", tmp_path) == WORK
    assert len(fake_api["calls"]) == 3


def test_server_error_after_all_retries_stops(fake_api, tmp_path):
    fake_api["responses"] = [FakeResponse(status=503) for _ in range(openalex.RETRIES + 1)]
    with pytest.raises(openalex.OpenAlexError, match="W1: OpenAlex returned HTTP 503"):
        openalex.get_work("W1", tmp_path)
    assert list(tmp_path.iterdir()) == []  # nothing bad gets cached


def test_api_key_is_sent_but_never_shown_in_errors(fake_api, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "secret123")
    fake_api["responses"] = [FakeResponse(status=403)]
    with pytest.raises(openalex.OpenAlexError) as error:
        openalex.get_work("W1", tmp_path)
    assert fake_api["calls"][0][1] == {"api_key": "secret123"}
    assert "secret123" not in str(error.value)


def test_network_error_message_hides_the_url(fake_api, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "secret123")
    fake_api["responses"] = [requests.ConnectionError("Max retries exceeded with url: /works/W1?api_key=secret123")]
    with pytest.raises(openalex.OpenAlexError) as error:
        openalex.get_work("W1", tmp_path)
    assert "secret123" not in str(error.value) and error.value.__cause__ is None


def test_no_key_means_no_key_parameter(fake_api, tmp_path):
    fake_api["responses"] = [FakeResponse(json_data=WORK)]
    openalex.get_work("W1", tmp_path)
    assert fake_api["calls"][0][1] == {}


@pytest.mark.parametrize("bad", ["10.1/x", "doi:abc", "PMC123", "w1", "pmid:12a"])
def test_bad_keys_are_rejected(bad, tmp_path):
    with pytest.raises(ValueError, match="not a lookup key"):
        openalex.get_work(bad, tmp_path)