# Tests for src/corpus/records.py. Run with:  pytest tests/test_corpus_records.py
# tests/fixtures/openalex/tsemberis_2004.json is a real OpenAlex work (Tsemberis et al. 2004,
# the Housing First trial), trimmed to the fields records.py reads.

import json
from pathlib import Path

import pytest

from corpus.records import corpus_record, field_coverage, lookup_key, openalex_id, rebuild_abstract

FIXTURES = Path(__file__).parent / "fixtures" / "openalex"


def real_work():
    return json.loads((FIXTURES / "tsemberis_2004.json").read_text(encoding="utf-8"))


# --- corpus_record -----------------------------------------------------------

def test_real_work_becomes_a_corpus_record():
    record = corpus_record(real_work(), "reference")
    assert record["paper_id"] == "doi:10.2105/ajph.94.4.651"  # same format as papers.jsonl
    assert record["title"].startswith("Housing First, Consumer Choice, and Harm Reduction")
    assert (record["date"], record["year"]) == ("2004-04-01", 2004)
    assert record["url"] == "https://doi.org/10.2105/ajph.94.4.651"
    assert record["venue"] == "American Journal of Public Health"
    assert (record["pmid"], record["openalex_id"], record["type"]) == ("15054020", "W2132295262", "article")
    assert (record["source"], record["cited_by"]) == ("reference", [])


def test_real_abstract_is_rebuilt_in_reading_order():
    abstract = corpus_record(real_work(), "reference")["abstract"]
    assert abstract.startswith("OBJECTIVES: We examined the longitudinal effects of a Housing First program")
    assert abstract.endswith("without compromising psychiatric or substance abuse symptoms.")


def test_work_without_doi_uses_openalex_id_and_landing_page():
    work = real_work()
    work["doi"] = None
    work["primary_location"]["landing_page_url"] = "https://example.org/report"
    record = corpus_record(work, "search")
    assert record["paper_id"] == "openalex:W2132295262"
    assert (record["doi"], record["url"]) == (None, "https://example.org/report")


def test_missing_fields_become_none():
    work = {"id": "https://openalex.org/W9", "title": None, "display_name": "Only a display name"}
    record = corpus_record(work, "search")
    assert record["title"] == "Only a display name"
    assert record["url"] == "https://openalex.org/W9"
    assert [record[f] for f in ("abstract", "date", "venue", "doi", "pmid")] == [None] * 5


def test_record_has_schema_fields_in_order():
    keys = list(corpus_record(real_work(), "reference"))
    assert keys[:5] == ["paper_id", "title", "date", "url", "abstract"]


# --- rebuild_abstract --------------------------------------------------------

def test_abstract_formatting_tags_removed_but_less_than_kept():
    index = {"<p>Rates": [0], "fell": [1], "(p<0.05": [2], "vs": [3], ">": [4], "baseline).</p>": [5]}
    assert rebuild_abstract(index) == "Rates fell (p<0.05 vs > baseline)."


@pytest.mark.parametrize("empty", [None, {}])
def test_no_abstract(empty):
    assert rebuild_abstract(empty) is None


# --- lookup_key and openalex_id ----------------------------------------------

@pytest.mark.parametrize("reference, expected", [
    ({"doi": "10.2105/ajph.94.4.651", "pmid": "15054020"}, "doi:10.2105/ajph.94.4.651"),  # DOI preferred
    ({"doi": None, "pmid": "15054020"}, "pmid:15054020"),
    ({"doi": "10.1/has space", "pmid": None}, None),  # broken DOI from the reference text
    ({"doi": None, "pmid": None}, None),               # e.g. a report or website
])
def test_lookup_key(reference, expected):
    assert lookup_key(reference) == expected


def test_openalex_id():
    assert openalex_id("https://openalex.org/W2132295262") == "W2132295262"
    assert openalex_id("W1") == "W1"
    with pytest.raises(ValueError):
        openalex_id("https://openalex.org/A5044439622")  # an author, not a work


# --- field_coverage ----------------------------------------------------------

def test_field_coverage_counts_present_fields():
    full = corpus_record(real_work(), "reference")
    empty = corpus_record({"id": "W9"}, "search")
    assert field_coverage([full, empty]) == {"works": 2, "abstract": 1, "date": 1, "doi": 1, "venue": 1}