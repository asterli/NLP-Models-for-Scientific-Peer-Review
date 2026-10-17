# Tests for scripts/build_corpus.py. Run with:  pytest tests/test_build_corpus.py
# No internet needed: openalex.get_work is swapped for a fake that answers from a dict.

import json

import pytest

import build_corpus
from corpus import openalex


def work(work_id, doi=None, refs=()):
    return {"id": f"https://openalex.org/{work_id}", "doi": f"https://doi.org/{doi}" if doi else None,
            "title": f"Title {work_id}", "publication_date": "2010-01-01",
            "referenced_works": [f"https://openalex.org/{r}" for r in refs]}


# What the fake OpenAlex knows. Source paper A cites: a DOI it has, a PMID it has, a DOI it
# doesn't have, and a report with no ID. A's OpenAlex reference list adds W4 (no DOI in our XML)
# and repeats W1. Source paper B cites W1 too.
FAKE_OPENALEX = {
    "doi:10.1/a": work("WA", "10.1/a", refs=["W1", "W4"]),
    "doi:10.1/one": work("W1", "10.1/one"),
    "pmid:222": work("W2", "10.1/two"),
    "W4": work("W4"),
    "doi:10.1/b": work("WB", "10.1/b"),
}
PAPERS = [
    {"paper_id": "doi:10.1/a", "pmcid": "PMC1", "doi": "10.1/a", "references": [
        {"ref_id": "R1", "doi": "10.1/one", "pmid": None},
        {"ref_id": "R2", "doi": None, "pmid": "222"},
        {"ref_id": "R3", "doi": "10.1/missing", "pmid": None},
        {"ref_id": "R4", "doi": None, "pmid": None},
    ]},
    {"paper_id": "doi:10.1/b", "pmcid": "PMC2", "doi": "10.1/b", "references": [
        {"ref_id": "B1", "doi": "10.1/one", "pmid": None},
    ]},
]


@pytest.fixture
def fake_openalex(monkeypatch):
    asked = []

    def fake_get_work(key, cache_dir):
        asked.append(key)
        return FAKE_OPENALEX.get(key)

    monkeypatch.setattr(openalex, "get_work", fake_get_work)
    return asked


def run(tmp_path, *extra):
    papers = tmp_path / "papers.jsonl"
    papers.write_text("".join(json.dumps(p) + "\n" for p in PAPERS))
    out = tmp_path / "corpus"
    build_corpus.main(["--papers", str(papers), "--out", str(out), "--cache", str(tmp_path / "cache"), *extra])
    read = lambda name: [json.loads(line) for line in (out / name).read_text().splitlines()]
    return read("corpus.jsonl"), read("citation_links.jsonl"), json.loads((out / "coverage.json").read_text())


def test_each_work_appears_once_with_everyone_who_cites_it(fake_openalex, tmp_path):
    corpus, _, _ = run(tmp_path)
    assert [r["paper_id"] for r in corpus] == ["doi:10.1/one", "doi:10.1/two", "openalex:W4"]
    assert corpus[0]["cited_by"] == ["doi:10.1/a", "doi:10.1/b"]


def test_citation_links_point_references_at_corpus_works(fake_openalex, tmp_path):
    _, links, _ = run(tmp_path)
    assert links == [
        {"paper_id": "doi:10.1/a", "ref_id": "R1", "corpus_id": "doi:10.1/one", "matched_by": "doi"},
        {"paper_id": "doi:10.1/a", "ref_id": "R2", "corpus_id": "doi:10.1/two", "matched_by": "pmid"},
        {"paper_id": "doi:10.1/b", "ref_id": "B1", "corpus_id": "doi:10.1/one", "matched_by": "doi"},
    ]


def test_coverage_counts_what_was_found_and_missing(fake_openalex, tmp_path):
    _, _, coverage = run(tmp_path)
    a = coverage["papers"][0]
    assert (a["references"], a["linked"], a["not_in_openalex"], a["no_id"]) == (4, 2, 1, 1)
    assert (a["added_from_openalex_list"], a["source_in_openalex"]) == (1, True)
    assert coverage["summary"] == {"works": 3, "abstract": 0, "date": 3, "doi": 2, "venue": 0, "links": 3}


def test_works_already_found_are_not_fetched_again(fake_openalex, tmp_path):
    run(tmp_path)
    assert "W1" not in fake_openalex  # W1 was already found by its DOI
    assert fake_openalex.count("doi:10.1/one") == 2  # once per citing paper; the second is a cache hit in real use


def test_refuses_to_overwrite_without_flag(fake_openalex, tmp_path):
    run(tmp_path)
    with pytest.raises(SystemExit, match="already has corpus.jsonl"):
        run(tmp_path)
    run(tmp_path, "--overwrite")


def test_openalex_error_stops_and_writes_nothing(monkeypatch, tmp_path):
    def failing(key, cache_dir):
        raise openalex.OpenAlexError(f"{key}: OpenAlex returned HTTP 503")

    monkeypatch.setattr(openalex, "get_work", failing)
    with pytest.raises(SystemExit, match="run again to continue"):
        run(tmp_path)
    assert not (tmp_path / "corpus").exists()