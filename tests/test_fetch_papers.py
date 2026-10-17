# Tests for scripts/fetch_papers.py. Run with:  pytest tests/test_fetch_papers.py

# No internet needed: the test puts a paper in the cache folder first, so the fetch finds it there.

import json
import shutil
from pathlib import Path

import pytest

import fetch_papers

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def workspace(tmp_path):
    # A cache that already holds one paper, and a PMCID list pointing at it.
    (tmp_path / "xml").mkdir()
    (tmp_path / "meta").mkdir()
    shutil.copy(FIXTURES / "research_superscript.xml", tmp_path / "xml" / "PMC0000001.xml")
    (tmp_path / "meta" / "PMC0000001.json").write_text(json.dumps({"firstPublicationDate": "2024-03-05"}))
    ids = tmp_path / "selected.txt"
    ids.write_text("# picked papers\nPMC0000001   # cohort study, Housing First\npmc0000001\n")
    return tmp_path, ids


def run_fetch(tmp_path, ids, *extra):
    out = tmp_path / "papers.jsonl"
    fetch_papers.main(["fetch", "--ids", str(ids), "--cache", str(tmp_path), "--out", str(out), *extra])
    return out


def test_fetch_writes_one_record_per_paper(workspace, capsys):
    tmp_path, ids = workspace
    out = run_fetch(tmp_path, ids)
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == 1  # the duplicate line was dropped
    assert records[0]["paper_id"] == "doi:10.1234/jhh.2024.001"
    printed = capsys.readouterr().out
    assert "wrote 1 papers" in printed and "check these before labeling" in printed


def test_fetch_refuses_to_overwrite_without_flag(workspace):
    tmp_path, ids = workspace
    run_fetch(tmp_path, ids)
    with pytest.raises(SystemExit, match="already exists"):
        run_fetch(tmp_path, ids)
    run_fetch(tmp_path, ids, "--overwrite")  # allowed when asked for


def test_pmid_in_the_list_is_reported_with_its_line_number(tmp_path):
    ids = tmp_path / "selected.txt"
    ids.write_text("PMC1\n42827296\n")
    with pytest.raises(SystemExit, match="line 2.*looks like a PMID"):
        fetch_papers.read_pmcids(ids)
