# Tests for scripts/diff_papers.py. Run with:  pytest tests/test_diff_papers.py

import json

import diff_papers


def paper(*texts, pmcid="PMC1", cites=None):
    sentences = [{"sentence_id": f"s{i:03d}", "section": "introduction", "text": t, "cites": (cites or {}).get(i, [])}
                 for i, t in enumerate(texts, start=1)]
    return {"pmcid": pmcid, "paper_id": "doi:10.1/x", "sentences": sentences, "references": []}


def test_identical_papers_have_no_changes():
    assert diff_papers.diff_paper(paper("A.", "B."), paper("A.", "B.")) == ([], False)


def test_text_fix_in_place_keeps_ids():
    lines, ids_shift = diff_papers.diff_paper(paper("BMI was 27 kg/m[2].", "B."), paper("BMI was 27 kg/m^2.", "B."))
    assert not ids_shift
    assert lines == ["  s001 edited:", "    - BMI was 27 kg/m[2].", "    + BMI was 27 kg/m^2."]


def test_cites_change_is_reported():
    lines, ids_shift = diff_papers.diff_paper(paper("A [1-4].", cites={1: ["R1", "R4"]}),
                                              paper("A [1-4].", cites={1: ["R1", "R2", "R3", "R4"]}))
    assert lines == ["  s001 cites: ['R1', 'R4'] -> ['R1', 'R2', 'R3', 'R4']"] and not ids_shift


def test_restored_sentence_shifts_later_ids():
    lines, ids_shift = diff_papers.diff_paper(paper("A.", "C."), paper("A.", "Funding fell.", "C."))
    assert ids_shift
    assert lines[0] == "  sentences change at s002: 0 became 1; 1 later sentence(s) get new IDs"


def test_sentence_added_at_the_end_moves_no_ids():
    lines, ids_shift = diff_papers.diff_paper(paper("A."), paper("A.", "B."))
    assert not ids_shift and "0 later sentence(s)" in lines[0]


def test_main_prints_a_summary(tmp_path, capsys):
    old, new = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
    old.write_text(json.dumps(paper("A.", "C.")) + "\n" + json.dumps(paper("X.", pmcid="PMC2")) + "\n")
    new.write_text(json.dumps(paper("A.", "B.", "C.")) + "\n" + json.dumps(paper("X.", pmcid="PMC2")) + "\n")
    diff_papers.main([str(old), str(new)])
    assert "1 unchanged, 0 changed with the same IDs, 1 with shifted IDs: PMC1" in capsys.readouterr().out