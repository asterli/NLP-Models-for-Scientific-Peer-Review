import json

import pytest

from score_predictions import main


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


def run(tmp_path, *args):
    out = tmp_path / "results" / "out.json"
    main([*args, "--out", str(out)])
    return json.loads(out.read_text(encoding="utf-8"))


def test_claims_writes_scores_and_counts(tmp_path):
    gold = write_jsonl(tmp_path / "gold.jsonl", [
        {"paper_id": "p1", "sentence_id": "s001", "label": "CLAIM"},
        {"paper_id": "p1", "sentence_id": "s002", "label": "Background"},  # case doesn't matter
        {"paper_id": "p1", "sentence_id": "s003", "label": "claim"},
    ])
    pred = write_jsonl(tmp_path / "pred.jsonl", [
        {"paper_id": "p1", "sentence_id": "s001"},
        {"paper_id": "p1", "sentence_id": "s002"},
        {"paper_id": "p9", "sentence_id": "s001"},  # unlabeled paper, ignored
    ])
    results = run(tmp_path, "claims", "--pred", pred, "--gold", gold)
    assert results["task"] == "claims"
    assert (results["tp"], results["fp"], results["fn"]) == (1, 1, 1)
    assert results["counts"] == {"papers": 1, "sentences": 3, "gold_claims": 2,
                                 "predicted_claims": 2, "ignored_predictions": 1}


@pytest.mark.parametrize("rows, message", [
    ([{"paper_id": "p1", "sentence_id": "s001", "label": "MAYBE"}], "expected CLAIM or BACKGROUND"),
    ([{"paper_id": "p1", "sentence_id": "s001", "label": "CLAIM"},
      {"paper_id": "p1", "sentence_id": "s001", "label": "BACKGROUND"}], "labeled twice"),
])
def test_bad_claim_gold_stops(tmp_path, rows, message):
    gold = write_jsonl(tmp_path / "gold.jsonl", rows)
    pred = write_jsonl(tmp_path / "pred.jsonl", [])
    with pytest.raises(SystemExit, match=message):
        run(tmp_path, "claims", "--pred", pred, "--gold", gold)


def test_novelty_joins_on_comparison_id_in_any_order(tmp_path):
    gold = write_jsonl(tmp_path / "gold.jsonl", [
        {"comparison_id": "c1", "label": "distinct"},
        {"comparison_id": "c2", "label": "overlapping"},
    ])
    pred = write_jsonl(tmp_path / "pred.jsonl", [
        {"comparison_id": "c2", "predicted": "overlapping"},
        {"comparison_id": "c1", "predicted": "distinct"},
    ])
    results = run(tmp_path, "novelty", "--pred", pred, "--gold", gold)
    assert results["accuracy"] == 1.0
    assert results["count"] == 2


def test_novelty_missing_prediction_stops(tmp_path):
    gold = write_jsonl(tmp_path / "gold.jsonl", [{"comparison_id": "c1", "label": "distinct"}])
    pred = write_jsonl(tmp_path / "pred.jsonl", [])
    with pytest.raises(SystemExit, match="no prediction for 1"):
        run(tmp_path, "novelty", "--pred", pred, "--gold", gold)


def test_novelty_bad_label_stops_cleanly(tmp_path):
    gold = write_jsonl(tmp_path / "gold.jsonl", [{"comparison_id": "c1", "label": "distinct"}])
    pred = write_jsonl(tmp_path / "pred.jsonl", [{"comparison_id": "c1", "predicted": "Distinct"}])
    with pytest.raises(SystemExit, match="labels not in"):
        run(tmp_path, "novelty", "--pred", pred, "--gold", gold)


def test_retrieval_recall_at_k(tmp_path):
    ranked = write_jsonl(tmp_path / "run.jsonl", [
        {"query_id": "q1", "ranked": ["a", "x"]},
        {"query_id": "q2", "ranked": ["y", "z", "b"]},
    ])
    qrels = write_jsonl(tmp_path / "qrels.jsonl", [
        {"query_id": "q1", "relevant": ["a"]},
        {"query_id": "q2", "relevant": ["b"]},
    ])
    results = run(tmp_path, "retrieval", "--run", ranked, "--qrels", qrels, "--k", "2")
    assert results["recall@2"] == 0.5
    assert results["inputs"] == {"run": ranked, "qrels": qrels}