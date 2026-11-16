import json

import pytest

import baseline_novelty
from comparison_baselines import (
    LABELS, load_development_data, load_split, majority_label, messages, parse_answer, score,
)


def write_rows(path, split, labels):
    rows = [
        {"comp_id": f"{split}-{i}", "newer_paper_id": split, "claim_text": "A new claim",
         "older_passage": "An older passage", "label": label, "m16_split": split,
         "reason": "Human explanation must not enter the prompt"}
        for i, label in enumerate(labels)
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def test_majority_uses_train_and_does_not_read_test(tmp_path, monkeypatch):
    train, dev, out = tmp_path / "train.jsonl", tmp_path / "dev.jsonl", tmp_path / "result"
    write_rows(train, "train", ["Overlapping", "Overlapping", "Distinct"])
    write_rows(dev, "dev", ["Distinct", "Distinct", "Overlapping"])
    (tmp_path / "test.jsonl").write_text("not valid json")
    monkeypatch.setattr(baseline_novelty, "run_qwen", lambda *a: pytest.fail("Must not load Qwen"))
    baseline_novelty.main(["--train", str(train), "--dev", str(dev), "--output", str(out), "--majority-only"])
    run = json.loads((out / "run.json").read_text())
    metrics = json.loads((out / "majority_metrics.json").read_text())
    assert run["complete"] and run["majority_label"] == "overlapping"
    assert metrics["correct"] == 1 and metrics["count"] == 3
    assert metrics["macro_f1"] == pytest.approx(1 / 6)
    with pytest.raises(SystemExit):
        baseline_novelty.main(["--train", str(train), "--dev", str(dev), "--output", str(out), "--majority-only"])


def test_invalid_predictions_are_errors_without_fourth_class_average():
    rows = [
        {"comparison_id": "1", "gold": "overlapping", "predicted": "overlapping"},
        {"comparison_id": "2", "gold": "overlapping", "predicted": "invalid"},
        {"comparison_id": "3", "gold": "distinct", "predicted": "distinct"},
        {"comparison_id": "4", "gold": "insufficient_evidence", "predicted": "distinct"},
    ]
    metrics = score(rows)
    assert metrics["accuracy"] == 0.5
    assert metrics["per_class"]["overlapping"]["recall"] == 0.5
    assert metrics["macro_f1"] == pytest.approx(4 / 9)
    assert metrics["confusion"]["overlapping"]["invalid"] == 1
    assert metrics["invalid_predictions"] == 1
    assert set(metrics["per_class"]) == set(LABELS)


@pytest.mark.parametrize("prompt,answer,expected", [
    ("label", "distinct\n", "distinct"),
    ("label", "I think distinct", "invalid"),
    ("explanation", "Different intervention.\nFINAL: distinct", "distinct"),
    ("explanation", "FINAL: overlapping\nFINAL: distinct", "invalid"),
    ("explanation", "Cannot assess this.", "invalid"),
    ("explanation", "FINAL: insufficient_evidence", "insufficient_evidence"),
    ("explanation", "FINAL: distinct but maybe overlapping", "invalid"),
])
def test_output_parsing(prompt, answer, expected):
    assert parse_answer(answer, prompt) == expected


def test_loader_keeps_only_allowed_fields(tmp_path):
    path = tmp_path / "dev.jsonl"
    write_rows(path, "dev", ["Insufficient evidence"])
    row = load_split(path, "dev")[0]
    assert row["label"] == "insufficient_evidence"
    assert "reason" not in row
    for prompt in ("label", "explanation"):
        msg = messages(row["claim"], row["prior_work"], prompt)
        assert msg[1]["content"] == "New claim:\nA new claim\n\nOlder passage:\nAn older passage"
        assert "Human explanation" not in json.dumps(msg)


def test_rejects_test_as_development(tmp_path):
    path = tmp_path / "wrong.jsonl"
    write_rows(path, "test", ["Distinct"])
    with pytest.raises(ValueError, match="expected dev"):
        load_split(path, "dev")


def test_rejects_reused_source_paper(tmp_path):
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    write_rows(train, "train", ["Distinct"])
    rows = write_rows(dev, "dev", ["Distinct"])
    rows[0]["newer_paper_id"] = "train"
    dev.write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="source_paper_id overlap"):
        load_development_data(train, dev)


@pytest.mark.parametrize("change", [{"label": "unknown"}, {"older_passage": " "}, {"comp_id": None}])
def test_rejects_invalid_data(tmp_path, change):
    path = tmp_path / "data.jsonl"
    rows = write_rows(path, "train", ["Distinct"])
    path.write_text(json.dumps(rows[0] | change) + "\n")
    with pytest.raises(ValueError):
        load_split(path, "train")


def test_majority_tie_is_deterministic():
    assert majority_label([{"label": "distinct"}, {"label": "overlapping"}]) == "overlapping"
