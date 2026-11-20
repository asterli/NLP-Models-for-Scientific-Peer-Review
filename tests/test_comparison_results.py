import pytest

from evaluation.comparison_results import error_examples, evaluate_predictions


@pytest.fixture
def examples():
    rows = [
        {"comparison_id": str(i), "source_paper_id": "paper", "claim": f"claim {i}",
         "prior_work": f"evidence {i}", "label": label}
        for i, label in enumerate(("overlapping", "distinct", "insufficient_evidence"))
    ]
    predictions = [
        {"comparison_id": r["comparison_id"], "source_paper_id": r["source_paper_id"],
         "gold": r["label"], "predicted": "distinct"}
        for r in rows
    ]
    return rows, predictions


def test_aligns_by_id_not_file_order(examples):
    rows, predictions = examples
    result = evaluate_predictions(rows, list(reversed(predictions)))
    assert result["correct"] == 1
    assert result["count"] == 3
    assert result["macro_f1"] == pytest.approx(1 / 6)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "wrong_gold", "wrong_source"])
def test_rejects_incomplete_or_mismatched_predictions(examples, mutation):
    rows, predictions = examples
    if mutation == "missing":
        predictions.pop()
    elif mutation == "duplicate":
        predictions.append(predictions[0])
    elif mutation == "extra":
        predictions.append(predictions[0] | {"comparison_id": "unknown"})
    elif mutation == "wrong_gold":
        predictions[0]["gold"] = "distinct"
    else:
        predictions[0]["source_paper_id"] = "other"
    with pytest.raises(ValueError):
        evaluate_predictions(rows, predictions)


def test_error_selection_is_stable_and_preserves_evidence(examples):
    rows, predictions = examples
    errors = error_examples(rows, list(reversed(predictions)))
    assert [r["comparison_id"] for r in errors] == ["0", "2"]
    assert errors[0]["claim"] == "claim 0"
    assert errors[0]["prior_work"] == "evidence 0"
    assert len(error_examples(rows, predictions, limit=1)) == 1
