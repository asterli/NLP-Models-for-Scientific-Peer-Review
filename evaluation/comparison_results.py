import json
from pathlib import Path

from comparison_baselines import score


def read_predictions(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def evaluate_predictions(rows, predictions):
    expected = {row["comparison_id"]: row for row in rows}
    if len(expected) != len(rows):
        raise ValueError("Duplicate comparison IDs in evaluation data")
    actual = {row["comparison_id"]: row for row in predictions}
    if len(actual) != len(predictions):
        raise ValueError("Duplicate prediction IDs")
    if actual.keys() != expected.keys():
        raise ValueError("Predictions must cover exactly the evaluation comparisons")
    ordered = []
    for row in rows:
        prediction = actual[row["comparison_id"]]
        if prediction["gold"] != row["label"]:
            raise ValueError(f"Gold label mismatch: {row['comparison_id']}")
        if prediction["source_paper_id"] != row["source_paper_id"]:
            raise ValueError(f"Source paper mismatch: {row['comparison_id']}")
        ordered.append(prediction)
    return score(ordered)


def error_examples(rows, predictions, limit=10):
    if limit < 1:
        raise ValueError("Error limit must be positive")
    evaluate_predictions(rows, predictions)
    inputs = {row["comparison_id"]: row for row in rows}
    errors = sorted(
        (p for p in predictions if p["gold"] != p["predicted"]),
        key=lambda p: p["comparison_id"],
    )
    return [
        p | {"claim": inputs[p["comparison_id"]]["claim"],
             "prior_work": inputs[p["comparison_id"]]["prior_work"]}
        for p in errors[:limit]
    ]
