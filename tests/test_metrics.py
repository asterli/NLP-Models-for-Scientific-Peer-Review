# Tests for src/evaluation/metrics.py. Run with:  pytest tests/test_metrics.py
# The "hand-scored" cases reproduce numbers reported by teammates, worked out by hand.

import pytest

from evaluation.metrics import claim_scores, classification_scores, prf


def approx(x):
    return pytest.approx(x, abs=1e-3)


# --- prf -----------------------------------------------------------------------

def test_prf_basic():
    s = prf({"a", "b", "c"}, {"b", "c", "d", "e"})
    assert (s["tp"], s["fp"], s["fn"]) == (2, 1, 2)
    assert (s["precision"], s["recall"], s["f1"]) == (approx(2 / 3), approx(0.5), approx(4 / 7))


def test_prf_empty_sets_give_zero_not_error():
    assert prf(set(), set())["f1"] == 0.0
    assert prf(set(), {"a"})["precision"] == 0.0


# --- claims --------------------------------------------------------------------

def test_hand_scored_m10_case():
    # Brandon's M10 check: 12 predicted claims, all correct, 33 gold claims in 132 sentences.
    gold = {("p", f"s{i:03d}"): ("CLAIM" if i <= 33 else "BACKGROUND") for i in range(1, 133)}
    predicted = {("p", f"s{i:03d}") for i in range(1, 13)}
    s = claim_scores(predicted, gold)
    assert (s["precision"], s["recall"], s["f1"]) == (approx(1.0), approx(12 / 33), approx(0.533))
    assert s["counts"] == {"papers": 1, "sentences": 132, "gold_claims": 33,
                           "predicted_claims": 12, "ignored_predictions": 0}


def test_predictions_on_unlabeled_papers_are_ignored_not_wrong():
    gold = {("p1", "s001"): "CLAIM", ("p1", "s002"): "BACKGROUND"}
    predicted = {("p1", "s001"), ("p2", "s005")}  # p2 has no gold labels yet
    s = claim_scores(predicted, gold)
    assert (s["precision"], s["counts"]["ignored_predictions"]) == (1.0, 1)


def test_hand_scored_always_distinct_baseline():
    # Dylan's M17 pilot: 19 distinct, 9 overlapping, 2 insufficient. The baseline always says distinct.
    gold = ["distinct"] * 19 + ["overlapping"] * 9 + ["insufficient_evidence"] * 2
    scores = classification_scores(gold, ["distinct"] * 30)
    assert scores["correct"] == 19
    assert scores["accuracy"] == approx(19 / 30)
    # distinct: P = 19/30, R = 1, so F1 = 38/49. The other two classes score 0.
    assert scores["per_class"]["distinct"]["f1"] == approx(38 / 49)
    assert scores["macro_f1"] == approx(38 / 49 / 3)  # 0.259, same as Dylan's majority row
    assert scores["per_class"]["insufficient_evidence"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 2}
    assert scores["confusion"]["overlapping"]["distinct"] == 9


def test_two_class_run_with_custom_labels():
    scores = classification_scores(["yes", "no", "yes"], ["yes", "yes", "yes"], labels=("yes", "no"))
    assert scores["accuracy"] == approx(2 / 3)
    assert scores["per_class"]["no"]["support"] == 1
    # yes: P = 2/3, R = 1, so F1 = 0.8. no: F1 = 0.
    assert scores["macro_f1"] == approx(0.8 / 2)


@pytest.mark.parametrize("gold, predicted, message", [
    (["distinct"], [], "1 gold labels but 0 predictions"),
    (["distinct"], ["Distinct"], "labels not in"),
])
def test_bad_classification_input(gold, predicted, message):
    with pytest.raises(ValueError, match=message):
        classification_scores(gold, predicted)

