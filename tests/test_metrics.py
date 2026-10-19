# Tests for src/evaluation/metrics.py. Run with:  pytest tests/test_metrics.py
# The "hand-scored" cases reproduce numbers reported by teammates, worked out by hand.

import pytest

from evaluation.metrics import claim_scores, prf


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