# Metrics for the three components (ticket M02). Pure functions: no files, no internet.
#   claims:    precision / recall / F1 over CLAIM sentences, keyed by (paper_id, sentence_id)
#   novelty:   per-class precision / recall / F1, macro-F1, accuracy, confusion matrix
#   retrieval: Recall@k, averaged over queries that have at least one relevant work
# Any rate with a zero denominator is reported as 0.0, never as an error.

from __future__ import annotations


def prf(predicted: set, gold: set) -> dict:
    # Precision, recall, and F1 of a predicted set against a gold set, with the raw counts.
    tp = len(predicted & gold)
    fp, fn = len(predicted - gold), len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def claim_scores(predicted: set[tuple[str, str]], gold_labels: dict[tuple[str, str], str]) -> dict:
    # predicted: (paper_id, sentence_id) pairs a model called CLAIM.
    # gold_labels: (paper_id, sentence_id) -> "CLAIM" or "BACKGROUND" for every labeled sentence.
    # Only labeled sentences count, so predictions on unlabeled papers are ignored, not wrong.
    labeled = set(gold_labels)
    gold_claims = {key for key, label in gold_labels.items() if label == "CLAIM"}
    scores = prf(predicted & labeled, gold_claims)
    scores["counts"] = {
        "papers": len({paper for paper, _ in labeled}),
        "sentences": len(labeled),
        "gold_claims": len(gold_claims),
        "predicted_claims": len(predicted & labeled),
        "ignored_predictions": len(predicted - labeled),
    }
    return scores