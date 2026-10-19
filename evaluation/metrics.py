# Metrics for the three components (ticket M02). Pure functions: no files, no internet.
#   claims:    precision / recall / F1 over CLAIM sentences, keyed by (paper_id, sentence_id)
#   novelty:   per-class precision / recall / F1, macro-F1, accuracy, confusion matrix
#   retrieval: Recall@k, averaged over queries that have at least one relevant work
# Any rate with a zero denominator is reported as 0.0, never as an error.

from __future__ import annotations

NOVELTY_LABELS = ("overlapping", "distinct", "insufficient_evidence")


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


def classification_scores(gold: list[str], predicted: list[str], labels=NOVELTY_LABELS) -> dict:
    # gold[i] and predicted[i] are the labels for the same item (e.g. one claim / prior-paper pair).
    if len(gold) != len(predicted):
        raise ValueError(f"{len(gold)} gold labels but {len(predicted)} predictions")
    unknown = (set(gold) | set(predicted)) - set(labels)
    if unknown:
        raise ValueError(f"labels not in {list(labels)}: {sorted(unknown)}")

    per_class = {}
    for label in labels:
        # Positions with this label, so each class is just prf on two sets.
        gold_idx = {i for i, g in enumerate(gold) if g == label}
        pred_idx = {i for i, p in enumerate(predicted) if p == label}
        scores = prf(pred_idx, gold_idx)
        per_class[label] = {k: scores[k] for k in ("precision", "recall", "f1")} | {"support": len(gold_idx)}

    # Rows are gold labels, columns are predictions.
    confusion = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, predicted):
        confusion[g][p] += 1

    correct = sum(g == p for g, p in zip(gold, predicted))
    return {
        "accuracy": correct / len(gold) if gold else 0.0,
        "correct": correct,
        # Every label counts, even one never predicted, so ignoring a class is penalized.
        "macro_f1": sum(c["f1"] for c in per_class.values()) / len(labels),
        "per_class": per_class,
        "confusion": confusion,
        "count": len(gold),
    }


def recall_at_k(ranked: dict[str, list[str]], relevant: dict[str, set[str]], k: int = 10) -> dict:
    # ranked: query_id -> work ids, best first. relevant: query_id -> the work ids that count as hits.
    # Queries with nothing relevant can't be scored (0/0), so they're skipped but counted.
    scored = {q: rel for q, rel in relevant.items() if rel}
    # A query with no results gets [] and scores 0. set() stops a repeated id counting twice.
    per_query = {q: len(set(ranked.get(q, [])[:k]) & rel) / len(rel) for q, rel in scored.items()}
    return {
        f"recall@{k}": sum(per_query.values()) / len(per_query) if per_query else 0.0,
        "queries": len(per_query),
        "queries_without_relevant": len(relevant) - len(scored),
        "queries_without_results": sum(1 for q in scored if not ranked.get(q)),
        "per_query": per_query,
    }
