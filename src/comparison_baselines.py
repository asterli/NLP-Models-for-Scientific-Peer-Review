import json
import re
from collections import Counter
from pathlib import Path

from evaluation.metrics import NOVELTY_LABELS, prf


LABELS = NOVELTY_LABELS
RULES = """Compare a new scientific claim with the supplied older passage, using only these two passages.
Overlapping: the older passage already says what the claim says.
Distinct: the claim says something the older passage does not.
Insufficient evidence: you cannot tell from these two passages.
Apply these rules:
1. Same outcome is not enough. If the intervention is different, it is a different contribution.
2. Do not judge whether the claim is true. Judge only whether the older passage already says it.
3. Main point test: if the older passage covers the main point and the claim just adds something small, choose Overlapping. If a main part is missing or contradicted, choose Distinct.
Do not infer missing study details."""
PROMPTS = {
    "label": RULES + "\nReturn exactly one label: overlapping, distinct, or insufficient_evidence.",
    "explanation": RULES + (
        "\nFirst compare the interventions and the main outcomes. Explain briefly whether the main "
        "contribution is already present, materially different, or impossible to assess from the "
        "supplied text. Then end with exactly FINAL: overlapping, FINAL: distinct, or "
        "FINAL: insufficient_evidence."
    ),
}


def load_split(path, split):
    rows, seen = [], set()
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        for key in ("comp_id", "newer_paper_id", "claim_text", "older_passage", "label"):
            if not isinstance(raw.get(key), str) or not raw[key].strip():
                raise ValueError(f"{path}:{line_number}: missing {key}")
        if raw.get("m16_split") != split:
            raise ValueError(f"{path}:{line_number}: expected {split} assignment")
        label = raw["label"].strip().lower().replace(" ", "_")
        if label not in LABELS:
            raise ValueError(f"Unknown label: {raw['label']}")
        if raw["comp_id"] in seen:
            raise ValueError(f"Duplicate comparison: {raw['comp_id']}")
        seen.add(raw["comp_id"])
        rows.append({
            "comparison_id": raw["comp_id"], "source_paper_id": raw["newer_paper_id"],
            "claim": raw["claim_text"], "prior_work": raw["older_passage"], "label": label,
        })
    if not rows:
        raise ValueError(f"Empty split: {path}")
    return rows


def load_development_data(train_path, dev_path):
    train, dev = load_split(train_path, "train"), load_split(dev_path, "dev")
    for key in ("comparison_id", "source_paper_id"):
        overlap = {r[key] for r in train} & {r[key] for r in dev}
        if overlap:
            raise ValueError(f"Training/development {key} overlap: {sorted(overlap)}")
    return train, dev


def majority_label(train):
    counts = Counter(row["label"] for row in train)
    if not counts:
        raise ValueError("Training data is empty")
    return max(LABELS, key=lambda label: counts[label])


def messages(claim, prior_work, prompt):
    return [
        {"role": "system", "content": PROMPTS[prompt]},
        {"role": "user", "content": f"New claim:\n{claim}\n\nOlder passage:\n{prior_work}"},
    ]


def parse_answer(answer, prompt):
    text = answer.strip()
    if prompt == "label":
        return text if text in LABELS else "invalid"
    if prompt != "explanation":
        raise ValueError(f"Unknown prompt: {prompt}")
    matches = re.findall(r"\bFINAL:\s*(overlapping|distinct|insufficient_evidence)\b", text)
    if len(matches) != 1 or not re.search(r"\bFINAL:\s*" + matches[0] + r"\s*$", text):
        return "invalid"
    return matches[0]


def score(records):
    if not records:
        raise ValueError("No predictions to score")
    if len({r["comparison_id"] for r in records}) != len(records):
        raise ValueError("Duplicate prediction IDs")
    if any(r["gold"] not in LABELS or r["predicted"] not in (*LABELS, "invalid") for r in records):
        raise ValueError("Unknown gold or predicted label")
    per_class = {}
    confusion = {g: {p: 0 for p in (*LABELS, "invalid")} for g in LABELS}
    for label in LABELS:
        gold = {i for i, r in enumerate(records) if r["gold"] == label}
        predicted = {i for i, r in enumerate(records) if r["predicted"] == label}
        per_class[label] = prf(predicted, gold) | {"support": len(gold)}
    for row in records:
        confusion[row["gold"]][row["predicted"]] += 1
    correct = sum(r["gold"] == r["predicted"] for r in records)
    return {
        "count": len(records), "correct": correct, "accuracy": correct / len(records),
        "macro_f1": sum(per_class[l]["f1"] for l in LABELS) / len(LABELS),
        "per_class": per_class, "confusion": confusion,
        "invalid_predictions": sum(r["predicted"] == "invalid" for r in records),
    }
