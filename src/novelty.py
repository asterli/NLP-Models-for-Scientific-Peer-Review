from pathlib import Path
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = ("overlapping", "distinct", "insufficient_evidence")


def load_comparisons(path):
    rows = []
    seen = set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("Each comparison must be a JSON object")
        for field in ("comparison_id", "claim", "prior_work", "label"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing or empty {field}")
        if row["label"] not in LABELS:
            raise ValueError(f"Unknown label: {row['label']}")
        if row["comparison_id"] in seen:
            raise ValueError(f"Duplicate comparison: {row['comparison_id']}")
        seen.add(row["comparison_id"])
        rows.append(row)
    if not rows:
        raise ValueError("No comparisons found")
    return rows


def encode_pairs(tokenizer, claims, passages, max_length):
    if not claims or len(claims) != len(passages):
        raise ValueError("Provide equally sized, nonempty claim and passage lists")
    if any(not isinstance(text, str) or not text.strip() for text in [*claims, *passages]):
        raise ValueError("Claim and prior-work text must be nonempty strings")
    inputs = tokenizer(claims, passages, padding=True, truncation=False, return_tensors="pt")
    if inputs["input_ids"].shape[1] > max_length:
        raise ValueError(f"Text pair exceeds {max_length} tokens; select a shorter evidence passage")
    return inputs


class NoveltyClassifier:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True, use_safetensors=True
        ).eval()
        if self.model.config.id2label != dict(enumerate(LABELS)):
            raise ValueError("Checkpoint label mapping does not match the novelty labels")
        self.max_length = self.model.config.novelty_max_length

    def predict(self, claim, prior_work):
        inputs = encode_pairs(self.tokenizer, [claim], [prior_work], self.max_length)
        with torch.inference_mode():
            probabilities = self.model(**inputs).logits.softmax(dim=-1)[0]
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Non-finite model scores")
        index = probabilities.argmax().item()
        # Softmax scores are not calibrated confidence.
        return {"label": LABELS[index], "score": probabilities[index].item()}
