"""Shared prompt, data loading, target formatting and output parsing for the claim extractor
(M10 prompted baseline and M11/M12 fine-tuned model). Pure Python: no torch, no GPU.

The fine-tuned model is trained on exactly the per-sentence format of the M10 "v2" prompt,
so baseline and fine-tuned outputs are parsed by the same function and scored the same way.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

LABELS = ("CLAIM", "BACKGROUND")

DEFINITIONS = (
    "You label sentences from the abstract and introduction of a research paper.\n"
    "A CLAIM is a sentence stating the paper's own objective, contribution, finding, or conclusion "
    "(including implications of its findings).\n"
    "Everything else is BACKGROUND: prior work, motivation, gaps, general context, and descriptions of "
    "this study's own design, setting, recruitment, or sample, unless the paper presents them as its contribution.\n"
)

# Identical to PROMPTS["v2"] in scripts/baseline_claims.py. Keep them in sync.
SYSTEM_V2 = DEFINITIONS + (
    "\nDecide each sentence separately, in order:\n"
    "1. Does it state a specific objective of THIS paper, a contribution it makes, or a finding or conclusion "
    "from its own analysis? If yes -> CLAIM. A stated study aim counts even before results are reported. "
    "A result sentence from this paper's own analysis counts even if it only reports numbers. "
    "A conclusion or implication drawn from this paper's findings counts.\n"
    "2. Does it mainly describe prior research, general context, a research gap, or why more research is needed? "
    "If yes -> BACKGROUND, unless it also clearly states a current-paper objective, finding, or conclusion.\n"
    "3. Does it describe this study's design, setting, data collection, recruitment, eligibility, or sample? "
    "If it is a routine description -> BACKGROUND. Only CLAIM if presented as the paper's contribution.\n"
    "4. A sentence mixing background with a clear current-paper objective, finding, or conclusion is CLAIM.\n"
    "Judge by meaning in context, not by keywords or section headings.\n\n"
    "Output exactly one line per sentence, in the same order, of the form\n"
    "s001: BACKGROUND\n"
    "s002: CLAIM\n"
    "Label every sentence. Output nothing else."
)

_LINE = re.compile(r"(s\d{3})\s*[:\-–]\s*(CLAIM|BACKGROUND)\b", re.I)


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(split: str, papers_path="data/papers/papers.jsonl", splits_dir="data/claims/splits") -> list[dict]:
    """Papers in `split` ("train" | "dev" | "test" | "all"), each with a `labels` dict sentence_id -> label.

    Labels come from data/claims/splits/<split>.jsonl; sentences/sections from papers.jsonl.
    Papers that have no labeled sentences are dropped (and only labeled sentences are kept),
    so the model is never trained on or scored against unlabeled text.
    """
    papers = {p["paper_id"]: p for p in read_jsonl(papers_path)}
    names = ["train", "dev", "test"] if split == "all" else [split]
    labels: dict[str, dict[str, str]] = {}
    for name in names:
        for row in read_jsonl(Path(splits_dir) / f"{name}.jsonl"):
            label = row["label"].strip().upper()
            if label not in LABELS:
                raise ValueError(f"{name}: {row['paper_id']}/{row['sentence_id']} has label {row['label']!r}")
            labels.setdefault(row["paper_id"], {})[row["sentence_id"]] = label
    out = []
    for pid, lab in labels.items():
        if pid not in papers:
            raise KeyError(f"split paper {pid} not in {papers_path}")
        p = dict(papers[pid])
        p["sentences"] = [s for s in p["sentences"] if s["sentence_id"] in lab]
        p["labels"] = lab
        out.append(p)
    out.sort(key=lambda p: p["paper_id"])
    return out


def user_text(paper: dict) -> str:
    lines = [f"{s['sentence_id']} [{s['section']}]: {s['text']}" for s in paper["sentences"]]
    return "Paper title: " + paper["title"] + "\n\nSentences:\n" + "\n".join(lines)


def messages(paper: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_V2}, {"role": "user", "content": user_text(paper)}]


def target_text(paper: dict) -> str:
    """Gold assistant turn: one `sXXX: LABEL` line per sentence, in order."""
    return "\n".join(f"{s['sentence_id']}: {paper['labels'][s['sentence_id']]}" for s in paper["sentences"])


def parse_claim_ids(raw: str, valid: set[str] | dict) -> list[str]:
    """Sentence IDs the model labeled CLAIM, in order, deduplicated, restricted to `valid` IDs."""
    seen, out = set(), []
    for sid, label in _LINE.findall(raw):
        if label.upper() == "CLAIM" and sid in valid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def parse_all_labels(raw: str, valid: set[str] | dict) -> dict[str, str]:
    """sentence_id -> label for every parsable line (first occurrence wins). Missing IDs are absent."""
    out: dict[str, str] = {}
    for sid, label in _LINE.findall(raw):
        if sid in valid and sid not in out:
            out[sid] = label.upper()
    return out


def claim_rows(paper: dict, claim_ids: list[str]) -> list[dict]:
    """Predictions in the shared Claim schema (docs/data-schema.md)."""
    by_id = {s["sentence_id"]: s for s in paper["sentences"]}
    return [{"claim_id": f"{paper['paper_id']}#{sid}", "paper_id": paper["paper_id"],
             "sentence_id": sid, "text": by_id[sid]["text"]} for sid in claim_ids]
