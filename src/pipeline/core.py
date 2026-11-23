"""Run one paper through extraction, retrieval and comparison, recording a status at every step.

The steps are plain callables so the real components and test stand-ins plug in the same way:

    extract(paper)                          -> list of Claim dicts (docs/data-schema.md, section 2)
    retrieve(claim_text, paper, k)          -> (list of Retrieval Candidate dicts, excluded-count dict)
    compare(claim_text, evidence_text)      -> one of LABELS

Nothing is invented when a step has nothing to work with:
  paper status      ok | no_claims | extraction_error
  claim status      retrieved | no_eligible_candidates | retrieval_error
  comparison status compared | no_evidence | not_run | error
Only `compared` carries a label. `insufficient_evidence` is a model decision about supplied text and is
never used to fill in for a pair that was skipped or failed.
"""
from __future__ import annotations

LABELS = ("overlapping", "distinct", "insufficient_evidence")

# Errors the comparison classifier documents for a single pair (overlong input, invalid output).
# Anything else (e.g. a missing GPU) is a setup problem and stops the run.
PAIR_ERRORS = (ValueError, RuntimeError)


class ClaimsMismatch(ValueError):
    """Claims that do not point at real sentences of the paper (wrong ID or altered text)."""


def check_claims(paper: dict, claims: list[dict]) -> list[dict]:
    """Each claim must be a sentence of this paper, with its original text, in the shared Claim schema."""
    by_id = {s["sentence_id"]: s for s in paper.get("sentences") or []}
    seen = set()
    for c in claims:
        for field in ("claim_id", "paper_id", "sentence_id", "text"):
            if not isinstance(c.get(field), str) or not c[field].strip():
                raise ClaimsMismatch(f"claim is missing `{field}`: {c!r}")
        if c["paper_id"] != paper["paper_id"]:
            raise ClaimsMismatch(f"claim {c['claim_id']} belongs to {c['paper_id']}, not {paper['paper_id']}")
        sentence = by_id.get(c["sentence_id"])
        if sentence is None:
            raise ClaimsMismatch(f"claim {c['claim_id']}: no sentence {c['sentence_id']} in {paper['paper_id']}")
        if c["text"] != sentence["text"]:
            raise ClaimsMismatch(f"claim {c['claim_id']}: text differs from the paper's sentence {c['sentence_id']}")
        if c["sentence_id"] in seen:
            raise ClaimsMismatch(f"{paper['paper_id']}: sentence {c['sentence_id']} claimed twice")
        seen.add(c["sentence_id"])
    return claims


def paper_summary(paper: dict) -> dict:
    return {k: paper.get(k) for k in ("paper_id", "title", "date", "url", "doi", "pmcid")}


def compare_pair(claim: dict, candidate: dict, compare, rank_allowed: bool) -> dict:
    """One Novelty Prediction (docs/data-schema.md, section 4) plus the status of how it was produced."""
    out = {"claim_id": claim["claim_id"], "prior_paper_id": candidate["paper_id"], "label": None,
           "score": None, "reason": None, "status": None, "detail": None}
    evidence = candidate.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip():
        out.update(status="no_evidence",
                   detail=f"no passage to compare ({candidate.get('evidence_type') or 'missing evidence'})")
    elif compare is None:
        out.update(status="not_run", detail="comparison classifier not loaded")
    elif not rank_allowed:
        out.update(status="not_run", detail="below --compare-top")
    else:
        try:
            label = compare(claim["text"], evidence)
        except PAIR_ERRORS as e:
            out.update(status="error", detail=f"{type(e).__name__}: {e}")
        else:
            if label in LABELS:
                out.update(status="compared", label=label)
            else:
                out.update(status="error", detail=f"classifier returned an unknown label: {label!r}")
    return out


def run_paper(paper: dict, extract, retrieve, compare=None, k: int = 10, compare_top: int | None = None) -> dict:
    """Run one paper. `compare=None` runs extraction and retrieval only (pairs are `not_run`)."""
    record = {"paper": paper_summary(paper), "status": "ok", "detail": None,
              "n_sentences": len(paper.get("sentences") or []), "claims": []}
    try:
        claims = check_claims(paper, extract(paper))
    except Exception as e:  # noqa: BLE001 -- one paper's extraction failure is recorded, not fatal to the run
        record.update(status="extraction_error", detail=f"{type(e).__name__}: {e}")
        return record
    if not claims:
        record.update(status="no_claims", detail="the extractor found no claim sentences")
        return record

    sections = {s["sentence_id"]: s.get("section") for s in paper["sentences"]}
    for claim in claims:
        entry = {**claim, "section": sections.get(claim["sentence_id"]), "status": "retrieved", "detail": None,
                 "excluded_counts": {}, "candidates": []}
        record["claims"].append(entry)
        try:
            candidates, excluded = retrieve(claim["text"], paper, k)
        except Exception as e:  # noqa: BLE001
            entry.update(status="retrieval_error", detail=f"{type(e).__name__}: {e}")
            continue
        entry["excluded_counts"] = dict(excluded or {})
        if not candidates:
            entry.update(status="no_eligible_candidates",
                         detail="no corpus work passed the F10 filters (older, not the paper itself, dated)")
            continue
        for i, cand in enumerate(candidates):
            allowed = compare_top is None or i < compare_top
            entry["candidates"].append({**cand, "comparison": compare_pair(claim, cand, compare, allowed)})
    return record


def predictions(records: list[dict]) -> list[dict]:
    """Flatten every claim-candidate pair, in order, as Novelty Prediction rows with their status."""
    rows = []
    for rec in records:
        for claim in rec["claims"]:
            for cand in claim["candidates"]:
                rows.append({"paper_id": rec["paper"]["paper_id"], "sentence_id": claim["sentence_id"],
                             "rank": cand.get("rank"), "retrieval_score": cand.get("score"),
                             "evidence": cand.get("evidence"), **cand["comparison"]})
    return rows


def summarize(records: list[dict]) -> dict:
    count = lambda xs: {k: xs.count(k) for k in sorted(set(xs))}  # noqa: E731
    claims = [c for r in records for c in r["claims"]]
    pairs = [cand["comparison"] for c in claims for cand in c["candidates"]]
    return {
        "papers": len(records), "paper_status": count([r["status"] for r in records]),
        "claims": len(claims), "claim_status": count([c["status"] for c in claims]),
        "pairs": len(pairs), "pair_status": count([p["status"] for p in pairs]),
        "labels": count([p["label"] for p in pairs if p["status"] == "compared"]),
    }
