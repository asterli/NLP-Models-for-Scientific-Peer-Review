"""Readable, evidence-linked Markdown report of a pipeline run."""
from __future__ import annotations

LABEL_TEXT = {"overlapping": "**Overlapping**", "distinct": "Distinct", "insufficient_evidence": "Insufficient evidence"}
STATUS_TEXT = {"no_evidence": "— not compared: title only, no passage",
               "not_run": "— not compared", "error": "⚠ error"}


def _cell(text) -> str:
    return str(text if text is not None else "").replace("|", "\\|").replace("\n", " ")


def _link(title, url) -> str:
    title = _cell(title) or "(untitled)"
    return f"[{title}]({url})" if url else title


def _verdict(comparison: dict) -> str:
    if comparison["status"] == "compared":
        return LABEL_TEXT[comparison["label"]]
    text = STATUS_TEXT.get(comparison["status"], comparison["status"])
    if comparison["status"] == "error" or (comparison["status"] == "not_run"
                                           and comparison.get("detail") != "comparison classifier not loaded"):
        text += f" ({_cell(comparison['detail'])})"
    return text


def _models(components: dict) -> list[str]:
    lines = []
    claims = components.get("claims") or {}
    if claims.get("mode") == "live":
        lines.append(f"- Claims: F06 extractor, live: `{claims.get('base_model')}` + adapter `{claims.get('adapter')}`")
    elif claims:
        ext = claims.get("extractor") or {}
        source = f" (made by adapter `{ext.get('adapter')}`)" if ext.get("adapter") else ""
        lines.append(f"- Claims: saved F06 extractor output `{claims.get('claims_file')}`{source}")
    ret = components.get("retrieval") or {}
    lines.append(f"- Retrieval: F10-filtered BM25 over `{ret.get('corpus')}` ({ret.get('corpus_works')} works), "
                 f"uncertain dates {ret.get('uncertain_dates')}")
    cmp_ = components.get("comparison")
    if cmp_:
        lines.append(f"- Comparison: F09 classifier `{cmp_.get('base_model')}` (MLX 4-bit, rev "
                     f"`{(cmp_.get('base_model_revision') or '')[:10]}`) + adapter `{cmp_.get('adapter')}`")
    else:
        lines.append("- Comparison: **not run** (`--no-comparison`); no labels in this report")
    return lines


def render(result: dict) -> str:
    run, s = result["run"], result["summary"]
    code = run["code"]
    out = ["# Evidence-linked claim report", "",
           f"Run {run['started']} · code `{(code.get('commit') or 'unknown')[:10]}`"
           + (" (uncommitted changes)" if code.get("dirty") else "") + f" · settings: top {result['settings']['k']} "
           f"retrieved per claim, compared {result['settings']['compare_top'] or 'all'} with evidence", ""]
    out += _models(result["components"]) + [""]
    out += [f"**Summary:** {s['papers']} paper(s), {s['claims']} claim(s), {s['pairs']} claim–prior-work pair(s). "
            f"Pair status: {s['pair_status'] or '{}'}. Labels: {s['labels'] or '{}'}.", "",
            "> How to read this: labels compare one claim with one short passage from one older work. "
            "*Overlapping* means the classifier judged the passage already states the claim's main point; it is "
            "not a verdict on the whole paper's novelty. The classifier returns no score; the retrieval score is "
            "BM25 rank strength, not confidence. The corpus is mostly works the source papers already cite, so "
            "missing prior work here says nothing about the wider literature.", ""]

    for rec in result["papers"]:
        p = rec["paper"]
        out += ["---", "", f"## {_link(p['title'], p.get('url'))}", "",
                f"`{p['paper_id']}` · published {p.get('date')} · {rec['n_sentences']} sentences · "
                f"{len(rec['claims'])} claim(s)", ""]
        if rec["status"] != "ok":
            out += [f"**{rec['status'].replace('_', ' ').capitalize()}:** {_cell(rec['detail'])}. "
                    "Nothing was retrieved or compared for this paper.", ""]
            continue
        for claim in rec["claims"]:
            out += [f"### {claim['sentence_id']} ({claim.get('section')})", "", f"> {claim['text']}", ""]
            excluded = claim.get("excluded_counts") or {}
            if excluded:
                out += ["Filtered out before ranking: " + ", ".join(f"{k} {v}" for k, v in sorted(excluded.items())), ""]
            if claim["status"] != "retrieved":
                out += [f"**{claim['status'].replace('_', ' ').capitalize()}:** {_cell(claim['detail'])}. "
                        "No comparison was made.", ""]
                continue
            out += ["| # | Older work | Date | BM25 | Evidence passage | Comparison |",
                    "|---:|---|---|---:|---|---|"]
            for c in claim["candidates"]:
                flags = f" <br>*{', '.join(c['flags'])}*" if c.get("flags") else ""
                evidence = f"“{_cell(c['evidence'])}”" if c.get("evidence") else "*(no abstract; title only)*"
                out.append(f"| {c.get('rank')} | {_link(c.get('title'), c.get('url'))}{flags} | {c.get('date') or ''} | "
                           f"{c['score']:.2f} | {evidence} | {_verdict(c['comparison'])} |")
            out.append("")
    return "\n".join(out) + "\n"
