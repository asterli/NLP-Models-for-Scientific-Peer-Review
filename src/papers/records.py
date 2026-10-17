# Converts a parsed article into a paper record (docs/data-schema.md, "Paper")
# Then checks the record for problems before labeling.

# build_record() is where each sentence gets their sentence_ids: (s001, s002, etc.)
# Order is based on reading order.
from __future__ import annotations

from dataclasses import asdict

from papers.europepmc import PaperMeta
from papers.jats import NON_RESEARCH_TYPES, Article, normalize_doi
from papers.sentences import split_sentences

# Warning thresholds. A sentence outside these is often a bad split, so a person should check it.
SHORT_SENTENCE_CHARS = 25  # e.g. "See Figure 1." left over from a figure reference
LONG_SENTENCE_WORDS = 80   # e.g. two sentences glued together after "U.S."


def build_record(pmcid: str, article: Article, meta: PaperMeta) -> dict:
    doi = article.doi or normalize_doi(meta.doi)
    sentences = []
    for paragraph in article.paragraphs:
        for s in split_sentences(paragraph):
            sentences.append({
                "sentence_id": f"s{len(sentences) + 1:03d}",
                "section": paragraph.section,
                "text": s.text,
                "cites": s.cites,
            })

    return {
        "paper_id": f"doi:{doi}" if doi else f"pmcid:{pmcid}",
        "title": article.title or meta.title,
        "date": meta.date,
        "url": f"https://doi.org/{doi}" if doi else f"https://europepmc.org/article/PMC/{pmcid}",
        "text": " ".join(s["text"] for s in sentences),
        "sentences": sentences,
        "references": [asdict(r) for r in article.references],
        "authors": meta.authors,
        "venue": article.venue or meta.journal,
        "doi": doi,
        "pmcid": pmcid,
        "article_type": article.article_type,
        "source": "europepmc",
        "split": None,
    }


def check_record(record: dict) -> list[str]:
    # Plain-English warnings about a record. IDEALLY we want an empty list returned to verify that nothing is off.
    sentences = record["sentences"]
    sections = {s["section"] for s in sentences}
    warnings = []

    if not sentences:
        warnings.append("no sentences extracted")
    if not any(section.startswith("abstract") for section in sections):
        warnings.append("no abstract found")
    if "introduction" not in sections:
        warnings.append("no introduction found (check the XML's section titles)")
    if record["article_type"] in NON_RESEARCH_TYPES:
        warnings.append(f"article type is '{record['article_type']}', which usually has no findings to label")
    if not record["date"]:
        warnings.append("missing publication date (needed to filter out newer prior work)")

    short = [s["sentence_id"] for s in sentences if len(s["text"]) < SHORT_SENTENCE_CHARS]
    long = [s["sentence_id"] for s in sentences if len(s["text"].split()) > LONG_SENTENCE_WORDS]
    if short:
        warnings.append(f"very short sentences, maybe bad splits: {', '.join(short)}")
    if long:
        warnings.append(f"very long sentences, maybe missed splits: {', '.join(long)}")

    if not record["references"]:
        warnings.append("no reference list found")
    elif not any(r["doi"] for r in record["references"]):
        warnings.append("no DOIs in the reference list (the corpus builder can't use these references)")
    return warnings
