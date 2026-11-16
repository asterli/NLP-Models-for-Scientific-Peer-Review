"""F10 (M23) - eligibility filters on top of the M22 BM25 retriever.

Every BM25 candidate goes through these checks, in order:

  1. not_scholarly_work  excluded: no title, or OpenAlex type 'paratext' (journal issue pages, editorial boards).
  2. submitted_paper     excluded: same DOI / PMID / PMCID / OpenAlex ID as the submitted paper, or the same
                         normalized title (catches other versions of it, e.g. a preprint).
  3. date check          see DATE POLICY below.
  4. duplicate_version   excluded: a later-ranked copy of a paper already kept (same DOI, PMID, PMCID, OpenAlex ID,
                         or same normalized title AND same year). Same title in different years is NOT merged.
                         The kept copy lists the merged copies in `duplicate_ids`.

DATE POLICY (explicit cutoff rule)
  Cutoff = the submitted paper's date. A candidate is eligible only if it is certainly published BEFORE that date.
  Each date is turned into the range of days it could mean:
      full date YYYY-MM-DD   -> that one day              (precision "day")
      YYYY-01-01             -> the whole year YYYY       (precision "year"; OpenAlex uses Jan 1 when only the
                                                           year is known, so we can't trust the day)
      only a year            -> the whole year            (precision "year")
      nothing                -> unknown                   (precision "missing")
  Then:
      candidate range ends before the cutoff range starts   -> eligible
      candidate range starts on/after the cutoff range ends -> excluded as 'same_date' (same day) or 'after_cutoff'
      anything else (same year with a year-only date,
      missing date, or no submitted date)                    -> 'uncertain'
  Uncertain candidates are NEVER silently treated as eligible:
      uncertain="exclude" (default) -> dropped and counted as 'date_uncertain'
      uncertain="flag"              -> returned with eligibility="uncertain" so a person can check them

EVIDENCE
  `evidence` is the abstract sentence that best matches the query. Title-only records (no abstract) get
  evidence=None and evidence_type="title_only"; the title is never passed off as a passage.
  Every candidate carries traceable sources: paper_id, url, doi, pmid, openalex_id, cited_by, cited_by_submitted.

Usage
  python retrieval/m23_filters.py --corpus data/corpus/corpus.jsonl --query "..." \
      --submitted-doi 10.1007/s11524-021-00550-1 --submitted-date 2021-06-28
  python retrieval/m23_filters.py --corpus data/corpus/corpus.jsonl --queries retrieval/outputs/m22/sample_claims.jsonl --out retrieval/outputs/m23
In code:
  from m23_filters import FilteredRetriever
  FR = FilteredRetriever("data/corpus/corpus.jsonl")
  hits = FR.search(claim, submitted={"doi": "...", "date": "YYYY-MM-DD", "title": "..."}, k=10)
  hits, excluded = FR.search(claim, submitted, k=10, explain=True)   # excluded = counts per reason
"""
import argparse, datetime as dt, json, os, re, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m22_bm25_retrieval import Retriever, sentences, tokenize

# ---------- normalizing ----------
def norm_title(t):
    t = re.sub(r"<[^>]+>", "", (t or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t)).strip()

def norm_doi(x):
    if not x: return None
    m = re.search(r"10\.\d{4,9}/\S+", str(x).lower())
    return m.group(0).rstrip(".") if m else None

def ids_of(d):
    """All identifiers of a record, normalized, for matching."""
    out = set()
    doi = norm_doi(d.get("doi")) or norm_doi(d.get("paper_id"))
    if doi: out.add("doi:" + doi)
    for k in ("pmid", "pmcid", "openalex_id"):
        v = str(d.get(k) or "").strip().lower()
        if v:
            v = v.rsplit("/", 1)[-1]                       # https://openalex.org/W123 -> w123
            if k == "pmid": v = v.removeprefix("pmid:")
            out.add(f"{k}:{v}")
    if str(d.get("paper_id", "")).startswith("openalex:"): out.add("openalex_id:" + d["paper_id"].split(":", 1)[1].lower())
    return out

# ---------- dates ----------
def date_range(date, year=None, jan1_is_year_only=True):
    """Return (first_day, last_day, precision). precision is 'day', 'year' or 'missing'."""
    s = str(date or "").strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        y, mo, d = map(int, m.groups())
        if not (jan1_is_year_only and mo == 1 and d == 1):
            day = dt.date(y, mo, d); return day, day, "day"
        year = y
    elif re.fullmatch(r"\d{4}(-\d{2})?", s):
        year = int(s[:4])                                  # YYYY or YYYY-MM: treat as the whole year
    if year and re.fullmatch(r"\d{4}", str(year)):
        y = int(year); return dt.date(y, 1, 1), dt.date(y, 12, 31), "year"
    return None, None, "missing"

def date_status(cand, cutoff, jan1_is_year_only=True):
    """'eligible', 'same_date', 'after_cutoff' or 'uncertain', plus the candidate's date precision.
    cand / cutoff are (date, year) tuples."""
    clo, chi, cprec = date_range(*cand, jan1_is_year_only=jan1_is_year_only)
    slo, shi, _ = date_range(*cutoff, jan1_is_year_only=False)   # submitted dates come from Europe PMC, trust the day
    if clo is None or slo is None: return "uncertain", cprec
    if chi < slo: return "eligible", cprec
    if clo >= shi: return ("same_date" if (cprec == "day" and clo == slo == shi) else "after_cutoff"), cprec
    return "uncertain", cprec

# ---------- retriever ----------
class FilteredRetriever:
    def __init__(self, corpus_path, uncertain="exclude", jan1_is_year_only=True, **kw):
        assert uncertain in ("exclude", "flag")
        self.R = Retriever(corpus_path, **kw)
        self.docs = self.R.docs
        self.by_pid = {d["paper_id"]: d for d in self.docs}
        self.uncertain, self.jan1 = uncertain, jan1_is_year_only

    def _evidence(self, d, qset):
        sents = sentences(d.get("abstract") or "")
        if not sents: return None, "title_only"
        return max(sents, key=lambda x: len(qset & set(tokenize(x)))), "abstract_sentence"

    def search(self, query, submitted=None, k=10, explain=False, uncertain=None):
        """submitted: dict with any of doi, pmid, pmcid, openalex_id, paper_id, title, date (YYYY-MM-DD), year."""
        policy = uncertain or self.uncertain
        sub = submitted or {}
        sub_ids, sub_title = ids_of(sub), norm_title(sub.get("title"))
        cutoff = (sub.get("date"), sub.get("year"))
        qset = set(tokenize(query))
        kept, excluded, seen = [], Counter(), {}
        for h in self.R.search(query, k=len(self.docs)):            # rank everything, then filter
            d = self.by_pid[h["paper_id"]]
            if not d.get("title") or d.get("type") == "paratext":
                excluded["not_scholarly_work"] += 1; continue
            if (ids_of(d) & sub_ids) or (sub_title and norm_title(d["title"]) == sub_title):
                excluded["submitted_paper"] += 1; continue
            status, prec = date_status((d.get("date"), d.get("year")), cutoff, self.jan1)
            if status in ("same_date", "after_cutoff"):
                excluded[status] += 1; continue
            if status == "uncertain" and policy == "exclude":
                excluded["date_uncertain"] += 1; continue
            year = str(d.get("year") or (d.get("date") or "")[:4])
            keys = ids_of(d) | {"title:" + norm_title(d["title"]) + "|" + year}
            dup_of = next((seen[x] for x in keys if x in seen), None)
            if dup_of is not None:
                dup_of["duplicate_ids"].append(d["paper_id"]); excluded["duplicate_version"] += 1; continue
            evidence, etype = self._evidence(d, qset)
            flags = []
            if prec == "missing": flags.append("date_missing")
            elif prec == "year": flags.append("date_year_only")
            if status == "uncertain": flags.append("date_uncertain")
            if etype == "title_only": flags.append("no_abstract")
            cited_by = d.get("cited_by") or []
            c = {"rank": len(kept) + 1, "score": h["score"], "paper_id": d["paper_id"], "title": d["title"],
                 "date": d.get("date"), "year": d.get("year"), "date_precision": prec, "venue": d.get("venue"),
                 "type": d.get("type"), "url": d.get("url"), "doi": d.get("doi"), "pmid": d.get("pmid"),
                 "openalex_id": d.get("openalex_id"), "evidence": evidence, "evidence_type": etype,
                 "eligibility": "eligible" if status == "eligible" else "uncertain", "flags": flags,
                 "cited_by": cited_by,
                 "cited_by_submitted": bool(sub_ids & set().union(*[ids_of({"paper_id": x}) for x in cited_by])) if cited_by else False,
                 "duplicate_ids": []}
            kept.append(c)
            for x in keys: seen[x] = c
            if len(kept) == k: break
        return (kept, dict(excluded)) if explain else kept

def submitted_from_row(r):
    """Build the submitted-paper dict from a papers.jsonl, M15 or M08-style row."""
    return {"paper_id": r.get("paper_id") if str(r.get("paper_id", "")).startswith(("doi:", "openalex:")) else None,
            "doi": r.get("doi") or r.get("newer_link") or r.get("newer_paper_id"), "pmid": r.get("pmid") or r.get("newer_pmid"),
            "pmcid": r.get("pmcid") or r.get("newer_pmcid"), "title": r.get("title") or r.get("newer_title") or r.get("paper_title"),
            "date": r.get("date") or r.get("newer_pub_date") or r.get("pub_date")}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--query"); ap.add_argument("--queries")
    ap.add_argument("--submitted-doi"); ap.add_argument("--submitted-date"); ap.add_argument("--submitted-title")
    ap.add_argument("--uncertain", choices=["exclude", "flag"], default="exclude")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="retrieval/outputs/m23")
    a = ap.parse_args()
    FR = FilteredRetriever(a.corpus, uncertain=a.uncertain)
    if a.query:
        sub = {"doi": a.submitted_doi, "date": a.submitted_date, "title": a.submitted_title}
        hits, ex = FR.search(a.query, sub, a.k, explain=True)
        print("excluded:", ex)
        for h in hits:
            print(f"{h['rank']:>2}. [{h['score']:.2f}] {h['title']} ({h['date']}) {h['url']} {h['flags'] or ''}")
            print(f"    > {h['evidence'] or '(title only, no passage)'}")
    if a.queries:
        os.makedirs(a.out, exist_ok=True)
        rows = [json.loads(l) for l in open(a.queries, encoding="utf-8") if l.strip()]
        res, lines, tot = [], [], Counter()
        for r in rows:
            claim = r.get("claim_text") or r.get("source_excerpt") or r.get("text")
            sub = submitted_from_row(r)
            hits, ex = FR.search(claim, sub, a.k, explain=True); tot.update(ex)
            qid = r.get("comp_id") or r.get("sentence_id") or r.get("query_id")
            res.append({"query_id": qid, "claim_text": claim, "submitted": sub, "excluded_counts": ex, "candidates": hits})
            lines += [f"=== {qid} (submitted {sub.get('date')}): {claim}", f"    excluded: {ex}"] + \
                     [f"  {h['rank']:>2}. [{h['score']:.2f}] {h['title']} ({h['date']}) {h['url']}"
                      + (f"  FLAGS={h['flags']}" if h['flags'] else "") + ("  [cited by submitted paper]" if h['cited_by_submitted'] else "")
                      + f"\n      > {h['evidence'] or '(title only, no passage)'}" for h in hits] + [""]
        with open(f"{a.out}/m23_filtered_results.jsonl", "w", encoding="utf-8") as f:
            for x in res: f.write(json.dumps(x, ensure_ascii=False) + "\n")
        open(f"{a.out}/m23_filtered_results.txt", "w", encoding="utf-8").write("\n".join(lines))
        print(f"wrote {len(res)} queries -> {a.out}/  | total excluded: {dict(tot)}")
