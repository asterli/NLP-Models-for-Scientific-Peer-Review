"""M22 - BM25 retrieval baseline over the M21 background corpus.

Plain Python (no installs). Use as a script or import `Retriever` in the pipeline (M03).

Examples
  # one claim
  python m22_bm25_retrieval.py --corpus corpus.jsonl --query "Housing First improved housing stability compared with TAU"
  # many claims from a JSONL file (one object per line with a 'claim_text' field, e.g. m15_labeled_v2.jsonl)
  python m22_bm25_retrieval.py --corpus corpus.jsonl --queries m15_labeled_v2.jsonl --out m22_results

Optional (preview of M23 filters): --before YYYY-MM-DD drops candidates published on/after that date;
--exclude ID drops the submitted paper itself. When --queries rows carry newer_pub_date / newer_link,
those are applied automatically per claim.
"""
import argparse, json, math, os, re
from collections import Counter

STOP = set("""a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having he her here
hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on
once only or other our ours ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were what when where which while
who whom why will with would you your yours yourself yourselves et al vs versus among within without study studies
using used use however may might
ci rr rrr irr hr aor smd sd ratio ratios interval confidence significant significantly difference differences
mean median participant participants group groups compared month months year years result results""".split())

def tokenize(text):
    toks = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", (text or "").lower())
    out = []
    for t in toks:
        if t in STOP or len(t) < 2 or re.fullmatch(r"[0-9][0-9-]*", t): continue   # drop stop/stat words and bare numbers
        # light stemming: plural / common suffixes
        for suf in ("ations", "ation", "ings", "ing", "ies", "es", "s"):
            if t.endswith(suf) and len(t) - len(suf) >= 4:
                t = t[: -len(suf)] + ("y" if suf == "ies" else "")
                break
        out.append(t)
    return out

def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text or "") if s.strip()]

class Retriever:
    """BM25 over title + abstract. k1/b are the standard defaults."""
    def __init__(self, corpus_path, k1=1.5, b=0.75, title_weight=2):
        self.docs = [json.loads(l) for l in open(corpus_path, encoding="utf-8") if l.strip()]
        self.k1, self.b = k1, b
        self.tf, self.len = [], []
        df = Counter()
        for d in self.docs:
            toks = tokenize(d.get("title", "")) * title_weight + tokenize(d.get("abstract", ""))
            c = Counter(toks); self.tf.append(c); self.len.append(len(toks)); df.update(c.keys())
        self.N = len(self.docs); self.avg = sum(self.len) / max(1, self.N)
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def _score(self, q, i):
        tf, L, s = self.tf[i], self.len[i], 0.0
        for t in q:
            f = tf.get(t)
            if f: s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
        return s

    def search(self, query, k=10, before=None, exclude=()):
        q = tokenize(query); qset = set(q)
        exclude = {str(e).lower() for e in exclude if e}
        hits = []
        for i, d in enumerate(self.docs):
            if str(d.get("paper_id", "")).lower() in exclude or str(d.get("doi", "")).lower() in exclude: continue
            if before and d.get("date") and d["date"] >= before: continue
            s = self._score(q, i)
            if s > 0: hits.append((s, i))
        hits.sort(reverse=True)
        out = []
        for rank, (s, i) in enumerate(hits[:k], 1):
            d = self.docs[i]
            sents = sentences(d.get("abstract", "")) or [d.get("title", "")]
            best = max(sents, key=lambda x: len(qset & set(tokenize(x))))   # most query-overlapping sentence
            out.append({"rank": rank, "score": round(s, 3), "paper_id": d.get("paper_id"), "title": d.get("title"),
                        "date": d.get("date"), "venue": d.get("venue"), "url": d.get("url"), "doi": d.get("doi"),
                        "has_abstract": bool(d.get("abstract")), "relevant_text": best})
        return out

def doi_of(link):
    m = re.search(r"10\.\d{4,9}/\S+", link or "")
    return ("doi:" + m.group(0).lower()) if m else None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--query"); ap.add_argument("--queries")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--before"); ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--out", default="m22_results")
    a = ap.parse_args()
    R = Retriever(a.corpus)
    print(f"indexed {R.N} papers (avg {R.avg:.0f} tokens)")
    if a.query:
        for h in R.search(a.query, a.k, a.before, a.exclude):
            print(f"{h['rank']:>2}. [{h['score']:.2f}] {h['title']} ({(h['date'] or '')[:4]})\n    {h['relevant_text'][:200]}")
    if a.queries:
        os.makedirs(a.out, exist_ok=True)
        rows = [json.loads(l) for l in open(a.queries, encoding="utf-8") if l.strip()]
        results, lines = [], []
        for r in rows:
            claim = r.get("claim_text") or r.get("text") or r.get("query")
            before = a.before or r.get("newer_pub_date")
            excl = a.exclude + [doi_of(r.get("newer_link"))]
            hits = R.search(claim, a.k, before, excl)
            results.append({"query_id": r.get("comp_id") or r.get("sentence_id"), "claim_text": claim, "candidates": hits})
            lines += [f"=== {results[-1]['query_id']}: {claim}"] + \
                     [f"  {h['rank']:>2}. [{h['score']:.2f}] {h['title']} ({(h['date'] or '')[:4]}) {h['url']}\n      > {h['relevant_text'][:220]}" for h in hits] + [""]
        with open(f"{a.out}/m22_retrieval_results.jsonl", "w", encoding="utf-8") as f:
            for x in results: f.write(json.dumps(x, ensure_ascii=False) + "\n")
        open(f"{a.out}/m22_retrieval_results.txt", "w", encoding="utf-8").write("\n".join(lines))
        print(f"wrote {len(results)} queries -> {a.out}/")
