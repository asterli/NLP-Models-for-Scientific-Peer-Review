# M22 – BM25 Retrieval Baseline


## Run

One claim:
```
python retrieval/m22_bm25_retrieval.py --corpus data/corpus/corpus.jsonl --query "your claim here"
```

Many claims (any JSONL with a `claim_text` field):
```
python retrieval/m22_bm25_retrieval.py --corpus data/corpus/corpus.jsonl --queries outputs/m22/sample_claims.jsonl --out outputs/m22
```

In code:
```python
from m22_bm25_retrieval import Retriever
Retriever("data/corpus/corpus.jsonl").search(claim, k=10)
```

## How it works

- BM25 (k1 = 1.5, b = 0.75) over title (counted twice) + abstract.
- Lowercase, stop words removed, light stemming. Bare numbers and stats words (CI, RR, HR, p-values, …) are ignored so they don't drive matches.
- Each candidate returns: `rank`, `score`, `paper_id`, `title`, `date`, `venue`, `url`, `doi`, and `relevant_text` (the abstract sentence that overlaps the claim most).

## Early filters (preview of M23)

- `--before YYYY-MM-DD` drops candidates published on/after that date.
- `--exclude <paper id>` drops the submitted paper itself.
- With `--queries`, each claim automatically drops candidates published on/after its paper's date and the paper itself.

## Quick check (not the official M24 eval)

For the 67 M15 claim → older-paper pairs whose older paper is in the corpus, the known older paper appears in:

| Top 10 | Top 5 | Top 1 |
|---|---|---|
| 42% | 25% | 7% |

## Known issues

- Some corpus entries have mismatched abstracts (e.g., the 1977 "Self-efficacy" paper), which show up as noisy results.
- 130 of 645 papers have no abstract, so they can only match on title.
