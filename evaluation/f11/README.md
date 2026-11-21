# F11 - Retrieval evaluation

F11 evaluates whether the existing **F10-filtered BM25 retriever** finds known-relevant older research for a scientific claim. It reports **Recall@10** and provides a callable `PriorWorkRetriever` for the downstream F12 pipeline. F11 does **not** train or introduce a new ranking model.

Evaluation script: `scripts/f11_eval_retrieval.py`.
Qrels builder: `scripts/f11_build_qrels.py`.
Callable retriever: `retrieval/f11_retriever.py`.
Tests: `tests/test_f11_eval_retrieval.py`, `tests/test_f11_build_qrels.py` and
`tests/test_f11_retriever.py`.

## Evaluation data

We selected **66 claim-paper pairs across 30 claims** from the M15 rounds 2-3 comparisons in the F03 development/test splits, requiring the older paper to be identifiable in the M21 corpus by DOI. The corpus contains **644 works**. This is a set of existing comparison pairs, **not** a complete relevance assessment of the corpus.

Two reviewers (Win and Dylan) independently assessed all 66 pairs using the same criteria and the newer paper's title/abstract plus the older paper's abstract. They marked population, intervention/exposure, and outcome as *Same*, *Related*, or *Different* (with *None* allowed for intervention). A pair counted as **relevant (Yes)** when the population and outcome were Same or Related; the intervention could differ. Reviewers recorded reasons and could disagree with the spreadsheet's suggested label.

- **Initial agreement:** 58/66 pairs (87.9%); Cohen's kappa = 0.758.
- **Adjudication:** All 8 disagreements were resolved and accepted by both reviewers.
- **Final judgments:** 35 Yes and 31 No. Eight claims had no Yes judgments, leaving **22 evaluable queries**, **35 relevant pairs**, and **25 unique relevant papers**.

**Exposure:** Dylan did not see retrieval results before labeling. Win had seen the preliminary BM25 evaluation for the first 47 pairs before their revised review from his preliminary run; the 19 added pairs were labeled without seeing results.

The final review is in `evaluation/f11/f11_relevance_review.csv` (builder input) and `.xlsx` (full review context). `scripts/f11_build_qrels.py` generates `queries.jsonl` and `qrels.jsonl` from the final CSV judgments; the generated files should not be edited manually.

## Results

Official evaluation (`results/f11/`), using the adjudicated labels:

| Metric                                             |                            Result |
| -------------------------------------------------- | --------------------------------: |
| **Macro Recall@10** (average across queries) |                  **0.6523** |
| Evaluated queries                                  |                                22 |
| Relevant pairs found in top 10                     |                           22 / 35 |
| Micro recall (pooled pairs)                        |                            0.6286 |
| Missed relevant pairs                              |                                13 |
| Miss breakdown                                     | 13 ranking; 0 eligibility; 0 data |

All 13 misses were eligible papers ranked below position 10. The earlier **0.544 Recall@10** was measured on a *different* preliminary set (21 queries, 30 relevant pairs), so the scores should **not** be interpreted as an improvement in BM25.

## Reproduce

From the repository root:

```bash
python scripts/f11_build_qrels.py --review evaluation/f11/f11_relevance_review.csv --splits-dir data/novelty/splits --out-dir evaluation/f11
python scripts/f11_eval_retrieval.py --corpus data/corpus/corpus.jsonl --queries evaluation/f11/queries.jsonl --qrels evaluation/f11/qrels.jsonl --out results/f11
pytest tests/test_f11_build_qrels.py tests/test_f11_eval_retrieval.py tests/test_f11_retriever.py
```

The evaluation writes `results.json`, `run.jsonl` (ranked paper IDs), `misses.jsonl` (miss reasons/ranks), and `retrieval_details.jsonl` (retrieved candidates and evidence). The committed `results.json` records the code commit used for the run.

For downstream use:

```python
from retrieval.f11_retriever import PriorWorkRetriever

retriever = PriorWorkRetriever("data/corpus/corpus.jsonl")
candidates = retriever.retrieve(claim_text, submitted_paper, k=10)
```

`submitted_paper` must contain a publication date and identifying metadata, or may be an F03-style `newer_*` record. Candidates are ranked by BM25 and include paper ID, title, score, date, URL, and an evidence sentence when an abstract exists. Missing abstracts yield `evidence=None`. Use `explain=True` for F10 exclusion counts.

## Limitations

- **Incomplete relevance judgments:** Only the 66 selected pairs were judged, so Recall@10 measures retrieval of **known relevant papers**, not every relevant paper in the corpus.
- **Selection scope:** The evaluation uses 66 claim-paper pairs from the F03 development and test splits, covering 30 claims. These pairs originated in M15 rounds 2-3 and were matched to the retrieval corpus using DOI-formatted paper IDs. An additional 12 unique pairs from earlier labeling rounds were identified later; their DOI links were available, but their paper IDs used short identifiers instead. These pairs were not included in the final evaluation.
- **Small, context-dependent benchmark:** Eight claims without positive judgments were excluded from recall; a few labels can materially change the score. Reviewers received more context than the claim-only BM25 ranking query, and the annotation exposure noted above may influence judgments.
- **Lexical retrieval:** BM25 can miss conceptually related work when the claim and older paper use different terminology, particularly when abstracts are missing.
