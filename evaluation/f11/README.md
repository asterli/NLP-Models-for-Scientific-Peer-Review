# F11 - Retrieval evaluation

Measures how often the F10 filtered BM25 retriever (`retrieval/m23_filters.py`) finds known-relevant prior work for a claim. The main number is **Recall@10** over a frozen set of 21 claims.

Evaluation script: `scripts/f11_eval_retrieval.py`.
Callable retriever: `retrieval/f11_retriever.py`.
Tests: `tests/test_f11_eval_retrieval.py` and
`tests/test_f11_retriever.py`.

## Files

| File                                         | What it is                                                                                                                                                  |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `evaluation/f11/queries.jsonl`             | The 21 claims, plus the submitted (newer) paper's DOI, PMID, PMCID, title and date. F10 uses these to drop the paper itself and anything not older than it. |
| `evaluation/f11/qrels.jsonl`               | The answer key:`{"query_id", "relevant": [corpus paper ids]}`. 21 queries, 30 relevant pairs, 23 unique papers.                                           |
| `evaluation/f11/F11_relevance_review.xlsx` | The review record: every candidate pair, the judgment, and notes. Change judgments here first, then regenerate the qrels.                                   |

Both `.jsonl` files were committed before any BM25 results were looked at, and are not edited after seeing results.


## Callable retriever for F12

`retrieval/f11_retriever.py` provides `PriorWorkRetriever`, a reusable interface for the downstream F12 pipeline. It wraps F10's `FilteredRetriever` (`retrieval/m23_filters.py`), which uses the M22 BM25 baseline.

This is an interface to the existing retriever, **not a new retrieval or ranking model**. The corpus is loaded once when `PriorWorkRetriever` is initialized.

### Usage

Run from the repository root:

```python
import json
from retrieval.f11_retriever import PriorWorkRetriever

with open("evaluation/f11/queries.jsonl", encoding="utf-8") as f:
    query = json.loads(next(f))

retriever = PriorWorkRetriever("data/corpus/corpus.jsonl")

candidates = retriever.retrieve(
    query["claim_text"],
    query,
    k=10
)
```

The submitted-paper argument accepts either:

- A plain dictionary containing metadata such as `doi`, `pmid`, `pmcid`, `paper_id`, `title`, and `date`.
- An F03-style record containing fields such as `newer_paper_id`, `newer_link`, `newer_title`, and `newer_pub_date`.

A missing submitted-paper date raises `ValueError`, rather than silently returning no eligible candidates.

The default `uncertain="exclude"` applies F10's strict date policy. Set `uncertain="flag"` during initialization to retain and flag uncertain-date candidates.

### Returned candidates

Candidates are ordered by BM25 ranking score and contain the shared Retrieval Candidate fields from `docs/data-schema.md`:

| Field | Description |
| --- | --- |
| `paper_id` | Identifier of the retrieved older paper |
| `title` | Paper title |
| `score` | BM25 ranking score, not calibrated confidence |
| `date` | Publication date |
| `url` | Source URL |
| `evidence` | Best-matching sentence from the abstract, if available |

Additional fields include `rank`, `doi`, `pmid`, `evidence_type`, `eligibility`, `flags`, and `duplicate_ids`.

For records without abstracts, `evidence` is `None` and `evidence_type` is `"title_only"`. The retriever does not invent supporting passages.

To inspect F10's exclusion counts:

```python
candidates, excluded_counts = retriever.retrieve(
    query["claim_text"],
    query,
    k=10,
    explain=True
)
```

### Connection to F11 evaluation

`scripts/f11_eval_retrieval.py` uses `PriorWorkRetriever.retrieve()` to evaluate the same callable interface exposed to F12.

The evaluator also accesses the wrapped F10 retriever through `retriever.filtered` for full-depth ranking and miss analysis.

This keeps retrieval behavior consistent between evaluation and downstream pipeline usage.

### Tests

```bash
pytest tests/test_f11_retriever.py tests/test_f11_eval_retrieval.py
```

The callable-retriever tests use a synthetic corpus to check candidate ordering, required schema fields, filtering of submitted and newer papers, evidence handling, F03-style metadata, top-k limits, and missing-date errors.

The preliminary Recall@10 baseline was obtained before adding this wrapper. The updated evaluation script should be rerun to verify that the callable interface produces the same result.

## Run

```
python scripts/f11_eval_retrieval.py --corpus data/corpus/corpus.jsonl \
    --queries evaluation/f11/queries.jsonl --qrels evaluation/f11/qrels.jsonl \
    --out results/f11 [--k 10] [--uncertain exclude]
pytest tests/test_f11_eval_retrieval.py
```

`--uncertain exclude` (the default, same as F10) drops candidates whose date can't confirm they are older than the claim. `--uncertain flag` keeps them, which is useful for checking what the date policy costs.

Outputs in `--out`:

| File                        | Contents                                                                                                                                             |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `results.json`            | Recall@k (from `evaluation/metrics.py`), per-query recall, miss breakdown, eligible-only recall, settings, git commit                              |
| `run.jsonl`               | Top k per query,`{"query_id", "ranked": [...]}`. Scores the same through `score_predictions.py retrieval`.                                       |
| `misses.jsonl`            | One line per relevant paper not in the top k, with its reason and filtered rank                                                                      |
| `retrieval_details.jsonl` | Full F10 candidates per query, with evidence sentences. Its `excluded_counts` only covers exclusions seen before reaching k, not the whole corpus. |

## How the queries and qrels were built

**Source.** 25 claims from the F03 dev/test comparisons whose older paper is in the M21 corpus, giving 47 claim-paper candidate pairs.

**The question.** For each pair: would a reviewer checking this claim want this older paper in the top 10? This is paper-level relevance, not the passage-level novelty label. A Distinct or Insufficient-evidence pair can still be relevant. Judgments used the full abstract, since BM25 only sees title + abstract.

| Label  | Meaning                                                                                                                                                              |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Yes    | Same or closely related outcome and exposure/intervention, in a comparable population. Agreeing, contradicting, or different results on the same question all count. |
| No     | Only shares the broad topic, or a different population/intervention/outcome with no useful comparison.                                                               |
| Unsure | Not settled by the abstract. All Unsures were resolved before freezing.                                                                                              |

**Gray-zone rule.** A different intervention with the same population and outcome counts as Yes (for example, Housing First evidence for a shelter-based psychiatric services claim). These rows are marked "gray-zone rule" in the review sheet.

**Process.** Each pair was screened and judged by a human reviewer (Win), who added notes on judgment calls. These qrels are human-verified by one reviewer, not independent multi-annotator labels.

**Excluded queries.** R092, R051, W033 and W131 had no candidate paper judged relevant, so they can't be scored (0/0). They were dropped before any BM25 results, leaving 21 queries. Replacement papers were not added, because candidates found by keyword search over titles would favor a lexical retriever.

## Miss reasons

Every relevant paper outside the top k gets one reason. Eligibility is checked before score: a paper F10 removes can't count however it ranks.

| Group       | Reason                          | Meaning                                                                     |
| ----------- | ------------------------------- | --------------------------------------------------------------------------- |
| ranking     | `ranked_below_k`              | Passed F10, but BM25 ranked it past k. The filtered rank is recorded.       |
| ranking     | `no_term_overlap`             | BM25 score 0: no shared terms with the claim (M22 drops zero-score papers). |
| eligibility | `submitted_paper`             | It is the claim's own paper or another version of it.                       |
| eligibility | `same_date`, `after_cutoff` | Not older than the claim's paper.                                           |
| eligibility | `date_uncertain`              | Date can't confirm it is older (only with `--uncertain exclude`).         |
| eligibility | `not_scholarly_work`          | No title, or an OpenAlex paratext record.                                   |
| eligibility | `duplicate_version`           | Merged by F10 into another record of the same paper.                        |
| data        | `not_in_corpus`               | The qrels id is not in the corpus file.                                     |

`recall@k_eligible_only` re-scores with the eligibility-group papers removed, so ranking failures and F10 filtering are reported separately.

## Results

**Preliminary**, run on the PR #20 branch before merge. To be replaced by the official run on main.

| Metric                   | Value                                                                   |
| ------------------------ | ----------------------------------------------------------------------- |
| Recall@10                | 0.544 (21 queries; 16 of 30 relevant pairs found)                       |
| Recall@10, eligible only | 0.544                                                                   |
| Misses                   | 14: 13 `ranked_below_k`, 1 `no_term_overlap`, 0 eligibility, 0 data |

- **F10 cost nothing on this set.** Every relevant paper predates its claim's paper.
- **5 of 14 misses are near misses** (filtered rank 11-20). Recall@20 would include them.
- **8 misses are far (rank 25+).** 3 of them are gray-zone Yes judgments (R094's two papers and the R189 hotel I/Q paper), where the relevant paper describes a different intervention in different words.
- **Title-only papers are hard to reach.** The Lancet review used for W123 and W125 has no abstract in the corpus: W123 has no term overlap, and W125 lands at rank 231.
- 8 of 21 queries score 0.

## Known limits

- **Small set.** 21 queries, most with one relevant paper, so one query moves Recall@10 by about 5 points.
- **Incomplete judgments.** Only the 47 candidate claim–paper pairs were reviewed. Additional relevant papers may exist in the corpus but are not included in our qrels. Therefore, Recall@10 measures retrieval of known relevant papers, rather than recall over all relevant literature. Unjudged retrieved papers are not independently assessed for relevance.
- **One reviewer**.
- **Gray-zone leniency** Our relevance criteria include scientifically useful papers discussing different interventions or settings. These papers may be harder for lexical retrieval methods such as BM25 to rank highly. A sensitivity analysis excluding gray-zone judgments could help quantify this effect, but it would not replace the primary metric.
- **Title-only papers** (130 of 644 corpus records) can only match on title.
- **Not comparable to the M22 quick check** (42% Recall@10 on 67 M15 pairs). That used a different pair set and no F10 filtering.
- **Duplicates.** A relevant paper merged into another record counts as a `duplicate_version` miss even if the kept copy is in the top k. None of the 23 relevant papers has a duplicate in the current corpus.
