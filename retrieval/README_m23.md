# F10 – Retrieval filters

`retrieval/m23_filters.py` wraps the M22 BM25 retriever and keeps only prior work that is eligible to compare against a claim. Tests are in `tests/test_retrieval_filters.py`.

## What gets excluded

| Reason (`excluded` count key) | Rule |
|---|---|
| `not_scholarly_work` | The record has no title, or its OpenAlex type is `paratext` (journal issue pages, editorial boards). |
| `submitted_paper` | It's the submitted paper itself. Matched by DOI, PMID, PMCID or OpenAlex ID, or by the same normalized title, which catches other versions such as a preprint. |
| `same_date` / `after_cutoff` | It was published on or after the submitted paper's date. |
| `date_uncertain` | The date can't confirm it is older (see the date policy below). |
| `duplicate_version` | It's a lower-ranked copy of a paper already kept: same DOI, PMID, PMCID or OpenAlex ID, or the same normalized title **and** the same year. The same title in different years is kept, because those are different papers. |

## Date policy

The cutoff is the submitted paper's date, and a candidate is eligible only if it is **certainly** published before that date. First, each date is read as the range of days it could mean:

| Candidate date | Read as | `date_precision` |
|---|---|---|
| `YYYY-MM-DD` | that one day | `day` |
| `YYYY-01-01` | the whole year (OpenAlex uses Jan 1 when only the year is known) | `year` |
| year only | the whole year | `year` |
| none | unknown | `missing` |

Then the candidate is sorted into one of four cases:

| Case | Example (cutoff 2020-06-15) | Result |
|---|---|---|
| Ends before the cutoff | 2020-06-14, or year-only 2019 | eligible |
| Same day | 2020-06-15 | excluded (`same_date`) |
| Later | 2021-02-01, or year-only 2021 | excluded (`after_cutoff`) |
| Can't tell | year-only 2020, no date, or no submitted date | uncertain |

Uncertain dates are never silently treated as eligible:
- `uncertain="exclude"` (the default) drops them and counts them.
- `uncertain="flag"` returns them with `eligibility: "uncertain"`, so a person can check them.

Set `jan1_is_year_only=False` to trust Jan 1 dates as real days.

## Returned candidate fields

- **Identity and sources:** `rank`, `score`, `paper_id`, `title`, `date`, `year`, `date_precision`, `venue`, `type`, `url`, `doi`, `pmid`, `openalex_id`.
- **`evidence`:** the abstract sentence that best matches the claim (`evidence_type: "abstract_sentence"`). Title-only records get `evidence: null` and `evidence_type: "title_only"`. The title is never used as a passage.
- **`eligibility`:** `eligible`, or `uncertain` (only with the `flag` policy).
- **`flags`:** any of `date_year_only`, `date_missing`, `date_uncertain`, `no_abstract`.
- **`cited_by`, `cited_by_submitted`:** the source papers that cite this work, and whether the submitted paper is one of them.
- **`duplicate_ids`:** other records merged into this one.

## Run

```
python retrieval/m23_filters.py --corpus data/corpus/corpus.jsonl --query "Housing First reduced ED visits" \
    --submitted-doi 10.1007/s11524-021-00550-1 --submitted-date 2021-06-28
python retrieval/m23_filters.py --corpus data/corpus/corpus.jsonl \
    --queries retrieval/outputs/m22/sample_claims.jsonl --out retrieval/outputs/m23 [--uncertain flag]
pytest tests/test_retrieval_filters.py
```

In code:

```python
from m23_filters import FilteredRetriever
FR = FilteredRetriever("data/corpus/corpus.jsonl")
hits, excluded = FR.search(claim, {"doi": "...", "date": "YYYY-MM-DD", "title": "..."}, k=10, explain=True)
```

## Tests

17 tests in total.

The fake-corpus tests cover:
- **Self matches:** by DOI (including a DOI given as a URL), PMID, and title of another version.
- **Dates:** before, same day and after; year-only in the previous, same and next year; missing dates; a missing submitted date; both uncertain policies.
- **Duplicates:** DOI (case-insensitive), PMID, same title in the same year (merged) and in a different year (kept).
- **Evidence:** title-only records get no passage, and abstract evidence is a real abstract sentence.
- **Other:** non-scholarly records are excluded, and every candidate has traceable fields.

One more test runs all 20 source papers against the real corpus. It checks that no result is the source paper itself, dated on or after it, or a title-only record with a passage.

On the 8 M22 sample claims, the filters removed 3 submitted-paper matches and 11 later papers. One year-only result fell in the same year as its source paper and was dropped as uncertain: "The Long-Term Effects of a Housing First Intervention…", dated 2021-01-01, for claim M14-11.
