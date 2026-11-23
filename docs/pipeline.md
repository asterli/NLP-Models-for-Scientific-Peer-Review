# F12: end-to-end pipeline

`scripts/run_pipeline.py` runs paper text through the actual components and saves machine-readable results plus an evidence-linked report:

```
paper (sentences with IDs)
  → F06 claim extractor      src/claims/extractor.py      (Qwen2.5-7B + M11 LoRA, CUDA)
  → F11 retrieval            retrieval/f11_retriever.py   (F10-filtered BM25 over data/corpus/corpus.jsonl)
  → F09 comparison           src/mlx_comparison.py        (Qwen2.5-7B 4-bit + F08 LoRA, Apple MLX)
  → results.json, predictions.jsonl, report.md
```

The pipeline calls these components as their owners published them; it does not retrain or change them. Code: `src/pipeline/` (`core.py` runs the steps, `components.py` adapts each component, `report.py` writes the report). Tests: `tests/test_pipeline.py` (no GPU or model needed).

## Two machines

The claim extractor's adapter is a PEFT/CUDA adapter on the cluster. The comparison classifier's adapter only loads in MLX on Apple Silicon. No single machine currently runs both, so the normal route has two steps.

**1. On the GPU cluster:** extract claims (`requirements-train.txt` environment).

```bash
python scripts/run_pipeline.py extract --split test \
    --claims-adapter /project2/xiangren_1987/group_40/runs/m11/r16_lr2e-4_s544/best \
    --out outputs/claims_test.json
```

This writes a claims file that lists every paper, including papers with zero claims and papers where extraction failed. It also records the adapter hashes, the raw model output and any sentences the model left unlabeled. Copy the file to the Mac.

**2. On an Apple Silicon Mac:** retrieve, compare and report. Set up the comparison environment and pinned base model first; see [comparison_local.md](comparison_local.md).

```bash
source .venv-comparison/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py run --split test \
    --claims-file outputs/claims_test.json \
    --local-model-dir /path/to/qwen-4bit \
    --out-dir outputs/pipeline_test
```

`--claims-file` also accepts an F06 predictions JSONL, such as `data/claims/predictions/finetuned_test_r16_lr2e-4_s544.jsonl` (the committed M12 fine-tuned test output). That format only lists papers that have claims. A paper missing from it is reported as an extraction error, not as "no claims".

On a machine that has both a CUDA GPU and MLX, `--claims-adapter` runs extraction live inside `run`.

## Options

| Option | Meaning |
|---|---|
| `--paper-id ID` (repeatable), `--split train/dev/test`, `--paper-file F`, `--text-file F --title T --date YYYY-MM-DD [--doi D]` | Which papers to run. Use exactly one. `--split` uses the frozen M08 claim splits. A text file has paragraphs separated by blank lines. A paragraph that is only `Abstract` or `Introduction` starts that section. A date is required so retrieval can drop newer work. |
| `--k 10` | Candidates retrieved per claim. F11 evaluated Recall@10. |
| `--compare-top N` | Compare only the top N candidates per claim. The default compares all candidates. |
| `--uncertain exclude/flag` | F10 policy for works whose date cannot confirm they are older. The default, `exclude`, is the setting that was evaluated. |
| `--comparison-adapter` | Defaults to the frozen `artifacts/comparison/local_20261114/adapter`. |
| `--no-comparison` | Runs extraction and retrieval only. Pairs are recorded as `not_run`, with no labels. |
| `--out-dir` | Must be new. The pipeline refuses to overwrite earlier results. |

`python scripts/run_sample.py` is a quick no-GPU demonstration. It runs one test paper with the saved F06 predictions and live retrieval, and skips comparison.

The exit code is 1 if any paper, claim or pair hit an error. Outputs are still written.

## What is compared

For each claim, the classifier sees the claim sentence and the candidate's `evidence`. This is F10's best-matching sentence from the older work's abstract, sent as `evidence_kind="passage"`. We compare against this sentence rather than the whole abstract because the classifier was trained on short older passages (median 29 words, maximum 81); whole corpus abstracts have a median of 213 words. Evidence is never truncated or generated. An input that exceeds the classifier's 4,096-token limit is recorded as an error for that pair.

## Empty and error cases

Nothing is invented when a step has nothing to work with. Each level records a status:

| Level | Status | Meaning |
|---|---|---|
| paper | `ok` | Claims were extracted. |
| paper | `no_claims` | The extractor found no claim sentences. Nothing is retrieved. |
| paper | `extraction_error` | The extractor failed, or returned a claim whose sentence ID or text does not match the paper. The pipeline checks every claim against the paper's own sentences. |
| claim | `retrieved` | Candidates were found. |
| claim | `no_eligible_candidates` | No corpus work passed the F10 filters. `excluded_counts` says why. |
| claim | `retrieval_error` | Retrieval failed for this claim. |
| pair | `compared` | Only this status carries a label: `overlapping`, `distinct` or `insufficient_evidence`. |
| pair | `no_evidence` | The candidate is title-only, so there was no passage to compare. |
| pair | `not_run` | The pair was not compared because of `--no-comparison` or because it fell below `--compare-top`. |
| pair | `error` | The classifier rejected the input or returned an invalid label. The detail is kept. |

`insufficient_evidence` is a model decision about supplied text. It is never used to fill in for a pair that was skipped or failed. Classifier setup failures, such as a missing GPU or base-model files that do not match the frozen hashes, stop the run.

## Outputs

- **`results.json`:** the full nested record, paper → claims → candidates → comparison. It also holds a `summary` of status and label counts, and run provenance:
  - `run.code`: git commit, branch, and uncommitted files
  - `run.environment`: Python, platform and package versions
  - `data`: papers source and its SHA-256, plus paper IDs
  - `components.claims`: adapter or claims-file hashes, plus the extractor provenance carried over from `extract`
  - `components.retrieval`: corpus SHA-256, work count, BM25 settings and date policy
  - `components.comparison`: base model repo and revision, adapter hashes, labels, seed and epoch
  - `settings`: the run settings
- **`predictions.jsonl`:** one row per claim–candidate pair. Each row is in the shared Novelty Prediction schema (`claim_id`, `prior_paper_id`, `label`, `score: null`, `reason: null`), plus `paper_id`, `sentence_id`, `rank`, `retrieval_score`, `evidence`, `status` and `detail`.
- **`report.md`:** one section per paper and claim, with the original sentence and its ID. A table lists each candidate's linked title, date, BM25 score, quoted evidence passage and the comparison label or reason it was skipped.

Candidates keep F11's `paper_id`, `url`, `doi`, `pmid`, `date`, `evidence_type`, `eligibility`, `flags` and `duplicate_ids`.

## Limitations

- **What a label covers:** a label compares one claim with one abstract sentence. It does not judge a paper's novelty. The classifier returns no score. The retrieval score is BM25 rank strength, not confidence.
- **Corpus coverage:** the corpus is mostly works the source papers already cite. A claim with no overlapping candidate says nothing about the wider literature.
- **Retrieval misses:** F11 Recall@10 is 0.652 on 22 queries. A relevant prior work can be missing from the top 10.
- **Evidence splitting:** the F10 evidence sentence comes from a simple splitter. A few abstracts with missing spaces after periods yield very long "sentences". These are passed through unchanged, so they are outside the classifier's training distribution.
- **Errors compound:** extraction errors (F06 test F1 0.86), retrieval misses and classifier errors (F09 test macro-F1 0.634) combine. F13 separates them on held-out papers.
- **Hardware:** a single-command live run needs CUDA and MLX together, which no single machine provides. The two-step route is the supported path.
