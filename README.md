# Fine-Tuned NLP Models for Automated Scientific Peer Review Assistance

## Project

We are developing an NLP pipeline that extracts claims from scientific papers, retrieves relevant prior work, and assesses claims against that work.

## Pipeline

Paper → Claim Extraction → Related-Work Retrieval → Novelty Classification

## Setup

Python 3.10+ recommended.

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Pipeline (F12)

```bash
python scripts/run_sample.py
```

This quick sample needs no GPU. It runs one held-out paper through the saved F06 claim predictions and live retrieval, skips the comparison classifier, and writes `results.json`, `predictions.jsonl` and `report.md` to `outputs/sample_<timestamp>/`.

The full pipeline is `scripts/run_pipeline.py`: paper → claim extractor → retrieval → comparison classifier → evidence-linked report. The extractor needs a CUDA GPU and the classifier needs Apple Silicon, so the full pipeline runs in two steps. See [docs/pipeline.md](docs/pipeline.md).

## Data Formats

See `docs/data-schema.md` for the shared JSON fields.

## Source Papers (M05)

- `data/papers/papers.jsonl`: the 20 source papers (abstract + introduction, split into sentences), in the format in `docs/data-schema.md`. Label against this file; don't edit it by hand.
- Why each paper was picked: `data/raw/selected_pmcids.txt`. Search results: `data/raw/candidates.csv`.
- Sentence IDs must not change once labeling starts, so never run `fetch --overwrite` on this file. To test a parser change, rebuild into a separate file and compare with `scripts/diff_papers.py` (usage at the top of that script).
- For M08: PMC12314730 and PMC8382791 (both At Home/Chez Soi) must go in the same split.

## Background Corpus (M21)

- `data/corpus/corpus.jsonl`: 644 older works cited by the 20 source papers, from OpenAlex. Format and caveats: `docs/data-schema.md`. Per-paper coverage: `data/corpus/coverage.json`.
- Rebuild with `python scripts/build_corpus.py`. The first run needs internet; lookups are cached in `data/raw/openalex/`. An OpenAlex API key is optional: copy `.env.example` to `.env`.

## Components

- [F12 end-to-end pipeline: commands, outputs, empty/error cases](docs/pipeline.md)
- [F07–F09 comparison classifier: results and saved model](results/comparison/local_20261114/README.md)
- [Reproduce training, evaluate, or run comparison inference](docs/comparison_local.md)
- [M17 pilot fine-tuning: status, results, and usage](docs/m17.md)

- `data/`: datasets (`papers/` source papers, `raw/` search results and picks, `claims/` labeling guide)
- `src/papers/`: paper collection and parsing (M05)
- `src/corpus/`: background corpus from OpenAlex (M21)
- `scripts/`: runnable commands, including `run_pipeline.py` (F12) and `run_sample.py`
- `src/pipeline/`: F12 pipeline (orchestration, component adapters, report, provenance)
- `tests/`: tests, run with `pytest` (no internet needed)
- `docs/`: data schema and labeling guides
- `outputs/`: pipeline runs (`run_sample.py` and `run_pipeline.py` write here)

## Reproducibility

Record dataset versions, model configurations, random seeds, and evaluation settings as experiments are added.
