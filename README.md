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

## Run the Sample

```bash
python scripts/run_sample.py
```

The command writes a placeholder result to `outputs/sample_output.json`.

## Data Formats

See `docs/data-schema.md` for the shared JSON fields.

## Source Papers (M05)

- `data/papers/papers.jsonl`: the 20 source papers (abstract + introduction, split into sentences), in the format in `docs/data-schema.md`. Label against this file; don't edit it by hand.
- Why each paper was picked: `data/raw/selected_pmcids.txt`. Search results: `data/raw/candidates.csv`.
- Sentence IDs must not change once labeling starts, so never run `fetch --overwrite` on this file. To test a parser change, rebuild into a separate file and compare with `scripts/diff_papers.py` (usage at the top of that script).
- For M08: PMC12314730 and PMC8382791 (both At Home/Chez Soi) must go in the same split.

## Components

- `data/`: datasets (`papers/` source papers, `raw/` search results and picks, `claims/` labeling guide)
- `src/papers/`: paper collection and parsing (M05)
- `scripts/`: runnable commands (`fetch_papers.py`, `diff_papers.py`, `run_sample.py`; the others are empty placeholders)
- `tests/`: tests, run with `pytest` (no internet needed)
- `docs/`: data schema and labeling guides
- Planned, not created yet: `evaluation/`, `models/`, `outputs/` (`run_sample.py` creates `outputs/`)

## Reproducibility

Record dataset versions, model configurations, random seeds, and evaluation settings as experiments are added.
