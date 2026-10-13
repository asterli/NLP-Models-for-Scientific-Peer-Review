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

## Components

- `data/`: datasets
- `src/`: NLP component implementations
- `scripts/`: runnable commands
- `evaluation/`: evaluation code and metrics
- `models/`: model configuration and artifact references
- `outputs/`: generated results
- `docs/`: documentation

## Reproducibility

Record dataset versions, model configurations, random seeds, and evaluation settings as experiments are added.
