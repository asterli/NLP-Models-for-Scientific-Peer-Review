# M17 evaluation

`pilot.jsonl` contains the 30 labeled comparisons. Inputs are `claim` and `prior_work`; `label` is the target, and `source_pmcid` determines the held-out paper. Human explanations are not inputs.

Each run holds out one of four source papers and excludes exact claim/passage overlaps across training and evaluation. Training sizes are 24, 19, 21, and 24. Seeds: 544 and 2026. These are exploratory pilot results; the pilot was used for model and prompt selection.

| Model | Correct / 30 | Macro-F1 |
|---|---:|---:|
| Always distinct | 19 | 0.259 |
| Qwen, rules and label only | 8 | 0.226 |
| Qwen, rules and written explanation | 19 | 0.259 |
| Qwen + LoRA, seed 544 | 20 | 0.417 |
| Qwen + LoRA, seed 2026 | 20 | 0.497 |
| DistilBERT fine-tuned, either seed | 17 | 0.246 |

`reference_predictions.json` contains the predictions behind these results. Qwen predictions differ on 11 comparisons between seeds; neither seed correctly identifies either insufficient-evidence example. Scores are uncalibrated.

## Run

Use Python 3.12 and separate environments for the two dependency sets. Qwen uses MLX on an Apple Silicon Mac; DistilBERT runs on CPU. Run from the repository root:

```bash
python3.12 -m venv evaluation/m17/.venv-qwen
evaluation/m17/.venv-qwen/bin/pip install -r evaluation/m17/requirements-qwen.txt
evaluation/m17/.venv-qwen/bin/python evaluation/m17/run.py qwen

python3.12 -m venv evaluation/m17/.venv-distilbert
evaluation/m17/.venv-distilbert/bin/pip install -r evaluation/m17/requirements-distilbert.txt
evaluation/m17/.venv-distilbert/bin/python evaluation/m17/run.py distilbert
```

Model downloads use pinned revisions in `run.py`. An existing matching checkpoint can be supplied with `--model-dir PATH`. Use `--output PATH` for a fresh rerun directory. Outputs contain predictions, per-class metrics, macro-F1, confusion matrices, training logs, and Qwen adapters. `summary.json` compares rerun predictions with the reference; numerical differences can occur across hardware or library versions.

Qwen: 4-bit `mlx-community/Qwen2.5-7B-Instruct-4bit`, LoRA rank 8, scale 16, query/value projections in all 28 layers, Adam at 1e-5, 3 epochs, batch size 1. Training uses the correct label token as its target. DistilBERT: `distilbert/distilbert-base-uncased`, full fine-tuning, AdamW at 2e-5, 10 epochs, batch size 4. Both clip gradients at 1. Prompts and label mappings are in `common.py` and `qwen.py`.
