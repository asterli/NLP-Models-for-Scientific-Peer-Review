# Local comparison classifier

The selected classifier is 4-bit Qwen2.5-7B-Instruct with a LoRA adapter, loaded through MLX on Apple Silicon. Use `artifacts/comparison/local_20261114/adapter`; no retraining is needed to use this saved model. Linux/CUDA cannot directly load this MLX adapter.

Create a Python 3.12 environment on an Apple Silicon Mac and install the dependencies. Download the pinned model:

```bash
python3.12 -m venv .venv-comparison
source .venv-comparison/bin/activate
python -m pip install -r requirements-comparison-mlx.txt
hf download mlx-community/Qwen2.5-7B-Instruct-4bit --revision c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed --local-dir /path/to/qwen-4bit
```

Run from the repository root. Output directories must be new. Keep the Mac powered and awake during execution.

```bash
python scripts/baseline_comparison_mlx.py --local-model-dir /path/to/qwen-4bit --model-source configs/comparison_mlx_model.json --output /path/to/results/f07
python scripts/finetune_comparison_mlx.py --local-model-dir /path/to/qwen-4bit --model-source configs/comparison_mlx_model.json --learning-rate 0.00001 --output /path/to/results/lr1e-5
python scripts/predict_comparison_mlx.py --local-model-dir /path/to/qwen-4bit --adapter /path/to/results/lr1e-5/adapter --verify-run /path/to/results/lr1e-5
python scripts/finetune_comparison_mlx.py --local-model-dir /path/to/qwen-4bit --model-source configs/comparison_mlx_model.json --learning-rate 0.00005 --output /path/to/results/lr5e-5
python scripts/predict_comparison_mlx.py --local-model-dir /path/to/qwen-4bit --adapter /path/to/results/lr5e-5/adapter --verify-run /path/to/results/lr5e-5
python scripts/evaluate_comparison_mlx.py freeze --baseline /path/to/results/f07 --training-runs /path/to/results/lr1e-5 /path/to/results/lr5e-5 --output /path/to/results/selection
python scripts/evaluate_comparison_mlx.py evaluate --protocol /path/to/results/selection/frozen_protocol.json --local-model-dir /path/to/qwen-4bit --output /path/to/results/test
```

F07 selects between label-only and explanation-then-label prompts using development macro-F1. F08 trains on label tokens plus EOS, with the prompt masked out: three epochs, seed 544, batch size one, accumulation eight, rank-eight LoRA on all query/value projections, scale 16, AdamW with bias correction and gradient clipping. Each learning rate's best epoch is selected on development data; ties favor the earlier epoch, then the lower learning rate across runs. No human explanations are supplied to the model.

The final evaluation compares the training-majority label, selected prompted baseline, and selected fine-tuned model on identical test pairs. Invalid output counts as an error. Saved artifacts include raw predictions, per-class precision/recall, macro-F1, confusion matrices, source-paper bootstrap comparisons, and ten error examples per method where available. These uncertainty intervals do not measure variation across training seeds. Do not use final test scores to select another model.

Secondary analyses also score the other baseline prompt and recover only unambiguous capitalization, spacing, or final-period differences. They distinguish classification gains from output-format compliance without changing model selection or primary strict scores.

Standalone inference for the selected classifier:

```bash
python scripts/predict_comparison_mlx.py --local-model-dir /path/to/qwen-4bit --adapter artifacts/comparison/local_20261114/adapter --claim "Claim text" --prior-work "Older passage"
```

This is the selected adapter recorded in `frozen_protocol.json`. `src/mlx_comparison.py` exposes `MLXComparisonClassifier(adapter_path, local_model_dir).predict(claim, prior_work, evidence_kind="passage")`, returning `{"label": ...}`. Missing or overlong inputs and invalid model output raise explicit errors. Callers must identify the evidence kind: `"title"` is rejected, but title-only text is not automatically detected. No calibrated confidence is returned. Adapter files require the pinned base model and MLX runtime; they are not PEFT/CUDA adapters.

### Pipeline integration

Run from the repository root, with the repository root and `src` on Python's import path (`PYTHONPATH=src:. python scripts/your_pipeline.py`). Load the classifier once and reuse it for successive claim–evidence pairs:

```python
from mlx_comparison import MLXComparisonClassifier

classifier = MLXComparisonClassifier(
    "artifacts/comparison/local_20261114/adapter",
    "/path/to/qwen-4bit",
)

def compare(claim, candidate):
    result = classifier.predict(
        claim["text"], candidate["evidence"], evidence_kind="passage"
    )
    return {
        "claim_id": claim["claim_id"],
        "prior_paper_id": candidate["paper_id"],
        "label": result["label"],
        "score": None,
        "reason": None,
    }
```

`claim` and `candidate` follow the [shared schema](data-schema.md). The exact output labels are `overlapping`, `distinct`, and `insufficient_evidence`. Use actual passage text in `candidate["evidence"]`; map retrieval's `relevant_text` to that field if needed. An abstract is also supported with `evidence_kind="abstract"`. Keep the original claim's paper/sentence IDs and the candidate's date, URL, and evidence alongside the prediction in the report. Retrieval scores remain on the candidate; they are not classifier confidence.

The model generates neither a reason nor a score: serialize these as JSON `null`. This is distinct from an `insufficient_evidence` prediction, which is a model decision on supplied text.

Skip classification when there is no extracted claim, no eligible candidate, or no passage/abstract; record the missing evidence in the report without inventing a label. `ValueError` reports empty input, unsupported evidence kind, or input exceeding 4,096 tokens including the rules and chat template. Supply a shorter relevant passage rather than silently truncating it. `RuntimeError` reports invalid generated labels or an unavailable Apple GPU. Model loading also fails if base files are missing or their hashes differ. Keep errors separate from valid predictions and show an unavailable/error status for that pair. The example lets exceptions propagate so the pipeline can apply its own error handling.

Model, data, source-code hashes, package versions, settings, epoch results, and reload checks are recorded with each run. Preserve those artifacts together with the saved adapter. Final training uses all training comparisons; development and test examples are never training targets.

To reproduce the published test evaluation with the included adapter, run:

```bash
python scripts/evaluate_comparison_mlx.py evaluate --protocol results/comparison/local_20261114/frozen_protocol.json --adapter artifacts/comparison/local_20261114/adapter --local-model-dir /path/to/qwen-4bit --output /path/to/new/test-replay
```

The frozen protocol retains original run locations to preserve its recorded hash. The adapter override makes replay portable; weights and configuration must still match the frozen hashes. Retraining may produce numerical differences across hardware or library versions; the saved predictions permit exact metric verification without retraining.

### Training-seed variation

Repeat the selected setting with seeds 545 and 546, selecting each checkpoint on development macro-F1. Use new output directories:

```bash
for seed in 545 546; do
  python scripts/finetune_comparison_mlx.py --local-model-dir /path/to/qwen-4bit --model-source configs/comparison_mlx_model.json --learning-rate 0.00005 --epochs 3 --seed "$seed" --output "/path/to/results/seed$seed"
  python scripts/predict_comparison_mlx.py --local-model-dir /path/to/qwen-4bit --adapter "/path/to/results/seed$seed/adapter" --verify-run "/path/to/results/seed$seed"
done
```

Compare each run's selected development metrics with seed 544 using the mean, sample standard deviation, and range of macro-F1. Saved results are in [f08/seeds](../results/comparison/local_20261114/f08/seeds/summary.json). These runs do not replace the packaged final classifier.
