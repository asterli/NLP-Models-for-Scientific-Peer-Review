"""Run the saved MLX classifier or verify its development predictions after reloading."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from comparison_baselines import PROMPTS, load_split
from evaluation.comparison_results import evaluate_predictions, read_predictions
from mlx_comparison import MLXComparisonClassifier, file_digest, predict_records


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def validate_saved_settings(settings, run):
    for key in ("model", "model_provenance", "labels", "system_prompt", "prompt_name", "precision",
                "seed", "generation", "max_input_tokens", "max_new_tokens", "best_epoch",
                "best_development_macro_f1"):
        if settings[key] != run[key]:
            raise ValueError(f"Saved classifier {key} differs from the training run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--local-model-dir", type=Path, required=True)
    parser.add_argument("--claim")
    parser.add_argument("--prior-work")
    parser.add_argument("--evidence-kind", choices=("passage", "abstract", "title"), default="passage")
    parser.add_argument("--verify-run", type=Path)
    parser.add_argument("--dev", type=Path, default=ROOT / "data/novelty/splits/dev.jsonl")
    args = parser.parse_args()
    if args.verify_run:
        if args.claim is not None or args.prior_work is not None:
            parser.error("Choose prediction or reload verification")
        if args.adapter.resolve() != (args.verify_run / "adapter").resolve():
            parser.error("Adapter must belong to the run being verified")
        run_path = args.verify_run / "training_run.json"
        run = json.loads(run_path.read_text())
        run["reload_verified"] = False
        save(run_path, run)
        save(args.verify_run / "reload_check.json", {"passed": False, "fresh_process": True, "status": "started"})
        if not run["complete"]:
            raise ValueError("Training did not finish")
        if file_digest(args.dev) != run["data"]["dev"]["sha256"]:
            raise ValueError("Development data differs from the training run")
        if run["prompt_name"] != "label" or run["system_prompt"] != PROMPTS["label"]:
            raise ValueError("Training prompt does not match the label-only protocol")
        settings = json.loads((args.adapter / "classifier_config.json").read_text())
        validate_saved_settings(settings, run)
        rows = load_split(args.dev, "dev")
        expected = read_predictions(args.verify_run / f"dev_epoch_{run['best_epoch']}.jsonl")
        evaluate_predictions(rows, expected)
    elif not args.claim or not args.prior_work:
        parser.error("Provide --claim and --prior-work")

    classifier = MLXComparisonClassifier(args.adapter, args.local_model_dir)
    if not args.verify_run:
        print(json.dumps(classifier.predict(args.claim, args.prior_work, args.evidence_kind)))
        return
    validate_saved_settings(classifier.settings, run)
    records = predict_records(
        classifier.model, classifier.tokenizer, rows, classifier.settings["prompt_name"],
        classifier.settings["max_input_tokens"], classifier.settings["max_new_tokens"],
        classifier.settings["system_prompt"],
    )
    metrics = evaluate_predictions(rows, records)
    old = {r["comparison_id"]: r for r in expected}
    mismatches = [r["comparison_id"] for r in records if
                  (r["predicted"], r["raw_output"]) !=
                  (old[r["comparison_id"]]["predicted"], old[r["comparison_id"]]["raw_output"])]
    predictions_path = args.verify_run / "reload_predictions.jsonl"
    predictions_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    report = {
        "passed": not mismatches, "fresh_process": True, "count": len(records),
        "mismatched_ids": mismatches, "metrics": metrics,
        "adapter_sha256": file_digest(args.adapter / "adapters.safetensors"),
        "predictions_sha256": file_digest(predictions_path),
        "source_sha256": file_digest(Path(__file__)),
    }
    save(args.verify_run / "reload_check.json", report)
    if mismatches:
        raise RuntimeError("Reloaded outputs differ; inspect reload_check.json")
    run["reload_verified"] = True
    save(run_path, run)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
