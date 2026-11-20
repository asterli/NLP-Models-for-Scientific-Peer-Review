"""Train MLX Qwen LoRA adapters using training data and select checkpoints on development data."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from comparison_baselines import LABELS, PROMPTS, load_development_data
from evaluation.comparison_results import evaluate_predictions
from mlx_comparison import MODEL_ID, file_digest, load_model, model_manifest, predict_records, runtime_manifest, encode_prompt
from qwen_comparison import training_example


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def loss_fn(model, inputs, labels):
    import mlx.core as mx
    import mlx.nn as nn

    logits = model(inputs[:, :-1]).astype(mx.float32)
    mask = labels[:, 1:] != -100
    targets = mx.maximum(labels[:, 1:], 0)
    return (nn.losses.cross_entropy(logits, targets) * mask).sum() / mask.sum()


def train_epoch(model, examples, optimizer, accumulation, rng):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_map

    order = list(range(len(examples)))
    rng.shuffle(order)
    model.train()
    value_and_grad = nn.value_and_grad(model, loss_fn)
    losses = []
    for start in range(0, len(order), accumulation):
        indices = order[start:start + accumulation]
        gradients = None
        for index in indices:
            example = examples[index]
            loss, current = value_and_grad(model, mx.array([example["input_ids"]]), mx.array([example["labels"]]))
            current = tree_map(lambda x: x / len(indices), current)
            gradients = current if gradients is None else tree_map(lambda x, y: x + y, gradients, current)
            mx.eval(loss, gradients)
            value = loss.item()
            if not math.isfinite(value):
                raise RuntimeError("Non-finite training loss")
            losses.append(value)
        gradients, norm = optim.clip_grad_norm(gradients, 1.0)
        mx.eval(norm)
        if not math.isfinite(norm.item()):
            raise RuntimeError("Non-finite gradient norm")
        optimizer.update(model, gradients)
        mx.eval(model.trainable_parameters(), optimizer.state)
        print(json.dumps({"examples": min(start + accumulation, len(order)),
                          "loss": sum(losses[-len(indices):]) / len(indices), "gradient_norm": norm.item(),
                          "peak_memory_gib": mx.get_peak_memory() / 2**30}), flush=True)
    return sum(losses) / len(losses)


def save_adapter(path, model, tokenizer, adapter_config, classifier_config):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    path.mkdir(exist_ok=True, parents=True)
    mx.save_safetensors(str(path / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    tokenizer.save_pretrained(str(path))
    save(path / "adapter_config.json", adapter_config)
    save(path / "classifier_config.json", classifier_config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/novelty/splits/train.jsonl")
    parser.add_argument("--dev", type=Path, default=ROOT / "data/novelty/splits/dev.jsonl")
    parser.add_argument("--local-model-dir", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=544)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    args = parser.parse_args()
    if not 0 < args.learning_rate < 1 or min(args.epochs, args.accumulation, args.max_input_tokens) < 1:
        parser.error("Invalid training settings")
    if args.output.exists():
        parser.error("Choose a new output directory")
    train, dev = load_development_data(args.train, args.dev)
    base_manifest = model_manifest(args.local_model_dir, args.model_source)

    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.tuner.trainer import grad_checkpoint

    model, tokenizer = load_model(args.local_model_dir, args.seed)
    examples = [training_example(tokenizer, row, args.max_input_tokens) for row in train]
    dev_lengths = [len(encode_prompt(tokenizer, row["claim"], row["prior_work"], max_input_tokens=args.max_input_tokens)) for row in dev]
    if max(max(len(e["input_ids"]) for e in examples), max(dev_lengths) + 256) > model.args.max_position_embeddings:
        raise ValueError("Training or development sequence exceeds the model context limit")
    adapter_config = {
        "fine_tune_type": "lora", "num_layers": len(model.layers),
        "lora_parameters": {"rank": 8, "scale": 16.0, "dropout": 0.0,
                            "keys": ["self_attn.q_proj", "self_attn.v_proj"]},
    }
    model.freeze()
    linear_to_lora_layers(model, len(model.layers), adapter_config["lora_parameters"])
    grad_checkpoint(model.layers[0])
    optimizer = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.0, bias_correction=True)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    manifest = {
        "complete": False, "reload_verified": False, "model": MODEL_ID, "model_provenance": base_manifest,
        "labels": list(LABELS), "system_prompt": PROMPTS["label"], "prompt_name": "label",
        "target": "label_tokens_and_eos", "precision": "mlx_4bit", "quantization": base_manifest["quantization"],
        "generation": {"max_new_tokens": 256, "temperature": 0.0}, "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": 256,
        "seed": args.seed, "learning_rate": args.learning_rate, "epochs": args.epochs,
        "batch_size": 1, "gradient_accumulation": args.accumulation,
        "optimizer": "AdamW", "bias_correction": True, "weight_decay": 0.0, "max_gradient_norm": 1.0,
        "lora": adapter_config,
        "trainable_parameters": sum(p.size for _, p in tree_flatten(model.trainable_parameters())),
        "selection_metric": "development_macro_f1", "checkpoint_tie_break": "earlier_epoch",
        "runtime": runtime_manifest(),
        "data": {name: {"sha256": file_digest(path), "count": len(rows),
                         "class_counts": dict(Counter(r["label"] for r in rows))}
                 for name, path, rows in (("train", args.train, train), ("dev", args.dev, dev))},
        "source_sha256": {p: file_digest(ROOT / p) for p in (
            "scripts/finetune_comparison_mlx.py", "src/mlx_comparison.py", "src/qwen_comparison.py",
            "src/comparison_baselines.py", "evaluation/comparison_results.py", "evaluation/metrics.py")},
    }
    save(args.output / "training_run.json", manifest)
    best = -math.inf
    history = []
    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        mean_loss = train_epoch(model, examples, optimizer, args.accumulation, rng)
        records = predict_records(model, tokenizer, dev, output_path=args.output / f"dev_epoch_{epoch}.jsonl",
                                  max_input_tokens=args.max_input_tokens)
        metrics = evaluate_predictions(dev, records)
        save(args.output / f"metrics_epoch_{epoch}.json", metrics)
        history.append({"epoch": epoch, "loss": mean_loss, "macro_f1": metrics["macro_f1"]})
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            manifest.update(best_epoch=epoch, best_development_macro_f1=best)
            classifier_config = {key: manifest[key] for key in (
                "model", "model_provenance", "labels", "system_prompt", "prompt_name", "precision", "seed",
                "generation", "max_input_tokens", "max_new_tokens", "best_epoch", "best_development_macro_f1")}
            save_adapter(args.output / "adapter", model, tokenizer, adapter_config, classifier_config)
        manifest.update(history=history, elapsed_seconds=time.perf_counter() - started,
                        peak_memory_gib=mx.get_peak_memory() / 2**30)
        save(args.output / "training_run.json", manifest)
        print(json.dumps(history[-1]), flush=True)
    manifest.update(complete=True, elapsed_seconds=time.perf_counter() - started)
    save(args.output / "training_run.json", manifest)


if __name__ == "__main__":
    main()
