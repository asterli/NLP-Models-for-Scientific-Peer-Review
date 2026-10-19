"""M17 pilot check: train a few batches and save a standalone classifier."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from novelty import LABELS, encode_pairs, load_comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="JSONL: comparison_id, claim, prior_work, label")
    parser.add_argument("--base-model", required=True, help="Local pretrained model directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=544)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or not 0 < args.learning_rate < 1:
        parser.error("Steps and batch size must be positive; learning rate must be between 0 and 1")
    if args.output.exists():
        parser.error("Output already exists; choose a fresh directory")
    rows = load_comparisons(args.data)
    torch.set_num_threads(4)
    set_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, local_files_only=True, use_safetensors=True,
        num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    model.config.novelty_max_length = model.config.max_position_embeddings
    inputs = encode_pairs(tokenizer, [r["claim"] for r in rows],
                          [r["prior_work"] for r in rows], model.config.novelty_max_length)
    targets = torch.tensor([LABELS.index(r["label"]) for r in rows])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    probe_name, probe = next(iter(model.base_model.named_parameters()))
    before = probe.detach().clone()
    batches = [indices for indices in torch.randperm(len(rows)).split(args.batch_size)]
    losses, trained_ids = [], []
    model.train()
    for step in range(args.steps):
        indices = batches[step % len(batches)]
        batch = {key: value[indices] for key, value in inputs.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch, labels=targets[indices]).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(loss.item())
        trained_ids.extend(rows[i]["comparison_id"] for i in indices.tolist())
        print(json.dumps({"step": step + 1, "loss": loss.item()}), flush=True)
    delta = (probe.detach() - before).abs().max().item()
    if delta == 0:
        raise RuntimeError("Pretrained encoder weights did not change")
    model.eval()
    with torch.inference_mode():
        expected_logits = model(**inputs).logits
    if not torch.isfinite(expected_logits).all():
        raise RuntimeError("Non-finite logits after training")
    args.output.mkdir(parents=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    report = {
        "purpose": "M17 technical check only; not held-out evaluation",
        "base_model": args.base_model, "device": "cpu", "seed": args.seed,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "input_fields": ["claim", "prior_work"], "labels": list(LABELS),
        "sample_count": len(rows), "label_counts": dict(Counter(r["label"] for r in rows)),
        "trained_comparison_ids": trained_ids, "steps": args.steps,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "losses": losses, "encoder_probe": probe_name, "encoder_max_change": delta,
        "max_pair_tokens": inputs["input_ids"].shape[1], "truncated_examples": 0,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "elapsed_seconds": time.perf_counter() - started,
        "comparison_ids": [r["comparison_id"] for r in rows],
        "pre_save_logits": expected_logits.tolist(),
    }
    (args.output / "training_run.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved pilot classifier to {args.output}")


if __name__ == "__main__":
    main()
