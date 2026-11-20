"""Run F07 comparison baselines on the development split."""

import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from comparison_baselines import LABELS, PROMPTS, load_development_data, majority_label, messages, parse_answer, score


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prediction(row, label, **extra):
    return {
        "comparison_id": row["comparison_id"], "source_paper_id": row["source_paper_id"],
        "gold": row["label"], "predicted": label, **extra,
    }


def run_qwen(args, dev, run):
    import torch
    from huggingface_hub import model_info
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Run Qwen on an allocated CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This command requires a GPU supporting bfloat16, such as A100")
    if args.local_model_dir:
        model_source = args.local_model_dir.resolve()
        revision = model_source.name
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("--local-model-dir must name a cached snapshot with a 40-character revision")
        if model_source.parent.name != "snapshots" or model_source.parent.parent.name != "models--" + args.model.replace("/", "--"):
            raise ValueError("Snapshot cache path does not match the requested model")
        load_options = {"local_files_only": True}
    else:
        model_source = args.model
        revision = model_info(args.model, revision=args.revision).sha
        load_options = {"revision": revision}
    tokenizer = AutoTokenizer.from_pretrained(model_source, **load_options)
    encoded = {}
    for prompt in PROMPTS:
        encoded[prompt] = []
        for row in dev:
            tokens = tokenizer.apply_chat_template(
                messages(row["claim"], row["prior_work"], prompt),
                tokenize=True, add_generation_prompt=True,
            )
            if len(tokens) > args.max_input_tokens:
                raise ValueError(f"{row['comparison_id']}: {len(tokens)} tokens exceeds input limit; no text truncated")
            encoded[prompt].append(tokens)
    set_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        model_source, torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="sdpa", **load_options,
    ).eval()
    if max(len(ids) for batch in encoded.values() for ids in batch) + args.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Input and output exceed the model context limit")
    generation = GenerationConfig.from_dict(model.generation_config.to_dict())
    generation.update(
        do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens,
        temperature=None, top_p=None, top_k=None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    run["qwen"] = {
        "model": args.model, "revision": revision, "precision": "bfloat16",
        "local_model_dir": str(model_source) if args.local_model_dir else None,
        "local_files_only": bool(args.local_model_dir),
        "quantization": None, "attention": "sdpa", "seed": args.seed,
        "max_input_tokens": args.max_input_tokens, "generation": generation.to_dict(),
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "packages": {p: version(p) for p in ("torch", "transformers", "accelerate", "huggingface-hub")},
        "prompts": PROMPTS,
    }
    save(args.output / "run.json", run)
    results = {}
    for prompt, rows in encoded.items():
        records = []
        with (args.output / f"qwen_{prompt}.jsonl").open("x") as output:
            for row, tokens in zip(dev, rows):
                inputs = torch.tensor([tokens], device="cuda")
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=inputs, attention_mask=torch.ones_like(inputs), generation_config=generation,
                    )
                completion = generated[0, len(tokens):].tolist()
                answer = tokenizer.decode(completion, skip_special_tokens=True)
                record = prediction(
                    row, parse_answer(answer, prompt), raw_output=answer,
                    input_tokens=len(tokens), output_tokens=len(completion),
                )
                output.write(json.dumps(record) + "\n")
                output.flush()
                records.append(record)
                print(json.dumps({"prompt": prompt, "completed": len(records), "total": len(dev)}), flush=True)
        results[prompt] = score(records)
        save(args.output / f"qwen_{prompt}_metrics.json", results[prompt])
    # Prefer label-only output when development macro-F1 ties.
    winner = max(PROMPTS, key=lambda p: results[p]["macro_f1"])
    save(args.output / "selected_baseline.json", {
        "selection_split": "dev", "selection_metric": "macro_f1", "tie_break": "label",
        "prompt_name": winner, "system_prompt": PROMPTS[winner],
        "development_macro_f1": {p: m["macro_f1"] for p, m in results.items()},
        "model": args.model, "revision": revision, "precision": "bfloat16",
        "generation": generation.to_dict(), "seed": args.seed,
        "train_sha256": run["data"]["train"]["sha256"],
        "dev_sha256": run["data"]["dev"]["sha256"],
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/novelty/splits/train.jsonl")
    parser.add_argument("--dev", type=Path, default=ROOT / "data/novelty/splits/dev.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--majority-only", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-model-dir", type=Path, help="Existing Hugging Face snapshot directory")
    parser.add_argument("--seed", type=int, default=544)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args(argv)
    if args.max_input_tokens < 1 or args.max_new_tokens < 1:
        parser.error("Token limits must be positive")
    train, dev = load_development_data(args.train, args.dev)
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    run = {
        "complete": False, "evaluation_split": "dev", "labels": LABELS,
        "python": platform.python_version(), "code_commit": commit.stdout.strip() or None,
        "source_sha256": {p: digest(ROOT / p) for p in (
            "src/comparison_baselines.py", "scripts/baseline_novelty.py", "evaluation/metrics.py",
        )},
        "data": {
            name: {"sha256": digest(path), "count": len(rows), "class_counts": dict(Counter(r['label'] for r in rows))}
            for name, path, rows in (("train", args.train, train), ("dev", args.dev, dev))
        },
        "mode": "majority" if args.majority_only else "majority_and_qwen",
        "majority_label": majority_label(train), "majority_tie_break": LABELS,
    }
    save(args.output / "run.json", run)
    records = [prediction(row, run["majority_label"]) for row in dev]
    (args.output / "majority.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    save(args.output / "majority_metrics.json", score(records))
    if not args.majority_only:
        run_qwen(args, dev, run)
    run.update(complete=True, elapsed_seconds=time.perf_counter() - start)
    save(args.output / "run.json", run)
    print(f"Saved development results to {args.output}")


if __name__ == "__main__":
    main()
