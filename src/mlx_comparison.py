"""Qwen comparison inference on Apple Silicon with an MLX quantized base model."""

from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

from comparison_baselines import LABELS, PROMPTS, messages, parse_answer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MLX_MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"


def file_digest(path):
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest(model_dir, source_json):
    model_dir = Path(model_dir)
    source = json.loads(Path(source_json).read_text())
    revision = source.get("revision", "")
    if source.get("repo") != MLX_MODEL_ID or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Expected a pinned MLX Qwen2.5-7B-Instruct-4bit source")
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("model_type") != "qwen2" or config.get("quantization", {}).get("bits") != 4:
        raise ValueError("Expected a 4-bit Qwen2 model")
    files = sorted(p for p in model_dir.iterdir() if p.is_file() and p.suffix in (".json", ".safetensors", ".txt", ".model"))
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Base model weights are missing")
    return {
        "model": MODEL_ID, "repo": source["repo"], "revision": revision,
        "quantization": config["quantization"],
        "file_sha256": {p.name: file_digest(p) for p in files},
    }


def runtime_manifest():
    import mlx.core as mx

    return {
        "backend": "mlx", "python": platform.python_version(), "platform": platform.platform(),
        "device": dict(mx.metal.device_info()),
        "packages": {p: version(p) for p in ("mlx", "mlx-lm", "mlx-metal", "transformers", "tokenizers")},
    }


def load_model(model_dir, seed=544):
    import mlx.core as mx
    from mlx_lm import load

    if not mx.metal.is_available():
        raise RuntimeError("An Apple Silicon GPU is required")
    mx.set_cache_limit(256 * 1024**2)
    mx.random.seed(seed)
    model, tokenizer = load(str(Path(model_dir).resolve()))
    return model, tokenizer


def encode_prompt(tokenizer, claim, prior_work, prompt_name="label", max_input_tokens=4096, system_prompt=None):
    if prompt_name not in PROMPTS:
        raise ValueError("Unknown prompt")
    if any(not isinstance(text, str) or not text.strip() for text in (claim, prior_work)):
        raise ValueError("A nonempty claim and evidence passage are required")
    conversation = messages(claim, prior_work, prompt_name)
    if system_prompt is not None:
        conversation[0]["content"] = system_prompt
    tokens = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True)
    if len(tokens) > max_input_tokens:
        raise ValueError("Input exceeds token limit; supply a shorter evidence passage")
    return tokens


def predict_records(model, tokenizer, rows, prompt_name="label", max_input_tokens=4096,
                    max_new_tokens=256, system_prompt=None, output_path=None, progress=True):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    tokens = [encode_prompt(tokenizer, row["claim"], row["prior_work"], prompt_name,
                            max_input_tokens, system_prompt) for row in rows]
    context = model.args.max_position_embeddings
    if any(len(t) + max_new_tokens > context for t in tokens):
        raise ValueError("Input and output exceed the model context limit")
    model.eval()
    records = []
    destination = Path(output_path).open("x") if output_path is not None else None
    try:
        for row, encoded in zip(rows, tokens):
            start = time.perf_counter()
            answer = generate(model, tokenizer, prompt=encoded, max_tokens=max_new_tokens,
                              sampler=make_sampler(temp=0), verbose=False)
            record = {
                "comparison_id": row["comparison_id"], "source_paper_id": row["source_paper_id"],
                "gold": row["label"], "predicted": parse_answer(answer, prompt_name), "raw_output": answer,
                "input_tokens": len(encoded),
                "output_tokens": len(tokenizer.encode(answer, add_special_tokens=False)),
                "elapsed_seconds": time.perf_counter() - start,
            }
            records.append(record)
            if destination:
                destination.write(json.dumps(record) + "\n")
                destination.flush()
            if progress:
                print(json.dumps({"prompt": prompt_name, "completed": len(records), "total": len(rows)}), flush=True)
    finally:
        if destination:
            destination.close()
    return records


class MLXComparisonClassifier:
    def __init__(self, adapter_path, local_model_dir):
        from mlx_lm.tuner.utils import load_adapters

        adapter_path, local_model_dir = Path(adapter_path), Path(local_model_dir)
        self.settings = json.loads((adapter_path / "classifier_config.json").read_text())
        if self.settings["labels"] != list(LABELS) or self.settings["model"] != MODEL_ID:
            raise ValueError("Unsupported classifier configuration")
        for name, expected in self.settings["model_provenance"]["file_sha256"].items():
            if file_digest(local_model_dir / name) != expected:
                raise ValueError(f"Base model file differs from training: {name}")
        self.model, self.tokenizer = load_model(local_model_dir, self.settings["seed"])
        self.model = load_adapters(self.model, str(adapter_path)).eval()

    def predict(self, claim, prior_work, evidence_kind="passage"):
        if evidence_kind not in ("passage", "abstract"):
            raise ValueError("Supply passage or abstract evidence; title-only evidence is not supported")
        row = {"comparison_id": "inference", "source_paper_id": "inference", "claim": claim,
               "prior_work": prior_work, "label": "distinct"}
        record = predict_records(
            self.model, self.tokenizer, [row], self.settings["prompt_name"],
            self.settings["max_input_tokens"], self.settings["generation"]["max_new_tokens"],
            self.settings["system_prompt"], progress=False,
        )[0]
        if record["predicted"] not in LABELS:
            raise RuntimeError(f"Invalid classifier response: {record['raw_output']!r}")
        return {"label": record["predicted"]}
