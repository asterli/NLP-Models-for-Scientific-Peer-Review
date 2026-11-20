import json
from pathlib import Path

from comparison_baselines import LABELS, PROMPTS, messages, parse_answer


MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def snapshot_revision(path):
    path = Path(path).resolve()
    revision = path.name
    if (len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision)
            or path.parent.name != "snapshots"
            or path.parent.parent.name != "models--" + MODEL_ID.replace("/", "--")):
        raise ValueError("Expected a cached Qwen2.5-7B-Instruct snapshot directory")
    return revision


def prompt_tokens(tokenizer, claim, prior_work, max_input_tokens, system_prompt=None):
    if any(not isinstance(text, str) or not text.strip() for text in (claim, prior_work)):
        raise ValueError("A nonempty claim and evidence passage are required")
    conversation = messages(claim, prior_work, "label")
    if system_prompt is not None:
        conversation[0]["content"] = system_prompt
    tokens = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True)
    if len(tokens) > max_input_tokens:
        raise ValueError("Input exceeds token limit; supply a shorter evidence passage")
    return tokens


def training_example(tokenizer, row, max_input_tokens):
    if row["label"] not in LABELS:
        raise ValueError("Unknown target label")
    prefix = prompt_tokens(tokenizer, row["claim"], row["prior_work"], max_input_tokens)
    target = tokenizer.encode(row["label"], add_special_tokens=False)
    if not target or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must provide label tokens and an end token")
    target.append(tokenizer.eos_token_id)
    return {"input_ids": prefix + target, "labels": [-100] * len(prefix) + target}


def generation_settings(model, tokenizer, max_new_tokens=256):
    from transformers import GenerationConfig

    config = GenerationConfig.from_dict(model.generation_config.to_dict())
    config.update(
        do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
        temperature=None, top_p=None, top_k=None,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        use_cache=True,
    )
    return config


def generate_label(model, tokenizer, tokens, generation):
    import torch

    inputs = torch.tensor([tokens], device=model.device)
    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs, attention_mask=torch.ones_like(inputs), generation_config=generation,
        )
    answer = tokenizer.decode(output[0, len(tokens):].tolist(), skip_special_tokens=True)
    return {"label": parse_answer(answer, "label"), "raw_output": answer}


def development_predictions(model, tokenizer, rows, max_input_tokens, generation):
    model.eval()
    records = []
    for row in rows:
        tokens = prompt_tokens(tokenizer, row["claim"], row["prior_work"], max_input_tokens)
        answer = generate_label(model, tokenizer, tokens, generation)
        records.append({
            "comparison_id": row["comparison_id"], "source_paper_id": row["source_paper_id"],
            "gold": row["label"], "predicted": answer["label"], "raw_output": answer["raw_output"],
        })
    return records


class QwenComparisonClassifier:
    def __init__(self, adapter_path, local_model_dir=None):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        self.settings = json.loads((Path(adapter_path) / "classifier_config.json").read_text())
        if self.settings["labels"] != list(LABELS) or self.settings["model"] != MODEL_ID:
            raise ValueError("Unsupported classifier configuration")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A CUDA GPU supporting bfloat16 is required")
        if local_model_dir:
            if snapshot_revision(local_model_dir) != self.settings["revision"]:
                raise ValueError("Base model revision differs from the saved classifier")
            source, options = str(local_model_dir), {"local_files_only": True}
        else:
            source, options = MODEL_ID, {"revision": self.settings["revision"]}
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(
            source, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa", **options,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False).eval()
        self.generation = GenerationConfig.from_dict(self.settings["generation"])

    def predict(self, claim, prior_work, evidence_kind="passage"):
        if evidence_kind not in ("passage", "abstract"):
            raise ValueError("Supply passage or abstract evidence; title-only evidence is not supported")
        tokens = prompt_tokens(
            self.tokenizer, claim, prior_work, self.settings["max_input_tokens"], self.settings["system_prompt"],
        )
        result = generate_label(self.model, self.tokenizer, tokens, self.generation)
        if result["label"] not in LABELS:
            raise RuntimeError(f"Invalid classifier response: {result['raw_output']!r}")
        return {"label": result["label"]}
