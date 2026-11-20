"""Callable claim extractor (M12 / F06 hand-off to the pipeline).

    from claims.extractor import ClaimExtractor
    extractor = ClaimExtractor("/project2/xiangren_1987/group_40/runs/m11/r16_lr2e-4_s544/best")
    claims = extractor.extract(paper)        # paper: a Paper record (docs/data-schema.md)

Input: a Paper dict with `paper_id`, `title` and `sentences` (each with `sentence_id`, `section`, `text`).
Output: Claim dicts (`claim_id`, `paper_id`, `sentence_id`, `text`), in reading order. `text` is the
original sentence, never model-generated text. A paper with no sentences, or no claims, returns [].

Same prompt, decoding and parser as scripts/predict_claims.py, which produced the M12 test numbers.
Needs a GPU. Load the model once and call extract() per paper.
"""
from __future__ import annotations

from pathlib import Path

from .prompting import claim_rows, messages, parse_all_labels, parse_claim_ids

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class ClaimExtractor:
    def __init__(self, adapter_path, model=DEFAULT_MODEL, load_4bit=False, max_new_tokens=700, generate=None):
        """`generate` (messages, max_new_tokens) -> str replaces the model; used by the no-GPU tests."""
        self.max_new_tokens = max_new_tokens
        self.last_raw = None        # raw model text of the last extract() call
        self.last_unlabeled = []    # sentence_ids the model gave no CLAIM/BACKGROUND label (treated as not claims)
        if generate is not None:
            self._generate = generate
            return
        if not (Path(adapter_path) / "adapter_config.json").is_file():
            raise FileNotFoundError(f"no LoRA adapter at {adapter_path}")
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        gpu = torch.cuda.get_device_name(0)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() and "V100" not in gpu else torch.float16
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                                   bnb_4bit_use_double_quant=True) if load_4bit else None
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(model)
        base = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, quantization_config=quant, device_map="cuda")
        self._model = PeftModel.from_pretrained(base, str(adapter_path)).eval()
        self._generate = self._generate_hf

    def _generate_hf(self, msgs, max_new_tokens):
        prompt = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self._tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with self._torch.no_grad():
            gen = self._model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                       pad_token_id=self._tok.eos_token_id)
        return self._tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def extract(self, paper: dict) -> list[dict]:
        self.last_raw, self.last_unlabeled = None, []
        for field in ("paper_id", "title"):
            if not isinstance(paper.get(field), str) or not paper[field].strip():
                raise ValueError(f"paper is missing `{field}`")
        sentences = paper.get("sentences") or []
        if not sentences:
            return []
        ids = []
        for s in sentences:
            for field in ("sentence_id", "section", "text"):
                if not isinstance(s.get(field), str) or not s[field].strip():
                    raise ValueError(f"{paper['paper_id']}: a sentence is missing `{field}`")
            ids.append(s["sentence_id"])
        if len(set(ids)) != len(ids):
            raise ValueError(f"{paper['paper_id']}: duplicate sentence_id")
        # one output line per sentence (~8 tokens); only raises the budget for papers longer than the M12 ones
        budget = max(self.max_new_tokens, 10 * len(ids) + 20)
        raw = self._generate(messages(paper), budget)
        self.last_raw = raw
        labeled = parse_all_labels(raw, set(ids))
        self.last_unlabeled = [sid for sid in ids if sid not in labeled]
        return claim_rows(paper, parse_claim_ids(raw, set(ids)))
