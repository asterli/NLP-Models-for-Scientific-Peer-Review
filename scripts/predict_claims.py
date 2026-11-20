"""M11/M12: reload a saved LoRA adapter and label a split, writing predictions in the shared Claim schema.

  python scripts/predict_claims.py --adapter /project2/.../runs/m11/<run>/best --split dev \
      --out data/claims/predictions/finetuned_dev.jsonl
  python scripts/score_predictions.py claims --pred data/claims/predictions/finetuned_dev.jsonl \
      --gold data/claims/splits/dev.jsonl --out results/m11_dev.json

Same prompt and parser as the fine-tuned training loop and the M10 v2 baseline, so scores are comparable.
Run on --split test only once in M12, after settings are frozen.
"""
import argparse, json, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from claims.prompting import load_split, messages, parse_claim_ids, claim_rows  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--adapter", required=True)
p.add_argument("--split", default="dev", choices=["train", "dev", "test", "all"])
p.add_argument("--out", required=True)
p.add_argument("--raw-out", default=None, help="default: <adapter>/raw_<split>.jsonl")
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--load-4bit", action="store_true", help="must match how the adapter was trained")
p.add_argument("--max-new-tokens", type=int, default=700)
args = p.parse_args()

gpu = torch.cuda.get_device_name(0)
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() and "V100" not in gpu else torch.float16
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                           bnb_4bit_use_double_quant=True) if args.load_4bit else None
tok = AutoTokenizer.from_pretrained(args.model)
base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, quantization_config=quant, device_map="cuda")
model = PeftModel.from_pretrained(base, args.adapter).eval()

papers = load_split(args.split)
raw_out = Path(args.raw_out or Path(args.adapter) / f"raw_{args.split}.jsonl")
Path(args.out).parent.mkdir(parents=True, exist_ok=True); raw_out.parent.mkdir(parents=True, exist_ok=True)
n, t0 = 0, time.time()
with open(args.out, "w") as fo, raw_out.open("w") as fr, torch.no_grad():
    for q in papers:
        prompt = tok.apply_chat_template(messages(q), tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        raw = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        ids = parse_claim_ids(raw, {s["sentence_id"] for s in q["sentences"]})
        fr.write(json.dumps({"paper_id": q["paper_id"], "adapter": args.adapter, "raw": raw, "parsed": ids}) + "\n")
        for row in claim_rows(q, ids):
            fo.write(json.dumps(row) + "\n")
        n += len(ids); print(f"{q['paper_id']}: {len(ids)}/{len(q['sentences'])} claims")
print(f"done: {n} claims from {len(papers)} papers in {time.time() - t0:.0f}s -> {args.out}")
