"""M11: LoRA fine-tune Qwen2.5-7B-Instruct as a per-sentence CLAIM/BACKGROUND labeler.

One training example per paper: the M10 v2 prompt (numbered sentences) -> gold "sXXX: LABEL" lines.
Loss is computed on the assistant turn only. After every epoch the model labels the DEV split with
greedy decoding and is scored with evaluation/metrics.claim_scores; the adapter with the best dev F1
is kept as <out>/best, the final epoch as <out>/last. Test is never touched here (that is M12).

  python scripts/train_claims.py --out /project2/xiangren_1987/group_40/runs/m11/r16_lr2e-4_s544
  python scripts/predict_claims.py --adapter .../best --split dev --out data/claims/predictions/finetuned_dev.jsonl
"""
import argparse, json, random, subprocess, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))
from claims.prompting import load_split, messages, target_text, parse_claim_ids, claim_rows  # noqa: E402
from evaluation.metrics import claim_scores  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--out", required=True, help="run directory (outside the repo, e.g. .../group_40/runs/m11/<name>)")
p.add_argument("--train-split", default="train")
p.add_argument("--dev-split", default="dev")
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--grad-accum", type=int, default=2, help="papers per optimizer step (batch size is 1 paper)")
p.add_argument("--lora-r", type=int, default=16)
p.add_argument("--lora-alpha", type=int, default=32)
p.add_argument("--lora-dropout", type=float, default=0.05)
p.add_argument("--targets", default="q_proj,k_proj,v_proj,o_proj")
p.add_argument("--seed", type=int, default=544)
p.add_argument("--load-4bit", action="store_true", help="QLoRA; use on V100 (32 GB) or if memory is tight")
p.add_argument("--max-new-tokens", type=int, default=700)
p.add_argument("--select-by", default="f1", choices=["f1", "recall", "precision"])
args = p.parse_args()

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
log = (out / "train_log.jsonl").open("a")
def emit(**rec):
    rec["t"] = round(time.time() - T0, 1); print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()

T0 = time.time()
random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
assert torch.cuda.is_available(), "no GPU visible"
gpu = torch.cuda.get_device_name(0)
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() and "V100" not in gpu else torch.float16
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
                           bnb_4bit_use_double_quant=True) if args.load_4bit else None

train = load_split(args.train_split); dev = load_split(args.dev_split)
gold_dev = {(q["paper_id"], sid): lab for q in dev for sid, lab in q["labels"].items()}
try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
except Exception:
    commit = None
config = vars(args) | {"gpu": gpu, "dtype": str(dtype), "git_commit": commit,
                       "train_papers": [q["paper_id"] for q in train], "dev_papers": [q["paper_id"] for q in dev],
                       "train_sentences": sum(len(q["sentences"]) for q in train),
                       "train_claims": sum(v == "CLAIM" for q in train for v in q["labels"].values())}
(out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
emit(stage="data", train_papers=len(train), dev_papers=len(dev), train_sentences=config["train_sentences"])

tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, quantization_config=quant, device_map="cuda")
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False
model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                         target_modules=args.targets.split(","), task_type="CAUSAL_LM"))
trainable = sum(q.numel() for q in model.parameters() if q.requires_grad)
emit(stage="model", trainable_params=trainable, load_4bit=args.load_4bit)

def encode(paper):
    """input_ids for prompt+gold answer; labels = -100 on the prompt so only the answer is learned."""
    prompt = tok.apply_chat_template(messages(paper), tokenize=False, add_generation_prompt=True)
    full = prompt + target_text(paper) + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    assert f_ids[:len(p_ids)] == p_ids, "prompt tokenization is not a prefix of the full sequence"
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return torch.tensor([f_ids], device="cuda"), torch.tensor([labels], device="cuda")

@torch.no_grad()
def predict(papers):
    model.eval(); model.config.use_cache = True
    rows, raws = [], []
    for q in papers:
        prompt = tok.apply_chat_template(messages(q), tokenize=False, add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        raw = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        ids = parse_claim_ids(raw, {s["sentence_id"] for s in q["sentences"]})
        rows += claim_rows(q, ids); raws.append({"paper_id": q["paper_id"], "raw": raw, "parsed": ids})
    model.train(); model.config.use_cache = False
    return rows, raws

def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=0.0)
steps_per_epoch = max(1, len(train) // args.grad_accum)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / (steps_per_epoch * args.epochs)))
rng = random.Random(args.seed)
best, history = None, []
model.train(); torch.cuda.reset_peak_memory_stats()
for epoch in range(1, args.epochs + 1):
    order = list(train); rng.shuffle(order); losses = []
    for i, q in enumerate(order, 1):
        ids, labels = encode(q)
        loss = model(input_ids=ids, labels=labels).loss / args.grad_accum
        loss.backward(); losses.append(loss.item() * args.grad_accum)
        if i % args.grad_accum == 0 or i == len(order):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    rows, raws = predict(dev)
    scores = claim_scores({(r["paper_id"], r["sentence_id"]) for r in rows}, gold_dev)
    rec = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "dev": {k: scores[k] for k in ("precision", "recall", "f1")},
           "dev_predicted": scores["counts"]["predicted_claims"], "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    history.append(rec); emit(stage="epoch", **rec)
    model.save_pretrained(out / "last")
    if best is None or scores[args.select_by] > best[1]:
        best = (epoch, scores[args.select_by])
        model.save_pretrained(out / "best")
        write_jsonl(out / "dev_predictions_best.jsonl", rows); write_jsonl(out / "dev_raw_best.jsonl", raws)
        (out / "dev_scores_best.json").write_text(json.dumps({"epoch": epoch} | scores, indent=2) + "\n")

summary = {"best_epoch": best[0], f"best_dev_{args.select_by}": best[1], "history": history,
           "seconds": round(time.time() - T0), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
emit(stage="done", best_epoch=best[0], best=best[1], seconds=summary["seconds"])
