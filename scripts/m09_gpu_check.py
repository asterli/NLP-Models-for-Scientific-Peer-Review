"""M09 smoke test: load base model on GPU, train a few LoRA steps on dummy data,
save the adapter, reload it into a fresh model, generate once. Records memory and time."""
import argparse, json, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--out", default="/project2/xiangren_1987/group_40/runs/m09_adapter")
p.add_argument("--steps", type=int, default=5)
p.add_argument("--load-4bit", action="store_true")
args = p.parse_args()

assert torch.cuda.is_available(), "no GPU visible"
dev = torch.cuda.get_device_name(0)
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=dtype) if args.load_4bit else None

tok = AutoTokenizer.from_pretrained(args.model)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype,
                                             quantization_config=quant, device_map="cuda")
load_s = time.time() - t0

model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                         target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
model.print_trainable_parameters()

# dummy examples in the M11 shape: numbered sentences -> claim sentence IDs
examples = [
    ("1. Prior work shows X.\n2. We find that Y improves Z.\n3. Our method was tested on 20 papers.", "2"),
    ("1. Homelessness is a risk factor.\n2. This study aimed to assess A.\n3. Data were collected in 2019.", "2"),
    ("1. We propose B.\n2. B outperforms C.\n3. C was introduced in 2015.", "1, 2"),
]
def chat(text, ans=None):
    msgs = [{"role": "user", "content": "Which numbered sentences are claims?\n" + text}]
    if ans is None:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    msgs.append({"role": "assistant", "content": ans})
    return tok.apply_chat_template(msgs, tokenize=False)

opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=2e-4)
model.train()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
for step in range(args.steps):
    text, ans = examples[step % len(examples)]
    batch = tok(chat(text, ans), return_tensors="pt").to("cuda")
    loss = model(**batch, labels=batch["input_ids"]).loss
    loss.backward(); opt.step(); opt.zero_grad()
    print(f"step {step} loss {loss.item():.4f}")
train_s = time.time() - t0
peak_gb = torch.cuda.max_memory_allocated() / 1e9

Path(args.out).mkdir(parents=True, exist_ok=True)
model.save_pretrained(args.out)
del model; torch.cuda.empty_cache()

# fresh reload: base + saved adapter
base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype,
                                            quantization_config=quant, device_map="cuda")
model = PeftModel.from_pretrained(base, args.out).eval()
ids = tok(chat(examples[0][0]), return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**ids, max_new_tokens=10, do_sample=False)
print("reloaded generation:", repr(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)))

report = {"gpu": dev, "dtype": str(dtype), "load_4bit": args.load_4bit,
          "model_load_s": round(load_s, 1), "train_steps": args.steps,
          "train_s": round(train_s, 1), "peak_mem_gb": round(peak_gb, 2), "adapter_dir": args.out}
print(json.dumps(report, indent=2))
Path(args.out, "m09_report.json").write_text(json.dumps(report, indent=2))
