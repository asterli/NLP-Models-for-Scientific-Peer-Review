"""M10: prompted claim-extraction baseline with the untuned base model.
Numbers each paper's sentences, asks the model which are claims, writes predictions
in the shared Claim schema (claim_id, paper_id, sentence_id, text) plus raw outputs."""
import argparse, json, re, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--papers", default="data/papers/papers.jsonl")
p.add_argument("--split", default="all", help="train|dev|test|all")
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--out", default="data/claims/predictions/baseline_qwen7b.jsonl")
p.add_argument("--raw-out", default="/project2/xiangren_1987/group_40/runs/m10_raw_outputs.jsonl")
p.add_argument("--max-new-tokens", type=int, default=200)
args = p.parse_args()

SYSTEM = (
    "You label sentences from the abstract and introduction of a research paper.\n"
    "A CLAIM is a sentence stating the paper's own objective, contribution, finding, or conclusion "
    "(including implications of its findings).\n"
    "Everything else is BACKGROUND: prior work, motivation, gaps, general context, and descriptions of "
    "this study's own design, setting, recruitment, or sample, unless the paper presents them as its contribution.\n"
    "Respond with only a JSON object of the form {\"claims\": [\"s004\", \"s012\"]} listing the sentence IDs "
    "that are CLAIMs, in order. Use an empty list if there are none."
)

def build_prompt(paper):
    lines = [f"{s['sentence_id']} [{s['section']}]: {s['text']}" for s in paper["sentences"]]
    return "Paper title: " + paper["title"] + "\n\nSentences:\n" + "\n".join(lines)

def parse_ids(raw, valid):
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        ids = json.loads(m.group(0))["claims"] if m else []
    except Exception:
        ids = []
    if not ids:
        ids = re.findall(r"s\d{3}", raw)
    seen, out = set(), []
    for i in ids:
        if i in valid and i not in seen:
            seen.add(i); out.append(i)
    return out

papers = [json.loads(l) for l in open(args.papers)]
if args.split != "all":
    papers = [q for q in papers if q.get("split") == args.split]
print(f"{len(papers)} papers")

tok = AutoTokenizer.from_pretrained(args.model)
dtype = torch.float16 if "V100" in torch.cuda.get_device_name(0) else torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="cuda").eval()

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.raw_out).parent.mkdir(parents=True, exist_ok=True)
n_claims, t0 = 0, time.time()
with open(args.out, "w") as fo, open(args.raw_out, "w") as fr:
    for paper in papers:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(paper)}]
        ids = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tok(ids, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False)
        raw = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        by_id = {s["sentence_id"]: s for s in paper["sentences"]}
        pred = parse_ids(raw, by_id)
        fr.write(json.dumps({"paper_id": paper["paper_id"], "raw": raw, "parsed": pred}) + "\n")
        for sid in pred:
            fo.write(json.dumps({"claim_id": f"{paper['paper_id']}#{sid}", "paper_id": paper["paper_id"],
                                 "sentence_id": sid, "text": by_id[sid]["text"]}) + "\n")
        n_claims += len(pred)
        print(f"{paper['paper_id']}: {len(pred)}/{len(paper['sentences'])} claims")
print(f"done: {n_claims} claims from {len(papers)} papers in {time.time()-t0:.0f}s -> {args.out}")
