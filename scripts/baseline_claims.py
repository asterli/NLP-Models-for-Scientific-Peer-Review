"""M10: prompted claim-extraction baseline with the untuned base model.
Numbers each paper's sentences, asks the model which are claims, writes predictions
in the shared Claim schema (claim_id, paper_id, sentence_id, text) plus raw outputs.

Prompt variants (select with --prompt; tune on dev only):
  v1  zero-shot, model returns a JSON list of CLAIM sentence IDs   (Oct 18 reference run)
  v2  model labels EVERY sentence CLAIM/BACKGROUND, one per line, following the M06 decision procedure
"""
import argparse, json, re, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--papers", default="data/papers/papers.jsonl")
p.add_argument("--split", default="all", help="train|dev|test|all (membership read from data/claims/splits/)")
p.add_argument("--prompt", default="v1", choices=["v1", "v2"])
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--out", default="data/claims/predictions/baseline_qwen7b.jsonl")
p.add_argument("--raw-out", default="/project2/xiangren_1987/group_40/runs/m10_raw_outputs.jsonl")
p.add_argument("--max-new-tokens", type=int, default=None, help="default: 200 for v1, 700 for v2")
args = p.parse_args()

DEFINITIONS = (
    "You label sentences from the abstract and introduction of a research paper.\n"
    "A CLAIM is a sentence stating the paper's own objective, contribution, finding, or conclusion "
    "(including implications of its findings).\n"
    "Everything else is BACKGROUND: prior work, motivation, gaps, general context, and descriptions of "
    "this study's own design, setting, recruitment, or sample, unless the paper presents them as its contribution.\n"
)

PROMPTS = {
    # v1: unchanged from the Oct 18 zero-shot run
    "v1": DEFINITIONS +
          "Respond with only a JSON object of the form {\"claims\": [\"s004\", \"s012\"]} listing the sentence IDs "
          "that are CLAIMs, in order. Use an empty list if there are none.",
    # v2: per-sentence decision, following the M06 guide's decision procedure
    "v2": DEFINITIONS +
          "\nDecide each sentence separately, in order:\n"
          "1. Does it state a specific objective of THIS paper, a contribution it makes, or a finding or conclusion "
          "from its own analysis? If yes -> CLAIM. A stated study aim counts even before results are reported. "
          "A result sentence from this paper's own analysis counts even if it only reports numbers. "
          "A conclusion or implication drawn from this paper's findings counts.\n"
          "2. Does it mainly describe prior research, general context, a research gap, or why more research is needed? "
          "If yes -> BACKGROUND, unless it also clearly states a current-paper objective, finding, or conclusion.\n"
          "3. Does it describe this study's design, setting, data collection, recruitment, eligibility, or sample? "
          "If it is a routine description -> BACKGROUND. Only CLAIM if presented as the paper's contribution.\n"
          "4. A sentence mixing background with a clear current-paper objective, finding, or conclusion is CLAIM.\n"
          "Judge by meaning in context, not by keywords or section headings.\n\n"
          "Output exactly one line per sentence, in the same order, of the form\n"
          "s001: BACKGROUND\n"
          "s002: CLAIM\n"
          "Label every sentence. Output nothing else.",
}
SYSTEM = PROMPTS[args.prompt]
max_new = args.max_new_tokens or (200 if args.prompt == "v1" else 700)

def build_prompt(paper):
    lines = [f"{s['sentence_id']} [{s['section']}]: {s['text']}" for s in paper["sentences"]]
    return "Paper title: " + paper["title"] + "\n\nSentences:\n" + "\n".join(lines)

def parse_ids(raw, valid):
    if args.prompt == "v2":
        ids = [m.group(1) for m in re.finditer(r"(s\d{3})\s*[:\-–]\s*CLAIM\b", raw, re.I)]
    else:
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
    ids = {json.loads(l)["paper_id"] for l in open(f"data/claims/splits/{args.split}.jsonl") if l.strip()}
    papers = [q for q in papers if q["paper_id"] in ids]
print(f"{len(papers)} papers, prompt={args.prompt}, max_new_tokens={max_new}")

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
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
        raw = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        by_id = {s["sentence_id"]: s for s in paper["sentences"]}
        pred = parse_ids(raw, by_id)
        fr.write(json.dumps({"paper_id": paper["paper_id"], "prompt": args.prompt, "raw": raw, "parsed": pred}) + "\n")
        for sid in pred:
            fo.write(json.dumps({"claim_id": f"{paper['paper_id']}#{sid}", "paper_id": paper["paper_id"],
                                 "sentence_id": sid, "text": by_id[sid]["text"]}) + "\n")
        n_claims += len(pred)
        print(f"{paper['paper_id']}: {len(pred)}/{len(paper['sentences'])} claims")
print(f"done: {n_claims} claims from {len(papers)} papers in {time.time()-t0:.0f}s -> {args.out}")
