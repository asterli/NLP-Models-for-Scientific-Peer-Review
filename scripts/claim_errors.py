"""M12: list every error in a claim prediction file, with the sentence text, for error analysis. No GPU.

  python scripts/claim_errors.py --pred data/claims/predictions/finetuned_test_r16_lr2e-4_s544.jsonl \
      --gold data/claims/splits/test.jsonl --out results/m12_test_errors_finetuned.txt

pred: one line per sentence the model called CLAIM ({"paper_id", "sentence_id", ...}).
gold: every labeled sentence ({"paper_id", "sentence_id", "text", "label"}).
"""
import argparse, json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--pred", required=True)
p.add_argument("--gold", required=True)
p.add_argument("--out", default=None, help="also write the listing to this file")
args = p.parse_args()

def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(l) for l in f if l.strip()]

gold = {(r["paper_id"], r["sentence_id"]): r for r in read(args.gold)}
pred = {(r["paper_id"], r["sentence_id"]) for r in read(args.pred)} & set(gold)
missed = [k for k, r in gold.items() if r["label"].strip().upper() == "CLAIM" and k not in pred]
false_pos = sorted(k for k in pred if gold[k]["label"].strip().upper() != "CLAIM")
n_gold = sum(r["label"].strip().upper() == "CLAIM" for r in gold.values())

lines = [f"pred: {args.pred}", f"gold: {args.gold}",
         f"{len(gold)} sentences, {n_gold} gold claims, {len(pred)} predicted, "
         f"{len(missed)} missed, {len(false_pos)} false positives", ""]
for title, keys in (("MISSED (gold CLAIM, predicted BACKGROUND)", missed),
                    ("FALSE POSITIVE (gold BACKGROUND, predicted CLAIM)", false_pos)):
    lines.append(f"--- {title}: {len(keys)} ---")
    for paper_id, sid in keys:
        lines.append(f"{paper_id}  {sid}  {gold[(paper_id, sid)]['text']}")
    lines.append("")
text = "\n".join(lines)
print(text)
if args.out:
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n", encoding="utf-8")
