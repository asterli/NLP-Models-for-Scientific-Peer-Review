"""M12: GPU check that the callable extractor reproduces the committed fine-tuned DEV predictions.

  python scripts/check_extractor.py --adapter /project2/xiangren_1987/group_40/runs/m11/r16_lr2e-4_s544/best

Uses dev, not test. Exits 1 if the extractor's claims differ from
data/claims/predictions/finetuned_dev_r16_lr2e-4_s544.jsonl.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from claims.extractor import ClaimExtractor  # noqa: E402
from claims.prompting import load_split, read_jsonl  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--adapter", required=True)
p.add_argument("--expected", default="data/claims/predictions/finetuned_dev_r16_lr2e-4_s544.jsonl")
p.add_argument("--load-4bit", action="store_true")
args = p.parse_args()

extractor = ClaimExtractor(args.adapter, load_4bit=args.load_4bit)
got = []
for paper in load_split("dev"):
    claims = extractor.extract(paper)
    got += claims
    print(f"{paper['paper_id']}: {len(claims)}/{len(paper['sentences'])} claims, unlabeled {extractor.last_unlabeled}")

key = lambda r: (r["paper_id"], r["sentence_id"])
expected = sorted(read_jsonl(args.expected), key=key)
got = sorted(got, key=key)
if got == expected:
    print(f"OK: extractor reproduces {args.expected} ({len(got)} claims)")
else:
    e, g = {key(r) for r in expected}, {key(r) for r in got}
    print(f"MISMATCH: expected {len(expected)} claims, got {len(got)}")
    print("  only in expected:", sorted(e - g))
    print("  only from extractor:", sorted(g - e))
    sys.exit(1)
