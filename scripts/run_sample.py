"""Quick no-GPU sample of the F12 pipeline on one held-out test paper.

Uses the saved F06 fine-tuned test predictions and the live F11 retriever; the comparison classifier is
skipped (pairs are recorded `not_run`). For the full pipeline, see docs/pipeline.md.

    python scripts/run_sample.py        # -> outputs/sample_<timestamp>/{results.json,predictions.jsonl,report.md}
"""
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_pipeline  # noqa: E402

out = ROOT / "outputs" / f"sample_{dt.datetime.now():%Y%m%d_%H%M%S}"
sys.exit(run_pipeline.main([
    "run", "--paper-id", "doi:10.1001/jamanetworkopen.2021.38464",
    "--claims-file", str(ROOT / "data/claims/predictions/finetuned_test_r16_lr2e-4_s544.jsonl"),
    "--no-comparison", "--out-dir", str(out),
]))
