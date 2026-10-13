import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sample_output = {
    "paper_id": "paper_001",
    "claims": [
        {
            "claim_id": "claim_001",
            "paper_id": "paper_001",
            "sentence_id": "s001",
            "text": "Example claim for testing."
        }
    ],
    "candidates": [
        {
            "paper_id": "paper_002",
            "title": "Example Prior Work",
            "score": 2.5,
            "date": "2024-01-01",
            "url": "https://example.org/paper_002",
            "evidence": "Example passage for testing."
        }
    ],
    "predictions": [
        {
            "claim_id": "claim_001",
            "prior_paper_id": "paper_002",
            "label": "insufficient_evidence",
            "score": None,
            "reason": "Placeholder prediction; no model has run."
        }
    ]
}

output_path = OUTPUT_DIR / "sample_output.json"

with output_path.open("w", encoding="utf-8") as f:
    json.dump(sample_output, f, indent=2)

print(f"Sample output saved to {output_path}")
