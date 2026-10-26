import json
from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/claims/M08_dataset.xlsx")
OUTPUT_DIR = Path("data/claims/splits")

SHEET_NAME = "Sentence_Dataset"


def main():
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

    required_columns = {
        "paper_id",
        "sentence_id",
        "source_excerpt",
        "final_label",
        "split",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Basic validation
    if df[list(required_columns)].isnull().any().any():
        raise ValueError("Dataset contains missing required values.")

    if not df["final_label"].isin(["CLAIM", "BACKGROUND"]).all():
        raise ValueError("Found an invalid claim label.")

    if not df["split"].isin(["train", "dev", "test"]).all():
        raise ValueError("Found an invalid split.")

    # Important M08 leakage check:
    # each paper must occur in exactly one split.
    paper_split_counts = df.groupby("paper_id")["split"].nunique()

    leaking_papers = paper_split_counts[paper_split_counts > 1]

    if not leaking_papers.empty:
        raise ValueError(
            "Some papers appear in multiple splits:\n"
            + "\n".join(leaking_papers.index)
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "dev", "test"]:
        split_df = df[df["split"] == split]

        output_file = OUTPUT_DIR / f"{split}.jsonl"

        with output_file.open("w", encoding="utf-8") as f:
            for _, row in split_df.iterrows():

                record = {
                    "paper_id": str(row["paper_id"]),
                    "sentence_id": str(row["sentence_id"]),
                    "text": str(row["source_excerpt"]),
                    "label": str(row["final_label"]),
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"{split}: "
            f"{split_df['paper_id'].nunique()} papers, "
            f"{len(split_df)} sentences -> {output_file}"
        )


if __name__ == "__main__":
    main()