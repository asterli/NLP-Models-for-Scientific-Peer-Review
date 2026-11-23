"""Adapters from the pipeline's step signatures (pipeline.core) to the real components.

  F06 claims      LiveClaims (CUDA, loads the LoRA adapter) or SavedClaims (a file the extractor wrote)
  F11 retrieval   Retrieval  (PriorWorkRetriever: F10-filtered BM25 over data/corpus/corpus.jsonl)
  F09 comparison  MLXComparison (Apple Silicon, the frozen local_20261114 adapter)

The extractor needs CUDA and the comparison classifier needs Apple MLX, so no one machine runs both.
`LiveClaims` + `save_claims` produce a claims file on the cluster; `SavedClaims` reads it on the Mac.
Each adapter exposes `provenance()` so the run records which model and data produced its results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .provenance import file_sha256, files_sha256, rel

ROOT = Path(__file__).resolve().parents[2]
CLAIMS_FILE_KIND = "f12_claims"


# ---------------------------------------------------------------- F06 claim extraction
class LiveClaims:
    """Runs the F06 ClaimExtractor (needs a CUDA GPU and the M11 adapter directory)."""

    def __init__(self, adapter_path, load_4bit=False, extractor=None):
        self.adapter_path = Path(adapter_path)
        if extractor is None:
            from claims.extractor import ClaimExtractor
            extractor = ClaimExtractor(self.adapter_path, load_4bit=load_4bit)
        self.extractor, self.load_4bit = extractor, load_4bit
        self.unlabeled, self.raw = {}, {}

    def __call__(self, paper):
        claims = self.extractor.extract(paper)
        self.unlabeled[paper["paper_id"]] = list(self.extractor.last_unlabeled)
        self.raw[paper["paper_id"]] = self.extractor.last_raw
        return claims

    def provenance(self):
        from claims.extractor import DEFAULT_MODEL
        return {"component": "F06 claim extractor", "mode": "live", "base_model": DEFAULT_MODEL,
                "adapter": rel(self.adapter_path), "load_4bit": self.load_4bit,
                "adapter_sha256": files_sha256(self.adapter_path, ("adapter_config.json", "adapter_model.safetensors")),
                "source_sha256": {p: file_sha256(ROOT / p) for p in ("src/claims/extractor.py", "src/claims/prompting.py")}}


def save_claims(path, papers, live: LiveClaims, provenance: dict):
    """Write a claims file that keeps zero-claim papers, so `no claims` is distinguishable from `not run`."""
    out = {"kind": CLAIMS_FILE_KIND, "extractor": live.provenance(), "run": provenance, "papers": {}}
    for paper in papers:
        try:
            claims = live(paper)
            entry = {"status": "ok", "claims": claims}
        except Exception as e:  # noqa: BLE001 -- recorded, the run reports it as extraction_error
            entry = {"status": "error", "error": f"{type(e).__name__}: {e}", "claims": []}
        entry.update(unlabeled=live.unlabeled.get(paper["paper_id"], []), raw=live.raw.get(paper["paper_id"]))
        out["papers"][paper["paper_id"]] = entry
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


class SavedClaims:
    """Claims the F06 extractor already produced, read from either

    - a claims file written by `run_pipeline.py extract` (lists every paper, including zero-claim ones), or
    - a predictions JSONL from scripts/predict_claims.py (one row per claim; a paper with no rows is an
      error, because that format cannot tell a paper with no claims from one that was never run).
    The pipeline still checks every claim against the paper's own sentence IDs and text.
    """

    def __init__(self, path):
        self.path = Path(path)
        text = self.path.read_text(encoding="utf-8-sig")
        self.extractor_info = None
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            doc = None
        if isinstance(doc, dict) and doc.get("kind") == CLAIMS_FILE_KIND:
            self.format = "claims_file"
            self.extractor_info = doc.get("extractor")
            self.by_paper = doc["papers"]
        else:
            self.format = "predictions_jsonl"
            self.by_paper = {}
            for n, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or "paper_id" not in row:
                    raise ValueError(f"{self.path}:{n}: not a Claim row")
                self.by_paper.setdefault(row["paper_id"], {"status": "ok", "claims": []})["claims"].append(row)

    def __call__(self, paper):
        entry = self.by_paper.get(paper["paper_id"])
        if entry is None:
            hint = (" (a predictions JSONL only lists papers with claims; use a claims file from "
                    "`run_pipeline.py extract` to record zero-claim papers)") if self.format == "predictions_jsonl" else ""
            raise LookupError(f"{paper['paper_id']} is not in {self.path}{hint}")
        if entry.get("status") == "error":
            raise RuntimeError(f"extraction failed when the claims file was made: {entry.get('error')}")
        return entry["claims"]

    def provenance(self):
        return {"component": "F06 claim extractor", "mode": "saved", "format": self.format,
                "claims_file": rel(self.path), "claims_file_sha256": file_sha256(self.path),
                "extractor": self.extractor_info}


# ---------------------------------------------------------------- F11 retrieval
class Retrieval:
    """The F11 callable retriever, unchanged. `uncertain="exclude"` is the F10/F11-evaluated default."""

    def __init__(self, corpus_path, uncertain="exclude"):
        sys.path.insert(0, str(ROOT / "retrieval"))
        from f11_retriever import PriorWorkRetriever
        self.corpus_path, self.uncertain = Path(corpus_path), uncertain
        self.retriever = PriorWorkRetriever(str(self.corpus_path), uncertain=uncertain)

    def __call__(self, claim_text, paper, k):
        submitted = {key: paper.get(key) for key in ("paper_id", "doi", "pmid", "pmcid", "title", "date")}
        return self.retriever.retrieve(claim_text, submitted, k=k, explain=True)

    def provenance(self):
        bm25 = self.retriever.filtered.R
        return {"component": "F11 retrieval (F10-filtered BM25)", "corpus": rel(self.corpus_path),
                "corpus_sha256": file_sha256(self.corpus_path), "corpus_works": len(bm25.docs),
                "bm25": {"k1": bm25.k1, "b": bm25.b}, "uncertain_dates": self.uncertain,
                "source_sha256": {p: file_sha256(ROOT / p) for p in
                                  ("retrieval/f11_retriever.py", "retrieval/m23_filters.py",
                                   "retrieval/m22_bm25_retrieval.py")}}


# ---------------------------------------------------------------- F09 comparison
class MLXComparison:
    """The frozen F09 classifier. Compares the claim with the retrieved evidence passage.

    The evidence is F10's best-matching abstract sentence, sent as evidence_kind="passage": the classifier
    was trained on short older passages (median 29 words), not whole abstracts.
    """

    def __init__(self, adapter_path, local_model_dir, classifier=None):
        self.adapter_path, self.local_model_dir = Path(adapter_path), Path(local_model_dir)
        if classifier is None:
            for p in (str(ROOT), str(ROOT / "src")):
                if p not in sys.path:
                    sys.path.insert(0, p)
            from mlx_comparison import MLXComparisonClassifier  # verifies base-model hashes on load
            classifier = MLXComparisonClassifier(self.adapter_path, self.local_model_dir)
        self.classifier = classifier

    def __call__(self, claim_text, evidence):
        return self.classifier.predict(claim_text, evidence, evidence_kind="passage")["label"]

    def provenance(self):
        cfg = json.loads((self.adapter_path / "classifier_config.json").read_text())
        mp = cfg.get("model_provenance", {})
        return {"component": "F09 comparison classifier", "base_model": cfg.get("model"),
                "base_model_repo": mp.get("repo"), "base_model_revision": mp.get("revision"),
                "base_model_dir": rel(self.local_model_dir),
                "base_model_files_verified": "file hashes checked against classifier_config.json at load",
                "adapter": rel(self.adapter_path),
                "adapter_sha256": files_sha256(self.adapter_path,
                                               ("adapters.safetensors", "adapter_config.json", "classifier_config.json")),
                "labels": cfg.get("labels"), "prompt_name": cfg.get("prompt_name"), "seed": cfg.get("seed"),
                "best_epoch": cfg.get("best_epoch"), "evidence_kind": "passage",
                "source_sha256": {p: file_sha256(ROOT / p) for p in
                                  ("src/mlx_comparison.py", "src/comparison_baselines.py")}}
