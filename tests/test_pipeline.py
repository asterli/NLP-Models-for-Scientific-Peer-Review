# F12 pipeline tests. No GPU, no model: real F11 retriever on a small fake corpus (and on the real corpus for
# the CLI test), stand-ins for the F06 extractor and F09 classifier. Run with:  pytest tests/test_pipeline.py
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from claims.extractor import ClaimExtractor  # noqa: E402
from pipeline import core, inputs, report  # noqa: E402
from pipeline.components import LiveClaims, MLXComparison, Retrieval, SavedClaims, save_claims  # noqa: E402
import run_pipeline  # noqa: E402

PAPER = {"paper_id": "doi:10.1/self", "title": "Housing First trial for veterans", "date": "2020-06-15",
         "doi": "10.1/self", "url": "https://doi.org/10.1/self",
         "sentences": [{"sentence_id": "s001", "section": "abstract", "text": "Homelessness is common."},
                       {"sentence_id": "s002", "section": "abstract:results",
                        "text": "Housing First improved housing stability among veterans."},
                       {"sentence_id": "s003", "section": "introduction",
                        "text": "Mindfulness apps reduced zebrafish anxiety."}]}
ABS = "Housing First improved housing stability for homeless adults. Participants were followed for two years."
CORPUS = [
    {"paper_id": "doi:10.1/a", "title": "Housing First housing stability trial", "date": "2015-06-01", "year": 2015,
     "abstract": ABS, "url": "https://doi.org/10.1/a", "doi": "10.1/a"},
    {"paper_id": "doi:10.1/titleonly", "title": "Housing First and housing stability", "date": "2014-02-01",
     "year": 2014, "abstract": None, "url": "https://doi.org/10.1/titleonly", "doi": "10.1/titleonly"},
    {"paper_id": "doi:10.1/self", "title": "Housing First trial for veterans", "date": "2020-06-15", "year": 2020,
     "abstract": ABS, "url": "https://doi.org/10.1/self", "doi": "10.1/self"},
    {"paper_id": "doi:10.1/new", "title": "Zebrafish anxiety and mindfulness apps", "date": "2022-03-01",
     "year": 2022, "abstract": "Mindfulness apps reduced zebrafish anxiety in a lab.", "url": "https://doi.org/10.1/new",
     "doi": "10.1/new"},
]


def claim(sid, paper=PAPER):
    text = next(s["text"] for s in paper["sentences"] if s["sentence_id"] == sid)
    return {"claim_id": f"{paper['paper_id']}#{sid}", "paper_id": paper["paper_id"], "sentence_id": sid, "text": text}


class FakeClassifier:
    def __init__(self, label="overlapping", fail_on=None, exc=ValueError):
        self.calls, self.label, self.fail_on, self.exc = [], label, fail_on, exc

    def __call__(self, claim_text, evidence):
        self.calls.append((claim_text, evidence))
        if self.fail_on and self.fail_on in evidence:
            raise self.exc("Input exceeds token limit")
        return self.label


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps(d) + "\n" for d in CORPUS), encoding="utf-8")
    return path


@pytest.fixture
def retrieve(corpus):
    return Retrieval(corpus)


# ---------------------------------------------------------------- core flow
def test_ids_links_and_evidence_flow_through_to_the_comparison(retrieve):
    clf = FakeClassifier()
    rec = core.run_paper(PAPER, lambda p: [claim("s002")], retrieve, clf, k=10)
    assert rec["status"] == "ok" and rec["paper"]["paper_id"] == "doi:10.1/self"
    (c,) = rec["claims"]
    assert (c["claim_id"], c["sentence_id"], c["section"], c["status"]) == \
        ("doi:10.1/self#s002", "s002", "abstract:results", "retrieved")
    by_id = {x["paper_id"]: x for x in c["candidates"]}
    assert set(by_id) == {"doi:10.1/a", "doi:10.1/titleonly"}          # self and newer work filtered (F10)
    a = by_id["doi:10.1/a"]
    assert a["url"] == "https://doi.org/10.1/a" and a["date"] == "2015-06-01" and a["evidence"] in ABS
    assert a["comparison"] == {"claim_id": "doi:10.1/self#s002", "prior_paper_id": "doi:10.1/a",
                               "label": "overlapping", "score": None, "reason": None,
                               "status": "compared", "detail": None}
    assert clf.calls == [(claim("s002")["text"], a["evidence"])]       # the classifier saw exactly this evidence


def test_title_only_candidates_are_not_compared(retrieve):
    clf = FakeClassifier()
    rec = core.run_paper(PAPER, lambda p: [claim("s002")], retrieve, clf)
    t = next(x for x in rec["claims"][0]["candidates"] if x["paper_id"] == "doi:10.1/titleonly")
    assert t["evidence"] is None
    assert t["comparison"]["status"] == "no_evidence" and t["comparison"]["label"] is None
    assert all(evidence for _, evidence in clf.calls)


def test_no_claims_skips_retrieval_and_comparison():
    def boom(*a):
        raise AssertionError("must not be called")
    rec = core.run_paper(PAPER, lambda p: [], boom, boom)
    assert rec["status"] == "no_claims" and rec["claims"] == []


def test_no_eligible_candidates_is_recorded_without_a_comparison(retrieve):
    clf = FakeClassifier()
    rec = core.run_paper(PAPER, lambda p: [claim("s003")], retrieve, clf)  # only match is newer than the paper
    (c,) = rec["claims"]
    assert c["status"] == "no_eligible_candidates" and c["candidates"] == []
    assert c["excluded_counts"].get("after_cutoff") == 1 and clf.calls == []


def test_extraction_failure_is_recorded_per_paper():
    def fail(paper):
        raise RuntimeError("CUDA out of memory")
    rec = core.run_paper(PAPER, fail, None)
    assert rec["status"] == "extraction_error" and "CUDA out of memory" in rec["detail"]


@pytest.mark.parametrize("bad, message", [
    ({**claim("s002"), "text": "Housing First improved stability."}, "text differs"),
    ({**claim("s002"), "sentence_id": "s099"}, "no sentence s099"),
    ({**claim("s002"), "paper_id": "doi:10.1/other"}, "belongs to"),
    ({k: v for k, v in claim("s002").items() if k != "claim_id"}, "missing `claim_id`"),
])
def test_claims_must_be_original_sentences_of_the_paper(bad, message):
    rec = core.run_paper(PAPER, lambda p: [bad], None)
    assert rec["status"] == "extraction_error" and message in rec["detail"]


def test_retrieval_error_is_recorded_per_claim():
    def fail(*a):
        raise ValueError("submitted paper needs a 'date'")
    rec = core.run_paper(PAPER, lambda p: [claim("s002")], fail, FakeClassifier())
    assert rec["claims"][0]["status"] == "retrieval_error"


def test_classifier_errors_are_kept_separate_and_the_run_continues(retrieve):
    clf = FakeClassifier(fail_on="Housing First")
    rec = core.run_paper(PAPER, lambda p: [claim("s002")], retrieve, clf)
    cmp_ = rec["claims"][0]["candidates"][0]["comparison"]
    assert cmp_["status"] == "error" and cmp_["label"] is None and "token limit" in cmp_["detail"]


def test_unknown_label_is_an_error_not_a_prediction(retrieve):
    rec = core.run_paper(PAPER, lambda p: [claim("s002")], retrieve, FakeClassifier(label="novel"))
    cmp_ = rec["claims"][0]["candidates"][0]["comparison"]
    assert cmp_["status"] == "error" and cmp_["label"] is None


def test_setup_errors_from_the_classifier_stop_the_run(retrieve):
    with pytest.raises(OSError):
        core.run_paper(PAPER, lambda p: [claim("s002")], retrieve, FakeClassifier(fail_on="Housing", exc=OSError))


def test_without_a_classifier_pairs_are_not_run(retrieve):
    rec = core.run_paper(PAPER, lambda p: [claim("s002")], retrieve, None)
    statuses = {c["paper_id"]: c["comparison"]["status"] for c in rec["claims"][0]["candidates"]}
    assert statuses == {"doi:10.1/a": "not_run", "doi:10.1/titleonly": "no_evidence"}


def test_compare_top_limits_classifier_calls(retrieve):
    clf = FakeClassifier()
    rec = core.run_paper(PAPER, lambda p: [claim("s002")], retrieve, clf, compare_top=1)
    first, second = rec["claims"][0]["candidates"]
    assert first["comparison"]["status"] == "compared" and second["comparison"]["status"] in ("not_run", "no_evidence")
    assert len(clf.calls) <= 1


def test_predictions_and_summary(retrieve):
    recs = [core.run_paper(PAPER, lambda p: [claim("s002"), claim("s003")], retrieve, FakeClassifier()),
            core.run_paper(PAPER, lambda p: [], retrieve, FakeClassifier())]
    rows = core.predictions(recs)
    assert [r["prior_paper_id"] for r in rows] == ["doi:10.1/a", "doi:10.1/titleonly"]
    assert rows[0]["paper_id"] == "doi:10.1/self" and rows[0]["sentence_id"] == "s002" and rows[0]["evidence"] in ABS
    s = core.summarize(recs)
    assert s["paper_status"] == {"no_claims": 1, "ok": 1}
    assert s["claim_status"] == {"no_eligible_candidates": 1, "retrieved": 1}
    assert s["pair_status"] == {"compared": 1, "no_evidence": 1} and s["labels"] == {"overlapping": 1}


# ---------------------------------------------------------------- component adapters
def test_saved_predictions_jsonl(tmp_path):
    path = tmp_path / "pred.jsonl"
    path.write_text(json.dumps(claim("s002")) + "\n")
    saved = SavedClaims(path)
    assert saved(PAPER) == [claim("s002")] and saved.format == "predictions_jsonl"
    rec = core.run_paper({**PAPER, "paper_id": "doi:10.1/missing"}, saved, None)
    assert rec["status"] == "extraction_error" and "only lists papers with claims" in rec["detail"]
    assert saved.provenance()["claims_file_sha256"]


def test_claims_file_round_trip_keeps_zero_claim_papers(tmp_path):
    # the real ClaimExtractor with a stand-in generator, so the claims file is what extract would write
    outputs = {"doi:10.1/self": "s001: BACKGROUND\ns002: CLAIM\ns003: BACKGROUND",
               "doi:10.1/empty": "s001: BACKGROUND"}
    gen_paper = {}
    def generate(msgs, n):
        return outputs[gen_paper["id"]]
    extractor = ClaimExtractor("unused", generate=generate)
    class Tracking(LiveClaims):
        def __call__(self, paper):
            gen_paper["id"] = paper["paper_id"]
            return super().__call__(paper)
    empty = {**PAPER, "paper_id": "doi:10.1/empty", "sentences": PAPER["sentences"][:1]}
    broken = {**PAPER, "paper_id": "doi:10.1/broken", "sentences": [{"sentence_id": "s001"}]}
    path = tmp_path / "claims.json"
    save_claims(path, [PAPER, empty, broken], Tracking("adapter-dir", extractor=extractor), {"command": "test"})

    saved = SavedClaims(path)
    assert saved.format == "claims_file" and saved.provenance()["extractor"]["adapter"] == "adapter-dir"
    assert core.run_paper(PAPER, saved, None)["claims"][0]["claim_id"] == "doi:10.1/self#s002"
    assert core.run_paper(empty, saved, None)["status"] == "no_claims"
    assert core.run_paper(broken, saved, None)["status"] == "extraction_error"


def test_mlx_adapter_sends_the_evidence_as_a_passage():
    class Fake:
        def predict(self, claim_text, prior_work, evidence_kind="passage"):
            self.args = (claim_text, prior_work, evidence_kind)
            return {"label": "distinct"}
    fake = Fake()
    cmp_ = MLXComparison(ROOT / "artifacts/comparison/local_20261114/adapter", "/models/qwen", classifier=fake)
    assert cmp_("claim", "older passage") == "distinct" and fake.args == ("claim", "older passage", "passage")
    prov = cmp_.provenance()
    assert prov["labels"] == list(core.LABELS) and prov["adapter_sha256"]["adapters.safetensors"]


def test_retrieval_provenance(retrieve, corpus):
    prov = retrieve.provenance()
    assert prov["corpus_works"] == len(CORPUS) and prov["corpus_sha256"] and prov["uncertain_dates"] == "exclude"


# ---------------------------------------------------------------- inputs and report
def test_paper_from_text(tmp_path):
    path = tmp_path / "paper.txt"
    path.write_text("Abstract\n\nWe tested the drug. It worked.\n\n# Introduction\n\nPrior work studied Y.\n")
    paper = inputs.paper_from_text(path, "A title", "2024-02-03", doi="https://doi.org/10.5/ABC")
    assert paper["paper_id"] == "doi:10.5/abc"
    assert [(s["sentence_id"], s["section"], s["text"]) for s in paper["sentences"]] == [
        ("s001", "abstract", "We tested the drug."), ("s002", "abstract", "It worked."),
        ("s003", "introduction", "Prior work studied Y.")]
    with pytest.raises(ValueError):
        inputs.paper_from_text(path, "A title", "2024")


def test_papers_need_a_date(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({k: v for k, v in PAPER.items() if k != "date"}))
    with pytest.raises(ValueError, match="date"):
        inputs.papers_from_file(path)


def test_report_shows_every_status(retrieve):
    recs = [core.run_paper(PAPER, lambda p: [claim("s002"), claim("s003")], retrieve, FakeClassifier()),
            core.run_paper({**PAPER, "paper_id": "doi:10.1/x"}, lambda p: [], retrieve, None)]
    result = {"run": {"started": "now", "code": {"commit": "abc", "dirty": False}},
              "settings": {"k": 10, "compare_top": None},
              "components": {"claims": {"mode": "saved", "claims_file": "c.jsonl"}, "retrieval": {}, "comparison": None},
              "summary": core.summarize(recs), "papers": recs}
    md = report.render(result)
    assert "[Housing First housing stability trial](https://doi.org/10.1/a)" in md
    assert "**Overlapping**" in md and "title only" in md
    assert "No eligible candidates" in md and "No claims" in md and "not run" in md


# ---------------------------------------------------------------- command line
SAVED_TEST_CLAIMS = ROOT / "data/claims/predictions/finetuned_test_r16_lr2e-4_s544.jsonl"


def test_cli_runs_real_claims_and_retrieval(tmp_path, monkeypatch):
    """Real papers, real saved F06 test predictions, real corpus and F11 retriever; stand-in classifier."""
    clf = FakeClassifier(label="distinct")
    monkeypatch.setattr(run_pipeline, "MLXComparison",
                        lambda adapter, model_dir: MLXComparison(adapter, model_dir, classifier=type(
                            "C", (), {"predict": lambda self, c, e, evidence_kind: {"label": clf(c, e)}})()))
    out = tmp_path / "run"
    code = run_pipeline.main(["run", "--split", "test", "--claims-file", str(SAVED_TEST_CLAIMS),
                              "--local-model-dir", "/models/qwen", "--compare-top", "2", "--out-dir", str(out)])
    assert code == 0
    result = json.loads((out / "results.json").read_text())
    assert result["summary"]["papers"] == 3 and result["summary"]["claims"] == 25
    assert result["run"]["code"]["commit"] and result["components"]["retrieval"]["corpus_works"] == 644
    assert result["components"]["claims"]["claims_file_sha256"] and result["data"]["papers_source_sha256"]
    rows = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == result["summary"]["pairs"]
    compared = [r for r in rows if r["status"] == "compared"]
    assert compared and all(r["label"] == "distinct" and r["rank"] <= 2 for r in compared)
    assert len(clf.calls) == len(compared)
    md = (out / "report.md").read_text()
    assert "](https://doi.org/" in md and "Distinct" in md


def test_cli_requires_a_comparison_choice_and_a_new_out_dir(tmp_path):
    args = ["run", "--split", "test", "--claims-file", str(SAVED_TEST_CLAIMS)]
    with pytest.raises(SystemExit):
        run_pipeline.main(args + ["--out-dir", str(tmp_path / "a")])           # neither model dir nor --no-comparison
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "results.json").write_text("{}")
    with pytest.raises(SystemExit):
        run_pipeline.main(args + ["--no-comparison", "--out-dir", str(tmp_path / "b")])
    assert (tmp_path / "b" / "results.json").read_text() == "{}"



def test_code_version_lists_dirty_files_by_full_path(monkeypatch):
    from pipeline import provenance
    outputs = {("status", "--porcelain"): " M README.md\n?? src/pipeline/\n", ("rev-parse", "HEAD"): "abc\n",
               ("rev-parse", "--abbrev-ref", "HEAD"): "main\n"}
    monkeypatch.setattr(provenance, "_git",
                        lambda *a, strip=True: outputs[a].strip() if strip else outputs[a])
    assert provenance.code_version() == {"commit": "abc", "branch": "main", "dirty": True,
                                         "dirty_files": ["README.md", "src/pipeline/"]}
    outputs[("status", "--porcelain")] = ""
    assert provenance.code_version()["dirty"] is False
