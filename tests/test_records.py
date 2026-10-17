"""Tests for src/papers/records.py. Run with:  pytest tests/test_records.py"""

from pathlib import Path

from papers.europepmc import PaperMeta
from papers.jats import parse_article
from papers.records import build_record, check_record

FIXTURES = Path(__file__).parent / "fixtures"
META = PaperMeta(date="2024-03-05", authors=["Doe J"], title=None, doi=None, journal=None)


def record_for(name, meta=META, pmcid="PMC0000001"):
    article = parse_article((FIXTURES / f"{name}.xml").read_bytes())
    return build_record(pmcid, article, meta)


def test_record_has_schema_fields_in_order():
    keys = list(record_for("research_superscript"))
    assert keys[:6] == ["paper_id", "title", "date", "url", "text", "sentences"]


def test_sentences_are_numbered_across_the_whole_paper():
    sentences = record_for("research_superscript")["sentences"]
    assert [s["sentence_id"] for s in sentences] == [f"s{n:03d}" for n in range(1, len(sentences) + 1)]
    assert sentences[0]["section"] == "abstract:background"
    assert sentences[3] == {"sentence_id": "s004", "section": "introduction",
                            "text": "About 1.4 million people used shelters in 2017.[1]", "cites": ["R1"]}


def test_paper_id_and_url_use_the_doi():
    record = record_for("research_superscript")
    assert record["paper_id"] == "doi:10.1234/jhh.2024.001"
    assert record["url"] == "https://doi.org/10.1234/jhh.2024.001"


def test_paper_without_doi_falls_back_to_pmcid():
    record = record_for("commentary_no_sections", pmcid="PMC42")
    assert record["paper_id"] == "pmcid:PMC42"
    assert record["url"] == "https://europepmc.org/article/PMC/PMC42"


def test_text_is_built_from_sentences():
    record = record_for("research_brackets")
    assert record["text"] == " ".join(s["text"] for s in record["sentences"])


def test_metadata_comes_through():
    record = record_for("research_superscript")
    assert (record["date"], record["authors"], record["split"]) == ("2024-03-05", ["Doe J"], None)
    assert record["references"][0]["doi"] == "10.1000/shelter.1"


def test_warnings_for_a_research_paper_flag_only_short_sentences():
    # "See Figure 1." and "The effect was p small." are both under 25 characters
    assert check_record(record_for("research_superscript")) == [
        "very short sentences, maybe bad splits: s006, s007"]


def test_warnings_for_a_commentary():
    no_date = PaperMeta(date=None, authors=[], title=None, doi=None, journal=None)
    warnings = check_record(record_for("commentary_no_sections", meta=no_date))
    assert "no abstract found" in warnings
    assert any("article-commentary" in w for w in warnings)
    assert any("missing publication date" in w for w in warnings)
    assert "no reference list found" in warnings


def test_long_sentence_is_flagged():
    record = record_for("research_brackets")
    record["sentences"][0]["text"] = "word " * 81
    assert any("very long sentences" in w and "s001" in w for w in check_record(record))
