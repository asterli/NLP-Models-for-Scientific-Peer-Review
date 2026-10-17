# tests for src/papers/jats.py

# The XML files in tests/fixtures/ are small made-up papers 
# each built to cover specific tests seen in real Europe PMC XML (noted at the top of each file).

from pathlib import Path

import pytest

from papers.jats import _clean_title, normalize_abstract_heading, normalize_doi, parse_article

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    return parse_article((FIXTURES / f"{name}.xml").read_bytes())


def cited(paragraph):
    """[(ref_id, the text that cites it), ...] for a paragraph."""
    return [(c.ref_id, paragraph.text[c.start:c.end]) for c in paragraph.citations]


# --- metadata --------------------------------------------------------------

def test_metadata():
    a = load("research_superscript")
    assert a.title == "Housing First and emergency care use: a cohort study"  # <italic> flattened
    assert a.doi == "10.1234/jhh.2024.001"  # lowercased
    assert a.venue == "Journal of Housing and Health"
    assert a.article_type == "research-article"


def test_commentary_is_identifiable():
    assert load("commentary_no_sections").article_type == "article-commentary"


# --- abstract --------------------------------------------------------------

def test_structured_abstract_uses_normalized_headings():
    sections = [p.section for p in load("research_superscript").paragraphs if p.section.startswith("abstract")]
    assert sections == ["abstract:background", "abstract:results", "abstract:conclusions"]


def test_graphical_abstract_and_summary_are_skipped():
    text = " ".join(p.text for p in load("research_superscript").paragraphs)
    assert "Graphical abstract" not in text and "Plain-language summary" not in text


def test_unstructured_abstract():
    first = load("research_brackets").paragraphs[0]
    assert (first.section, first.text) == ("abstract", "We studied outreach. It worked.")


@pytest.mark.parametrize("heading, expected", [
    ("Findings:", "results"),
    ("Methods and analysis", "methods"),
    ("Design, setting, and participants", "methods"),
    ("Main Outcome Measures", "methods"),
    ("Aims and objectives", "objective"),
    ("Conclusions and relevance", "conclusions"),
    ("Limitations", "limitations"),  # unknown headings keep their own name
])
def test_heading_normalization(heading, expected):
    assert normalize_abstract_heading(heading) == expected


# --- introduction ----------------------------------------------------------

def test_intro_found_by_title_and_includes_subsections():
    intro = [p for p in load("research_superscript").paragraphs if p.section == "introduction"]
    assert len(intro) == 2  # one paragraph in "1. Background", one in its subsection "1.1"


def test_intro_found_by_sec_type():
    assert any(p.section == "introduction" for p in load("research_brackets").paragraphs)


def test_methods_figures_and_boxes_are_excluded():
    text = " ".join(p.text for p in load("research_superscript").paragraphs)
    for unwanted in ("Not part of the excerpt", "Logic model caption", "Box text"):
        assert unwanted not in text


def test_body_without_sections_uses_leading_paragraphs():
    intro = load("commentary_no_sections").paragraphs
    assert [p.text for p in intro] == ["First paragraph of opinion.", "Second paragraph."]


def test_whitespace_is_collapsed():
    intro = [p for p in load("research_superscript").paragraphs if p.section == "introduction"]
    assert "  " not in intro[0].text and "\n" not in intro[0].text


# --- citations -------------------------------------------------------------

def test_superscript_citations_get_brackets():
    p = next(p for p in load("research_superscript").paragraphs if p.section == "introduction")
    assert p.text == "About 1.4 million people used shelters in 2017.[1] Prior work found improved stability.[2],[3]"
    assert cited(p) == [("R1", "[1]"), ("R2", "[2]"), ("R3", "[3]")]


def test_bracketed_and_author_year_citations_are_left_alone():
    p = next(p for p in load("research_brackets").paragraphs if p.section == "introduction")
    assert p.text.startswith("Outreach improves access [1, 2]. Housing matters (Smith et al., 2019).")
    assert cited(p)[:3] == [("B1", "1"), ("B2", "2"), ("B3", "Smith et al., 2019")]


def test_one_citation_can_point_to_several_references():
    p = next(p for p in load("research_brackets").paragraphs if p.section == "introduction")
    assert cited(p)[3:] == [("B1", "[1, 2]"), ("B2", "[1, 2]")]


def test_affiliation_links_are_not_citations():
    a = load("research_superscript")
    assert all(c.ref_id.startswith("R") for p in a.paragraphs for c in p.citations)


# --- references ------------------------------------------------------------

def test_references():
    refs = {r.ref_id: r for r in load("research_superscript").references}
    assert (refs["R1"].doi, refs["R1"].pmid, refs["R1"].year) == ("10.1000/shelter.1", "111", 2018)
    assert refs["R2"].doi == "10.2105/ajph.94.4.651"  # DOI found in plain text, trailing "." dropped
    assert (refs["R3"].doi, refs["R3"].title, refs["R3"].year) == (None, "A Book With No DOI", 2010)


# --- helpers ---------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("https://doi.org/10.1/ABC", "10.1/abc"),
    ("doi:10.1/x", "10.1/x"),
    ("", None),
    (None, None),
])
def test_normalize_doi(raw, expected):
    assert normalize_doi(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("1. Introduction", "introduction"),
    ("2.1 Background", "background"),
    ("IV. Methods", "methods"),
    ("Methods/Design", "methods design"),
    ("X ray findings", "x ray findings"),  # a word, not a numeral: kept
])
def test_clean_title(raw, expected):
    assert _clean_title(raw) == expected


# --- quirks found in the first real run (Oct 16) -----------------------------

def test_jama_key_points_box_is_skipped_for_the_real_abstract():
    sections = [p.section for p in load("jama_key_points").paragraphs]
    assert "abstract:question" not in sections and "abstract:meaning" not in sections
    assert sections[:5] == ["abstract:background", "abstract:methods", "abstract:methods",
                            "abstract:results", "abstract:conclusions"]


def test_registration_and_keywords_are_dropped():
    text = " ".join(p.text for p in load("jama_key_points").paragraphs)
    assert "NCT00000001" not in text and "Keywords" not in text


@pytest.mark.parametrize("heading, expected", [
    ("Principal findings", "results"),         # contains "finding", doesn't start with it
    ("Study setting and design", "methods"),
    ("Importance", "background"),
    ("Data collection", "methods"),
])
def test_heading_normalization_by_contained_word(heading, expected):
    assert normalize_abstract_heading(heading) == expected


def test_bare_superscript_numbers_become_bracketed_citations():
    p = load("bare_superscripts").paragraphs[0]
    assert p.text.startswith("Homelessness is linked to emergency department use[1], [2] and inpatient care. [3]")
    assert cited(p)[:3] == [("b1", "[1]"), ("b2", "[2]"), ("b3", "[3]")]


def test_exponent_after_a_digit_is_not_a_citation():
    p = load("bare_superscripts").paragraphs[0]
    assert "Costs reach 106 dollars." in p.text  # the superscript 6 stays plain text
    assert all(ref_id != "b6" for ref_id, _ in cited(p))


def test_link_without_ref_type_still_counts_when_it_points_to_a_reference():
    p = load("bare_superscripts").paragraphs[0]
    assert cited(p)[-1] == ("b4", "[4]")