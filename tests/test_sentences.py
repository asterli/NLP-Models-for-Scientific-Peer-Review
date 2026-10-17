# Tests for src/papers/sentences.py. Run with:  pytest tests/test_sentences.py

# One test per non-simple/tricky case. The last group documents the known misses, so a change in
# that behavior shows up here instead of quietly in the data.


import pytest

from papers.jats import Citation, Paragraph
from papers.sentences import split_sentences


def split(text):
    return [s.text for s in split_sentences(Paragraph("introduction", text))]


# --- Some cases ---------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    # 1. citation right after the period stays with the first sentence
    ("Shelters were full in 2017.[1] Prior work agrees.",
     ["Shelters were full in 2017.[1]", "Prior work agrees."]),
    # 2. several citation groups in a row
    ("Stability improved.[2],[3] Next came costs.",
     ["Stability improved.[2],[3]", "Next came costs."]),
    # 3. et al.
    ("Tsemberis et al. found gains. They were large.",
     ["Tsemberis et al. found gains.", "They were large."]),
    # 4. common abbreviations, including before a capital or digit
    ("See Fig. 2 for details. Costs fell vs. Treatment First. Done.",
     ["See Fig. 2 for details.", "Costs fell vs. Treatment First.", "Done."]),
    # 5. initials
    ("J. Smith reported gains. Others agreed.",
     ["J. Smith reported gains.", "Others agreed."]),
    # 6. sentence ending with a bracket
    ("Visits fell (95% CI 0.6 to 0.9). Effects were larger in women.",
     ["Visits fell (95% CI 0.6 to 0.9).", "Effects were larger in women."]),
    # 7. decimals
    ("About 1.4 million people used shelters. Most were adults.",
     ["About 1.4 million people used shelters.", "Most were adults."]),
    # 8. next word starts lowercase: no split, even for abbreviations not on our list
    ("Effects were larger in older adults, e.g. those aged 40+. This held.",
     ["Effects were larger in older adults, e.g. those aged 40+.", "This held."]),
    # 9. author-year citation containing et al.
    ("Housing matters (Smith et al., 2019). Next we review costs.",
     ["Housing matters (Smith et al., 2019).", "Next we review costs."]),
    # question marks and quotes
    ('Does it work? The answer is "yes." Most agree.',
     ["Does it work?", 'The answer is "yes."', "Most agree."]),
    # curly quotes, as most journals use (written as escapes: \u201c opening, \u201d closing)
    ("They called it \u201chome.\u201d \u201cIt worked,\u201d one said.",
     ["They called it \u201chome.\u201d", "\u201cIt worked,\u201d one said."]),
])
def test_splits(text, expected):
    assert split(text) == expected


def test_single_sentence_and_empty_paragraph():
    assert split("Just one sentence.") == ["Just one sentence."]
    assert split("") == []


# --- citations and positions -------------------------------------------------

def test_citations_go_to_the_sentence_they_start_in():
    text = "Shelters were full in 2017.[1] Prior work agrees [2,3]. No citation here."
    cite = lambda ref_id, marker: Citation(ref_id, text.index(marker), text.index(marker) + len(marker))
    paragraph = Paragraph("introduction", text, [cite("R1", "[1]"), cite("R2", "[2,3]"), cite("R3", "[2,3]")])
    assert [s.cites for s in split_sentences(paragraph)] == [["R1"], ["R2", "R3"], []]


def test_repeated_citation_listed_once():
    text = "A [1] and again [1]."
    paragraph = Paragraph("introduction", text, [Citation("R1", 2, 5), Citation("R1", 16, 19)])
    assert split_sentences(paragraph)[0].cites == ["R1"]


def test_sentences_are_exact_slices_of_the_paragraph():
    paragraph = Paragraph("introduction", "First one.[1] Second one (n=5). Third.")
    for s in split_sentences(paragraph):
        assert paragraph.text[s.start:s.end] == s.text

# --- cases found in the first real run (Oct 16) -----------------------------

@pytest.mark.parametrize("text, expected", [
    # a space after the comma between citation groups
    ("Access is linked to homelessness.[1], [3] Given the interplay, this matters.",
     ["Access is linked to homelessness.[1], [3]", "Given the interplay, this matters."]),
    # a space between the period and the citation
    ("More research is needed on these models. [11] In April 2020, a program began.",
     ["More research is needed on these models. [11]", "In April 2020, a program began."]),
    # a dash between citation groups
    ("Rates are higher than the general population.[1]-[4] Within Canada, drinking is common.",
     ["Rates are higher than the general population.[1]-[4]", "Within Canada, drinking is common."]),
    # "v." for versus
    ("Fewer inpatient days were found for HF v. TAU (0.62). The HF group did better.",
     ["Fewer inpatient days were found for HF v. TAU (0.62).", "The HF group did better."]),
])
def test_real_run_cases(text, expected):
    assert split(text) == expected

def test_page_number_does_not_end_a_sentence():  # found in code review (Oct 16)
    assert split("As noted (Smith, 2019, p. 23), housing helps. Next.") == [
        "As noted (Smith, 2019, p. 23), housing helps.", "Next."]


# --- known misses (documented, not fixed) ------------------------------------

@pytest.mark.parametrize("text", [
    "Programs grew in the U.S. The next study was in Canada.",  # abbreviation ending a sentence
    "Participants joined Group B. Then they were followed up.",  # sentence ending in an initial
    "Housing improved. mRNA levels were not measured.",          # sentence starting lowercase
])
def test_known_misses_stay_as_one_sentence(text):
    assert split(text) == [text]
