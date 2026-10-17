# Splits a paragraph into sentences. Also tracks what each sentence cites

# Input comes as a paragraph from jats.py. 
# Each sentence is a slice of the paragraph's text.
# A citation in a sentence means that sentence is the citer, not the next sentence.

# Splitting is decided based on:
#   1. Find every possible end: . ! or ? (plus any citations written right after it,
#      like "2017.[1]"), followed by a space and a capital letter, digit, quote, or opening parenthesis.
#   2. Skip ends that come right after an abbreviation ("et al.", "Fig.") or an initial ("J.").

# Known misses, kept on purpose to keep the rules simple: an abbreviation that really ends a
# sentence ("...in the U.S. The next..."), a sentence ending in a single capital letter
# ("...Group B. Then..."), and a sentence starting lowercase ("mRNA levels..."). All three
# produce one overly long sentence, which fetch_papers.py flags for a person to fix.

from __future__ import annotations

import re
from dataclasses import dataclass

from papers.jats import Paragraph


@dataclass
class Sentence:
    text: str
    start: int  # paragraph.text[start:end] == text
    end: int
    cites: list[str]  # ref_ids of the references this sentence cites, in order


# Words that end with a period without ending the sentence (lowercase, final period removed).
ABBREVIATIONS = {
    "al", "e.g", "i.e", "vs", "v", "cf", "ca", "approx", "fig", "figs", "eq", "no", "vol",
    "ref", "refs", "dr", "mr", "mrs", "ms", "st", "inc", "ltd", "u.s", "u.k",
}

# One numbered citation group, like "[3]", "[1,2]", or "[4-6]" (any of the dashes from jats.py).
_CITATION = r"\[[\d\s,;\-\u2013\u2014\u2212]+\]"

# A possible sentence end. \u201d and \u2019 are curly closing quotes, \u201c a curly opening one.
_POSSIBLE_END = re.compile(rf"""
    [.!?][\"\u201d\u2019]?                          # end punctuation, maybe a closing quote
    (?:\s*{_CITATION}(?:\s*[,;\-\u2013\u2014]\s*{_CITATION})*)?   # citations after it
    (?=\s+[A-Z0-9\"\u201c(])                        # then a space and the next sentence's start
    """, re.VERBOSE)

_LAST_WORD = re.compile(r"(\S+)$")


def split_sentences(paragraph: Paragraph) -> list[Sentence]:
    text = paragraph.text
    ends = [m.end() for m in _POSSIBLE_END.finditer(text) if not _follows_abbreviation(text, m.start())]

    sentences, start = [], 0
    for end in ends + [len(text)]:
        sentence = _make_sentence(paragraph, start, end)
        if sentence:
            sentences.append(sentence)
        start = end
    return sentences


def _follows_abbreviation(text: str, punct_index: int) -> bool:
    # True if the word just before this period is an abbreviation or a single initial.
    match = _LAST_WORD.search(text[:punct_index])
    if not match:
        return False
    word = match.group(1).lstrip("([")
    return word.lower() in ABBREVIATIONS or (len(word) == 1 and word.isupper())


def _make_sentence(paragraph: Paragraph, start: int, end: int) -> Sentence | None:
    # The sentence between two positions, with surrounding spaces trimmed off.
    text = paragraph.text
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start == end:
        return None
    cites = [c.ref_id for c in paragraph.citations if start <= c.start < end]
    return Sentence(text[start:end], start, end, list(dict.fromkeys(cites)))  # dedupe, keep order
