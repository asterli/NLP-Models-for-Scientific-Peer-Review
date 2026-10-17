# Read a paper's Journal Article Tag Suite (JATS) XML from Europe PMC.

# Extracts: 
# 1) title, DOI, journal, article type
# 2) abstract + introduction as paragraphs in order
# 3) reference list

# citations inside the paragraphs are kept as positions (which characters cite which reference),
# so sentences.py can tell which sentence cites what without changing the paper's text.
# Numeric superscripts that are not citations (exponents like 10<sup>6</sup>) are written as "10^6".
# Sentence splitting will be done inside sentences.py.
 
# Referenced: https://jats.nlm.nih.gov/archiving/tag-library/

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

# -------------------------
# This module returns: 

@dataclass
class Citation:
    # one citation inside a paragraph. paragraph.text[start:end] is the marker. for example: "[2]".
    ref_id: str					# this matches Reference. ref_id, stuff like: "R2"
    start: int
    end: int

@dataclass
class Paragraph:
    section: str				# this is stuff like: "abstract", "abstract:results", "introduction"    
    text: str
    citations: list[Citation] = field(default_factory=list)

@dataclass
class Reference:
    ref_id: str
    label: str | None			# number shown in paper
    doi: str | None				# lowercase, no leading https://doi.org/ prefix
    pmid: str | None
    title: str | None
    year: int | None

@dataclass
class Article:
    title: str | None
    doi: str | None
    venue: str | None 			# name of journal
    article_type: str | None 	# "research-article", "review-article", "article-commentary"
    paragraphs: list[Paragraph] # start with the abstract first, then introduction
    references: list[Reference] 

# ---------------
# there were multiple article types that did not have their own findings. 
# we don't want these so we warn during the fetch.
NON_RESEARCH_TYPES = {"article-commentary", "editorial", "letter", "correction", "retraction",
                        "news", "book-review", "obituary", "reply"}


# some of the abstract headings are different depending on the journal
# for example: results vs findings, etc.
# map them together to a small set so the same kind of sentences get the same section label assigned
# heading will be matched by a word it contains, e.g. 'Principal findings' = results.
# for example: "Methods and analysis" will be matched to "methods"
# if heading has no matches, they will keep their name.
ABSTRACT_HEADINGS = {
    "background": ("background", "introduction", "context", "rationale", "importance"),
    "objective": ("objective", "aim", "purpose", "research question"),
    "methods": ("method", "design", "setting", "participant", "data source", "data collection",
                "eligibility", "study selection", "outcome", "exposure", "intervention", "analysis"),
    "results": ("result", "finding"),
    "conclusions": ("conclusion", "interpretation", "implication", "discussion"),
}

ABSTRACT_METADATA = ("keyword", "key word", "trial registration", "registration", "clinicaltrials gov",
                     "supplementary", "funding", "conflict of interest")

# section titles that count as introduction when section is not tagged as sec-type="intro"
INTRO_TITLES = ("introduction", "background")

# skip/remove non-paragraph elements, like figures, tables, formulas, etc.
SKIPPED_ELEMENTS = {"fig", "table-wrap", "table", "disp-formula", "graphic", "inline-graphic",
                    "media", "supplementary-material", "boxed-text", "caption", "label"}

# Abstracts that are extra summaries rather than the paper's main abstract.
SKIPPED_ABSTRACT_TYPES = {"graphical", "teaser", "toc", "summary", "author-summary", "editor-summary", "key-points"}

_WHITESPACE = re.compile(r"\s+")
# A numbered citation label: "2", "1,2", "3-5". Ranges can use a hyphen (-), en dash (\u2013),
# em dash (\u2014), or minus sign (\u2212) depending on the journal, so all four are accepted.
_NUMERIC_LABEL = re.compile(r"[\d\s,;\-\u2013\u2014\u2212]+")
_DOI_IN_TEXT = re.compile(r"10\.\d{4,9}/[^\s\"<>]+")
_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.|www\.)?doi\.org/|doi:)\s*")
# Trial/review registry IDs: NCT01234567, ISRCTN12345678, CRD42020123456.
_REGISTRY_ID = re.compile(r"\b(?:NCT\d{8}|ISRCTN\d+|CRD\d{6,})\b")
# The text right before a superscript ends in a digit (10^6) or a lone letter (R^2, kg/m^2),
# so the superscript is an exponent, not a citation.
_EXPONENT_BASE = re.compile(r"(?:\d|\b[^\W\d_])$")
# Wrappers around the paper that have their own reference lists (e.g. peer-review reports).
_ATTACHED_DOCUMENTS = ("sub-article", "response")
# ---------------
# Public function

def parse_article(xml_bytes: bytes) -> Article:
    # Parse one paper's full-text JATS XML.
    root = ET.fromstring(xml_bytes)
    _strip_namespaces(root)

    meta = root.find("front/article-meta")
    if meta is None:
        meta = ET.Element("article-meta")  # keeps the lookups below simple

    references = _references(root)
    cite_index = _CitationIndex(references, root)
    paragraphs = _abstract_paragraphs(meta, cite_index)
    body = root.find("body")
    if body is not None:
        paragraphs += _introduction_paragraphs(body, cite_index)

    return Article(
        title=_text_of(meta.find("title-group/article-title")),
        doi=normalize_doi(_text_of(meta.find("article-id[@pub-id-type='doi']"))),
        venue=_text_of(root.find("front/journal-meta//journal-title")),
        article_type=root.get("article-type"),
        paragraphs=[p for p in paragraphs if p.text],
        references=references,
    )


# ----------------------
# Abstract

def _abstract_paragraphs(meta: ET.Element, cites: _CitationIndex) -> list[Paragraph]:
    abstract = _main_abstract(meta)
    if abstract is None:
        return []
    paragraphs = []
    for child in abstract:
        if child.tag == "sec":  # structured abstract: one <sec> per heading
            heading = _text_of(child.find("title")) or ""
            if is_abstract_metadata(heading):
                continue
            section = ("abstract:" + normalize_abstract_heading(heading)).rstrip(":")
            paragraphs += [_read_paragraph(p, section, cites) for p in _paragraph_elements(child)]
        elif child.tag == "p":  # unstructured abstract
            paragraphs.append(_read_paragraph(child, "abstract", cites))
    return [p for p in paragraphs if not is_abstract_metadata(p.text)]

def is_abstract_metadata(text: str) -> bool:
    # True for 'Keywords: ...', 'Trial registration', and similar non-content lines.
    # Starting with one of these words is not enough ("Funding for housing fell." is a real
    # sentence). It must also be a short heading, a labeled line ("Funding: none"),
    # or contain a registry ID, DOI, or link.
    cleaned = _clean_title(text)
    if not cleaned.startswith(ABSTRACT_METADATA):
        return False
    is_heading = len(cleaned.split()) <= 3
    is_labeled = ":" in " ".join(text.split()[:4])
    has_id = bool(_REGISTRY_ID.search(text) or _DOI_IN_TEXT.search(text) or "http" in text.lower())
    return is_heading or is_labeled or has_id

def _main_abstract(meta: ET.Element) -> ET.Element | None:
    # The paper's main abstract, skipping graphical abstracts and plain-language summaries.
    for abstract in meta.findall("abstract"):
        if (abstract.get("abstract-type") or "").lower() not in SKIPPED_ABSTRACT_TYPES:
            return abstract
    return None


def normalize_abstract_heading(heading: str) -> str:
    """'Findings:' -> 'results', 'Methods and analysis' -> 'methods', 'Setting' -> 'methods'.
    Unknown headings come back cleaned but otherwise unchanged ('Limitations' -> 'limitations')."""
    cleaned = _clean_title(heading)
    for group, words in ABSTRACT_HEADINGS.items():
        if any(re.search(rf"\b{word}", cleaned) for word in words):  # \b: match whole-word starts only
            return group
    return cleaned

# ------------------
# Introduction

def _introduction_paragraphs(body: ET.Element, cites: _CitationIndex) -> list[Paragraph]:

    
    # search the intro paragraphs by following these rules
    # 1) top level section tagged as sec-type="intro"
    # 2) top level section titled "Introduction" or "Background" (numbering like "1." is ignored)
    # 3) paragraphs at the start of the body, before the first section (some journals don't put the intro in a section)

    # if none match, we assume no intro

    sections = body.findall("sec")
    intro = next((s for s in sections if "intro" in (s.get("sec-type") or "").lower()), None)
    if intro is None:
        intro = next((s for s in sections
                      if _clean_title(_text_of(s.find("title")) or "").startswith(INTRO_TITLES)), None)
    if intro is not None:
        return [_read_paragraph(p, "introduction", cites) for p in _paragraph_elements(intro)]

    leading = []
    for child in body:
        if child.tag == "sec":
            break
        if child.tag == "p":
            leading.append(_read_paragraph(child, "introduction", cites))
    return leading


# -----------------
# Paragraphs and citations
def _paragraph_elements(container: ET.Element):
    # Every <p> inside a section and its subsections, in reading order,
    # skipping paragraphs that belong to figures, tables, or boxes.
    for child in container:
        if child.tag in SKIPPED_ELEMENTS or child.tag == "title":
            continue
        if child.tag == "p":
            yield child
        else:
            yield from _paragraph_elements(child)

def _read_paragraph(p: ET.Element, section: str, cites: _CitationIndex) -> Paragraph:
    builder = _ParagraphBuilder()

    def walk(node: ET.Element, in_sup: bool = False) -> None:
        if node.tag in SKIPPED_ELEMENTS:
            return
        if cites.is_citation_link(node):
            label="".join(node.itertext())
            ref_ids = (node.get("rid") or "").split()  # one xref can point to several refs
            # A numbered label can name more refs than the link points to: "1-4" linked only to
            # ref 1, or the second half of a range split across two links ([1]-[4]). Add them.
            first = builder.range_start()
            numbers = f"{first}-{label}" if first else label
            if _NUMERIC_LABEL.fullmatch(numbers.strip()):
                by_label = cites.ids_for_labels(numbers)
                ref_ids = by_label + [r for r in ref_ids if r not in by_label]
            builder.add_citation(label, ref_ids, superscript=in_sup)
            return
        if _is_numeric_superscript(node):
            if cites.is_bare_citation(builder.tail()):
                builder.add_citation(node.text, cites.ids_for_labels(node.text), superscript=True)
            else:
                builder.add_text("^" + node.text.strip())  # an exponent: 10^6, kg/m^2
            return
        builder.add_text(node.text)
        for child in node:
            walk(child, in_sup or node.tag == "sup")
            builder.add_text(child.tail)  # text after a child belongs to its parent

    walk(p)
    return Paragraph(section=section, text=builder.text(), citations=builder.citations)

class _ParagraphBuilder:
    # build paragraph piece by piece, collapse whitespace and record where each citation lands.
    # positions are recorded while paragraph is built 

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._length = 0
        self.citations: list[Citation] = []

    def add_text(self, text: str | None) -> None:
        if not text:
            return
        text = _WHITESPACE.sub(" ", text)
        if self._ends_with_space():
            text = text.lstrip(" ")  # never two spaces in a row, never a leading space
        self._append(text)

    def add_citation(self, label: str, ref_ids: list[str], superscript: bool = False) -> None:
        label = _WHITESPACE.sub(" ", label).strip()
        # Superscript citations have no brackets in the XML ("2017.<sup>1</sup>"), which would
        # read as "2017.1". Wrap bare numbers in brackets: always for superscripts (even inside
        # parentheses, "(e.g., Housing First<sup>3</sup>)"), and otherwise only outside brackets.
        # Labels already in brackets ("[<xref>2</xref>]"), linked years ("(FEANTSA <xref>2018</xref>)"),
        # and author-year labels ("Smith et al., 2019") are left alone.
        if label and _NUMERIC_LABEL.fullmatch(label) and (superscript or not self._inside_brackets()):
            label = f"[{label}]"
        start = self._length
        self._append(label)
        self.citations += [Citation(ref_id, start, self._length) for ref_id in ref_ids]

    def text(self) -> str:
        return "".join(self._parts).rstrip()

    def tail(self, n: int = 2) -> str:
        # The last n characters of the text so far.
        return "".join(self._parts)[-n:]

    def range_start(self) -> str | None:
        # If the text so far ends with a citation and a dash ("[1]-", "1\u2013"), the next
        # citation closes a range. Returns that first label ("1"), otherwise None.
        if not self.citations:
            return None
        text, last = "".join(self._parts), self.citations[-1]
        if text[last.end:].strip() in ("-", "\u2013", "\u2014", "\u2212"):
            return text[last.start:last.end].strip("[] ")
        return None

    def _append(self, text: str) -> None:
        if text:
            self._parts.append(text)
            self._length += len(text)

    def _ends_with_space(self) -> bool:
        return not self._parts or self._parts[-1].endswith(" ")

    def _inside_brackets(self) -> bool:
        """True if the text so far has an opening [ or ( that hasn't been closed yet."""
        for part in reversed(self._parts):
            for ch in reversed(part):
                if ch in "])":
                    return False
                if ch in "[(":
                    return True
        return False


# ------------------
# Citation Index

class _CitationIndex:
    # Knows the paper's references, so it can recognize citations that publishers mark up in
    # different ways:

    # - a link to a reference (<xref rid="R2">), even when it isn't tagged ref-type="bibr"
    # - a plain superscript number (<sup>2</sup>) with no link at all, matched to a reference by
    #   its label (the number shown in the reference list). Only in papers that never tag
    #   citations with ref-type="bibr"; in those that do, a bare <sup>2</sup> is an exponent.
    def __init__(self, references: list[Reference], root: ET.Element) -> None:
        self._ids = {r.ref_id for r in references}
        self._by_label = {r.label.strip(".[]() "): r.ref_id for r in references if r.label}
        self._links_tagged = any(x.get("ref-type") == "bibr" for x in _iter_main(root, "xref"))

    def is_citation_link(self, node: ET.Element) -> bool:
        if node.tag != "xref":
            return False
        rids = (node.get("rid") or "").split()
        return node.get("ref-type") == "bibr" or (bool(rids) and all(r in self._ids for r in rids))

    def is_bare_citation(self, text_before: str) -> bool:
        # Is a numeric <sup> a citation? Only in papers with numbered but untagged references,
        # and not right after a digit or lone letter (10<sup>6</sup>, R<sup>2</sup>).
        return (bool(self._by_label) and not self._links_tagged
                and not _EXPONENT_BASE.search(text_before))

    def ids_for_labels(self, text: str) -> list[str]:
        # '1, 3-5' -> the ref_ids of the references labeled 1, 3, 4, and 5 (unknown labels skipped).
        labels = []
        for part in re.split(r"[,;]", text):
            ends = re.split(r"\s*[\-\u2013\u2014\u2212]\s*", part.strip())
            if len(ends) == 2 and ends[0].isdigit() and ends[1].isdigit():
                labels += [str(n) for n in range(int(ends[0]), int(ends[1]) + 1)]
            elif part.strip():
                labels.append(part.strip())
        return [self._by_label[label] for label in labels if label in self._by_label]



# -----------------
# References

def _references(root: ET.Element) -> list[Reference]:
    refs = []
    for ref in _iter_main(root, "ref"):
        all_text = " ".join(ref.itertext())
        year = _text_of(ref.find(".//year"))
        refs.append(Reference(
            ref_id=ref.get("id") or "",
            label=_text_of(ref.find("label")),
            doi=_reference_doi(ref, all_text),
            pmid=_text_of(ref.find(".//pub-id[@pub-id-type='pmid']")),
            title=_text_of(ref.find(".//article-title")) or _text_of(ref.find(".//source")),
            year=int(year[:4]) if year and year[:4].isdigit() else None,
        ))
    return refs


def _reference_doi(ref: ET.Element, all_text: str) -> str | None:
    # A reference's DOI, either tagged or written in its text
    # (some journals have only 'doi:10.xxxx/...' as plain text or a link).
    doi = _text_of(ref.find(".//pub-id[@pub-id-type='doi']"))
    if not doi:
        match = _DOI_IN_TEXT.search(all_text)
        doi = match.group(0).rstrip(".,;)") if match else None
    return normalize_doi(doi)


def normalize_doi(doi: str | None) -> str | None:
    # 'https://doi.org/10.1/ABC' -> '10.1/abc'. DOIs are case-insensitive, so we lowercase.
    if not doi:
        return None
    doi = _DOI_PREFIX.sub("", doi.strip().lower()).strip()
    return doi or None

# ------------------
# Small helpers

def _strip_namespaces(root: ET.Element) -> None:
    # '{http://www.w3.org/1998/Math/MathML}math' -> 'math', so lookups can use plain names.
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]


def _iter_main(node: ET.Element, tag: str):
    # Like node.iter(tag), but skips documents attached to the paper (peer-review reports),
    # whose reference lists reuse labels like "1" and would point citations at the wrong paper.
    for child in node:
        if child.tag == tag:
            yield child
        if child.tag not in _ATTACHED_DOCUMENTS:
            yield from _iter_main(child, tag)


def _is_numeric_superscript(node: ET.Element) -> bool:
    # <sup>2</sup> or <sup>1,3-5</sup> with no tags inside. Needs a digit: a lone <sup>,</sup>
    # between two superscript citations is just a comma.
    text = (node.text or "").strip()
    is_number = bool(_NUMERIC_LABEL.fullmatch(text)) and any(c.isdigit() for c in text)
    return node.tag == "sup" and len(node) == 0 and is_number


def _text_of(el: ET.Element | None) -> str | None:
    # All text inside an element, whitespace collapsed. None if the element is missing or empty.
    if el is None:
        return None
    text = _WHITESPACE.sub(" ", "".join(el.itertext())).strip()
    return text or None


def _clean_title(title: str) -> str:
    # example: '1. Introduction:' -> 'introduction', 'Methods/Design' -> 'methods design'.
    title = title.lower()
    title = re.sub(r"^(?:\d+(?:\.\d+)*\.?|[ivx]+[.)])\s+", "", title)  # leading "1.", "2.1", "iv."
    title = re.sub(r"[^a-z ]+", " ", title)
    return _WHITESPACE.sub(" ", title).strip()
