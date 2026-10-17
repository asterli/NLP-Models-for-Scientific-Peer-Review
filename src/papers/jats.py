# Read a paper's Journal Article Tag Suite (JATS) XML from Europe PMC.

# Extracts: 
# 1) title, DOI, journal, article type
# 2) abstract + introduction as paragraphs in order
# 3) reference list

# citations inside the paragraphs are kept as positions (which characters cite which reference),
# so sentences.py can tell which sentence cites what without changing the paper's text.
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
    cite_index = _CitationIndex(references)
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
    return _clean_title(text).startswith(ABSTRACT_METADATA)

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

    def walk(node: ET.Element) -> None:
        if node.tag in SKIPPED_ELEMENTS:
            return
        if cites.is_citation_link(node):
            ref_ids = (node.get("rid") or "").split()  # one xref can point to several refs
            builder.add_citation("".join(node.itertext()), ref_ids)
            return
        if cites.is_bare_superscript(node, builder.last_char()):
            builder.add_citation(node.text, cites.ids_for_labels(node.text))
            return
        builder.add_text(node.text)
        for child in node:
            walk(child)
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

    def add_citation(self, label: str, ref_ids: list[str]) -> None:
        label = _WHITESPACE.sub(" ", label).strip()
        # Superscript citations have no brackets in the XML ("2017.<sup>1</sup>"), which would
        # read as "2017.1". Wrap bare numbers in brackets; leave labels that are already inside
        # brackets ("[<xref>2</xref>]") and author-year labels ("Smith et al., 2019") alone.
        if label and _NUMERIC_LABEL.fullmatch(label) and not self._inside_brackets():
            label = f"[{label}]"
        start = self._length
        self._append(label)
        self.citations += [Citation(ref_id, start, self._length) for ref_id in ref_ids]

    def text(self) -> str:
        return "".join(self._parts).rstrip()

    def last_char(self) -> str:
        return self._parts[-1][-1] if self._parts else ""

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
    #   its label (the number shown in the reference list)
    def __init__(self, references: list[Reference]) -> None:
        self._ids = {r.ref_id for r in references}
        self._by_label = {r.label.strip(".[]() "): r.ref_id for r in references if r.label}

    def is_citation_link(self, node: ET.Element) -> bool:
        if node.tag != "xref":
            return False
        rids = (node.get("rid") or "").split()
        return node.get("ref-type") == "bibr" or (bool(rids) and all(r in self._ids for r in rids))

    def is_bare_superscript(self, node: ET.Element, previous_char: str) -> bool:
        # <sup>2</sup> with nothing inside it, in a paper whose references are numbered. Not
        # right after a digit, so exponents like 10<sup>6</sup> aren't mistaken for citations.
        return (node.tag == "sup" and len(node) == 0 and bool(self._by_label)
                and _NUMERIC_LABEL.fullmatch((node.text or "").strip() or "x") is not None
                and not previous_char.isdigit())

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
    for ref in root.iter("ref"):
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
    doi = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi or None

# ------------------
# Small helpers

def _strip_namespaces(root: ET.Element) -> None:
    # '{http://www.w3.org/1998/Math/MathML}math' -> 'math', so lookups can use plain names.
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]


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
