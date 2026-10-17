# Shared Data Schema

## 1. Paper

- `paper_id`: unique paper identifier `doi:<lowercase DOI>`, or `pmcid:<PMCID>` if the paper has no DOI
- `title`: paper title
- `date`: publication or submission date, if known
- `url`: source URL
- `text`: the full excerpt (abstract + whole introduction). Always generated from `sentences`; never edit by hand
- `sentences`: the excerpt split into sentences, in reading order
	- `sentence_id`: `s001`, `s002`, ... unique within the paper only, so refer to a sentence as (`paper_id`, `sentence_id`). Labels point to these, so they must not change once labeling starts
	- `section` : where the sentence came from : `abstract`, `abstract:<heading>` (e.g. `abstract:results`), or `introduction`
	- `text` : the sentence. Numbered citations are shown as `[n]`, author-year citations as written (`Smith et al., 2019`), and exponents as `^` (`kg/m^2`)
	- `cites` : the `ref_id`s of the references this sentence cites
- `references`: the paper's reference list. Each entry: `ref_id`, `label`, `doi`, `pmid`, `title`, `year` (missing values are `null`)
- `split`: `train`, `dev`, or `test`, set when splits are frozen (M08); `null` until then
- Other metadata: `authors`, `venue`, `doi`, `pmcid`, `article_type` (e.g. `research-article`), `source`

## 2. Claim

- `claim_id`: unique claim identifier
- `paper_id`: ID of the source paper
- `sentence_id`: ID of the original sentence
- `text`: extracted claim text

## 3. Retrieval Candidate

- `paper_id`: ID of the retrieved prior-work paper
- `title`: paper title
- `score`: retrieval ranking score
- `date`: publication date, if known
- `url`: source URL
- `evidence`: relevant passage from the paper

## 4. Novelty Prediction

- `claim_id`: ID of the claim being assessed
- `prior_paper_id`: ID of the comparison paper
- `label`: `overlapping`, `distinct`, or `insufficient_evidence`
- `score`: raw model score, if available
- `reason`: brief explanation supported by the evidence

## 5. Corpus Document

One older work in `data/corpus/corpus.jsonl`, the pool that retrieval searches.

- `paper_id`: `doi:<lowercase DOI>`, or `openalex:<work ID>` if there is no DOI (same format as Paper)
- `title`, `date` (YYYY-MM-DD), `url`
- `abstract`: rebuilt from OpenAlex; `null` for about 20% of works, so only their title is searchable
- `venue`, `year`, `type`, `doi`, `pmid`, `openalex_id`
- `source`: `reference` (cited by a source paper) or `search` (topic search, planned)
- `cited_by`: `paper_id`s of the source papers that cite this work

Not filtered by date. Some works are newer than some source papers, and 5 source papers are in the corpus because other source papers cite them. Retrieval (M23) must drop works dated after the source paper, and the source paper itself.

## 6. Citation Link

One line of `data/corpus/citation_links.jsonl`: which corpus work a source paper's reference points to.

- `paper_id`, `ref_id`: the source paper and one of its `references`
- `corpus_id`: the `paper_id` of the matching corpus document
- `matched_by`: `doi` or `pmid`

To find the older papers a sentence cites: sentence `cites` -> `ref_id` -> `corpus_id`. PMC7576524 and PMC11773779 have no reference DOIs, so they have no links; their cited works are in the corpus with `cited_by` only.

## General Rules

- Use JSON for individual records and JSONL for datasets.
- Keep IDs consistent across all components.
- Preserve source text and evidence for evaluation.
- Represent missing dates or scores as `null`.
- Do not describe raw model scores as calibrated confidence.
