# Shared Data Schema

## 1. Paper

- `paper_id`: unique paper identifier `doi:<lowercase DOI>`, or `pmcid:<PMCID>` if the paper has no DOI
- `title`: paper title
- `date`: publication or submission date, if known
- `url`: source URL
- `text`: the full excerpt (abstract + whole introduction). Always generated from `sentences`; never edit by hand
- `sentences`: the excerpt split into sentences, in reading order
	- `sentence_id`: `s001`, `s002`, ... unique within the paper. Labels point to these, so they must not change once labeling starts
	- `section` : where the sentence came from : `abstract`, `abstract:<heading>` (e.g. `abstract:results`), or `introduction`
	- `text` : the sentence, with citations shown as `[n]`
	- `cites` : the `ref_id`s of the references this sentence cites
-`references`: the paper's reference list. Each entry: `ref_id`, `label`, `doi`, `pmid`, `title`, `year` (missing values are `null`)
- `split`: `train`, `dev`, or `test`, set when splits are frozen (M08); `null` until then
- Other metadata: `authors`, `venue`, `doi`, `pmcid`, `source`

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

## General Rules

- Use JSON for individual records and JSONL for datasets.
- Keep IDs consistent across all components.
- Preserve source text and evidence for evaluation.
- Represent missing dates or scores as `null`.
- Do not describe raw model scores as calibrated confidence.
