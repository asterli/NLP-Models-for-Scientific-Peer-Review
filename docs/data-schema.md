# Shared Data Schema

## 1. Paper

- `paper_id`: unique paper identifier
- `title`: paper title
- `date`: publication or submission date, if known
- `url`: source URL
- `text`: abstract or introduction excerpt

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
