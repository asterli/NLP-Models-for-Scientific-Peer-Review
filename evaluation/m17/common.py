import json, re, os
from pathlib import Path
PACKAGE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('M17_OUTPUT', PACKAGE / 'outputs')).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
LABELS = ['overlapping', 'distinct', 'insufficient_evidence']
LETTERS = ['A', 'B', 'C']
ROWS = [json.loads(l) for l in (PACKAGE / 'pilot.jsonl').read_text().splitlines()]
GROUPS = sorted({r['source_pmcid'] for r in ROWS})
RULES = 'Compare a new scientific claim with the supplied older passage, using only these two passages.\nA = Overlapping: the older passage already says what the claim says.\nB = Distinct: the claim says something the older passage does not.\nC = Insufficient evidence: you cannot tell from these two passages.\nApply these rules:\n1. Same outcome is not enough. If the intervention is different, it is a different contribution.\n2. Do not judge whether the claim is true. Judge only whether the older passage already says it.\n3. Main point test: if the older passage covers the main point and the claim just adds something small, choose Overlapping. If a main part is missing or contradicted, choose Distinct.\nReturn exactly one letter: A, B, or C.'

def user_text(r):
    return 'New claim:\n' + r['claim'] + '\n\nOlder passage:\n' + r['prior_work']

def messages(r):
    return [{'role': 'system', 'content': RULES}, {'role': 'user', 'content': user_text(r)}]

def norm(s):
    return re.sub('\\W', '', s.lower())

def split(group):
    test = [r for r in ROWS if r['source_pmcid'] == group]
    train = [r for r in ROWS if r['source_pmcid'] != group]
    test_claims = {norm(r['claim']) for r in test}
    test_prior = {norm(r['prior_work']) for r in test}
    dropped = [r for r in train if norm(r['prior_work']) in test_claims or norm(r['claim']) in test_prior]
    train = [r for r in train if r not in dropped]
    assert not {norm(r['claim']) for r in train} & test_claims
    return (train, test, dropped)

def metrics(records):
    matrix = [[sum((r['gold'] == a and r['predicted'] == b for r in records)) for b in LABELS] for a in LABELS]
    per = {}
    for i, label in enumerate(LABELS):
        tp = matrix[i][i]
        support = sum((r['gold'] == label for r in records))
        predicted = sum((r['predicted'] == label for r in records))
        p = tp / predicted if predicted else 0.0
        rec = tp / support if support else 0.0
        per[label] = {'precision': p, 'recall': rec, 'f1': 2 * p * rec / (p + rec) if p + rec else 0.0, 'support': support}
    return {'n': len(records), 'accuracy': sum((matrix[i][i] for i in range(3))) / len(records), 'macro_f1': sum((x['f1'] for x in per.values())) / 3, 'per_class': per, 'confusion': matrix, 'invalid_predictions': sum((r['predicted'] not in LABELS for r in records))}

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')
