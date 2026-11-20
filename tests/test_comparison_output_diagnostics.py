import pytest

from comparison_baselines import parse_answer
from evaluation.comparison_output_diagnostics import format_only_label, normalize_predictions


@pytest.mark.parametrize('answer,label', [
    ('Distinct', 'distinct'), (' DISTINCT. ', 'distinct'),
    ('Insufficient evidence', 'insufficient_evidence'),
    ('INSUFFICIENT_EVIDENCE.', 'insufficient_evidence'),
    ('Overlapping', 'overlapping'),
])
def test_recovers_only_simple_label_format(answer, label):
    assert format_only_label(answer, 'label') == label


@pytest.mark.parametrize('answer', [
    'The label is distinct', 'distinct or overlapping', 'distinct..',
    '**Distinct**', '"distinct"', 'insufficient-evidence',
    'distinct because the intervention differs', '',
])
def test_does_not_guess_labels_in_prose_or_change_other_format(answer):
    assert format_only_label(answer, 'label') == 'invalid'


@pytest.mark.parametrize('answer,label', [
    ('The intervention differs. Final: Distinct.', 'distinct'),
    ('The evidence is incomplete. final: Insufficient evidence.', 'insufficient_evidence'),
    ('Same main result. FINAL: overlapping', 'overlapping'),
])
def test_explanation_requires_one_explicit_final_label(answer, label):
    assert format_only_label(answer, 'explanation') == label


@pytest.mark.parametrize('answer', [
    'Distinct', 'The result is distinct.',
    'Final: distinct\nFINAL: overlapping',
    'Final: unknown\nFINAL: distinct',
    'FINAL: distinct. More explanation.',
    'FINAL: distinct or overlapping',
    'FINAL: **distinct**', 'FINAL: distinct..',
])
def test_explanation_rejects_ambiguous_missing_or_nonterminal_markers(answer):
    assert format_only_label(answer, 'explanation') == 'invalid'


def test_diagnostic_keeps_primary_records_and_counts_correct_and_wrong_recoveries():
    answers = ['distinct', 'Distinct', 'Overlapping.', 'distinct or overlapping']
    records = [{'raw_output': answer, 'predicted': parse_answer(answer, 'label'), 'gold': 'distinct'}
               for answer in answers]
    original = [dict(record) for record in records]
    result = normalize_predictions(records, 'label')
    assert records == original
    assert [r['predicted'] for r in result['records']] == ['distinct', 'distinct', 'overlapping', 'invalid']
    assert result['counts'] == {'total': 4, 'strict_invalid': 3, 'format_recovered': 2,
                                'recovered_correct': 1, 'recovered_incorrect': 1, 'remaining_invalid': 1}
    assert [r['strict_predicted'] for r in result['records']] == [r['predicted'] for r in records]


def test_diagnostic_rejects_raw_output_prediction_mismatch():
    with pytest.raises(ValueError, match='strict parsing'):
        normalize_predictions([{'raw_output': 'distinct', 'predicted': 'overlapping', 'gold': 'distinct'}], 'label')


def test_diagnostic_does_not_reinterpret_valid_primary_predictions():
    raw = 'Final: uncertain. FINAL: distinct'
    assert parse_answer(raw, 'explanation') == 'distinct'
    result = normalize_predictions([{'raw_output': raw, 'predicted': 'distinct', 'gold': 'distinct'}], 'explanation')
    assert result['records'][0]['predicted'] == 'distinct'
    assert result['counts']['format_recovered'] == 0
