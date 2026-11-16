import pytest

from evaluation.comparison_significance import paired_comparison


def data(spec):
    rows, fine, base = [], [], []
    for i, (paper, gold, fine_label, base_label) in enumerate(spec):
        row = {'comparison_id': str(i), 'source_paper_id': paper, 'label': gold}
        rows.append(row)
        common = {'comparison_id': str(i), 'source_paper_id': paper, 'gold': gold}
        fine.append(common | {'predicted': fine_label})
        base.append(common | {'predicted': base_label})
    return rows, fine, base


def test_identical_predictions_have_zero_difference_and_p_one():
    args = data([('a', 'distinct', 'distinct', 'distinct'),
                 ('b', 'overlapping', 'invalid', 'invalid')])
    result = paired_comparison(*args, draws=100)
    assert result['differences']['macro_f1'] == {'estimate': 0, 'percentile_95_ci': [0, 0]}
    assert result['differences']['accuracy'] == {'estimate': 0, 'percentile_95_ci': [0, 0]}
    assert result['mcnemar']['p_value'] == 1


def test_cluster_bootstrap_preserves_whole_papers_and_paired_direction():
    args = data([('a', 'distinct', 'distinct', 'overlapping'),
                 ('a', 'distinct', 'distinct', 'overlapping'),
                 ('b', 'distinct', 'overlapping', 'distinct')])
    result = paired_comparison(*args, draws=1000)
    assert result['source_paper_count'] == 2
    assert result['differences']['accuracy']['estimate'] == pytest.approx(1 / 3)
    # Sampling the same source twice gives two whole copies, despite duplicate IDs.
    assert result['differences']['accuracy']['percentile_95_ci'] == [-1, 1]
    assert result == paired_comparison(args[0], list(reversed(args[1])), list(reversed(args[2])), draws=1000)
    swapped = paired_comparison(args[0], args[2], args[1], draws=1000)
    assert swapped['differences']['accuracy']['estimate'] == pytest.approx(-1 / 3)


def test_exact_mcnemar_and_invalid_predictions_count_as_errors():
    args = data([(str(i), 'distinct', 'distinct', 'invalid') for i in range(6)])
    result = paired_comparison(*args, draws=100)
    assert result['discordant_correctness'] == {'fine_tuned_only_correct': 6, 'baseline_only_correct': 0}
    assert result['mcnemar']['p_value'] == pytest.approx(2 / 64)
    assert result['differences']['accuracy']['percentile_95_ci'] == [1, 1]
    assert result['differences']['macro_f1']['estimate'] == pytest.approx(1 / 3)


def test_single_source_does_not_claim_an_uncertainty_interval():
    args = data([('a', 'distinct', 'distinct', 'overlapping')])
    result = paired_comparison(*args)
    assert result['bootstrap']['draws'] == 0
    assert result['differences']['accuracy']['percentile_95_ci'] is None


def test_validation_rejects_mismatched_predictions():
    rows, fine, base = data([('a', 'distinct', 'distinct', 'distinct')])
    base[0]['gold'] = 'overlapping'
    with pytest.raises(ValueError, match='Gold label mismatch'):
        paired_comparison(rows, fine, base)


@pytest.mark.parametrize('draws', [0, -1, 1.5, True])
def test_invalid_draw_count(draws):
    with pytest.raises(ValueError, match='positive integer'):
        paired_comparison(*data([('a', 'distinct', 'distinct', 'distinct')]), draws=draws)
