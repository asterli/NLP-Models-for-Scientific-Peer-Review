"""Paired uncertainty estimates for fixed classifiers on the same comparisons."""

from collections import defaultdict
import math
import random

from comparison_baselines import LABELS
from evaluation.comparison_results import evaluate_predictions


def _metrics(matrix):
    count = sum(sum(row) for row in matrix)
    correct = sum(matrix[i][i] for i in range(len(LABELS)))
    f1 = []
    for i in range(len(LABELS)):
        denominator = sum(matrix[i]) + sum(row[i] for row in matrix)
        f1.append(2 * matrix[i][i] / denominator if denominator else 0.0)
    return {'accuracy': correct / count, 'macro_f1': sum(f1) / len(LABELS)}


def _percentile(values, quantile):
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def paired_comparison(rows, fine_tuned, baseline, draws=10000, seed=544):
    """Return fine-tuned-minus-baseline differences; selection must precede evaluation."""
    if not isinstance(draws, int) or isinstance(draws, bool) or draws < 1:
        raise ValueError('draws must be a positive integer')
    fine_metrics = evaluate_predictions(rows, fine_tuned)
    base_metrics = evaluate_predictions(rows, baseline)
    fine_by_id = {r['comparison_id']: r for r in fine_tuned}
    base_by_id = {r['comparison_id']: r for r in baseline}
    label_index = {label: i for i, label in enumerate((*LABELS, 'invalid'))}
    grouped = defaultdict(list)
    fine_only = base_only = 0
    for row in rows:
        source = row['source_paper_id']
        if not isinstance(source, str) or not source.strip():
            raise ValueError('Every comparison needs a source paper ID')
        gold = label_index[row['label']]
        fine = label_index[fine_by_id[row['comparison_id']]['predicted']]
        base = label_index[base_by_id[row['comparison_id']]['predicted']]
        grouped[source].append((gold, fine, base))
        fine_only += fine == gold and base != gold
        base_only += base == gold and fine != gold

    groups = [grouped[key] for key in sorted(grouped)]
    samples = {'accuracy': [], 'macro_f1': []}
    rng = random.Random(seed)
    if len(groups) > 1:
        for _ in range(draws):
            matrices = [[[0] * (len(LABELS) + 1) for _ in LABELS] for _ in range(2)]
            for _ in groups:
                for gold, fine, base in groups[rng.randrange(len(groups))]:
                    matrices[0][gold][fine] += 1
                    matrices[1][gold][base] += 1
            fine_score, base_score = map(_metrics, matrices)
            for metric in samples:
                samples[metric].append(fine_score[metric] - base_score[metric])

    discordant = fine_only + base_only
    p_value = min(1.0, 2 * sum(math.comb(discordant, k)
                              for k in range(min(fine_only, base_only) + 1)) / 2**discordant)
    return {
        'direction': 'fine_tuned_minus_baseline',
        'count': len(rows), 'source_paper_count': len(groups),
        'fine_tuned_metrics': {k: fine_metrics[k] for k in samples},
        'baseline_metrics': {k: base_metrics[k] for k in samples},
        'differences': {
            metric: {
                'estimate': fine_metrics[metric] - base_metrics[metric],
                'percentile_95_ci': [_percentile(values, .025), _percentile(values, .975)] if values else None,
            } for metric, values in samples.items()
        },
        'bootstrap': {
            'method': 'paired source-paper cluster bootstrap',
            'draws': draws if len(groups) > 1 else 0, 'seed': seed,
            'resampled_unit': 'source_paper_id',
            'weighting': 'All comparisons in each sampled paper; pooled comparison-level metrics',
            'limitations': (
                'Intervals are exploratory with few source papers and condition on the observed papers. '
                'They do not measure variation from training seeds, model selection, or annotation uncertainty. '
                'No interval is reported with fewer than two source papers.'
            ),
        },
        'discordant_correctness': {'fine_tuned_only_correct': fine_only, 'baseline_only_correct': base_only},
        'mcnemar': {
            'method': 'exact two-sided binomial McNemar', 'p_value': p_value,
            'interpretation': 'Descriptive only: assumes independent comparison pairs, which may fail within source papers.',
        },
    }
