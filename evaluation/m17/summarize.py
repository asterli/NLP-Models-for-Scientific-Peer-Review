from collections import Counter
import json
from common import GROUPS, LABELS, PACKAGE, ROOT, ROWS, metrics, save, split


def summarize(predictions):
    gold = {row['comparison_id']: row for row in ROWS}
    result = {}
    for name, records in predictions.items():
        assert len(records) == len(gold)
        assert {r['comparison_id'] for r in records} == set(gold)
        assert all(r['gold'] == gold[r['comparison_id']]['label'] for r in records)
        assert all(r['source_pmcid'] == gold[r['comparison_id']]['source_pmcid'] for r in records)
        result[name] = metrics(records)
    return result


def main():
    predictions = {'majority': []}
    for group in GROUPS:
        train, test, _ = split(group)
        label = Counter(r['label'] for r in train).most_common(1)[0][0]
        predictions['majority'].extend({
            'comparison_id': r['comparison_id'], 'source_pmcid': group,
            'gold': r['label'], 'predicted': label,
        } for r in test)
    for name in ['qwen_zero', 'qwen_reason']:
        path = ROOT / f'{name}.json'
        if path.exists():
            predictions[name] = json.loads(path.read_text())['records']
    for model, pattern in [('qwen', 'qwen_{group}_{seed}/results.json'),
                           ('distilbert', 'encoder_{group}_{seed}.json')]:
        for seed in [544, 2026]:
            paths = [ROOT / pattern.format(group=g, seed=seed) for g in GROUPS]
            if all(path.exists() for path in paths):
                predictions[f'{model}_{seed}'] = [
                    row for path in paths for row in json.loads(path.read_text())['records']]
    result = {'labels': LABELS, 'metrics': summarize(predictions)}
    reference = json.loads((PACKAGE / 'reference_predictions.json').read_text())
    result['reference_metrics'] = summarize(reference)
    result['prediction_differences_from_reference'] = {
        name: sum(r['predicted'] != {x['comparison_id']: x['predicted'] for x in reference[name]}[r['comparison_id']]
                  for r in records) for name, records in predictions.items()
    }
    save(ROOT / 'predictions.json', predictions)
    save(ROOT / 'summary.json', result)
    for name, values in result['metrics'].items():
        print(f"{name}: accuracy={values['accuracy']:.4f}, macro-F1={values['macro_f1']:.4f}")


if __name__ == '__main__':
    main()
