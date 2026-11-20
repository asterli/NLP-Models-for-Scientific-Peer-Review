"""Freeze development-selected models, then evaluate them once on held-out comparisons."""

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from comparison_baselines import LABELS, PROMPTS, load_development_data, load_split, majority_label
from evaluation.comparison_results import evaluate_predictions, error_examples, read_predictions
from evaluation.comparison_significance import paired_comparison
from evaluation.comparison_output_diagnostics import normalize_predictions
from mlx_comparison import file_digest, load_model, predict_records, MLXComparisonClassifier, runtime_manifest


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def freeze(args):
    train, dev = load_development_data(args.train, args.dev)
    baseline = json.loads((args.baseline / 'selected_baseline.json').read_text())
    baseline_run = json.loads((args.baseline / 'run.json').read_text())
    if not baseline_run['complete']:
        raise ValueError('Baseline run is incomplete')
    if baseline['train_sha256'] != file_digest(args.train) or baseline['dev_sha256'] != file_digest(args.dev):
        raise ValueError('Baseline data hashes differ')
    if baseline['system_prompt'] != PROMPTS[baseline['prompt_name']]:
        raise ValueError('Baseline prompt has changed')
    if (baseline_run['model_provenance'] != baseline['model_provenance'] or
        baseline_run['generation'] != baseline['generation'] or
        any(baseline_run['data'][s]['sha256'] != baseline[f'{s}_sha256'] for s in ('train', 'dev'))):
        raise ValueError('Baseline manifest differs from its selected configuration')
    baseline_scores = {p: evaluate_predictions(dev, read_predictions(args.baseline / f'qwen_{p}.jsonl'))['macro_f1']
                       for p in PROMPTS}
    if (baseline_scores != baseline['development_macro_f1'] or
        max(PROMPTS, key=baseline_scores.get) != baseline['prompt_name']):
        raise ValueError('Baseline selection differs from saved development predictions')
    candidates = []
    for path in args.training_runs:
        run = json.loads((path / 'training_run.json').read_text())
        if not run['complete'] or not run['reload_verified']:
            raise ValueError('Every training candidate must finish and pass fresh-process reload')
        if run['model_provenance'] != baseline['model_provenance']:
            raise ValueError('Base model differs across experiments')
        for split, data_path in [('train', args.train), ('dev', args.dev)]:
            if run['data'][split]['sha256'] != file_digest(data_path):
                raise ValueError('Training data hashes differ')
        epoch_scores = {epoch: evaluate_predictions(dev, read_predictions(path / f'dev_epoch_{epoch}.jsonl'))['macro_f1']
                        for epoch in range(1, run['epochs'] + 1)}
        best_epoch = max(epoch_scores, key=epoch_scores.get)
        if best_epoch != run['best_epoch'] or epoch_scores[best_epoch] != run['best_development_macro_f1']:
            raise ValueError('Training selection differs from saved development predictions')
        settings = json.loads((path / 'adapter/classifier_config.json').read_text())
        if json.loads((path / 'adapter/adapter_config.json').read_text()) != run['lora']:
            raise ValueError('Adapter configuration differs from the training run')
        for key in ('model_provenance', 'prompt_name', 'system_prompt', 'generation', 'seed',
                    'best_epoch', 'best_development_macro_f1', 'max_input_tokens'):
            if settings[key] != run[key]:
                raise ValueError('Saved classifier differs from the selected training run')
        check = json.loads((path / 'reload_check.json').read_text())
        if not check['passed'] or check['adapter_sha256'] != file_digest(path / 'adapter/adapters.safetensors'):
            raise ValueError('Adapter differs from the verified reload')
        candidates.append({'run': str(path.resolve()), 'learning_rate': run['learning_rate'],
                           'best_epoch': run['best_epoch'], 'macro_f1': run['best_development_macro_f1']})
    candidates.sort(key=lambda c: (-c['macro_f1'], c['learning_rate'], c['run']))
    selected = candidates[0]
    adapter = Path(selected['run']) / 'adapter'
    values = {'majority': evaluate_predictions(dev, [
        {'comparison_id': r['comparison_id'], 'source_paper_id': r['source_paper_id'],
         'gold': r['label'], 'predicted': majority_label(train)} for r in dev])['macro_f1'],
        'prompted': baseline['development_macro_f1'][baseline['prompt_name']],
        'fine_tuned': selected['macro_f1']}
    protocol = {
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'selection_split': 'dev',
        'selection_metric': 'macro_f1', 'training_tie_break': 'lower learning rate, then run path',
        'deployment_tie_break': ['majority', 'prompted', 'fine_tuned'],
        'recommended_method_by_development': max(values, key=values.get),
        'development_macro_f1': values, 'candidates': candidates, 'selected_training': selected,
        'adapter': str(adapter), 'adapter_sha256': file_digest(adapter / 'adapters.safetensors'),
        'classifier_config_sha256': file_digest(adapter / 'classifier_config.json'),
        'adapter_config_sha256': file_digest(adapter / 'adapter_config.json'),
        'baseline': baseline, 'majority_label': majority_label(train),
        'train_sha256': file_digest(args.train), 'dev_sha256': file_digest(args.dev),
        'labels': list(LABELS),
        'comparison_plan': 'Evaluate majority, development-selected prompted baseline, and development-selected fine-tuned model on identical test pairs. No test-based model selection.',
        'secondary_analysis': 'Also score the other predeclared baseline prompt and conservative format-only normalization for all model outputs. These analyses do not change selection or primary strict scores.',
    }
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / 'frozen_protocol.json', protocol)
    print(json.dumps(protocol), flush=True)


def evaluate(args):
    protocol = json.loads(args.protocol.read_text())
    train, dev = load_development_data(args.train, args.dev)
    for split, path in [('train', args.train), ('dev', args.dev)]:
        if file_digest(path) != protocol[f'{split}_sha256']:
            raise ValueError('Data changed after protocol freeze')
    adapter = getattr(args, 'adapter', None) or Path(protocol['adapter'])
    if (file_digest(adapter / 'adapters.safetensors') != protocol['adapter_sha256'] or
        file_digest(adapter / 'classifier_config.json') != protocol['classifier_config_sha256'] or
        file_digest(adapter / 'adapter_config.json') != protocol['adapter_config_sha256']):
        raise ValueError('Selected adapter changed after protocol freeze')
    for name, digest in protocol['baseline']['model_provenance']['file_sha256'].items():
        if file_digest(args.local_model_dir / name) != digest:
            raise ValueError('Base model differs from the frozen baseline')
    test = load_split(args.test, 'test')
    for field in ('comparison_id', 'source_paper_id'):
        if {r[field] for r in test} & {r[field] for r in train + dev}:
            raise ValueError(f'Test {field} overlaps training/development')
    args.output.mkdir(parents=True, exist_ok=False)
    run = {'complete': False, 'evaluation_split': 'test', 'test_sha256': file_digest(args.test),
           'protocol_sha256': file_digest(args.protocol), 'count': len(test), 'runtime': runtime_manifest(),
           'source_sha256': {p: file_digest(ROOT / p) for p in ['scripts/evaluate_comparison_mlx.py',
               'src/mlx_comparison.py', 'src/comparison_baselines.py', 'evaluation/comparison_results.py',
               'evaluation/comparison_significance.py', 'evaluation/comparison_output_diagnostics.py', 'evaluation/metrics.py']}}
    save(args.output / 'run.json', run)
    records = {'majority': [{'comparison_id': r['comparison_id'], 'source_paper_id': r['source_paper_id'],
                            'gold': r['label'], 'predicted': protocol['majority_label']} for r in test]}
    (args.output / 'majority.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records['majority']))
    model, tokenizer = load_model(args.local_model_dir)
    base = protocol['baseline']
    records['prompted'] = predict_records(model, tokenizer, test, base['prompt_name'],
        base['generation']['max_input_tokens'], base['generation']['max_new_tokens'], base['system_prompt'],
        output_path=args.output / 'prompted.jsonl')
    alternate = next(p for p in PROMPTS if p != base['prompt_name'])
    records['alternate_prompted'] = predict_records(model, tokenizer, test, alternate,
        base['generation']['max_input_tokens'], base['generation']['max_new_tokens'], PROMPTS[alternate],
        output_path=args.output / 'alternate_prompted.jsonl')
    del model, tokenizer
    gc.collect()
    import mlx.core as mx
    mx.clear_cache()
    classifier = MLXComparisonClassifier(adapter, args.local_model_dir)
    settings = classifier.settings
    records['fine_tuned'] = predict_records(classifier.model, classifier.tokenizer, test,
        settings['prompt_name'], settings['max_input_tokens'], settings['generation']['max_new_tokens'],
        settings['system_prompt'], output_path=args.output / 'fine_tuned.jsonl')
    metrics = {name: evaluate_predictions(test, rows) for name, rows in records.items()}
    save(args.output / 'metrics.json', metrics)
    save(args.output / 'paired_comparisons.json', {name: paired_comparison(test, records['fine_tuned'], records[name])
                                                for name in ('majority', 'prompted', 'alternate_prompted')})
    normalized = {'majority': records['majority']}
    diagnostics = {}
    for name, prompt in [('prompted', base['prompt_name']), ('alternate_prompted', alternate),
                         ('fine_tuned', settings['prompt_name'])]:
        diagnostic = normalize_predictions(records[name], prompt)
        normalized[name] = diagnostic['records']
        diagnostics[name] = {'counts': diagnostic['counts'],
                             'metrics': evaluate_predictions(test, normalized[name])}
    diagnostics['paired_comparisons'] = {name: paired_comparison(test, normalized['fine_tuned'], normalized[name])
                                        for name in ('majority', 'prompted', 'alternate_prompted')}
    save(args.output / 'formatting_diagnostics.json', diagnostics)
    save(args.output / 'error_examples.json', {name: error_examples(test, rows) for name, rows in records.items()})
    run.update(complete=True, peak_memory_gib=mx.get_peak_memory()/2**30,
               primary_methods=['majority', 'prompted', 'fine_tuned'], alternate_prompt=alternate)
    save(args.output / 'run.json', run)
    print(json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'evaluate'))
    parser.add_argument('--train', type=Path, default=ROOT / 'data/novelty/splits/train.jsonl')
    parser.add_argument('--dev', type=Path, default=ROOT / 'data/novelty/splits/dev.jsonl')
    parser.add_argument('--test', type=Path, default=ROOT / 'data/novelty/splits/test.jsonl')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--training-runs', type=Path, nargs='+')
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--adapter', type=Path, help='Relocated saved adapter; frozen hashes must still match')
    parser.add_argument('--local-model-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'freeze':
        if not args.baseline or not args.training_runs:
            parser.error('freeze requires --baseline and --training-runs')
        freeze(args)
    else:
        if not args.protocol or not args.local_model_dir:
            parser.error('evaluate requires --protocol and --local-model-dir')
        evaluate(args)


if __name__ == '__main__':
    main()
