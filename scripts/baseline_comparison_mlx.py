"""Evaluate the majority and two prompted Qwen baselines on development data."""

import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from comparison_baselines import LABELS, PROMPTS, load_development_data, majority_label
from evaluation.comparison_results import evaluate_predictions
from mlx_comparison import model_manifest, load_model, predict_records


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=ROOT / 'data/novelty/splits/train.jsonl')
    parser.add_argument('--dev', type=Path, default=ROOT / 'data/novelty/splits/dev.jsonl')
    parser.add_argument('--local-model-dir', type=Path, required=True)
    parser.add_argument('--model-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    train, dev = load_development_data(args.train, args.dev)
    provenance = model_manifest(args.local_model_dir, args.model_source)
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    run = {
        'complete': False, 'evaluation_split': 'dev', 'seed': 544,
        'model_provenance': provenance, 'prompts': PROMPTS,
        'generation': {'temperature': 0, 'max_new_tokens': 256, 'max_input_tokens': 4096},
        'labels': list(LABELS), 'majority_label': majority_label(train),
        'python': platform.python_version(),
        'packages': {p: version(p) for p in ('mlx', 'mlx-lm', 'transformers')},
        'data': {name: {'sha256': sha256(path.read_bytes()).hexdigest(), 'count': len(rows),
                        'class_counts': dict(Counter(r['label'] for r in rows))}
                 for name, path, rows in [('train', args.train, train), ('dev', args.dev, dev)]},
        'source_sha256': {p: sha256((ROOT / p).read_bytes()).hexdigest() for p in
                          ['scripts/baseline_comparison_mlx.py', 'src/mlx_comparison.py',
                           'src/comparison_baselines.py', 'evaluation/comparison_results.py', 'evaluation/metrics.py']},
    }
    save(args.output / 'run.json', run)
    majority = [{'comparison_id': r['comparison_id'], 'source_paper_id': r['source_paper_id'],
                 'gold': r['label'], 'predicted': run['majority_label']} for r in dev]
    (args.output / 'majority.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in majority))
    save(args.output / 'majority_metrics.json', evaluate_predictions(dev, majority))
    model, tokenizer = load_model(args.local_model_dir, seed=544)
    metrics = {}
    for prompt in PROMPTS:
        records = predict_records(model, tokenizer, dev, prompt_name=prompt,
                               output_path=args.output / f'qwen_{prompt}.jsonl')
        metrics[prompt] = evaluate_predictions(dev, records)
        save(args.output / f'qwen_{prompt}_metrics.json', metrics[prompt])
    selected = max(PROMPTS, key=lambda p: metrics[p]['macro_f1'])
    save(args.output / 'selected_baseline.json', {
        'selection_split': 'dev', 'selection_metric': 'macro_f1', 'tie_break': 'label',
        'prompt_name': selected, 'system_prompt': PROMPTS[selected],
        'development_macro_f1': {p: m['macro_f1'] for p, m in metrics.items()},
        'model_provenance': provenance, 'generation': run['generation'],
        'train_sha256': run['data']['train']['sha256'], 'dev_sha256': run['data']['dev']['sha256'],
    })
    import mlx.core as mx
    run.update(complete=True, elapsed_seconds=time.perf_counter()-start,
               peak_memory_gib=mx.get_peak_memory()/2**30)
    save(args.output / 'run.json', run)
    print(json.dumps({'selected_prompt': selected, 'metrics': metrics}), flush=True)


if __name__ == '__main__':
    main()
