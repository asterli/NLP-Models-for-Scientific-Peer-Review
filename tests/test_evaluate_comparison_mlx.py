import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import evaluate_comparison_mlx as runner
from comparison_baselines import LABELS, PROMPTS, load_split
from evaluation.comparison_results import evaluate_predictions
from mlx_comparison import file_digest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def change_json(path, **changes):
    value = json.loads(path.read_text())
    value.update(changes)
    write_json(path, value)


@pytest.fixture
def fixture(tmp_path):
    for split in ('train', 'dev'):
        (tmp_path / f'{split}.jsonl').write_text(''.join(json.dumps({
            'comp_id': f'{split}-{i}', 'newer_paper_id': f'{split}-paper',
            'claim_text': 'claim', 'older_passage': 'passage',
            'label': label, 'm16_split': split,
        }) + '\n' for i, label in enumerate(LABELS)))
    train, dev = tmp_path / 'train.jsonl', tmp_path / 'dev.jsonl'
    dev_rows = load_split(dev, 'dev')
    predictions = [{'comparison_id': r['comparison_id'], 'source_paper_id': r['source_paper_id'],
                    'gold': r['label'], 'predicted': r['label'], 'raw_output': r['label']} for r in dev_rows]
    metrics = evaluate_predictions(dev_rows, predictions)
    baseline = tmp_path / 'baseline'
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'model.safetensors').write_bytes(b'base weights')
    provenance = {'model': 'Qwen/Qwen2.5-7B-Instruct',
                  'file_sha256': {'model.safetensors': file_digest(model / 'model.safetensors')}}
    generation = {'temperature': 0, 'max_input_tokens': 4096, 'max_new_tokens': 256}
    data = {split: {'sha256': file_digest(path)} for split, path in [('train', train), ('dev', dev)]}
    write_json(baseline / 'run.json', {'complete': True, 'model_provenance': provenance,
                                     'data': data, 'generation': generation})
    selected = {'prompt_name': 'label', 'system_prompt': PROMPTS['label'],
                'selection_split': 'dev', 'selection_metric': 'macro_f1', 'tie_break': 'label',
                'train_sha256': data['train']['sha256'], 'dev_sha256': data['dev']['sha256'],
                'model_provenance': provenance, 'generation': generation,
                'development_macro_f1': {p: metrics['macro_f1'] for p in PROMPTS}}
    write_json(baseline / 'selected_baseline.json', selected)
    for prompt in PROMPTS:
        (baseline / f'qwen_{prompt}.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in predictions))
        write_json(baseline / f'qwen_{prompt}_metrics.json', metrics)
    training_runs = []
    for name, lr in [('larger_lr', 5e-5), ('smaller_lr', 1e-5)]:
        path = tmp_path / name
        training_runs.append(path)
        lora = {'fine_tune_type': 'lora', 'num_layers': 28,
                'lora_parameters': {'rank': 8, 'scale': 16, 'dropout': 0,
                                    'keys': ['self_attn.q_proj', 'self_attn.v_proj']}}
        run = {'complete': True, 'reload_verified': True, 'model_provenance': provenance,
               'data': data, 'learning_rate': lr, 'best_epoch': 1,
               'best_development_macro_f1': metrics['macro_f1'], 'epochs': 1,
               'history': [{'epoch': 1, 'macro_f1': metrics['macro_f1']}],
               'model': 'Qwen/Qwen2.5-7B-Instruct', 'labels': list(LABELS),
               'prompt_name': 'label', 'system_prompt': PROMPTS['label'], 'generation': generation,
               'max_input_tokens': 4096, 'max_new_tokens': 256, 'seed': 544, 'lora': lora}
        write_json(path / 'training_run.json', run)
        write_json(path / 'adapter/classifier_config.json', run)
        write_json(path / 'adapter/adapter_config.json', lora)
        (path / 'adapter/adapters.safetensors').write_bytes(b'adapter weights')
        write_json(path / 'reload_check.json', {'passed': True, 'fresh_process': True,
            'adapter_sha256': file_digest(path / 'adapter/adapters.safetensors')})
        (path / 'dev_epoch_1.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in predictions))
        write_json(path / 'metrics_epoch_1.json', metrics)
    return SimpleNamespace(train=train, dev=dev, test=tmp_path / 'UNREAD_TEST.jsonl',
        baseline=baseline, training_runs=training_runs, output=tmp_path / 'freeze', local_model_dir=model)


def test_freeze_uses_development_ties_and_never_loads_test(fixture, monkeypatch):
    monkeypatch.setattr(runner, 'load_split', lambda *a: pytest.fail('freeze accessed test data'))
    runner.freeze(fixture)
    protocol = json.loads((fixture.output / 'frozen_protocol.json').read_text())
    assert protocol['selected_training']['learning_rate'] == 1e-5
    assert protocol['recommended_method_by_development'] == 'prompted'
    assert not fixture.test.exists()


@pytest.mark.parametrize('field', ['complete', 'reload_verified'])
def test_freeze_rejects_unfinished_or_unverified_training(fixture, field):
    change_json(fixture.training_runs[0] / 'training_run.json', **{field: False})
    with pytest.raises(ValueError, match='finish and pass'):
        runner.freeze(fixture)
    assert not fixture.output.exists()


def test_freeze_rejects_baseline_from_different_data(fixture):
    change_json(fixture.baseline / 'selected_baseline.json', train_sha256='wrong')
    with pytest.raises(ValueError, match='hashes differ'):
        runner.freeze(fixture)


def test_freeze_rejects_different_base_weights(fixture):
    change_json(fixture.training_runs[0] / 'training_run.json', model_provenance={'wrong': True})
    with pytest.raises(ValueError, match='Base model differs'):
        runner.freeze(fixture)


def evaluation_args(fixture):
    runner.freeze(fixture)
    return SimpleNamespace(**(vars(fixture) | {'protocol': fixture.output / 'frozen_protocol.json',
                                              'output': fixture.output.parent / 'evaluation'}))


@pytest.mark.parametrize('filename', ['adapters.safetensors', 'classifier_config.json', 'adapter_config.json'])
def test_evaluate_rejects_changed_adapter_before_accessing_test(fixture, monkeypatch, filename):
    args = evaluation_args(fixture)
    protocol = json.loads(args.protocol.read_text())
    with (Path(protocol['adapter']) / filename).open('ab') as target:
        target.write(b'changed')
    monkeypatch.setattr(runner, 'load_split', lambda *a: pytest.fail('accessed test before integrity checks'))
    with pytest.raises(ValueError, match='adapter changed'):
        runner.evaluate(args)
    assert not args.output.exists()


def test_evaluate_rejects_changed_data_before_accessing_test(fixture, monkeypatch):
    args = evaluation_args(fixture)
    args.train.write_text(args.train.read_text() + '\n')
    monkeypatch.setattr(runner, 'load_split', lambda *a: pytest.fail('accessed test before integrity checks'))
    with pytest.raises(ValueError, match='Data changed'):
        runner.evaluate(args)


def test_evaluate_rejects_changed_base_before_accessing_test(fixture, monkeypatch):
    args = evaluation_args(fixture)
    (args.local_model_dir / 'model.safetensors').write_bytes(b'different weights')
    monkeypatch.setattr(runner, 'load_split', lambda *a: pytest.fail('accessed test before integrity checks'))
    with pytest.raises(ValueError, match='Base model differs'):
        runner.evaluate(args)
    assert not args.output.exists()


def test_freeze_rejects_stale_baseline_metrics(fixture):
    change_json(fixture.baseline / 'selected_baseline.json', development_macro_f1={p: 0.9 for p in PROMPTS})
    with pytest.raises(ValueError, match='selection differs'):
        runner.freeze(fixture)


def test_freeze_rejects_stale_training_metrics(fixture):
    change_json(fixture.training_runs[0] / 'training_run.json', best_development_macro_f1=0.9)
    with pytest.raises(ValueError, match='selection differs'):
        runner.freeze(fixture)


def test_freeze_rejects_adapter_changed_since_reload(fixture):
    (fixture.training_runs[0] / 'adapter/adapters.safetensors').write_bytes(b'not verified')
    with pytest.raises(ValueError, match='verified reload'):
        runner.freeze(fixture)


@pytest.mark.parametrize('field', ['model_provenance', 'generation', 'data'])
def test_freeze_rejects_baseline_manifest_disagreement(fixture, field):
    path = fixture.baseline / 'run.json'
    run = json.loads(path.read_text())
    if field == 'data':
        run['data']['dev']['sha256'] = 'different data'
    else:
        run[field] = {'different': True}
    write_json(path, run)
    with pytest.raises(ValueError, match='Baseline manifest differs'):
        runner.freeze(fixture)
    assert not fixture.output.exists()


def test_freeze_rejects_adapter_configuration_changed_before_freeze(fixture):
    path = fixture.training_runs[0] / 'adapter/adapter_config.json'
    config = json.loads(path.read_text())
    config['lora_parameters']['scale'] = 1
    write_json(path, config)
    with pytest.raises(ValueError, match='Adapter configuration differs'):
        runner.freeze(fixture)
    assert not fixture.output.exists()


@pytest.mark.parametrize('field', ['comparison_id', 'source_paper_id'])
def test_evaluate_rejects_train_test_overlap_before_model_load(fixture, monkeypatch, field):
    args = evaluation_args(fixture)
    train = load_split(args.train, 'train')
    test = dict(train[0], comparison_id='test-row', source_paper_id='test-paper')
    test[field] = train[0][field]
    monkeypatch.setattr(runner, 'load_split', lambda *a: [test])
    monkeypatch.setattr(runner, 'load_model', lambda *a: pytest.fail('model loaded despite overlap'))
    with pytest.raises(ValueError, match='overlaps'):
        runner.evaluate(args)
    assert not args.output.exists()
