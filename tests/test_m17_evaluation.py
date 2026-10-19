import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def evaluation(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / 'evaluation' / 'm17' / 'common.py'
    monkeypatch.setenv('M17_OUTPUT', str(tmp_path))
    spec = importlib.util.spec_from_file_location('m17_common', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paper_splits_and_cross_role_exclusions(evaluation):
    expected = [(24, 6, []), (19, 10, ['M14-11']), (21, 8, ['M14-01']), (24, 6, [])]
    held_out = []
    for group, (train_size, test_size, excluded) in zip(evaluation.GROUPS, expected):
        train, test, dropped = evaluation.split(group)
        assert (len(train), len(test)) == (train_size, test_size)
        assert [r['comparison_id'] for r in dropped] == excluded
        assert all(r['source_pmcid'] != group for r in train)
        assert all(r['source_pmcid'] == group for r in test)
        held_out.extend(r['comparison_id'] for r in test)
    assert len(held_out) == len(set(held_out)) == 30


def test_reference_metrics(evaluation):
    records = json.loads((evaluation.PACKAGE / 'reference_predictions.json').read_text())
    expected = {'majority': (19, 0.2585034013605442), 'qwen_zero': (8, 0.226),
                'qwen_reason': (19, 0.2585034013605442), 'qwen_544': (20, 0.41702534672405545),
                'qwen_2026': (20, 0.49673202614379086),
                'distilbert_544': (17, 0.2463768115942029),
                'distilbert_2026': (17, 0.2463768115942029)}
    gold = {r['comparison_id']: r['label'] for r in evaluation.ROWS}
    for name, rows in records.items():
        assert len(rows) == len({r['comparison_id'] for r in rows}) == 30
        assert all(r['gold'] == gold[r['comparison_id']] for r in rows)
        result = evaluation.metrics(rows)
        correct, f1 = expected[name]
        assert result['accuracy'] == pytest.approx(correct / 30)
        assert result['macro_f1'] == pytest.approx(f1, abs=0.0005)


def test_invalid_prediction_counts_as_error(evaluation):
    result = evaluation.metrics([
        {'gold': 'overlapping', 'predicted': 'overlapping'},
        {'gold': 'overlapping', 'predicted': 'invalid'},
    ])
    assert result['accuracy'] == 0.5
    assert result['per_class']['overlapping']['recall'] == 0.5
    assert result['invalid_predictions'] == 1


def test_prompt_excludes_human_label(evaluation):
    row = dict(evaluation.ROWS[0])
    prompt = evaluation.messages(row)
    row['label'] = 'insufficient_evidence'
    assert evaluation.messages(row) == prompt
