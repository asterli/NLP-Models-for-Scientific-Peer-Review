import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import BertTokenizerFast

from novelty import encode_pairs, load_comparisons


@pytest.fixture
def tokenizer(tmp_path):
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhousing\nhelps\n")
    return BertTokenizerFast(vocab_file=str(vocab))


def test_pairs_preserve_both_texts_and_reject_truncation(tokenizer):
    inputs = encode_pairs(tokenizer, ["housing"], ["helps"], 5)
    assert tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]) == [
        "[CLS]", "housing", "[SEP]", "helps", "[SEP]"
    ]
    with pytest.raises(ValueError, match="exceeds"):
        encode_pairs(tokenizer, ["housing housing"], ["helps"], 5)


@pytest.mark.parametrize("claims,passages", [([], []), (["housing"], []), ([""], ["helps"])])
def test_invalid_input_pairs(tokenizer, claims, passages):
    with pytest.raises(ValueError):
        encode_pairs(tokenizer, claims, passages, 512)


@pytest.mark.parametrize("change", [{"label": "unknown"}, {"claim": " "}, {"prior_work": None}])
def test_bad_training_rows(tmp_path, change):
    row = {"comparison_id": "sample", "claim": "housing", "prior_work": "helps", "label": "distinct"}
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(row | change) + "\n")
    with pytest.raises(ValueError):
        load_comparisons(path)


def test_duplicate_comparison_ids(tmp_path):
    row = {"comparison_id": "sample", "claim": "housing", "prior_work": "helps", "label": "distinct"}
    path = tmp_path / "data.jsonl"
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        load_comparisons(path)


@pytest.mark.parametrize("row", [None, [], "comparison", 1])
def test_non_object_rows(tmp_path, row):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="JSON object"):
        load_comparisons(path)
