import pytest

from qwen_comparison import MODEL_ID, prompt_tokens, snapshot_revision, training_example


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, conversation, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True}
        assert len(conversation) == 2
        return [1, 2, 3]

    def encode(self, text, add_special_tokens):
        assert text == "distinct" and not add_special_tokens
        return [10, 11]


def test_training_masks_prompt_and_includes_complete_label_and_end():
    row = {"claim": "new claim", "prior_work": "old passage", "label": "distinct"}
    example = training_example(Tokenizer(), row, 3)
    assert example["input_ids"] == [1, 2, 3, 10, 11, 99]
    assert example["labels"] == [-100, -100, -100, 10, 11, 99]


def test_inputs_are_not_silently_truncated():
    with pytest.raises(ValueError, match="token limit"):
        prompt_tokens(Tokenizer(), "claim", "passage", 2)
    with pytest.raises(ValueError, match="nonempty"):
        prompt_tokens(Tokenizer(), "claim", " ", 3)


def test_snapshot_identity_and_revision(tmp_path):
    snapshot = tmp_path / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / ("a" * 40)
    assert snapshot_revision(snapshot) == "a" * 40
    with pytest.raises(ValueError):
        snapshot_revision(tmp_path / "models--other" / "snapshots" / ("a" * 40))
