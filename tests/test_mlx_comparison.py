import json

import pytest

from mlx_comparison import MLX_MODEL_ID, encode_prompt, model_manifest
from comparison_baselines import PROMPTS


class Tokenizer:
    def apply_chat_template(self, conversation, **kwargs):
        self.conversation = conversation
        assert kwargs == {"tokenize": True, "add_generation_prompt": True}
        return [1, 2, 3]


def test_mlx_prompts_preserve_passages_and_validate_inputs():
    tokenizer = Tokenizer()
    assert encode_prompt(tokenizer, "claim", "evidence", "explanation") == [1, 2, 3]
    assert tokenizer.conversation[0]["content"] == PROMPTS["explanation"]
    assert tokenizer.conversation[1]["content"] == "New claim:\nclaim\n\nOlder passage:\nevidence"
    with pytest.raises(ValueError, match="nonempty"):
        encode_prompt(tokenizer, "claim", " ")
    with pytest.raises(ValueError, match="token limit"):
        encode_prompt(tokenizer, "claim", "evidence", max_input_tokens=2)
    with pytest.raises(ValueError, match="Unknown prompt"):
        encode_prompt(tokenizer, "claim", "evidence", "unregistered")


def test_model_manifest_pins_source_and_detects_weight_changes(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen2", "quantization": {"bits": 4, "group_size": 64}}))
    weights = model / "model.safetensors"
    weights.write_bytes(b"first")
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"repo": MLX_MODEL_ID, "revision": "a" * 40}))
    first = model_manifest(model, source)
    weights.write_bytes(b"second")
    second = model_manifest(model, source)
    assert first["file_sha256"]["model.safetensors"] != second["file_sha256"]["model.safetensors"]
    source.write_text(json.dumps({"repo": MLX_MODEL_ID, "revision": "main"}))
    with pytest.raises(ValueError, match="pinned"):
        model_manifest(model, source)
