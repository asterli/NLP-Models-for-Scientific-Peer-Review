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


@pytest.fixture
def training_model():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.array(1.0)

    return Model()


def test_accumulation_includes_partial_final_group(training_model, monkeypatch):
    import random
    import mlx.optimizers as optim
    import finetune_comparison_mlx as training

    monkeypatch.setattr(training, "loss_fn", lambda model, inputs, labels: model.weight * inputs.mean() * 0.1)
    optimizer = optim.SGD(learning_rate=0.1)
    examples = [{"input_ids": [value], "labels": [value]} for value in (1, 2, 3)]
    training.train_epoch(training_model, examples, optimizer, accumulation=2, rng=random.Random(0))
    # Seed zero orders examples 1, 3, 2: two mean gradients of 0.2 each.
    assert training_model.weight.item() == pytest.approx(0.96, abs=1e-6)
    assert optimizer.state["step"].item() == 2


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_nonfinite_loss_stops_before_update(training_model, monkeypatch, nonfinite):
    import random
    import mlx.optimizers as optim
    import finetune_comparison_mlx as training

    monkeypatch.setattr(training, "loss_fn", lambda model, inputs, labels: model.weight * inputs.mean())
    optimizer = optim.SGD(learning_rate=0.1)
    examples = [{"input_ids": [value], "labels": [1]} for value in (1.0, nonfinite)]
    with pytest.raises(RuntimeError, match="Non-finite training loss"):
        training.train_epoch(training_model, examples, optimizer, accumulation=2, rng=random.Random(0))
    assert training_model.weight.item() == 1.0
    assert optimizer.state["step"].item() == 0


def test_nonfinite_gradient_stops_before_update(training_model, monkeypatch):
    import random
    import mlx.core as mx
    import mlx.optimizers as optim
    import finetune_comparison_mlx as training

    training_model.weight = mx.array(0.0)
    monkeypatch.setattr(training, "loss_fn", lambda model, inputs, labels: mx.sqrt(model.weight))
    optimizer = optim.SGD(learning_rate=0.1)
    with pytest.raises(RuntimeError, match="Non-finite gradient norm"):
        training.train_epoch(training_model, [{"input_ids": [1], "labels": [1]}],
                             optimizer, accumulation=1, rng=random.Random(0))
    assert training_model.weight.item() == 0.0
    assert optimizer.state["step"].item() == 0
