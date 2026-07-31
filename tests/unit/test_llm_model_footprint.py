"""Reading a model's shape and footprint before the engine downloads it.

Every warning about a tensor-parallel degree needs two numbers, and on the real path neither
was available, so the branch that matters most — "this degree cannot hold this model" — could
never fire. The failure it exists to prevent then arrived as an allocation error, tens of
gigabytes and many minutes after the point where it was already knowable.

Both numbers come from metadata rather than weights, so the parsing is what has to be right.
The tests below pin the two ways it silently goes wrong: reading the attention-head count
where a model publishes a grouped key/value count, which overstates the cache by the GQA ratio
(a factor of eight on a modern model), and summing a repository that publishes its weights in
two formats, which doubles the footprint.
"""

from __future__ import annotations

import pytest

from batcher.ml.llm.engines.footprint import (
    ModelShape,
    declared_context,
    model_weight_bytes,
    shape_from_config,
    weight_bytes_from_files,
)
from batcher.ml.llm.sizing import sized_window

pytestmark = pytest.mark.unit


def test_a_grouped_query_model_reports_its_grouped_head_count() -> None:
    shape = shape_from_config(
        {
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "hidden_size": 8192,
            "max_position_embeddings": 131072,
        }
    )
    assert shape == ModelShape(
        layers=80,
        attention_heads=64,
        kv_heads=8,
        head_dim=128,
        hidden_size=8192,
        max_context=131072,
    )


def test_multi_head_attention_carries_a_kv_head_per_attention_head() -> None:
    # No grouped count published means every head holds its own key and value. Defaulting to
    # anything smaller would understate the cache by the ratio it invented.
    shape = shape_from_config(
        {"num_hidden_layers": 32, "num_attention_heads": 32, "hidden_size": 4096}
    )
    assert shape.kv_heads == 32


def test_an_explicit_head_dim_wins_over_the_derived_one() -> None:
    # Several recent models publish a head_dim that is not hidden_size / heads.
    shape = shape_from_config(
        {
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "hidden_size": 4096,
            "head_dim": 256,
        }
    )
    assert shape.head_dim == 256


@pytest.mark.parametrize(
    "config",
    [
        {"n_layer": 32, "n_head": 32, "n_embd": 4096},
        {"num_layers": 32, "num_heads": 32, "d_model": 4096},
    ],
)
def test_the_older_spellings_are_read_too(config: dict) -> None:
    shape = shape_from_config(config)
    assert shape is not None
    assert (shape.layers, shape.attention_heads, shape.head_dim) == (32, 32, 128)


def test_a_vision_language_model_is_read_from_its_language_tower() -> None:
    # The top level carries the vision encoder's much smaller numbers; the tower below it is
    # what holds a KV cache.
    shape = shape_from_config(
        {
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "text_config": {
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
            },
        }
    )
    assert shape.layers == 32
    assert shape.kv_heads == 8


def test_something_that_is_not_a_transformer_is_not_guessed_at() -> None:
    assert shape_from_config({}) is None
    assert shape_from_config({"hidden_size": 4096}) is None
    assert shape_from_config({"num_hidden_layers": 32}) is None
    assert shape_from_config("not a config") is None  # type: ignore[arg-type]


def test_a_boolean_is_not_a_layer_count() -> None:
    # `bool` is an `int` in Python, and a config carrying `use_cache: True` next to a missing
    # layer count would otherwise read as a one-layer model.
    assert shape_from_config({"num_hidden_layers": True, "num_attention_heads": 32}) is None


def test_a_repository_publishing_two_formats_is_not_counted_twice() -> None:
    # The classic overstatement: the same weights as safetensors and as a pickle. Summing both
    # advises a group twice the size the model needs.
    assert (
        weight_bytes_from_files(
            {
                "model-00001-of-00002.safetensors": 8,
                "model-00002-of-00002.safetensors": 8,
                "pytorch_model-00001-of-00002.bin": 8,
                "pytorch_model-00002-of-00002.bin": 8,
                "tokenizer.json": 4,
                "config.json": 1,
            }
        )
        == 16
    )


def test_a_repository_with_no_weights_reports_nothing() -> None:
    assert weight_bytes_from_files({"README.md": 100, "config.json": 2}) == 0
    assert weight_bytes_from_files({}) == 0


def test_a_zero_sized_entry_is_ignored() -> None:
    # The hub reports `size=None` for a file whose metadata was not requested; it must not
    # count as a weight file of unknown size.
    assert weight_bytes_from_files({"model.safetensors": 0, "other.bin": 5}) == 5


def test_a_local_directory_is_measured_without_a_network(tmp_path) -> None:
    (tmp_path / "model-00001.safetensors").write_bytes(b"x" * 1024)
    (tmp_path / "model-00002.safetensors").write_bytes(b"x" * 512)
    (tmp_path / "config.json").write_text("{}")
    assert model_weight_bytes(str(tmp_path)) == 1536


def test_an_empty_local_directory_reports_unknown_rather_than_zero(tmp_path) -> None:
    # Zero would read as "this model has no weights", which every fit check would clear.
    (tmp_path / "config.json").write_text("{}")
    assert model_weight_bytes(str(tmp_path)) is None


def test_a_local_config_is_read_as_json_without_executing_the_model_s_code(tmp_path) -> None:
    """A model that ships its own configuration class needs `trust_remote_code=True` to load
    through `transformers`, which runs code from the repository — far too much to spend on
    reading six integers. Reading the JSON covers those families and executes nothing."""
    import json

    from batcher.ml.llm.engines.footprint import model_shape

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "some_custom_arch",
                "auto_map": {"AutoConfig": "configuration_custom.CustomConfig"},
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "hidden_size": 4096,
            }
        )
    )
    shape = model_shape(str(tmp_path))
    assert (shape.layers, shape.kv_heads) == (32, 8)


def test_an_unreadable_config_is_not_guessed_at(tmp_path) -> None:
    from batcher.ml.llm.engines.footprint import model_shape

    (tmp_path / "config.json").write_text("{not json")
    assert model_shape(str(tmp_path)) is None


def test_the_declared_window_is_read_from_the_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        "batcher.ml.llm.engines.footprint.model_shape",
        lambda model: ModelShape(32, 32, 8, 128, 4096, max_context=8192),
    )
    assert declared_context("some-org/some-model") == 8192


def test_a_model_declaring_no_window_is_not_capped(monkeypatch) -> None:
    # A cap invented from a guess is worse than no cap: it truncates prompts the model could
    # have held.
    monkeypatch.setattr(
        "batcher.ml.llm.engines.footprint.model_shape",
        lambda model: ModelShape(32, 32, 8, 128, 4096, max_context=0),
    )
    assert declared_context("some-org/some-model") is None
    monkeypatch.setattr("batcher.ml.llm.engines.footprint.model_shape", lambda model: None)
    assert declared_context("some-org/some-model") is None


def test_a_window_above_the_declared_maximum_is_not_proposed() -> None:
    # Without the cap this proposes a window vLLM refuses, and the refusal is only discovered
    # by building the engine — weights loaded, allocation made, minutes spent, then a second
    # build to fall back.
    long_prompts = ["x" * 40_000]
    assert sized_window(long_prompts, {"max_tokens": 256}) == 30_720
    assert sized_window(long_prompts, {"max_tokens": 256}, 8192) is None
    assert sized_window(long_prompts, {"max_tokens": 256}, 131_072) == 30_720


def test_an_unreachable_model_leaves_the_advice_silent(monkeypatch) -> None:
    # No hub client, no network, a private repository: all of them mean "not measured", and
    # the only safe reading of that is to say nothing.
    monkeypatch.setattr("batcher.ml.llm.engines.footprint._local_weight_bytes", lambda model: None)
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", None)
    assert model_weight_bytes("some-org/some-model") is None
