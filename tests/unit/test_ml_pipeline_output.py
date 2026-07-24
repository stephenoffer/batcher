"""`transformers` pipeline result reduction — one salient value per row, all of them.

The bug: a token-classification / NER pipeline returns a *list* of entities per row, and
the old `result[0]` kept only the first. These assert every entity survives while the
single-value pipelines (classification, generation) still reduce to a scalar. Pure
function over fabricated pipeline results — no transformers, no GPU.
"""

from __future__ import annotations

from batcher.ml.inference.pipelines import _primary_output


def test_classification_reduces_to_the_label():
    assert _primary_output({"label": "POSITIVE", "score": 0.99}) == "POSITIVE"


def test_generation_unwraps_the_single_element_list():
    assert _primary_output([{"generated_text": "hello world"}]) == "hello world"


def test_ner_keeps_every_entity_not_just_the_first():
    ner = [{"entity": "PER", "word": "Ada"}, {"entity": "LOC", "word": "Paris"}]
    assert _primary_output(ner) == ner


def test_multi_label_classification_keeps_each_label():
    labels = [{"label": "a", "score": 0.6}, {"label": "b", "score": 0.4}]
    assert _primary_output(labels) == ["a", "b"]


def test_scalar_and_empty_pass_through():
    assert _primary_output(0.5) == 0.5
    assert _primary_output([]) is None


def test_converters_warn_when_dropping_non_numeric_columns():
    """to_torch_iterable/to_tf_dataset warned nothing when a string label was silently
    dropped; now they warn once (matching the loader)."""
    import warnings

    import pyarrow as pa
    import pytest

    pytest.importorskip("torch")
    from batcher.ml import to_torch_iterable

    batch = pa.record_batch({"x": [1, 2], "label": ["a", "b"]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        list(to_torch_iterable([batch]))
    assert any("non-numeric" in str(w.message) and "label" in str(w.message) for w in caught)
