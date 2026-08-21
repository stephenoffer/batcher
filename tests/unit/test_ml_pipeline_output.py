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


def test_speech_recognition_reduces_to_the_transcript():
    # An ASR pipeline returns {"text": ..., "chunks": [...]}. Without `text` in the key list
    # the whole dict fell through as the "scalar", so a transcription landed as a struct
    # column carrying the timing chunks beside the words.
    result = {"text": " the quick brown fox", "chunks": [{"timestamp": (0.0, 1.0)}]}
    assert _primary_output(result) == " the quick brown fox"


def test_a_label_still_wins_over_text_when_a_pipeline_reports_both():
    assert _primary_output({"label": "POSITIVE", "text": "raw input echoed back"}) == "POSITIVE"


class TestPipelineInputs:
    """What a batch column becomes on the way into a `transformers` pipeline.

    A pipeline is polymorphic in its input, so the *column type* is what decides the shape.
    """

    @staticmethod
    def _inputs(array):
        from batcher.ml.inference.pipelines import _pipeline_inputs

        return _pipeline_inputs(array)

    def test_a_text_column_passes_through_as_strings(self):
        import pyarrow as pa

        assert self._inputs(pa.array(["a", "b"])) == ["a", "b"]

    def test_a_binary_column_passes_through_as_bytes(self):
        # Encoded audio/image bytes: the pipeline decodes them itself.
        import pyarrow as pa

        assert self._inputs(pa.array([b"\x00\x01"], type=pa.binary())) == [b"\x00\x01"]

    def test_a_waveform_column_becomes_one_numpy_array_per_row(self):
        # The case that was broken: `to_pylist()` hands a feature extractor a Python list of
        # floats per row, which it rejects — so an ASR model over the documented decode path
        # could not run through ds.ml.infer at all.
        import numpy as np
        import pyarrow as pa

        rows = self._inputs(pa.array([[0.1, 0.2], [0.3, 0.4]], type=pa.list_(pa.float32())))
        assert len(rows) == 2
        assert isinstance(rows[0], np.ndarray)
        assert rows[0].dtype == np.dtype("float32")

    def test_a_tensor_column_keeps_its_shape(self):
        import numpy as np
        import pyarrow as pa

        from batcher.io.formats.ml.tensor import to_tensor_column

        column = to_tensor_column(np.zeros((2, 4, 4, 3), dtype="uint8"))
        rows = self._inputs(pa.chunked_array([column]).combine_chunks())
        assert rows[0].shape == (4, 4, 3)

    def test_a_column_with_no_array_form_falls_back_to_python_objects(self):
        import pyarrow as pa

        struct = pa.array([{"a": 1}], type=pa.struct([("a", pa.int64())]))
        assert self._inputs(struct) == [{"a": 1}]
