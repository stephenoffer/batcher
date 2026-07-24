"""Multi-channel audio decode error tolerance — parity with `ds.ml.download`'s on_error.

A corrupt/truncated audio row used to raise straight out of the decode UDF and kill the
whole batch; `on_error="null"` now nulls it and continues. Uses soundfile when present.
"""

from __future__ import annotations

import pytest

from batcher._internal.errors import PlanError
from batcher.ml.decode.media import _decode_audio_bytes, audio_dataset


def test_on_error_is_validated():
    import batcher as bt

    ds = bt.from_pydict({"bytes": [b""]})
    with pytest.raises(PlanError):
        audio_dataset(ds, mono=False, on_error="maybe")


def test_corrupt_row_nulls_under_on_error_null():
    pytest.importorskip("soundfile")
    assert _decode_audio_bytes(b"not-audio", None, mono=False, on_error="null") is None


def test_corrupt_row_raises_by_default():
    pytest.importorskip("soundfile")
    with pytest.raises(Exception):  # noqa: B017 - libsndfile raises its own error type
        _decode_audio_bytes(b"not-audio", None, mono=False, on_error="raise")


def test_null_input_is_null_regardless():
    assert _decode_audio_bytes(None, None, mono=False, on_error="raise") is None
