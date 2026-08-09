"""Video decode failure signal — a corrupt clip is no longer an indistinguishable black one.

`error_column` marks rows whose bytes were present but would not decode (as opposed to a
null-input row or a legitimately black clip). Monkeypatches the PyAV decoder, so no PyAV.

Every test here pins the **Python fallback**, because that decoder is what is being
driven. `video_dataset` picks its decoder from the engine's compiled features, so on a
`video`-enabled build the monkeypatched function would never be reached and these tests
would silently assert nothing about it.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

import batcher as bt
import batcher.ml.decode.video as video


def _fake_decode(data, num_frames, height, width, seek=False):
    if data == b"bad":
        return None  # undecodable
    return np.ones((num_frames, height, width, 3), dtype=np.uint8)


def _python_path(monkeypatch):
    """Force the PyAV fallback whatever the engine was built with."""
    monkeypatch.setattr(video, "engine_features", frozenset)
    monkeypatch.setattr(video, "_decode_video_bytes", _fake_decode)


def test_error_column_flags_only_decode_failures(monkeypatch):
    _python_path(monkeypatch)
    ds = bt.from_arrow(pa.table({"bytes": pa.array([b"good", b"bad", None], type=pa.binary())}))
    out = video.video_dataset(
        ds, size=(2, 2), num_frames=1, error_column="decode_failed"
    ).to_pydict()
    # good -> False, bad (present but undecodable) -> True, null input -> not a failure
    assert out["decode_failed"] == [False, True, False]


def test_error_column_is_absent_by_default(monkeypatch):
    _python_path(monkeypatch)
    ds = bt.from_arrow(pa.table({"bytes": pa.array([b"good"], type=pa.binary())}))
    out = video.video_dataset(ds, size=(2, 2), num_frames=1).to_pydict()
    assert "decode_failed" not in out


def test_an_undecodable_clip_is_null_not_a_black_frame(monkeypatch):
    """The frames column itself must distinguish "failed" from "black".

    `error_column` is opt-in, so if the frames of a failed row were zeros, a caller who
    did not ask for the extra column would have a training set with silent black samples
    in it and no way to find them. Nulling is also what the native kernel does, and the
    two decoders must not disagree about what the same call means.
    """
    _python_path(monkeypatch)
    ds = bt.from_arrow(pa.table({"bytes": pa.array([b"good", b"bad", None], type=pa.binary())}))
    frames = video.video_dataset(ds, size=(2, 2), num_frames=1).to_pydict()["frames"]
    assert frames[0] is not None
    assert frames[1] is None
    assert frames[2] is None
