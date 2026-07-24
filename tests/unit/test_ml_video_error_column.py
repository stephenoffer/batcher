"""Video decode failure signal — a corrupt clip is no longer an indistinguishable black one.

`error_column` marks rows whose bytes were present but would not decode (as opposed to a
null-input row or a legitimately black clip). Monkeypatches the PyAV decoder, so no PyAV.
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


def test_error_column_flags_only_decode_failures(monkeypatch):
    monkeypatch.setattr(video, "_decode_video_bytes", _fake_decode)
    ds = bt.from_arrow(
        pa.table({"bytes": pa.array([b"good", b"bad", None], type=pa.binary())})
    )
    out = video.video_dataset(
        ds, size=(2, 2), num_frames=1, error_column="decode_failed"
    ).to_pydict()
    # good -> False, bad (present but undecodable) -> True, null input -> not a failure
    assert out["decode_failed"] == [False, True, False]


def test_error_column_is_absent_by_default(monkeypatch):
    monkeypatch.setattr(video, "_decode_video_bytes", _fake_decode)
    ds = bt.from_arrow(pa.table({"bytes": pa.array([b"good"], type=pa.binary())}))
    out = video.video_dataset(ds, size=(2, 2), num_frames=1).to_pydict()
    assert "decode_failed" not in out
