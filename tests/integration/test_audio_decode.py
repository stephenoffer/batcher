"""Native audio decode (`.audio` accessor) — moves WAV/FLAC decode off the per-row
Python `map_batches` path into the Rust data plane (symphonia).

No DuckDB oracle for audio; we hand-encode a minimal PCM WAV and assert the decoded
metadata and waveform.
"""

from __future__ import annotations

import struct

import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.integration


def _wav(sample_rate: int, samples: list[int]) -> bytes:
    """A minimal mono 16-bit PCM WAV."""
    data = b"".join(struct.pack("<h", s) for s in samples)
    fmt = struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVE"
        + b"fmt "
        + fmt
        + b"data"
        + struct.pack("<I", len(data))
    )
    return header + data


def test_audio_decode_metadata():
    ds = bt.from_pydict({"a": [_wav(16000, [0, 100, -100, 0, 50]), None]})
    out = ds.select(d=col("a").audio.decode()).collect().to_pydict()["d"]
    assert out[0] == {
        "sample_rate": 16000,
        "channels": 1,
        "num_frames": 5,
        "duration_secs": 5 / 16000,
    }
    assert out[1] is None  # null bytes → null struct


def test_audio_to_waveform():
    # 16384/32768 = 0.5 normalized; -16384/32768 = -0.5.
    ds = bt.from_pydict({"a": [_wav(8000, [0, 16384, -16384]), b"not audio"]})
    out = ds.select(w=col("a").audio.to_waveform()).collect().to_pydict()["w"]
    assert [round(x, 3) for x in out[0]] == [0.0, 0.5, -0.5]
    assert out[1] is None  # undecodable bytes → null list


def test_audio_dataset_helper_takes_native_path():
    # The default (mono, native rate) `audio_dataset` must decode in the data plane
    # — no per-row Python `map_batches` — and match the native expression's output.
    from batcher.core.udf import has_map_batches
    from batcher.ml.decode import audio_dataset

    ds = bt.from_pydict({"bytes": [_wav(8000, [0, 16384, -16384]), None]})
    decoded = audio_dataset(ds)
    assert not has_map_batches(decoded._plan), "default audio_dataset must not use map_batches"
    out = decoded.collect().to_pydict()["waveform"]
    assert [round(x, 3) for x in out[0]] == [0.0, 0.5, -0.5]
    assert out[1] is None


def test_audio_resample_expression():
    # `.audio.resample` decodes + sinc-resamples natively; output length is the librosa
    # length, ceil(n * target / source). 400 frames at 8 kHz -> 800 at 16 kHz (upsample),
    # -> 200 at 4 kHz (downsample). Undecodable input -> null.
    samples = [int(16384 * (i % 7 - 3) / 3) for i in range(400)]
    ds = bt.from_pydict({"a": [_wav(8000, samples), b"not audio"]})
    up = ds.select(w=col("a").audio.resample(16000)).collect().to_pydict()["w"]
    assert len(up[0]) == 800
    assert up[1] is None
    down = ds.select(w=col("a").audio.resample(4000)).collect().to_pydict()["w"]
    assert len(down[0]) == 200
    # Resampling to the source rate is an exact passthrough.
    same = ds.select(w=col("a").audio.resample(8000)).collect().to_pydict()["w"]
    assert len(same[0]) == 400


def test_audio_dataset_resample_takes_native_path():
    # An explicit mono sample_rate now resamples natively (Rust sinc) — no per-row Python.
    from batcher.core.udf import has_map_batches
    from batcher.ml.decode import audio_dataset

    ds = bt.from_pydict({"bytes": [_wav(8000, [int(16384 * (i % 5 - 2) / 2) for i in range(200)])]})
    decoded = audio_dataset(ds, sample_rate=16000)
    assert not has_map_batches(decoded._plan), "mono resample must take the native path"
    out = decoded.collect().to_pydict()["waveform"]
    assert len(out[0]) == 400  # 200 frames at 8 kHz -> 400 at 16 kHz
