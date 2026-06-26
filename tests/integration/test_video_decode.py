"""Native video decode (`.video` accessor) — FFmpeg-backed, behind the optional
`video` cargo feature.

The engine decodes from a path, so each clip's bytes are written to a short-lived
temp file and probed. The default engine build does *not* enable `video`, so the
accessor degrades gracefully with a clear error; this test asserts whichever path
the running engine took (so CI passes either way).
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.integration


def _sample_video(tmp_path) -> bytes:
    """A 1s 320x240 10fps test clip via the system ffmpeg (skips if unavailable)."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg binary not available to generate a fixture")
    out = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=320x240:rate=10",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out.read_bytes()


def test_video_decode_metadata_or_graceful_error(tmp_path):
    video = _sample_video(tmp_path)
    ds = bt.from_pydict({"v": [video, None]})
    try:
        out = ds.select(d=col("v").video.decode()).collect().to_pydict()["d"]
    except RuntimeError as e:
        # Default build (no `video` feature) → clear, actionable error.
        assert "video" in str(e).lower() and "feature" in str(e).lower()
        return
    # Feature-enabled build → real metadata from the 320x240 10fps test source.
    assert out[0] == {
        "width": 320,
        "height": 240,
        "num_frames": 10,
        "duration_secs": 1.0,
        "fps": 10.0,
    }
    assert out[1] is None  # null bytes → null struct


def test_video_dataset_decodes_per_row_without_whole_batch_materialization():
    # The Python `video_dataset` path decodes one clip at a time (no `.to_pylist()`).
    # Null rows skip the PyAV decode and become all-zero frames, so this exercises the
    # per-row iteration + fixed-shape tensor output without needing PyAV installed.
    import numpy as np

    from batcher.ml.decode import video_dataset

    ds = bt.from_pydict({"bytes": [None, None]})
    out = video_dataset(ds, size=(8, 8), num_frames=2).collect()
    frames = out.column("frames")
    assert out.num_rows == 2
    # Fixed-shape tensor: each row is (num_frames, H, W, 3) = (2, 8, 8, 3) of zeros.
    arr = np.asarray(frames.to_pylist())
    assert arr.shape == (2, 2 * 8 * 8 * 3)
    assert not arr.any()  # undecodable/null clips → all-zero frames
