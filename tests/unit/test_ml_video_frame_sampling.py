"""Video frame sampling decodes a clip without retaining it.

The previous implementation built `[f.to_ndarray() for f in container.decode()]` — every
frame of the clip, as separate RGB arrays — and then sampled 8 of them. A 1-minute 1080p
clip at 30fps is ~1,800 x 6.2 MB ≈ 11 GB resident, to produce 8 frames.

The rewrite must be *indistinguishable in output*, so the test that matters is equality
against the retaining implementation, kept here verbatim as the oracle.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

av = pytest.importorskip("av")
Image = pytest.importorskip("PIL.Image")

from batcher.ml.decode import _decode_video_bytes  # noqa: E402

pytestmark = pytest.mark.unit


def _clip(n_frames: int, w: int = 64, h: int = 48, rate: int = 10) -> bytes:
    """A synthetic clip whose frames are distinguishable from one another."""
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        for i in range(n_frames):
            arr = np.full((h, w, 3), (i * 5) % 256, dtype=np.uint8)
            arr[:, :, 1] = (i * 3) % 256
            for packet in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buf.getvalue()


def _retaining_oracle(data: bytes, num_frames: int, height: int, width: int):
    """The original implementation — correct, and unusable on a real clip."""
    with av.open(io.BytesIO(data)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    if not frames:
        return None
    idx = np.linspace(0, len(frames) - 1, num=num_frames).astype(int)
    out = np.empty((num_frames, height, width, 3), dtype=np.uint8)
    for j, k in enumerate(idx):
        out[j] = np.asarray(Image.fromarray(frames[k]).resize((width, height)))
    return out


@pytest.mark.parametrize("num_frames", [1, 2, 4, 8, 16])
def test_matches_the_retaining_implementation(num_frames: int) -> None:
    data = _clip(50)
    assert np.array_equal(
        _retaining_oracle(data, num_frames, 24, 32),
        _decode_video_bytes(data, num_frames, 24, 32),
    )


def test_sampling_more_frames_than_the_clip_has() -> None:
    """`num_frames > total` repeats frames; it must repeat the same ones as before."""
    data = _clip(5)
    assert np.array_equal(
        _retaining_oracle(data, 12, 24, 32), _decode_video_bytes(data, 12, 24, 32)
    )


def test_single_frame_clip() -> None:
    data = _clip(1)
    out = _decode_video_bytes(data, 4, 24, 32)
    assert out.shape == (4, 24, 32, 3)
    # Every sampled index collapses onto the only frame there is.
    assert all(np.array_equal(out[0], out[i]) for i in range(4))


def test_peak_memory_does_not_grow_with_clip_length() -> None:
    """The whole point: a clip 6x longer must not cost 6x the memory.

    Measured as decoded-frame retention rather than RSS, which is too noisy to assert on:
    the oracle holds `total` frames, so its retention grows with the clip and the
    implementation's must not.
    """
    import tracemalloc

    def peak_bytes(fn, data) -> int:
        tracemalloc.start()
        fn(data, 4, 24, 32)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    short, long = _clip(20), _clip(120)
    grew = peak_bytes(_decode_video_bytes, long) - peak_bytes(_decode_video_bytes, short)
    oracle_grew = peak_bytes(_retaining_oracle, long) - peak_bytes(_retaining_oracle, short)

    assert oracle_grew > 0, "oracle should grow with clip length (guards the test itself)"
    assert grew < oracle_grew / 4, f"grew {grew} vs oracle {oracle_grew}"
