"""Media decode dimension/rate bounds — untrusted-shape inputs must not wrap or hang.

`image.to_tensor` / `image.resize` take a target ``width``/``height`` and `audio.resample`
takes a target ``rate``. These are ``i64`` in the plan IR but ``u32`` in the decoder. A
plain ``as u32`` cast silently wraps an out-of-range value, which is two distinct bugs:

* a value **past ``u32::MAX``** wraps *down* to a small one — ``to_tensor(2**32 + 5, 5)``
  silently produced a 5x5 tensor instead of erroring, and ``resample(2**32)`` wrapped to a
  **0 Hz** target that sent the sinc resampler into an infinite iterator (a hang);
* a **negative** value wraps *up* to ~4 billion — an unbounded allocation that aborts the
  process.

Both must be rejected with a clear error. There is no DuckDB oracle for media decode, so
the property under test is "a nonsensical dimension is a clean error, never a wrong-size
result, an OOM, or a hang".
"""

from __future__ import annotations

import struct

import pytest

import batcher as bt
from batcher.plan.expr_ir.image import _PNG_1X1

pytestmark = pytest.mark.differential

_U32_MAX = 2**32 - 1


def _img():
    return bt.from_pydict({"img": [_PNG_1X1]})


def _wav(sample_rate: int, n: int) -> bytes:
    data = b"".join(struct.pack("<h", i % 1000) for i in range(n))
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
    hdr += b"data" + struct.pack("<I", len(data))
    return hdr + data


@pytest.mark.parametrize("bad", [-1, 0, _U32_MAX + 1, _U32_MAX + 5, 2**40])
def test_to_tensor_rejects_out_of_range_dimensions(bad):
    """A width/height that is negative or past u32::MAX must error, not wrap.

    Before the fix, ``bad = 2**32 + 5`` silently returned a 5-pixel-wide tensor and a
    negative width attempted a multi-gigabyte allocation.
    """
    with pytest.raises(Exception):  # noqa: B017 - engine surfaces a typed decode error
        _img().select(t=bt.col("img").image.to_tensor(bad, 4)).to_pydict()
    with pytest.raises(Exception):  # noqa: B017
        _img().select(t=bt.col("img").image.to_tensor(4, bad)).to_pydict()


@pytest.mark.parametrize("bad", [-1, 0, _U32_MAX + 1, 2**40])
def test_resize_rejects_out_of_range_dimensions(bad):
    with pytest.raises(Exception):  # noqa: B017
        _img().select(t=bt.col("img").image.resize(bad, 2)).to_pydict()


def test_to_tensor_does_not_silently_truncate_a_large_dimension():
    """``to_tensor(2**32 + 5, 5)`` must not produce the same output as ``to_tensor(5, 5)``."""
    with pytest.raises(Exception):  # noqa: B017
        _img().select(t=bt.col("img").image.to_tensor(_U32_MAX + 6, 5)).to_pydict()


@pytest.mark.parametrize("bad", [_U32_MAX + 1, 2**40])
def test_resample_rejects_a_rate_past_u32_max(bad):
    """A rate past u32::MAX wraps to 0 Hz under ``as u32`` and hangs the resampler."""
    ds = bt.from_pydict({"a": [_wav(8000, 100)]})
    with pytest.raises(Exception):  # noqa: B017
        ds.select(w=bt.col("a").audio.resample(bad)).to_pydict()


def test_resample_still_works_for_a_normal_rate():
    """Guard against over-tightening: an ordinary rate must still resample."""
    ds = bt.from_pydict({"a": [_wav(8000, 100)]})
    out = ds.select(w=bt.col("a").audio.resample(4000)).to_pydict()["w"]
    assert len(out[0]) == 50
