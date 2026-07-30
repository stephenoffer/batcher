"""Silence trimming, peak normalization, and the zero-crossing rate, over real audio.

These decode in the data plane, so the only honest test builds actual WAV bytes and checks
what comes back. The properties pinned are the ones a speech pipeline depends on: trimming
removes the ends and not the middle, normalization changes the level without changing the
shape, and the crossing rate tracks the frequency it is supposed to measure.
"""

from __future__ import annotations

import io
import itertools
import math
import struct
import wave

import pytest

import batcher as bt

pytestmark = pytest.mark.integration

_RATE = 8000


def _wav(samples: list[float], rate: int = _RATE) -> bytes:
    """16-bit mono WAV bytes for a float signal in ``[-1, 1]``."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(
            b"".join(struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in samples)
        )
    return buf.getvalue()


def _tone(hz: float, n: int, amplitude: float = 0.4) -> list[float]:
    return [amplitude * math.sin(2 * math.pi * hz * i / _RATE) for i in range(n)]


def _select(clips: list[bytes | None], **exprs) -> dict:
    return bt.from_pydict({"b": clips}).select(**exprs).to_pydict()


# --- trim_silence ------------------------------------------------------------------


def test_trimming_removes_the_silent_ends():
    padded = _wav([0.0] * 800 + _tone(220, 1600) + [0.0] * 800)
    got = _select(
        [padded],
        raw=bt.col("b").audio.to_waveform().list.len(),
        trimmed=bt.col("b").audio.trim_silence().list.len(),
    )
    assert got["raw"] == [3200]
    assert 1500 < got["trimmed"][0] < 1700  # the tone, without its padding


def test_trimming_keeps_an_interior_pause():
    """Interior silence carries timing an acoustic model reads; only the ends go."""
    gapped = _wav(_tone(220, 800) + [0.0] * 800 + _tone(220, 800))
    got = _select([gapped], n=bt.col("b").audio.trim_silence().list.len())
    assert got["n"][0] > 2000  # both tones plus the gap between them


def test_a_silent_clip_trims_to_an_empty_list():
    """The shape that makes a silent-recording filter expressible."""
    silent = _wav([0.0] * 800)
    got = _select([silent], n=bt.col("b").audio.trim_silence().list.len())
    assert got["n"] == [0]


def test_a_silent_recording_can_be_filtered_out():
    clips = [_wav(_tone(220, 800)), _wav([0.0] * 800)]
    kept = (
        bt.from_pydict({"b": clips, "id": [1, 2]})
        .filter(bt.col("b").audio.trim_silence().list.len() > bt.lit(0))
        .to_pydict()
    )
    assert kept["id"] == [1]


def test_a_louder_threshold_trims_more():
    clip = _wav([0.02] * 400 + _tone(220, 800) + [0.02] * 400)
    got = _select(
        [clip],
        lenient=bt.col("b").audio.trim_silence(threshold_db=-60).list.len(),
        strict=bt.col("b").audio.trim_silence(threshold_db=-20).list.len(),
    )
    assert got["strict"][0] < got["lenient"][0]


def test_a_positive_threshold_is_rejected():
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError):
        bt.col("b").audio.trim_silence(threshold_db=6)


# --- peak_normalize ----------------------------------------------------------------


def test_normalization_brings_the_loudest_sample_to_full_scale():
    quiet = _wav(_tone(220, 800, amplitude=0.05))
    got = _select(
        [quiet],
        before=bt.col("b").audio.to_waveform().list.max_abs(),
        after=bt.col("b").audio.peak_normalize().list.max_abs(),
    )
    assert got["before"][0] < 0.06
    assert got["after"][0] == pytest.approx(1.0, abs=1e-4)


def test_normalization_preserves_the_length_and_the_shape():
    clip = _wav(_tone(220, 800, amplitude=0.2))
    got = _select(
        [clip],
        raw_len=bt.col("b").audio.to_waveform().list.len(),
        norm_len=bt.col("b").audio.peak_normalize().list.len(),
        raw_zcr=bt.col("b").audio.zero_crossing_rate(),
    )
    assert got["raw_len"] == got["norm_len"]
    # Scaling every sample by a positive gain cannot change where the signal crosses zero.
    normalized = (
        bt.from_pydict({"b": [clip]})
        .select(z=bt.col("b").audio.peak_normalize())
        .to_pydict()["z"][0]
    )
    pairs = itertools.pairwise(normalized)
    crossings = sum(1 for a, b in pairs if (a < 0) != (b < 0))
    assert crossings / (len(normalized) - 1) == pytest.approx(got["raw_zcr"][0], abs=1e-9)


def test_a_silent_clip_is_returned_unchanged_rather_than_amplified():
    silent = _wav([0.0] * 400)
    got = _select([silent], peak=bt.col("b").audio.peak_normalize().list.max_abs())
    assert got["peak"] == [0.0]


# --- zero_crossing_rate ------------------------------------------------------------


def test_the_crossing_rate_tracks_the_frequency():
    """A sine at f crosses zero 2f times a second, so the rate is 2f/sample_rate."""
    got = _select(
        [_wav(_tone(220, 4000)), _wav(_tone(880, 4000))],
        z=bt.col("b").audio.zero_crossing_rate(),
    )
    assert got["z"][0] == pytest.approx(2 * 220 / _RATE, abs=0.01)
    assert got["z"][1] == pytest.approx(2 * 880 / _RATE, abs=0.01)


def test_silence_never_crosses_zero():
    got = _select([_wav([0.0] * 800)], z=bt.col("b").audio.zero_crossing_rate())
    assert got["z"] == [0.0]


def test_the_crossing_rate_separates_a_tone_from_noise():
    """The voiced/unvoiced split this is normally used for."""
    tone = _wav(_tone(220, 4000))
    # Alternating samples are the highest-frequency signal representable, standing in for
    # broadband noise without needing a random source.
    noise = _wav([0.3 if i % 2 == 0 else -0.3 for i in range(4000)])
    got = _select([tone, noise], z=bt.col("b").audio.zero_crossing_rate())
    assert got["z"][0] < 0.1
    assert got["z"][1] > 0.9


# --- the shared null contract ------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda e: e.audio.trim_silence(),
        lambda e: e.audio.peak_normalize(),
        lambda e: e.audio.zero_crossing_rate(),
    ],
)
def test_null_and_undecodable_input_yields_null_rather_than_failing_the_batch(build):
    clips = [_wav(_tone(220, 400)), None, b"this is not audio at all"]
    got = bt.from_pydict({"b": clips}).select(v=build(bt.col("b"))).to_pydict()["v"]
    assert got[0] is not None
    assert got[1] is None
    assert got[2] is None
