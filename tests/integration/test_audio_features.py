"""The audio level, shaping and spectral expressions, over real encoded clips.

These cover what a speech pipeline does between reading a clip's bytes and handing it to a
model. What is asserted is what a naive implementation gets wrong while still returning
plausible numbers:

- an RMS that a single loud sample can move, which is the peak wearing a different name;
- a dBFS that reports `-inf` for silence, so a "quieter than X" filter accepts it *and* a
  negated "louder than X" filter accepts it too;
- an in-place pre-emphasis filter running forwards, which feeds itself its own output;
- a `pad_or_trim` that pads but does not truncate, so the corpus is still unbatchable;
- spectral descriptors that average silent frames in as "0 Hz", which makes a
  mostly-quiet recording look band-limited — the exact confusion rolloff exists to resolve.

Clips are synthesized as 16-bit PCM WAV in the test, so nothing here needs a fixture file
or an optional decoder.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable

import pytest

import batcher as bt

pytestmark = pytest.mark.integration

_RATE = 16_000


def _wav(samples: list[float], rate: int = _RATE) -> bytes:
    """A mono 16-bit PCM WAV of `samples`, each in ``-1..1``."""
    pcm = b"".join(struct.pack("<h", round(max(-1.0, min(1.0, s)) * 32767)) for s in samples)
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(pcm))
    )
    return header + pcm


def _tone(hz: float, n: int = _RATE, amplitude: float = 0.5) -> bytes:
    return _wav([amplitude * math.sin(2 * math.pi * hz * i / _RATE) for i in range(n)])


def _generated(n: int, f: Callable[[int], float]) -> bytes:
    return _wav([f(i) for i in range(n)])


def _one(clip: bytes, **exprs) -> dict:
    return bt.from_pydict({"clip": [clip]}).select(**exprs).to_pydict()


# --- level and hygiene -------------------------------------------------------------


def test_rms_tracks_level_where_a_peak_does_not() -> None:
    """One door slam must not make a quiet recording read as a loud one."""
    click = _generated(4000, lambda i: 1.0 if i == 0 else 0.01)
    steady = _generated(4000, lambda _: 0.5)
    got = (
        bt.from_pydict({"clip": [click, steady]})
        .select(level=bt.col("clip").audio.rms(), peak=bt.col("clip").audio.peak_dbfs())
        .to_pydict()
    )
    assert got["level"][0] < 0.05, "a single click moved the RMS"
    assert abs(got["level"][1] - 0.5) < 0.01
    assert got["peak"][0] > -0.5, "the peak should still see the click"


def test_digital_silence_has_a_null_level_rather_than_negative_infinity() -> None:
    """An infinity passes every threshold, in both directions."""
    silent = _generated(2000, lambda _: 0.0)
    got = _one(
        silent,
        db=bt.col("clip").audio.dbfs(),
        peak=bt.col("clip").audio.peak_dbfs(),
        level=bt.col("clip").audio.rms(),
    )
    assert got["db"] == [None]
    assert got["peak"] == [None]
    assert got["level"] == [0.0], "RMS is a plain amplitude, so silence is honestly zero"


def test_dbfs_agrees_with_the_definition() -> None:
    """Half full scale is -6.02 dB; a level in the wrong unit is a silently wrong filter."""
    got = _one(_generated(2000, lambda _: 0.5), db=bt.col("clip").audio.dbfs())
    assert abs(got["db"][0] - (-6.0206)) < 0.02


def test_clipping_and_silence_ratios_count_the_samples_they_name() -> None:
    half = _generated(2000, lambda i: 1.0 if i % 2 == 0 else 0.0)
    got = _one(
        half,
        hot=bt.col("clip").audio.clipping_ratio(),
        quiet=bt.col("clip").audio.silence_ratio(),
    )
    assert abs(got["hot"][0] - 0.5) < 0.01
    assert abs(got["quiet"][0] - 0.5) < 0.01


# --- waveform shaping --------------------------------------------------------------


def test_rms_normalize_lifts_a_quiet_clip_without_clipping_it() -> None:
    """Reaching the target by brute force would clip, which is worse than staying quiet."""
    quiet = _generated(4000, lambda i: 0.02 * math.sin(i / 20))
    got = _one(
        quiet,
        level=bt.col("clip").audio.rms_normalize().list.max(),
        loud=bt.col("clip").audio.rms_normalize().list.min(),
    )
    assert got["level"][0] <= 1.0
    assert got["loud"][0] >= -1.0


def test_pre_emphasis_reads_the_original_previous_sample() -> None:
    """A forward in-place filter feeds itself its own output, and the steady state shows it."""
    step = _generated(6, lambda i: 0.0 if i == 0 else 1.0)
    out = _one(step, y=bt.col("clip").audio.pre_emphasis())["y"][0]
    assert abs(out[1] - 1.0) < 1e-3, f"the edge should survive: {out}"
    # x - 0.97x = 0.03x, which only holds if each step read the input.
    assert abs(out[2] - 0.03) < 1e-3, f"the filter fed itself: {out}"


def test_pad_or_trim_gives_every_clip_exactly_the_same_length() -> None:
    """The property that makes a clip corpus batchable at all."""
    short = _generated(100, lambda _: 0.5)
    long = _generated(_RATE, lambda _: 0.5)
    got = (
        bt.from_pydict({"clip": [short, long]})
        .select(n=bt.col("clip").audio.pad_or_trim(0.25, _RATE).list.len())
        .to_pydict()
    )
    assert got["n"] == [4000, 4000]


def test_pad_or_trim_pads_with_silence_at_the_end() -> None:
    """Padding at the front would shift every timestamp a downstream model reads."""
    short = _generated(10, lambda _: 0.5)
    out = _one(short, y=bt.col("clip").audio.pad_or_trim(0.001, _RATE))["y"][0]
    assert abs(out[0] - 0.5) < 1e-3
    assert out[-1] == 0.0


def test_slice_reads_the_window_and_one_past_the_end_is_empty_not_null() -> None:
    """An empty region is a fact about the window, not a failure to read the clip."""
    clip = _tone(440.0)
    got = _one(
        clip,
        inside=bt.col("clip").audio.slice(0.25, 0.5).list.len(),
        past=bt.col("clip").audio.slice(10.0, 1.0).list.len(),
    )
    assert got["inside"] == [8000]
    assert got["past"] == [0]


def test_encode_wav_round_trips_and_resamples() -> None:
    """The loop a cleaned corpus has to close to be written back out as audio."""
    clip = _tone(440.0, n=4000)
    got = _one(
        clip,
        same=bt.col("clip").audio.encode_wav().audio.decode(),
        halved=bt.col("clip").audio.encode_wav(8000).audio.decode(),
    )
    assert got["same"][0]["sample_rate"] == _RATE
    assert got["same"][0]["num_frames"] == 4000
    assert got["halved"][0]["sample_rate"] == 8000
    assert abs(got["halved"][0]["num_frames"] - 2000) <= 2


def test_the_waveform_ops_compose_without_re_encoding() -> None:
    """The gap that made this namespace non-composable.

    Every waveform method hands back a `List<Float32>`, and the level and shaping methods
    used to insist on a container — so `trim_silence()` produced a value nothing else here
    could read, and a two-step clean was three impossible expressions. The ops that need no
    sample rate now read a waveform directly, which also means the second step costs no
    second decode.
    """
    noisy = _wav([0.0] * 500 + [0.3 * math.sin(i / 12) for i in range(4000)] + [0.0] * 500)
    got = _one(
        noisy,
        level=bt.col("clip").audio.trim_silence().audio.rms(),
        chained=bt.col("clip").audio.trim_silence().audio.rms_normalize().list.len(),
        raw=bt.col("clip").audio.to_waveform().audio.zero_crossing_rate(),
    )
    assert got["level"][0] > 0.0
    # The interior survives and only the silent ends go; the sine starts at zero, so the
    # first sample is part of the leading quiet.
    assert got["chained"][0] == 3999
    assert 0.0 < got["raw"][0] < 1.0


def test_a_waveform_answers_the_same_measure_an_encoded_clip_does() -> None:
    """One implementation serves both shapes, so a level cannot mean two things.

    16-bit PCM quantization makes the two differ in the last few digits, which is a
    property of the container rather than of the measure.
    """
    clip = _tone(440.0, n=4000)
    got = _one(
        clip,
        encoded=bt.col("clip").audio.rms(),
        decoded=bt.col("clip").audio.to_waveform().audio.rms(),
    )
    assert abs(got["encoded"][0] - got["decoded"][0]) < 1e-6


def test_an_op_that_needs_a_sample_rate_says_so_by_name() -> None:
    """Resampling, slicing by time and the spectral front ends are defined against a rate.

    A waveform column carries none, and the refusal has to say which op and what to do —
    the alternative reads as a type error about a column the caller never typed.
    """
    clip = _tone(440.0, n=2000)
    with pytest.raises(Exception, match="encode_wav"):
        _one(clip, x=bt.col("clip").audio.to_waveform().audio.spectral_centroid(_RATE))


def test_a_cleaned_waveform_can_be_written_back_out_as_audio() -> None:
    """The loop that lets a cleaned corpus leave the engine as audio, not as floats."""
    noisy = _wav([0.0] * 500 + [0.3 * math.sin(i / 12) for i in range(4000)] + [0.0] * 500)
    cleaned = bt.col("clip").audio.trim_silence().audio.rms_normalize()
    got = _one(noisy, meta=cleaned.audio.encode_wav(_RATE).audio.decode())
    assert got["meta"][0]["sample_rate"] == _RATE
    assert got["meta"][0]["num_frames"] == 3999


def test_encoding_a_waveform_needs_the_rate_its_samples_are_at() -> None:
    """Guessing would make every clip play at the wrong speed."""
    clip = _tone(440.0, n=2000)
    with pytest.raises(Exception, match="rate"):
        _one(clip, x=bt.col("clip").audio.to_waveform().audio.encode_wav())


# --- spectral ----------------------------------------------------------------------


def test_the_centroid_lands_on_the_tone_that_is_actually_there() -> None:
    got = (
        bt.from_pydict({"clip": [_tone(1000.0), _tone(4000.0)]})
        .select(hz=bt.col("clip").audio.spectral_centroid(_RATE))
        .to_pydict()
    )
    assert abs(got["hz"][0] - 1000.0) < 150.0
    assert got["hz"][1] > got["hz"][0]


def test_rolloff_finds_the_band_edge_a_level_measure_cannot_see() -> None:
    """The band-limited-then-upsampled recording is the one this exists to catch."""
    got = (
        bt.from_pydict({"clip": [_tone(500.0), _tone(6000.0)]})
        .select(
            edge=bt.col("clip").audio.spectral_rolloff(_RATE),
            level=bt.col("clip").audio.rms(),
        )
        .to_pydict()
    )
    assert got["edge"][0] < 2000.0
    assert got["edge"][1] > 4000.0
    # Both are equally loud, which is why no level measure separates them.
    assert abs(got["level"][0] - got["level"][1]) < 0.02


def test_flatness_separates_a_tone_from_broadband_noise() -> None:
    pseudo_noise = _generated(_RATE, lambda i: ((i * 1103515245 + 12345) % 2000) / 1000.0 - 1.0)
    got = (
        bt.from_pydict({"clip": [_tone(1000.0), pseudo_noise]})
        .select(f=bt.col("clip").audio.spectral_flatness(_RATE))
        .to_pydict()
    )
    assert got["f"][0] < 0.05
    assert got["f"][1] > got["f"][0] * 5


def test_silent_frames_do_not_drag_the_centroid_toward_dc() -> None:
    """Counting a silent frame as 0 Hz makes a mostly-quiet recording look band-limited."""
    tone_only = _tone(3000.0)
    half_silent = _wav(
        [0.5 * math.sin(2 * math.pi * 3000.0 * i / _RATE) for i in range(_RATE)] + [0.0] * _RATE
    )
    got = (
        bt.from_pydict({"clip": [tone_only, half_silent]})
        .select(hz=bt.col("clip").audio.spectral_centroid(_RATE))
        .to_pydict()
    )
    assert abs(got["hz"][0] - got["hz"][1]) < got["hz"][0] * 0.2


def test_the_linear_spectrogram_has_the_shape_its_framing_implies() -> None:
    clip = _tone(1000.0, n=_RATE)
    got = _one(clip, n=bt.col("clip").audio.spectrogram(_RATE, n_fft=64, hop_length=32).list.len())
    # centre padding gives 16000 + 64 samples, so 1 + (16064 - 64) / 32 = 501 frames.
    assert got["n"] == [33 * 501]


def test_a_bad_framing_argument_is_refused_at_plan_build() -> None:
    """Refused where it names the caller's method, not by the engine after the scan ran."""
    with pytest.raises(Exception, match="n_fft"):
        bt.col("clip").audio.spectrogram(_RATE, n_fft=0)
    with pytest.raises(Exception, match="rate"):
        bt.col("clip").audio.spectral_centroid(0)
    with pytest.raises(Exception, match="duration_secs"):
        bt.col("clip").audio.pad_or_trim(0.0, _RATE)


def test_every_new_operation_answers_null_for_a_null_or_undecodable_row() -> None:
    """The namespace's convention: a corrupt clip is a null, never a failed batch."""
    ds = bt.from_pydict({"clip": [None, b"not audio at all"]})
    got = ds.select(
        level=bt.col("clip").audio.rms(),
        shaped=bt.col("clip").audio.pre_emphasis(),
        fixed=bt.col("clip").audio.pad_or_trim(1.0, _RATE),
        wav=bt.col("clip").audio.encode_wav(),
        hz=bt.col("clip").audio.spectral_centroid(_RATE),
    ).to_pydict()
    for name, values in got.items():
        assert values == [None, None], f"{name} did not null an unreadable row"


def test_streaming_an_audio_pipeline_matches_collecting_it() -> None:
    """Same question for audio: a batch-scoped parameter is where the executors diverge.

    The framing arguments and the FFT plan are resolved once per batch, so a kernel that
    accidentally carried state across rows would show up here and nowhere else.
    """
    corpus = [_tone(200.0 + 40 * k, n=2000) for k in range(20)]
    ds = bt.from_pydict({"clip": corpus})
    query = ds.select(
        level=bt.col("clip").audio.rms(),
        hz=bt.col("clip").audio.spectral_centroid(_RATE),
        fixed=bt.col("clip").audio.pad_or_trim(0.1, _RATE).list.len(),
        cleaned=bt.col("clip").audio.trim_silence().audio.rms_normalize().list.len(),
    )
    collected = query.collect().to_pydict()
    streamed: dict[str, list] = {name: [] for name in collected}
    for batch in query.iter_batches(batch_size=4):
        for name, values in batch.to_pydict().items():
            streamed[name] += values
    assert streamed == collected
    # The whole point of `pad_or_trim`: every row the same width, whatever went in.
    assert len(set(collected["fixed"])) == 1
