"""A corrupt image or audio blob must measure as null, never as a plausible number.

Every `.image` and `.audio` metric takes an encoded blob and returns a measurement. The blobs a real
pipeline meets are not all images: a crawl truncates downloads, an object store returns an
error page as the body, and a column of images has nulls where the fetch failed. The
dangerous failure is not a crash -- a crash stops the job and gets fixed. It is
`brightness` returning `0.0` for an error page, because that is a plausible reading of a
dark image, and it flows into a filter threshold or a training set as data.

All fourteen image metrics and all seven audio metrics return null for null, for empty
bytes, and for bytes that are not media.

The real PNG is what makes that mean something. Without it "null for corrupt" is
indistinguishable from "null for everything", which a metric that simply never worked would
also satisfy. The image is seven by three pixels of RGB(10, 20, 30), so the expected
`aspect_ratio` and `mean_color` are known from its construction rather than read off the
implementation.

The audio fixture is the same idea: a 52-byte 8-bit mono WAV of eight known samples at
8000 Hz, so `decode` must report that rate and `to_waveform` must return those samples
scaled to [-1, 1].

Both are embedded as base64 rather than generated with Pillow or `wave`, so the file needs
no media library and the bytes under test cannot drift with a library upgrade.
"""

from __future__ import annotations

import base64
import inspect

import pytest

import batcher as bt

pytestmark = pytest.mark.unit

#: 7x3 RGB PNG, every pixel (10, 20, 30). 76 bytes.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAcAAAADCAIAAADQoYKSAAAAE0lEQVR4nGPkEpFjwABMmEI4RQEP9ABClMpXmAAAAABJRU5ErkJggg=="
)

#: A real image, then the three shapes of "not an image" a pipeline actually receives.
BLOBS = [PNG, None, b"", b"not-an-image-at-all"]
_REAL, _NULL, _EMPTY, _GARBAGE = range(4)


def _metrics(namespace: str = "image") -> list[str]:
    accessor = getattr(bt.col("b"), namespace)
    names = []
    for name in sorted(dir(accessor)):
        if name.startswith("_"):
            continue
        function = getattr(accessor, name, None)
        if not callable(function):
            continue
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            continue
        if [p for p in signature.parameters if not p.startswith("_")]:
            continue
        names.append(name)
    return names


#: 52-byte 8-bit mono WAV, 8000 Hz, samples [128, 200, 128, 60, 128, 200, 128, 60].
WAV = base64.b64decode("UklGRiwAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQgAAACAyIA8gMiAPA==")

AUDIO_BLOBS = [WAV, None, b"", b"not-audio-at-all"]

METRICS = _metrics()
AUDIO_METRICS = _metrics("audio")


def _evaluate(name: str) -> list:
    column = getattr(bt.col("b").image, name)()
    return bt.from_pydict({"b": BLOBS}).select(r=column).to_pydict()["r"]


def _evaluate_audio(name: str) -> list:
    column = getattr(bt.col("b").audio, name)()
    return bt.from_pydict({"b": AUDIO_BLOBS}).select(r=column).to_pydict()["r"]


def test_the_sweep_found_the_metrics():
    assert len(METRICS) >= 10, f"only {len(METRICS)} zero-argument .image metrics found"


@pytest.mark.parametrize("name", METRICS)
def test_a_corrupt_blob_measures_as_null(name):
    """Null, empty, and bytes that are not an image. A number here is a fabricated
    measurement that a threshold cannot tell from a real one."""
    result = _evaluate(name)
    for index, label in ((_NULL, "null"), (_EMPTY, "empty bytes"), (_GARBAGE, "garbage bytes")):
        assert result[index] is None, f"{name} returned {result[index]!r} for {label}"


@pytest.mark.parametrize("name", METRICS)
def test_the_same_metric_reads_a_real_image(name):
    """The control, per metric rather than once for the file. A metric that returned null
    for everything would satisfy the test above completely."""
    assert _evaluate(name)[_REAL] is not None, f"{name} could not read a valid 76-byte PNG"


class TestTheKnownImage:
    """Values fixed by how the fixture was constructed, not read off the implementation."""

    def test_the_aspect_ratio_is_seven_over_three(self):
        assert _evaluate("aspect_ratio")[_REAL] == pytest.approx(7 / 3)

    def test_the_mean_colour_is_the_fill_colour(self):
        mean = _evaluate("mean_color")[_REAL]
        assert mean["r"] == pytest.approx(10.0)
        assert mean["g"] == pytest.approx(20.0)
        assert mean["b"] == pytest.approx(30.0)

    def test_the_format_and_alpha_are_read_from_the_container(self):
        assert _evaluate("format")[_REAL] == "png"
        assert _evaluate("has_alpha")[_REAL] is False

    def test_decode_reports_the_constructed_size(self):
        decoded = _evaluate("decode")[_REAL]
        assert (decoded["width"], decoded["height"]) == (7, 3)


@pytest.mark.parametrize("name", AUDIO_METRICS)
def test_a_corrupt_audio_blob_measures_as_null(name):
    """The same property for `.audio`. A crawl truncates an mp3 exactly as it truncates a
    jpeg, and `rms` returning a number for the truncated half is the same failure."""
    result = _evaluate_audio(name)
    for index, label in ((_NULL, "null"), (_EMPTY, "empty bytes"), (_GARBAGE, "garbage bytes")):
        assert result[index] is None, f"audio.{name} returned {result[index]!r} for {label}"


@pytest.mark.parametrize("name", AUDIO_METRICS)
def test_the_same_audio_metric_reads_a_real_wav(name):
    assert _evaluate_audio(name)[_REAL] is not None, f"audio.{name} could not read a valid WAV"


def test_the_audio_sweep_found_the_metrics():
    assert len(AUDIO_METRICS) >= 5, f"only {len(AUDIO_METRICS)} zero-argument .audio metrics"


class TestTheKnownWaveform:
    """Values fixed by the samples the fixture was built from."""

    def test_decode_reports_the_constructed_sample_rate(self):
        assert _evaluate_audio("decode")[_REAL]["sample_rate"] == 8000

    def test_the_waveform_is_the_samples_scaled_to_unit_range(self):
        # 8-bit PCM is unsigned with 128 as silence: 128 -> 0.0, 200 -> +0.5625.
        waveform = _evaluate_audio("to_waveform")[_REAL]
        assert waveform[0] == pytest.approx(0.0)
        assert waveform[1] == pytest.approx(0.5625)
