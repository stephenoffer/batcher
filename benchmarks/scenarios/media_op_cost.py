"""What each `.image` / `.audio` operation costs per row, and the ranking that follows.

This is the reproducer for the media entries in
``python/batcher/kyber/expr_cost/weights.py``. That file's opening paragraph says its
numbers are measured rather than guessed, and lists the orderings a guessed table gets
wrong; the media families are the largest block in it, so they need a script anyone can
re-run rather than a number nobody can check.

It answers one question: how long does a single media expression take per row, with the
cost of reading the column subtracted. Nothing here compares engines, and nothing here is
a throughput number. The output is a ranking, and the ranking is what the optimizer
consumes: `filter_split` orders the conjuncts of a predicate by exactly these costs
(Krishnamurthy-Boral-Zaniolo rank), so a header probe that is cheaper than a full decode
has to be *known* to be cheaper or it will not be run first.

Two results are worth re-checking whenever the kernels change, because both are
counter-intuitive enough that a hand-written table gets them backwards:

* `to_tensor_f32` costs several times `to_tensor`. The float conversion and per-channel
  normalization cost more than the decode-and-resize they follow.
* `is_grayscale` costs a full decode, while `has_alpha` beside it costs a header read.
  Alpha is a field in the container header; grayscale-in-fact is a question about pixels.

The corpus is synthesized, so no network, cloud store or fixture file is needed. Sizes
match the reference inputs the weights table names: a 512x512 JPEG and a 3-second 16 kHz
mono WAV. Media cost scales with resolution and duration far more strongly than a string
function's does with string length, so changing `--size` or `--seconds` moves every number
and only the *ranking* carries over. Image *content* matters as much as size and not
uniformly: a decode-bound op such as `dhash` runs several times faster on a compressible
frame than on incompressible noise, while a convolution such as `blur` barely moves,
because its work is in the filter rather than in the decode.

Run:
    python benchmarks/scenarios/media_op_cost.py
    python benchmarks/scenarios/media_op_cost.py --size 1024 --runs 5
    python benchmarks/scenarios/media_op_cost.py --family audio
"""

from __future__ import annotations

import argparse
import io
import math
import struct
import time

# One weights-table unit is ~0.2 ns/row, fixed by an entry already in that file:
# `regexp_matches` is 48.0 units and measures 9.5 ns/row net of a bare projection. The
# `units` column below applies that directly, so a drift shows up as a number that no
# longer matches `weights.py` rather than as a ranking someone has to eyeball.
_UNITS_PER_US = 48.0 / 0.00954

# Row counts an op climbs through until one timed pass is long enough to trust. Capped so
# the corpus stays holdable: 8,192 512x512 JPEGs is already several hundred megabytes, and
# the cheapest ops are within ~30% of their converged value there.
_SIZES = (64, 256, 1024, 4096, 8192)


def _jpeg(size: int, seed: int) -> bytes:
    """A structured (compressible) JPEG, the shape a real frame has.

    Random noise is a pathological worst case for a JPEG decoder and would overstate
    every decode here, so the frame is a gradient with a shape drawn over it.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    ramp = np.linspace(0, 255, size, dtype=np.uint8)
    plane = np.tile(ramp, (size, 1))
    img = Image.fromarray(np.dstack([plane, plane[::-1], np.roll(plane, seed % size)]))
    ImageDraw.Draw(img).ellipse(
        (size // 4, size // 4, 3 * size // 4, 3 * size // 4), fill=(20, 180, 90)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _wav(seconds: float, rate: int) -> bytes:
    """A 16-bit mono PCM WAV of a 440 Hz tone -- smaller than a fixture and unambiguously
    licensed, and every audio kernel here decodes it the same way it would a real clip."""
    n = int(seconds * rate)
    pcm = b"".join(
        struct.pack("<h", int(16000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(n)
    )
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(pcm))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )


def _image_ops(size: int) -> dict:
    """Every `.image` op the engine has, with arguments that keep the shapes comparable."""
    half = max(4, size // 2)
    return {
        "format": lambda c: c.image.format(),
        "aspect_ratio": lambda c: c.image.aspect_ratio(),
        "has_alpha": lambda c: c.image.has_alpha(),
        "exif_orientation": lambda c: c.image.exif_orientation(),
        "decode": lambda c: c.image.decode(),
        "ahash": lambda c: c.image.ahash(),
        "dhash": lambda c: c.image.dhash(),
        "phash": lambda c: c.image.phash(),
        "to_tensor": lambda c: c.image.to_tensor(224, 224),
        "to_grayscale": lambda c: c.image.to_grayscale(224, 224),
        "center_crop": lambda c: c.image.center_crop(224, 224),
        "resize": lambda c: c.image.resize(224, 224),
        "thumbnail": lambda c: c.image.thumbnail(224),
        "letterbox": lambda c: c.image.letterbox(224, 224),
        "is_grayscale": lambda c: c.image.is_grayscale(),
        "brightness": lambda c: c.image.brightness(),
        "colorfulness": lambda c: c.image.colorfulness(),
        "entropy": lambda c: c.image.entropy(),
        "mean_color": lambda c: c.image.mean_color(),
        "sharpness": lambda c: c.image.sharpness(),
        "encode(jpeg)": lambda c: c.image.encode("jpeg"),
        "encode(png)": lambda c: c.image.encode("png"),
        "to_tensor_f32": lambda c: c.image.to_tensor_f32(224, 224),
        "convert": lambda c: c.image.convert("RGB"),
        "auto_orient": lambda c: c.image.auto_orient(),
        "pad": lambda c: c.image.pad(size + half, size + half),
        "rotate": lambda c: c.image.rotate(90),
        "flip_horizontal": lambda c: c.image.flip_horizontal(),
        "flip_vertical": lambda c: c.image.flip_vertical(),
        "invert": lambda c: c.image.invert(),
        "posterize": lambda c: c.image.posterize(4),
        "solarize": lambda c: c.image.solarize(128),
        "equalize": lambda c: c.image.equalize(),
        "autocontrast": lambda c: c.image.autocontrast(),
        "adjust_brightness": lambda c: c.image.adjust_brightness(1.1),
        "adjust_contrast": lambda c: c.image.adjust_contrast(1.1),
        "adjust_saturation": lambda c: c.image.adjust_saturation(1.1),
        "adjust_hue": lambda c: c.image.adjust_hue(0.1),
        "blur": lambda c: c.image.blur(1.0),
        "sharpen": lambda c: c.image.sharpen(1.0),
    }


def _audio_ops(rate: int) -> dict:
    """Every `.audio` op the engine has, at the clip's own sample rate where one is asked for."""
    return {
        "decode": lambda c: c.audio.decode(),
        "dbfs": lambda c: c.audio.dbfs(),
        "rms": lambda c: c.audio.rms(),
        "peak_dbfs": lambda c: c.audio.peak_dbfs(),
        "clipping_ratio": lambda c: c.audio.clipping_ratio(),
        "silence_ratio": lambda c: c.audio.silence_ratio(),
        "zero_crossing_rate": lambda c: c.audio.zero_crossing_rate(),
        "encode_wav": lambda c: c.audio.encode_wav(),
        "slice": lambda c: c.audio.slice(0.5, 1.0),
        "spectral_rolloff": lambda c: c.audio.spectral_rolloff(rate),
        "spectral_centroid": lambda c: c.audio.spectral_centroid(rate),
        "spectral_flatness": lambda c: c.audio.spectral_flatness(rate),
        "spectral_bandwidth": lambda c: c.audio.spectral_bandwidth(rate),
        "pad_or_trim": lambda c: c.audio.pad_or_trim(2.0, rate),
        "to_waveform": lambda c: c.audio.to_waveform(),
        "trim_silence": lambda c: c.audio.trim_silence(),
        "peak_normalize": lambda c: c.audio.peak_normalize(),
        "pre_emphasis": lambda c: c.audio.pre_emphasis(),
        "rms_normalize": lambda c: c.audio.rms_normalize(),
        "spectrogram": lambda c: c.audio.spectrogram(rate),
        "mel_spectrogram": lambda c: c.audio.mel_spectrogram(rate),
        "mfcc": lambda c: c.audio.mfcc(rate),
        "resample": lambda c: c.audio.resample(rate // 2),
    }


def _sized(corpus: list, n: int, cache: dict) -> object:
    """A dataset of `n` rows drawn from `corpus`, built once per size and reused.

    Building it per op -- let alone per timing attempt -- dominated the measurement: a
    corpus of thousands of 512x512 JPEGs is hundreds of megabytes to assemble, and every
    op was assembling its own.
    """
    import batcher as bt

    if n not in cache:
        cache[n] = bt.from_pydict({"k": [corpus[i % len(corpus)] for i in range(n)]})
    return cache[n]


def _per_row_us(corpus: list, build, runs: int, budget_s: float, cache: dict) -> float:
    """Best-of-`runs` microseconds per row for one expression, less a bare projection.

    Subtracting the bare projection is what makes this the *function's own* work rather
    than the function plus the cost of reading a wide binary column, and it is the same
    correction the rest of the weights table was measured with.

    The row count is chosen per op rather than fixed, because no single one is right for
    both ends of this range. At 200 rows the cheap header probes are dominated by per-call
    fixed cost and read several times high -- `image.format` measures 7.5 us/row at 200
    rows against 0.45 at 25,600 -- while at 25,600 rows one convolution op takes minutes.
    So each op climbs through `_SIZES` until a timed pass clears `budget_s`, or until the
    corpus would be too large to hold, whichever comes first.
    """
    import batcher as bt

    for n in _SIZES:
        dataset = _sized(corpus, n, cache)

        def timed(make, d=dataset) -> float:
            make(d).collect()  # warm: first call pays plan build and one-time setup
            best = float("inf")
            for _ in range(runs):
                started = time.perf_counter()
                make(d).collect()
                best = min(best, time.perf_counter() - started)
            return best

        full = timed(lambda d: d.select(x=build(bt.col("k"))))
        if full >= budget_s or n == _SIZES[-1]:
            bare = timed(lambda d: d.select(x=bt.col("k")))
            return max(0.0, full - bare) / n * 1e6
    raise AssertionError("unreachable: the loop returns on its last size")


def _report(title: str, timings: dict[str, float]) -> None:
    """Print the ranking, and the weights-table units each timing implies."""
    scale = _UNITS_PER_US
    print(f"\n{title}  (1 unit = ~0.2 ns/row, per `regexp_matches` = 48.0)")
    print(f"  {'op':<22} {'us/row':>10} {'units':>9} {'x cheapest':>11}")
    print(f"  {'-' * 55}")
    cheapest = min(t for t in timings.values() if t > 0) or 1.0
    for name, us in sorted(timings.items(), key=lambda kv: kv[1]):
        print(f"  {name:<22} {us:>10.1f} {us * scale:>9.0f} {us / cheapest:>10.1f}x")


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-row cost of each media expression")
    parser.add_argument(
        "--budget",
        type=float,
        default=0.15,
        help="seconds one timed pass must reach before a per-row figure is trusted",
    )
    parser.add_argument("--runs", type=int, default=3, help="best-of-N timed repeats")
    parser.add_argument("--size", type=int, default=512, help="source JPEG edge, in pixels")
    parser.add_argument("--seconds", type=float, default=3.0, help="clip length, in seconds")
    parser.add_argument("--rate", type=int, default=16000, help="clip sample rate, in Hz")
    parser.add_argument(
        "--family",
        choices=("image", "audio", "both"),
        default="both",
        help="which expression family to measure",
    )
    args = parser.parse_args()

    import batcher as bt

    print(f"engine: {bt.versions().get('engine_profile', 'unknown')}")

    if args.family in ("image", "both"):
        corpus = [_jpeg(args.size, i) for i in range(16)]
        cache: dict = {}
        timings = {}
        for name, build in _image_ops(args.size).items():
            try:
                timings[name] = _per_row_us(corpus, build, args.runs, args.budget, cache)
            except Exception as exc:  # an op this engine build lacks must not abort the rest
                print(f"  (image.{name} skipped: {type(exc).__name__}: {str(exc)[:60]})")
        _report(f".image on a {args.size}x{args.size} JPEG", timings)

    if args.family in ("audio", "both"):
        timings = {}
        cache = {}
        clips = [_wav(args.seconds, args.rate)]
        for name, build in _audio_ops(args.rate).items():
            try:
                timings[name] = _per_row_us(clips, build, args.runs, args.budget, cache)
            except Exception as exc:
                print(f"  (audio.{name} skipped: {type(exc).__name__}: {str(exc)[:60]})")
        _report(f".audio on a {args.seconds:g}s {args.rate} Hz mono WAV", timings)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
