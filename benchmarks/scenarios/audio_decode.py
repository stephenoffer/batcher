"""Audio-decode benchmark: Batcher's native data-plane decode vs a Python baseline.

Audio ML preprocessing (Whisper, wav2vec, CLAP) begins by decoding a corpus of clips to
PCM waveforms. Batcher does this in the Rust data plane — ``col("bytes").audio.decode()``
(symphonia), fanned across every core — where the Python-ecosystem alternative is a
per-clip ``soundfile.read`` loop under the GIL, exactly what a pandas/Ray-``map_batches``
user writes. This measures both on the same in-memory corpus of encoded WAV bytes (one
shared column, byte-identical to every path) and, per the harness discipline, verifies the
two agree on the decoded frame count before trusting any timing.

The interesting regime is a corpus *smaller than one 16,384-row morsel* — a few thousand
clips — where naive scheduling would decode the whole batch on a single core. Batcher's
byte-aware worker sizing splits the byte-heavy column into many morsels and its media
kernels fan out per row, so the decode uses the whole machine.

Run:
    python benchmarks/scenarios/audio_decode.py                 # 2,000 clips
    python benchmarks/scenarios/audio_decode.py --clips 8000
"""

from __future__ import annotations

import argparse
import io
import time

import numpy as np
import pyarrow as pa


def _wav_bytes(sample_rate: int, num_frames: int, freq: float) -> bytes:
    """Encode one mono 16-bit PCM WAV of a pure tone — deterministic for a given input."""
    import soundfile as sf

    t = np.arange(num_frames)
    signal = (0.5 * np.sin(2 * np.pi * freq * t / sample_rate)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, signal, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _corpus(clips: int, sample_rate: int = 44_100) -> pa.Table:
    """A shared table of ``clips`` encoded WAV blobs (one ``bytes`` column)."""
    rng = np.random.default_rng(0)
    blobs = [
        _wav_bytes(sample_rate, sample_rate, int(rng.integers(200, 2000))) for _ in range(clips)
    ]
    return pa.table({"bytes": pa.array(blobs, type=pa.binary())})


def _batcher_frames(table: pa.Table) -> int:
    """Batcher: native parallel decode → total decoded frame count."""
    import batcher as bt
    from batcher import col

    out = (
        bt.from_arrow(table)
        .select(meta=col("bytes").audio.decode())
        .select(nf=col("meta").struct.field("num_frames"))
        .agg(total=col("nf").sum())
        .collect()
    )
    return out.column("total")[0].as_py()


def _python_frames(table: pa.Table) -> int:
    """Python baseline: per-clip ``soundfile.read`` under the GIL → total frame count."""
    import soundfile as sf

    total = 0
    for blob in table.column("bytes").to_pylist():
        samples, _ = sf.read(io.BytesIO(blob), dtype="float32")
        total += len(samples)
    return total


_TARGET_SR = 16000  # the Whisper / wav2vec input rate — the canonical resample target


def _batcher_resample(table: pa.Table) -> int:
    """Batcher: native decode + sinc resample to 16 kHz → total resampled frame count."""
    import batcher as bt
    from batcher import col

    out = (
        bt.from_arrow(table)
        .select(w=col("bytes").audio.resample(_TARGET_SR))
        .select(n=col("w").list.len())
        .agg(total=col("n").sum())
        .collect()
    )
    return out.column("total")[0].as_py()


def _python_resample(table: pa.Table) -> int:
    """Python baseline: per-clip ``soundfile.read`` + ``soxr`` resample → total frames.

    ``soxr`` is the high-quality SoX resampler ``librosa`` itself calls by default — the
    honest per-clip Python alternative to the native sinc kernel.
    """
    import soundfile as sf
    import soxr

    total = 0
    for blob in table.column("bytes").to_pylist():
        samples, sr = sf.read(io.BytesIO(blob), dtype="float32")
        total += len(soxr.resample(samples, sr, _TARGET_SR))
    return total


def _best_ms(fn, table: pa.Table, runs: int) -> tuple[float, int]:
    """Best-of-``runs`` wall-clock in ms plus the (checked-identical) result."""
    best = float("inf")
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn(table)
        best = min(best, time.perf_counter() - t0)
    return best * 1000, result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audio-decode benchmark (Batcher vs Python)")
    parser.add_argument("--clips", type=int, default=2000, help="number of audio clips")
    parser.add_argument("--runs", type=int, default=4, help="best-of-N timed repeats")
    args = parser.parse_args()

    try:
        import soundfile  # noqa: F401
    except ImportError:
        print("audio_decode benchmark needs soundfile: pip install 'batcher-engine[audio]'")
        return 0

    print(f"building {args.clips} WAV clips ...", flush=True)
    table = _corpus(args.clips)
    mib = sum(len(b.as_py()) for b in table.column("bytes")) / (1 << 20)

    bt_ms, bt_frames = _best_ms(_batcher_frames, table, args.runs)
    py_ms, py_frames = _best_ms(_python_frames, table, args.runs)

    # Correctness gate: both paths must decode the same number of frames.
    if bt_frames != py_frames:
        print(f"MISMATCH: batcher decoded {bt_frames} frames, python {py_frames}")
        return 1

    speedup = py_ms / bt_ms if bt_ms else float("inf")
    print(f"\ncorpus: {args.clips} clips, {mib:.0f} MiB, {bt_frames:,} frames")
    print(f"(best-of-{args.runs})\n")
    print(f"  {'engine':<26} {'ms':>10} {'Mframes/s':>12}")
    print(f"  {'-' * 50}")
    print(f"  {'batcher (native, parallel)':<26} {bt_ms:>10.1f} {bt_frames / bt_ms / 1e3:>12.1f}")
    print(f"  {'python (soundfile loop)':<26} {py_ms:>10.1f} {py_frames / py_ms / 1e3:>12.1f}")
    print(f"\n  DECODE: batcher is {speedup:.1f}x faster than the Python baseline.")

    # Resample to 16 kHz — the audio-ML preprocessing step. Gate on the (ratio-determined)
    # output frame count, which both the native sinc kernel and soxr produce identically.
    try:
        import soxr  # noqa: F401
    except ImportError:
        print("\n(resample comparison skipped: pip install soxr)")
        return 0
    rbt_ms, rbt_frames = _best_ms(_batcher_resample, table, args.runs)
    rpy_ms, rpy_frames = _best_ms(_python_resample, table, args.runs)
    if rbt_frames != rpy_frames:
        print(f"RESAMPLE MISMATCH: batcher {rbt_frames} frames, python {rpy_frames}")
        return 1
    rspeedup = rpy_ms / rbt_ms if rbt_ms else float("inf")
    print(f"\n  resample to {_TARGET_SR} Hz ({rbt_frames:,} output frames):")
    print(f"  {'batcher (native sinc)':<26} {rbt_ms:>10.1f}")
    print(f"  {'python (soundfile+librosa)':<26} {rpy_ms:>10.1f}")
    print(f"\n  RESAMPLE: batcher is {rspeedup:.1f}x faster than soundfile+librosa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
