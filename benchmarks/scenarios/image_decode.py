"""Image decode+resize ingest: batcher vs Ray Data vs Daft.

The physical-AI / computer-vision ingest hot path: read a corpus of JPEG frames, decode
each, and resize to the model input (``224x224``) — the first stage of every vision
training and batch-inference pipeline. batcher does the whole thing in the native data
plane: files read concurrently, then ``col(bytes).image.to_tensor`` decodes (zune-jpeg,
DCT-scaled for large frames) and resizes (SIMD ``fast_image_resize``) fanned across every
core, emitting a fixed-shape-tensor column directly (no per-batch re-type UDF). Ray Data
(``read_images``) and Daft (``url.download → decode_image → resize``) are the multimodal
competitors.

Frames are synthesized locally (structured, JPEG-compressible — representative of camera
input, unlike random noise which is a pathological worst case for any JPEG decoder), so the
benchmark needs no network or cloud corpus. Cross-engine pixel equality is not a sound gate
(decoders/resizers differ), so the check is that every engine processed the same **frame
count** at the same **output size** before its throughput is trusted.

Run:
    python benchmarks/scenarios/image_decode.py                 # 2,000 frames, 640x480 -> 224
    python benchmarks/scenarios/image_decode.py --frames 8000 --width 1280 --height 720
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time

import numpy as np

_TARGET = (224, 224)  # (H, W) model input


def _write_corpus(directory: str, frames: int, height: int, width: int) -> int:
    """Write ``frames`` structured JPEGs (a gradient plus shapes — compressible, like a
    real scene) at ``height x width``. Returns the number of files written."""
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.stack(
        [
            (xx * 255 // width).astype(np.uint8),
            (yy * 255 // height).astype(np.uint8),
            np.full((height, width), 128, np.uint8),
        ],
        axis=-1,
    )
    for i in range(frames):
        img = Image.fromarray(base.copy())
        draw = ImageDraw.Draw(img)
        for _ in range(5):
            x0, y0 = int(rng.integers(0, width - 60)), int(rng.integers(0, height - 60))
            color = tuple(int(v) for v in rng.integers(0, 255, 3))
            draw.ellipse([x0, y0, x0 + 60, y0 + 60], fill=color)
        img.save(os.path.join(directory, f"frame_{i:06d}.jpg"), format="JPEG", quality=90)
    return frames


def _batcher(directory: str) -> int:
    import batcher as bt

    d = bt.read.images(os.path.join(directory, "*.jpg"), decode=True, size=_TARGET).collect()
    height, width, _ = d.column("image").type.shape  # fixed-shape-tensor extension type
    assert (height, width) == _TARGET, (height, width)
    return d.num_rows


def _ray(directory: str) -> int:
    import ray
    import ray.data

    ray.init(ignore_reinit_error=True, logging_level="ERROR")
    rows = ray.data.read_images(directory, size=_TARGET).take_all()
    assert rows[0]["image"].shape[:2] == _TARGET
    return len(rows)


def _daft(directory: str) -> int:
    import daft
    from daft.functions import decode_image, download, resize

    paths = sorted(os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".jpg"))
    df = daft.from_pydict({"path": paths})
    img = resize(decode_image(download(daft.col("path"))), _TARGET[1], _TARGET[0])
    return df.with_column("img", img).collect().count_rows()


def _best_ms(fn, directory: str, runs: int) -> tuple[float, int]:
    """Best-of-``runs`` wall-clock (ms) plus the frame count (checked identical)."""
    best = float("inf")
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn(directory)
        best = min(best, time.perf_counter() - t0)
    return best * 1000, result


def main() -> int:
    parser = argparse.ArgumentParser(description="Image decode+resize (batcher vs Ray vs Daft)")
    parser.add_argument("--frames", type=int, default=2000, help="number of JPEG frames")
    parser.add_argument("--height", type=int, default=480, help="source frame height")
    parser.add_argument("--width", type=int, default=640, help="source frame width")
    parser.add_argument("--runs", type=int, default=3, help="best-of-N timed repeats")
    args = parser.parse_args()

    directory = tempfile.mkdtemp(prefix="bt_img_")
    try:
        print(f"writing {args.frames} JPEGs at {args.width}x{args.height} ...", flush=True)
        n = _write_corpus(directory, args.frames, args.height, args.width)

        results: dict[str, tuple[float, int]] = {}
        for name, fn in (("batcher", _batcher), ("ray data", _ray), ("daft", _daft)):
            try:
                results[name] = _best_ms(fn, directory, args.runs)
            except ImportError:
                print(f"  ({name} not installed — skipped)")
            except Exception as exc:  # one engine failing must not abort the rest
                print(f"  ({name} failed: {type(exc).__name__}: {exc})")

        # Correctness gate: every engine must process every frame.
        for name, (_, count) in results.items():
            if count != n:
                print(f"MISMATCH: {name} processed {count} of {n} frames")
                return 1

        print(f"\ncorpus: {n:,} frames, {args.width}x{args.height} -> {_TARGET[0]}x{_TARGET[1]}")
        print(f"(best-of-{args.runs})\n")
        print(f"  {'engine':<20} {'ms':>10} {'img/s':>10}")
        print(f"  {'-' * 42}")
        for name, (ms, _) in results.items():
            print(f"  {name:<20} {ms:>10.1f} {n / ms * 1e3:>10.0f}")
        if "batcher" in results:
            bt_ms = results["batcher"][0]
            print()
            for other in ("ray data", "daft"):
                if other in results:
                    print(f"  batcher is {results[other][0] / bt_ms:.2f}x faster than {other}.")
        return 0
    finally:
        shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
