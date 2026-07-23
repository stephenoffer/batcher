"""Point-cloud / LiDAR training-data loading: batcher vs Ray Data.

Physical-AI (autonomous-driving, robotics) training reads a corpus of point-cloud frames
— each an ``(P, 3)`` or ``(P, 4)`` array of points — off disk, batches them into tensors,
and feeds a model. The frames are stored as ``.npy`` shards (the ``read_numpy`` convention:
a leading axis of frames, each a fixed ``(P, C)`` cloud), so the loader's job is exactly
what an image loader does for pixels — read many files concurrently, assemble Arrow tensor
columns, and stream them to torch in bounded memory.

batcher reads the shards concurrently into a **fixed-shape-tensor** column (the shape rides
in Arrow field metadata, zero-copy across the FFI) and streams ``iter_torch_batches`` to
per-batch tensors; Ray Data's ``read_numpy → iter_torch_batches`` is the direct competitor.
Both run over the *same* on-disk shards and, per the harness discipline, must agree on the
total frame count before any timing is trusted.

Run:
    python benchmarks/scenarios/point_cloud_load.py                    # 200 shards x 100 frames
    python benchmarks/scenarios/point_cloud_load.py --shards 400 --points 8192
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time

import numpy as np


def _write_shards(directory: str, shards: int, frames: int, points: int) -> int:
    """Write ``shards`` ``.npy`` files of ``(frames, points, 3)`` float32 clouds.

    Returns the total number of frames (rows) written across every shard.
    """
    rng = np.random.default_rng(0)
    for i in range(shards):
        arr = rng.random((frames, points, 3), dtype=np.float32)
        np.save(os.path.join(directory, f"shard_{i:05d}.npy"), arr)
    return shards * frames


def _batcher_frames(directory: str, batch_size: int) -> int:
    """batcher: ``read.numpy`` (concurrent, tensor column) → streamed torch batches."""
    import batcher as bt
    from batcher.ml import iter_torch_batches

    ds = bt.read.numpy(os.path.join(directory, "*.npy"))
    total = 0
    for batch in iter_torch_batches(ds, batch_size=batch_size, columns=["data"], device=None):
        total += int(batch["data"].shape[0])
    return total


def _ray_frames(directory: str, batch_size: int) -> int:
    """Ray Data: ``read_numpy`` → ``iter_torch_batches`` (the direct competitor)."""
    import ray
    import ray.data

    ray.init(ignore_reinit_error=True, logging_level="ERROR")
    total = 0
    for batch in ray.data.read_numpy(directory).iter_torch_batches(batch_size=batch_size):
        column = next(iter(batch.values()))
        total += int(column.shape[0])
    return total


def _best_ms(fn, directory: str, batch_size: int, runs: int) -> tuple[float, int]:
    """Best-of-``runs`` wall-clock (ms) plus the frame count (checked identical)."""
    best = float("inf")
    result = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn(directory, batch_size)
        best = min(best, time.perf_counter() - t0)
    return best * 1000, result


def main() -> int:
    parser = argparse.ArgumentParser(description="Point-cloud loading (batcher vs Ray Data)")
    parser.add_argument("--shards", type=int, default=200, help="number of .npy shards")
    parser.add_argument("--frames", type=int, default=100, help="frames (rows) per shard")
    parser.add_argument("--points", type=int, default=4096, help="points per frame")
    parser.add_argument("--batch-size", type=int, default=256, help="torch batch size")
    parser.add_argument("--runs", type=int, default=3, help="best-of-N timed repeats")
    args = parser.parse_args()

    directory = tempfile.mkdtemp(prefix="bt_pcd_")
    try:
        print(
            f"writing {args.shards} shards x {args.frames} frames x {args.points}x3 ...",
            flush=True,
        )
        n_frames = _write_shards(directory, args.shards, args.frames, args.points)

        bt_ms, bt_frames = _best_ms(_batcher_frames, directory, args.batch_size, args.runs)
        try:
            ray_ms, ray_frames = _best_ms(_ray_frames, directory, args.batch_size, args.runs)
        except ImportError:
            print("Ray not installed; showing batcher only.")
            ray_ms, ray_frames = None, bt_frames

        # Correctness gate: every engine must stream the same number of frames.
        if bt_frames != n_frames or ray_frames != n_frames:
            print(f"MISMATCH: wrote {n_frames}, batcher {bt_frames}, ray {ray_frames}")
            return 1

        print(f"\ncorpus: {n_frames:,} LiDAR frames of {args.points}x3 points")
        print(f"(best-of-{args.runs})\n")
        print(f"  {'engine':<26} {'ms':>10} {'frames/s':>12}")
        print(f"  {'-' * 50}")
        print(f"  {'batcher':<26} {bt_ms:>10.1f} {n_frames / bt_ms * 1e3:>12.0f}")
        if ray_ms is not None:
            print(f"  {'ray data':<26} {ray_ms:>10.1f} {n_frames / ray_ms * 1e3:>12.0f}")
            print(f"\n  batcher is {ray_ms / bt_ms:.1f}x faster than Ray Data.")
        return 0
    finally:
        shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
