"""Putting a LiDAR sweep in world coordinates: batcher vs NumPy, and fused vs composed.

The innermost loop of every autonomous-driving pipeline. A sweep is a hundred thousand
points in the sensor's frame, a log is tens of thousands of sweeps, and almost nothing can
be asked of any of it until the points are in a common frame.

Two questions, and they are different:

1. **Against NumPy.** Pulling the sweep out of the engine and rotating it with NumPy is
   what an AV team does today, and it is the honest competitor — NumPy's rigid transform
   is a `(3, 3) @ (3, N)` matrix product in BLAS, which is very hard to beat on raw
   throughput. What it cannot do is stay in a query: the round trip materializes the sweep,
   and it does not spill, stream or distribute. This measures the arithmetic alone, which
   is the comparison least favourable to the engine and therefore the one worth publishing.

2. **Fused versus composed.** `se3_transform` lowers to three `Expr::Spatial` kernels; the
   same transform is also expressible in ordinary arithmetic as a rotation matrix, which is
   what a user would otherwise type. The fused kernel is a scalar row loop and the composed
   form is a chain of vectorized Arrow kernels, so this is not the foregone conclusion it
   looks like, and the answer decides whether the IR node earns its place on speed or only
   on ergonomics.

Both engines must agree to 1e-9 before any timing is reported.

Run:
    python benchmarks/scenarios/robotics/sweep_transform.py
    python benchmarks/scenarios/robotics/sweep_transform.py --points 20000000 --runs 5
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pyarrow as pa

import batcher as bt
from batcher import col

#: Column layout: a per-row pose (the ego pose at the moment each point was measured,
#: which is what motion compensation produces) plus the point itself.
_POSE = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
_POINT = ("x", "y", "z")


def _sweep(points: int) -> pa.Table:
    """A sweep of `points` returns with a per-row pose, as Arrow."""
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-60.0, 60.0, size=(points, 3))
    # A yaw-only ego rotation that drifts across the sweep, which is what a turning
    # vehicle produces and what makes the per-row pose non-constant.
    yaw = np.linspace(0.0, 0.4, points)
    return pa.table(
        {
            "x": pa.array(xyz[:, 0]),
            "y": pa.array(xyz[:, 1]),
            "z": pa.array(xyz[:, 2]),
            "tx": pa.array(np.full(points, 100.0)),
            "ty": pa.array(np.full(points, 50.0)),
            "tz": pa.array(np.full(points, 1.8)),
            "qx": pa.array(np.zeros(points)),
            "qy": pa.array(np.zeros(points)),
            "qz": pa.array(np.sin(yaw / 2.0)),
            "qw": pa.array(np.cos(yaw / 2.0)),
        }
    )


def _fused(ds: bt.Dataset) -> pa.Table:
    """The `Expr::Spatial` kernels, through the public helper.

    Collected to Arrow, not to Python. `to_pydict` would build fifteen million Python
    floats and swamp the thing being measured — the engine would be a rounding error on
    a benchmark of CPython's allocator.
    """
    return ds.select(**bt.se3_transform(_POSE, _POINT)).to_arrow()


def _composed(ds: bt.Dataset) -> pa.Table:
    """The same transform as ordinary arithmetic: normalize, build the matrix, apply it."""
    qx, qy, qz, qw = (col(c) for c in ("qx", "qy", "qz", "qw"))
    n = (qx * qx + qy * qy + qz * qz + qw * qw).sqrt()
    ux, uy, uz, uw = qx / n, qy / n, qz / n, qw / n
    px, py, pz = (col(c) for c in _POINT)
    return ds.select(
        x=(1 - 2 * (uy * uy + uz * uz)) * px
        + 2 * (ux * uy - uz * uw) * py
        + 2 * (ux * uz + uy * uw) * pz
        + col("tx"),
        y=2 * (ux * uy + uz * uw) * px
        + (1 - 2 * (ux * ux + uz * uz)) * py
        + 2 * (uy * uz - ux * uw) * pz
        + col("ty"),
        z=2 * (ux * uz - uy * uw) * px
        + 2 * (uy * uz + ux * uw) * py
        + (1 - 2 * (ux * ux + uy * uy)) * pz
        + col("tz"),
    ).to_arrow()


def _numpy(table: pa.Table) -> dict[str, np.ndarray]:
    """NumPy's rigid transform, applied per row because the pose varies per row.

    Written the way it is actually written: build the rotation matrices, then contract
    them against the points with `einsum`. A single `(3, 3) @ (3, N)` would be faster and
    is not applicable here, because each point has its own pose.
    """
    cols = {name: table[name].to_numpy() for name in (*_POSE, *_POINT)}
    qx, qy, qz, qw = (cols[c] for c in ("qx", "qy", "qz", "qw"))
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    ux, uy, uz, uw = qx / n, qy / n, qz / n, qw / n
    m = np.empty((len(ux), 3, 3))
    m[:, 0, 0] = 1 - 2 * (uy * uy + uz * uz)
    m[:, 0, 1] = 2 * (ux * uy - uz * uw)
    m[:, 0, 2] = 2 * (ux * uz + uy * uw)
    m[:, 1, 0] = 2 * (ux * uy + uz * uw)
    m[:, 1, 1] = 1 - 2 * (ux * ux + uz * uz)
    m[:, 1, 2] = 2 * (uy * uz - ux * uw)
    m[:, 2, 0] = 2 * (ux * uz - uy * uw)
    m[:, 2, 1] = 2 * (uy * uz + ux * uw)
    m[:, 2, 2] = 1 - 2 * (ux * ux + uy * uy)
    p = np.stack([cols["x"], cols["y"], cols["z"]], axis=1)
    out = np.einsum("nij,nj->ni", m, p)
    return {
        "x": out[:, 0] + cols["tx"],
        "y": out[:, 1] + cols["ty"],
        "z": out[:, 2] + cols["tz"],
    }


def _best_ms(fn, arg, runs: int) -> tuple[float, object]:
    """Best-of-`runs` wall clock in milliseconds, plus the last result."""
    fn(arg)  # warm: first call pays plan build and any lazy import
    best, result = float("inf"), None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = fn(arg)
        best = min(best, time.perf_counter() - t0)
    return best * 1000, result


def _column(result: pa.Table | dict, axis: str) -> np.ndarray:
    """One coordinate of a result, whichever shape it came back in."""
    if isinstance(result, pa.Table):
        return result.column(axis).to_numpy(zero_copy_only=False)
    return np.asarray(result[axis], dtype=float)


def _agree(a: pa.Table | dict, b: pa.Table | dict, label: str) -> bool:
    """True when two results match to 1e-9 on every coordinate."""
    for axis in ("x", "y", "z"):
        left = _column(a, axis)
        right = _column(b, axis)
        worst = float(np.max(np.abs(left - right)))
        if worst > 1e-9:
            print(f"MISMATCH vs {label} on {axis}: worst |delta| = {worst:g}")
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Rigid-body sweep transform")
    parser.add_argument("--points", type=int, default=5_000_000, help="returns in the sweep")
    parser.add_argument("--runs", type=int, default=3, help="best-of-N timed repeats")
    args = parser.parse_args()

    print(f"building a {args.points:,}-point sweep with a per-row pose ...", flush=True)
    table = _sweep(args.points)
    ds = bt.from_arrow(table)

    fused_ms, fused_out = _best_ms(_fused, ds, args.runs)
    composed_ms, composed_out = _best_ms(_composed, ds, args.runs)
    numpy_ms, numpy_out = _best_ms(_numpy, table, args.runs)

    # Correctness gate: nothing is timed on a path that disagrees.
    if not _agree(fused_out, composed_out, "composed"):
        return 1
    if not _agree(fused_out, numpy_out, "numpy"):
        return 1

    rows = args.points
    print(f"\nse3_transform over {rows:,} points -> 3 columns (best-of-{args.runs})\n")
    print(f"  {'path':<34} {'ms':>10} {'Mpoint/s':>12}")
    print(f"  {'-' * 58}")
    for label, ms in (
        ("batcher, fused Expr::Spatial", fused_ms),
        ("batcher, composed arithmetic", composed_ms),
        ("numpy einsum (materialized)", numpy_ms),
    ):
        print(f"  {label:<34} {ms:>10.1f} {rows / ms / 1e3:>12.1f}")
    print(f"\n  fused / composed = {fused_ms / composed_ms:.2f}x")
    print(f"  fused / numpy    = {fused_ms / numpy_ms:.2f}x")
    print(
        "\n  NumPy materializes the whole sweep and cannot spill, stream or distribute;\n"
        "  both batcher paths run inside a query that does all three."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
