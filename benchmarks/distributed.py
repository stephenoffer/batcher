"""Distributed equivalence and timing benchmark.

Exercises the mergeable execution path (partial / combine / finalize over a hash
shuffle) by running each query two ways: single-node and across several
partitions via ``collect(distributed=True, num_partitions=...)``. The mergeable
algebra guarantees the two results are identical, so this benchmark asserts that
equivalence first and only then reports timings. It is the single-node == many-
partition invariant from the engine contract, measured.

Run:
    source .venv/bin/activate
    python3 benchmarks/distributed.py           # default ~2M rows, 8 partitions
    python3 benchmarks/distributed.py 5000000 16

Requires the optional ``ray`` extra. Without it the benchmark exits cleanly with
a skip message rather than failing.
"""

from __future__ import annotations

import sys
import time

import batcher as bt
from batcher import col, count
from contexts import SyntheticContext
from harness import bench, results_match

try:
    import ray  # noqa: F401

    HAVE_RAY = True
except ImportError:
    HAVE_RAY = False

DEFAULT_SCALE = 2_000_000


def build_plans(ctx: SyntheticContext):
    """Return (name, plan) pairs; each plan is a non-collected Dataset.

    These mirror the mergeable operators (aggregate, join, distinct) the
    distributed path composes from ``partial / combine / finalize``.
    """
    return [
        ("groupby-agg", ctx.bf.group_by("k1").agg(s=col("price").sum(), n=count())),
        ("groupby-2key", ctx.bf.group_by("k1", "k2").agg(s=col("qty").sum(), n=count())),
        (
            "join+groupby",
            ctx.bf.join(ctx.bd, on="dim_key", how="inner")
            .group_by("region")
            .agg(s=col("price").sum(), n=count()),
        ),
        ("distinct", ctx.bf.select("k1", "k2").distinct()),
    ]


def main() -> int:
    if not HAVE_RAY:
        print("ray is not installed; skipping distributed benchmark.")
        print("Install the optional extra with:  uv pip install -e '.[ray]'")
        return 0

    scale = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SCALE
    num_partitions = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    runs = 3

    print(f"Batcher distributed benchmark  (engine {bt.engine_version()})")
    print(f"scale = {scale:,} rows, num_partitions = {num_partitions}, best-of-{runs}\n")

    t0 = time.perf_counter()
    ctx = SyntheticContext.build(scale)
    print(f"generated data in {time.perf_counter() - t0:.2f}s\n")

    rows = []
    any_mismatch = False
    for name, plan in build_plans(ctx):
        print(f"running {name} ...", flush=True)
        single = plan.collect()
        dist = plan.collect(distributed=True, num_partitions=num_partitions)
        ok, msg = results_match(single, dist)
        if not ok:
            any_mismatch = True

        def run_dist(p=plan):
            return p.collect(distributed=True, num_partitions=num_partitions)

        sn_ms = bench(plan.collect, runs=runs)
        di_ms = bench(run_dist, runs=runs)
        speedup = f"{sn_ms / di_ms:.2f}x" if di_ms else "-"
        rows.append((name, sn_ms, di_ms, speedup, "OK" if ok else f"MISMATCH: {msg}"))

    print()
    headers = ["query", "single_ms", "dist_ms", "single/dist", "status"]
    widths = [len(h) for h in headers]
    table = [[n, f"{s:.1f}", f"{d:.1f}", sp, st] for (n, s, d, sp, st) in rows]
    for row in table:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells):
        return "  ".join(
            c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(cells)
        )

    print(fmt(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in table:
        print(fmt(row))

    print()
    if any_mismatch:
        print("Distributed result diverged from single-node. This is a correctness bug.")
        return 1
    print("Distributed results match single-node on every query.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
