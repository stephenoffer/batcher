"""Distributed equivalence and timing benchmark.

Exercises the mergeable execution path (partial / combine / finalize over a hash
shuffle) by running each query two ways: single-node and across several
partitions via ``collect(distributed=True, num_partitions=...)``. The mergeable
algebra guarantees the two results are identical, so this benchmark asserts that
equivalence first and only then reports timings. It is the single-node == many-
partition invariant from the engine contract, measured.

Run:
    source .venv/bin/activate
    python3 benchmarks/distributed.py           # TPC-H scale 1, 8 partitions
    python3 benchmarks/distributed.py 10 16     # scale 10, 16 partitions

Reads the public TPC-H tables (``sources.py`` — no data is generated). Requires the
optional ``ray`` extra; without it the benchmark exits cleanly with a skip message.
"""

from __future__ import annotations

import sys
import time

import batcher as bt
import engines as engines_mod
from batcher import col, count
from context import Context
from harness import bench, results_match

try:
    import ray  # noqa: F401

    HAVE_RAY = True
except ImportError:
    HAVE_RAY = False

DEFAULT_SCALE = 1.0


def build_plans(ctx: Context):
    """Return (name, plan) pairs; each plan is a non-collected Dataset.

    These mirror the mergeable operators (aggregate, join, distinct) the
    distributed path composes from ``partial / combine / finalize``, over the real
    TPC-H ``lineitem`` ⋈ ``orders`` tables.
    """
    lineitem = ctx.handle("lineitem", "batcher")
    # orders keyed to lineitem so the equi-join is on a shared column name.
    orders = ctx.handle("orders", "batcher").rename({"o_orderkey": "l_orderkey"})
    return [
        (
            "groupby-agg",
            lineitem.group_by("l_returnflag").agg(s=col("l_extendedprice").sum(), n=count()),
        ),
        (
            "groupby-2key",
            lineitem.group_by("l_returnflag", "l_linestatus").agg(
                s=col("l_quantity").sum(), n=count()
            ),
        ),
        (
            "join+groupby",
            lineitem.join(orders, on="l_orderkey", how="inner")
            .group_by("o_orderpriority")
            .agg(s=col("l_extendedprice").sum(), n=count()),
        ),
        ("distinct", lineitem.select("l_returnflag", "l_linestatus").distinct()),
    ]


def run(scale: float = DEFAULT_SCALE, num_partitions: int = 8, runs: int = 3) -> int:
    """Run the single-node vs many-partition equivalence + timing benchmark."""
    if not HAVE_RAY:
        print("ray is not installed; skipping distributed benchmark.")
        print("Install the optional extra with:  uv pip install -e '.[ray]'")
        return 0

    print(f"Batcher distributed benchmark  (engine {bt.engine_version()})")
    print(f"TPC-H scale = {scale}, num_partitions = {num_partitions}, best-of-{runs}\n")

    t0 = time.perf_counter()
    ctx = Context.build("tpch", scale, engines_mod.resolve(["batcher"]))
    print(f"loaded data in {time.perf_counter() - t0:.2f}s\n")

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


def main() -> int:
    scale = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SCALE
    num_partitions = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    return run(scale, num_partitions)


if __name__ == "__main__":
    raise SystemExit(main())
