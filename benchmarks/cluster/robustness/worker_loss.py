"""What a lost worker costs: recovery time and result equality under injected failure.

Every distributed claim in this repository is a throughput claim. None of them says what
happens when a worker dies mid-query, which is the property a shuffle's design is actually
for. This measures it directly: run one grouped aggregate through the Flight shuffle, kill
workers at a chosen phase, and report both halves of the answer.

* **Correctness** — the recovered result must equal the single-node result exactly. This is
  the load-bearing half. A lost bucket reads back as EMPTY rather than as an error, so a
  defect here drops rows silently instead of failing.
* **Cost** — wall time against the no-fault arm, which is the number nobody publishes.

Two failure phases, because they recover by different mechanisms and cost differently:

``reduce``  the worker dies after the map barrier, so its *published buckets* vanish. With
            ``shuffle_replication = 2`` a survivor holds a byte-identical copy and the
            reducer re-fetches; with ``1`` there is no copy and the map round is re-run.
``map``     the worker dies before it publishes anything, so its *source split* is
            relocated onto a survivor. This is the likelier spot-instance failure, and the
            more expensive one, because the map phase reads the input.

Absolute times here are from a single-box local Ray cluster: this measures the recovery
mechanism, not cluster-scale recovery. The ratios are the result; the milliseconds are not.

Run:
    python benchmarks/cluster/robustness/worker_loss.py
    python benchmarks/cluster/robustness/worker_loss.py --rows 4000000 --workers 8
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

import batcher as bt
from batcher import col, count
from batcher.config import Config, DistributedConfig, config_context

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))


def _data(rows: int, keys: int) -> pa.Table:
    rng = np.random.default_rng(19)
    return pa.table(
        {
            "k": rng.integers(0, keys, rows).astype("int64"),
            "v": rng.integers(0, 100, rows).astype("int64"),
        }
    )


def _norm(t: pa.Table) -> set:
    return {tuple(r.values()) for r in t.to_pylist()}


def _run(plan_src, workers: int, *, kill: set[int] | None, phase: str, replication: int):
    """One execution of the distributed aggregate, optionally killing `kill` at `phase`."""
    from batcher.dist.flight_aggregate import execute_aggregate_flight

    ds = plan_src()
    kw: dict = {}
    if kill and phase == "reduce":
        kw["_fault_inject"] = kill
    elif kill and phase == "map":
        kw["_fault_inject_map"] = kill
    cfg = Config(distributed=DistributedConfig(shuffle_replication=replication))
    with config_context(cfg):
        t0 = time.perf_counter()
        out = execute_aggregate_flight([], ds._plan, ds._sources, workers=workers, **kw)
        return (time.perf_counter() - t0) * 1000.0, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2_000_000)
    ap.add_argument("--keys", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    from _ray_cluster import init_test_ray, shutdown_test_ray

    started = init_test_ray(args.workers * 2)
    try:
        table = _data(args.rows, args.keys)

        def plan_src():
            return bt.from_arrow(table).group_by("k").agg(s=col("v").sum(), n=count())

        expected = _norm(plan_src().collect())

        arms = [
            ("no fault, replication=1", None, "reduce", 1),
            ("no fault, replication=2", None, "reduce", 2),
            ("1 lost at reduce, replication=1", {1}, "reduce", 1),
            ("1 lost at reduce, replication=2", {1}, "reduce", 2),
            ("2 lost at reduce, replication=2", {1, 3}, "reduce", 2),
            ("1 lost during map", {1}, "map", 1),
            ("2 lost during map", {1, 3}, "map", 1),
        ]

        print(
            f"\nWorker loss: {args.rows:,} rows, {args.keys} keys, {args.workers} workers, "
            f"best of {args.rounds}, local Ray cluster"
        )
        hdr = f"{'arm':<34}{'ms':>9}{'vs no fault':>13}{'result':>10}"
        print(hdr)
        print("-" * len(hdr))
        base = None
        for label, kill, phase, repl in arms:
            times, ok = [], True
            for _ in range(args.rounds):
                ms, out = _run(plan_src, args.workers, kill=kill, phase=phase, replication=repl)
                times.append(ms)
                ok = ok and _norm(out) == expected
            best = min(times)
            if base is None:
                base = best
            verdict = "IDENTICAL" if ok else "*** WRONG ***"
            print(f"{label:<34}{best:>9.1f}{best / base:>12.2f}x{verdict:>14}")
        print("-" * len(hdr))
        print(f"median round spread: {statistics.median(times):.1f} ms")
        print("Every arm must read IDENTICAL; a lost bucket returns empty, never an error.")
    finally:
        shutdown_test_ray(started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
