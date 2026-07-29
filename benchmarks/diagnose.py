"""Diagnose *why* a Batcher query is slow, rather than only how slow it is.

The comparison suite (`run.py`) answers "is Batcher ahead of DuckDB on this query". When the
answer is no, these subcommands answer the next four questions, and they are the ones that
found the sf10 join regression recorded in `BENCHMARK_RESULTS.md`:

``decompose``
    Wall time, CPU time and the ratio between them for a set of related shapes. The ratio is
    the useful number: a query at 10x parallelism on a 96-core box is not slow because its
    kernels are slow, and no amount of kernel work will fix it.
``executors``
    The same query under both single-node executors. They do not dominate one another —
    streaming bounds intermediates, materializing spills and partitions joins across cores —
    so a routing decision between them has to be measured, not assumed.
``time``
    Every run printed as it lands, cold one included. Best-of-N (what `run.py` reports) hides
    a cold run that costs 70x the warm ones, which is exactly what it did for TPC-H q5.
``mem``
    Peak RSS sampled from a side thread, so a run that is killed still reports how far it got.

Run from `benchmarks/`:

    python3 diagnose.py decompose --scale 10
    python3 diagnose.py executors --scale 10 --only q5
    python3 diagnose.py time --scale 10 --query tpch-q5 --repeat 3
    python3 diagnose.py mem --scale 10 --query tpch-q5 --executor materializing
"""

from __future__ import annotations

import argparse
import dataclasses
import threading
import time

import batcher as bt
from batcher.config import active_config, set_config
from sources import tables as src
from suites.standard.tpch import QUERIES

# Join shapes that isolate one cost each: the join alone, the join plus a payload gather, and
# the join plus a grouped aggregate. Subtracting them attributes the time.
SHAPES = {
    "agg-only": "SELECT SUM(l_extendedprice*(1-l_discount)) FROM lineitem",
    "agg-grouped": (
        "SELECT l_returnflag, SUM(l_extendedprice*(1-l_discount)) "
        "FROM lineitem GROUP BY l_returnflag"
    ),
    "join-count": "SELECT count(*) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey",
    "join-sum": (
        "SELECT SUM(l.l_extendedprice) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey"
    ),
    "join-agg-str": (
        "SELECT o.o_orderpriority, SUM(l.l_extendedprice*(1-l.l_discount)) AS revenue "
        "FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey GROUP BY o.o_orderpriority"
    ),
    "join-agg-int": (
        "SELECT o.o_shippriority, SUM(l.l_extendedprice*(1-l.l_discount)) AS revenue "
        "FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey GROUP BY o.o_shippriority"
    ),
    "join-semi": (
        "SELECT count(*) FROM lineitem l WHERE EXISTS "
        "(SELECT 1 FROM orders o WHERE o.o_orderkey = l.l_orderkey)"
    ),
}

TPCH_TABLES = ("lineitem", "orders", "customer", "part", "partsupp", "supplier", "nation", "region")


def _session(loaded: dict, tables: tuple[str, ...] | None = None) -> bt.Session:
    sess = bt.Session()
    for name, table in loaded.items():
        if tables is None or name in tables:
            sess.register(name, bt.from_arrow(table))
    return sess


def _set_executor(base, executor: str, parallelism: int = 0) -> None:
    set_config(
        base.replace(
            execution=dataclasses.replace(
                base.execution,
                streaming=executor == "streaming",
                parallelism=parallelism,
            )
        )
    )


def _best(sess: bt.Session, sql: str, repeat: int) -> tuple[float, float]:
    """`(wall, cpu)` of the fastest of `repeat` runs, in seconds."""
    best = (float("inf"), 0.0)
    for _ in range(repeat):
        t0, c0 = time.perf_counter(), time.process_time()
        sess.sql(sql).collect()
        wall, cpu = time.perf_counter() - t0, time.process_time() - c0
        if wall < best[0]:
            best = (wall, cpu)
    return best


def _rss_gb() -> float:
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * 4096 / 1e9


def cmd_decompose(args: argparse.Namespace) -> int:
    loaded = src.load_tables("tpch", args.scale, None)
    sess = _session(loaded, ("lineitem", "orders"))
    for name, sql in SHAPES.items():
        if args.only and args.only not in name:
            continue
        wall, cpu = _best(sess, sql, args.repeat)
        print(
            f"{name:14s} wall {wall * 1000:8.1f}ms  cpu {cpu * 1000:9.1f}ms  "
            f"parallelism {cpu / wall:5.1f}x",
            flush=True,
        )
    return 0


def cmd_executors(args: argparse.Namespace) -> int:
    loaded = src.load_tables("tpch", args.scale, None)
    base = active_config()
    faster = 0
    total = 0
    for name, sql in QUERIES.items():
        if args.only and args.only not in name:
            continue
        timings = []
        for executor in ("streaming", "materializing"):
            _set_executor(base, executor)
            try:
                timings.append(_best(_session(loaded), sql, args.repeat)[0] * 1000)
            except Exception as exc:
                print(f"{name} {executor}: ERROR {exc}", flush=True)
                timings.append(float("nan"))
        total += 1
        faster += timings[1] < timings[0]
        print(
            f"{name:10s} streaming {timings[0]:9.1f}ms   materializing {timings[1]:9.1f}ms   "
            f"mat/stream {timings[1] / timings[0]:.2f}x",
            flush=True,
        )
    set_config(base)
    print(f"\nmaterializing faster on {faster}/{total}")
    return 0


def cmd_time(args: argparse.Namespace) -> int:
    loaded = src.load_tables("tpch", args.scale, None)
    _set_executor(active_config(), args.executor, args.parallelism)
    sess = _session(loaded)
    sql = QUERIES[args.query]
    for i in range(args.repeat):
        t0 = time.perf_counter()
        rows = sess.sql(sql).collect()
        print(
            f"{args.query} {args.executor} run {i}: "
            f"{(time.perf_counter() - t0) * 1000:.0f} ms ({rows.num_rows} rows)",
            flush=True,
        )
    return 0


def cmd_mem(args: argparse.Namespace) -> int:
    loaded = src.load_tables("tpch", args.scale, None)
    print(f"sources: {sum(t.nbytes for t in loaded.values()) / 1e9:.1f} GB arrow", flush=True)
    _set_executor(active_config(), args.executor, args.parallelism)
    sess = _session(loaded)

    peak = _rss_gb()
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not stop.wait(0.05):
            peak = max(peak, _rss_gb())

    watcher = threading.Thread(target=sample, daemon=True)
    watcher.start()
    print(f"baseline RSS {_rss_gb():.1f} GB, running {args.query}", flush=True)
    t0 = time.perf_counter()
    rows = sess.sql(QUERIES[args.query]).collect()
    wall = (time.perf_counter() - t0) * 1000
    stop.set()
    watcher.join(timeout=1)
    print(
        f"{args.query} {args.executor}: {wall:.0f} ms, {rows.num_rows} rows, peak RSS {peak:.1f} GB"
    )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scale", type=float, default=10.0)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--only", default=None, help="substring filter on the case/query name")
    p.add_argument("--query", default="tpch-q5")
    p.add_argument("--executor", choices=("streaming", "materializing"), default="streaming")
    p.add_argument("--parallelism", type=int, default=0, help="worker count; 0 = all cores")
    p.add_argument("command", choices=("decompose", "executors", "time", "mem"))
    args = p.parse_args()
    return {
        "decompose": cmd_decompose,
        "executors": cmd_executors,
        "time": cmd_time,
        "mem": cmd_mem,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
