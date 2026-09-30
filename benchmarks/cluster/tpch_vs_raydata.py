"""Batcher vs Ray Data on Ray's own TPC-H benchmark, correctness-gated, on a live cluster.

The queries are `tpch_raydata.QUERIES`, a statement-for-statement port of Ray's
`release/nightly_tests/dataset/tpch/tpch_q*.py`. Ray's numbers come from running those
scripts unmodified (see ``--ray-scripts``), or from the Ray Data team's published SF1000
report, which states its cluster alongside every figure.

Correctness comes first, as everywhere in this suite: with ``--reference DIR`` every result
is compared with the one Ray produced for the same query at the same scale (captured by the
Ray run with ``RAY_TPCH_CAPTURE=DIR``) before its time is reported. Rows are compared as a
multiset with a float tolerance, and a query Ray sorts is additionally checked to come back
in that order, since a multiset comparison cannot see a sort bug.

Timing mirrors Ray's harness: one query at a time, construction inside the timer, cold
object-store reads (the worker scan cache is off unless ``--warm``). The fleet is warmed
once before the first timer, as the Spark port of this benchmark warms its executors.

Run:
    python benchmarks/cluster/tpch_vs_raydata.py --sf 1000 --queries all --distributed
    python benchmarks/cluster/tpch_vs_raydata.py --sf 1 --reference /tmp/ray_sf1 --local
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tpch_raydata import QUERIES, load_tables

from envinfo import require_release_build

BASE_URI = "s3://ray-benchmark-data/tpch/parquet"

# The ORDER BY of each query Ray sorts, as (column, descending) pairs, so a result can be
# checked for order as well as content. q6/q14/q17/q19 return one row and are omitted.
SORT_KEYS: dict[str, list[tuple[str, bool]]] = {
    "q1": [("l_returnflag", False), ("l_linestatus", False)],
    "q2": [("s_acctbal", True), ("n_name", False), ("s_name", False), ("p_partkey", False)],
    "q3": [("revenue", True), ("o_orderdate", False)],
    "q4": [("o_orderpriority", False)],
    "q5": [("revenue", True)],
    "q7": [("n_name_supp", False), ("n_name_cust", False), ("l_year", False)],
    "q8": [("o_year", False)],
    "q9": [("n_name", False), ("o_year", True)],
    "q10": [("revenue", True)],
    "q11": [("value", True)],
    "q12": [("l_shipmode", False)],
    "q13": [("custdist", True), ("c_count", True)],
    "q15": [("s_suppkey", False)],
    "q16": [("supplier_cnt", True), ("p_brand", False), ("p_type", False), ("p_size", False)],
    "q18": [("o_totalprice", True), ("o_orderdate", False)],
    "q20": [("s_name", False)],
    "q21": [("numwait", True), ("s_name", False)],
    "q22": [("cntrycode", False)],
}
# Ray keeps a LIMIT's ties in whatever order its sort left them, so for these the rows past
# the sort keys' last distinct value may legitimately differ; they are compared on keys only.
_LIMITED = {"q2", "q21"}


def _init_cluster() -> None:
    """Ship the working-tree Batcher to the cluster, forwarding every BATCHER_* variable."""
    import ray

    import batcher as bt
    from batcher.config import active_config, set_config

    env = {k: v for k, v in os.environ.items() if k.startswith("BATCHER_")}
    package = os.path.dirname(os.path.abspath(bt.__file__))
    shared = os.environ.get("BATCHER_CLUSTER_CODE_DIR")
    if shared:
        # A directory every node mounts (an Anyscale workspace's /mnt/cluster_storage): the
        # package is copied there and put on the workers' path, rather than zipped, uploaded
        # and unpacked per node on every edit -- minutes each time with the engine inside.
        # rsync, not a copy: it skips unchanged files and replaces a changed one by rename,
        # so a worker that still has the old engine mapped is not pulled out from under.
        import subprocess

        dest = os.path.join(shared, "batcher")
        os.makedirs(shared, exist_ok=True)
        subprocess.run(["rsync", "-a", "--delete", f"{package}/", dest], check=True)
        runtime_env = {"env_vars": env | {"PYTHONPATH": shared}}
    else:
        runtime_env = {"py_modules": [package], "env_vars": env}
    base = active_config()
    set_config(
        base.replace(
            distributed=dataclasses.replace(
                base.distributed, ray_address="auto", runtime_env=runtime_env
            )
        )
    )
    if not ray.is_initialized():
        ray.init(
            address="auto", runtime_env=runtime_env, logging_level="ERROR", log_to_driver=False
        )
    _prime_runtime_env()


def _prime_runtime_env() -> None:
    """Install the shipped runtime env on every node before anything is timed or placed.

    Each node unpacks the working-tree package (tens of MB with the engine) before its first
    worker can import it, which takes minutes on a fresh cluster -- longer than the fleet's
    placement deadline, so the first distributed query failed with a `ResourceError` or ran
    narrow. A zero-CPU task pinned to each node waits it out, with no deadline of its own.
    """
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy as Pin

    @ray.remote(num_cpus=0)
    def ready() -> bool:
        import batcher  # noqa: F401  (the import is the point: it needs the env unpacked)

        return True

    t = time.perf_counter()
    nodes = [n["NodeID"] for n in ray.nodes() if n.get("Alive")]
    ray.get([ready.options(scheduling_strategy=Pin(n, soft=False)).remote() for n in nodes])
    print(f"runtime env ready on {len(nodes)} nodes in {time.perf_counter() - t:.1f}s", flush=True)


class _Monitor:
    """Per-node CPU busy % and NIC receive rate, sampled once a second by a zero-CPU actor.

    Low CPU across a query is the signal that it is waiting -- on the network, on the
    driver, or on a scheduling gap -- rather than computing, so every timed query can say
    which of its nodes did the work.
    """

    def __init__(self) -> None:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy as Pin

        @ray.remote(num_cpus=0)
        class Sampler:
            def __init__(self) -> None:
                self.samples: list[tuple[float, float]] = []
                self.running = False

            @staticmethod
            def _read() -> tuple[float, int, int, int]:
                with open("/proc/stat") as f:
                    cpu = [int(x) for x in f.readline().split()[1:]]
                rx = 0
                with open("/proc/net/dev") as f:
                    for line in f.readlines()[2:]:
                        name, rest = line.split(":", 1)
                        if name.strip() != "lo":
                            rx += int(rest.split()[0])
                return time.time(), sum(cpu), cpu[3] + cpu[4], rx

            def start(self) -> None:
                import threading

                self.samples, self.running = [], True

                def loop() -> None:
                    prev = self._read()
                    while self.running:
                        time.sleep(1.0)
                        cur = self._read()
                        busy = 1 - (cur[2] - prev[2]) / max(1, cur[1] - prev[1])
                        rate = (cur[3] - prev[3]) / 1e6 / max(1e-3, cur[0] - prev[0])
                        self.samples.append((100 * busy, rate))
                        prev = cur

                threading.Thread(target=loop, daemon=True).start()

            def stop(self) -> list[tuple[float, float]]:
                self.running = False
                return self.samples

        nodes = [n for n in ray.nodes() if n["Alive"]]
        self.names = [n["NodeManagerAddress"] for n in nodes]
        self.actors = [
            Sampler.options(scheduling_strategy=Pin(n["NodeID"], soft=False)).remote()
            for n in nodes
        ]

    def start(self) -> None:
        import ray

        ray.get([a.start.remote() for a in self.actors])

    def report(self) -> str:
        import ray

        rows = []
        for name, samples in zip(
            self.names, ray.get([a.stop.remote() for a in self.actors]), strict=True
        ):
            if samples:
                cpu = sum(c for c, _ in samples) / len(samples)
                rx = sum(r for _, r in samples) / len(samples)
                rows.append(f"{name}: cpu {cpu:4.0f}% rx {rx:5.0f} MB/s")
        return "\n    ".join(rows)


def _norm(v: object) -> object:
    if isinstance(v, float):
        return None if math.isnan(v) else float(f"{v:.9g}")
    return v


def _rows(tbl: pa.Table, cols: list[str]) -> list[tuple]:
    data = [tbl.column(c).to_pylist() for c in cols]
    return [tuple(_norm(v) for v in row) for row in zip(*data, strict=True)]


def _close(a: object, b: object) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    return str(a) == str(b)


def check(name: str, got: pa.Table, ref: pa.Table) -> str | None:
    """Why `got` differs from Ray's `ref`, or `None` when they agree."""
    cols = [c for c in ref.column_names if c in got.column_names]
    missing = sorted(set(ref.column_names) - set(got.column_names))
    if missing:
        return f"columns missing vs ray: {missing}"
    if got.num_rows != ref.num_rows:
        return f"{got.num_rows} rows, ray has {ref.num_rows}"
    keys = SORT_KEYS.get(name, [])
    if keys:
        order = [(c, "descending" if d else "ascending") for c, d in keys]
        resorted = got.sort_by(order)
        if _rows(resorted, [c for c, _ in keys]) != _rows(got, [c for c, _ in keys]):
            return "not in the query's sort order"
    compare = [c for c, _ in keys] if name in _LIMITED else cols
    mine = sorted(_rows(got, compare), key=repr)
    theirs = sorted(_rows(ref, compare), key=repr)
    for a, b in zip(mine, theirs, strict=True):
        if not all(_close(x, y) for x, y in zip(a, b, strict=True)):
            return f"row differs: {a} vs ray {b}"
    return None


def check_standard(got: pa.Table, ref: pa.Table) -> str | None:
    """Why `got` differs from DuckDB's `ref` for a standard query, or `None` when they agree.

    Every standard query orders its whole output, so the rows are compared in order; a
    multiset comparison could not see a sort bug. Columns are matched by position, since
    the SQL names them and both engines keep that order.
    """
    if got.num_columns != ref.num_columns:
        return f"{got.num_columns} columns, duckdb has {ref.num_columns}"
    if got.num_rows != ref.num_rows:
        return f"{got.num_rows} rows, duckdb has {ref.num_rows}"
    mine = _rows(got.rename_columns(ref.column_names), ref.column_names)
    theirs = _rows(ref, ref.column_names)
    for i, (a, b) in enumerate(zip(mine, theirs, strict=True)):
        if not all(_close(x, y) for x, y in zip(a, b, strict=True)):
            same = sorted(mine, key=repr) == sorted(theirs, key=repr)
            return f"row {i} differs{' (order only)' if same else ''}: {a} vs duckdb {b}"
    return None


def _read_reference(root: str, name: str) -> pa.Table | None:
    """`tpch_<name>.parquet` under `root` (a local path or a URI), or None when absent."""
    import pyarrow.fs as pafs

    fs, base = pafs.FileSystem.from_uri(root if "://" in root else str(Path(root).resolve()))
    path = f"{base}/tpch_{name}.parquet"
    if fs.get_file_info(path).type == pafs.FileType.NotFound:
        return None
    return pq.read_table(path, filesystem=fs)


def _standard_queries() -> dict:
    """Standard TPC-H, as the repo's DuckDB-checked SQL, over the same renamed scans."""
    import batcher as bt
    from suites.standard.tpch import QUERIES as SQL

    def build(sql: str):
        def run(tables: dict, _sf: int) -> bt.Dataset:
            session = bt.Session()
            for table, ds in tables.items():
                session.register(table, ds)
            return session.sql(sql)

        return run

    return {name.removeprefix("tpch-"): build(sql) for name, sql in SQL.items()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sf", type=int, required=True)
    p.add_argument(
        "--suite",
        choices=("raydata", "standard"),
        default="raydata",
        help="Ray's release-test variants, or standard TPC-H SQL (what Daft's answers.py runs)",
    )
    p.add_argument("--queries", default="all")
    p.add_argument("--base-uri", default=BASE_URI)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--distributed", action="store_true", help="collect(distributed=True)")
    mode.add_argument("--local", action="store_true", help="collect(distributed=False)")
    p.add_argument("--workers", type=int, default=None, help="num_workers when distributed")
    p.add_argument("--runs", type=int, default=1, help="timed runs per query (reports each)")
    p.add_argument(
        "--reference",
        help="dir or URI of reference results (<tpch_qN>.parquet): Ray's for the raydata suite, "
        "`tpch_reference.py`'s DuckDB answers for the standard one",
    )
    p.add_argument("--warm", action="store_true", help="leave the worker scan cache on")
    p.add_argument(
        "--object-cache-gb",
        type=float,
        default=0.0,
        help="per-process block cache of remote object bytes, in GiB (the warm path "
        "Databricks' disk cache measures; 0 = off)",
    )
    p.add_argument("--out", help="append one JSON line per query here")
    p.add_argument("--save-results", help="write each query's first result here as <query>.parquet")
    p.add_argument("--monitor", action="store_true", help="print per-node CPU/network per query")
    p.add_argument("--verbose", action="store_true", help="log the executors' phase timings")
    args = p.parse_args()
    require_release_build()

    if args.verbose:
        import logging

        logging.basicConfig(format="    %(name)s: %(message)s")
        logging.getLogger("batcher.dist").setLevel(logging.INFO)
    if not args.warm:
        os.environ.setdefault("BATCHER_SCAN_CACHE_BYTES", "0")
    if args.object_cache_gb > 0:
        os.environ["BATCHER_OBJECT_CACHE_BYTES"] = str(int(args.object_cache_gb * (1 << 30)))
    builders = QUERIES if args.suite == "raydata" else _standard_queries()
    names = list(builders) if args.queries == "all" else args.queries.split(",")
    distributed: bool | str = True if args.distributed else (False if args.local else "auto")
    if distributed is not False:
        _init_cluster()

    import batcher as bt

    def collect(ds: bt.Dataset) -> pa.Table:
        kw = {"distributed": distributed}
        if distributed is True and args.workers:
            kw["num_workers"] = args.workers
        return ds.collect(**kw)

    tables = load_tables(args.base_uri, args.sf)
    t0 = time.perf_counter()
    # Warm-up: fleet, worker imports and object-store clients. A bare `count()` is answered
    # from the footers without a worker, so it warmed nothing and the first timed query of
    # each process paid for the cluster's start-up (minutes, once a runtime env ships).
    warm = tables["orders"].group_by("o_orderstatus").agg(s=bt.col("o_totalprice").sum())
    collect(warm)
    print(f"warm-up {time.perf_counter() - t0:.1f}s (untimed)", flush=True)

    monitor = _Monitor() if args.monitor and distributed is not False else None
    failures = 0
    for name in names:
        for run in range(args.runs):
            if monitor is not None:
                monitor.start()
            t = time.perf_counter()
            status = "ok"
            try:
                got = collect(builders[name](tables, args.sf))
                secs = time.perf_counter() - t
                if args.reference and run == 0:
                    ref = _read_reference(args.reference, name)
                    oracle = "duckdb" if args.suite == "standard" else "ray"
                    if ref is None:
                        status = f"ok (no {oracle} reference)"
                    else:
                        why = (
                            check_standard(got, ref)
                            if args.suite == "standard"
                            else check(name, got, ref)
                        )
                        status = f"ok (matches {oracle})" if why is None else f"WRONG: {why}"
                rows = got.num_rows
                if args.save_results and run == 0:
                    Path(args.save_results).mkdir(parents=True, exist_ok=True)
                    pq.write_table(got, Path(args.save_results) / f"tpch_{name}.parquet")
            except Exception as exc:
                secs, rows = time.perf_counter() - t, None
                traceback.print_exc()
                status = f"FAILED {type(exc).__name__}: {str(exc)[:300]}"
            failures += not status.startswith("ok")
            print(f"{name} run{run} {secs:.2f}s rows={rows} {status}", flush=True)
            if monitor is not None:
                print(f"    {monitor.report()}", flush=True)
            if args.out:
                with open(args.out, "a") as f:
                    rec = {"engine": "batcher", "sf": args.sf, "query": name, "run": run}
                    rec |= {"secs": secs, "rows": rows, "status": status}
                    rec |= {"distributed": distributed, "workers": args.workers}
                    f.write(json.dumps(rec) + "\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
