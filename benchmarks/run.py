"""Run the Batcher benchmark suite against the engines it claims to beat.

Benchmarks are registered by family under ``suites/`` and discovered through
``registry.REGISTRY``; this module is the thin CLI that selects the engines, loads the
public dataset (``sources`` — no data is generated), runs the cases, and reports
them. Correctness is verified before any timing is trusted (see ``harness.py``): a
query is only timed once the engines agree.

Run (single-node default lineup: batcher, duckdb, polars, pyarrow):
    source .venv/bin/activate
    python3 benchmarks/run.py                                  # TPC-H, scale 1
    python3 benchmarks/run.py --benchmark clickbench           # ClickBench (hits)
    python3 benchmarks/run.py --benchmark tpcds --scale 1      # TPC-DS, all 99 queries
    python3 benchmarks/run.py --benchmark job                  # Join Order Benchmark, 113 queries
    python3 benchmarks/run.py --benchmark h2o-groupby          # H2O.ai db-benchmark groupby
    python3 benchmarks/run.py --benchmark h2o-join             # H2O.ai db-benchmark joins
    python3 benchmarks/run.py --benchmark operators            # operator-mix
    python3 benchmarks/run.py --benchmark scan                 # parquet file-layout scan
    python3 benchmarks/run.py --benchmark images               # multimodal image ingest
    python3 benchmarks/run.py --benchmark all                  # every dataset except scan/images

    python3 benchmarks/run.py --engines batcher,duckdb,spark   # opt in to PySpark
    python3 benchmarks/run.py --tier multi                     # batcher, ray, daft
    python3 benchmarks/run.py --benchmark tpch --family tpch --only q1
    python3 benchmarks/run.py --benchmark scan --family scan-many_small
    python3 benchmarks/run.py --list                           # list, do not run

This is the single entrypoint: besides the engine-comparison datasets, it also
dispatches the standalone benchmarks (`--benchmark distributed | optimizer | shuffle`).
"""

from __future__ import annotations

import argparse
import dataclasses
import time

import batcher as bt
import engines as engines_mod
import suites  # noqa: F401  (import registers every benchmark)
from batcher.config import active_config, set_config
from context import CORPUS_BENCHMARKS, Context
from envinfo import require_release_build
from harness import compare, emit_result, print_table, run_isolated
from registry import REGISTRY

_SIZE_UNITS = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}


def _parse_size(text: str) -> int:
    """Parse a byte size like ``64GB`` / ``16G`` / ``1048576`` into bytes."""
    s = text.strip().upper().rstrip("B").rstrip("I")  # tolerate GB / GiB / G
    unit = s[-1] if s and s[-1] in _SIZE_UNITS else ""
    num = s[: -1 if unit else len(s)] or s
    return int(float(num) * _SIZE_UNITS[unit])


# Engine-comparison datasets (run through the correctness-gated compare()).
BENCHMARKS = (
    "tpch",
    "tpcds",
    "clickbench",
    "operators",
    "json",
    "job",
    "h2o-groupby",
    "h2o-join",
    "scan",
    "images",
)
# What `--benchmark all` sweeps. Four datasets are deliberately excluded. `scan` and
# `images` each re-read their corpus from object storage on every repeat, so a full run is
# tens of minutes. The two `h2o-*` datasets default to the db-benchmark's own 1e7-row tier,
# whose largest cases (a group-by on all six keys, a 1e7-row join) produce results as big as
# their inputs — the harness compares every row of those across every engine. `job` loads a
# 1.8 GiB real database (and downloads 1.2 GiB the first time). All five are opt-in
# (`--benchmark job`) for the same reason Spark is.
ALL_DATASETS = ("tpch", "tpcds", "clickbench", "operators", "json")
# Standalone benchmarks with their own reporting, dispatched by this single runner.
AUX = ("distributed", "optimizer", "shuffle")


def _runs_for(scale: float, benchmark: str) -> int:
    """Best-of-N: more repeats when the data is small enough to make them cheap.

    The corpus benchmarks (scan, images) re-read from object storage on every repeat (that
    is the point — the read is the measurement), so they stay at the floor rather than
    paying five full passes over a many-small-files corpus per case.
    """
    if benchmark in CORPUS_BENCHMARKS:
        return 2 if scale <= 10 else 1
    if scale <= 1:
        return 5
    return 3 if scale <= 10 else 2


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batcher benchmark suite")
    p.add_argument(
        "--benchmark",
        choices=(*BENCHMARKS, "all", *AUX),
        default="tpch",
        help="dataset (tpch/tpcds/clickbench/operators/json/job/h2o-groupby/h2o-join/"
        "scan/images) or aux (distributed/optimizer/shuffle); 'all' sweeps every dataset "
        "but scan/images/job/h2o-*, which are opt-in because of their size",
    )
    p.add_argument(
        "--engines",
        default=None,
        help="comma-separated engine lineup (default: the tier's lineup)",
    )
    p.add_argument(
        "--tier",
        choices=("single", "multi"),
        default="single",
        help="default lineup: single (batcher,duckdb,polars,pyarrow) or multi (batcher,ray,daft)",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="TPC-H / TPC-DS scale factor; for --benchmark scan, the corpus size "
        "(1=1GiB, 10=10GiB, ...); for --benchmark images, the image count "
        "(1=10, 10=100, 100=1000, ...); for --benchmark h2o-*, the db-benchmark row tier "
        "(1=1e7 rows, its smallest published size; 10=1e8)",
    )
    p.add_argument("--partitions", type=int, default=8, help="shuffle partitions (distributed aux)")
    p.add_argument("--source", default=None, help="override the dataset's parquet base URI")
    p.add_argument("--family", default=None, help="run only this family (exact match)")
    p.add_argument(
        "--only",
        default=None,
        help="run only cases whose name contains this substring; comma-separate several "
        "(e.g. --only q17,q72) to time an arbitrary subset in one process, which is what an "
        "A/B over a handful of queries needs — a process per query re-loads the tables and "
        "spends more wall time on the fixture than on the measurement",
    )
    p.add_argument(
        "--skip",
        action="append",
        default=None,
        metavar="SUBSTRING",
        help="skip cases whose name contains this substring (repeatable). For working "
        "around a query that takes the *process* down rather than raising — the harness "
        "catches an exception per engine, but nothing catches a SIGKILL, and one such "
        "query otherwise costs every result after it. Never use it to hide a wrong "
        "answer: a FAILED row is the benchmark working.",
    )
    p.add_argument(
        "--memory-bytes",
        default=None,
        help="pin Batcher's memory cap (e.g. 64GB) to force the bounded-envelope spill "
        "path instead of the auto-sensed host RAM — for exercising out-of-core at scale",
    )
    p.add_argument("--spill-dir", default=None, help="local scratch dir for spilled batches")
    p.add_argument(
        "--scan",
        action="store_true",
        help="scan mode: bind each table to a lazy native parquet scan instead of "
        "preloading Arrow (required at sf100+; SQL suites only). Combine with --source "
        "pointing at canonical-named parquet.",
    )
    p.add_argument(
        "--isolate",
        action="store_true",
        help="run each case in its own subprocess, so a query that takes the process "
        "down (OOM kill, native abort) costs one KILLED row instead of every result "
        "after it. Pays a dataset load per case; use it when a suite cannot complete.",
    )
    p.add_argument(
        "--isolate-case",
        default=None,
        help=argparse.SUPPRESS,  # internal: the child half of --isolate
    )
    p.add_argument("--list", action="store_true", help="list registered benchmarks and exit")
    p.add_argument(
        "--allow-debug-build",
        action="store_true",
        help="time an unoptimized (dev-profile) engine anyway; the ratios are not comparable",
    )
    return p.parse_args()


def _apply_memory_config(args: argparse.Namespace) -> None:
    """Pin Batcher's memory envelope / spill dir from the CLI, if given."""
    if args.memory_bytes is None and args.spill_dir is None:
        return
    cfg = active_config()
    overrides: dict[str, object] = {}
    if args.memory_bytes is not None:
        overrides["max_memory_bytes"] = _parse_size(args.memory_bytes)
    if args.spill_dir is not None:
        overrides["spill_dir"] = args.spill_dir
    set_config(cfg.replace(memory=dataclasses.replace(cfg.memory, **overrides)))


def _list_benchmarks() -> None:
    print(f"{len(REGISTRY.select())} registered benchmarks:\n")
    for ds in REGISTRY.datasets():
        print(f"[{ds}]")
        for case in REGISTRY.select(dataset=ds):
            print(f"  {case.family:<18} {case.name}")
        print()


def _run_dataset(benchmark: str, args: argparse.Namespace, engines: list) -> list:
    wanted = [t for t in (args.only or "").split(",") if t] or [None]
    cases = [
        c
        for token in wanted
        for c in REGISTRY.select(dataset=benchmark, family=args.family, name=token)
    ]
    cases = list(dict.fromkeys(cases))  # a case matching two tokens is still run once
    if args.skip:
        dropped = [c.name for c in cases if any(s in c.name for s in args.skip)]
        cases = [c for c in cases if not any(s in c.name for s in args.skip)]
        if dropped:
            # Say what was dropped. A silently shortened suite reads as full coverage.
            print(f"skipping {len(dropped)} case(s): {', '.join(dropped)}\n")
    if args.isolate_case is not None:
        # The child half of --isolate: exactly one case, selected by its full name so a
        # name that is a substring of another cannot pull in its neighbour.
        cases = [c for c in cases if c.name == args.isolate_case]
    if not cases:
        return []
    names = [e.name for e in engines]
    if args.isolate:
        results = run_isolated([c.name for c in cases])
        print(f"=== {benchmark} ({', '.join(names)}) ===")
        print_table(results, names)
        print()
        return results
    t0 = time.perf_counter()
    if benchmark in CORPUS_BENCHMARKS:
        ctx = Context.build_corpus(benchmark, args.scale, engines, args.source)
    elif args.scan:
        ctx = Context.build_scan(benchmark, args.scale, engines, args.source)
    else:
        ctx = Context.build(benchmark, args.scale, engines, args.source)
    runs = _runs_for(args.scale, benchmark)
    elapsed = time.perf_counter() - t0
    mode = "corpus" if benchmark in CORPUS_BENCHMARKS else ("scan" if args.scan else "loaded")
    print(f"{mode} {benchmark} (scale {args.scale}) in {elapsed:.2f}s, best-of-{runs}\n")
    results = []
    for case in cases:
        print(f"running {case.name} ...", flush=True)
        results.append(compare(case.name, case.build(ctx), names, runs=runs))
    if args.isolate_case is not None:
        for result in results:
            emit_result(result)
        return results
    print()
    print(f"=== {benchmark} ({', '.join(names)}) ===")
    print_table(results, names)
    print()
    return results


def _run_aux(which: str, args: argparse.Namespace) -> int:
    """Dispatch a standalone benchmark (its own reporting, not the compare() table)."""
    if which == "distributed":
        from internals import distributed

        return distributed.run(args.scale, args.partitions)
    if which == "optimizer":
        from internals import optimizer_bench

        return optimizer_bench.main()
    from internals import shuffle_vs_object_store

    return shuffle_vs_object_store.main(args.partitions)


def main() -> int:
    args = _parse_args()

    if args.list:
        _list_benchmarks()
        return 0

    if args.benchmark in AUX:
        return _run_aux(args.benchmark, args)

    _apply_memory_config(args)
    names = args.engines.split(",") if args.engines else engines_mod.default_names(args.tier)
    engines = engines_mod.resolve([n.strip() for n in names])
    print(f"Batcher benchmark suite  (engine {bt.engine_version()})")
    print(f"engines: {', '.join(e.name for e in engines)}\n")
    require_release_build(allow_debug=args.allow_debug_build)

    datasets = ALL_DATASETS if args.benchmark == "all" else (args.benchmark,)
    all_results = []
    for ds in datasets:
        all_results += _run_dataset(ds, args, engines)

    if not all_results:
        print("no benchmarks matched the selection.")
        return 0

    failed = [r for r in all_results if r.status in ("FAILED", "ERROR", "KILLED")]
    if failed:
        print(f"{len(failed)} query(ies) FAILED correctness, errored, or died.")
        return 1
    print("All correctness checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
