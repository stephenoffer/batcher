"""Run the Batcher benchmark suite against the engines it claims to beat.

Benchmarks are registered by family under ``suites/`` and discovered through
``registry.REGISTRY``; this module is the thin CLI that selects the engines, loads the
public dataset (``sources.py`` — no data is generated), runs the cases, and reports
them. Correctness is verified before any timing is trusted (see ``harness.py``): a
query is only timed once the engines agree.

Run (single-node default lineup: batcher, duckdb, polars, pyarrow):
    source .venv/bin/activate
    python3 benchmarks/run.py                                  # TPC-H, scale 1
    python3 benchmarks/run.py --benchmark clickbench           # ClickBench (hits)
    python3 benchmarks/run.py --benchmark tpcds --scale 1      # TPC-DS subset
    python3 benchmarks/run.py --benchmark operators            # operator-mix
    python3 benchmarks/run.py --benchmark all                  # every dataset

    python3 benchmarks/run.py --engines batcher,duckdb,spark   # opt in to PySpark
    python3 benchmarks/run.py --tier multi                     # batcher, ray, daft
    python3 benchmarks/run.py --benchmark tpch --family tpch --only q1
    python3 benchmarks/run.py --list                           # list, do not run

This is the single entrypoint: besides the engine-comparison datasets, it also
dispatches the standalone benchmarks (`--benchmark distributed | optimizer | shuffle`).
"""

from __future__ import annotations

import argparse
import time

import batcher as bt
import engines as engines_mod
import suites  # noqa: F401  (import registers every benchmark)
from context import Context
from harness import compare, print_table
from registry import REGISTRY

# Engine-comparison datasets (run through the correctness-gated compare()).
BENCHMARKS = ("tpch", "tpcds", "clickbench", "operators")
# Standalone benchmarks with their own reporting, dispatched by this single runner.
AUX = ("distributed", "optimizer", "shuffle")


def _runs_for(scale: float) -> int:
    """Best-of-N: more repeats when the data is small enough to make them cheap."""
    if scale <= 1:
        return 5
    return 3 if scale <= 10 else 2


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batcher benchmark suite")
    p.add_argument(
        "--benchmark",
        choices=(*BENCHMARKS, "all", *AUX),
        default="tpch",
        help="dataset (tpch/tpcds/clickbench/operators/all) or aux (distributed/optimizer/shuffle)",
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
    p.add_argument("--scale", type=float, default=1.0, help="TPC-H / TPC-DS scale factor")
    p.add_argument("--partitions", type=int, default=8, help="shuffle partitions (distributed aux)")
    p.add_argument("--source", default=None, help="override the dataset's parquet base URI")
    p.add_argument("--family", default=None, help="run only this family (exact match)")
    p.add_argument("--only", default=None, help="run only cases whose name contains this substring")
    p.add_argument("--list", action="store_true", help="list registered benchmarks and exit")
    return p.parse_args()


def _list_benchmarks() -> None:
    print(f"{len(REGISTRY.select())} registered benchmarks:\n")
    for ds in REGISTRY.datasets():
        print(f"[{ds}]")
        for case in REGISTRY.select(dataset=ds):
            print(f"  {case.family:<18} {case.name}")
        print()


def _run_dataset(benchmark: str, args: argparse.Namespace, engines: list) -> list:
    cases = REGISTRY.select(dataset=benchmark, family=args.family, name=args.only)
    if not cases:
        return []
    names = [e.name for e in engines]
    t0 = time.perf_counter()
    ctx = Context.build(benchmark, args.scale, engines, args.source)
    runs = _runs_for(args.scale)
    elapsed = time.perf_counter() - t0
    print(f"loaded {benchmark} (scale {args.scale}) in {elapsed:.2f}s, best-of-{runs}\n")
    results = []
    for case in cases:
        print(f"running {case.name} ...", flush=True)
        results.append(compare(case.name, case.build(ctx), names, runs=runs))
    print()
    print(f"=== {benchmark} ({', '.join(names)}) ===")
    print_table(results, names)
    print()
    return results


def _run_aux(which: str, args: argparse.Namespace) -> int:
    """Dispatch a standalone benchmark (its own reporting, not the compare() table)."""
    if which == "distributed":
        import distributed

        return distributed.run(args.scale, args.partitions)
    if which == "optimizer":
        import optimizer_bench

        return optimizer_bench.main()
    import shuffle_vs_object_store

    return shuffle_vs_object_store.main()


def main() -> int:
    args = _parse_args()

    if args.list:
        _list_benchmarks()
        return 0

    if args.benchmark in AUX:
        return _run_aux(args.benchmark, args)

    names = args.engines.split(",") if args.engines else engines_mod.default_names(args.tier)
    engines = engines_mod.resolve([n.strip() for n in names])
    print(f"Batcher benchmark suite  (engine {bt.engine_version()})")
    print(f"engines: {', '.join(e.name for e in engines)}\n")

    datasets = BENCHMARKS if args.benchmark == "all" else (args.benchmark,)
    all_results = []
    for ds in datasets:
        all_results += _run_dataset(ds, args, engines)

    if not all_results:
        print("no benchmarks matched the selection.")
        return 0

    failed = [r for r in all_results if r.status in ("FAILED", "ERROR")]
    if failed:
        print(f"{len(failed)} query(ies) FAILED correctness or errored.")
        return 1
    print("All correctness checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
