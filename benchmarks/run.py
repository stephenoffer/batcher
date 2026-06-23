"""Run the Batcher benchmark suite against DuckDB and Polars.

Benchmarks are registered by family under ``suites/`` and discovered through
``registry.REGISTRY``; this module is the thin CLI that selects, runs, and reports
them. Correctness is verified before any timing is trusted (see ``harness.py``):
a query is only timed once the three engines agree.

Run:
    source .venv/bin/activate
    python3 benchmarks/run.py                     # operator mix, ~10M rows
    python3 benchmarks/run.py 2000000             # custom synthetic row count
    python3 benchmarks/run.py --dataset tpch      # TPC-H subset (default sf 0.1)
    python3 benchmarks/run.py --dataset all       # both datasets
    python3 benchmarks/run.py --family joins      # only the joins family
    python3 benchmarks/run.py --only window       # only cases whose name matches
    python3 benchmarks/run.py --list              # list registered benchmarks, do not run
"""

from __future__ import annotations

import argparse
import time

import batcher as bt
import suites
from contexts import DEFAULT_ROWS, DEFAULT_SF, SyntheticContext, TpchContext
from harness import compare, print_table
from registry import REGISTRY


def _runs_for_synthetic(rows: int) -> int:
    return 3 if rows >= 5_000_000 else 5


def _runs_for_tpch(scale: float) -> int:
    return 5 if scale <= 0.2 else 3


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batcher benchmark suite")
    p.add_argument(
        "rows",
        nargs="?",
        type=int,
        default=None,
        help="synthetic fact-table row count (back-compat positional; default 10M)",
    )
    p.add_argument("--rows", dest="rows_opt", type=int, default=None, help="synthetic row count")
    p.add_argument("--sf", type=float, default=DEFAULT_SF, help="TPC-H scale factor")
    p.add_argument(
        "--dataset",
        choices=("synthetic", "tpch", "all"),
        default="synthetic",
        help="which dataset family to run (default: synthetic)",
    )
    p.add_argument("--family", default=None, help="run only this family (exact match)")
    p.add_argument("--only", default=None, help="run only cases whose name contains this substring")
    p.add_argument("--list", action="store_true", help="list registered benchmarks and exit")
    return p.parse_args()


def _list_benchmarks() -> None:
    print(f"{len(REGISTRY.select())} registered benchmarks:\n")
    for ds in REGISTRY.datasets():
        print(f"[{ds}]")
        for case in REGISTRY.select(dataset=ds):
            print(f"  {case.family:<14} {case.name}")
        print()


def _run_dataset(label: str, ctx, cases, runs: int) -> list:
    results = []
    for case in cases:
        print(f"running {case.name} ...", flush=True)
        results.append(compare(case.name, case.build(ctx), runs=runs))
    print()
    print(f"=== {label} ===")
    print_table(results)
    print()
    return results


def main() -> int:
    args = _parse_args()
    suites.load_all()

    if args.list:
        _list_benchmarks()
        return 0

    rows = args.rows_opt if args.rows_opt is not None else args.rows
    if rows is None:
        rows = DEFAULT_ROWS

    datasets = ("synthetic", "tpch") if args.dataset == "all" else (args.dataset,)
    print(f"Batcher benchmark suite  (engine {bt.engine_version()})")

    all_results = []
    for ds in datasets:
        cases = REGISTRY.select(dataset=ds, family=args.family, name=args.only)
        if not cases:
            continue
        t0 = time.perf_counter()
        if ds == "synthetic":
            ctx = SyntheticContext.build(rows)
            runs = _runs_for_synthetic(rows)
            built = f"synthetic {rows:,} rows"
        else:
            ctx = TpchContext.build(args.sf)
            runs = _runs_for_tpch(args.sf)
            built = f"TPC-H sf={args.sf} ({ctx.tables['lineitem'].num_rows:,} lineitem rows)"
        print(f"built {built} in {time.perf_counter() - t0:.2f}s, best-of-{runs}\n")
        all_results += _run_dataset(ds, ctx, cases, runs)

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
