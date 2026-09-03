"""Run the Batcher benchmark suite against the engines it claims to beat.

Benchmarks are registered by family under ``suites/`` and discovered through
``registry.REGISTRY``; this module is the thin CLI that selects the engines, loads the
public dataset (``sources`` — no data is generated), runs the cases, and reports
them. Correctness is verified before any timing is trusted (see ``harness.py``): a
query is only timed once the engines agree.

Run (single-node default lineup: batcher, duckdb, polars, pyarrow, daft):
    source .venv/bin/activate
    python3 benchmarks/run.py                                  # TPC-H, scale 1
    python3 benchmarks/run.py --benchmark clickbench           # ClickBench (hits)
    python3 benchmarks/run.py --benchmark tpcds --scale 1      # TPC-DS, all 99 queries
    python3 benchmarks/run.py --benchmark job                  # Join Order Benchmark, 113 queries
    python3 benchmarks/run.py --benchmark h2o-groupby          # H2O.ai db-benchmark groupby
    python3 benchmarks/run.py --benchmark h2o-join             # H2O.ai db-benchmark joins
    python3 benchmarks/run.py --benchmark operators            # operator-mix
    python3 benchmarks/run.py --benchmark scan                 # parquet file-layout scan
    python3 benchmarks/run.py --benchmark cache                # result-cache tiers + lookup join
    python3 benchmarks/run.py --benchmark images               # multimodal image ingest
    python3 benchmarks/run.py --benchmark all                  # every dataset except scan/images

    python3 benchmarks/run.py --engines batcher,duckdb,spark   # opt in to PySpark
    python3 benchmarks/run.py --tier multi                     # batcher, daft
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

import engines as engines_mod
import suites  # noqa: F401  (import registers every benchmark)
from batcher.config import active_config, set_config
from context import CORPUS_BENCHMARKS, Context
from envinfo import machine_fingerprint, require_quiet_box, require_release_build
from harness import (
    compare,
    emit_result,
    format_repeats,
    format_summary,
    format_unstable,
    print_table,
    run_isolated,
    summarize,
)
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
AUX = ("distributed", "optimizer", "shuffle", "cache")


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
        help="default lineup: single (batcher,duckdb,polars,pyarrow,daft) or multi "
        "(batcher,daft). `ray` is registered but in no default — pass it explicitly. "
        "`duckdb_arrow` likewise; see engines/lineup.py for why each is opt-in",
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
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="run the whole selection N times and report the geomean spread. A suite "
        "geomean quoted to three decimals from a single run claims a precision no single "
        "run supports — the operator mix measures at a 4.1%% spread across three runs on "
        "one box. Use this before quoting a figure.",
    )
    p.add_argument(
        "--allow-busy-box",
        action="store_true",
        help="time on a contended machine anyway; the numbers are not comparable to any other run",
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
        # Say which mode produced these numbers. `--isolate` is not a neutral packaging
        # choice: measured on TPC-H sf1 over four alternated passes, the suite geomean reads
        # **0.725 isolated against 0.693 in-process** — a 4.4% difference from a flag that
        # changes nothing about the queries, against a within-mode pass-to-pass spread of
        # under 1%. Run alone, DuckDB does not care which mode it is in (+1.3%, per-query
        # signs 11 of 22) while Batcher gains 3.8% from the shared process, so what the
        # shared-process board credits into the ratio is Batcher's cross-query carry-over —
        # a credit the comparator has nothing to gain from. The two modes otherwise print an
        # identical table, which is how a board saved from one gets compared against a board
        # saved from the other.
        print(f"isolated {benchmark} (scale {args.scale}), one process per case")
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
    print(
        f"{mode} {benchmark} (scale {args.scale}) in {elapsed:.2f}s, best-of-{runs}, "
        "one process for every case"
    )
    # Say what `--only` actually selected. It matches on *substring*, so `--only q1` pulls in
    # q10 through q19 as well — twelve cases, and the geomean printed underneath is then a
    # mean over all twelve rather than the one query the reader asked for. The `--isolate`
    # path already selects by exact name for precisely this reason; the in-process path
    # cannot, because a substring match is what makes `--only q17,q72` useful. So it says so.
    if args.only:
        matched = ", ".join(c.name for c in cases)
        print(f"--only {args.only!r} matched {len(cases)} case(s): {matched}")
    print()
    results = []
    for case in cases:
        print(f"running {case.name} ...", flush=True)
        results.append(
            compare(case.name, case.build(ctx), names, runs=runs, ordered_by=case.ordered_by)
        )
    if args.isolate_case is not None:
        for result in results:
            emit_result(result)
        return results
    print()
    print(f"=== {benchmark} ({', '.join(names)}) ===")
    print_table(results, names)
    print(format_summary(summarize(results, names), runs, repeated=args.repeat > 1))
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
    if which == "cache":
        from internals import cache_bench

        return cache_bench.main()
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
    # The fingerprint, not just the engine version. `machine_fingerprint`'s own docstring
    # says it "is the first record of every result document the benchmark harnesses write",
    # and gives the reason: BENCHMARK_RESULTS.md accumulated numbers from at least four
    # machines whose ratios differ by an order of magnitude, and two rows without it
    # attached are not a comparison. Until now the *concurrency* harness was the only one
    # that printed it, so every headline suite emitted a table with no record of the box it
    # was measured on — and the hardware table in `docs/benchmarks/methodology.md` had to be
    # maintained by hand against numbers that carried no evidence of where they came from.
    #
    # Both core counts are printed because when they disagree the difference is usually the
    # whole story: `cpu_count_available` is what the cgroup grants and what the engine will
    # use, `cpu_count_logical` is what the kernel advertises.
    fp = machine_fingerprint()
    load = fp["load_per_core_at_start"]
    load_text = "load/core unmeasurable" if load is None else f"load/core {load:.2f}"
    print(
        f"Batcher benchmark suite  ({fp['engine_profile']} engine {fp['engine']}, {fp['git_sha']})"
    )
    print(f"  host {fp['host']}  ({fp['cpu_model']})")
    print(
        f"  {fp['cpu_count_available']} of {fp['cpu_count_logical']} cores available, "
        f"{fp['memory_bytes'] / (1 << 30):.0f} GiB, {load_text}"
    )
    print(f"engines: {', '.join(e.name for e in engines)}\n")
    # Before any case runs, so an engine that must own a piece of global setup can take it.
    # Ray Data is the one that does: the job-level `runtime_env` carrying `benchmarks/` to
    # its workers can only be attached by whoever calls `ray.init`, and Batcher leads this
    # lineup — see `Engine.prepare`.
    for engine in engines:
        engine.prepare()
    require_release_build(allow_debug=args.allow_debug_build)
    # `envinfo` has shipped this guard since it was written, and until now the *concurrency*
    # benchmark was its only caller — so the suite that produces every headline ratio
    # (TPC-H, TPC-DS, ClickBench, JOB, H2O, the operator mix) would record numbers on a
    # contended box without so much as a warning. That is not a smaller version of the same
    # measurement: `require_quiet_box`'s own docstring records DuckDB timing *slower* at 8
    # threads than at 1 under load 25, which is not a fact about DuckDB, and several deltas
    # in BENCHMARK_RESULTS.md are explicitly disavowed for it. A benchmark that cannot
    # refuse an unusable environment reports the neighbour's load as an engine difference.
    require_quiet_box(allow_busy=args.allow_busy_box)

    datasets = ALL_DATASETS if args.benchmark == "all" else (args.benchmark,)
    names = [e.name for e in engines]
    all_results = []
    per_run = []
    per_run_results = []
    for i in range(max(1, args.repeat)):
        if args.repeat > 1:
            print(f"--- repeat {i + 1} of {args.repeat} ---")
        run_results = []
        for ds in datasets:
            run_results += _run_dataset(ds, args, engines)
        all_results += run_results
        per_run_results.append(run_results)
        per_run.append(summarize(run_results, names))
    # The spread, not another decimal place. A geomean from one run carries no evidence
    # about its own stability, and quoting it to three decimals asserts some.
    if args.repeat > 1:
        print(format_repeats(per_run))
        # The geomean carries a spread and the rows above it do not, which is the wrong way
        # round: averaging 22 queries cancels most of the noise, a single row keeps all of it.
        print(format_unstable(per_run_results, names))

    if not all_results:
        print("no benchmarks matched the selection.")
        return 0

    # `DIVERGENT` is deliberately not in this list. It marks a row where every difference
    # is one `harness/divergences.py` has recorded, with a citation naming which engine is
    # right — three of the four recorded so far are cases where a *comparator* is the wrong
    # one. Failing the run on those would put standing pressure to change Batcher to match
    # a comparator's deviation, which is the outcome that module exists to prevent. The row
    # still never reads OK, still carries no ratio, and still prints its reason.
    failed = [r for r in all_results if r.status in ("FAILED", "ERROR", "KILLED")]
    divergent = [r for r in all_results if r.status == "DIVERGENT"]
    degenerate = [r for r in all_results if r.status == "DEGENERATE"]
    if divergent:
        print(
            f"{len(divergent)} query(ies) DIVERGENT: a recorded semantic difference, not a "
            "defect — see the notes above and harness/divergences.py."
        )
    if degenerate:
        # Not a failure: TPC-DS q17 legitimately returns zero rows at sf1. But a row where
        # every engine returned nothing must never read as a fast pass, so it is counted
        # out loud rather than folded into the OK count.
        print(
            f"{len(degenerate)} query(ies) DEGENERATE: every engine returned a result "
            "carrying no information. Check the data actually loaded before reading these."
        )
    if failed:
        print(f"{len(failed)} query(ies) FAILED correctness, errored, or died.")
        return 1
    print("All correctness checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
