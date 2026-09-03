"""Strong-scaling ladder: does N times the cluster give N times the throughput?

Every other benchmark here answers "is Batcher faster than DuckDB on this box". This one
answers the question a distributed engine is actually bought for: **hold the data fixed,
multiply the machines, and see whether the wall time divides.** The single-node suites can
be won by a fast kernel; this can only be won by the scheduling, the shuffle and the scan
assignment all staying out of the way as the fleet grows.

The measurement is a ladder. One query shape is run at each rung of a worker-count
sequence (``1,2,4`` by default, one worker per node under SPREAD, so the rungs are
1x/2x/4x the cores), and the rung-1 time is the baseline every later rung is divided by.
An **efficiency** column reports that speedup against the ideal, so a rung that adds
hardware and buys nothing is visible as a number rather than inferred from two timings.

A ``driver`` row is reported beside the ladder: the same query run single-node, in this
process, on the driver's own cores. It is the figure `BENCHMARK_RESULTS.md` records for a
single large box, and it is what "distributed is worth it above N nodes" is measured
against. It is **not** the ladder's baseline — the driver and a worker are different
machines with different page caches and different distances to the data, and using one as
the denominator for the other would charge the fleet for that difference.

Correctness is gated the way every benchmark here gates it: the mergeable algebra says
every rung must produce the same rows as the driver, so a rung that disagrees is reported
FAILED and its timing is not quoted. That is the single-node == distributed invariant
(`CLAUDE.md` #7), measured on real hardware rather than asserted.

Each case runs in **its own process** by default, because a session that has already run two
other queries does not measure the same thing (see :func:`_one_process_per_case` — it is worth
a factor of two). `--in-process` opts out.

Run:
    python benchmarks/scenarios/scaling/ladder.py                      # sf100, all cases
    python benchmarks/scenarios/scaling/ladder.py --rungs 1,2,4 --only scan-agg
    python benchmarks/scenarios/scaling/ladder.py --scale 10 --source /mnt/.../tpch_named_sf10
    python benchmarks/scenarios/scaling/ladder.py --no-driver         # ladder only, faster
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import pyarrow as pa

# Resolved from this file rather than the working directory, for the reason the sibling
# scenarios spell out: the cwd-relative form only imports when launched from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from envinfo import machine_fingerprint, require_quiet_box, require_release_build
from harness import order_violation, results_match
from sources import scan_rename, table_uris

#: A cluster-visible TPC-H mirror, preferred when it exists. Any base holding
#: ``{base}/{table}/*.parquet`` works; positional column names are renamed on read.
#:
#: Every worker must be able to read the source, and it must not be the *driver's* disk: the
#: point of the ladder is that a rung's readers are on N different machines. A shared mount or
#: an object store both qualify; `/mnt/cluster_storage` is the shared mount the other cluster
#: benchmarks here already use.
_MIRROR = "/mnt/cluster_storage/tpch_sf100"
DEFAULT_SCALE = 100


def default_source() -> str:
    """The mirror when this box has one, else the public S3 base every other suite reads.

    Not a constant, because a hard-coded absolute path is a benchmark that runs on exactly one
    machine — the failure `scenarios/scale_bench.py` records in its own comments. The S3
    fallback is correct everywhere and slower everywhere, so the mirror wins when it is there.
    """
    from pathlib import Path as _Path

    from sources import TPCH_BASE

    return _MIRROR if _Path(_MIRROR).is_dir() else TPCH_BASE


#: Where the engine package is shipped from, so a worker imports the tree under test rather
#: than whatever is installed on the node image. Derived, never a workspace-specific path.
_PKG = str(Path(__file__).resolve().parents[3] / "python" / "batcher")


def _bind(uris: dict[str, str], renames: dict[str, dict[str, str]], name: str, cols: list[str]):
    """A lazy scan of `name` projected to `cols`, under canonical column names.

    The projection is applied as part of the bind rather than left to the optimizer, so
    every rung reads the same bytes and the ladder measures scheduling rather than how well
    pushdown happened to fire at that fan-out.
    """
    import batcher as bt

    ren = renames.get(name, {})
    back = {v: k for k, v in ren.items()}  # canonical -> positional, {} when already named
    ds = bt.read.parquet(uris[name])
    return ds.select(*[bt.col(back.get(c, c)).alias(c) for c in cols])


def cases(uris: dict[str, str], renames: dict[str, dict[str, str]]) -> dict:
    """The query shapes the ladder runs, one per mergeable operator family.

    Each returns a **small** result on purpose. A rung's timing must measure the fleet, and a
    query that ships 150M rows back to the driver measures the driver's single-threaded
    collect instead — which cannot scale with the cluster and so flattens every ladder it
    appears in, whatever the engine does.
    """
    import batcher as bt
    from batcher import col

    def li(*cols):
        return _bind(uris, renames, "lineitem", list(cols))

    def orders(*cols):
        return _bind(uris, renames, "orders", list(cols))

    def scan_agg():
        """TPC-H q1's shape: scan, filter, two-key group-by, five aggregates."""
        net = col("l_extendedprice") * (1 - col("l_discount"))
        return (
            li(
                "l_quantity",
                "l_extendedprice",
                "l_discount",
                "l_tax",
                "l_returnflag",
                "l_linestatus",
                "l_shipdate",
            )
            .filter(col("l_shipdate") <= bt.lit("1998-09-01").cast("date32"))
            .group_by("l_returnflag", "l_linestatus")
            .agg(
                qty=col("l_quantity").sum(),
                base=col("l_extendedprice").sum(),
                disc=net.sum(),
                charge=(net * (1 + col("l_tax"))).sum(),
                n=col("l_quantity").count(),
            )
        )

    def groupby_high_card():
        """150M groups: the shuffle is the whole query, folded back to one row."""
        return (
            li("l_orderkey", "l_extendedprice")
            .group_by("l_orderkey")
            .agg(s=col("l_extendedprice").sum())
            .agg(groups=bt.count(), total=col("s").sum())
        )

    def hash_join():
        """lineitem ⋈ orders on the order key, reduced to a per-priority total."""
        return (
            li("l_orderkey", "l_extendedprice", "l_discount")
            .join(
                orders("o_orderkey", "o_orderpriority"),
                left_on="l_orderkey",
                right_on="o_orderkey",
                how="inner",
            )
            .group_by("o_orderpriority")
            .agg(rev=(col("l_extendedprice") * (1 - col("l_discount"))).sum())
        )

    def sort_topn():
        """A global order over the whole relation, cut to 100 rows.

        Only the sort key is projected. `l_extendedprice` has ties, and which *row* supplies a
        tied value is not determined by the query — so carrying a second column would make a
        legitimate tie-break difference between rungs read as a divergence. The 100 values
        themselves are determined, and their order is checked directly by `order_violation`.
        """
        return li("l_extendedprice").sort("l_extendedprice", descending=True).limit(100)

    def distinct_count():
        """Distinct over a 20M-value key, counted."""
        return li("l_partkey").distinct().agg(n=bt.count())

    def window_dedup():
        """row_number() over a partitioned, ordered frame, kept at rank 1 and counted."""
        return (
            li("l_orderkey", "l_linenumber", "l_extendedprice")
            .with_columns(
                r=bt.row_number().over(partition_by="l_orderkey", order_by="l_linenumber")
            )
            .filter(col("r") == 1)
            .agg(n=bt.count(), t=col("l_extendedprice").sum())
        )

    return {
        "scan-agg": scan_agg,
        "groupby-high-card": groupby_high_card,
        "hash-join": hash_join,
        "sort-topn": sort_topn,
        "distinct": distinct_count,
        "window-dedup": window_dedup,
    }


#: Cases whose output order is part of the answer. `results_match` sorts both sides by design,
#: so it would sort away the very property a sort case exists to test — the shape `lint-tests`
#: calls a check that cannot fail. Each rung's own table is held to this order separately.
ORDERED: dict[str, list] = {"sort-topn": [("l_extendedprice", True)]}


def _time(fn, runs: int) -> tuple[float, pa.Table]:
    """Best-of-`runs` after one untimed warm-up, which also spawns the fleet for the rung."""
    out = fn()
    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best, out


def _drop_session_fleet() -> None:
    """Tear the warm session fleet down, so the next rung gets the width it asked for.

    Without this the ladder measures nothing. The session fleet is reused across `collect()`
    calls and is respawned only when it is **too narrow** for the request, so a `num_workers=1`
    query arriving after a 4-worker one silently borrows all four actors and reports a
    1-worker timing for a 4-worker run. That is correct behaviour for a session — a wider
    fleet is never wrong, only warmer — and fatal for a benchmark that varies the width
    downward inside one process.

    It is also exactly the shape of measurement error this repo keeps finding: the first
    version of this ladder ran its cases in order, so case 1 (`scan-agg`, whose rungs really
    did resize upward) reported a clean 3.25x while every later case reported a flat line —
    and a flat line beside a clean one reads as a property of those operators rather than of
    the harness. It is not skipped when reuse is off; `release_session_fleet` is a no-op with
    no fleet cached, and refuses outright while one is leased.
    """
    from batcher.dist.fleet import release_session_fleet

    release_session_fleet()


def _configure(address: str) -> None:
    """Point the distributed executor at the live cluster and ship this tree to it."""
    from batcher.config import active_config, set_config

    cfg = active_config()
    set_config(
        cfg.replace(
            distributed=dataclasses.replace(
                cfg.distributed,
                ray_address=address,
                runtime_env={
                    "py_modules": [_PKG],
                    "env_vars": {"AWS_DEFAULT_REGION": "us-west-2", "AWS_REGION": "us-west-2"},
                },
            )
        )
    )


def _rung(w) -> str:
    """The row label for a rung: `w4` for a pinned width, `auto` for the engine's own."""
    return "auto" if w == "auto" else f"w{w}"


def run_case(name, build, rungs, runs, want_driver):
    """Run one case down the ladder, returning `{rung -> ms}` and the correctness verdicts."""
    times: dict[str, float] = {}
    verdicts: dict[str, str] = {}
    reference = None
    if want_driver:
        try:
            # `distributed=False`, never the default: `collect()`'s default is `"auto"`, which
            # resolves to the cluster the moment Ray is initialized — and the first rung
            # initializes it. Left implicit, every case after the first reported a
            # *distributed* run in the `driver` row and the ladder compared the fleet against
            # itself.
            ms, out = _time(lambda: build().collect(distributed=False), runs)
        except Exception as exc:  # a rung that cannot run is reported, never silently dropped
            verdicts["driver"] = f"FAILED {type(exc).__name__}: {exc}"
        else:
            # The reference is held to the case's own order before anything is compared to
            # it: an unordered reference would make every rung agree with it and the ladder
            # would report a passing sort that nothing had checked.
            disorder = order_violation(out, ORDERED.get(name, []))
            verdicts["driver"] = "OK" if not disorder else f"UNORDERED {disorder}"
            times["driver"], reference = ms, out
    for w in [*rungs, "auto"]:
        _drop_session_fleet()
        # `auto` is the fan-out the engine chooses for this cluster, which is what a user
        # gets and what a "cluster vs one node" claim has to be made against. Every numbered
        # rung passes `num_workers`, and that argument takes a *different* branch of the
        # executor's sizing — so a ladder built only from numbered rungs can be entirely
        # healthy while the default path is not. It was: the default fan-out ran this join in
        # 25.3s against 10.4s for the same width requested explicitly.
        pick = {} if w == "auto" else {"num_workers": w}
        try:
            ms, out = _time(lambda k=pick: build().collect(distributed=True, **k), runs)
        except Exception as exc:
            verdicts[_rung(w)] = f"FAILED {type(exc).__name__}: {exc}"
            continue
        if reference is None:
            reference = out
            times[_rung(w)] = ms
            continue
        ok, msg = results_match(reference, out)
        disorder = order_violation(out, ORDERED.get(name, []))
        if ok and disorder:
            ok, msg = False, f"rows match but the result is not ordered: {disorder}"
        verdicts[_rung(w)] = "OK" if ok else f"MISMATCH {msg}"
        if ok:
            times[_rung(w)] = ms
    return times, verdicts


def _report(name, times, verdicts, rungs) -> None:
    """One case's row block: ms, speedup over the first rung, and efficiency against ideal."""
    base_key = f"w{rungs[0]}"
    base = times.get(base_key)
    print(f"\n{name}")
    print(f"  {'rung':<10}{'ms':>10}{'vs w' + str(rungs[0]):>10}{'efficiency':>12}   {'check':<10}")
    for key in ["driver", *[f"w{w}" for w in rungs], "auto"]:
        if key not in times:
            if key in verdicts:
                print(f"  {key:<10}{'-':>10}{'-':>10}{'-':>12}   {verdicts[key]}")
            continue
        ms = times[key]
        if base and key.startswith("w"):
            speedup = base / ms
            width = int(key[1:]) / rungs[0]
            eff = f"{100 * speedup / width:.0f}%"
            print(f"  {key:<10}{ms:>10.0f}{speedup:>9.2f}x{eff:>12}   {verdicts.get(key, ''):<10}")
        elif key == "auto" and base:
            print(
                f"  {key:<10}{ms:>10.0f}{base / ms:>9.2f}x{'-':>12}   {verdicts.get(key, ''):<10}"
            )
        else:
            print(f"  {key:<10}{ms:>10.0f}{'-':>10}{'-':>12}   {verdicts.get(key, ''):<10}")


def _one_process_per_case(args, names: list[str]) -> None:
    """Re-run this script once per case, so no case inherits another's session.

    A ladder measures an operator; a process that has already run two other queries measures
    the *session*. Both matter and they are not the same number: `hash-join` at sf100 scales
    **3.14x on four workers alone in a fresh process and 1.68x** when `scan-agg` and
    `groupby-high-card` ran before it — same query, same data, same widths, same box. Whatever
    carries across (the `MetadataHub`'s learned routes are the obvious candidate; this has not
    been isolated) halves the scale-out, so a multi-case run in one process silently reports
    the wrong figure for every case after the first.

    That is the second measurement trap in this one file — the session fleet was the first —
    and both have the same shape: state that is *correct* for a session and *fatal* for a
    benchmark that varies one thing. So the isolation is the default rather than a flag anyone
    has to know to pass. `--in-process` opts out, for measuring the session on purpose.
    """
    import subprocess
    import sys

    for name in names:
        argv = [sys.executable, __file__, "--in-process", "--only", name]
        for flag, value in (
            ("--scale", str(args.scale)),
            ("--rungs", args.rungs),
            ("--runs", str(args.runs)),
            ("--address", args.address),
        ):
            argv += [flag, value]
        if args.source:
            argv += ["--source", args.source]
        if args.no_driver:
            argv.append("--no-driver")
        subprocess.run(argv, check=False)


def main() -> None:
    require_release_build()
    print(machine_fingerprint())
    require_quiet_box()
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=DEFAULT_SCALE)
    ap.add_argument("--source", default="", help="parquet base; default: see default_source()")
    ap.add_argument("--rungs", default="1,2,4", help="worker counts, comma separated")
    ap.add_argument("--runs", type=int, default=2, help="timed runs per rung (best-of)")
    ap.add_argument("--only", default="", help="comma-separated case-name substrings")
    ap.add_argument("--no-driver", action="store_true", help="skip the single-node reference row")
    ap.add_argument("--address", default="auto")
    ap.add_argument(
        "--in-process",
        action="store_true",
        help="run every selected case in THIS process — measures the session, not the operator",
    )
    args = ap.parse_args()

    _configure(args.address)
    rungs = [int(x) for x in args.rungs.split(",")]
    source = args.source or default_source()
    uris = table_uris("tpch", args.scale, source)
    renames = scan_rename("tpch", uris)
    built = cases(uris, renames)
    picked = {
        k: v for k, v in built.items() if not args.only or any(s in k for s in args.only.split(","))
    }
    if not args.in_process and len(picked) > 1:
        _one_process_per_case(args, list(picked))
        return
    print(f"source={source}  scale=sf{args.scale}  rungs={rungs}  runs={args.runs}")

    for name, build in picked.items():
        times, verdicts = run_case(name, build, rungs, args.runs, not args.no_driver)
        _report(name, times, verdicts, rungs)


if __name__ == "__main__":
    main()
