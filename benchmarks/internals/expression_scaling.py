"""Does an expression cost the same per row at 1 M rows as it does at 32 M?

Every other benchmark here answers "how fast", at one size, against a competitor. This one
answers a different question that no single-size timing can: **is the cost per row flat as
the relation grows?** A shape that is 3 ns/row at 1 M and 40 ns/row at 32 M does not have a
constant-factor problem, it has a scaling problem, and the two are fixed by entirely
different work. Reporting one number at one scale hides the difference completely.

So the unit here is **ns/row at each scale**, and the verdict is the *trend* across them:

- flat or falling -> linear (falling is normal and healthy at the small end, where a fixed
  per-query cost of a millisecond or two is still amortizing);
- rising -> super-linear, and the growth factor says by how much.

The shapes are chosen to separate the parts of the expression evaluator that can each
degrade on their own: the JIT-compiled arithmetic subset, the interpreted string and
temporal kernels, the branch-evaluating forms (`CASE`, `COALESCE`), null handling, and the
group-by whose hash table stops being cache-resident somewhere in this range. Group-by
cardinality is swept *separately from row count*, because it is the cardinality and not the
row count that decides whether the table fits in a core's private cache -- sweeping only
rows would move both at once and attribute the result to the wrong axis.

The measurement is min-of-N, not mean: this is a floor, and on a shared box the mean
measures the neighbours. Run it somewhere quiet. On a contended 16-core host the same case
here read 3.6 ms and 10.5 ms in consecutive runs, so a difference under ~30 % is not
resolvable and the trend, not the absolute, is what to read.

Run:
    python benchmarks/internals/expression_scaling.py
    python benchmarks/internals/expression_scaling.py --scales 1000000,8000000,32000000
    python benchmarks/internals/expression_scaling.py --only case,agg
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from collections.abc import Callable

import numpy as np
import pyarrow as pa

import batcher as bt

#: Row counts to sweep. Spans the range where a group-by's hash table stops fitting in a
#: core's private cache, which is where the interesting behaviour is.
SCALES = (1_000_000, 4_000_000, 16_000_000)
#: Distinct group counts, as a fraction of the row count, for the group-by sweep.
GROUP_FRACTIONS = (0.000_01, 0.001, 0.1, 0.5)
_REPEATS = 3
#: ns/row growth from the smallest scale to the largest above which a case is called
#: super-linear. Generous on purpose -- see the note about resolvability in the module
#: docstring. A genuine scaling defect in this range shows up as 2x or more, not 1.4x.
SUPERLINEAR_FACTOR = 1.5


def _columns(n: int, seed: int = 0) -> dict[str, object]:
    """One table carrying every column shape the cases below need."""
    rng = np.random.default_rng(seed)
    return {
        "i64": rng.integers(0, 1_000_000, n),
        "f64": rng.random(n) * 1000.0,
        "g64": rng.random(n) * 1000.0,
        # A 16-value string key (dictionary-friendly) and a near-unique one.
        "s_lo": rng.choice([f"cat{i:02d}" for i in range(16)], n),
        "s_hi": np.array([f"id-{v}" for v in rng.integers(0, n, n)]),
        # Half null, so the null-handling paths are exercised rather than skipped.
        "nulls": pa.array(rng.random(n) * 100, mask=(rng.random(n) < 0.5)),
        "ts": pa.array(rng.integers(0, 10**9, n).astype("int64"), type=pa.timestamp("us")),
    }


def _cases() -> dict[str, Callable[[object], object]]:
    """Each case maps a dataset to an unexecuted `Dataset`; the harness collects it."""
    return {
        # --- the JIT-compiled subset: arithmetic and comparison over numerics ---
        "arith_chain": lambda d: d.select(o=(bt.col("f64") * 2.0 + bt.col("g64")) / 3.0 - 1.0),
        "cmp_chain": lambda d: d.filter(
            (bt.col("f64") > 100.0) & (bt.col("g64") < 900.0) & (bt.col("i64") % 7 == 0)
        ),
        "cast": lambda d: d.select(o=bt.col("i64").cast("float64") + 0.5),
        # --- branch-evaluating forms. Both evaluate every branch at full width today, so
        # --- the string variant pays four string kernels where one row needs one.
        "case_num4": lambda d: d.select(
            o=bt.when(bt.col("i64") < 250_000)
            .then(bt.col("f64") * 1.0)
            .when(bt.col("i64") < 500_000)
            .then(bt.col("f64") * 2.0)
            .when(bt.col("i64") < 750_000)
            .then(bt.col("f64") * 3.0)
            .otherwise(bt.col("f64") * 4.0)
        ),
        "case_str4": lambda d: d.select(
            o=bt.when(bt.col("i64") < 250_000)
            .then(bt.col("s_hi").str.to_uppercase())
            .when(bt.col("i64") < 500_000)
            .then(bt.col("s_hi").str.to_lowercase())
            .when(bt.col("i64") < 750_000)
            .then(bt.col("s_hi").str.reverse())
            .otherwise(bt.col("s_hi"))
        ),
        "coalesce4": lambda d: d.select(
            o=bt.coalesce(bt.col("nulls"), bt.col("f64") * 2.0, bt.col("g64") * 3.0, bt.lit(0.0))
        ),
        # --- interpreted kernels: strings and temporal ---
        "str_upper": lambda d: d.select(o=bt.col("s_lo").str.to_uppercase()),
        "str_contains": lambda d: d.filter(bt.col("s_hi").str.contains("7")),
        "dt_extract": lambda d: d.select(o=bt.col("ts").dt.year() + bt.col("ts").dt.month()),
        # --- a repeated subexpression, which Kyber's CSE rule should evaluate once ---
        "cse4": lambda d: d.select(
            a=(bt.col("f64") * bt.col("g64")) + 1.0,
            b=(bt.col("f64") * bt.col("g64")) + 2.0,
            c=(bt.col("f64") * bt.col("g64")) + 3.0,
            d=(bt.col("f64") * bt.col("g64")) + 4.0,
        ),
        # --- grouping by a string key, which hashes bytes rather than a native value ---
        "agg_str": lambda d: d.group_by("s_lo").agg(s=bt.col("f64").sum()),
    }


def _best_ms(fn: Callable[[], object], repeats: int = _REPEATS) -> float:
    """Best-of-N wall time in ms. A floor, so the neighbours on a shared box can only
    make it worse and never better -- which is what makes a *rise* across scales real."""
    fn()  # warm: first run pays JIT compile and any one-time lookup
    best = float("inf")
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def _trend(per_row: list[float]) -> str:
    """Verdict on a case's ns/row across the scales.

    Measured from the **cheapest** scale to the largest, not from the first. Every case here
    is U-shaped: a fixed per-query cost of a millisecond or two dominates at the small end
    and amortizes away, and only after that does any real scaling defect start to show. A
    first-to-last reading adds those two opposite movements together and cancels them --
    group-by over 0.5 x rows distinct groups runs 26.7 -> 22.3 -> 38.3 ns/row, which reads as
    a harmless 1.4x end to end while the actual degradation, once the fixed cost is gone, is
    22.3 -> 38.3. The minimum is where amortizing stops, so it is the honest baseline for
    what happens afterwards.
    """
    known = [p for p in per_row if p == p and p > 0]
    if len(known) < 2:
        return ""
    floor = min(known)
    growth = known[-1] / floor
    if growth >= SUPERLINEAR_FACTOR:
        return f"SUPER-LINEAR {growth:.1f}x (from {floor:.1f} ns/row)"
    amortized = known[0] / known[-1]
    return f"amortizing {amortized:.1f}x" if amortized >= 1.1 else "linear"


def _run_expressions(scales: tuple[int, ...], only: set[str] | None) -> int:
    cases = _cases()
    if only:
        cases = {k: v for k, v in cases.items() if any(o in k for o in only)}
    if not cases:
        return 0
    print("\n=== per-row cost of an expression, by relation size ===")
    print("(ns/row; flat or falling is linear, rising is a scaling defect)\n")
    data = {n: bt.from_pydict(_columns(n)) for n in scales}
    header = "case".ljust(14) + "".join(f"{n // 1_000_000:>10}M" for n in scales) + "   verdict"
    print(header)
    print("-" * len(header))
    flagged = 0
    for name, build in cases.items():
        per_row: list[float] = []
        row = name.ljust(14)
        for n in scales:
            try:
                ms = _best_ms(lambda n=n, build=build: build(data[n]).collect())
            # A broken case must not stop the sweep.
            except Exception as exc:
                row += f"{type(exc).__name__[:10]:>11}"
                per_row.append(float("nan"))
                continue
            per_row.append(ms * 1e6 / n)
            row += f"{per_row[-1]:>11.1f}"
        verdict = _trend(per_row)
        flagged += verdict.startswith("SUPER")
        print(f"{row}   {verdict}", flush=True)
    return flagged


def _run_group_by(scales: tuple[int, ...]) -> int:
    """Group-by swept over cardinality *and* rows, because only cardinality decides
    whether the hash table stays in cache and only rows decide how often it is probed."""
    print("\n=== per-row cost of GROUP BY, by relation size and distinct groups ===")
    print("(ns/row; the cardinality axis is the one that leaves cache)\n")
    header = "groups/rows".ljust(14) + "".join(f"{n // 1_000_000:>10}M" for n in scales)
    print(header + "   verdict")
    print("-" * (len(header) + 11))
    flagged = 0
    for frac in GROUP_FRACTIONS:
        per_row: list[float] = []
        row = f"{frac:<14g}"
        for n in scales:
            groups = max(2, int(n * frac))
            rng = np.random.default_rng(7)
            ds = bt.from_pydict({"k": rng.integers(0, groups, n), "v": rng.random(n)})
            try:
                ms = _best_ms(lambda ds=ds: ds.group_by("k").agg(s=bt.col("v").sum()).collect())
            except Exception as exc:
                row += f"{type(exc).__name__[:10]:>11}"
                per_row.append(float("nan"))
                continue
            per_row.append(ms * 1e6 / n)
            row += f"{per_row[-1]:>11.1f}"
        verdict = _trend(per_row)
        flagged += verdict.startswith("SUPER")
        print(f"{row}   {verdict}", flush=True)
    return flagged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scales",
        default=",".join(str(s) for s in SCALES),
        help="comma-separated row counts to sweep",
    )
    parser.add_argument("--only", default="", help="comma-separated substrings of case names")
    args = parser.parse_args()
    scales = tuple(int(s) for s in args.scales.split(","))
    only = {o.strip() for o in args.only.split(",") if o.strip()} or None

    print(f"engine profile: {bt.versions().get('engine_profile')}")
    if bt.versions().get("engine_profile") != "release":
        print("WARNING: not a release engine — these numbers measure a debug build.")

    flagged = _run_expressions(scales, only)
    if not only or any("agg" in o or "group" in o for o in only):
        flagged += _run_group_by(scales)

    print(f"\n{flagged} shape(s) flagged super-linear.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
