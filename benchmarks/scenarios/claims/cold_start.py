"""Process cold start: what the first query in a fresh process costs.

Every other suite here reports a **warm** number. `harness.compare` calls each engine once
untimed and then takes best-of-N, and `latency_bench` explicitly discards its first calls
because they "pay import, JIT warm-up and a cold plan". That is the right measurement for
throughput, and for a long-lived process it is the only honest one — nobody cares what the
thousandth query's predecessor cost.

It is the wrong measurement for the claim the project actually makes about small queries.
`.claude/rules/performance.md` says: *"Small queries: sub-second, low fixed overhead. Don't
add per-query setup cost (spinning thread pools, compiling unconditionally, allocating
large buffers) that hurts the small case to help the large one."* A warm-up followed by
best-of-N cannot see that cost at all — it is designed not to. So the one number that would
falsify the claim is the one number nothing recorded.

And the amount hidden is not symmetric across engines. Measured in-process on this box, a
2M-row filter + group-by:

    batcher   390.0 ms on the first call, then ~8.4 ms  (46x)
    duckdb     77.9 ms on the first call, then ~51.2 ms (1.5x)

So the shared "one warm-up" convention forgives Batcher ~380 ms and DuckDB ~27 ms. Whether
that is fair depends entirely on the workload, which is exactly why it should be a published
number rather than a convention buried in `bench()`.

It is **not** a result cache, and the warm number is not an illusion. Varying the filter
literal on every call — which defeats a plan cache keyed on `LogicalPlan.content_key()`,
since that includes literal values — leaves the steady state at ~7 ms. Batcher really is
several times faster than DuckDB on that shape once warm. It is also several times slower to
*get* there, and only one of those two facts was being reported.

What this script measured the first time it ran (96-core box, load 0.07/core, release
engine, 100,000 rows, min over 5 fresh processes):

    engine       import ms    first ms    total ms   vs batcher
    duckdb            59.2        22.3        81.5       10.98x
    polars           104.0        44.2       148.9        6.01x
    batcher          544.6       348.6       894.9            —

Two things to take from that. **Batcher is ~11x DuckDB to first answer**, which no warm
suite could show. And the larger half is the *import*, not the query: `python -X importtime`
attributes 575 ms to `import batcher`, of which **559 ms is `batcher.api`** — the control
plane eagerly pulling in `session`, `functions`, `dataset.frame`, `terminal.core`,
`merge.builder` and `orchestration` before the user has expressed any intent at all. For
comparison the compiled engine is cheap: `batcher._native` adds about 21 ms on top of the
Python package, and `pyarrow` accounts for 155 ms of the total.

That decomposition is why this belongs in `benchmarks/` rather than in a note somewhere: it
is a measurable, attributable, and therefore fixable number, and it stayed invisible for as
long as every suite began by throwing the first run away.

## What is measured

One **subprocess per sample**, because that is the only way to get a genuinely cold
interpreter: within a process, the thread pool, the allocator arenas, the JIT and the
imported modules all stay warm, and the second measurement of "cold start" in the same
process is not cold. The child reports three timings from its own clock:

``import``
    Wall time to import the engine's module. For a serverless invocation or a CLI run this
    is paid before any user code runs at all.
``first``
    Wall time from a loaded module to the first answer, including whatever the engine
    builds lazily on first use.
``total``
    The two together, which is what the user actually waits for.

Each is reported as the **minimum** over the samples: process start-up contends with
whatever else is on the box, and contention can only ever add time.

The result is verified against DuckDB before any of it is reported — a cold path that
returns the wrong answer is not a fast cold path.

Run:
    python benchmarks/scenarios/claims/cold_start.py
    python benchmarks/scenarios/claims/cold_start.py --samples 10 --rows 200000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# `parents[2]`, matching every sibling in this package: the shared benchmark modules
# live in `benchmarks/`, which is two levels up from `scenarios/claims/`. This read
# `parent.parent` while the file sat one level higher; the count is a function of the
# file's depth, so it moves when the file does.
_BENCH_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BENCH_ROOT))

from envinfo import machine_fingerprint, require_quiet_box, require_release_build  # noqa: E402
from harness.compare import _ORACLE_PREFERENCE  # noqa: E402

#: The query every engine answers. Deliberately tiny: this measures fixed cost, so the
#: compute must not be large enough to show up beside it. It still has a filter, a grouped
#: aggregate and a sort, so no engine can answer it without building a real pipeline.
_QUESTION = "rows where v > 2, grouped by k, summed, ordered by k"

#: The child program, one per engine. Each prints a single JSON line so the parent needs no
#: parsing rules, and each times *its own* import rather than being timed from outside —
#: measuring from the parent would fold in interpreter start-up, which is Python's cost and
#: not the engine's, and would be counted identically for every engine anyway.
#:
#: The row count arrives as ``sys.argv[1]`` rather than being interpolated into the source.
#: Templating it in means escaping every ``%`` and ``{`` the program contains, which is a
#: silent source of syntax errors in a string nothing type-checks — the first draft of this
#: file shipped exactly that and every child died with `SyntaxError` on a modulo operator.
_CHILDREN: dict[str, str] = {
    "batcher": """
import json, sys, time
rows = int(sys.argv[1])
t0 = time.perf_counter()
import batcher as bt
t1 = time.perf_counter()
data = {"k": [i % 7 for i in range(rows)], "v": [i % 5 for i in range(rows)]}
out = (
    bt.from_pydict(data)
    .filter(bt.col("v") > 2)
    .group_by("k")
    .agg(s=bt.col("v").sum())
    .sort("k")
    .to_pydict()
)
t2 = time.perf_counter()
print(json.dumps({"import": t1 - t0, "first": t2 - t1, "answer": [out["k"], out["s"]]}))
""",
    "duckdb": """
import json, sys, time
rows = int(sys.argv[1])
t0 = time.perf_counter()
import duckdb
t1 = time.perf_counter()
con = duckdb.connect()
con.execute(
    "CREATE TABLE t AS SELECT range % 7 AS k, range % 5 AS v FROM range(" + str(rows) + ")"
)
out = con.execute(
    "SELECT k, sum(v) AS s FROM t WHERE v > 2 GROUP BY k ORDER BY k"
).fetchall()
t2 = time.perf_counter()
print(json.dumps({
    "import": t1 - t0, "first": t2 - t1,
    "answer": [[r[0] for r in out], [r[1] for r in out]],
}))
""",
    "polars": """
import json, sys, time
rows = int(sys.argv[1])
t0 = time.perf_counter()
import polars as pl
t1 = time.perf_counter()
df = pl.DataFrame({"k": [i % 7 for i in range(rows)], "v": [i % 5 for i in range(rows)]})
out = (
    df.lazy()
    .filter(pl.col("v") > 2)
    .group_by("k")
    .agg(pl.col("v").sum().alias("s"))
    .sort("k")
    .collect()
)
t2 = time.perf_counter()
print(json.dumps({
    "import": t1 - t0, "first": t2 - t1,
    "answer": [out["k"].to_list(), out["s"].to_list()],
}))
""",
}


def _one_sample(engine: str, rows: int) -> dict | None:
    """Run one child process and return its timings, or None when the engine is absent."""
    finished = subprocess.run(
        [sys.executable, "-c", _CHILDREN[engine], str(rows)],
        capture_output=True,
        text=True,
        check=False,
    )
    if finished.returncode != 0:
        return None
    return json.loads(finished.stdout.strip().splitlines()[-1])


def measure(engine: str, rows: int, samples: int) -> dict | None:
    """Minimum import / first-answer / total time over `samples` fresh processes.

    Returns None when the engine is not installed here, so a missing comparator reports as
    absent rather than as a failure.
    """
    results = [_one_sample(engine, rows) for _ in range(samples)]
    produced = [r for r in results if r is not None]
    if not produced:
        return None
    best = {
        "import": min(r["import"] for r in produced),
        "first": min(r["first"] for r in produced),
        "answer": produced[0]["answer"],
    }
    # The minimum of a sum is not the sum of the minima, and what a user waits for is one
    # process's total — so take the best *total*, not the two best halves added together.
    best["total"] = min(r["import"] + r["first"] for r in produced)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000, help="rows in the tiny table")
    parser.add_argument("--samples", type=int, default=5, help="fresh processes per engine")
    parser.add_argument(
        "--allow-busy-box", action="store_true", help="measure on a contended machine anyway"
    )
    parser.add_argument(
        "--allow-debug-build", action="store_true", help="measure an unoptimized engine anyway"
    )
    args = parser.parse_args()

    require_release_build(allow_debug=args.allow_debug_build)
    # Process start-up is *more* sensitive to a busy box than steady-state throughput is,
    # not less: it is dominated by page faults, dynamic linking and thread-pool creation,
    # all of which queue behind a neighbour's load.
    require_quiet_box(allow_busy=args.allow_busy_box)

    print(machine_fingerprint())
    print(f"\ncold start: import -> first answer, {args.rows:,} rows, {_QUESTION}")
    print(f"min over {args.samples} fresh processes per engine\n")

    measured = {name: measure(name, args.rows, args.samples) for name in _CHILDREN}
    available = {name: value for name, value in measured.items() if value is not None}
    if "batcher" not in available:
        print("batcher failed to produce a result; nothing to report")
        return 1

    # Correctness before timing, the same contract `harness.compare` enforces — and the
    # oracle comes from `harness`'s own preference order rather than being named here, so
    # this file cannot become a second opinion about which engine is trustworthy. The rule
    # that order encodes is that Batcher is never its own oracle: a comparator's bug then
    # gets reported as Batcher's, which is how "Daft computes q6 wrong" was once recorded
    # against Batcher.
    oracle_name = next((n for n in _ORACLE_PREFERENCE if n in available), None)
    oracle = available[oracle_name] if oracle_name else None
    if oracle is not None:
        for name, value in available.items():
            if value["answer"] != oracle["answer"]:
                print(f"FAILED: {name} disagrees with {oracle_name} — no timing is reported")
                print(f"  {name}: {value['answer']}\n  {oracle_name}: {oracle['answer']}")
                return 1
    else:
        print("NOTE: no oracle engine is installed here, so no result was verified.\n")

    header = f"{'engine':10}  {'import ms':>10}  {'first ms':>10}  {'total ms':>10}"
    print(f"{header}  {'vs batcher':>10}")
    print("-" * 58)
    batcher_total = available["batcher"]["total"]
    for name, value in sorted(available.items(), key=lambda kv: kv[1]["total"]):
        ratio = "—" if name == "batcher" else f"{batcher_total / value['total']:.2f}x"
        print(
            f"{name:10}  {value['import'] * 1000:>10.1f}  {value['first'] * 1000:>10.1f}"
            f"  {value['total'] * 1000:>10.1f}  {ratio:>10}"
        )

    print(
        "\n'vs batcher' is batcher_total / engine_total, so above 1.00 means Batcher is\n"
        "slower to first answer. This is the number the warm suites amortize away, and the\n"
        "one a CLI invocation, a serverless call or a short script actually pays."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
