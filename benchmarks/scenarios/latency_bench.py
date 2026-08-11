"""Small-query latency: the per-query fixed cost a transactional workload pays.

Every other suite here measures **throughput** — one large query, timed once, compared
across engines. That measurement is blind to the thing an OLTP-shaped workload is made of:
thousands of tiny queries, where the answer is a handful of rows and almost all of the
elapsed time is control plane. `competitive_architecture.md` ceiling 8 records the gap this
script exists to track, and records that it is *not* the optimizer -- a warm plan-cache hit
plans in 0.002 ms, so what remains is FFI, Arrow hand-off and orchestration.

Two shapes are timed separately, because they fail differently:

`repeated`
    The identical query, over and over. Hits `kyber.plan_cache` (and, through
    `Session.sql`, the parsed-AST cache), so it measures the irreducible per-query floor.
`parameterized`
    The same query shape with a different literal each time -- `WHERE id = 41`, then
    `42`. This is the defining shape of a transactional workload and the worst case for a
    plan cache keyed on the lowered IR: `LogicalPlan.content_key()` includes literal
    values, so every distinct parameter misses and pays a full optimize.

The gap between the two is the prize a prepared-statement API would collect.

**SQLite is here as the transactional reference**, not as a competitor Batcher claims to
beat. It answers a primary-key lookup from a B-tree in microseconds; Batcher has no index
and scans (see the audit's Gap 1.2). Reporting it keeps the comparison honest about what
"low latency transactional" costs elsewhere, rather than only comparing analytics engines
to each other.

Percentiles, not means: a mean hides the tail, and the tail is what a latency SLO is
written against. CPU time is reported alongside wall time because this box is routinely
shared -- wall p50 moves by 2-3x under another session's test run while CPU p50 barely
moves, so CPU is the number to trust when the machine is busy and the one to quote when
comparing two builds of Batcher.

Run:
    python benchmarks/scenarios/latency_bench.py
    python benchmarks/scenarios/latency_bench.py --rows 1000000 --iterations 500
    python benchmarks/scenarios/latency_bench.py --engines batcher,duckdb
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Shapes are deliberately tiny. A query whose *execution* is milliseconds would bury the
#: fixed cost this script is trying to see.
DEFAULT_ROWS = 100_000
DEFAULT_ITERATIONS = 300
#: Discarded before timing starts: the first calls pay import, JIT warm-up and a cold plan
#: cache, none of which a steady-state latency figure should carry.
WARMUP = 30


@dataclass(frozen=True)
class Timing:
    """One engine's measured latency for one shape, in milliseconds."""

    wall_p50: float
    wall_p99: float
    cpu_p50: float

    @classmethod
    def measure(cls, run, iterations: int) -> Timing:
        """Time `run(i)` over `iterations` calls, after `WARMUP` untimed ones.

        `run` takes the iteration index so a parameterized shape can vary its literal;
        a repeated shape simply ignores it.
        """
        for i in range(WARMUP):
            run(i)
        wall: list[float] = []
        cpu: list[float] = []
        for i in range(iterations):
            w0, c0 = time.perf_counter(), time.process_time()
            run(i)
            wall.append((time.perf_counter() - w0) * 1000.0)
            cpu.append((time.process_time() - c0) * 1000.0)
        wall.sort()
        return cls(
            wall_p50=statistics.median(wall),
            wall_p99=wall[min(len(wall) - 1, int(len(wall) * 0.99))],
            cpu_p50=statistics.median(cpu),
        )


def _rows(n: int) -> dict[str, list]:
    """The one table every engine is given, as plain Python columns.

    `id` is unique and ascending so a point lookup selects exactly one row; `grp` has low
    cardinality so the aggregate shape has something to group by.
    """
    return {
        "id": list(range(n)),
        "grp": [i % 100 for i in range(n)],
        "v": [float(i) * 1.5 for i in range(n)],
    }


def _normalized(table) -> list[tuple]:
    """A result as sorted plain tuples, so two engines' answers can be compared.

    Sorted because this script never times an ordered query, and floats are rounded: an
    engine summing in a different order is not a wrong answer (see the reassociation
    clause in `.claude/rules/python-control-plane.md`).

    Defensive against a `RecordBatchReader`, though the engines below are all written to
    return a materialized `Table`: handing back an undrained reader would stop the timed
    call short of producing the result and post a number for work it had not done.
    """
    if isinstance(table, pa.RecordBatchReader):
        table = table.read_all()
    rows = []
    for row in table.to_pylist():
        rows.append(
            tuple(round(v, 6) if isinstance(v, float) else v for _, v in sorted(row.items()))
        )
    return sorted(rows, key=repr)


class BatcherEngine:
    """Batcher through both surfaces a caller actually uses: DataFrame and SQL."""

    name = "batcher"

    def __init__(self, data: dict[str, list]) -> None:
        import batcher as bt

        self._bt = bt
        self._session = bt.Session()
        self._table = self._session.register("t", bt.from_pydict(data))

    def point_lookup(self, key: int) -> pa.Table:
        return self._table.filter(self._bt.col("id") == key).collect()

    def filter_project(self, key: int) -> pa.Table:
        return self._table.filter(self._bt.col("id") > key).select("id", "v").limit(10).collect()

    def aggregate(self, key: int) -> pa.Table:
        return (
            self._table.filter(self._bt.col("id") > key)
            .group_by("grp")
            .agg(total=self._bt.col("v").sum())
            .collect()
        )

    def sql_point_lookup(self, key: int) -> pa.Table:
        return self._session.sql(f"SELECT * FROM t WHERE id = {key}").collect()


class DuckDBEngine:
    """DuckDB over its own native storage -- how every published DuckDB result runs it.

    Deliberately not an Arrow-registered view: that strips its zone maps and puts the scan
    inside the timed region, which is the measurement error
    `competitive_architecture.md` retires as claim 2.
    """

    name = "duckdb"

    def __init__(self, data: dict[str, list]) -> None:
        import duckdb

        self._con = duckdb.connect()
        self._con.register("staging", pa.table(data))
        self._con.execute("CREATE TABLE t AS SELECT * FROM staging")
        self._con.unregister("staging")

    def _run(self, sql: str, key: int) -> pa.Table:
        """Execute and **materialize**, so the timed call does the same work Batcher's
        `collect()` does. `.arrow()` hands back a lazy `RecordBatchReader` on some
        versions, which would stop the clock before the rows existed."""
        return self._con.execute(sql, [key]).fetch_arrow_table()

    def point_lookup(self, key: int) -> pa.Table:
        return self._run("SELECT * FROM t WHERE id = ?", key)

    def filter_project(self, key: int) -> pa.Table:
        return self._run("SELECT id, v FROM t WHERE id > ? LIMIT 10", key)

    def aggregate(self, key: int) -> pa.Table:
        return self._run("SELECT grp, SUM(v) AS total FROM t WHERE id > ? GROUP BY grp", key)

    sql_point_lookup = point_lookup


class SQLiteEngine:
    """The transactional reference: a real primary-key index, answered from a B-tree.

    Present to size the gap, not as a rival. It loses badly on the aggregate shape, which
    is the honest mirror image of it winning the point lookup.
    """

    name = "sqlite"

    def __init__(self, data: dict[str, list]) -> None:
        import sqlite3

        self._con = sqlite3.connect(":memory:")
        self._con.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, grp INTEGER, v REAL)")
        self._con.executemany(
            "INSERT INTO t VALUES (?, ?, ?)",
            zip(data["id"], data["grp"], data["v"], strict=True),
        )
        self._con.commit()

    def _arrow(self, sql: str, params: list, names: tuple[str, ...]) -> pa.Table:
        rows = self._con.execute(sql, params).fetchall()
        cols = list(zip(*rows, strict=True)) if rows else [[] for _ in names]
        return pa.table(dict(zip(names, cols, strict=True)))

    def point_lookup(self, key: int) -> pa.Table:
        return self._arrow("SELECT id, grp, v FROM t WHERE id = ?", [key], ("id", "grp", "v"))

    def filter_project(self, key: int) -> pa.Table:
        return self._arrow("SELECT id, v FROM t WHERE id > ? LIMIT 10", [key], ("id", "v"))

    def aggregate(self, key: int) -> pa.Table:
        return self._arrow(
            "SELECT grp, SUM(v) AS total FROM t WHERE id > ? GROUP BY grp",
            [key],
            ("grp", "total"),
        )

    sql_point_lookup = point_lookup


ENGINES = {"batcher": BatcherEngine, "duckdb": DuckDBEngine, "sqlite": SQLiteEngine}

#: `(label, method, varies)` — `varies` marks the parameterized shapes, where the literal
#: changes every call and a content-keyed plan cache therefore cannot hit.
SHAPES = (
    ("point-lookup (repeated)", "point_lookup", False),
    ("point-lookup (parameterized)", "point_lookup", True),
    ("sql point-lookup (parameterized)", "sql_point_lookup", True),
    ("filter+project+limit (parameterized)", "filter_project", True),
    ("group-by agg (parameterized)", "aggregate", True),
)


def _verify(engines: list, method: str, key: int) -> str | None:
    """Check every engine answers `method(key)` identically; return a complaint or `None`.

    Correctness before timing, the same rule `harness.py` applies: a fast wrong answer is a
    bug, and an engine that quietly returns nothing would otherwise post the best number
    here.
    """
    reference = _normalized(getattr(engines[0], method)(key))
    for engine in engines[1:]:
        got = _normalized(getattr(engine, method)(key))
        if got != reference:
            return f"{engine.name} disagrees with {engines[0].name} on {method}"
    return None


def _build_engines(requested: list[str], data: dict[str, list]) -> list:
    """Construct each requested engine, naming any whose optional dependency is absent.

    Skipped by name rather than silently dropped: a shorter table would otherwise read as a
    complete comparison.
    """
    engines = []
    for name in requested:
        try:
            engines.append(ENGINES[name](data))
        except ImportError:
            print(f"note: {name} not installed, skipping")
    return engines


def _row_for(shape: tuple[str, str, bool], engines: list, rows: int, iterations: int) -> str:
    """Time one shape across every engine that expresses it, as a printable table row.

    Returns a `SKIPPED:` row when the engines disagree — correctness before timing, the
    same rule `harness.py` applies, so a wrong answer can never post the best number.
    """
    label, method, varies = shape
    usable = [e for e in engines if hasattr(e, method)]
    if not usable:
        return ""
    complaint = _verify(usable, method, 7)
    if complaint is not None:
        return f"{label:<38}{'SKIPPED: ' + complaint:>26}"
    cells = []
    for engine in usable:
        call = getattr(engine, method)
        # A repeated shape ignores the index on purpose: that is what makes it hit the
        # plan cache and measure the floor rather than the re-optimization.
        run = (lambda i, c=call: c(i % rows)) if varies else (lambda _i, c=call: c(7))
        t = Timing.measure(run, iterations)
        cells.append(f"{t.wall_p50:7.3f} /{t.wall_p99:7.3f} /{t.cpu_p50:7.3f}")
    return f"{label:<38}" + "".join(f"{c:>26}" for c in cells)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--engines",
        default=",".join(ENGINES),
        help=f"comma-separated subset of: {', '.join(ENGINES)}",
    )
    args = parser.parse_args()

    requested = [name.strip() for name in args.engines.split(",") if name.strip()]
    unknown = [name for name in requested if name not in ENGINES]
    if unknown:
        parser.error(f"unknown engine(s): {', '.join(unknown)}")

    engines = _build_engines(requested, _rows(args.rows))
    if not engines:
        print("no engines available")
        return 1

    print(f"Small-query latency  rows={args.rows:,}  iterations={args.iterations}")
    print("CPU p50 is the figure to compare when the box is shared.\n")
    header = f"{'shape':<38}" + "".join(f"{e.name:>26}" for e in engines)
    print(header)
    print(f"{'':<38}" + "".join(f"{'wall p50 / p99 / cpu':>26}" for _ in engines))
    print("-" * len(header))

    failures = 0
    for shape in SHAPES:
        row = _row_for(shape, engines, args.rows, args.iterations)
        if not row:
            continue
        if "SKIPPED:" in row:
            failures += 1
        print(row)

    print(
        "\nThe repeated-vs-parameterized gap on the same shape is the plan-cache miss a "
        "prepared-statement API would remove."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
