"""How much of TPC-DS Batcher's SQL front-end can actually parse and plan.

# Why this exists

`benchmarks/suites/standard/tpcds.py` runs 7 of the 99 queries, and its docstring says
expanding "is mechanical once a query's tables are added to `sources.TPCDS_TABLES`". That is
the wrong binding constraint. Adding tables is trivial; the constraint is the **SQL surface**.
Which of the 99 Batcher can express was an opinion. This makes it a measurement.

It runs parse-and-plan only — no data, no execution, seconds to complete — against empty
registered schemas. So it answers "can the front-end express this query", which is the
question the roadmap needs, without a data generator or a scale factor.

# Neither the queries nor the schemas are ours, and neither is invented

Both come from DuckDB's `tpcds` extension: `tpcds_queries()` ships the 99 official texts with
the validation-default substitution parameters, and `dsdgen(sf=0)` creates all 24 tables empty
with the official column names and types.

That sourcing is deliberate. An earlier draft of this file hand-typed the 24 `CREATE TABLE`
statements from memory — which would have produced schemas that looked right, were subtly
wrong, and would have reported a coverage number measuring the typo rather than the engine.
If the extension is unavailable this script refuses to run rather than fall back to a partial
or remembered set.

# What the result is and is not

It is a **coverage** measurement: parsed / planned / unsupported, with a reason per failure.
It is **not** a performance result, not an audited TPC result, and not a claim that a planned
query returns the right answer — planning proves the front-end accepts it, nothing more.

Usage:
    python benchmarks/internals/tpcds_coverage.py
    python benchmarks/internals/tpcds_coverage.py --failures    # print each reason
    python benchmarks/internals/tpcds_coverage.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass


@dataclass
class Outcome:
    """What happened to one query."""

    query: str
    status: str  # "planned" | "unsupported"
    reason: str = ""
    error_type: str = ""


def _tpcds_connection():
    """A DuckDB connection with `tpcds` loaded and the 24 tables created empty.

    Raises:
        SystemExit: If the extension is unavailable. Refusing beats reporting coverage over a
            remembered schema or a partial query set.
    """
    try:
        import duckdb
    except ImportError:
        sys.exit("duckdb supplies the official TPC-DS texts and schemas (pip install duckdb)")
    con = duckdb.connect()
    try:
        con.execute("INSTALL tpcds; LOAD tpcds;")
        con.execute("CALL dsdgen(sf=0)")  # schemas only; every table has zero rows
    except Exception as exc:
        sys.exit(f"could not load DuckDB's tpcds extension: {exc}")
    return con


def official_queries(con) -> dict[str, str]:
    """The 99 official TPC-DS texts, keyed `q1`..`q99`.

    Args:
        con: A connection from `_tpcds_connection`.

    Returns:
        Query name to SQL text.
    """
    rows = con.execute("SELECT query_nr, query FROM tpcds_queries() ORDER BY query_nr").fetchall()
    if len(rows) != 99:
        sys.exit(f"expected 99 official queries, got {len(rows)}")
    return {f"q{nr}": sql for nr, sql in rows}


def official_tables(con) -> dict:
    """The 24 TPC-DS tables as empty Arrow tables carrying the official schema.

    Args:
        con: A connection from `_tpcds_connection`.

    Returns:
        Table name to an empty `pyarrow.Table` with that table's schema.
    """
    names = [
        row[0]
        for row in con.execute(
            "SELECT table_name FROM information_schema.tables ORDER BY table_name"
        ).fetchall()
    ]
    # `.arrow()` yields a RecordBatchReader on current DuckDB; `read_all` pins it to a Table,
    # which is what the translator accepts.
    return {name: con.execute(f"SELECT * FROM {name} LIMIT 0").arrow().read_all() for name in names}


def _classify(exc: BaseException) -> str:
    """Reduce an exception to a short, groupable reason.

    Grouping is the point: 99 distinct messages is a list, while "12 queries need ROLLUP" is
    a roadmap item.
    """
    text = str(exc)
    # Ordered most-specific first. Decimal leads because it is the single biggest cause and
    # it masquerades as several others: a window over a Decimal column reads as a window gap,
    # and a UNION of Decimal against Float64 reads as a set-operation gap. Grouping them under
    # the operator would split one fix across four roadmap items.
    patterns = [
        (r"Decimal", "decimal type support"),
        (r"IN/EXISTS subquery combined with OR", "disjunctive IN/EXISTS subquery"),
        (r"unsupported SQL expression: Star", "star expansion in an expression"),
        (r"\bROLLUP\b|\bCUBE\b|GROUPING SETS", "grouping sets / rollup / cube"),
        (r"\bRECURSIVE\b", "recursive CTE"),
        (r"\bLATERAL\b", "lateral join"),
        (r"__jk_", "synthesized join key out of scope"),
        (r"\bINTERSECT\b|\bEXCEPT\b|set operation", "set operation (intersect/except)"),
        (r"\bEXCLUDE\b|IGNORE NULLS", "window frame option"),
        (r"window|OVER\b", "window function"),
        (r"correlated", "correlated subquery"),
        (r"unknown column|unknown identifier|references unknown", "name resolution"),
        (r"unsupported function|unknown function|no such function", "scalar function"),
        (r"\bCAST\b|incompatible|type", "type / cast"),
        (r"parse|syntax|unexpected", "parse"),
    ]
    for pattern, label in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return label
    return "other"


def measure(queries: dict[str, str], tables: dict[str, object]) -> list[Outcome]:
    """Parse and plan each query against the empty schemas. No data, no execution."""
    import batcher as bt

    session = bt.Session()
    outcomes: list[Outcome] = []
    for name, sql in queries.items():
        # A TPC-DS "query" may be several statements (q14/q23/q24/q39 are two). Planning the
        # last one is not the same as planning the query, so every statement must plan.
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        try:
            for statement in statements:
                dataset = session.sql(statement, **tables)
                _ = dataset.schema  # force name resolution and type inference
            outcomes.append(Outcome(query=name, status="planned"))
        except BaseException as exc:
            outcomes.append(
                Outcome(
                    query=name,
                    status="unsupported",
                    reason=_classify(exc),
                    error_type=type(exc).__name__,
                )
            )
    return outcomes


def report(outcomes: list[Outcome], *, show_failures: bool) -> None:
    """Print the coverage summary and the grouped reasons."""
    planned = [o for o in outcomes if o.status == "planned"]
    unsupported = [o for o in outcomes if o.status == "unsupported"]

    print(f"TPC-DS front-end coverage: {len(planned)}/{len(outcomes)} queries plan")
    print("  (parse + plan only, against empty schemas — NOT an execution or performance result)")
    print()

    if unsupported:
        print("Unsupported, grouped by reason — this list is the SQL-parity roadmap:")
        for reason, count in Counter(o.reason for o in unsupported).most_common():
            names = " ".join(o.query for o in unsupported if o.reason == reason)
            print(f"  {count:3d}  {reason:34s} {names}")
        print()

    if show_failures:
        for outcome in unsupported:
            print(f"  {outcome.query}: [{outcome.error_type}] {outcome.reason}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures", action="store_true", help="print each failing query")
    parser.add_argument("--json", metavar="PATH", help="write the outcomes as JSON")
    args = parser.parse_args()

    con = _tpcds_connection()
    outcomes = measure(official_queries(con), official_tables(con))
    report(outcomes, show_failures=args.failures)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump([asdict(o) for o in outcomes], handle, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
