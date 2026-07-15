"""Property: a random relational pipeline agrees with DuckDB (the master invariant).

The other property modules pin *specific* invariants (chunk-independence, optimizer
result-invariance). This one is the broad fuzzer: Hypothesis draws a random table and a
random *chain* of relational steps — filter / project / with-column / distinct /
group-by-aggregate — builds the identical query in Batcher and in nested DuckDB SQL, and
asserts the two multisets match. It finds bugs in the *space between* the enumerated
differential cases: operator interactions no hand-written test happened to compose.

Draws deliberately avoid the two known, out-of-scope divergences so a counterexample is a
*new* bug, not a re-report:
  * signed zero (``-0.0``): the engine compares floats by total order (``-0.0 < 0.0``),
    ledger B26 — an open, separately-owned question. Float columns here never carry a
    ``-0.0`` that a comparison/sort could observe.
  * IEEE division: ``a / 0`` and ``a // 0`` follow Polars/IEEE (``±inf``), not SQL
    ``NULL``. The pipeline uses only ``+``/``-``/``*``, never division.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import batcher as bt
from batcher import col

pytest.importorskip("batcher._native", reason="native engine not built")
duckdb = pytest.importorskip("duckdb")

pytestmark = [pytest.mark.property, pytest.mark.differential]


def _coerce(v: object) -> object:
    """Canonicalize a scalar so int/float widening and NaN payloads compare equal."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if v != v:
            return "<nan>"
        if not math.isfinite(v):
            return float(v)
        r = round(v, 6)
        return int(r) if r == int(r) else r
    from decimal import Decimal

    if isinstance(v, Decimal):
        f = float(v)
        r = round(f, 6)
        return int(r) if r == int(r) else r
    return v


def _rowset(table: pa.Table) -> list[tuple]:
    cols = sorted(table.column_names)
    rows = [tuple(_coerce(r[c]) for c in cols) for r in table.to_pylist()]
    return sorted(rows, key=lambda t: tuple((v is None, str(type(v)), str(v)) for v in t))


_SCHEMA = pa.schema([("k", pa.int64()), ("i", pa.int64()), ("j", pa.int64()), ("s", pa.string())])
_ints = st.one_of(st.none(), st.sampled_from([-3, -1, 0, 1, 2, 3, 7]))
_strs = st.one_of(st.none(), st.sampled_from(["a", "b", "", "c"]))


@st.composite
def _table(draw: st.DrawFn) -> pa.Table:
    n = draw(st.integers(min_value=0, max_value=24))
    return pa.table(
        {
            "k": pa.array(draw(st.lists(st.integers(0, 3), min_size=n, max_size=n)), pa.int64()),
            "i": pa.array(draw(st.lists(_ints, min_size=n, max_size=n)), pa.int64()),
            "j": pa.array(draw(st.lists(_ints, min_size=n, max_size=n)), pa.int64()),
            "s": pa.array(draw(st.lists(_strs, min_size=n, max_size=n)), pa.string()),
        },
        schema=_SCHEMA,
    )


# One step = a name plus its parameters; kept as data so the same draw drives both engines.
_CMP = st.sampled_from([">", "<", ">=", "<=", "=", "<>"])
_ARITH = st.sampled_from(["+", "-", "*"])


@st.composite
def _plan(draw: st.DrawFn) -> list[tuple]:
    steps: list[tuple] = []
    for n in range(draw(st.integers(min_value=1, max_value=4))):
        kind = draw(st.sampled_from(["filter", "withcol", "select", "distinct", "groupby"]))
        if kind == "filter":
            steps.append(
                (
                    "filter",
                    draw(st.sampled_from(["k", "i", "j"])),
                    draw(_CMP),
                    draw(st.integers(-1, 2)),
                )
            )
        elif kind == "withcol":
            steps.append(
                (
                    "withcol",
                    f"c{n}",
                    draw(st.sampled_from(["k", "i", "j"])),
                    draw(_ARITH),
                    draw(st.sampled_from(["k", "i", "j"])),
                )
            )
        elif kind == "select":
            steps.append(
                (
                    "select",
                    tuple(
                        sorted(
                            draw(
                                st.lists(
                                    st.sampled_from(["k", "i", "j", "s"]),
                                    min_size=1,
                                    max_size=4,
                                    unique=True,
                                )
                            )
                        )
                    ),
                )
            )
        elif kind == "distinct":
            steps.append(("distinct",))
        else:
            steps.append(
                (
                    "groupby",
                    draw(st.sampled_from(["k", "s"])),
                    draw(st.sampled_from(["k", "i", "j"])),
                )
            )
    return steps


def _cmp_expr(c: str, op: str, v: int):
    e = col(c)
    lit = bt.lit(v)
    return {
        ">": e > lit,
        "<": e < lit,
        ">=": e >= lit,
        "<=": e <= lit,
        "=": e == lit,
        "<>": e != lit,
    }[op]


def _apply(ds: bt.Dataset, sql: str, cols: set[str], grouped: bool, step: tuple):
    """Apply one step to the Batcher dataset and the parallel DuckDB SQL string."""
    kind = step[0]
    if kind == "filter":
        _, c, op, v = step
        if c not in cols:
            return None
        return (
            ds.filter(_cmp_expr(c, op, v)),
            f"SELECT * FROM ({sql}) WHERE {c} {op} {v}",
            cols,
            grouped,
        )
    if kind == "withcol":
        _, name, a, o, b = step
        if a not in cols or b not in cols:
            return None
        e = {"+": col(a) + col(b), "-": col(a) - col(b), "*": col(a) * col(b)}[o]
        return (
            ds.with_columns(**{name: e}),
            f"SELECT *, ({a}{o}{b}) {name} FROM ({sql})",
            cols | {name},
            grouped,
        )
    if kind == "select":
        _, keep = step
        keep = tuple(c for c in keep if c in cols)
        if not keep:
            return None
        return ds.select(*keep), f"SELECT {','.join(keep)} FROM ({sql})", set(keep), grouped
    if kind == "distinct":
        return ds.distinct(), f"SELECT DISTINCT * FROM ({sql})", cols, grouped
    if kind == "groupby":
        _, gk, vc = step
        if grouped or gk not in cols or vc not in cols:
            return None
        agg = ds.group_by(gk).agg(g_s=col(vc).sum(), g_n=bt.count(), g_mx=col(vc).max())
        gsql = (
            f"SELECT {gk}, SUM({vc}) g_s, COUNT(*) g_n, MAX({vc}) g_mx FROM ({sql}) GROUP BY {gk}"
        )
        return agg, gsql, {gk, "g_s", "g_n", "g_mx"}, True
    return None


_PROP = settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


@_PROP
@given(_table(), _plan())
def test_random_pipeline_matches_duckdb(table: pa.Table, plan: list[tuple]) -> None:
    """A random filter/project/with-column/distinct/group-by chain == the same DuckDB query."""
    ds: bt.Dataset = bt.from_arrow(table)
    sql = "SELECT k, i, j, s FROM t"
    cols = {"k", "i", "j", "s"}
    grouped = False
    for step in plan:
        applied = _apply(ds, sql, cols, grouped, step)
        if applied is None:  # step not applicable to the current schema — skip it
            continue
        ds, sql, cols, grouped = applied

    con = duckdb.connect()
    try:
        con.register("t", table)
        expected = con.sql(sql).to_arrow_table()
        actual = ds.collect()
        assert set(actual.column_names) == set(expected.column_names), (
            f"columns: {sorted(actual.column_names)} vs {sorted(expected.column_names)}\nsql={sql}"
        )
        got = _rowset(actual)
        want = _rowset(expected.select(actual.column_names))
        assert got == want, f"\nsql={sql}\nbatcher: {got}\nduckdb:  {want}"
    finally:
        con.close()
