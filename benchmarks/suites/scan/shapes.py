"""The query shapes the scan benchmark runs against each file layout.

Every shape is one SQL string over the table ``t`` (bound by each SQL engine to a fresh
scan of the corpus) plus optional native callables for the two engines that have no SQL
surface — PyArrow (Acero) and Ray Data — so the whole lineup competes on the layout
question rather than only the SQL half of it.

The shapes are chosen to separate the costs a file layout actually moves:

- ``count`` / ``minmax`` read no data pages at all when the engine trusts parquet
  footer statistics, so they isolate **listing + metadata** — the pure small-files tax.
- ``sum1`` reads one of sixteen columns (**projection pushdown**); ``sumwide`` reads all
  sixteen, and is I/O-bound.
- ``filter`` / ``filter_agg`` select ~1% of rows, so an engine that skips row groups on
  statistics wins (**predicate pushdown**).
- ``groupby`` / ``distinct`` / ``topn`` push the cost past the scan into the operators,
  and show how much of a layout's penalty survives once real work follows it.

Two constraints shaped the SQL. The columns are ``int64`` drawn uniformly from
``[0, 2^63)``, so a bare ``SUM(column0)`` overflows 64 bits — engines disagree there by
*design* (DuckDB widens to ``HUGEINT``, others wrap), which would report an engine bug
that is really a benchmark bug. Every sum is therefore taken over a bounded expression.
For the same reason ``filter`` compares against a fraction of ``2^63`` rather than a
magic constant: :data:`SELECTIVE` cuts ~1% of rows on uniform data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

# The corpus schema: 16 uniformly-random int64 columns.
COLUMNS = tuple(f"column{i}" for i in range(16))

# ~1% of the uniform [0, 2^63) domain — a selective predicate with a stable selectivity
# at every scale, and one an engine can answer from row-group statistics.
SELECTIVE = 2**63 // 100

# Group-key modulus: 100 groups, small enough to stay a single-pass hash aggregate.
GROUPS = 100

_INT64 = pa.int64()
_FLOAT64 = pa.float64()


@dataclass(frozen=True)
class Shape:
    """One query shape: its SQL, and the native forms of the engines lacking SQL."""

    name: str
    sql: str
    #: engine name -> callable taking that engine's scan handle, returning an Arrow table
    native: dict[str, Callable[[Any], pa.Table]] = field(default_factory=dict)


def _scalar(name: str, value: object, dtype: pa.DataType = _INT64) -> pa.Table:
    return pa.table({name: pa.array([value], type=dtype)})


# --------------------------------------------------------------------------- #
# PyArrow (Acero) natives — `d` is a `pyarrow.dataset.Dataset`
# --------------------------------------------------------------------------- #
def _pa_count(d: Any) -> pa.Table:
    return _scalar("c", d.count_rows())


def _pa_minmax(d: Any) -> pa.Table:
    col = d.to_table(columns=["column0"])["column0"]
    return pa.table({"lo": pa.array([pc.min(col).as_py()]), "hi": pa.array([pc.max(col).as_py()])})


def _mod(col: pa.ChunkedArray, modulus: int) -> pa.ChunkedArray:
    """``col % modulus`` — Arrow exposes no modulo kernel, but integer divide truncates."""
    return pc.subtract(col, pc.multiply(pc.divide(col, modulus), modulus))


def _pa_sum1(d: Any) -> pa.Table:
    col = d.to_table(columns=["column0"])["column0"]
    return _scalar("s", pc.sum(_mod(col, 1000)).as_py())


def _pa_sumwide(d: Any) -> pa.Table:
    table = d.to_table(columns=list(COLUMNS))
    total = sum(pc.sum(_mod(table[c], 1000)).as_py() for c in COLUMNS)
    return _scalar("s", total)


def _pa_filter(d: Any) -> pa.Table:
    import pyarrow.dataset as ds

    return _scalar("c", d.count_rows(filter=ds.field("column0") < SELECTIVE))


def _pa_filter_agg(d: Any) -> pa.Table:
    import pyarrow.dataset as ds

    table = d.to_table(columns=["column1"], filter=ds.field("column0") < SELECTIVE)
    return _scalar("a", pc.mean(table["column1"]).as_py(), _FLOAT64)


def _pa_groupby(d: Any) -> pa.Table:
    col = d.to_table(columns=["column0"])["column0"]
    keyed = pa.table({"g": _mod(col, GROUPS)})
    agg = keyed.group_by("g").aggregate([([], "count_all")])
    return pa.table({"g": agg["g"], "n": agg["count_all"]})


def _pa_distinct(d: Any) -> pa.Table:
    col = d.to_table(columns=["column0"])["column0"]
    return _scalar("dd", len(pc.unique(col)))


def _pa_topn(d: Any) -> pa.Table:
    table = d.to_table(columns=["column0"])
    idx = pc.select_k_unstable(table, k=10, sort_keys=[("column0", "ascending")])
    return table.take(idx)


# --------------------------------------------------------------------------- #
# Ray Data natives — `rd` is a `ray.data.Dataset`
# --------------------------------------------------------------------------- #
def _ray_count(rd: Any) -> pa.Table:
    return _scalar("c", rd.count())


def _ray_minmax(rd: Any) -> pa.Table:
    return pa.table({"lo": pa.array([rd.min("column0")]), "hi": pa.array([rd.max("column0")])})


def _ray_filter(rd: Any) -> pa.Table:
    return _scalar("c", rd.filter(expr=f"column0 < {SELECTIVE}").count())


def _ray_filter_agg(rd: Any) -> pa.Table:
    mean = rd.filter(expr=f"column0 < {SELECTIVE}").mean("column1")
    return _scalar("a", mean, _FLOAT64)


SHAPES: tuple[Shape, ...] = (
    Shape(
        "count",
        "SELECT COUNT(*) AS c FROM t",
        {"pyarrow": _pa_count, "ray": _ray_count},
    ),
    Shape(
        "minmax",
        "SELECT MIN(column0) AS lo, MAX(column0) AS hi FROM t",
        {"pyarrow": _pa_minmax, "ray": _ray_minmax},
    ),
    Shape(
        "sum1",
        "SELECT SUM(column0 % 1000) AS s FROM t",
        {"pyarrow": _pa_sum1},
    ),
    Shape(
        "sumwide",
        "SELECT " + " + ".join(f"SUM({c} % 1000)" for c in COLUMNS) + " AS s FROM t",
        {"pyarrow": _pa_sumwide},
    ),
    Shape(
        "filter",
        f"SELECT COUNT(*) AS c FROM t WHERE column0 < {SELECTIVE}",
        {"pyarrow": _pa_filter, "ray": _ray_filter},
    ),
    Shape(
        "filter_agg",
        f"SELECT AVG(column1) AS a FROM t WHERE column0 < {SELECTIVE}",
        {"pyarrow": _pa_filter_agg, "ray": _ray_filter_agg},
    ),
    Shape(
        "groupby",
        f"SELECT column0 % {GROUPS} AS g, COUNT(*) AS n FROM t GROUP BY column0 % {GROUPS}",
        {"pyarrow": _pa_groupby},
    ),
    Shape(
        "distinct",
        "SELECT COUNT(DISTINCT column0) AS dd FROM t",
        {"pyarrow": _pa_distinct},
    ),
    Shape(
        "topn",
        "SELECT column0 FROM t ORDER BY column0 LIMIT 10",
        {"pyarrow": _pa_topn},
    ),
)
