"""Every aggregate that accepts ``DISTINCT``, against DuckDB — and the ones that decline.

`SUM(DISTINCT x)` and its four neighbours were already served by the pre-dedup rewrite
(`agg_rewrites.rewrite_distinct_aggs`): the query deduplicates on the group keys plus `x`
once up front and the aggregate itself becomes an ordinary one. The rewrite is *generic* —
it constrains the aggregate only to have one input column — but the reach of the argument
handling was not. Only the aggregates whose sqlglot node reaches `literals._AGG_FUNCS`
unwrapped a `DISTINCT`; every aggregate served by `expressions/aggregates.py` (the typed
nodes, the composites, and the anonymous DuckDB names) let the `Distinct` node fall
through to the *scalar* translator, which rejected it as ``unsupported SQL expression:
Distinct``.

So eleven aggregates DuckDB accepts with `DISTINCT` could not be spelled that way at all,
and the error named a sqlglot class rather than the function the user wrote. This file
pins the whole vocabulary in one place — which aggregates take `DISTINCT`, and that the
answer matches DuckDB — so the two halves cannot drift apart again.

The three that still decline do so for one structural reason: `stddev_pop`, `var_pop` and
`sem` are *composite*, built from several aggregate leaves over more than one input, so
there is no single column for the dedup to redirect. They are declined by name, with the
subquery rewrite that does work, rather than by a message about a parse node.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential


@pytest.fixture
def t(duck):
    # Within-group duplicates (a: 2,2), a NULL, and a group that is entirely NULL (c).
    table = pa.table(
        {
            "k": pa.array(["a", "a", "a", "b", "b", "c"], pa.string()),
            "v": pa.array([2, 2, 5, 3, None, None], pa.int64()),
            "w": pa.array([1.5, 1.5, -2.0, 4.0, 0.0, None], pa.float64()),
        }
    )
    duck.register("t", table)
    return table


def _run(t, duck, sql):
    assert_same(bt.sql(sql, t=bt.from_arrow(t)).collect(), duck.sql(sql))


#: Every single-input aggregate the SQL front-end accepts a ``DISTINCT`` argument for.
#: `sum`/`avg`/`count`/`min`/`max`/`median`/`stddev_samp`/`var_samp` already worked; the
#: rest are the ones this file's fix added.
DISTINCT_AGGS = [
    "sum",
    "avg",
    "count",
    "min",
    "max",
    "median",
    "stddev_samp",
    "var_samp",
    "bit_and",
    "bit_or",
    "bit_xor",
    "kurtosis",
    "skewness",
    "approx_count_distinct",
    "product",
    "entropy",
    "mad",
    "any_value",
]


@pytest.mark.parametrize("fn", DISTINCT_AGGS)
def test_distinct_aggregate_matches_duckdb(t, duck, fn):
    _run(t, duck, f"SELECT k, {fn}(DISTINCT v) AS a FROM t GROUP BY k")


@pytest.mark.parametrize("fn", DISTINCT_AGGS)
def test_distinct_aggregate_without_group_by(t, duck, fn):
    """The same aggregates over the whole relation — one group, no keys to dedup on."""
    _run(t, duck, f"SELECT {fn}(DISTINCT v) AS a FROM t")


def test_distinct_quantile_matches_duckdb(t, duck):
    """A *parameterized* aggregate: the fraction must survive the DISTINCT unwrap."""
    _run(t, duck, "SELECT k, quantile_cont(DISTINCT v, 0.5) AS a FROM t GROUP BY k")


def test_distinct_over_a_float_column(t, duck):
    """`DISTINCT` over a float column — the dedup runs on the key encoder's identity."""
    _run(t, duck, "SELECT k, product(DISTINCT w) AS p, avg(DISTINCT w) AS a FROM t GROUP BY k")


def test_distinct_mixed_with_a_decomposable_plain_aggregate(t, duck):
    """The two-level rewrite: a newly-distinct aggregate beside plain mergeable ones."""
    _run(
        t,
        duck,
        "SELECT k, bit_or(DISTINCT v) AS a, count(*) AS c, sum(w) AS s, max(v) AS m "
        "FROM t GROUP BY k",
    )


def test_distinct_over_an_expression(t, duck):
    """The deduped input is an expression, not a bare column."""
    _run(t, duck, "SELECT k, product(DISTINCT v * 2) AS a FROM t GROUP BY k")


@pytest.mark.parametrize("fn", ["stddev_pop", "var_pop", "sem"])
def test_composite_distinct_declines_by_name(t, fn):
    """A composite aggregate declines naming *itself*, not the parse node it choked on."""
    with pytest.raises(NotImplementedError) as excinfo:
        bt.sql(f"SELECT k, {fn}(DISTINCT v) AS a FROM t GROUP BY k", t=bt.from_arrow(t)).collect()
    message = str(excinfo.value)
    assert "DISTINCT" in message
    assert "Distinct" not in message.replace("DISTINCT", "")  # not the sqlglot node name
    assert "subquery" in message


def test_two_different_distinct_expressions_still_decline(t):
    """One dedup pass cannot serve two different DISTINCT expressions."""
    with pytest.raises(NotImplementedError, match="two different DISTINCT"):
        bt.sql(
            "SELECT k, product(DISTINCT v) AS a, sum(DISTINCT w) AS b FROM t GROUP BY k",
            t=bt.from_arrow(t),
        ).collect()


def test_composite_plain_aggregate_beside_a_distinct_one_declines_cleanly(t):
    """A composite *plain* aggregate has no mergeable partial to pre-aggregate.

    It used to raise a bare ``AttributeError: Expr has no attribute 'func'`` out of
    `Session.sql()` — the guard asked every plain aggregate for its function tag, and a
    composite is an `Expr` with no tag at all.
    """
    with pytest.raises(NotImplementedError, match="mixing a DISTINCT aggregate"):
        bt.sql(
            "SELECT k, regr_slope(v, w) AS a, sum(DISTINCT v) AS b FROM t GROUP BY k",
            t=bt.from_arrow(t),
        ).collect()
