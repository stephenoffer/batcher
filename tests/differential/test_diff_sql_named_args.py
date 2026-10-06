"""Named SQL arguments (``name => value``) are honoured or refused, never silently dropped.

sqlglot files a named argument in whatever slot of a typed node comes next, or drops it when
the node has no slot left, and no translator handler read it. So
``round(x, 0, mode => 'half_to_even')`` answered with the default tie rule (2.5 -> 3.0),
and a made-up name on ``round`` or ``lpad`` was accepted and ignored. Each case below failed
that way before the fix.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
from tests._harness import assert_same

import batcher as bt

pytestmark = pytest.mark.differential


@pytest.fixture
def t(duck):
    table = pa.table(
        {
            "x": [0.5, 1.5, 2.5, -2.5, 3.25, None],
            "s": ["a.b", "xyz", "", None, "q.q", "a"],
        }
    )
    duck.register("t", table)
    return table


def test_round_without_mode_matches_duckdb_round(duck, t):
    query = "SELECT x, round(x) AS r0, round(x, 1) AS r1 FROM t"
    assert_same(bt.sql(query, t=t).collect(), duck.sql(query))


@pytest.mark.parametrize("digits", ["", ", 0", ", 1"])
def test_round_half_to_even_matches_duckdb_round_even(duck, t, digits):
    """``mode => 'half_to_even'`` is DuckDB's ``round_even``: 2.5 -> 2.0, not 3.0."""
    ours = f"SELECT x, round(x{digits}, mode => 'half_to_even') AS r FROM t"
    theirs = f"SELECT x, round_even(x{digits or ', 0'}) AS r FROM t"
    assert_same(bt.sql(ours, t=t).collect(), duck.sql(theirs))


def test_round_mode_equals_the_dataframe_spelling(t):
    sql = bt.sql("SELECT round(x, 0, mode => 'half_to_even') AS r FROM t", t=t).to_pydict()
    frame = bt.from_arrow(t).select(r=bt.col("x").round(0, mode="half_to_even")).to_pydict()
    assert sql == frame
    assert sql["r"][2] == 2.0  # 2.5 ties to even


def test_round_half_away_from_zero_is_the_default(duck, t):
    query = "SELECT x, round(x, 0, mode => 'half_away_from_zero') AS r FROM t"
    assert_same(bt.sql(query, t=t).collect(), duck.sql("SELECT x, round(x, 0) AS r FROM t"))


def test_round_rejects_an_unknown_mode_value(t):
    with pytest.raises(bt.PlanError, match="half_to_even"):
        bt.sql("SELECT round(x, 0, mode => 'half_even') AS r FROM t", t=t)


@pytest.mark.parametrize(
    ("query", "argument"),
    [
        ("SELECT round(x, 0, foo => 1) FROM t", "foo"),
        ("SELECT round(x, mode => 'half_to_even', foo => 1) FROM t", "foo"),
        ("SELECT lpad(s, 3, 'x', z => 1) FROM t", "z"),
        ("SELECT upper(s, k => 1) FROM t", "k"),
        ("SELECT abs(x, k => 1) FROM t", "k"),
        ("SELECT coalesce(s, k => 'a') FROM t", "k"),
    ],
)
def test_a_named_argument_nothing_consumes_is_refused(t, query, argument):
    with pytest.raises(bt.SQLUnsupportedError, match=repr(argument)) as raised:
        bt.sql(query, t=t).collect()
    assert isinstance(raised.value, NotImplementedError)
    assert raised.value.line == 1


def test_a_named_argument_reaches_a_keyword_only_parameter(duck, t):
    """``literal => false`` is `.str.contains`'s keyword: '.' becomes a regex wildcard."""
    query = "SELECT s, str_contains(s, '.', literal => false) AS m FROM t"
    frame = bt.from_arrow(t).select("s", m=bt.col("s").str.contains(".", literal=False))
    assert bt.sql(query, t=t).to_pydict() == frame.to_pydict()
    assert_same(
        bt.sql(query, t=t).collect(), duck.sql("SELECT s, regexp_matches(s, '.') AS m FROM t")
    )


@pytest.mark.parametrize(
    ("query", "match"),
    [
        ("SELECT str_contains(s, '.', literall => false) FROM t", "no parameter named"),
        ("SELECT str_contains(s, '.', literal => s) FROM t", "constant bool"),
        ("SELECT str_contains(s, '.', pattern => 'a') FROM t", "twice"),
    ],
)
def test_a_bad_keyword_on_a_derived_function_is_refused(t, query, match):
    with pytest.raises(bt.SQLUnsupportedError, match=match):
        bt.sql(query, t=t)
