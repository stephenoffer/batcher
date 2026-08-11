"""String and list functions over an all-null (Arrow ``Null``-typed) column match DuckDB.

A column that is entirely null carries Arrow's ``Null`` data type — what ``SELECT NULL AS x``,
a ``from_pydict`` column of all ``None``, a left join that matched nothing, and a batch of
generations that all failed each produce. It is a real type, not an error.

The aggregate kernels had exactly this bug and it was fixed (``test_diff_agg_null_dtype``,
``coerce_null_call_inputs``). The string and list families were missed in that sweep, so
arithmetic, ``cast``, ``coalesce`` and ``is_null`` all answered NULL for such a column while
every ``.str`` method and ``.list`` kernel raised "expected a Utf8 argument, got Null" and
failed the whole query.

That asymmetry bites hardest on the AI path, where nulls are the *documented* outcome of a
generation the engine could not produce — so parsing a batch whose generations all failed
killed the run. DuckDB returns NULL for every one of these, which is what the coercion in
``eval/str`` and ``eval/list_ops/coerce`` now does too.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential


@pytest.fixture
def t(duck):
    tbl = pa.table(
        {
            "g": pa.array([1, 2], pa.int64()),
            "s": pa.array([None, None], pa.null()),
            "m": pa.array(["ab", None], pa.string()),
        }
    )
    duck.register("t", tbl)
    return tbl


def test_upper_of_an_all_null_column_is_null(duck, t):
    out = bt.from_arrow(t).select(g=col("g"), r=col("s").str.upper()).collect()
    assert_same(out, duck.sql("select g, upper(s::VARCHAR) as r from t"))


def test_length_of_an_all_null_column_is_null(duck, t):
    out = bt.from_arrow(t).select(g=col("g"), r=col("s").str.len()).collect()
    assert_same(out, duck.sql("select g, length(s::VARCHAR) as r from t"))


def test_contains_on_an_all_null_column_is_null(duck, t):
    out = bt.from_arrow(t).select(g=col("g"), r=col("s").str.contains("a")).collect()
    assert_same(out, duck.sql("select g, contains(s::VARCHAR, 'a') as r from t"))


def test_replace_on_an_all_null_column_is_null(duck, t):
    out = bt.from_arrow(t).select(g=col("g"), r=col("s").str.replace("a", "b")).collect()
    assert_same(out, duck.sql("select g, replace(s::VARCHAR, 'a', 'b') as r from t"))


def test_a_mixed_column_is_unaffected(duck, t):
    """The fix must not change a column that already had a string type."""
    out = bt.from_arrow(t).select(g=col("g"), u=col("m").str.upper(), n=col("m").str.len())
    assert_same(out.collect(), duck.sql("select g, upper(m) as u, length(m) as n from t"))


def test_list_length_of_an_all_null_column_is_null(duck):
    """The `.list` namespace rejected `Null` the same way the string one did."""
    tbl = pa.table({"g": pa.array([1, 2], pa.int64()), "e": pa.array([None, None], pa.null())})
    duck.register("lt", tbl)
    out = bt.from_arrow(tbl).select(g=col("g"), r=col("e").list.len()).collect()
    assert_same(out, duck.sql("select g, len(e::BIGINT[]) as r from lt"))


def test_a_generation_batch_that_wholly_failed_still_parses() -> None:
    """The shape this actually protects: every row's generation came back null.

    `ds.ml.generate` reports a row the engine could not produce as null, so a batch the
    provider refused wholesale is an all-null column — and the parse step downstream used to
    take the entire query down with it.
    """
    refuses = lambda: lambda prompts: [None] * len(prompts)  # noqa: E731
    ds = bt.from_pydict({"q": ["a", "b"]}).ml.generate(refuses, prompt_column="q")

    parsed = ds.select(answer=bt.extract_json("response")).to_pydict()

    assert parsed["answer"] == [None, None]


@pytest.mark.parametrize(
    ("method", "sql"),
    [
        ("abs", "abs(s::DOUBLE)"),
        ("sqrt", "sqrt(s::DOUBLE)"),
        ("floor", "floor(s::DOUBLE)"),
        ("ceil", "ceil(s::DOUBLE)"),
        ("ln", "ln(s::DOUBLE)"),
    ],
)
def test_math_over_an_all_null_column_is_null(duck, t, method, sql):
    """The math family rejected `Null` the same way the string and list ones did.

    One arm in `eval_math` covers all of them, mirroring the `Int64`/`Decimal` promotion
    directly above it.
    """
    out = bt.from_arrow(t).select(g=col("g"), r=getattr(col("s"), method)()).collect()
    assert_same(out, duck.sql(f"select g, {sql} as r from t"))


def test_a_real_numeric_column_is_unaffected(duck):
    """The promotion must not disturb a column that already had a numeric type."""
    tbl = pa.table({"g": pa.array([1, 2, 3], pa.int64()), "x": pa.array([-2.5, None, 4.0])})
    duck.register("nt", tbl)
    out = bt.from_arrow(tbl).select(g=col("g"), a=col("x").abs()).collect()
    assert_same(out, duck.sql("select g, abs(x) as a from nt"))


@pytest.mark.parametrize(
    ("expr", "sql"),
    [
        (lambda c: c.dt.year(), "year(s::TIMESTAMP)"),
        (lambda c: c.dt.month(), "month(s::TIMESTAMP)"),
        (lambda c: c.list.contains(1), "list_contains(s::BIGINT[], 1)"),
    ],
)
def test_the_remaining_families_are_null_over_a_null_column(duck, t, expr, sql):
    """Temporal and list membership had the same gap as string, list, math and map.

    Between them the six families cover essentially every expression an AI pipeline runs
    over a column whose batch came back empty.
    """
    out = bt.from_arrow(t).select(g=col("g"), r=expr(col("s"))).collect()
    assert_same(out, duck.sql(f"select g, {sql} as r from t"))


def test_struct_field_of_an_all_null_column_is_null(duck, t):
    """DuckDB has no `Null`-typed struct to cast to, so this is pinned directly."""
    out = bt.from_arrow(t).select(g=col("g"), r=col("s").struct.field("a")).collect()
    assert out.column("r").to_pylist() == [None, None]


def test_a_real_struct_column_is_unaffected() -> None:
    ds = bt.from_pydict({"d": [{"a": 1}, None, {"a": 3}]})
    assert ds.select(r=col("d").struct.field("a")).to_pydict()["r"] == [1, None, 3]
