"""`GroupBy.std()` / `.var()` and the rest of the no-argument shorthands, against DuckDB.

`tools/api_exercise_coverage.py` measures which public callables the suite actually *runs*,
as opposed to which are documented. Over the whole 2,764-callable surface it found 43
unexercised, and two of them were **`GroupBy.std` and `GroupBy.var`** — core relational
aggregates with no test in the suite at all.

The expression spelling was covered (`agg(v=col("v").var())` appears in several files). The
*shorthand* was not, and it is a different thing: `agg(...)` is told which column to reduce,
while `ds.group_by("g").std()` has to **choose** — "every non-key numeric column by default",
per its docstring. That choice is the whole content of these methods and it is where they
can go wrong:

- including the group key in the reduction,
- reducing a string column with a numeric aggregate, or refusing to reduce one with `min`,
- silently dropping a column that should have been reduced.

Every one of those produces a well-formed result with a plausible schema. Only a comparison
against an engine that made the same choices catches them, and only if the aggregate is
named per column so the *selection* is visible in the output.

They had a `.. doctest::` in the docstring, which `just docs` executes — so this is precisely
the gap that tool exists to name: documented, demonstrated, and never run by the suite that
gates a commit.

Measured against duckdb 1.5.5: the two engines agree exactly, including on a sample variance
over a single-row group, which is undefined and null in both rather than zero.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

pytest.importorskip("duckdb")

#: A float column, an integer column, a string column, and a two-row group whose values are
#: identical — so a zero standard deviation is distinguishable from a null one.
_MIXED = pa.table(
    {
        "g": ["a", "a", "b", "b"],
        "x": [1.0, 3.0, 10.0, 10.0],
        "y": [2, 4, 6, 8],
        "s": ["p", "q", "r", "s"],
    }
)

#: Group "a" has one row, so its *sample* variance and standard deviation are undefined.
_UNEVEN = pa.table({"g": ["a", "b", "b"], "x": [5.0, 1.0, 3.0]})


@pytest.mark.parametrize(
    ("method", "sql_fn"),
    [
        ("std", "stddev_samp"),
        ("var", "var_samp"),
        ("sum", "sum"),
        ("mean", "avg"),
    ],
)
def test_a_numeric_shorthand_reduces_every_numeric_column_and_no_others(duck, method, sql_fn):
    """`std`/`var`/`sum`/`mean` take the numeric columns; the string column is left out.

    Naming the aggregate per column in the SQL is deliberate: it makes the *selection* part
    of what is compared. A shorthand that reduced `s` as well, or that dropped `y`, would
    differ from this query in its column set and fail on that alone.
    """
    duck.register("t", _MIXED)
    got = getattr(bt.from_arrow(_MIXED).group_by("g"), method)().collect()
    assert got.column_names == ["g", "x", "y"], (
        f"`{method}()` must reduce exactly the non-key numeric columns, not {got.column_names}"
    )
    assert_same(got, duck.sql(f"SELECT g, {sql_fn}(x) AS x, {sql_fn}(y) AS y FROM t GROUP BY g"))


@pytest.mark.parametrize("method", ["min", "max"])
def test_min_and_max_also_reduce_the_string_column(duck, method):
    """The counterpart, and the reason "numeric" is not the rule for every shorthand.

    `min`/`max` are defined over strings, so excluding `s` would silently narrow the result
    of a documented default. Both engines include it.
    """
    duck.register("t", _MIXED)
    got = getattr(bt.from_arrow(_MIXED).group_by("g"), method)().collect()
    assert got.column_names == ["g", "x", "y", "s"]
    assert_same(
        got,
        duck.sql(
            f"SELECT g, {method}(x) AS x, {method}(y) AS y, {method}(s) AS s FROM t GROUP BY g"
        ),
    )


@pytest.mark.parametrize(("method", "sql_fn"), [("std", "stddev_samp"), ("var", "var_samp")])
def test_a_single_row_group_has_no_sample_spread(duck, method, sql_fn):
    """Undefined, therefore null — not zero.

    This is the edge these two aggregates are most likely to get wrong, because 0.0 is the
    plausible-looking answer and would pass any test that only checked the shape. It is also
    the one place `std` and `var` differ from `sum` and `mean`, which are perfectly happy
    with one row.
    """
    duck.register("t", _UNEVEN)
    got = getattr(bt.from_arrow(_UNEVEN).group_by("g"), method)().collect()
    assert got.to_pydict()["x"][0] is None, "a one-row group has no sample spread"
    assert_same(got, duck.sql(f"SELECT g, {sql_fn}(x) AS x FROM t GROUP BY g"))


def test_a_zero_spread_group_is_zero_and_not_null(duck):
    """The other half of the pair above, which is what makes it an assertion rather than a
    coincidence: group "b" of `_MIXED` holds two identical values, so its spread is a real
    zero. A shorthand that returned null for "no variation" would satisfy the null test and
    fail this one.
    """
    duck.register("t", _MIXED)
    got = bt.from_arrow(_MIXED).group_by("g").std().collect().to_pydict()
    by_group = dict(zip(got["g"], got["x"], strict=True))
    assert by_group["b"] == 0.0
    assert by_group["a"] == pytest.approx(2.0**0.5)


def test_the_group_key_is_never_reduced(duck):
    """A numeric group key must stay a key, not become an aggregate of itself.

    With a string key the mistake is invisible — `s` would be excluded from a numeric
    shorthand anyway — so the key has to be numeric for this to check anything.
    """
    numeric_key = pa.table({"k": [1, 1, 2], "v": [10.0, 20.0, 30.0]})
    duck.register("t", numeric_key)
    got = bt.from_arrow(numeric_key).group_by("k").sum().collect()
    assert got.column_names == ["k", "v"]
    assert sorted(got.to_pydict()["k"]) == [1, 2], "the key column holds keys, not sums"
    assert_same(got, duck.sql("SELECT k, sum(v) AS v FROM t GROUP BY k"))
