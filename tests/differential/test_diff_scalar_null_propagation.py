"""A scalar function of NULL is NULL, across the whole accessor surface.

SQL's null propagation is not a per-function courtesy, it is the rule every scalar
function obeys: no input value, no output value. Two functions here did not, and both
were *silent* — a plausible value where the answer is unknown, so nothing raised and
nothing in the result looked wrong:

* ``.str.capitalize()`` returned ``''``. It was built on the DuckDB ``concat`` function,
  which deliberately coalesces NULL to the empty string, rather than on ``||``, which
  propagates. ``''`` is also the right answer for an empty-string *input*, so the two
  cases were indistinguishable once the value left the expression.
* ``.dt.days_in_year()`` returned ``365``. It was a ``CASE WHEN is_leap_year() THEN 366
  ELSE 365``, and a NULL date makes the condition NULL, which is not *true*, so every
  null row fell through to the ELSE and claimed a 365-day year.

The neighbouring functions (``upper``, ``strip``, ``days_in_month``) were correct
throughout, which is what made the two stand out and is why the sweep below exists: it
holds the *whole* zero-argument accessor surface to the rule rather than the two names
that happened to break it, so the next composition built on a null-swallowing helper
fails here instead of in a user's result.
"""

from __future__ import annotations

import datetime as dt
import inspect

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col

pytestmark = pytest.mark.differential

#: The accessor namespaces whose methods are scalar functions of the column.
_NAMESPACES = ("str", "dt", "list", "json", "struct")


def _zero_arg_methods():
    """Every accessor method callable with no arguments, as (namespace, name, expr).

    Discovered from the live surface rather than listed, so a method added later is
    covered without anyone remembering to add it here.
    """
    out = []
    for ns in _NAMESPACES:
        acc = getattr(col("s"), ns)
        for name in sorted(m for m in dir(acc) if not m.startswith("_")):
            meth = getattr(acc, name)
            if not callable(meth):
                continue
            try:
                params = inspect.signature(meth).parameters.values()
                if any(
                    p.default is inspect.Parameter.empty
                    and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                    for p in params
                ):
                    continue
                out.append((f"{ns}.{name}", meth()))
            except Exception:  # a method this column's type cannot build
                continue
    return out


def test_every_zero_arg_accessor_propagates_null():
    """f(NULL) IS NULL, for every accessor method that takes no arguments.

    A method the engine cannot evaluate over a string column is skipped, not failed --
    this is about the null rule, not about which types each function accepts (that is
    `plan.types.domains`' job).
    """
    tbl = pa.table({"s": pa.array([None, "abc"], pa.string())})
    checked, violations = 0, []
    for label, expr in _zero_arg_methods():
        try:
            got = bt.from_arrow(tbl).select(r=expr).to_pydict()["r"]
        except Exception:  # unsupported for this input type
            continue
        if len(got) != 2:
            continue
        checked += 1
        if got[0] is not None:
            violations.append(f"{label}(NULL) -> {got[0]!r}")

    # Guards the sweep itself: if a refactor made every method unbuildable, the loop
    # above would pass by checking nothing.
    assert checked > 50, f"expected a broad surface, only checked {checked}"
    assert not violations, "scalar functions that swallowed a NULL:\n  " + "\n  ".join(violations)


@pytest.mark.parametrize("value", [None, "", "hELLO wORLD", "a", "ABC", "  x", "123"])
def test_capitalize_matches_duckdb(duck, value):
    """`capitalize` against DuckDB's `||`-based spelling, which propagates NULL."""
    tbl = pa.table({"s": pa.array([value], pa.string())})
    got = bt.from_arrow(tbl).select(s=col("s"), r=col("s").str.capitalize()).collect()

    duck.register("t", tbl)
    assert_same(got, duck.sql("select s, upper(s[1]) || lower(s[2:]) as r from t"))


@pytest.mark.parametrize(
    "value",
    [None, dt.date(2020, 3, 1), dt.date(2021, 3, 1), dt.date(2000, 1, 1), dt.date(1900, 1, 1)],
)
def test_days_in_year_matches_duckdb(duck, value):
    """`days_in_year` against DuckDB, including both century leap-rule edges."""
    tbl = pa.table({"d": pa.array([value], pa.date32())})
    got = bt.from_arrow(tbl).select(d=col("d"), r=col("d").dt.days_in_year()).collect()

    duck.register("t", tbl)
    assert_same(
        got,
        duck.sql("select d, dayofyear(make_date(year(d), 12, 31))::BIGINT as r from t"),
    )


def test_days_in_month_and_days_in_year_agree_about_null():
    """The pair that diverged: one returned NULL, the other 365, for the same input."""
    tbl = pa.table({"d": pa.array([None], pa.date32())})
    got = (
        bt.from_arrow(tbl)
        .select(m=col("d").dt.days_in_month(), y=col("d").dt.days_in_year())
        .to_pydict()
    )
    assert got == {"m": [None], "y": [None]}


@pytest.mark.parametrize(
    "lists",
    [
        [[1, 2, 3]],
        [[]],
        [None],
        [[None, None]],
        [[1, None, 3]],
        [[1, 2, 3], [], None, [None, None], [5]],
    ],
)
def test_list_join_empty_and_null_match_duckdb(duck, lists):
    """`list.join` distinguishes *no elements* from *no non-null elements*.

    An empty list joins to the empty string while a null list -- and a list whose
    elements are all null, which also leaves nothing to join -- yields null. The
    authoritative oracle is DuckDB's ``array_to_string``, not
    ``list_aggregate(l, 'string_agg', ...)``: the two disagree on the empty list, and
    only the former matches what this engine computes. The public docstring claimed
    "a null or empty list yields null", which was wrong about the empty case; this
    pins the real contract so the prose cannot drift from it again.
    """
    tbl = pa.table({"l": pa.array(lists, pa.list_(pa.int64()))})
    got = bt.from_arrow(tbl).select(l=col("l"), r=col("l").list.join(",")).collect()

    duck.register("t", tbl)
    assert_same(got, duck.sql("select l, array_to_string(l, ',') as r from t"))
