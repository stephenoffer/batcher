"""Mixing a DISTINCT aggregate with a plain one that combines with itself.

`SUM(DISTINCT x)` is answered by a two-level aggregate: level 1 groups by the group keys
*plus* `x`, which dedups `x` implicitly, and level 2 aggregates over that. Any plain
aggregate in the same query has to survive that pre-aggregation, so it needs a
**single-column mergeable partial** — a level-1 value per sub-group that one level-2
aggregate can fold into the group's true answer.

Only five were listed as having one (`count`, `count_star`, `sum`, `min`, `max`), so
`SELECT SUM(DISTINCT x), BOOL_OR(flag) ... GROUP BY g` was refused. But level 1 *partitions*
the group's rows — every row lands in exactly one sub-group — and over a partition the
boolean folds, the bitwise folds, `product` and `any_value` are each associative and
commutative, so they combine with themselves. That is what these cover.

`ANY_VALUE` is compared loosely on purpose: SQL leaves which row it picks unspecified, so an
exact match against DuckDB would be asserting an implementation detail of both engines. What
is actually promised — that the answer is a value the group contains — is asserted instead.

The fixture carries duplicates within a group (so `DISTINCT` changes the answer), NULLs, a
NULL group key, and an all-NULL group, because those are where a two-level fold diverges from
a one-level one if the combine step is wrong.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_T = pa.table(
    {
        "g": pa.array(["a", "a", "a", "b", "b", None, None, "c"], pa.string()),
        "x": pa.array([1, 1, 2, 3, 3, 5, 5, None], pa.int64()),
        "y": pa.array([2, 2, 6, 4, 12, 7, 7, None], pa.int64()),
        "f": pa.array([1.5, 1.5, 2.0, 3.0, 4.0, 5.0, 5.0, None], pa.float64()),
        "b": pa.array([True, False, True, True, True, None, False, None], pa.bool_()),
    }
)

#: Plain aggregates that must survive the DISTINCT pre-aggregation.
_PLAIN = [
    "BOOL_AND(b)",
    "BOOL_OR(b)",
    "BIT_AND(y)",
    "BIT_OR(y)",
    "BIT_XOR(y)",
    "PRODUCT(f)",
    "SUM(y)",
    "COUNT(y)",
    "COUNT(*)",
    "MIN(y)",
    "MAX(y)",
]
_DISTINCT = ["SUM(DISTINCT x)", "AVG(DISTINCT x)", "MIN(DISTINCT x)", "MAX(DISTINCT x)"]


@pytest.mark.parametrize("plain", _PLAIN)
@pytest.mark.parametrize("distinct", _DISTINCT)
def test_a_distinct_aggregate_mixes_with_a_self_combining_one(duck, distinct, plain):
    query = f"SELECT g, {distinct} AS dv, {plain} AS pv FROM t GROUP BY g"
    duck.register("t", _T)
    assert_same(bt.sql(query, t=_T).collect(), duck.sql(query))


@pytest.mark.parametrize("distinct", _DISTINCT)
def test_any_value_beside_a_distinct_aggregate_returns_a_value_from_the_group(distinct):
    """`ANY_VALUE` picks an unspecified row, so its *contract* is membership, not a value.

    Comparing it to DuckDB row-for-row would assert an implementation detail of both
    engines. The two-level fold could still break it — by returning a value from another
    group, or a NULL where the group has non-NULL rows — and that is what this catches.
    """
    query = f"SELECT g, {distinct} AS dv, ANY_VALUE(y) AS pv FROM t GROUP BY g"
    got = bt.sql(query, t=_T).collect().to_pydict()

    members: dict[object, set] = {}
    for group, value in zip(_T.column("g").to_pylist(), _T.column("y").to_pylist(), strict=True):
        members.setdefault(group, set()).add(value)

    assert set(got["g"]) == set(members)
    for group, picked in zip(got["g"], got["pv"], strict=True):
        available = members[group]
        if available == {None}:
            assert picked is None, f"group {group!r} has only NULLs but picked {picked!r}"
        else:
            assert picked in available, f"group {group!r}: {picked!r} not in {available}"


@pytest.mark.parametrize(
    "plain",
    ["AVG(y)", "STDDEV_SAMP(f)", "VAR_SAMP(f)", "MEDIAN(y)", "COUNT(DISTINCT y)"],
)
def test_an_aggregate_with_no_single_column_partial_is_still_refused(plain):
    """A mean needs a sum *and* a count, so one level-2 column cannot reconstruct it.

    Refused rather than approximated, and the message names the aggregate at fault so the
    subquery workaround is obvious.
    """
    query = f"SELECT g, SUM(DISTINCT x) AS dv, {plain} AS pv FROM t GROUP BY g"
    with pytest.raises(NotImplementedError, match="DISTINCT"):
        bt.sql(query, t=_T).to_arrow()


def test_several_self_combining_aggregates_at_once(duck):
    """The realistic shape: one DISTINCT count beside a handful of ordinary rollups."""
    query = (
        "SELECT g, SUM(DISTINCT x) AS dv, BOOL_OR(b) AS anyb, BIT_XOR(y) AS parity, "
        "COUNT(*) AS n, MAX(y) AS hi FROM t GROUP BY g"
    )
    duck.register("t", _T)
    assert_same(bt.sql(query, t=_T).collect(), duck.sql(query))
