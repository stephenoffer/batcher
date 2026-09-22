"""Which *side* of a join the implied `IS NOT NULL` may reduce, for every join type.

A join on `l.k = r.k` can never match a null key, so the optimizer may add an implied
`IS NOT NULL` beneath the join and drop those rows early. That is sound only on a side the
join does not preserve. Pushing it onto a *preserved* side is a wrong-answer bug: a LEFT
join must still emit its null-key left rows (padded with nulls), and an ANTI join must emit
them too, since a null key matches nothing and is therefore in the anti-result by
definition.

The existing plan-shape check in `test_diff_runtime_filters.py` cannot see this. Its helper
looks for an `IsNotNull` over a column *named* `k` anywhere in the optimized plan, and both
inputs call their key `k` -- so it answers `True` for a push on either side. Measured, the
optimizer is right in every case:

    inner  both inputs reduced       right  left input only
    left   right input only          semi   both inputs reduced
    full   neither input reduced     anti   right input only

but `_pushed_not_null` returns `True` for five of those six without distinguishing them. A
regression that moved the filter to the preserving side of a LEFT or ANTI join would leave
that assertion passing. Only the row-level differential comparison would fail, and it would
report "wrong rows" rather than naming the cause -- which is the job a plan-shape test
exists to do.

So this file pins the side, derives its cases from `JOIN_TYPES` so a seventh join type has
to be classified rather than silently skipped, and backs the table with the consequence it
protects: the null-key rows the preserved side owes, counted against DuckDB. Asserting the
shape without the consequence would leave the table itself unchecked -- it would be a
statement about the optimizer's current behaviour rather than about what is correct.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _harness import assert_same
from batcher.io.source import source_statistics
from batcher.kyber.optimizer import Optimizer
from batcher.plan.expr_ir import Col, IsNotNull
from batcher.plan.expr_rewrite import split_conjuncts
from batcher.plan.logical import Filter, Join
from batcher.plan.logical.join import JOIN_TYPES
from batcher.plan.visitor import walk

pytestmark = pytest.mark.differential

pytest.importorskip("duckdb")

#: The inputs each join type may reduce -- that is, the ones it does not preserve. Written
#: out rather than derived from the optimizer, so that a change has to be argued here.
_REDUCIBLE: dict[str, frozenset[str]] = {
    "inner": frozenset({"left", "right"}),
    "semi": frozenset({"left", "right"}),
    "left": frozenset({"right"}),
    "right": frozenset({"left"}),
    "anti": frozenset({"right"}),
    "full": frozenset(),
}

_SQL_JOIN = {
    "inner": "INNER JOIN",
    "left": "LEFT JOIN",
    "right": "RIGHT JOIN",
    "full": "FULL OUTER JOIN",
    "semi": "SEMI JOIN",
    "anti": "ANTI JOIN",
}

#: Two null keys on the left, one on the right, plus key 3 which matches nothing -- so the
#: rows a wrong-side push would delete are present on both sides and are distinguishable.
_LEFT = pa.table({"k": [1, 2, 2, None, 3, None], "v": [10, 20, 21, 98, 30, 99]})
_RIGHT = pa.table({"k": [1, 1, 2, 9, None], "w": [5, 6, 7, 8, 9]})


def _optimized(ds):
    stats = [source_statistics(s) for s in ds._sources]
    return Optimizer(sources=ds._sources, source_stats=stats).logical_rewrite(ds._plan)


def _reduced_sides(ds) -> frozenset[str]:
    """The join inputs that carry an `IS NOT NULL` on the key, by *position* not by name."""
    for node in walk(_optimized(ds)):
        if isinstance(node, Join):
            found = set()
            for label, child in (("left", node.left), ("right", node.right)):
                for sub in walk(child):
                    if not isinstance(sub, Filter):
                        continue
                    if any(
                        isinstance(c, IsNotNull)
                        and isinstance(c.input, Col)
                        and c.input.name == "k"
                        for c in split_conjuncts(sub.predicate)
                    ):
                        found.add(label)
            return frozenset(found)
    return frozenset()


@pytest.fixture(scope="module")
def joined(tmp_path_factory):
    """Build the join over *Parquet* inputs, which is what lets the rule fire at all.

    `push_is_not_null_from_join_key` declines a **resident** source on purpose: the predicate
    earns its place by sinking into a scan that answers it from null counts and skips row
    groups, and Arrow already in memory can do neither, so the push would copy nearly every
    row to drop rows the join drops anyway (its docstring carries the TPC-DS measurements).
    `bt.from_arrow` is resident, so a fixture built that way reduces nothing on every join
    type, and the side table below would read as a uniform "no push" -- passing only if
    `_REDUCIBLE` were empty for all six, which is exactly the regression it exists to catch.
    """
    directory = tmp_path_factory.mktemp("null_key_reduction")
    pq.write_table(_LEFT, directory / "l.parquet")
    pq.write_table(_RIGHT, directory / "r.parquet")

    def build(how: str):
        return bt.read.parquet(str(directory / "l.parquet")).join(
            bt.read.parquet(str(directory / "r.parquet")), on="k", how=how
        )

    return build


#: Batcher emits ONE key column, coalesced across the sides -- so a right-only row of a
#: right/full join reports the right key there where `SELECT l.k` would be NULL in SQL.
_COALESCED = ("right", "full")


def _cols_sql(how: str) -> str:
    if how in ("semi", "anti"):
        return "l.k, l.v"
    key = "COALESCE(l.k, r.k)" if how in _COALESCED else "l.k"
    return f"{key} AS k, l.v AS v, r.w AS w"


def test_every_join_type_is_classified():
    """A seventh join type must be placed here before the tests below can pass over it."""
    assert set(_REDUCIBLE) == set(JOIN_TYPES)


@pytest.mark.parametrize("how", sorted(JOIN_TYPES))
def test_the_reduction_reaches_exactly_the_sides_the_join_does_not_preserve(joined, how):
    assert _reduced_sides(joined(how)) == _REDUCIBLE[how], (
        f"a `{how}` join may reduce {sorted(_REDUCIBLE[how]) or 'neither input'}; "
        "reducing a preserved side drops rows the join owes"
    )


@pytest.mark.parametrize("how", sorted(JOIN_TYPES))
def test_the_rows_a_preserved_side_owes_actually_survive(duck, joined, how):
    """The consequence, which is what makes the table above a claim about correctness.

    Counting only the *null-key* output rows isolates exactly the rows a wrong-side push
    would delete. A whole-result comparison would catch it too, but a count that moves from
    3 to 0 names the defect where "the multisets differ" does not.
    """
    duck.register("l", _LEFT)
    duck.register("r", _RIGHT)
    got = joined(how).collect().to_pydict()
    expected = (
        duck.sql(f"SELECT {_cols_sql(how)} FROM l {_SQL_JOIN[how]} r ON l.k = r.k")
        .to_arrow_table()
        .to_pydict()
    )
    assert sum(k is None for k in got["k"]) == sum(k is None for k in expected["k"]), (
        f"a `{how}` join emitted the wrong number of null-key rows"
    )


@pytest.mark.parametrize("how", sorted(JOIN_TYPES))
def test_the_whole_result_still_matches_the_oracle(duck, joined, how):
    """The backstop: the side table must not be bought with a wrong answer elsewhere."""
    duck.register("l", _LEFT)
    duck.register("r", _RIGHT)
    assert_same(
        joined(how).collect(),
        duck.sql(f"SELECT {_cols_sql(how)} FROM l {_SQL_JOIN[how]} r ON l.k = r.k"),
    )


def test_a_name_only_check_cannot_tell_the_sides_apart(joined):
    """The control that justifies this file existing beside the name-based helper.

    Both inputs name their key `k`, so "is there an `IsNotNull` on `k` in this plan?" is
    answered `True` by a push on either side. This is not a criticism of that helper in the
    tests it serves -- it is why the assertions above are written by position instead.
    """
    by_name = {
        how: any(
            isinstance(c, IsNotNull) and isinstance(c.input, Col) and c.input.name == "k"
            for node in walk(_optimized(joined(how)))
            if isinstance(node, Filter)
            for c in split_conjuncts(node.predicate)
        )
        for how in ("left", "anti", "inner")
    }
    assert by_name == {"left": True, "anti": True, "inner": True}
    assert _reduced_sides(joined("left")) != _reduced_sides(joined("inner")), (
        "the positional check separates the two cases the name-only check merges"
    )
