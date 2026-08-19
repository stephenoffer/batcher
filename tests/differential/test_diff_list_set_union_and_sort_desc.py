"""``list.set_union`` and ``list.sort_desc``, against DuckDB's list functions.

Two methods on the ``.list`` accessor that no test called. Both are easy to get almost
right: a union that keeps duplicates is a concatenation, and a descending sort that puts
nulls at the wrong end disagrees with every other engine at exactly the row a user checks
first.

DuckDB has both (``list_union`` and ``list_sort(..., 'DESC')``), so it is the oracle. Set
semantics make element *order* unspecified, so the union is compared as a multiset while
the sort -- where order is the entire point -- is compared element by element, in order,
and never through a sorting comparison.
"""

from __future__ import annotations

import itertools

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential

duckdb = pytest.importorskip("duckdb")


@pytest.fixture(scope="module")
def duck():
    return duckdb.connect()


#: Written out rather than inferred: a fixture holding only nulls and empty lists infers
#: as ``list<null>`` in pyarrow, and unioning that against ``list<int64>`` fails on the
#: element type rather than on anything this module is about.
_SCHEMA = pa.schema([("a", pa.list_(pa.int64())), ("b", pa.list_(pa.int64()))])


def test_set_union_matches_duckdb_on_lists_that_are_present(duck):
    """Every element of either side, once. Order is unspecified, so it is not compared.

    DuckDB has no ``list_union``, so the oracle is written from the definition:
    ``list_distinct(list_concat(a, b))``. Two rows are held out and pinned separately -- a
    null *list*, where the two engines disagree, and a null *element*, which DuckDB's
    ``list_distinct`` removes by design and Batcher keeps.
    """
    left = [[3, 1, 2], [5, 5, 1], [], [1, 2]]
    right = [[2, 9], [1, 5], [7], []]
    table = pa.table({"a": left, "b": right}, schema=_SCHEMA)
    duck.register("t", table)
    got = bt.from_arrow(table).select(u=col("a").list.set_union(col("b"))).to_pydict()["u"]
    want = (
        duck.sql("SELECT list_distinct(list_concat(a, b)) u FROM t")
        .to_arrow_table()
        .to_pydict()["u"]
    )
    for i, (ours, theirs) in enumerate(zip(got, want, strict=True)):
        assert sorted(ours) == sorted(theirs), f"row {i}: {ours} vs {theirs}"


def test_a_null_element_is_a_member_of_the_union(duck):
    """Batcher keeps a null element and deduplicates it, as Polars and Spark do.

    DuckDB's ``list_distinct`` strips nulls out of the list entirely, which is documented
    behaviour on its side and the reason it cannot be the oracle for this row. Polars'
    ``set_union`` -- which this method's docstring names as its reference spelling -- keeps
    one, and so does Batcher.
    """
    table = pa.table({"a": [[1, None, 2]], "b": [[2, None]]}, schema=_SCHEMA)
    got = bt.from_arrow(table).select(u=col("a").list.set_union(col("b"))).to_pydict()["u"][0]
    assert sorted(got, key=lambda v: (v is None, v)) == [1, 2, None]
    assert got.count(None) == 1, "the null element is deduplicated like any other"

    polars = pytest.importorskip("polars")
    theirs = (
        polars.DataFrame(
            {"a": [[1, None, 2]], "b": [[2, None]]},
            schema={"a": polars.List(polars.Int64), "b": polars.List(polars.Int64)},
        )
        .select(polars.col("a").list.set_union("b"))
        .to_series()
        .to_list()[0]
    )
    assert sorted(got, key=lambda v: (v is None, v)) == sorted(theirs, key=lambda v: (v is None, v))

    duck.register("n", table)
    stripped = (
        duck.sql("SELECT list_distinct(list_concat(a, b)) u FROM n")
        .to_arrow_table()
        .to_pydict()["u"][0]
    )
    assert None not in stripped, (
        "the departure is only real while DuckDB's list_distinct still removes nulls"
    )


def test_set_union_deduplicates_rather_than_concatenating():
    """The property that separates it from ``list.concat``, on a row that has both."""
    table = pa.table({"a": [[5, 5, 1]], "b": [[1, 5]]})
    ds = bt.from_arrow(table)
    united = ds.select(u=col("a").list.set_union(col("b"))).to_pydict()["u"][0]
    concatenated = ds.select(c=col("a").list.concat(col("b"))).to_pydict()["c"][0]
    assert sorted(united) == [1, 5], "the union of {5,1} and {1,5} has two elements"
    assert len(concatenated) == 5, "concatenation keeps every element, which is the difference"


def test_set_union_is_commutative_when_both_lists_are_present():
    """A set operation must not depend on which side it was written on."""
    left = [[3, 1, 2], [5, 5, 1], [], [1, 2], [1, None, 2]]
    right = [[2, 9], [1, 5], [7], [], [2, None]]
    ds = bt.from_arrow(pa.table({"a": left, "b": right}, schema=_SCHEMA))
    forward = ds.select(u=col("a").list.set_union(col("b"))).to_pydict()["u"]
    backward = ds.select(u=col("b").list.set_union(col("a"))).to_pydict()["u"]
    key = lambda v: (v is None, v)  # noqa: E731
    for i, (one, other) in enumerate(zip(forward, backward, strict=True)):
        assert sorted(one, key=key) == sorted(other, key=key), f"row {i}"


def test_sort_desc_matches_duckdb_element_by_element(duck):
    """Order is the whole point here, so this compares in order and never sorts first."""
    values = [[3, 1, 2], [5, 5, 1], [], [1], None, [2, None, 1]]
    table = pa.table({"a": values})
    duck.register("s", table)
    got = bt.from_arrow(table).select(d=col("a").list.sort_desc()).to_pydict()["d"]
    want = duck.sql("SELECT list_sort(a, 'DESC') d FROM s").to_arrow_table().to_pydict()["d"]
    assert got == want, f"{got}\nvs duckdb\n{want}"


def test_sort_desc_really_descends_and_keeps_every_element():
    """Asserted directly, because comparing a sort with a sorting comparison proves nothing."""
    values = [[3, 1, 2], [5, 5, 1], [10, -3, 0, 7]]
    got = (
        bt.from_arrow(pa.table({"a": values})).select(d=col("a").list.sort_desc()).to_pydict()["d"]
    )
    for original, ordered in zip(values, got, strict=True):
        assert ordered == sorted(original, reverse=True), f"{original} sorted to {ordered}"
        assert len(ordered) == len(original), "sorting must not drop a duplicate"
        for earlier, later in itertools.pairwise(ordered):
            assert earlier >= later, f"{ordered} is not descending"


def test_sort_desc_is_the_reverse_of_sort_on_a_list_with_no_nulls():
    """Cross-check against the ascending spelling, which is implemented separately."""
    values = [[3, 1, 2], [5, 5, 1], [10, -3, 0, 7], []]
    ds = bt.from_arrow(pa.table({"a": values}))
    got = ds.select(up=col("a").list.sort(), down=col("a").list.sort_desc()).to_pydict()
    for ascending, descending in zip(got["up"], got["down"], strict=True):
        assert descending == list(reversed(ascending)), f"{ascending} vs {descending}"


def test_a_null_list_on_the_right_is_read_as_an_empty_one():
    """Batcher's null model for the list set operations, which is not Spark's or Polars'.

    A null on the **left** propagates; a null on the **right** behaves as an empty list. So
    ``null union [1]`` is null while ``[1] union null`` is ``[1]``, and the operation is not
    commutative when exactly one side is missing. Spark's ``array_union`` and Polars'
    ``set_union`` -- both named as the reference in the docstrings -- return null when
    *either* side is null.

    The behaviour is consistent across ``union``, ``intersect`` and ``difference``, so it is
    a deliberate model rather than a slip in one operator, and it is pinned here rather than
    changed: altering it is a data-plane semantics decision across the whole family, not a
    test fix. What this test guarantees is that the divergence cannot move silently.
    """
    table = pa.table({"a": [None, [1, 2], None], "b": [[1], None, None]}, schema=_SCHEMA)
    ds = bt.from_arrow(table)
    got = ds.select(
        union=col("a").list.set_union(col("b")),
        intersect=col("a").list.set_intersection(col("b")),
        difference=col("a").list.set_difference(col("b")),
    ).to_pydict()
    assert got["union"] == [None, [1, 2], None]
    assert got["intersect"] == [None, [], None]
    assert got["difference"] == [None, [1, 2], None]

    flipped = ds.select(u=col("b").list.set_union(col("a"))).to_pydict()["u"]
    assert flipped[0] == [1], "the same pair in the other order answers differently"
    assert got["union"][0] is None


def test_sort_desc_leaves_an_empty_list_empty_and_a_null_list_null():
    """The distinction the sort must keep: a missing list is not an empty one."""
    table = pa.table({"a": [None, [], [2, 1]]}, schema=pa.schema([("a", pa.list_(pa.int64()))]))
    got = bt.from_arrow(table).select(d=col("a").list.sort_desc()).to_pydict()["d"]
    assert got == [None, [], [2, 1]]
