"""`sorted_by` is claimed only when the footers *prove* it.

`SourceStatistics.sorted_by` existed and Kyber consumed it to delete a redundant `Sort`,
but no structured source ever set it — so a dataset written sorted by Spark, Hive or
DuckDB, which record `sorting_columns` in the footer, was re-sorted on every read.

This is the most dangerous statistic in the bundle. Every other one is a *bound*: wrong,
it makes a plan slower. Wrong sortedness makes the optimizer delete a sort that was doing
real work, and the query returns rows in the wrong order — with no error, and invisible to
any order-independent assertion.

So most of these tests are adversarial: the file that *declares* itself sorted and is not,
the two files each sorted but not ordered against each other, the descending declaration.
Every one must yield `()`. A false negative costs a sort that was going to happen anyway;
a false positive is a wrong answer.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher.io.formats.structured.parquet.source import ParquetSource
from batcher.plan.stats import SortOrder

pytestmark = pytest.mark.unit


def _table(keys) -> pa.Table:
    keys = list(keys)
    return pa.table({"k": pa.array(keys, pa.int64()), "v": ["x"] * len(keys)})


def _sorted_by(directory) -> tuple[SortOrder, ...]:
    return ParquetSource(str(directory)).statistics().sorted_by


_ASCENDING = [pq.SortingColumn(0, descending=False, nulls_first=False)]
_DESCENDING = [pq.SortingColumn(0, descending=True, nulls_first=False)]


# ---- the claim is made when it is true ---------------------------------------


def test_a_sorted_declared_file_is_recognized(tmp_path) -> None:
    pq.write_table(
        _table(range(1000)),
        str(tmp_path / "a.parquet"),
        sorting_columns=_ASCENDING,
        row_group_size=250,
    )

    assert _sorted_by(tmp_path) == (SortOrder("k"),)


def test_two_files_ordered_across_the_boundary_are_recognized(tmp_path) -> None:
    pq.write_table(_table(range(0, 500)), str(tmp_path / "a.parquet"), sorting_columns=_ASCENDING)
    pq.write_table(
        _table(range(500, 1000)), str(tmp_path / "b.parquet"), sorting_columns=_ASCENDING
    )

    assert _sorted_by(tmp_path) == (SortOrder("k"),)


# ---- and refused whenever it cannot be proved --------------------------------


def test_a_sorted_file_that_does_not_declare_it_is_not_claimed(tmp_path) -> None:
    """No declaration, no claim — the data happens to be sorted, which is not a promise."""
    pq.write_table(_table(range(1000)), str(tmp_path / "a.parquet"), row_group_size=250)

    assert _sorted_by(tmp_path) == ()


def test_a_file_that_lies_about_being_sorted_is_caught(tmp_path) -> None:
    """The case that makes this a prover: the declaration is wrong, and only the
    row-group bounds reveal it."""
    shuffled = np.random.default_rng(0).permutation(1000)
    pq.write_table(
        _table(shuffled),
        str(tmp_path / "a.parquet"),
        sorting_columns=_ASCENDING,
        row_group_size=250,
    )

    assert _sorted_by(tmp_path) == ()


def test_a_descending_declaration_is_claimed_as_descending(tmp_path) -> None:
    """A descending key is a real ordering and is recorded as one — not discarded.

    `SortOrder` carries the direction, so there is nothing to reinterpret: the claim says
    descending, and only a descending `Sort` is elided against it."""
    pq.write_table(
        _table(range(999, -1, -1)),
        str(tmp_path / "a.parquet"),
        sorting_columns=_DESCENDING,
        row_group_size=250,
    )

    assert _sorted_by(tmp_path) == (SortOrder("k", descending=True),)


def test_a_descending_declaration_over_ascending_data_is_refused(tmp_path) -> None:
    """The declaration is never taken on trust: the footer bounds must agree with it."""
    pq.write_table(
        _table(range(1000)),
        str(tmp_path / "a.parquet"),
        sorting_columns=_DESCENDING,
        row_group_size=250,
    )

    assert _sorted_by(tmp_path) == ()


def test_descending_files_out_of_order_are_refused(tmp_path) -> None:
    """Read in name order, `a` (499..0) precedes `b` (999..500), so the concatenation
    does not descend even though each file does."""
    pq.write_table(
        _table(range(499, -1, -1)), str(tmp_path / "a.parquet"), sorting_columns=_DESCENDING
    )
    pq.write_table(
        _table(range(999, 499, -1)), str(tmp_path / "b.parquet"), sorting_columns=_DESCENDING
    )

    assert _sorted_by(tmp_path) == ()


def test_descending_files_ordered_across_the_boundary_are_recognized(tmp_path) -> None:
    pq.write_table(
        _table(range(999, 499, -1)), str(tmp_path / "a.parquet"), sorting_columns=_DESCENDING
    )
    pq.write_table(
        _table(range(499, -1, -1)), str(tmp_path / "b.parquet"), sorting_columns=_DESCENDING
    )

    assert _sorted_by(tmp_path) == (SortOrder("k", descending=True),)


def test_files_disagreeing_about_direction_are_refused(tmp_path) -> None:
    """One file ascending and one descending is not one ordering."""
    pq.write_table(_table(range(0, 500)), str(tmp_path / "a.parquet"), sorting_columns=_ASCENDING)
    pq.write_table(
        _table(range(999, 499, -1)), str(tmp_path / "b.parquet"), sorting_columns=_DESCENDING
    )

    assert _sorted_by(tmp_path) == ()


def test_files_each_sorted_but_out_of_order_are_refused(tmp_path) -> None:
    """A directory of individually-sorted files is not a sorted relation. Read in name
    order, `b` (0..499) follows `a` (500..999), so the concatenation is not ascending."""
    pq.write_table(
        _table(range(500, 1000)), str(tmp_path / "a.parquet"), sorting_columns=_ASCENDING
    )
    pq.write_table(_table(range(0, 500)), str(tmp_path / "b.parquet"), sorting_columns=_ASCENDING)

    assert _sorted_by(tmp_path) == ()


def test_a_mix_of_declared_and_undeclared_files_is_refused(tmp_path) -> None:
    pq.write_table(_table(range(0, 500)), str(tmp_path / "a.parquet"), sorting_columns=_ASCENDING)
    pq.write_table(_table(range(500, 1000)), str(tmp_path / "b.parquet"))

    assert _sorted_by(tmp_path) == ()


def test_nulls_in_the_key_are_refused(tmp_path) -> None:
    """Nulls-last cannot be verified from min/max bounds alone."""
    keys = pa.array([None, *range(999)], pa.int64())
    table = pa.table({"k": keys, "v": ["x"] * 1000})
    pq.write_table(table, str(tmp_path / "a.parquet"), sorting_columns=_ASCENDING)

    assert _sorted_by(tmp_path) == ()


# ---- the unit under the integration ------------------------------------------


def test_an_unreadable_footer_voids_the_proof() -> None:
    """A gap in the evidence is not a pass: the missing file could hold anything."""
    from batcher.io.stats.sortedness import proved_sorted_by

    assert proved_sorted_by([None]) == ()
    assert proved_sorted_by([]) == ()


def test_the_claim_reaches_the_planner(tmp_path) -> None:
    """It is only worth proving because something consumes it."""
    import batcher as bt

    pq.write_table(
        _table(range(1000)),
        str(tmp_path / "a.parquet"),
        sorting_columns=_ASCENDING,
        row_group_size=250,
    )
    ordered = bt.read.parquet(str(tmp_path)).sort("k").collect()

    # Whatever the optimizer decides, the answer must still be sorted.
    assert ordered.column("k").to_pylist() == sorted(ordered.column("k").to_pylist())


# ---- what the proved ordering buys, end to end -------------------------------
#
# The claim is only worth proving because the planner deletes work on it. These pin the two
# rewrites it enables, and — because both change the *order* of the result, which an
# order-independent comparison cannot see — each one also asserts the rows themselves.


def _ops(ds) -> list[str]:
    """The optimized plan's operators, outermost first, read out of `explain()`."""
    return [line.strip().split()[0] for line in ds.explain().strip().splitlines()]


def test_a_resort_matching_a_proved_descending_source_is_deleted(tmp_path) -> None:
    pq.write_table(
        _table(range(999, -1, -1)),
        str(tmp_path / "a.parquet"),
        sorting_columns=_DESCENDING,
        row_group_size=250,
    )
    ds = bt.read.parquet(str(tmp_path)).sort("k", descending=True)

    assert "sort" not in _ops(ds)
    got = ds.collect().column("k").to_pylist()
    assert got == sorted(got, reverse=True)


def test_a_top_n_matching_a_proved_descending_source_becomes_a_limit(tmp_path) -> None:
    """``ORDER BY ts DESC LIMIT n`` over a newest-first table is the standard recent-events
    query. Against a proved ordering it reads `n` rows instead of heap-sorting the table.

    The rewrite is `sort_elimination_from_ordering`'s: the query reaches REWRITE as a `Limit`
    above a *plain* `Sort`, so removing the sort leaves the limit sitting on the scan. There
    is no separate top-N rule, and a top-N-shaped one would never fire here."""
    pq.write_table(
        _table(range(999, -1, -1)),
        str(tmp_path / "a.parquet"),
        sorting_columns=_DESCENDING,
        row_group_size=250,
    )
    ds = bt.read.parquet(str(tmp_path)).sort("k", descending=True).limit(5)

    assert _ops(ds) == ["limit", "scan"]
    assert ds.collect().column("k").to_pylist() == [999, 998, 997, 996, 995]


def test_a_top_n_against_the_opposite_direction_keeps_its_sort(tmp_path) -> None:
    """The rewrite must not fire on a direction the source does not deliver."""
    pq.write_table(
        _table(range(999, -1, -1)),
        str(tmp_path / "a.parquet"),
        sorting_columns=_DESCENDING,
        row_group_size=250,
    )
    ds = bt.read.parquet(str(tmp_path)).sort("k").limit(5)

    assert "sort" in _ops(ds)
    assert ds.collect().column("k").to_pylist() == [0, 1, 2, 3, 4]
