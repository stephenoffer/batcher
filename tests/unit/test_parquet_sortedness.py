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

from batcher.io.formats.structured.parquet.source import ParquetSource

pytestmark = pytest.mark.unit


def _table(keys) -> pa.Table:
    keys = list(keys)
    return pa.table({"k": pa.array(keys, pa.int64()), "v": ["x"] * len(keys)})


def _sorted_by(directory) -> tuple[str, ...]:
    return ParquetSource(str(directory)).statistics().sorted_by


_ASCENDING = [pq.SortingColumn(0, descending=False, nulls_first=False)]


# ---- the claim is made when it is true ---------------------------------------


def test_a_sorted_declared_file_is_recognized(tmp_path) -> None:
    pq.write_table(
        _table(range(1000)),
        str(tmp_path / "a.parquet"),
        sorting_columns=_ASCENDING,
        row_group_size=250,
    )

    assert _sorted_by(tmp_path) == ("k",)


def test_two_files_ordered_across_the_boundary_are_recognized(tmp_path) -> None:
    pq.write_table(_table(range(0, 500)), str(tmp_path / "a.parquet"), sorting_columns=_ASCENDING)
    pq.write_table(
        _table(range(500, 1000)), str(tmp_path / "b.parquet"), sorting_columns=_ASCENDING
    )

    assert _sorted_by(tmp_path) == ("k",)


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


def test_a_descending_declaration_is_refused(tmp_path) -> None:
    """Descending is a different ordering than `sorted_by` denotes; reinterpreting it
    would be exactly the wrong-order bug."""
    pq.write_table(
        _table(range(999, -1, -1)),
        str(tmp_path / "a.parquet"),
        sorting_columns=[pq.SortingColumn(0, descending=True)],
        row_group_size=250,
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
