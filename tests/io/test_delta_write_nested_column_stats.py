"""A Delta table with a struct column could not be written at all.

Every write records per-file statistics, because they are the *next* query's file-skipping
index (`io/formats/lakehouse/delta/_commit.py`). The bounds already skipped nested columns —
"an absent stat is always sound, a wrong one silently loses rows" — but the **null count**
was recorded for every column including a struct, as a plain number. Delta's protocol says a
struct's `nullCount` is an *object* mirroring the struct's own fields, so the commit carried
`{"st": 1}` where the reader demands `{"st": {"k": 1}}`, and the Delta kernel refused the
whole commit:

    CommitError: Kernel error: Json error: whilst decoding field 'nullCount' …

What kept it hidden is that it did not fail on every such table. A struct column **alone**
committed fine, and so did `int + struct`; it took a second column of a particular type in
the same file for the kernel to reach the strict decode. The reproducer that found it was
`boolean + struct`, and nothing about booleans is the cause — which is exactly why the tests
below cover the struct against several neighbours rather than the one pair that happened to
fail first.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pyarrow as pa
import pytest

import batcher as bt

pytest.importorskip("deltalake", reason="the Delta writer needs deltalake")


def _struct_column() -> pa.Array:
    return pa.array([{"k": 1}, {"k": None}, None], pa.struct([("k", pa.int64())]))


@pytest.mark.parametrize(
    ("name", "neighbour"),
    [
        ("b", pa.array([True, False, None])),
        ("i", pa.array([1, 2, None], pa.int64())),
        ("s", pa.array(["a", "", None])),
        ("f", pa.array([1.5, -0.0, None], pa.float64())),
        ("d", pa.array([dt.date(2024, 3, 5), None, None], pa.date32())),
    ],
)
def test_a_struct_column_commits_beside_any_neighbour(tmp_path, name, neighbour):
    """The failure depended on which other column shared the file, so vary that."""
    table = pa.table({name: neighbour, "st": _struct_column()})
    path = str(tmp_path / f"delta_{name}")
    bt.from_arrow(table).write.delta(path)
    back = bt.read_delta(path).collect()
    assert back.num_rows == 3
    assert back.schema.field("st").type == pa.struct([("k", pa.int64())])


def test_every_column_type_commits_in_one_table(tmp_path):
    """The whole type matrix in a single file — the shape that first failed."""
    table = pa.table(
        {
            "i": pa.array([1, 2, None], pa.int64()),
            "f": pa.array([1.5, -0.0, None], pa.float64()),
            "s": pa.array(["a", "", None]),
            "b": pa.array([True, False, None]),
            "ts": pa.array(
                [dt.datetime(2024, 3, 5, 13, 45, 30), dt.datetime(1969, 6, 1), None],
                pa.timestamp("us"),
            ),
            "d": pa.array([dt.date(2024, 3, 5), dt.date(1969, 6, 1), None], pa.date32()),
            "l": pa.array([[1, 2], [], None], pa.list_(pa.int64())),
            "st": _struct_column(),
            "m": pa.array([Decimal("1.25"), Decimal("-2.50"), None], pa.decimal128(10, 2)),
        }
    )
    path = str(tmp_path / "delta_all")
    bt.from_arrow(table).write.delta(path)
    back = bt.read_delta(path).collect()
    assert back.sort_by("i").to_pydict() == table.sort_by("i").to_pydict()


def test_the_leaf_columns_still_carry_their_statistics(tmp_path):
    """Skipping the nested column must not cost the stats the skipping index is for.

    The fix drops one entry, not the index: a `WHERE` on a leaf column must still prune,
    which it can only do if that column's bounds and null count reached the commit.
    """
    from batcher.io.formats.lakehouse.delta._commit import collect_file_stats

    table = pa.table(
        {
            "i": pa.array([1, 5, None], pa.int64()),
            "st": _struct_column(),
            "l": pa.array([[1], None, None], pa.list_(pa.int64())),
        }
    )
    stats = collect_file_stats(table)
    assert stats["min_values"] == {"i": 1}
    assert stats["max_values"] == {"i": 5}
    # The leaf keeps its count; the nested columns contribute none, because the protocol
    # spells theirs as an object and an absent statistic is sound.
    assert stats["null_counts"] == {"i": 1}
    assert stats["num_records"] == 3


def test_a_written_delta_table_still_prunes_on_a_leaf_predicate(tmp_path):
    """End to end: the statistics survive the fix and a filtered read still answers."""
    path = str(tmp_path / "delta_prune")
    table = pa.table({"i": pa.array([1, 2, 3], pa.int64()), "st": _struct_column()})
    bt.from_arrow(table).write.delta(path)
    out = bt.read_delta(path).filter(bt.col("i") > 1).collect()
    assert sorted(out.to_pydict()["i"]) == [2, 3]
