"""Hive-partitioned writes group rows once, not once per partition.

`_hive_partition` used to select each distinct partition with its own mask: a full-table
comparison per key column, then a full `table.filter`. That is O(partitions x rows), and
partition counts are exactly what grows — a date x region x tenant layout reaches
thousands. Measured at 200k rows: 100 partitions 0.13 s, 500 0.48 s, 2,000 1.47 s,
10,000 **10.85 s**, perfectly linear.

It now sorts by the key columns once and slices the contiguous runs. The behaviour must
be identical, and two details are why this is not a one-liner:

* **NULL and NaN.** `NULL == NULL` is NULL and `NaN == NaN` is False, so naive run
  detection would start a new run on every such row and shatter that partition into one
  file per row. `group_by` put them in a single group; so must this.
* **Row order within a partition.** The old form `filter`ed rows out of the original
  table, preserving their relative order. The sort is stable, so that still holds.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from batcher.io.base.sink import FileSink

pytestmark = pytest.mark.unit


def _mask_based(table: pa.Table, cols: list[str]):
    """The previous implementation, kept as the oracle."""
    keys = table.group_by(cols).aggregate([])
    for i in range(keys.num_rows):
        key_vals = [(c, keys.column(c)[i].as_py()) for c in cols]
        mask = None
        for c, v in key_vals:
            col = table.column(c)
            if v is None:
                eq = pc.is_null(col)
            elif isinstance(v, float) and v != v:
                eq = pc.is_nan(col)
            else:
                eq = pc.equal(col, pa.scalar(v, table.schema.field(c).type))
            mask = eq if mask is None else pc.and_(mask, eq)
        yield key_vals, table.filter(mask).drop_columns(cols)


def _normalize(groups) -> dict:
    out = {}
    for key_vals, sub in groups:
        key = tuple((c, "NaN" if isinstance(v, float) and v != v else v) for c, v in key_vals)
        out[key] = sub.to_pydict()
    return out


_NAN = float("nan")

_CASES = {
    "simple": pa.table({"v": list(range(20)), "p": [str(i % 4) for i in range(20)]}),
    "nulls": pa.table(
        {
            "v": list(range(12)),
            "p": ["a", "b", None, "a", None, "b", "a", None, "b", "a", "b", None],
        }
    ),
    "nan": pa.table(
        {
            "v": list(range(9)),
            "p": pa.array([1.0, _NAN, 2.0, _NAN, 1.0, 2.0, _NAN, 1.0, 2.0], pa.float64()),
        }
    ),
    "multi_column": pa.table(
        {
            "v": list(range(24)),
            "p1": [str(i % 3) for i in range(24)],
            "p2": [str(i % 2) for i in range(24)],
        }
    ),
    "single_partition": pa.table({"v": [1, 2, 3], "p": ["x", "x", "x"]}),
    "empty": pa.table({"v": pa.array([], pa.int64()), "p": pa.array([], pa.string())}),
    "all_null": pa.table({"v": [1, 2, 3], "p": pa.array([None, None, None], pa.string())}),
    "every_row_its_own": pa.table({"v": list(range(8)), "p": [str(i) for i in range(8)]}),
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_matches_the_mask_based_implementation(name: str) -> None:
    table = _CASES[name]
    cols = [c for c in table.column_names if c.startswith("p")]

    assert _normalize(FileSink._hive_partition(table, cols)) == _normalize(_mask_based(table, cols))


def test_nulls_form_one_partition_not_one_per_row() -> None:
    """`NULL == NULL` is NULL, so naive run detection shatters the null partition."""
    table = _CASES["all_null"]

    groups = list(FileSink._hive_partition(table, ["p"]))

    assert len(groups) == 1
    assert groups[0][1].num_rows == 3


def test_nans_form_one_partition_not_one_per_row() -> None:
    """`NaN == NaN` is False, the same trap by a different route."""
    table = pa.table({"v": [1, 2, 3], "p": pa.array([_NAN, _NAN, _NAN], pa.float64())})

    groups = list(FileSink._hive_partition(table, ["p"]))

    assert len(groups) == 1
    assert groups[0][1].num_rows == 3


def test_row_order_within_a_partition_is_preserved() -> None:
    """The sort must be stable: the previous form filtered out of the original table,
    so rows kept their relative order, and a written file's row order is observable."""
    table = pa.table({"v": list(range(100)), "p": [str(i % 3) for i in range(100)]})

    for key_vals, sub in FileSink._hive_partition(table, ["p"]):
        values = sub.column("v").to_pylist()
        assert values == sorted(values), f"partition {key_vals} was reordered"


def test_every_row_is_written_exactly_once() -> None:
    """A partitioned write must not drop or duplicate rows."""
    table = pa.table(
        {"v": list(range(500)), "p": [None if i % 7 == 0 else str(i % 11) for i in range(500)]}
    )

    written = [
        v for _, sub in FileSink._hive_partition(table, ["p"]) for v in sub.column("v").to_pylist()
    ]

    assert sorted(written) == list(range(500))


def test_the_partition_column_is_dropped_from_the_payload() -> None:
    """Hive layout encodes the key in the path, so it must not also be in the file."""
    table = _CASES["multi_column"]

    for _, sub in FileSink._hive_partition(table, ["p1", "p2"]):
        assert sub.column_names == ["v"]


def test_cost_does_not_grow_with_the_partition_count() -> None:
    """The whole point. The mask form was linear in partitions; this must not be."""
    import time

    rows = 100_000

    def elapsed(partitions: int) -> float:
        table = pa.table({"v": list(range(rows)), "p": [str(i % partitions) for i in range(rows)]})
        start = time.perf_counter()
        list(FileSink._hive_partition(table, ["p"]))
        return time.perf_counter() - start

    few, many = elapsed(50), elapsed(5000)

    # A 100x increase in partitions must not cost anything like 100x the time. The mask
    # form did exactly that; a generous bound still fails it and is robust to a noisy box.
    assert many < few * 20, f"{few:.3f}s at 50 partitions, {many:.3f}s at 5000"
