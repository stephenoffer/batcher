"""Wave-5 IO base-class regressions — partitioned-write row conservation.

`FileSink._hive_partition` special-cased a NULL partition key (matching it with
``is_null`` so its rows are not dropped) but not a NaN one. Because ``group_by``
places NaN in its own group yet ``col == NaN`` is False for every row, a float
partition column containing NaN selected zero rows for the NaN group — the rows
were silently dropped from the partitioned write (S1 data loss). Every input row
MUST land in exactly one partition.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.io.base.sink import FileSink

pytestmark = pytest.mark.unit


def _partition_rows(table: pa.Table, cols: list[str]) -> int:
    return sum(sub.num_rows for _, sub in FileSink._hive_partition(table, cols))


def test_nan_partition_key_conserves_rows() -> None:
    # Two of the five rows carry a NaN partition key.
    table = pa.table(
        {
            "k": pa.array([1.0, float("nan"), 2.0, float("nan"), 1.0], pa.float64()),
            "v": [10, 20, 30, 40, 50],
        }
    )
    assert _partition_rows(table, ["k"]) == table.num_rows

    # The NaN rows must survive, in their own NaN partition.
    nan_rows: list[int] = []
    for kv, sub in FileSink._hive_partition(table, ["k"]):
        key = kv[0][1]
        if isinstance(key, float) and key != key:  # NaN
            nan_rows = sorted(sub.column("v").to_pylist())
    assert nan_rows == [20, 40]


def test_null_partition_key_still_conserves_rows() -> None:
    # The pre-existing NULL handling must remain intact.
    table = pa.table(
        {
            "k": pa.array([1.0, None, 2.0, None], pa.float64()),
            "v": [1, 2, 3, 4],
        }
    )
    assert _partition_rows(table, ["k"]) == table.num_rows


def test_nan_and_null_partition_keys_together() -> None:
    table = pa.table(
        {
            "k": pa.array([float("nan"), None, 1.0, float("nan"), None], pa.float64()),
            "v": [1, 2, 3, 4, 5],
        }
    )
    assert _partition_rows(table, ["k"]) == table.num_rows
