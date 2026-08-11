"""`schema_mode="strict"` says so when it is about to drop a later file's columns.

The sibling of the partition-column warning, and the same kind of loss: in strict mode the
first file's schema stands for every file, so a directory written over months -- where a
column was added partway -- reads back without it. Every row is returned and nothing is
raised, so the result is indistinguishable from a correct one for a table that never had
the column.
"""

from __future__ import annotations

import os
import warnings

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher._internal.errors import DataWarning

pytestmark = pytest.mark.integration


def _read(tmp_path, tables, **opts):
    for i, table in enumerate(tables):
        pq.write_table(table, os.path.join(str(tmp_path), f"f{i}.parquet"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = bt.read.parquet(str(tmp_path), **opts).to_pydict()
    return result, [w for w in caught if issubclass(w.category, DataWarning)]


def test_a_column_added_by_a_later_file_is_reported(tmp_path):
    result, warned = _read(tmp_path, [pa.table({"a": [1]}), pa.table({"a": [2], "b": ["x"]})])
    assert sorted(result) == ["a"], "strict mode still returns the first file's columns"
    assert len(warned) == 1
    message = str(warned[0].message)
    assert "'b'" in message
    assert "schema_mode='union'" in message


def test_matching_schemas_say_nothing(tmp_path):
    _, warned = _read(tmp_path, [pa.table({"a": [1]}), pa.table({"a": [2]})])
    assert warned == []


def test_the_evolution_modes_say_nothing_because_they_drop_nothing(tmp_path):
    result, warned = _read(
        tmp_path,
        [pa.table({"a": [1]}), pa.table({"a": [2], "b": ["x"]})],
        schema_mode="union",
    )
    assert sorted(result) == ["a", "b"]
    assert warned == []


def test_a_single_file_says_nothing(tmp_path):
    result, warned = _read(tmp_path, [pa.table({"a": [1], "b": ["x"]})])
    assert sorted(result) == ["a", "b"]
    assert warned == []


def test_the_opposite_shape_was_already_loud(tmp_path):
    """A later file *missing* a declared column raises rather than warns.

    Worth pinning next to the warning, because the asymmetry is the point: a column the
    reader cannot find is an error it can raise, and a column it never looked for is a
    silent narrowing that only a warning can surface.
    """
    from batcher._internal.errors import SchemaError

    with pytest.raises(SchemaError, match="missing column"):
        _read(tmp_path, [pa.table({"a": [1], "b": ["x"]}), pa.table({"a": [2]})])


def test_every_row_is_still_returned(tmp_path):
    result, _ = _read(tmp_path, [pa.table({"a": [1]}), pa.table({"a": [2], "b": ["x"]})])
    assert sorted(result["a"]) == [1, 2]
