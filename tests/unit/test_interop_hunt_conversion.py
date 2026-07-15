"""Conversion-fidelity tests for the framework-interop ``from_*`` constructors.

Each test pins a distinct data-fidelity defect in ``batcher.io.interop`` /
``batcher.api.session`` and FAILS without its fix:

* ``from_pylist`` / ``from_items`` inferred the schema from the *first row only*
  (``pa.Table.from_pylist``), silently dropping every key absent from that row —
  data loss. The contract (and DuckDB) is the ordered UNION of keys, nulls for
  missing cells.
* ``from_items([])`` / ``from_pandas`` / ``from_polars`` of an empty (0-row,
  >=1-column) frame crashed with "Schema and number of arrays unequal" because the
  empty-schema batch was built with zero arrays instead of one empty array per field.
* ``from_pandas`` kept the pandas index, leaking pyarrow's internal
  ``__index_level_0__`` (or the index name) into the public schema — DuckDB, Polars,
  and Ray Data all drop it.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt


def test_from_pylist_unions_keys_across_rows() -> None:
    """A key only in a later row must survive as its own column (no data loss)."""
    ds = bt.from_pylist([{"a": 1}, {"b": 2}])
    got = ds.to_pydict()
    assert got == {"a": [1, None], "b": [None, 2]}


def test_from_pylist_first_row_subset() -> None:
    """When the first row lacks a key later rows have, the column is still built."""
    ds = bt.from_pylist([{"a": 1}, {"a": 2, "b": 3}])
    assert ds.to_pydict() == {"a": [1, 2], "b": [None, 3]}


def test_from_pylist_key_order_is_first_seen() -> None:
    """The union preserves first-seen key order across rows."""
    ds = bt.from_pylist([{"a": 1, "b": 2}, {"c": 3, "a": 4}])
    assert ds.to_arrow().schema.names == ["a", "b", "c"]


def test_from_items_dict_unions_keys() -> None:
    """`from_items` over dicts follows the same union-of-keys contract."""
    ds = bt.from_items([{"a": 1}, {"b": 2}])
    assert ds.to_pydict() == {"a": [1, None], "b": [None, 2]}


def test_from_items_empty_list() -> None:
    """An empty item list yields an empty (0-row) dataset, not a crash."""
    assert bt.from_items([]).to_arrow().num_rows == 0


def test_interop_source_from_empty_single_column_table() -> None:
    """A 0-row, 1-column table wrapped as a Source keeps its schema, no arity crash."""
    from batcher.io.interop import from_pydict

    src = from_pydict({"item": pa.array([], type=pa.int64())})
    assert src.schema().names == ["item"]


@pytest.mark.parametrize("ncols", [1, 3])
def test_from_pandas_empty_frame(ncols: int) -> None:
    """An empty pandas frame with columns keeps its schema and does not crash."""
    pd = pytest.importorskip("pandas")
    cols = {f"c{i}": pd.Series([], dtype="int64") for i in range(ncols)}
    ds = bt.from_pandas(pd.DataFrame(cols))
    schema = ds.to_arrow().schema
    assert schema.names == list(cols)
    assert ds.to_arrow().num_rows == 0


def test_from_polars_empty_frame() -> None:
    """An empty polars frame with a column keeps its schema and does not crash."""
    pl = pytest.importorskip("polars")
    ds = bt.from_polars(pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)}))
    assert ds.to_arrow().schema.names == ["a"]
    assert ds.to_arrow().num_rows == 0


def test_from_pandas_drops_index() -> None:
    """A meaningful pandas index must NOT leak into the schema (matches DuckDB/Polars)."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index([10, 20, 30], name="id"))
    names = bt.from_pandas(df).to_arrow().schema.names
    assert names == ["a"]
    assert "__index_level_0__" not in names
    assert "id" not in names


def test_from_pandas_string_index_no_magic_column() -> None:
    """An unnamed non-range index must not surface as ``__index_level_0__``."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"a": [1, 2]}, index=["x", "y"])
    assert bt.from_pandas(df).to_arrow().schema.names == ["a"]
