"""Parquet reads that bypass the Python file handle must return exactly what it returned.

`pq.read_table` handed a Python file object round-trips every read through the interpreter,
which serializes the reader's own decode threads — so a wide projection gets *superlinearly*
slower (sf100 `lineitem`, 600M rows: 1 column 648 ms either way; 4 columns 2,831 ms through a
handle vs 1,653 ms when pyarrow owns the I/O). `FileSystem.native_read_target` hands pyarrow
the filesystem and path so it can do that I/O itself.

The fast path is a *performance* change only. These tests pin that: same rows, same schema,
same projection and predicate semantics, and a clean fallback for backends that cannot expose
a native target.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from batcher.io.filesystem import resolve_filesystem
from batcher.io.formats import ParquetSource

pytestmark = pytest.mark.unit


@pytest.fixture
def table() -> pa.Table:
    return pa.table(
        {
            "a": pa.array([1, 2, 3, 4], type=pa.int64()),
            "b": pa.array([10.0, 20.0, 30.0, 40.0], type=pa.float64()),
            "c": pa.array(["w", "x", "y", "z"]),
        }
    )


@pytest.fixture
def parquet_path(tmp_path, table: pa.Table) -> str:
    path = str(tmp_path / "t.parquet")
    pq.write_table(table, path)
    return path


def _rows(batches) -> list[dict]:
    if not batches:
        return []
    return pa.Table.from_batches(batches).to_pylist()


def test_a_local_filesystem_offers_a_native_read_target(parquet_path: str):
    fs = resolve_filesystem(parquet_path)
    target = fs.native_read_target(parquet_path)
    assert target is not None
    _pafs, in_path = target
    assert in_path == parquet_path


def test_the_native_read_equals_the_file_handle_read(parquet_path: str, table: pa.Table):
    """The whole contract: same bytes in, same rows out."""
    source = ParquetSource(parquet_path)
    native = source.read()
    assert _rows(native) == table.to_pylist()


def test_a_projection_reads_the_same_columns_either_way(parquet_path: str):
    source = ParquetSource(parquet_path)
    got = _rows(source.read(projection=["a", "c"]))
    assert got == [{"a": 1, "c": "w"}, {"a": 2, "c": "x"}, {"a": 3, "c": "y"}, {"a": 4, "c": "z"}]


def test_a_pushed_predicate_reads_the_same_rows_either_way(parquet_path: str):
    from batcher.plan.expr_ir import col

    source = ParquetSource(parquet_path)
    predicate = (col("a") > 2).to_ir()
    rows = _rows(source.read(projection=["a"], predicate=predicate))
    assert sorted(r["a"] for r in rows) == [3, 4]


def test_a_backend_without_a_native_target_falls_back_to_the_handle(parquet_path: str, table):
    """A read-through byte cache serves reads through `open`, so bypassing it would
    silently disable it. Such a backend returns None and the handle path must still work."""
    source = ParquetSource(parquet_path)

    class NoNativeTarget:
        def __init__(self, inner):
            self._inner = inner

        def native_read_target(self, path):
            return None

        def __getattr__(self, name):
            return getattr(self._inner, name)

    source._fs = NoNativeTarget(source._fs)
    assert source._read_by_path(parquet_path, None) is None  # declined
    assert _rows(source.read()) == table.to_pylist()  # and the handle read is correct


def test_an_empty_parquet_file_reads_as_no_rows(tmp_path):
    path = str(tmp_path / "empty.parquet")
    schema = pa.schema([("a", pa.int64())])
    pq.write_table(pa.table({"a": pa.array([], type=pa.int64())}, schema=schema), path)
    assert _rows(ParquetSource(path).read()) == []


# ---- native reader predicate pushdown (footer-statistics row-group pruning) ----------


def _multi_rg_parquet(tmp_path) -> str:
    path = str(tmp_path / "rg.parquet")
    # 4 row-groups of 250: column `a` = 0..999, so each group owns a disjoint 250-range.
    pq.write_table(
        pa.table({"a": list(range(1000)), "s": [str(i % 4) for i in range(1000)]}),
        path,
        row_group_size=250,
    )
    return path


def test_native_predicate_translation_pushable_and_not():
    from batcher.io.predicate import to_native_predicate
    from batcher.plan.expr_ir import col

    # Column-vs-literal comparison (with the column on the right → operator flips).
    assert to_native_predicate((col("a") >= 500).to_ir()) == {
        "node": "cmp",
        "col": "a",
        "op": "ge",
        "lit": 500,
    }
    assert to_native_predicate((5 < col("a")).to_ir()) == {  # noqa: SIM300 (tests operator flip)
        "node": "cmp",
        "col": "a",
        "op": "gt",
        "lit": 5,
    }
    # AND of two pushable terms.
    both = to_native_predicate(((col("a") >= 500) & (col("s") == "1")).to_ir())
    assert both == {
        "node": "and",
        "left": {"node": "cmp", "col": "a", "op": "ge", "lit": 500},
        "right": {"node": "cmp", "col": "s", "op": "eq", "lit": "1"},
    }
    # A temporal literal is not pushable to the native zone-map (unit ambiguity) → None.
    temporal = {
        "e": "binary",
        "op": "ge",
        "left": col("a").to_ir(),
        "right": {"e": "lit", "value": {"date": 100}},
    }
    assert to_native_predicate(temporal) is None


def test_native_filtered_read_prunes_row_groups_superset_safe(tmp_path):
    import json

    import batcher._native as nat

    from batcher.io.predicate import to_native_predicate
    from batcher.plan.expr_ir import col

    path = _multi_rg_parquet(tmp_path)
    pred = json.dumps(to_native_predicate((col("a") >= 500).to_ir()))

    # Pruned read: only row-groups 2+3 (a in [500,999]) survive → 500 rows, all >= 500.
    out = nat.read_parquet_filtered(path, [], ["a"], 65536, pred)
    vals = sorted(v for b in out for v in b.column(0).to_pylist())
    assert len(vals) == 500
    assert min(vals) == 500 and max(vals) == 999  # a strict superset would still be >= 500

    # No row-group can match → provably empty.
    none = json.dumps(to_native_predicate((col("a") > 100000).to_ir()))
    assert nat.read_parquet_filtered(path, [], ["a"], 65536, none) == []

    # A non-pushable / malformed predicate reads everything (never under-reads).
    assert sum(b.num_rows for b in nat.read_parquet_filtered(path, [], ["a"], 65536, "x")) == 1000


def test_native_scan_batches_pushes_predicate(tmp_path):
    """The distributed native scan path applies predicate pruning and stays a superset."""
    from batcher.dist.executors.scan_read import _native_scan_batches
    from batcher.io.splits import parquet_row_group_splits
    from batcher.plan.expr_ir import col

    path = _multi_rg_parquet(tmp_path)
    splits = parquet_row_group_splits(path, None)  # one split per row-group
    result = _native_scan_batches(splits, ["a"], (col("a") >= 500).to_ir())
    assert result is not None
    vals = sorted(v for b in result for v in b.column(0).to_pylist())
    # Row-group pruning keeps groups 2+3; every returned value satisfies the predicate's
    # necessary condition (>= 500), and the true matches are all present.
    assert set(range(500, 1000)) <= set(vals)
    assert all(v >= 500 for v in vals)
