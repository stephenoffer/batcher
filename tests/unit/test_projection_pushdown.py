"""Projection pushdown computes the minimal per-source column set."""

from __future__ import annotations

import batcher as bt
from batcher import col, count
from batcher.kyber.rules.projections import required_columns_per_source


def _req(ds):
    return required_columns_per_source(ds._plan)


def test_select_prunes_to_used_columns():
    ds = bt.from_pydict({"a": [1], "b": [2], "c": [3]}).select("a")
    assert _req(ds) == {0: ["a"]}


def test_filter_keeps_predicate_columns():
    ds = bt.from_pydict({"a": [1], "b": [2], "c": [3]}).filter(col("b") > 0).select("a")
    assert _req(ds) == {0: ["a", "b"]}


def test_aggregate_keeps_keys_and_inputs():
    ds = bt.from_pydict({"a": [1], "b": [2], "c": [3]}).group_by("a").agg(s=col("c").sum())
    assert _req(ds) == {0: ["a", "c"]}


def test_count_star_reads_one_column():
    ds = bt.from_pydict({"a": [1], "b": [2], "c": [3]}).group_by().agg(n=count())
    # No values needed, but one column is read to preserve row count.
    req = _req(ds)
    assert req[0] == ["a"] and len(req[0]) == 1


def test_join_prunes_each_side_keeping_keys():
    emp = bt.from_pydict({"id": [1], "name": ["x"], "dept_id": [10]})
    dept = bt.from_pydict({"dept_id": [10], "dept": ["eng"], "budget": [99]})
    ds = emp.join(dept, on="dept_id").select("name", "dept")
    req = _req(ds)
    # Left side (source 0): name + the join key. Right side (source 1): dept + key.
    assert req[0] == ["dept_id", "name"]
    assert req[1] == ["dept", "dept_id"]  # budget pruned


def test_pushdown_does_not_change_results():
    import pyarrow as pa

    class SpySource:
        def __init__(self, table):
            self._t = table
            self.last_projection = None

        def schema(self):
            return self._t.schema

        def read(self, projection=None):
            self.last_projection = projection
            t = self._t if projection is None else self._t.select(projection)
            return t.to_batches()

    spy = SpySource(pa.table({"a": [1, 2, 3], "b": [10, 20, 30], "c": [9, 9, 9]}))
    from batcher.api.dataset import Dataset
    from batcher.plan.logical import Scan
    from batcher.plan.schema import SchemaRef

    ds = Dataset(Scan(0, SchemaRef.from_arrow(spy.schema())), [spy])
    out = ds.filter(col("a") > 1).select("a", s=col("a") + col("b")).collect().to_pydict()

    assert out == {"a": [2, 3], "s": [22, 33]}
    # 'c' was never read.
    assert spy.last_projection == ["a", "b"]
