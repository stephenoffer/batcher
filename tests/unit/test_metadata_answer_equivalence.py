"""The hard invariant: a metadata answer always equals the executed answer.

For every plan where `count()` / `is_empty()` / a global aggregate is answered
from metadata, that answer MUST equal a full execution — bit for bit. These
tests also pin the provenance firewall: a filtered count is never answered from
metadata, and `count_distinct` is answered only from an exact distinct count.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from batcher import col, count
from batcher.api.terminal import _collect
from batcher.api.terminal.metadata_answer import metadata_count as _answer_count


@pytest.fixture
def pq_path(tmp_path):
    table = pa.table({"x": list(range(100)), "g": [i % 7 for i in range(100)]})
    path = str(tmp_path / "t.parquet")
    pq.write_table(table, path)
    return path


def _ds(pq_path):
    return bt.read.parquet(pq_path)


# --- count(): metadata answer == execution, whenever an answer is produced ---


@pytest.mark.parametrize(
    "build",
    [
        lambda d: d,
        lambda d: d.limit(10),
        lambda d: d.limit(10, offset=5),
        lambda d: d.limit(500),  # n > rows
        lambda d: d.select(a=col("x")),
        lambda d: d.sort("x"),
        lambda d: d.sort("x").limit(3),
        lambda d: d.agg(c=count()),
        lambda d: d.limit(0),
    ],
)
def test_metadata_count_matches_execution(pq_path, build):
    ds = build(_ds(pq_path))
    answer = _answer_count(ds._plan, ds._sources)
    executed = _collect(ds._plan, ds._sources, ds.columns).num_rows
    if answer is not None:
        assert answer == executed
    # The plain count() API must agree with execution regardless of which path it took.
    assert ds.count() == executed


def test_filter_count_not_answered_from_metadata(pq_path):
    # The firewall: a filtered row count is never EXACT, so no metadata answer.
    ds = _ds(pq_path).filter(col("x") > 50)
    assert _answer_count(ds._plan, ds._sources) is None
    assert ds.count() == 49  # but execution is correct


def test_union_count_is_exact_and_matches(pq_path):
    ds = _ds(pq_path).union(_ds(pq_path))
    answer = _answer_count(ds._plan, ds._sources)
    assert answer == 200
    assert ds.count() == 200


# --- global aggregate: metadata answer == execution ---


def test_global_aggregate_min_max_matches_execution(pq_path):
    ds = _ds(pq_path).agg(mn=col("x").min(), mx=col("x").max(), c=count())
    meta = ds.to_pydict()
    assert meta == {"mn": [0], "mx": [99], "c": [100]}


def test_count_distinct_executes_without_exact_ndv(pq_path):
    # Parquet footers don't give an exact distinct count → must execute, correctly.
    ds = _ds(pq_path).agg(n=col("g").n_unique())
    assert ds.to_pydict() == {"n": [7]}


def test_is_empty_matches_execution(pq_path):
    ds = _ds(pq_path)
    assert ds.is_empty() is False
    assert ds.limit(0).is_empty() is True
    assert ds.filter(col("x") > 1000).is_empty() == (
        _collect(ds.filter(col("x") > 1000)._plan, ds._sources, ds.columns).num_rows == 0
    )


def test_schema_matches_execution(pq_path):
    ds = _ds(pq_path).select(a=col("x"), b=col("g"))
    meta_schema = ds.schema
    executed_schema = _collect(ds._plan, ds._sources, ds.columns).schema
    assert meta_schema.names == executed_schema.names


# --- provably-empty joins: count()/is_empty() answer 0 from metadata ---


def test_inner_join_empty_side_counts_zero_from_metadata(pq_path):
    # An EXACT-empty side makes an inner join EXACT-empty, so count() answers 0
    # from metadata (and equals execution) without running the join.
    ds = _ds(pq_path).limit(0).join(_ds(pq_path), on="x")
    answer = _answer_count(ds._plan, ds._sources)
    executed = _collect(ds._plan, ds._sources, ds.columns).num_rows
    assert executed == 0
    assert answer == 0  # not None → the metadata shortcut fired
    assert ds.is_empty() is True


def test_left_join_empty_left_counts_zero_from_metadata(pq_path):
    # LEFT join is left-driven: an empty left → empty result, answered from metadata.
    ds = _ds(pq_path).limit(0).join(_ds(pq_path), on="x", how="left")
    answer = _answer_count(ds._plan, ds._sources)
    assert answer == 0
    assert _collect(ds._plan, ds._sources, ds.columns).num_rows == 0


def test_left_join_empty_right_not_claimed_empty(pq_path):
    # The firewall: a LEFT join with an empty *right* keeps every left row (null-
    # extended), so it is NOT provably empty — no metadata answer, execution correct.
    ds = _ds(pq_path).join(_ds(pq_path).limit(0), on="x", how="left")
    assert _answer_count(ds._plan, ds._sources) is None
    assert ds.count() == _collect(ds._plan, ds._sources, ds.columns).num_rows


def test_asof_join_empty_left_counts_zero_from_metadata(pq_path):
    # ASOF is left-style: an empty left → empty result, answered from metadata.
    ds = _ds(pq_path).limit(0).join_asof(_ds(pq_path), on="x")
    answer = _answer_count(ds._plan, ds._sources)
    assert answer == 0
    assert _collect(ds._plan, ds._sources, ds.columns).num_rows == 0
