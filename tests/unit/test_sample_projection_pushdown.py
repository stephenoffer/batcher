"""Projection pushdown must not prune columns beneath a `Sample`.

`Sample` keeps a row iff a seeded hash of **all** its values falls under the fraction. So
unlike every other schema-preserving operator, dropping a column beneath it changes which
ROWS survive. Pushing a projection through it made `sample(f).select("k")` return a
different row count than `sample(f)` — a rewrite that silently changed the result.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher.kyber.rules.projections import required_columns_per_source

pytestmark = pytest.mark.unit


@pytest.fixture
def table():
    n = 4_000
    return pa.table(
        {
            "k": (np.arange(n) % 20).astype("int64"),
            "v": (np.arange(n) % 97).astype("int64"),
            "w": (np.arange(n) % 13).astype("int64"),
        }
    )


def test_scan_beneath_a_sample_keeps_every_column(table):
    """The plan-level fact: the scan under a `Sample` must read its full schema."""
    plan = bt.from_arrow(table).sample(0.1, seed=42).select("k")._plan
    assert sorted(required_columns_per_source(plan)[0]) == ["k", "v", "w"]


def test_scan_above_a_sample_still_prunes(table):
    """A projection that does NOT cross a sample must still prune (no perf regression)."""
    plan = bt.from_arrow(table).select("k", "v").group_by("k").agg(s=col("v").sum())._plan
    assert sorted(required_columns_per_source(plan)[0]) == ["k", "v"]


def test_a_projection_after_a_sample_does_not_change_the_sampled_rows(table):
    """The semantics the rewrite must preserve: sample the relation, THEN project."""
    ds = bt.from_arrow(table)
    full = ds.sample(0.1, seed=42).collect()
    projected = ds.sample(0.1, seed=42).select("k").collect()
    assert projected.num_rows == full.num_rows
    assert projected.column("k").to_pylist() == full.column("k").to_pylist()


def test_an_aggregate_over_a_sample_sees_the_sampled_rows(table):
    """`sample(f).group_by(k).agg(sum(v))` must total the same as summing the sampled rows."""
    import pyarrow.compute as pc

    ds = bt.from_arrow(table)
    sampled = ds.sample(0.1, seed=42).collect()
    grouped = ds.sample(0.1, seed=42).group_by("k").agg(s=col("v").sum()).collect()
    assert pc.sum(grouped.column("s")).as_py() == pc.sum(sampled.column("v")).as_py()
