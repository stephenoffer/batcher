"""A wide aggregate must not go through the combiner tree and come back empty.

`shuffle_fan_in` (8 by default) is the point where the distributed aggregate stops reducing
its buckets flat and starts folding them through a tree of combiners. The two halves are meant
to be interchangeable — any reducer count is result-correct under the mergeable algebra — so
nothing about crossing that line should change an answer.

It changed the answer to *nothing*. When the aggregate was moved off the fixed ticket stage 0
onto a reserved stage block, `_tree_reduce` kept addressing its leaf partials at the literal
stage 0 and numbering its interior levels 1, 2, 3. Past `shuffle_fan_in` reducers it therefore
fetched tickets nobody had published, every bucket came back empty, and the query returned zero
rows — silently, because an unregistered ticket reads back as an empty bucket rather than an
error (the epoch invariant in `dist/shuffle_replication.py`). Measured on TPC-H q1 at sf10:
four rows at eight workers, **zero** at twelve, same data.

These cases run the same aggregate either side of the threshold and demand the same answer.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col, count
from batcher.config import active_config, config_context

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _need_ray():
    pytest.importorskip("ray")


@pytest.fixture(scope="module")
def table() -> pa.Table:
    rng = np.random.default_rng(11)
    return pa.table(
        {
            "k": rng.integers(0, 6, 30_000).astype("int64"),
            "wide": rng.integers(0, 5_000, 30_000).astype("int64"),
            "v": rng.integers(0, 100, 30_000).astype("int64"),
        }
    )


def _sorted(rows: list[dict], key: str) -> list[tuple]:
    return sorted(tuple(sorted(r.items())) for r in rows)


@pytest.mark.parametrize("key", ["k", "wide"])
def test_the_combiner_tree_returns_what_the_flat_reduce_does(table, key):
    """Either side of `shuffle_fan_in`, same answer. Both a low-cardinality key (which leaves
    most buckets empty) and a high-cardinality one, because the failure was independent of
    cardinality and a single-key test would not have said so."""
    ds = bt.from_arrow(table).group_by(key).agg(n=count(), s=col("v").sum())
    single = _sorted(ds.collect().to_pylist(), key)
    flat = _sorted(ds.collect(distributed=True, num_workers=4).to_pylist(), key)
    tree = _sorted(ds.collect(distributed=True, num_workers=12).to_pylist(), key)
    assert flat == single, "the flat reduce must match single-node"
    assert tree == single, "crossing shuffle_fan_in must not change the answer"


def test_a_tree_deep_enough_to_need_several_levels_still_agrees(table):
    """A fan-in of 2 forces the maximum number of interior levels for this worker count, so
    every level's ticket stage is exercised rather than just the first."""
    ds = bt.from_arrow(table).group_by("k").agg(n=count())
    single = _sorted(ds.collect().to_pylist(), "k")
    cfg = active_config()
    deep = cfg.replace(flow_control=dataclasses.replace(cfg.flow_control, shuffle_fan_in=2))
    with config_context(deep):
        got = _sorted(ds.collect(distributed=True, num_workers=8).to_pylist(), "k")
    assert got == single


def test_two_aggregates_in_one_query_do_not_share_ticket_stages(table):
    """The reserved block must be per-aggregate. Two wide aggregates on one fleet addressing
    the same stages would have the second overwrite the first's buckets — the collision the
    stage block exists to prevent, now that a tree claims more than one stage."""
    ds = bt.from_arrow(table)
    left = ds.group_by("k").agg(a=count())
    right = ds.group_by("k").agg(b=col("v").sum())
    joined = left.join(right, on="k")
    single = _sorted(joined.collect().to_pylist(), "k")
    got = _sorted(joined.collect(distributed=True, num_workers=12).to_pylist(), "k")
    assert got == single
