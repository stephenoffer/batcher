"""Distributed join equivalence: single-node == multi-partition for joins.

Regression coverage for a bug where the non-adaptive distributed path shipped the
*raw* logical plan to the workers. A SQL comma join (``FROM a, b WHERE a.k = b.k``)
raw-lowers to an inner join on a constant ``__cross_key`` with the equality stranded
in a Filter above it; distributed raw, every row hashed to one bucket (a cross
product) and the shuffle collapsed onto a single reducer. Distributing the *optimized*
logical plan (``derive_join_keys`` turns the WHERE-equality into real join keys) is the
fix, and these assert the result is identical single-node vs distributed.
"""

from __future__ import annotations

import collections

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential

_LEFT = pa.table({"k": [1, 2, 3, 1, 2, 5, 3, 4] * 8, "v": list(range(64))})
_RIGHT = pa.table({"k": [1, 2, 3, 4], "g": ["a", "b", "c", "d"]})


def _multiset(d: dict) -> collections.Counter:
    return collections.Counter(tuple(row) for row in zip(*d.values(), strict=True))


def test_explicit_join_single_node_equals_distributed():
    ds = bt.from_arrow(_LEFT).join(bt.from_arrow(_RIGHT), left_on="k", right_on="k", how="inner")
    single = _multiset(ds.collect().to_pydict())
    multi = _multiset(ds.collect(distributed=True, num_workers=3).to_pydict())
    assert single == multi


def test_comma_join_single_node_equals_distributed():
    # The regression: a SQL comma join must hash by the real key (k), not a constant
    # cross key, or the distributed shuffle concentrates every row on one reducer.
    s = bt.Session()
    s.register("l", _LEFT)
    s.register("r", _RIGHT)
    q = s.sql("SELECT v, g FROM l, r WHERE l.k = r.k")
    single = _multiset(q.collect().to_pydict())
    multi = _multiset(q.collect(distributed=True, num_workers=3).to_pydict())
    assert single == multi


def test_fused_join_aggregate_single_node_equals_distributed():
    # group_by the join key ⇒ each group is bucket-local, so the reducer aggregates its
    # joined bucket (exchange elimination) — the result must still match single-node.
    ds = (
        bt.from_arrow(_LEFT)
        .join(bt.from_arrow(_RIGHT), left_on="k", right_on="k", how="inner")
        .group_by("k")
        .agg(total=col("v").sum(), n=col("v").count())
    )
    single = _multiset(ds.collect().to_pydict())
    multi = _multiset(ds.collect(distributed=True, num_workers=3).to_pydict())
    assert single == multi
