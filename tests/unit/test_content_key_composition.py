"""`LogicalPlan.content_key` composes its children's keys and keeps the identity it promises.

The key is "this subtree is the same computation": equal IR and equal identity suffixes (a
scan's schema). It used to hash every node's whole-subtree JSON, quadratic in plan size; it
now hashes each node's own IR with its children's keys standing in for their IR. These pin
that the composition kept the contract, in both directions, including at depth.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.plan.visitor import children

pytestmark = pytest.mark.unit


def _table(y_type=None):
    y = pa.array([4, 5, 6], type=y_type or pa.int64())
    return pa.table({"x": pa.array([1, 2, 3], type=pa.int64()), "y": y})


def _plan(literal: int = 1, y_type=None, table=None):
    t = table if table is not None else _table(y_type)
    left = bt.from_arrow(t).filter(bt.col("x") > literal)
    right = bt.from_arrow(t).group_by("x").agg(n=bt.col("y").count())
    return left.join(right, on="x").select("x", "y", "n").sort("x")._plan


def test_independently_built_equal_plans_key_equal():
    t = _table()
    assert _plan(table=t).content_key() == _plan(table=t).content_key()


def test_a_literal_at_depth_changes_the_root_key():
    t = _table()
    assert _plan(1, table=t).content_key() != _plan(2, table=t).content_key()


def test_a_scan_schema_at_depth_changes_the_root_key():
    """A scan's IR is only its source id; its schema is in its identity suffix. (A narrow
    integer is recorded already widened, so the types differ here by kind, not width.)"""
    assert _plan(y_type=pa.int64()).content_key() != _plan(y_type=pa.float64()).content_key()


def test_the_root_payload_names_its_children_by_key():
    """The composition itself: a child's key appears in its parent's payload, so a parent's
    key is built from it rather than from a re-serialized subtree."""
    plan = _plan()
    kids = list(children(plan))
    assert kids
    payload = plan._content_payload()
    for kid in kids:
        assert kid.content_key() in payload
