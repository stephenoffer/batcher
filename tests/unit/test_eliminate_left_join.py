"""Plan-shape unit tests for `eliminate_left_join`."""

from __future__ import annotations

import json

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.joins import eliminate_left_join


def _fact():
    return bt.from_pydict({"k": [1, 2, 2, 3], "v": [10, 20, 30, 40]})


def _dim_unique():
    # GROUP BY k → provably unique on k.
    return (
        bt.from_pydict({"k": [1, 1, 2, 3], "p": [5, 6, 7, 8]}).group_by("k").agg(tot=col("p").sum())
    )


def _has_join(ir) -> bool:
    return '"hash_join"' in json.dumps(ir)


def test_rule_registered():
    assert "eliminate_left_join" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_left_join_to_aggregate_eliminated():
    ds = _fact().join(_dim_unique(), on="k", how="left").select("k", "v")
    ir = Optimizer().optimize(ds._plan).ir
    assert not _has_join(ir)  # the redundant left join is gone


def test_using_right_column_not_eliminated():
    ds = _fact().join(_dim_unique(), on="k", how="left").select("k", "v", "tot")
    ir = Optimizer().optimize(ds._plan).ir
    assert _has_join(ir)  # `tot` comes from R → join is needed


def test_inner_join_not_eliminated():
    ds = _fact().join(_dim_unique(), on="k", how="inner").select("k", "v")
    ir = Optimizer().optimize(ds._plan).ir
    assert _has_join(ir)  # inner also drops unmatched L rows → not eliminable


def test_non_unique_right_not_eliminated():
    # Plain (non-aggregated, no stats) right side → uniqueness can't be proven.
    dim = bt.from_pydict({"k": [1, 1, 2], "p": [5, 6, 7]})
    ds = _fact().join(dim, on="k", how="left").select("k", "v")
    ir = Optimizer().optimize(ds._plan).ir
    assert _has_join(ir)


def test_distinct_right_eliminated():
    dim = bt.from_pydict({"k": [1, 1, 2, 3]}).distinct()  # unique on its only column
    ds = _fact().join(dim, on="k", how="left").select("k", "v")
    ir = Optimizer().optimize(ds._plan).ir
    assert not _has_join(ir)


def test_direct_call_noop_when_not_project_over_join():
    # The rule matches Project nodes; a Project over a Scan (not a Join) → None.
    assert eliminate_left_join(_fact().select("k")._plan, None) is None
