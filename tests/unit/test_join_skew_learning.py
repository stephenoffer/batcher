"""Metadata-driven join-skew learning: hot keys measured by the detection pre-pass
are persisted by join shape and reused on later runs, so salting engages without
re-running the pre-pass. The loop is result-preserving (salting changes scheduling,
not the joined relation), so these tests pin the persistence semantics; the
distributed equivalence suite proves correctness end to end.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.dist.skew import join_skew_key, load_learned_hot_keys, persist_hot_keys

pytestmark = pytest.mark.unit


def _join_plan():
    a = pa.table({"id": [1, 2, 3], "v": [10, 20, 30]})
    b = pa.table({"id": [1, 2, 3], "w": [1, 2, 3]})
    return bt.from_arrow(a).join(bt.from_arrow(b), on="id")._plan


def test_join_skew_key_is_stable_and_shape_specific():
    plan = _join_plan()
    k1 = join_skew_key("LIR", "RIR", plan)
    k2 = join_skew_key("LIR", "RIR", plan)
    assert k1 == k2 and len(k1) == 16  # deterministic short hash
    # A different shape (different side IR) keys differently.
    assert join_skew_key("OTHER", "RIR", plan) != k1


def test_learned_hot_keys_round_trip_and_none_vs_empty():
    plan = _join_plan()
    key = join_skew_key("LIR", "RIR", plan)

    # Never measured → None (so the caller knows to run the pre-pass).
    assert load_learned_hot_keys(key) is None

    # A measured non-empty hot set round-trips and is what later runs salt on.
    persist_hot_keys(key, ["7", "42"])
    assert load_learned_hot_keys(key) == ["7", "42"]

    # A measured EMPTY result ("not skewed") is distinct from never-measured, so a
    # non-skewed shape never re-runs the pre-pass.
    empty_key = join_skew_key("LIR2", "RIR2", plan)
    persist_hot_keys(empty_key, [])
    assert load_learned_hot_keys(empty_key) == []
    assert load_learned_hot_keys(empty_key) is not None
