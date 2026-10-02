"""`with_exact_join_keys` proves an in-memory join key's uniqueness by counting it.

Uniqueness licenses the additive aggregate pushdowns, and an estimate must never stand in for
it (an HLL count once "proved" a non-unique key unique and halved a `SUM`). So the count is
EXACT, only for integer columns a join actually names, and only below a row cap.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.api import source_stats as ss
from batcher.api.source_stats import collect_source_stats, with_exact_join_keys
from batcher.plan.stats import Provenance

pytestmark = pytest.mark.unit


def _joined(left: pa.Table, right: pa.Table, on: tuple[str, str]):
    ds = bt.from_arrow(left).join(bt.from_arrow(right), left_on=on[0], right_on=on[1])
    stats = collect_source_stats(ds._sources, None, need_columns=set())
    return ds, with_exact_join_keys(ds._sources, stats, ds._plan)


def test_an_integer_join_key_gets_an_exact_distinct_count():
    left = pa.table({"k": [1, 2, 2, 3], "v": [1, 2, 3, 4]})
    right = pa.table({"rk": [1, 2, 3], "name": ["a", "b", "c"]})
    _, (lstats, rstats) = _joined(left, right, ("k", "rk"))
    assert (lstats.columns["k"].ndv, rstats.columns["rk"].ndv) == (3, 3)
    assert lstats.columns["k"].ndv_is_exact and rstats.columns["rk"].ndv_is_exact
    # Only join keys are counted: a column the join does not name is left alone.
    assert rstats.columns.get("name") is None or rstats.columns["name"].ndv is None


def test_a_string_key_is_not_counted():
    left = pa.table({"k": ["x", "y"], "v": [1, 2]})
    right = pa.table({"rk": ["x", "y"], "w": [3, 4]})
    _, (lstats, _r) = _joined(left, right, ("k", "rk"))
    assert lstats.columns.get("k") is None or lstats.columns["k"].ndv is None


def test_a_relation_over_the_cap_is_not_counted(monkeypatch):
    left = pa.table({"k": [1, 2, 3], "v": [1, 2, 3]})
    right = pa.table({"rk": [1, 2, 3], "w": [1, 2, 3]})
    monkeypatch.setattr(ss, "_EXACT_KEY_MAX_ROWS", 2)
    _, (lstats, _r) = _joined(left, right, ("k", "rk"))
    assert lstats.columns.get("k") is None or lstats.columns["k"].ndv is None
    # Positive control: at the default cap the same key is counted.
    monkeypatch.undo()
    _, (lstats, _r) = _joined(left, right, ("k", "rk"))
    assert lstats.columns["k"].ndv == 3
    assert lstats.columns["k"].ndv_provenance is Provenance.EXACT


def test_a_comma_join_equality_counts_as_a_key():
    session = bt.Session()
    session.register("a", pa.table({"k": [1, 2, 3]}))
    session.register("b", pa.table({"j": [1, 1, 2]}))
    ds = session.sql("SELECT * FROM a, b WHERE k = j")
    stats = collect_source_stats(ds._sources, None, need_columns=set())
    counted = with_exact_join_keys(ds._sources, stats, ds._plan)
    assert sorted(c.columns["k" if "k" in c.columns else "j"].ndv for c in counted) == [2, 3]
