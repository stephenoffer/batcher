"""The positive control for every `"__cross_key" not in explain()` assertion.

A cartesian product lowers to a hash join on a synthetic `__cross_key` column, and
`explain()` names that key. Several SQL tests prove a comma join became an equi-join by
asserting the token is *absent*; that only means something while a real cross join still
renders it. This file pins that it does, and that the same relation joined on a real key
does not, so the absence assertions elsewhere keep discriminating the two plans.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same

pytestmark = pytest.mark.differential

_T = {"id": [1, 2, 3], "g": ["a", "b", "a"]}
_U = {"k": [1, 2], "w": [100, 300]}


def _session() -> bt.Session:
    s = bt.Session()
    s.register("t", bt.from_pydict(_T))
    s.register("u", bt.from_pydict(_U))
    return s


def test_a_sql_cross_join_renders_its_synthetic_key(duck):
    query = "SELECT t.id, u.w FROM t, u"
    ds = _session().sql(query)
    assert "__cross_key" in ds.explain()
    duck.register("t", pa.table(_T))
    duck.register("u", pa.table(_U))
    assert_same(ds.collect(), duck.sql(query))


def test_a_dataframe_cross_join_renders_its_synthetic_key():
    t, u = bt.from_pydict(_T), bt.from_pydict(_U)
    assert "__cross_key" in t.join(u, how="cross").explain()


def test_the_same_join_on_a_real_key_does_not():
    ds = _session().sql("SELECT t.id, u.w FROM t, u WHERE t.id = u.k")
    assert "__cross_key" not in ds.explain()
