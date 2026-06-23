"""Pythonic indexing sugar on `Dataset`: `ds[col]`, `ds[[cols]]`, `ds[:n]`, `len`."""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Col, Expr

pytestmark = pytest.mark.integration


def _ds():
    return bt.from_pydict({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})


def test_getitem_column_returns_expr():
    e = _ds()["a"]
    assert isinstance(e, Expr) and isinstance(e, Col)
    # Usable like any expression.
    out = _ds().select(doubled=_ds()["a"] * 2).to_pydict()
    assert out["doubled"] == [2, 4, 6, 8, 10]


def test_getitem_list_projects():
    ds = _ds()[["b", "a"]]
    assert ds.columns == ["b", "a"]


def test_getitem_slice():
    assert _ds()[:2].to_pydict()["a"] == [1, 2]
    assert _ds()[2:4].to_pydict()["a"] == [3, 4]
    assert _ds()[1:].to_pydict()["a"] == [2, 3, 4, 5]


def test_getitem_slice_rejects_step_and_negative():
    with pytest.raises(PlanError):
        _ds()[::2]
    with pytest.raises(PlanError):
        _ds()[-1:]


def test_len():
    assert len(_ds()) == 5
    assert len(_ds().filter(bt.col("a") > 2)) == 3
