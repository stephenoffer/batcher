"""Differential coverage for `sequence` (DuckDB `generate_series`, inclusive)."""

from __future__ import annotations

import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential


def test_sequence_matches_generate_series(duck):
    from conftest import assert_same

    ds = bt.from_pydict({"a": [1, 2, 10], "b": [5, 2, 8]})
    duck.register("t", ds.collect())
    out = ds.select(s=bt.sequence(col("a"), col("b"))).collect()
    assert_same(out, duck.sql("SELECT generate_series(a, b) AS s FROM t"))


def test_sequence_with_step_and_literals(duck):
    from conftest import assert_same

    ds = bt.from_pydict({"a": [0, 1]})
    duck.register("t", ds.collect())
    out = ds.select(s=bt.sequence(0, 10, 2)).collect()
    assert_same(out, duck.sql("SELECT generate_series(0, 10, 2) AS s FROM t"))
