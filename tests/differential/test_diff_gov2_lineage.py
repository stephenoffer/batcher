"""End-to-end column lineage through the public API for the ASOF join.

Lineage is a pure plan analysis (nothing executes), but it must be correct on the plan
the *public* surface builds. This exercises `Dataset.join_asof(...).lineage()` and asserts
each output column names exactly the source column it is fed by — the regression being that
an ASOF join fell through to the opaque catch-all, which attributed a left-derived column
to the right source and dropped its true left origin.
"""

from __future__ import annotations

import os
import tempfile

import pytest

import batcher as bt

pytestmark = pytest.mark.differential


def test_asof_join_lineage_names_the_correct_side() -> None:
    d = tempfile.mkdtemp()
    trades = os.path.join(d, "trades.parquet")
    quotes = os.path.join(d, "quotes.parquet")
    bt.from_pydict({"t": [1, 2, 3], "price": [10.0, 20.0, 30.0]}).write(trades, format="parquet")
    bt.from_pydict({"t": [1, 2, 3], "price": [9.0, 19.0, 29.0]}).write(quotes, format="parquet")

    joined = bt.read.parquet(trades).join_asof(bt.read.parquet(quotes), on="t", suffix="_q")
    lin = joined.lineage()

    # `price` is the left (trades) column; `price_q` is the right (quotes) column. Before
    # the fix both reported every origin from both sides (with the left origin dropped on
    # the same-named `price`/`t` collision), so `price` named the wrong table.
    assert lin["price"] == [f"{trades}.price"]
    assert lin["price_q"] == [f"{quotes}.price"]
    assert lin["t"] == [f"{trades}.t"]
