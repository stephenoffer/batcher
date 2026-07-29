"""Unit tests for the Markdown-formatting corpus metrics.

Each metric is a mergeable rate aggregate over the string primitives; these pin the corpus value
on hand-checkable inputs and confirm the rates compose inside ``group_by``.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.plan.functions.metrics.text import formatting as f

pytestmark = pytest.mark.unit


def test_heading_rate() -> None:
    ds = bt.from_pydict({"o": ["# Title\ntext", "plain text", "- item one"]})
    assert ds.agg(h=f.heading_rate("o")).to_pydict()["h"][0] == pytest.approx(1 / 3)


def test_bullet_list_rate() -> None:
    ds = bt.from_pydict({"o": ["- item one", "plain text", "line1\n* two"]})
    assert ds.agg(b=f.bullet_list_rate("o")).to_pydict()["b"][0] == pytest.approx(2 / 3)


def test_numbered_list_rate() -> None:
    ds = bt.from_pydict({"o": ["1. first", "plain text", "line\n2. second"]})
    assert ds.agg(n=f.numbered_list_rate("o")).to_pydict()["n"][0] == pytest.approx(2 / 3)


def test_markdown_link_rate() -> None:
    ds = bt.from_pydict({"o": ["see [docs](http://x)", "plain text", "[a](b)"]})
    assert ds.agg(m=f.markdown_link_rate("o")).to_pydict()["m"][0] == pytest.approx(2 / 3)


def test_table_rate() -> None:
    ds = bt.from_pydict({"o": ["| a | b |\n|---|---|", "plain text", "no table"]})
    assert ds.agg(t=f.table_rate("o")).to_pydict()["t"][0] == pytest.approx(1 / 3)


def test_code_block_present_rate() -> None:
    ds = bt.from_pydict({"o": ["```py\ncode\n```", "plain text", "no fence"]})
    assert ds.agg(c=f.code_block_present_rate("o")).to_pydict()["c"][0] == pytest.approx(1 / 3)


def test_group_by_composition() -> None:
    ds = bt.from_pydict(
        {
            "model": ["a", "a", "b", "b"],
            "o": ["# h", "plain", "# h1\n## h2", "- bullet"],
        }
    )
    out = ds.group_by("model").agg(h=f.heading_rate("o")).sort("model").to_pydict()
    assert out["model"] == ["a", "b"]
    assert out["h"][0] == pytest.approx(0.5)
    assert out["h"][1] == pytest.approx(0.5)
