"""Unit tests for the script / character-set composition metrics."""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.plan.functions.metrics.text import script as s

pytestmark = pytest.mark.unit

_CORPUS = ["hello 世界", "plain english", "привет", "مرحبا", "party 🎉"]


def test_cjk_rate() -> None:
    ds = bt.from_pydict({"o": _CORPUS})
    got = ds.agg(m=s.cjk_rate("o")).to_pydict()["m"][0]
    assert got == pytest.approx(1 / 5)


def test_cyrillic_rate() -> None:
    ds = bt.from_pydict({"o": _CORPUS})
    got = ds.agg(m=s.cyrillic_rate("o")).to_pydict()["m"][0]
    assert got == pytest.approx(1 / 5)


def test_arabic_rate() -> None:
    ds = bt.from_pydict({"o": _CORPUS})
    got = ds.agg(m=s.arabic_rate("o")).to_pydict()["m"][0]
    assert got == pytest.approx(1 / 5)


def test_emoji_rate() -> None:
    ds = bt.from_pydict({"o": _CORPUS})
    got = ds.agg(m=s.emoji_rate("o")).to_pydict()["m"][0]
    assert got == pytest.approx(1 / 5)


def test_latin_only_rate() -> None:
    ds = bt.from_pydict({"o": _CORPUS})
    got = ds.agg(m=s.latin_only_rate("o")).to_pydict()["m"][0]
    assert got == pytest.approx(1 / 5)


def test_group_by() -> None:
    ds = bt.from_pydict(
        {"lang": ["a", "a", "b"], "o": ["世界", "ok", "привет"]}
    )
    got = ds.group_by("lang").agg(m=s.cjk_rate("o")).sort("lang").to_pydict()
    assert got["lang"] == ["a", "b"]
    assert got["m"] == pytest.approx([0.5, 0.0])
