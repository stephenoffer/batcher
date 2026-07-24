"""Token and cost aggregates — sizing an LLM run from the data.

These aggregate the tokenizer-free ``estimate_tokens`` heuristic (characters over
``chars_per_token``, truncated to an int) to a corpus number, so they are pinned to that arithmetic
on a hand-countable batch: the total tokens, the over-budget fraction, and a per-row quantile.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def test_total_token_estimate_sums_the_per_row_estimate() -> None:
    # "12345678" -> 8 // 4 = 2 ; "1234" -> 1 ; total 3.
    ds = bt.from_pydict({"o": ["12345678", "1234"]})
    assert ds.agg(t=bt.total_token_estimate("o", chars_per_token=4.0)).to_pydict()["t"][0] == 3


def test_total_token_estimate_scales_with_chars_per_token() -> None:
    # 8 chars: at 2 chars/token -> 4 tokens ; at 8 -> 1 token.
    ds = bt.from_pydict({"o": ["12345678"]})
    assert ds.agg(t=bt.total_token_estimate("o", chars_per_token=2.0)).to_pydict()["t"][0] == 4
    assert ds.agg(t=bt.total_token_estimate("o", chars_per_token=8.0)).to_pydict()["t"][0] == 1


def test_token_budget_exceed_rate_is_the_over_budget_fraction() -> None:
    # "abcd" -> 1 token (fits 2) ; "abcdefghijkl" -> 3 tokens (exceeds 2). Rate 1/2.
    ds = bt.from_pydict({"o": ["abcd", "abcdefghijkl"]})
    got = ds.agg(r=bt.token_budget_exceed_rate("o", budget=2)).to_pydict()["r"][0]
    assert got == pytest.approx(0.5)


def test_token_budget_exceed_rate_is_zero_when_all_fit() -> None:
    ds = bt.from_pydict({"o": ["a", "bb", "ccc"]})
    got = ds.agg(r=bt.token_budget_exceed_rate("o", budget=100)).to_pydict()["r"][0]
    assert got == pytest.approx(0.0)


def test_token_estimate_quantile_reads_the_tail() -> None:
    # per-row token estimates: 1, 2, 3 ; median ~ 2, max ~ 3.
    ds = bt.from_pydict({"o": ["1234", "12345678", "123456789012"]})
    median = ds.agg(p=bt.token_estimate_quantile("o", q=0.5)).to_pydict()["p"][0]
    top = ds.agg(p=bt.token_estimate_quantile("o", q=1.0)).to_pydict()["p"][0]
    assert round(median) == 2
    assert round(top) == 3


def test_token_cost_metrics_compose_with_group_by() -> None:
    ds = bt.from_pydict({"model": ["a", "a", "b"], "o": ["1234", "12345678", "1234"]})
    out = ds.group_by("model").agg(t=bt.total_token_estimate("o")).sort("model").to_pydict()
    assert out["model"] == ["a", "b"]
    # Group a sums to three tokens, group b to one.
    assert out["t"] == [3, 1]
