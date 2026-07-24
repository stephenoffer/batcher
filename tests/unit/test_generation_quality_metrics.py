"""Reference-free generation-quality metrics — corpus signals over an output column alone.

These score a generated-text column without a gold reference (diversity, verbosity, empty rate,
refusal rate, truncation proxy). Each is a single mergeable aggregate, so the contract to pin is
the *corpus value*: the exact number over a small hand-computable batch, plus that it composes
inside ``group_by`` the way any other aggregate does.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.unit


def test_distinct_token_ratio_is_unique_over_total_averaged() -> None:
    # "cat sat mat" -> 3 unique / 3 = 1.0 ; "cat cat cat cat" -> 1/4 = 0.25 ; mean 0.625.
    ds = bt.from_pydict({"o": ["cat sat mat", "cat cat cat cat"]})
    got = ds.agg(d=bt.distinct_token_ratio("o")).to_pydict()["d"][0]
    assert got == pytest.approx(0.625)


def test_distinct_token_ratio_normalizes_case_and_punctuation() -> None:
    # SQuAD normalization folds case and strips punctuation, so these are one repeated token.
    ds = bt.from_pydict({"o": ["Cat, cat. CAT!"]})
    got = ds.agg(d=bt.distinct_token_ratio("o")).to_pydict()["d"][0]
    assert got == pytest.approx(1.0 / 3.0)


def test_mean_output_tokens_averages_the_estimate() -> None:
    # estimate = floor(len / chars_per_token): 8/4=2, 4/4=1 ; mean 1.5.
    ds = bt.from_pydict({"o": ["12345678", "1234"]})
    got = ds.agg(t=bt.mean_output_tokens("o", chars_per_token=4.0)).to_pydict()["t"][0]
    assert got == pytest.approx(1.5)


def test_empty_generation_rate_counts_blank_and_whitespace() -> None:
    ds = bt.from_pydict({"o": ["an answer", "   ", "", "another"]})
    got = ds.agg(e=bt.empty_generation_rate("o")).to_pydict()["e"][0]
    assert got == pytest.approx(0.5)


def test_refusal_rate_matches_the_is_refusal_detector() -> None:
    ds = bt.from_pydict(
        {
            "o": [
                "I'm sorry, I can't help with that.",
                "As an AI, I am unable to comply.",
                "The capital of France is Paris.",
                "42",
            ]
        }
    )
    got = ds.agg(r=bt.refusal_rate("o")).to_pydict()["r"][0]
    assert got == pytest.approx(0.5)


def test_truncation_rate_excludes_empty_and_needs_terminal_punctuation() -> None:
    # "A sentence." ends in '.' -> ok ; "cut off" -> truncated ; "" -> excluded.
    # non-empty = 2, truncated = 1 -> 0.5.
    ds = bt.from_pydict({"o": ["A sentence.", "cut off", ""]})
    got = ds.agg(t=bt.truncation_rate("o")).to_pydict()["t"][0]
    assert got == pytest.approx(0.5)


def test_generation_quality_metrics_compose_with_group_by() -> None:
    ds = bt.from_pydict(
        {
            "model": ["a", "a", "b", "b"],
            "o": ["good answer here", "the the the the", "fine", "ok"],
        }
    )
    out = ds.group_by("model").agg(d=bt.distinct_token_ratio("o")).to_pydict()
    assert set(out["model"]) == {"a", "b"}
    assert len(out["d"]) == 2
