"""Proportional stratified subsampling.

Where the balancing samplers equalize the classes, `stratified_sample` preserves their
relative sizes, so the tests pin exactly that: each stratum keeps ``floor(fraction * n)`` rows,
the selection is reproducible from the seed and partition-independent, and a fraction outside
``(0, 1]`` is rejected.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.sampling import class_counts, stratified_sample

pytestmark = pytest.mark.unit


def test_keeps_the_fraction_of_every_stratum() -> None:
    ds = bt.from_pydict({"g": ["a"] * 100 + ["b"] * 20, "x": list(range(120))})
    assert class_counts(stratified_sample(ds, "g", 0.5, seed=1), "g") == {"a": 50, "b": 10}


def test_uses_floor_for_uneven_fractions() -> None:
    ds = bt.from_pydict({"g": ["a"] * 7 + ["b"] * 3, "x": list(range(10))})
    # floor(0.5 * 7) = 3, floor(0.5 * 3) = 1.
    assert class_counts(stratified_sample(ds, "g", 0.5, seed=0), "g") == {"a": 3, "b": 1}


def test_is_reproducible_from_the_seed() -> None:
    ds = bt.from_pydict({"g": ["a"] * 50 + ["b"] * 50, "x": list(range(100))})
    first = set(stratified_sample(ds, "g", 0.4, seed=7).to_pydict()["x"])
    second = set(stratified_sample(ds, "g", 0.4, seed=7).to_pydict()["x"])
    assert first == second


def test_a_different_seed_selects_different_rows() -> None:
    ds = bt.from_pydict({"g": ["a"] * 100, "x": list(range(100))})
    a = set(stratified_sample(ds, "g", 0.5, seed=1).to_pydict()["x"])
    b = set(stratified_sample(ds, "g", 0.5, seed=2).to_pydict()["x"])
    assert a != b


def test_fraction_one_keeps_everything() -> None:
    ds = bt.from_pydict({"g": ["a", "a", "b"], "x": [1, 2, 3]})
    assert stratified_sample(ds, "g", 1.0).count() == 3


def test_rejects_a_fraction_out_of_range() -> None:
    ds = bt.from_pydict({"g": ["a"], "x": [1]})
    with pytest.raises(PlanError, match="fraction must be"):
        stratified_sample(ds, "g", 1.5)


def test_tiny_stratum_can_drop_out() -> None:
    # floor(0.1 * 5) = 0, so a stratum of 5 contributes nothing at a 10% sample.
    ds = bt.from_pydict({"g": ["a"] * 100 + ["b"] * 5, "x": list(range(105))})
    counts = class_counts(stratified_sample(ds, "g", 0.1, seed=0), "g")
    assert counts.get("a") == 10
    assert counts.get("b", 0) == 0
