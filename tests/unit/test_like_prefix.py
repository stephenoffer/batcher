"""Plan-shape unit tests for `like_prefix_to_range`."""

from __future__ import annotations

import json

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.normalize import _prefix_upper_bound, like_prefix_to_range
from batcher.plan.logical import Filter


def _t():
    return bt.from_pydict({"name": ["apple", "banana"], "v": [1, 2]})


def test_rule_registered():
    assert "like_prefix_to_range" in {r.name for r in DEFAULT_REGISTRY.rules()}


def test_prefix_like_becomes_range():
    plan = _t().filter(col("name").str.like("ab%"))._plan
    out = like_prefix_to_range(plan)
    assert isinstance(out, Filter)
    pred = out.predicate.to_ir()
    assert pred["op"] == "and"
    # lower bound col >= 'ab', upper bound col < 'ac'
    assert pred["left"]["op"] == "ge" and pred["left"]["right"]["value"]["str"] == "ab"
    assert pred["right"]["op"] == "lt" and pred["right"]["right"]["value"]["str"] == "ac"


def test_full_optimizer_drops_like():
    plan = _t().filter(col("name").str.like("ap%"))._plan
    ir = json.dumps(Optimizer().optimize(plan).ir)
    assert '"like"' not in ir
    assert '"ge"' in ir and '"lt"' in ir


def test_idempotent():
    plan = _t().filter(col("name").str.like("ab%"))._plan
    once = like_prefix_to_range(plan)
    assert like_prefix_to_range(once).to_ir() == once.to_ir()


# --- _prefix_upper_bound -------------------------------------------------------


def test_upper_bound_basic():
    assert _prefix_upper_bound("abc%") == "abd"
    assert _prefix_upper_bound("a%") == "b"


def test_upper_bound_rejects_non_prefix():
    assert _prefix_upper_bound("%") is None  # empty prefix
    assert _prefix_upper_bound("a_c%") is None  # underscore wildcard
    assert _prefix_upper_bound("a%b") is None  # not anchored at end
    assert _prefix_upper_bound("ab\\%%") is None  # escape in prefix
    assert _prefix_upper_bound("abc") is None  # no trailing %


def test_upper_bound_rejects_unsafe_last_char():
    assert _prefix_upper_bound("a~%") is not None  # '~' = 0x7E, incrementable
    assert _prefix_upper_bound("a\x7f%") is None  # 0x7F not safely incrementable
