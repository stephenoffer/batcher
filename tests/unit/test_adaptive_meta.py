"""Plan-shape unit tests for the `adaptive_meta` provenance-driven rules.

Each rule gets: a fires test (the rewrite yields the intended shape), an idempotence
test (rewriting twice equals once), and a does-not-fire test (the rule stays its hand
when the row count is not *provably* EXACT, or the cap/offset legitimately applies).
Result-correctness vs DuckDB lives in the differential suite.
"""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra.adaptive_meta import (  # importing registers the rules
    drop_inert_limit,
    empty_limit_past_offset,
    fold_exact_empty_input,
)
from batcher.plan.logical import Filter, Limit


def _rewrite(ds):
    """Run the full logical rewrite pipeline over a dataset's plan."""
    return Optimizer(sources=ds._sources).logical_rewrite(ds._plan)


def _ctx(ds):
    return Optimizer(sources=ds._sources)._context()


def _t(rows=3):
    return bt.from_pydict({"x": list(range(rows)), "y": [i * 10 for i in range(rows)]})


def _empty():
    return bt.from_pydict({"x": [], "y": []})


# --- registration --------------------------------------------------------------


def test_rules_registered():
    names = {r.name for r in DEFAULT_REGISTRY.rules()}
    assert {"drop_inert_limit", "empty_limit_past_offset", "fold_exact_empty_input"} <= names


# --- drop_inert_limit ----------------------------------------------------------


def test_drop_inert_limit_fires():
    ds = _t(3)
    out = _rewrite(ds.limit(10))  # cap 10 ≥ 3 exact rows → inert
    assert not isinstance(out, Limit)


def test_drop_inert_limit_boundary_equal():
    ds = _t(3)
    out = _rewrite(ds.limit(3))  # exactly the row count → still every row, drop it
    assert not isinstance(out, Limit)


def test_drop_inert_limit_kept_when_cap_binds():
    ds = _t(5)
    out = _rewrite(ds.limit(2))  # cap 2 < 5 rows → the limit truly caps, keep it
    assert isinstance(out, Limit) and out.n == 2


def test_drop_inert_limit_kept_without_exact_size():
    ds = _t(5)
    # A filter downgrades provenance away from EXACT, so the input size is only
    # estimated — the limit must NOT be dropped even though 100 ≥ any real count.
    out = _rewrite(ds.filter(col("x") > 0).limit(100))
    assert any(isinstance(n, Limit) for n in _walk(out))


def test_drop_inert_limit_kept_with_offset():
    ds = _t(3)
    plan = ds.limit(10, offset=1)._plan  # non-zero offset skips a row → not inert
    assert drop_inert_limit(plan, _ctx(ds)) is None


def test_drop_inert_limit_idempotent():
    ds = _t(3)
    once = _rewrite(ds.limit(10))
    twice = Optimizer(sources=ds._sources).logical_rewrite(once)
    assert once.to_ir() == twice.to_ir()


# --- empty_limit_past_offset ---------------------------------------------------


def test_empty_limit_past_offset_fires():
    ds = _t(3)
    out = _rewrite(ds.limit(10, offset=3))  # offset 3 ≥ 3 rows → skips everything
    assert isinstance(out, Limit) and out.n == 0


def test_empty_limit_past_offset_kept_within_range():
    ds = _t(5)
    plan = ds.limit(10, offset=2)._plan  # offset 2 < 5 rows → some rows survive
    assert empty_limit_past_offset(plan, _ctx(ds)) is None


def test_empty_limit_past_offset_ignores_zero_offset():
    ds = _t(3)
    plan = ds.limit(10)._plan  # offset 0 is drop_inert_limit's job, not this rule's
    assert empty_limit_past_offset(plan, _ctx(ds)) is None


def test_empty_limit_past_offset_idempotent():
    ds = _t(3)
    once = _rewrite(ds.limit(10, offset=3))
    twice = Optimizer(sources=ds._sources).logical_rewrite(once)
    assert once.to_ir() == twice.to_ir()


# --- fold_exact_empty_input ----------------------------------------------------


def test_fold_exact_empty_input_fires():
    ds = _empty()
    out = _rewrite(ds.filter(col("x") > 1).sort("x"))  # filter+sort over empty source
    assert isinstance(out, Limit) and out.n == 0


def test_fold_exact_empty_input_kept_when_nonempty():
    ds = _t(3)
    plan = ds.filter(col("x") > 1)._plan
    # A non-empty EXACT source: the filter output is unknown, never proven empty.
    assert fold_exact_empty_input(plan, _ctx(ds)) is None


def test_fold_exact_empty_input_skips_existing_marker():
    ds = _t(3)
    # `Filter` over the canonical empty marker is propagate_empty_relation's job;
    # this rule must not also fire on it (would be redundant / non-idempotent).
    plan = ds.limit(0).filter(col("x") > 1)._plan
    assert fold_exact_empty_input(plan, _ctx(ds)) is None


def test_fold_exact_empty_input_idempotent():
    ds = _empty()
    once = _rewrite(ds.filter(col("x") > 1).sort("x"))
    twice = Optimizer(sources=ds._sources).logical_rewrite(once)
    assert once.to_ir() == twice.to_ir()


# --- helpers -------------------------------------------------------------------


def _walk(node):
    from batcher.plan.visitor import walk

    return list(walk(node))


def test_filter_marker_helper_untouched_shape():
    # Sanity: a plain non-empty filter is left as a Filter by these rules.
    ds = _t(4)
    out = _rewrite(ds.filter(col("x") >= 0).limit(100))
    assert any(isinstance(n, Filter) for n in _walk(out))
