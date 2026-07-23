"""Kyber inference-pipeline rules: filter pushdown below an opaque `map_batches`,
and dropping a large column once its last consumer (the UDF) has passed.

`map_batches` is the black-box operator: `fn` returns the whole output batch and the
engine does not re-attach untouched inputs, so a predicate can be moved below it only
when every column it reads is *declared preserved*. These tests pin the plan shape the
rules produce and — the load-bearing half — that they fire on nothing else, because a
filter that moved on an undeclared column is a wrong answer, not a slow one.
"""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.rules.projections import push_filter_through_map_batches
from batcher.plan.logical import Filter, MapBatches, Project
from batcher.plan.visitor import children


def _ident(batch: object) -> object:
    """A UDF the tests never execute — only its declared column contracts matter here."""
    return batch


def _ds():
    return bt.from_pydict({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})


def _find(plan: object, kind: type) -> list:
    """Every node of type `kind` in `plan`, pre-order."""
    found = [plan] if isinstance(plan, kind) else []
    for child in children(plan):
        found.extend(_find(child, kind))
    return found


def _optimized(plan: object) -> object:
    # `MapBatches.to_ir()` raises by design, so optimize to a logical plan, not to IR.
    return Optimizer().logical_rewrite(plan)


# --- the pushdown: filter sinks below the UDF only on preserved columns -------


def test_filter_pushed_below_map_batches_on_preserved_column():
    plan = _ds().ml.map_batches(_ident, preserves_columns=["x"]).filter(col("x") < 5)._plan
    # The unit rule fires directly: MapBatches rises to the top, the filter sinks under it.
    out = push_filter_through_map_batches(plan, None)
    assert isinstance(out, MapBatches)
    assert isinstance(out.input, Filter)
    # And the full optimizer reaches the same shape.
    opt = _optimized(plan)
    assert isinstance(opt, MapBatches)
    assert isinstance(opt.input, Filter)


def test_filter_not_pushed_when_preserves_columns_unset():
    # preserves_columns=None (the default): the fn may rewrite anything, so the filter
    # MUST stay above the UDF. This is the test that stops a wrong answer.
    plan = _ds().ml.map_batches(_ident).filter(col("x") < 5)._plan
    assert push_filter_through_map_batches(plan, None) is None
    opt = _optimized(plan)
    assert isinstance(opt, Filter)
    assert isinstance(opt.input, MapBatches)


def test_filter_not_pushed_on_undeclared_column():
    # `x` is preserved but the predicate reads `y`, which is not declared preserved:
    # the fn may have rewritten `y`, so the filter must not move.
    plan = _ds().ml.map_batches(_ident, preserves_columns=["x"]).filter(col("y") < 30)._plan
    assert push_filter_through_map_batches(plan, None) is None
    opt = _optimized(plan)
    assert isinstance(opt, Filter)
    assert isinstance(opt.input, MapBatches)


def test_mixed_predicate_splits_preserved_conjunct_below():
    # `x < 5 AND y < 30`: only the `x` conjunct is preserved, so it sinks below the UDF
    # while the `y` conjunct stays above it.
    plan = (
        _ds()
        .ml.map_batches(_ident, preserves_columns=["x"])
        .filter((col("x") < 5) & (col("y") < 30))
        ._plan
    )
    out = push_filter_through_map_batches(plan, None)
    assert isinstance(out, Filter)  # residual `y` filter on top
    assert isinstance(out.input, MapBatches)  # UDF in the middle
    assert isinstance(out.input.input, Filter)  # pushed `x` filter at the bottom


def test_rule_is_noop_off_map_batches():
    plan = _ds().filter(col("x") < 5)._plan  # filter directly over a scan
    assert push_filter_through_map_batches(plan, None) is None


def test_optimizer_is_idempotent_on_pushdown():
    plan = _ds().ml.map_batches(_ident, preserves_columns=["x"]).filter(col("x") < 5)._plan
    once = _optimized(plan)
    twice = _optimized(once)
    assert repr(once) == repr(twice)  # a second pass changes nothing


# --- the drop: a big input-only column is freed once the UDF has passed -------


def _wide_ds():
    return bt.from_pydict(
        {"id": [1, 2, 3], "score": [0.1, 0.2, 0.3], "img": [b"aaa", b"bbb", b"ccc"]}
    )


def test_large_column_dropped_above_map_batches():
    # The UDF reads `img` (kept alive beneath it) but nothing above the sort needs it.
    # After optimization a Project drops `img` right above the UDF, so the sort never
    # carries the wide column.
    plan = (
        _wide_ds()
        .ml.map_batches(_ident, input_columns=["img"])
        .sort("score")
        .select("id", "score")
        ._plan
    )
    opt = _optimized(plan)
    mbs = _find(opt, MapBatches)
    assert len(mbs) == 1
    mb = mbs[0]
    # `img` is still read from the source beneath the UDF (the fn needs it) ...
    assert "img" in mb.input.available_columns()
    # ... but the UDF's output is immediately narrowed, so nothing above carries `img`.
    projects_over_mb = [p for p in _find(opt, Project) if mb in children(p)]
    assert projects_over_mb, "expected a pruning Project directly above the map_batches"
    assert all("img" not in p.available_columns() for p in projects_over_mb)
    assert "img" not in opt.available_columns()


def test_no_drop_when_output_column_is_consumed():
    # When the final result keeps every column, there is nothing to drop and no Project
    # is inserted above the UDF.
    plan = _wide_ds().ml.map_batches(_ident, input_columns=["img"])._plan
    opt = _optimized(plan)
    assert isinstance(opt, MapBatches)  # no wrapping Project
    assert set(opt.available_columns()) == {"id", "score", "img"}


def test_drop_optimizer_is_idempotent():
    plan = (
        _wide_ds()
        .ml.map_batches(_ident, input_columns=["img"])
        .sort("score")
        .select("id", "score")
        ._plan
    )
    once = _optimized(plan)
    twice = _optimized(once)
    assert repr(once) == repr(twice)
