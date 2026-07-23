"""Plan-shape, idempotence and refusal tests for the `window_extra` rules.

Each rule gets a *fires* test, an *end-to-end* test through the real `Optimizer`, and the
refusals that keep it correct: an estimated (not proven) ndv, a column that is not provably
constant, an explicit frame that may be empty, the last order key (whose removal would change
the operator), and a `rank_limit` window (which filters rows).

Result-correctness vs DuckDB lives in `tests/differential/test_diff_window_extra.py`.
"""

from __future__ import annotations

import batcher as bt
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.registry import DEFAULT_REGISTRY
from batcher.kyber.rules.extra import window_extra as m
from batcher.plan.expr_ir import Col
from batcher.plan.logical import (
    Limit,
    Project,
    SortKeySpec,
    Window,
    WindowFuncSpec,
)
from batcher.plan.source_stats import SourceStatistics
from batcher.plan.stats import ColumnStat, Provenance
from batcher.plan.visitor import walk

_RULE_NAMES = {
    "constant_window_function_folding",
    "dedupe_window_functions",
    "drop_order_key_after_unique_key_in_window_order",
    "drop_order_key_equal_to_partition_key",
    "drop_order_key_proven_constant",
    "drop_order_keys_under_unbounded_frames",
    "drop_partition_key_proven_constant",
    "drop_redundant_unbounded_frame",
    "rank_limit_zero_to_empty",
    "simplify_window_over_single_row_partition",
}


def _exact(**kw) -> ColumnStat:
    return ColumnStat(provenance=Provenance.EXACT, **kw)


def _stats(rows: int, **columns) -> list[SourceStatistics]:
    return [SourceStatistics(row_count=rows, columns=dict(columns))]


def _ds(rows: int = 4):
    ids = list(range(1, rows + 1))
    return bt.from_pydict(
        {
            "id": ids,
            "g": ["a", "b"] * (rows // 2) or ["a"],
            "k": [7] * rows,
            "x": [i * 5 for i in ids],
        }
    )


def _ctx(ds, stats):
    return Optimizer(sources=ds._sources, source_stats=stats)._context()


def _rewrite(ds, plan, stats):
    return Optimizer(sources=ds._sources, source_stats=stats).logical_rewrite(plan)


def _windows(plan):
    return [n for n in walk(plan) if isinstance(n, Window)]


# The constant column `k` (7 in every row) and the unique column `id`, both proven EXACT.
_CONST = _exact(min=7, max=7, null_count=0)
_UNIQUE = _exact(ndv=4, null_count=0)


def test_all_rules_registered():
    assert {r.name for r in DEFAULT_REGISTRY.rules()} >= _RULE_NAMES
    assert set(m.__all__) == _RULE_NAMES


# --- drop_partition_key_proven_constant ----------------------------------------


def _part_plan(ds):
    return ds.window(partition_by=["k", "g"], order_by=["x"], functions={"rn": "row_number"})._plan


def test_drop_partition_key_proven_constant_fires():
    ds = _ds()
    out = m.drop_partition_key_proven_constant(_part_plan(ds), _ctx(ds, _stats(4, k=_CONST)))
    assert isinstance(out, Window)
    assert [k.name for k in out.partition_keys] == ["g"]
    assert m.drop_partition_key_proven_constant(out, _ctx(ds, _stats(4, k=_CONST))) is None


def test_drop_partition_key_refuses_unproven_constant():
    ds = _ds()
    stats = _stats(4, k=ColumnStat(min=7, max=7, null_count=0, provenance=Provenance.SKETCH))
    assert m.drop_partition_key_proven_constant(_part_plan(ds), _ctx(ds, stats)) is None


def test_drop_partition_key_refuses_null_bearing_column():
    ds = _ds()
    stats = _stats(4, k=_exact(min=7, max=7, null_count=1))  # a NULL is its own partition
    assert m.drop_partition_key_proven_constant(_part_plan(ds), _ctx(ds, stats)) is None


def test_drop_partition_key_fires_end_to_end():
    ds = _ds()
    out = _rewrite(ds, _part_plan(ds), _stats(4, k=_CONST))
    assert [len(w.partition_keys) for w in _windows(out)] == [1]


# --- drop_order_key_proven_constant --------------------------------------------


def test_drop_order_key_proven_constant_fires():
    ds = _ds()
    plan = ds.window(partition_by=["g"], order_by=["k", "x"], functions={"rn": "row_number"})._plan
    out = m.drop_order_key_proven_constant(plan, _ctx(ds, _stats(4, k=_CONST)))
    assert isinstance(out, Window)
    assert [k.expr.name for k in out.order_keys] == ["x"]


def test_drop_order_key_proven_constant_keeps_the_last_key():
    ds = _ds()
    plan = ds.window(partition_by=["g"], order_by=["k"], functions={"rn": "row_number"})._plan
    # Dropping it would leave a ranking function with no ORDER BY — a different operator.
    assert m.drop_order_key_proven_constant(plan, _ctx(ds, _stats(4, k=_CONST))) is None


def test_drop_order_key_proven_constant_fires_end_to_end():
    ds = _ds()
    plan = ds.window(partition_by=["g"], order_by=["k", "x"], functions={"rn": "row_number"})._plan
    out = _rewrite(ds, plan, _stats(4, k=_CONST))
    assert [[k.expr.name for k in w.order_keys] for w in _windows(out)] == [["x"]]


# --- drop_order_key_equal_to_partition_key --------------------------------------


def test_drop_order_key_equal_to_partition_key_fires():
    ds = _ds()
    plan = ds.window(partition_by=["g"], order_by=["g", "x"], functions={"rn": "row_number"})._plan
    out = m.drop_order_key_equal_to_partition_key(plan, None)
    assert [k.expr.name for k in out.order_keys] == ["x"]
    assert m.drop_order_key_equal_to_partition_key(out, None) is None  # idempotent


def test_drop_order_key_equal_to_partition_key_keeps_the_last_key():
    ds = _ds()
    plan = ds.window(partition_by=["g"], order_by=["g"], functions={"rn": "row_number"})._plan
    assert m.drop_order_key_equal_to_partition_key(plan, None) is None


def test_drop_order_key_equal_to_partition_key_fires_end_to_end():
    ds = _ds()
    plan = ds.window(partition_by=["g"], order_by=["g", "x"], functions={"rn": "row_number"})._plan
    out = _rewrite(ds, plan, _stats(4))
    assert [[k.expr.name for k in w.order_keys] for w in _windows(out)] == [["x"]]


# --- drop_order_key_after_unique_key_in_window_order ----------------------------


def _unique_order_plan(ds):
    return ds.window(order_by=["id", "x"], functions={"rn": "row_number"})._plan


def test_drop_order_key_after_unique_key_fires():
    ds = _ds()
    out = m.drop_order_key_after_unique_key_in_window_order(
        _unique_order_plan(ds), _ctx(ds, _stats(4, id=_UNIQUE))
    )
    assert [k.expr.name for k in out.order_keys] == ["id"]


def test_drop_order_key_after_unique_key_refuses_estimated_ndv():
    ds = _ds()
    stats = _stats(4, id=ColumnStat(ndv=4, null_count=0, provenance=Provenance.SKETCH))
    assert (
        m.drop_order_key_after_unique_key_in_window_order(_unique_order_plan(ds), _ctx(ds, stats))
        is None
    )


def test_drop_order_key_after_unique_key_refuses_non_unique_ndv():
    ds = _ds()
    stats = _stats(4, id=_exact(ndv=2, null_count=0))  # ties exist → later keys still matter
    assert (
        m.drop_order_key_after_unique_key_in_window_order(_unique_order_plan(ds), _ctx(ds, stats))
        is None
    )


def test_drop_order_key_after_unique_key_fires_end_to_end():
    ds = _ds()
    out = _rewrite(ds, _unique_order_plan(ds), _stats(4, id=_UNIQUE))
    assert [[k.expr.name for k in w.order_keys] for w in _windows(out)] == [["id"]]


# --- drop_redundant_unbounded_frame ---------------------------------------------


def _unbounded_plan(ds, order_by=()):
    return ds.window(
        partition_by=["g"],
        order_by=list(order_by),
        functions={"s": ("sum", "x")},
        frame=(None, None),
    )._plan


def test_drop_redundant_unbounded_frame_fires():
    ds = _ds()
    out = m.drop_redundant_unbounded_frame(_unbounded_plan(ds), None)
    assert isinstance(out, Window)
    assert out.functions[0].frame is None  # the engine's default *is* the whole partition
    assert m.drop_redundant_unbounded_frame(out, None) is None  # idempotent


def test_drop_redundant_unbounded_frame_refuses_when_ordered():
    ds = _ds()
    # With an ORDER BY the default frame is the *running* one — the frame is not redundant.
    assert m.drop_redundant_unbounded_frame(_unbounded_plan(ds, ["x"]), None) is None


def test_drop_redundant_unbounded_frame_fires_end_to_end():
    ds = _ds()
    out = _rewrite(ds, _unbounded_plan(ds), _stats(4))
    assert [fn.frame for w in _windows(out) for fn in w.functions] == [None]


# --- drop_order_keys_under_unbounded_frames -------------------------------------


def test_drop_order_keys_under_unbounded_frames_fires():
    ds = _ds()
    out = m.drop_order_keys_under_unbounded_frames(_unbounded_plan(ds, ["x"]), None)
    assert isinstance(out, Window)
    assert out.order_keys == ()  # every function spans the partition → the sort is dead
    assert m.drop_order_keys_under_unbounded_frames(out, None) is None


def test_drop_order_keys_under_unbounded_frames_refuses_a_ranking_function():
    ds = _ds()
    win = ds.window(
        partition_by=["g"], order_by=["x"], functions={"s": ("sum", "x")}, frame=(None, None)
    )._plan
    ranked = Window(
        win.input,
        win.partition_keys,
        win.order_keys,
        (*win.functions, WindowFuncSpec("row_number", None, "rn")),
    )
    assert m.drop_order_keys_under_unbounded_frames(ranked, None) is None


def test_drop_order_keys_under_unbounded_frames_refuses_a_running_frame():
    ds = _ds()
    plan = ds.window(
        partition_by=["g"], order_by=["x"], functions={"s": ("sum", "x")}, frame=(None, 0)
    )._plan
    assert m.drop_order_keys_under_unbounded_frames(plan, None) is None


def test_drop_order_keys_under_unbounded_frames_fires_end_to_end():
    ds = _ds()
    out = _rewrite(ds, _unbounded_plan(ds, ["x"]), _stats(4))
    assert [w.order_keys for w in _windows(out)] == [()]


# --- dedupe_window_functions -----------------------------------------------------


def _dup_plan(ds):
    return ds.window(
        partition_by=["g"], order_by=["x"], functions={"a": ("sum", "x"), "b": ("sum", "x")}
    )._plan


def test_dedupe_window_functions_fires():
    ds = _ds()
    out = m.dedupe_window_functions(_dup_plan(ds), None)
    assert isinstance(out, Project)
    assert [i.alias for i in out.items] == ["id", "g", "k", "x", "a", "b"]  # schema preserved
    assert [fn.alias for fn in out.input.functions] == ["a"]  # computed once
    assert out.items[-1].expr.name == "a"  # `b` re-derived from it


def test_dedupe_window_functions_refuses_distinct_functions():
    ds = _ds()
    plan = ds.window(
        partition_by=["g"], order_by=["x"], functions={"a": ("sum", "x"), "b": ("min", "x")}
    )._plan
    assert m.dedupe_window_functions(plan, None) is None


def test_dedupe_window_functions_fires_end_to_end():
    ds = _ds()
    out = _rewrite(ds, _dup_plan(ds), _stats(4))
    assert [len(w.functions) for w in _windows(out)] == [1]


# --- constant_window_function_folding --------------------------------------------


def _const_fn_plan(ds, frame=None):
    return ds.window(
        partition_by=["g"], order_by=["x"], functions={"lo": ("min", "k")}, frame=frame
    )._plan


def test_constant_window_function_folding_fires():
    ds = _ds()
    out = m.constant_window_function_folding(_const_fn_plan(ds), _ctx(ds, _stats(4, k=_CONST)))
    assert isinstance(out, Project)
    assert not _windows(out)  # the only function folded → the window disappears
    assert [i.alias for i in out.items] == ["id", "g", "k", "x", "lo"]


def test_constant_window_function_folding_refuses_explicit_frame():
    ds = _ds()
    plan = _const_fn_plan(ds, frame=(1, 2))  # a following-only frame can be *empty* → NULL
    assert m.constant_window_function_folding(plan, _ctx(ds, _stats(4, k=_CONST))) is None


def test_constant_window_function_folding_refuses_non_constant_column():
    ds = _ds()
    plan = ds.window(partition_by=["g"], order_by=["x"], functions={"lo": ("min", "x")})._plan
    assert m.constant_window_function_folding(plan, _ctx(ds, _stats(4, k=_CONST))) is None


def test_constant_window_function_folding_refuses_count():
    ds = _ds()
    plan = ds.window(partition_by=["g"], order_by=["x"], functions={"n": ("count", "k")})._plan
    # COUNT depends on how many rows the frame holds, not only on the value.
    assert m.constant_window_function_folding(plan, _ctx(ds, _stats(4, k=_CONST))) is None


def test_constant_window_function_folding_fires_end_to_end():
    ds = _ds()
    out = _rewrite(ds, _const_fn_plan(ds), _stats(4, k=_CONST))
    assert not _windows(out)


# --- simplify_window_over_single_row_partition ------------------------------------


def _single_row_plan(ds):
    return ds.window(order_by=["x"], functions={"rn": "row_number", "lo": ("min", "x")})._plan


def test_simplify_window_over_single_row_partition_fires():
    ds = _ds(1)
    out = m.simplify_window_over_single_row_partition(_single_row_plan(ds), _ctx(ds, _stats(1)))
    assert isinstance(out, Project)
    assert not _windows(out)
    assert [i.alias for i in out.items] == ["id", "g", "k", "x", "rn", "lo"]


def test_simplify_window_refuses_estimated_row_count():
    ds = _ds(4)
    # A filter's output size is *estimated* (selectivity), never proven — even when it in fact
    # leaves one row. The rule must not fold on an estimate.
    filtered = ds.filter(bt.col("x") > 15)
    plan = filtered.window(order_by=["x"], functions={"rn": "row_number"})._plan
    assert m.simplify_window_over_single_row_partition(plan, _ctx(ds, _stats(4))) is None


def test_simplify_window_refuses_unfoldable_function():
    ds = _ds(1)
    plan = ds.window(order_by=["x"], functions={"s": ("sum", "x")})._plan  # SUM widens its type
    assert m.simplify_window_over_single_row_partition(plan, _ctx(ds, _stats(1))) is None


def test_simplify_window_over_multi_row_input_refuses():
    ds = _ds(4)
    ctx = _ctx(ds, _stats(4))
    assert m.simplify_window_over_single_row_partition(_single_row_plan(ds), ctx) is None


def test_simplify_window_fires_end_to_end():
    ds = _ds(1)
    out = _rewrite(ds, _single_row_plan(ds), _stats(1))
    assert not _windows(out)


# --- rank_limit_zero_to_empty -------------------------------------------------------


def _rank_limited(ds, limit: int):
    return Window(
        ds._plan,
        (Col("g"),),
        (SortKeySpec(Col("x")),),
        (WindowFuncSpec("row_number", None, "rn"),),
        rank_limit=limit,
    )


def test_rank_limit_zero_to_empty_fires():
    ds = _ds()
    out = m.rank_limit_zero_to_empty(_rank_limited(ds, 0), None)
    assert isinstance(out, Window) and out.rank_limit is None
    assert isinstance(out.input, Limit) and out.input.n == 0
    assert m.rank_limit_zero_to_empty(out, None) is None  # idempotent


def test_rank_limit_zero_to_empty_refuses_a_real_limit():
    ds = _ds()
    assert m.rank_limit_zero_to_empty(_rank_limited(ds, 1), None) is None


def test_rank_limit_zero_to_empty_fires_end_to_end():
    ds = _ds()
    out = _rewrite(ds, _rank_limited(ds, 0), _stats(4))
    assert any(isinstance(n, Limit) and n.n == 0 for n in walk(out))
