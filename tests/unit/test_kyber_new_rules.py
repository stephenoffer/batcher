"""Plan-shape tests for the new Kyber rules (B-batch). Correctness vs DuckDB lives
in tests/differential/test_diff_kyber_new_rules.py; here we assert each rule fires,
is a no-op when it shouldn't, and the optimizer stays idempotent."""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.optimizer import Optimizer


def _ir(ds):
    return Optimizer().optimize(ds._plan).ir


def _count_op(ir, op):
    n = 1 if ir.get("op") == op else 0
    for v in ir.values():
        if isinstance(v, dict) and "op" in v:
            n += _count_op(v, op)
        elif isinstance(v, list):
            n += sum(_count_op(e, op) for e in v if isinstance(e, dict) and "op" in e)
    return n


# --- B7 collapse_adjacent_windows ---------------------------------------------


def test_collapse_adjacent_windows_fuses():
    ds = bt.from_pydict({"g": [1, 1, 2], "v": [10, 20, 30]})
    q = ds.window(partition_by=["g"], order_by=["v"], functions={"r": "row_number"}).window(
        partition_by=["g"], order_by=["v"], functions={"mx": ("max", "v")}
    )
    assert _count_op(_ir(q), "window") == 1  # two windows fused into one pass


def test_collapse_adjacent_windows_noop_on_different_partition():
    ds = bt.from_pydict({"g": [1, 1, 2], "h": [1, 2, 2], "v": [10, 20, 30]})
    q = ds.window(partition_by=["g"], order_by=["v"], functions={"r": "row_number"}).window(
        partition_by=["h"], order_by=["v"], functions={"mx": ("max", "v")}
    )
    assert _count_op(_ir(q), "window") == 2  # different partitions → kept separate


# --- B8 push_filter_through_window --------------------------------------------


def test_push_filter_through_window_on_partition_key():
    ds = bt.from_pydict({"g": [1, 1, 2, 2], "v": [10, 20, 30, 40]})
    q = ds.window(partition_by=["g"], order_by=["v"], functions={"r": "row_number"}).filter(
        col("g") == 2
    )
    ir = _ir(q)
    # The window's input must now be a filter (the partition predicate was pushed below).
    win = ir if ir["op"] == "window" else ir.get("input", {})
    assert win.get("op") == "window"
    assert win["input"]["op"] == "filter"


def test_push_filter_through_window_noop_on_rank_predicate():
    ds = bt.from_pydict({"g": [1, 1, 2, 2], "v": [10, 20, 30, 40]})
    # A predicate on the window output (r) cannot move below the window.
    q = ds.window(partition_by=["g"], order_by=["v"], functions={"r": "row_number"}).filter(
        col("r") >= 2
    )
    ir = _ir(q)
    # Top filter remains above the window (not pushed). qualify fusion handles le/lt,
    # but `>=` is a lower bound, so it stays as a filter over the window.
    assert _count_op(ir, "filter") >= 1


# --- B12 distinct_over_single_row ---------------------------------------------


def test_distinct_dropped_over_scalar_aggregate():
    agg = bt.from_pydict({"x": [1, 2, 3]}).agg(s=col("x").sum())  # exactly one row
    ir = _ir(agg.distinct())
    assert _count_op(ir, "distinct") == 0  # dedup over a 1-row relation is pointless


def test_distinct_kept_over_multirow():
    ds = bt.from_pydict({"x": [1, 1, 2]})
    ir = _ir(ds.distinct())
    assert _count_op(ir, "distinct") == 1  # genuine dedup stays


# --- B6 projection_inlining_into_agg ------------------------------------------


def test_rename_inlined_into_aggregate():
    ds = bt.from_pydict({"a": [1, 1, 2], "b": [10, 20, 30]})
    q = ds.rename({"a": "k"}).group_by("k").agg(s=col("b").sum())
    assert _count_op(_ir(q), "project") == 0  # the rename projection is inlined away


# --- B11 nested_cast_removal --------------------------------------------------


def test_nested_same_type_cast_collapsed():
    import json

    ds = bt.from_pydict({"a": [1, 2, 3]})
    q = ds.with_columns(c=col("a").cast("int64").cast("int64"))
    # The doubled cast collapses: only one cast tag remains in the lowered IR.
    blob = json.dumps(_ir(q))
    assert blob.count('"cast"') == 1


# --- B10 or_to_in_and_range ---------------------------------------------------


def test_or_to_in_adds_range_bounds():
    import json

    ds = bt.from_pydict({"c": [1, 3, 5, 7, 9]})
    q = ds.filter(col("c").is_in([3, 7]))
    blob = json.dumps(_ir(q))
    # The IN's implied range (3 <= c <= 7) is added for zone-map pruning.
    assert '"ge"' in blob and '"le"' in blob


def test_or_to_in_idempotent():
    # The rule adds conjuncts, so it must guard against re-adding (else the fixpoint
    # never converges). The bounds must appear exactly once.
    import json

    ds = bt.from_pydict({"c": [1, 3, 5, 7, 9]})
    q = ds.filter(col("c").is_in([3, 7]))
    blob = json.dumps(_ir(q))
    assert blob.count('"ge"') == 1 and blob.count('"le"') == 1


# --- B3 join_to_semijoin ------------------------------------------------------


def _join_type(ir):
    if ir.get("op") == "hash_join":
        return ir.get("join_type")
    for v in ir.values():
        if isinstance(v, dict) and "op" in v:
            r = _join_type(v)
            if r:
                return r
    return None


def test_distinct_left_only_join_becomes_semi():
    emp = bt.from_pydict({"id": [1, 2, 3], "dept": [10, 20, 10]})
    dept = bt.from_pydict({"dept": [10, 20], "name": ["a", "b"]})
    q = emp.join(dept, on="dept").select("id", "dept").distinct()
    assert _join_type(_ir(q)) == "semi"


def test_join_kept_inner_when_right_columns_used():
    emp = bt.from_pydict({"id": [1, 2], "dept": [10, 20]})
    dept = bt.from_pydict({"dept": [10, 20], "name": ["a", "b"]})
    q = emp.join(dept, on="dept").select("id", "name").distinct()  # reads right column
    assert _join_type(_ir(q)) == "inner"  # cannot become semi


# --- B1 transitive_predicate_inference ----------------------------------------


def _scans_under_filter(ir, under=False, acc=None):
    if acc is None:
        acc = set()
    if ir.get("op") == "scan" and under:
        acc.add(ir.get("source_id"))
    nf = under or ir.get("op") == "filter"
    for v in ir.values():
        if isinstance(v, dict) and "op" in v:
            _scans_under_filter(v, nf, acc)
        elif isinstance(v, list):
            for e in v:
                if isinstance(e, dict) and "op" in e:
                    _scans_under_filter(e, nf, acc)
    return acc


def test_constraint_propagates_transitively_across_join_chain():
    a = bt.from_pydict({"k": [1, 2, 3, 4]})
    b = bt.from_pydict({"k": [1, 2, 3, 4]})
    c = bt.from_pydict({"k": [1, 2, 3, 4], "cv": [5, 6, 7, 8]})
    # a.k = b.k = c.k AND a.k > 2 → the bound must reach all three scans (incl. c, 2 hops).
    q = a.join(b, on="k").join(c, on="k").filter(col("k") > 2)
    assert _scans_under_filter(_ir(q)) == {0, 1, 2}


# --- B5 pre_aggregation_through_join ------------------------------------------


def _preagg_ctx(ds, ndv):
    from batcher.config import active_config
    from batcher.kyber.pass_base import OptimizerContext
    from batcher.kyber.stats.estimator import StatsEstimator

    est = StatsEstimator(ds._sources, learned={"__column_ndv__": ndv})
    return OptimizerContext(config=active_config(), sources=ds._sources, hub=None, estimator=est)


def _fact_dim_sum():
    from batcher.kyber.rules.agg_pushdown import pre_aggregation_through_join

    fact = bt.from_pydict({"k": [1, 1, 1, 2, 2, 3], "amt": [10, 20, 30, 40, 50, 60]})
    dim = (
        bt.from_pydict({"k": [1, 2, 3], "region": ["e", "w", "s"]})
        .group_by("k")
        .agg(region=col("region").max())
    )  # structurally unique on k
    q = fact.join(dim, on="k").group_by("region").agg(s=col("amt").sum())
    return q, pre_aggregation_through_join


def test_preagg_sum_pushed_below_unique_join():
    from batcher.plan.logical import Aggregate, Join

    q, rule = _fact_dim_sum()
    out = rule(q._plan, _preagg_ctx(q, {"k": 3.0}))
    assert isinstance(out, Aggregate) and isinstance(out.input, Join)
    assert isinstance(out.input.left, Aggregate)  # partial sum pushed below the join


def test_preagg_bails_when_right_not_unique():
    from batcher.kyber.rules.agg_pushdown import pre_aggregation_through_join

    fact = bt.from_pydict({"k": [1, 1, 2], "amt": [10, 20, 30]})
    dim_dup = bt.from_pydict({"k": [1, 1, 2], "r": ["e", "e2", "w"]})  # not unique
    q = fact.join(dim_dup, on="k").group_by("r").agg(s=col("amt").sum())
    # Fan-out would multiply the partial sums → must not fire.
    assert pre_aggregation_through_join(q._plan, _preagg_ctx(q, {"k": 2.0})) is None
