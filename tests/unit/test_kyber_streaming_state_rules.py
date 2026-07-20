"""Kyber's streaming *state-minimization* rules — the ones that shrink retained state.

Sibling of `test_kyber_streaming_rules.py`, which covers the pushdown half of the family
and the streaming analysis itself. Every rule here gets a **pair**: one test that it
fires, and one that it declines on the near-miss where firing would be wrong. For this
family the declining test is the load-bearing one — a wrongly narrowed dedup key merges
two groups that should stay apart, and that is a silently missing output row on a stream,
never an error.

**Why most tests call the rule function directly rather than running the optimizer.**
`Optimizer.logical_rewrite` cannot currently survive a plan containing a
`WatermarkStreamJoin`, or a `WatermarkDedup` whose predicate any NORMALIZE rule rewrites:
the fixpoint's change-detector (`kyber/optimizer/driver.py::_fingerprint`) falls back to
comparing plan *nodes* for the streaming operators, because they define no `to_ir()`, and
comparing two frozen dataclasses eventually compares two `Expr`s — whose `__eq__` builds
an expression rather than returning a bool, so `_run_phase` raises `PlanError`. That is a
pre-existing driver bug, reproducible with no rule from this module involved:

    Optimizer(None, [], None).logical_rewrite(<any WatermarkStreamJoin>)  # PlanError

`registry.rule` returns the decorated function unchanged precisely so a rule stays
unit-testable in isolation, so that is what these use. The narrowing rules that the
driver *can* carry today are additionally exercised end to end at the bottom of the file,
which is what proves they survive the other 300 rules rather than merely working alone.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.rules.streaming.state import (
    add_stream_join_key_null_rejection,
    deduplicate_stream_join_keys,
    deduplicate_watermark_dedup_subset,
    drop_constant_watermark_dedup_key,
    drop_watermark_dedup_key_determined_by_other_keys,
    drop_watermark_dedup_key_pinned_by_equality_filter,
    split_filter_conjuncts_into_stream_join_sides,
    split_filter_conjuncts_through_watermark_dedup,
)
from batcher.plan.expr_ir import Col, Lit, col, lit, referenced_columns
from batcher.plan.logical import (
    Filter,
    JoinOutputCol,
    Project,
    Projection,
    WatermarkDedup,
    WatermarkStreamJoin,
)
from batcher.plan.visitor import walk

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def ctx():
    """A real `OptimizerContext`; every rule here is structural and reads nothing from it."""
    return Optimizer(None, [], None)._context()


def _source(nullable_key: bool = False):
    keys = [1, None, 3] if nullable_key else [1, 2, 3]
    return bt.from_pydict({"k": keys, "v": [10, 20, 30], "t": [1, 2, 3], "r": [7, 7, 7]})._plan


def _dedup(plan, subset=("k",)):
    return WatermarkDedup(plan, subset=subset, event_time="t", lateness_micros=1_000)


def _project(plan, **items):
    """A `Project` over `plan`; each kwarg is an output alias bound to an expression."""
    return Project(plan, tuple(Projection(alias=a, expr=e) for a, e in items.items()))


def _right(nullable_key: bool = False):
    """A right-hand stream whose event-time column is renamed apart from the left's."""
    return _project(
        _source(nullable_key), k=Col("k"), v=Col("v"), rt=Col("t"), r=Col("r"), t=Col("t")
    )


def _stream_join(left=None, right=None, *, left_keys=("k",), right_keys=("k",)):
    output = (
        JoinOutputCol("left", "k", "k"),
        JoinOutputCol("left", "v", "v"),
        JoinOutputCol("left", "t", "t"),
        JoinOutputCol("right", "v", "rv"),
        JoinOutputCol("right", "rt", "rt"),
    )
    return WatermarkStreamJoin(
        left=_source() if left is None else left,
        right=_right() if right is None else right,
        left_keys=left_keys,
        right_keys=right_keys,
        output=output,
        left_time="t",
        right_time="rt",
        within_micros=5_000,
        lateness_micros=1_000,
    )


def _cols(node) -> set[str]:
    """The columns the `Filter` `node` reads (empty for anything else)."""
    return referenced_columns(node.predicate) if isinstance(node, Filter) else set()


# --- deduplicate_watermark_dedup_subset ----------------------------------------


def test_repeated_subset_column_collapses_to_one(ctx):
    """`(k, k, v)` and `(k, v)` induce the same key partition; the state stores less."""
    out = deduplicate_watermark_dedup_subset(_dedup(_source(), subset=("k", "k", "v")), ctx)
    assert out.subset == ("k", "v")


def test_distinct_subset_columns_are_not_collapsed(ctx):
    """The near miss: two *different* columns are not a repetition.

    Collapsing `(k, v)` to `(k,)` would merge rows that differ in `v` into one key and
    suppress output rows that must be emitted — silently, and only on a stream.
    """
    assert deduplicate_watermark_dedup_subset(_dedup(_source(), subset=("k", "v")), ctx) is None


# --- drop_constant_watermark_dedup_key -----------------------------------------


def test_literal_keyed_dedup_column_is_dropped(ctx):
    """A key defined as a literal is the same in every row, so it separates nothing."""
    child = _project(_source(), k=Col("k"), t=Col("t"), tag=Lit("eu"))
    assert drop_constant_watermark_dedup_key(_dedup(child, subset=("k", "tag")), ctx).subset == (
        "k",
    )


def test_non_literal_keyed_dedup_column_is_kept(ctx):
    """The near miss: a *column* is not a constant, however cheap it looks."""
    child = _project(_source(), k=Col("k"), t=Col("t"), tag=Col("r"))
    assert drop_constant_watermark_dedup_key(_dedup(child, subset=("k", "tag")), ctx) is None


def test_an_all_literal_subset_is_left_alone(ctx):
    """Never narrow to nothing: a keyless dedup would collapse the stream to one row."""
    child = _project(_source(), t=Col("t"), a=Lit(1), b=Lit(2))
    assert drop_constant_watermark_dedup_key(_dedup(child, subset=("a", "b")), ctx) is None


def test_constant_key_needs_a_projection_to_prove_it(ctx):
    """Without a child `Project` there is nothing proving the column constant."""
    assert drop_constant_watermark_dedup_key(_dedup(_source(), subset=("k", "r")), ctx) is None


# --- drop_watermark_dedup_key_determined_by_other_keys -------------------------


def test_renamed_key_determined_by_its_source_key_is_dropped(ctx):
    """`b := col("k")` agrees whenever `k` agrees, so `(k, b)` groups as `(k,)` does."""
    child = _project(_source(), k=Col("k"), t=Col("t"), b=Col("k"))
    out = drop_watermark_dedup_key_determined_by_other_keys(_dedup(child, subset=("k", "b")), ctx)
    assert out.subset == ("k",)


def test_derived_key_determined_by_its_source_key_is_dropped(ctx):
    """A row-local derivation (`cents := amount * 100`) is a function of the row."""
    child = _project(_source(), v=Col("v"), t=Col("t"), cents=col("v") * lit(100))
    dedup = _dedup(child, subset=("v", "cents"))
    out = drop_watermark_dedup_key_determined_by_other_keys(dedup, ctx)
    assert out.subset == ("v",)


def test_key_reading_a_non_key_column_is_kept(ctx):
    """The near miss: `b := col("v")` is not determined by `k`, only correlated with it.

    Dropping `b` here would key on `k` alone and suppress every later row of a key whose
    `v` changed — a real row loss that a bounded run, seeing each key once, cannot show.
    """
    child = _project(_source(), k=Col("k"), t=Col("t"), b=Col("v"))
    assert (
        drop_watermark_dedup_key_determined_by_other_keys(_dedup(child, subset=("k", "b")), ctx)
        is None
    )


def test_mutually_referencing_keys_are_not_both_dropped(ctx):
    """Each of `a := col("k")` and `b := col("k")` determines the other; keep one.

    Testing a key against keys still under consideration rather than against the ones
    already kept would drop both and leave a keyless dedup.
    """
    child = _project(_source(), k=Col("k"), t=Col("t"), a=Col("k"), b=Col("k"))
    out = drop_watermark_dedup_key_determined_by_other_keys(_dedup(child, subset=("a", "b")), ctx)
    assert out.subset == ("a",)


def test_a_constant_key_is_left_to_the_constant_rule(ctx):
    """Each rule owns one idea: a key reading *no* column is the constant case."""
    child = _project(_source(), k=Col("k"), t=Col("t"), tag=Lit("eu"))
    assert (
        drop_watermark_dedup_key_determined_by_other_keys(_dedup(child, subset=("k", "tag")), ctx)
        is None
    )


# --- drop_watermark_dedup_key_pinned_by_equality_filter ------------------------


def test_key_pinned_by_an_equality_filter_is_dropped(ctx):
    """Only rows with `r = 7` reach the dedup, so `r` cannot separate two of them."""
    child = Filter(_source(), col("r") == lit(7))
    out = drop_watermark_dedup_key_pinned_by_equality_filter(_dedup(child, subset=("r", "k")), ctx)
    assert out.subset == ("k",)


def test_key_pinned_by_a_singleton_in_list_is_dropped(ctx):
    """`r IN (7)` is `r = 7` spelled differently, and pins the column just as hard."""
    child = Filter(_source(), col("r").is_in([7]))
    out = drop_watermark_dedup_key_pinned_by_equality_filter(_dedup(child, subset=("r", "k")), ctx)
    assert out.subset == ("k",)


def test_key_constrained_by_an_inequality_filter_is_kept(ctx):
    """The near miss: `r > 7` admits many values of `r`, so `r` still splits keys.

    Only an equality (or a one-element `IN`) pins a column to a single non-null value;
    treating a range predicate as pinning would merge distinct keys into one.
    """
    child = Filter(_source(), col("r") > lit(7))
    assert (
        drop_watermark_dedup_key_pinned_by_equality_filter(_dedup(child, subset=("r", "k")), ctx)
        is None
    )


def test_key_pinned_by_a_multi_value_in_list_is_kept(ctx):
    """`r IN (7, 8)` leaves two possible values — not a constant."""
    child = Filter(_source(), col("r").is_in([7, 8]))
    assert (
        drop_watermark_dedup_key_pinned_by_equality_filter(_dedup(child, subset=("r", "k")), ctx)
        is None
    )


# --- split_filter_conjuncts_through_watermark_dedup ----------------------------


def test_mixed_conjunction_splits_across_the_dedup(ctx):
    """The key-constant half crosses; the half reading a non-key column stays above."""
    plan = Filter(_dedup(_source()), (col("k") > lit(0)) & (col("v") > lit(5)))
    out = split_filter_conjuncts_through_watermark_dedup(plan, ctx)
    assert _cols(out) == {"v"}  # residual stayed above
    dedup = out.input
    assert isinstance(dedup, WatermarkDedup)
    assert _cols(dedup.input) == {"k"}  # key-constant half went below


def test_a_wholly_non_key_conjunction_does_not_split(ctx):
    """The near miss: nothing may cross, so the dedup's first-per-key row is preserved.

    Pushing `v > 5` below would let a row the dedup would have suppressed become its
    key's first surviving row, changing which row is emitted.
    """
    plan = Filter(_dedup(_source()), (col("v") > lit(0)) & (col("v") > lit(5)))
    assert split_filter_conjuncts_through_watermark_dedup(plan, ctx) is None


def test_a_wholly_key_constant_conjunction_is_left_to_the_shipped_rule(ctx):
    """Nothing is left behind, so `push_filter_through_watermark_dedup` owns this case.

    Firing here too would let the two rules hand the plan back and forth forever.
    """
    plan = Filter(_dedup(_source()), (col("k") > lit(0)) & (col("k") < lit(9)))
    assert split_filter_conjuncts_through_watermark_dedup(plan, ctx) is None


# --- split_filter_conjuncts_into_stream_join_sides -----------------------------


def test_mixed_conjunction_splits_into_both_join_sides(ctx):
    """Left-only and right-only conjuncts reach their sides; the cross-side one stays."""
    predicate = (col("v") > lit(1)) & (col("rv") > lit(2)) & (col("v") > col("rv"))
    plan = Filter(_stream_join(), predicate)
    out = split_filter_conjuncts_into_stream_join_sides(plan, ctx)
    assert _cols(out) == {"v", "rv"}  # the cross-side conjunct stayed above
    join = out.input
    assert _cols(join.left) == {"v"}
    # The right conjunct was remapped out of the `rv` alias into the side's own name.
    assert _cols(join.right) == {"v"}


def test_a_wholly_cross_side_conjunction_does_not_split(ctx):
    """The near miss: a predicate naming both sides is a join condition, not a filter."""
    plan = Filter(_stream_join(), (col("v") > col("rv")) & (col("t") < col("rt")))
    assert split_filter_conjuncts_into_stream_join_sides(plan, ctx) is None


def test_a_wholly_one_sided_conjunction_is_left_to_the_shipped_rule(ctx):
    """Nothing is left behind, so `push_filter_into_stream_join_sides` owns this case."""
    plan = Filter(_stream_join(), (col("v") > lit(1)) & (col("v") < lit(9)))
    assert split_filter_conjuncts_into_stream_join_sides(plan, ctx) is None


def test_a_conjunct_naming_a_non_output_column_stays_above(ctx):
    """A column the join does not emit cannot be attributed to a side, so it must not move."""
    predicate = (col("v") > lit(1)) & (col("k") > lit(0)) & (col("t") < col("rt"))
    plan = Filter(_stream_join(), predicate)
    out = split_filter_conjuncts_into_stream_join_sides(plan, ctx)
    assert _cols(out) == {"t", "rt"}
    assert _cols(out.input.left) == {"v", "k"}


# --- deduplicate_stream_join_keys ----------------------------------------------


def test_repeated_join_key_pair_collapses(ctx):
    """`a = b AND a = b` is `a = b`; the buffers hash a narrower tuple."""
    out = deduplicate_stream_join_keys(
        _stream_join(left_keys=("k", "k"), right_keys=("k", "k")), ctx
    )
    assert out.left_keys == ("k",)
    assert out.right_keys == ("k",)


def test_a_repeated_column_in_distinct_pairs_is_kept(ctx):
    """The near miss: `(v, k)` vs `(k, k)` states `v = k AND k = k`, two conditions.

    Collapsing on the repeated right column alone would delete the `v = k` equality and
    widen the join — a different query, not a smaller one.
    """
    join = _stream_join(left_keys=("v", "k"), right_keys=("k", "k"))
    assert deduplicate_stream_join_keys(join, ctx) is None


def test_distinct_join_keys_are_untouched(ctx):
    assert deduplicate_stream_join_keys(_stream_join(), ctx) is None


# --- add_stream_join_key_null_rejection ----------------------------------------


def test_nullable_join_keys_get_a_null_rejecting_filter(ctx):
    """An inner equi-join never matches a NULL key, so those rows only occupy buffer."""
    join = _stream_join(left=_source(nullable_key=True), right=_right(nullable_key=True))
    out = add_stream_join_key_null_rejection(join, ctx)
    assert _cols(out.left) == {"k"}
    assert _cols(out.right) == {"k"}


def test_already_guarded_join_keys_get_no_second_filter(ctx):
    """The near miss: a side the user already null-guarded must not be guarded again.

    This is the only way this rule can be *wrong* — not by changing a result, but by
    matching its own output and never converging. The optimizer would then warn about a
    non-confluent phase and the plan would depend on `fixpoint_iterations`.
    """
    left = Filter(_source(nullable_key=True), col("k").is_not_null())
    right = Filter(_right(nullable_key=True), col("k").is_not_null())
    assert add_stream_join_key_null_rejection(_stream_join(left=left, right=right), ctx) is None


def test_null_rejection_is_not_added_twice(ctx):
    """Idempotence, stated as a fixpoint: applying the rule to its own output declines."""
    join = _stream_join(left=_source(nullable_key=True), right=_right(nullable_key=True))
    once = add_stream_join_key_null_rejection(join, ctx)
    assert add_stream_join_key_null_rejection(once, ctx) is None


def test_null_rejection_merges_into_an_existing_filter(ctx):
    """The guard joins the side's existing predicate rather than stacking a second node."""
    left = Filter(_source(nullable_key=True), col("v") > lit(1))
    join = _stream_join(left=left, right=_right(nullable_key=True))
    out = add_stream_join_key_null_rejection(join, ctx)
    assert _cols(out.left) == {"v", "k"}
    assert len([n for n in walk(out.left) if isinstance(n, Filter)]) == 1


# --- the fields that carry the state bound must survive every rewrite ----------


def test_narrowing_preserves_the_dedup_state_configuration(ctx):
    """A rewrite that dropped `event_time` or `lateness_micros` would unbound the state."""
    original = _dedup(_source(), subset=("k", "k"))
    out = deduplicate_watermark_dedup_subset(original, ctx)
    assert out.event_time == original.event_time
    assert out.lateness_micros == original.lateness_micros


def test_join_rewrites_preserve_the_interval_and_lateness(ctx):
    """`within_micros`/`lateness_micros` are the join's eviction bound — never dropped."""
    original = _stream_join(left_keys=("k", "k"), right_keys=("k", "k"))
    out = deduplicate_stream_join_keys(original, ctx)
    assert out.within_micros == original.within_micros
    assert out.lateness_micros == original.lateness_micros
    assert out.left_time == original.left_time
    assert out.right_time == original.right_time
    assert out.output == original.output


# --- end to end, through the whole optimizer -----------------------------------
#
# Only the dedup-narrowing rules can be run this way today; see the module docstring for
# the `_fingerprint` driver bug that stops a `WatermarkStreamJoin` plan (or a dedup whose
# predicate NORMALIZE rewrites) from reaching any rule at all.


def _rewrite(plan):
    return Optimizer(None, [], None).logical_rewrite(plan)


def test_subset_narrowing_survives_the_full_rule_set():
    """The narrowing holds up against the other 300 rules, not just in isolation."""
    child = _project(_source(), k=Col("k"), t=Col("t"), b=Col("k"), tag=Lit("eu"))
    out = _rewrite(_dedup(child, subset=("k", "b", "tag", "k")))
    dedup = next(n for n in walk(out) if isinstance(n, WatermarkDedup))
    assert dedup.subset == ("k",)
    assert dedup.event_time == "t"


def test_a_sound_subset_is_unchanged_by_the_full_rule_set():
    """The declining direction, end to end: three genuinely independent keys stay."""
    out = _rewrite(_dedup(_source(), subset=("k", "v", "r")))
    dedup = next(n for n in walk(out) if isinstance(n, WatermarkDedup))
    assert set(dedup.subset) == {"k", "v", "r"}
