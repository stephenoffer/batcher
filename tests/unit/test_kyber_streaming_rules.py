"""Kyber's streaming rules and the streaming analysis they gate on.

Plan-shape assertions live here; result correctness against DuckDB lives in
`tests/differential/`. Each rule gets a pair — one test that it fires, one that it
declines on the case where firing would be wrong — because for these rules the
declining case is the one that protects a correct answer.
"""

from __future__ import annotations

import pytest

import batcher as bt
from batcher.kyber.optimizer import Optimizer
from batcher.kyber.streaming import (
    STREAM_CLASSIFIED,
    blocking_operators,
    has_unbounded_input,
    is_blocking_under_stream,
    retains_unbounded_state,
    unbounded_scan_ids,
    unbounded_state_operators,
)
from batcher.plan.expr_ir import col, lit
from batcher.plan.logical import (
    Aggregate,
    Distinct,
    Filter,
    Sample,
    Sort,
    TransformWithState,
    Union,
    WatermarkDedup,
    Window,
)
from batcher.plan.visitor import walk

pytestmark = pytest.mark.unit


def _source():
    return bt.from_pydict({"k": [1, 2, 3], "v": [10, 20, 30], "t": [1, 2, 3]})


def _dedup(plan, subset=("k",)):
    return WatermarkDedup(plan, subset=subset, event_time="t", lateness_micros=1_000)


def _shape(plan) -> list[str]:
    return [type(n).__name__ for n in walk(plan)]


def _rewrite(plan):
    return Optimizer(None, [], None).logical_rewrite(plan)


# --- push_filter_through_watermark_dedup ---------------------------------------


def test_key_constant_filter_pushes_below_watermark_dedup():
    """A predicate over the dedup `subset` may cross it — and shrinks the seen-key state."""
    plan = Filter(_dedup(_source()._plan), col("k") > lit(0))
    shape = _shape(_rewrite(plan))
    assert shape.index("WatermarkDedup") < shape.index("Filter"), shape


def test_non_key_filter_does_not_push_below_watermark_dedup():
    """A predicate over a non-key column must NOT cross the dedup.

    The dedup keeps the *first* row per key. A predicate that accepts some rows of a key
    and rejects others would, if applied first, promote a different row to be that key's
    first — changing the output. Nothing about that failure is loud, so it is pinned here.
    """
    plan = Filter(_dedup(_source()._plan), col("v") > lit(0))
    shape = _shape(_rewrite(plan))
    assert shape.index("Filter") < shape.index("WatermarkDedup"), shape


def test_event_time_filter_does_not_push_below_watermark_dedup():
    """The event-time column is not key-constant — two rows of a key differ in it."""
    plan = Filter(_dedup(_source()._plan), col("t") > lit(0))
    shape = _shape(_rewrite(plan))
    assert shape.index("Filter") < shape.index("WatermarkDedup"), shape


def test_optimizing_a_streaming_node_preserves_its_fields():
    """The rewrite must not disturb the dedup's state-bearing configuration."""
    original = _dedup(_source()._plan)
    out = _rewrite(Filter(original, col("k") > lit(0)))
    assert isinstance(out, WatermarkDedup)
    assert out.subset == original.subset
    assert out.event_time == original.event_time
    assert out.lateness_micros == original.lateness_micros


# --- the streaming analysis ----------------------------------------------------


def test_full_sort_blocks_under_a_stream_but_topn_does_not():
    """A top-N keeps a bounded running best-N, so it can emit; a full sort cannot."""
    assert is_blocking_under_stream(Sort(_source()._plan, keys=(), limit=None))
    assert not is_blocking_under_stream(Sort(_source()._plan, keys=(), limit=10))


def test_distinct_blocks_under_a_stream():
    assert is_blocking_under_stream(Distinct(_source()._plan))


def test_a_distinct_carrying_a_limit_settles_on_a_prefix():
    """The fused early exit stops at `limit` distinct rows, so no last row is needed.

    The same reasoning that makes a top-N `Sort` non-blocking, and the state it holds is
    bounded by the same number.
    """
    capped = Distinct(_source()._plan, limit=10)
    assert not is_blocking_under_stream(capped)
    assert not retains_unbounded_state(capped)


def test_distinct_retains_state_nothing_releases():
    """One entry per distinct value, for the life of the query, with no eviction.

    Reported as bounded until this test existed — the exact shape `WatermarkDedup` was
    added to replace, and invisible to every bounded test because a bounded input
    releases the set at end-of-input.
    """
    assert retains_unbounded_state(Distinct(_source()._plan))


def test_a_distinct_union_blocks_and_leaks_but_union_all_does_neither():
    """`UNION` dedupes the concatenation; `UNION ALL` is a pass-through of both branches."""
    branches = (_source()._plan, _source()._plan)
    assert is_blocking_under_stream(Union(branches, distinct=True))
    assert retains_unbounded_state(Union(branches, distinct=True))
    assert not is_blocking_under_stream(Union(branches, distinct=False))
    assert not retains_unbounded_state(Union(branches, distinct=False))


def test_a_fixed_count_sample_blocks_without_leaking():
    """A reservoir holds exactly `n` rows — blocking, bounded, and the case that proves
    the two properties are independent rather than two names for one thing."""
    reservoir = Sample(_source()._plan, fraction=1.0, seed=7, n=100)
    assert is_blocking_under_stream(reservoir)
    assert not retains_unbounded_state(reservoir)


def test_a_fraction_sample_streams_freely():
    """A per-row seeded hash test — the same distinction `is_partition_independent` draws."""
    fraction = Sample(_source()._plan, fraction=0.5, seed=7)
    assert not is_blocking_under_stream(fraction)
    assert not retains_unbounded_state(fraction)


def test_a_full_sort_leaks_but_a_topn_is_bounded_by_its_limit():
    assert retains_unbounded_state(Sort(_source()._plan, keys=(), limit=None))
    assert not retains_unbounded_state(Sort(_source()._plan, keys=(), limit=10))


def test_a_window_blocks_and_holds_every_partition_it_has_seen():
    """A stream never closes a partition, so nothing releases the rows buffered for one."""
    plan = _source().with_columns(r=col("v").sum().over(partition_by="k"))._plan
    windows = [n for n in walk(plan) if isinstance(n, Window)]
    assert windows, _shape(plan)
    assert is_blocking_under_stream(windows[0])
    assert retains_unbounded_state(windows[0])


def test_transform_with_state_leaks_without_a_ttl_and_is_bounded_with_one():
    """`ttl_micros == 0` means "never expire", which the node's own docstring names as
    the shape this predicate is entitled to complain about — and which it did not."""

    def fn(key, rows, state):  # pragma: no cover — never called, this is a plan-shape test
        return rows, state

    forever = TransformWithState(_source()._plan, fn, ("k",), ("k", "v"), ttl_micros=0)
    expiring = TransformWithState(_source()._plan, fn, ("k",), ("k", "v"), ttl_micros=60_000_000)
    assert retains_unbounded_state(forever)
    assert not retains_unbounded_state(expiring)
    # Neither blocks: state is emitted per key per micro-batch, not at end-of-input.
    assert not is_blocking_under_stream(forever)
    assert not is_blocking_under_stream(expiring)


def test_the_watermark_bounded_nodes_neither_block_nor_leak():
    """The three nodes that exist to emit on an advancing watermark rather than at
    end-of-input. If one of these were classified as leaking, the streaming path would be
    reporting its own bounded operators as unbounded."""
    dedup = _dedup(_source()._plan)
    assert not is_blocking_under_stream(dedup)
    assert not retains_unbounded_state(dedup)
    join = _stream_join()
    assert not is_blocking_under_stream(join)
    assert not retains_unbounded_state(join)


def test_unbounded_state_operators_names_the_offenders():
    """A caller reporting to a user wants the nodes, not a bare bool — the state-side
    counterpart of `blocking_operators`."""
    plan = Distinct(_source().group_by("k").agg(s=col("v").sum())._plan)
    named = {type(n).__name__ for n in unbounded_state_operators(plan)}
    assert named == {"Distinct", "Aggregate"}, named
    assert unbounded_state_operators(_source()._plan) == []


def test_grouped_aggregate_does_not_block():
    """A grouped aggregate emits a running result per group — the streamable case."""
    agg = _source().group_by("k").agg(s=col("v").sum())._plan
    assert isinstance(agg, Aggregate)
    assert not is_blocking_under_stream(agg)
    assert blocking_operators(agg) == []


def test_grouped_aggregate_without_a_watermark_retains_unbounded_state():
    """Correct, and a memory leak: nothing ever evicts a group without a watermark."""
    agg = _source().group_by("k").agg(s=col("v").sum())._plan
    assert retains_unbounded_state(agg)


def _logical_node_types() -> dict[str, type]:
    """Every `LogicalPlan` node type defined under `batcher.plan.logical`.

    Deliberately not `LogicalPlan.__subclasses__()`. Every node is a
    `@dataclass(frozen=True, slots=True)`, and `slots=True` builds a *replacement* class
    — leaving the original registered as a subclass forever. That walk reports 42
    entries for 21 nodes, half of them classes nothing can ever instantiate, so a
    membership test against it fails for reasons that have nothing to do with streaming.
    Walking the modules names each node once.
    """
    import importlib
    import pkgutil

    import batcher.plan.logical as pkg
    from batcher.plan.logical.base import LogicalPlan

    found: dict[str, type] = {}
    for info in pkgutil.iter_modules(pkg.__path__):
        module = importlib.import_module(f"{pkg.__name__}.{info.name}")
        for name, obj in vars(module).items():
            if (
                isinstance(obj, type)
                and issubclass(obj, LogicalPlan)
                and obj is not LogicalPlan
                and obj.__module__ == module.__name__
            ):
                found[name] = obj
    return found


def test_every_logical_node_is_classified_for_streaming():
    """A node with no streaming decision silently takes the *permissive* default.

    Both predicates end in `return False` — "streams fine, retains nothing" — so a node
    added without a decision here claims a memory bound it may not have, and claims it
    where no bounded test can contradict it. That is how `Distinct` came to report it
    retained no state while holding one entry per distinct value forever. This is the
    same "every tag is classified" contract the device tier runs on.
    """
    nodes = _logical_node_types()
    assert len(nodes) >= 20, f"the node walk found only {sorted(nodes)} — it stopped working"
    missing = sorted(name for name, cls in nodes.items() if cls not in STREAM_CLASSIFIED)
    assert not missing, (
        f"{missing} have no streaming classification. Decide in `kyber.streaming`: can the "
        "operator emit before its input ends (`is_blocking_under_stream`), and does "
        "anything ever release its state (`retains_unbounded_state`)? Then add it to "
        "STREAM_CLASSIFIED."
    )


def test_stream_classified_names_only_real_nodes():
    """The guard is only worth its weight if it cannot go stale in the other direction."""
    live = set(_logical_node_types().values())
    stale = sorted(cls.__name__ for cls in STREAM_CLASSIFIED if cls not in live)
    assert not stale, f"{stale} are in STREAM_CLASSIFIED but no longer exist"


def test_bounded_plan_is_never_reported_unbounded():
    """`has_unbounded_input` reads the bound source, not a row estimate.

    An unknown row count means "not estimated" — which a finite source reports just as
    readily as a stream — so boundedness must not be inferred from cardinality.
    """
    ctx = Optimizer(None, [], None)._context()
    plan = _source()._plan
    assert not has_unbounded_input(plan, ctx)
    assert unbounded_scan_ids(plan, []) == frozenset()


# --- push_filter_into_stream_join_side, and what an outer join forbids ----------------


def _stream_join(how: str = "inner", *, predicate=None):
    """A two-stream interval join, optionally under a filter, for the pushdown rules."""
    from batcher.plan.logical import JoinOutputCol, WatermarkStreamJoin

    left = _source()._plan
    right = _source()._plan
    output = (
        JoinOutputCol("left", "k", "k"),
        JoinOutputCol("left", "t", "t"),
        JoinOutputCol("left", "v", "v"),
        JoinOutputCol("right", "v", "v_right"),
    )
    join = WatermarkStreamJoin(left, right, ("k",), ("k",), output, "t", "t", 1_000_000, 0, how)
    return Filter(join, predicate) if predicate is not None else join


def _the_join(plan):
    from batcher.plan.logical import WatermarkStreamJoin

    joins = [n for n in walk(plan) if isinstance(n, WatermarkStreamJoin)]
    assert len(joins) == 1
    return joins[0]


def _sides_carrying(plan, needle: str) -> set[str]:
    """Which sides of the join now carry a `Filter` mentioning `needle`.

    Keyed on the predicate's text rather than on "is there a Filter", because
    `reject_null_join_keys` also adds one and the two rules must be told apart: a test
    that only asked whether a side had *a* filter passed while the wrong rule fired.
    """
    join = _the_join(plan)
    return {
        side
        for side, node in (("left", join.left), ("right", join.right))
        if isinstance(node, Filter) and needle in repr(node.predicate)
    }


def test_a_side_pure_filter_pushes_into_an_inner_stream_join():
    """The rewrite that pays: a buffered row filtered before the join is never buffered."""
    rewritten = _rewrite(_stream_join("inner", predicate=col("v") > lit(5)))
    assert _sides_carrying(rewritten, "> lit(5)") == {"left"}


def test_a_filter_on_the_preserved_side_of_a_left_join_still_pushes():
    """A left row removed before the join simply never appears — same rows, less state."""
    rewritten = _rewrite(_stream_join("left", predicate=col("v") > lit(5)))
    assert _sides_carrying(rewritten, "> lit(5)") == {"left"}


def test_a_filter_on_the_null_supplying_side_of_a_left_join_does_not_push():
    """Pushing there does not remove output rows, it *replaces* matched rows with
    null-padded ones — a different answer, not a smaller one."""
    rewritten = _rewrite(_stream_join("left", predicate=col("v_right") > lit(5)))
    assert _sides_carrying(rewritten, "> lit(5)") == set()


def test_a_filter_on_the_null_supplying_side_of_a_right_join_does_not_push():
    rewritten = _rewrite(_stream_join("right", predicate=col("v") > lit(5)))
    assert _sides_carrying(rewritten, "> lit(5)") == set()


@pytest.mark.parametrize("predicate_col", ["v", "v_right"])
def test_a_full_outer_join_pushes_into_neither_side(predicate_col):
    """Both sides supply nulls, so neither can be filtered before the join."""
    rewritten = _rewrite(_stream_join("full", predicate=col(predicate_col) > lit(5)))
    assert _sides_carrying(rewritten, "> lit(5)") == set()


def test_the_pushdown_preserves_the_join_kind():
    """A rule that rebuilt the node without `how` would silently turn an outer join into
    an inner one — the rows would simply be missing."""
    assert _the_join(_rewrite(_stream_join("left", predicate=col("v") > lit(5)))).how == "left"


# --- reject_null_join_keys, and what an outer join forbids ----------------------------


def test_null_keys_are_rejected_on_both_sides_of_an_inner_stream_join():
    """A null-keyed row matches nothing and is buffered for the whole window regardless,
    so removing it is free state."""
    assert _sides_carrying(_rewrite(_stream_join("inner")), "is_not_null") == {"left", "right"}


@pytest.mark.parametrize(
    ("how", "expected"),
    [("left", {"right"}), ("right", {"left"}), ("full", set())],
)
def test_a_preserved_sides_null_keys_are_its_output_not_its_dead_weight(how, expected):
    """An outer join emits its preserved side's unmatched rows null-padded, and a
    null-keyed row is exactly an unmatched one — so removing it deletes output."""
    assert _sides_carrying(_rewrite(_stream_join(how)), "is_not_null") == expected
