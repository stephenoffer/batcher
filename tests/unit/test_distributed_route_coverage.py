"""Every operator shape either has a distributed route, or is named here as not having one.

The distributed operator matrix (`tests/differential/test_diff_distributed_operator_matrix.py`)
proves that what distributes is *correct*. It cannot prove that anything distributes at all:
it builds its inputs with `bt.from_arrow`, and an in-memory source is not splittable, so
`_unsupported` treats the plan as "no distributed data to speak of" and runs it on one node —
returning the right rows, from the driver, past a green assertion. Every missing route in this
engine is invisible to it by construction.

This file asks the other question: over a source that IS splittable, which route does the
dispatcher take? It runs the real `_dispatch` with every executor stubbed out, so no Ray, no
cluster, and no data — but also no second model of the routing rules to drift from the first.
That drift is precisely the bug class here: `_dispatcher_handles_aggregate_input` claimed the
dispatcher fused an aggregate over an ASOF join, `_dispatch` had no such branch, and the
mismatch turned off the staging that shape's only distributed path runs through — so
`join_asof(...).agg(...)` raised `PlanError` on real data while every in-memory test passed.

`UNROUTED` is the honest ledger of what distribution genuinely cannot do, with reasons. A shape
that gains a route must be deleted from it in the same commit. The list only shrinks.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher.kyber.optimizer import optimize_logical
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit

LEFT = pa.table(
    {
        "k": pa.array([1, 2, 3], pa.int64()),
        "v": pa.array([1.0, 2.0, 3.0]),
        "g": pa.array(["a", "b", "c"]),
        "t": pa.array([10, 20, 30], pa.int64()),
        "lo": pa.array([0.0, 1.0, 2.0]),
    }
)
RIGHT = pa.table(
    {
        "k": pa.array([1, 2], pa.int64()),
        "rv": pa.array([5.0, 6.0]),
        "rt": pa.array([5, 15], pa.int64()),
        "hi": pa.array([9.0, 9.0]),
    }
)

c = bt.col


def _l() -> bt.Dataset:
    return bt.from_arrow(LEFT)


def _r() -> bt.Dataset:
    return bt.from_arrow(RIGHT)


#: Shape name -> the query, in the reduced form a user writes. Each is optimized before being
#: routed, because Kyber's rewrites (cross join + inequality -> range join) are what the
#: dispatcher actually sees.
SHAPES: dict[str, object] = {
    "filter_project": lambda: _l().filter(c("v") > 1).select("k", "v"),
    "group_by": lambda: _l().group_by("k").agg(s=c("v").sum()),
    "global_agg": lambda: _l().agg(s=c("v").sum()),
    "distinct": lambda: _l().select("k").distinct(),
    "sort": lambda: _l().sort("v"),
    "limit": lambda: _l().limit(2),
    "sample_n": lambda: _l().sample(n=2, seed=1),
    "with_row_index": lambda: _l().with_row_index("i"),
    "union_all": lambda: _l().select("k").union(_l().select("k")),
    "union_distinct": lambda: _l().select("k").union(_l().select("k"), distinct=True),
    "grouped_union": lambda: (
        _l().select("k", "v").union(_l().select("k", "v")).group_by("k").agg(s=c("v").sum())
    ),
    "intersect": lambda: _l().select("k").intersect(_r().select("k")),
    "except": lambda: _l().select("k").except_(_r().select("k")),
    "join": lambda: _l().join(_r(), on="k"),
    "grouped_join": lambda: _l().join(_r(), on="k").group_by("k").agg(s=c("v").sum()),
    "global_agg_over_join": lambda: _l().join(_r(), on="k").agg(n=c("k").count()),
    "asof_join": lambda: _l().join_asof(_r(), left_on="t", right_on="rt", by="k"),
    "range_join": lambda: _l().cross_join(_r()).filter((c("v") >= c("lo")) & (c("v") <= c("hi"))),
    "window_partitioned": lambda: _l().with_columns(s=c("v").sum().over(partition_by="g")),
    "window_global_unordered": lambda: _l().with_columns(s=c("v").sum().over()),
}

#: Shapes the *one-shot* dispatcher cannot route on its own. Each is routed by the adaptive
#: staging loop instead, which the gate turns on for exactly this reason (`resolve_adaptive`),
#: so they still run distributed — they just do it stage by stage.
STAGED: frozenset[str] = frozenset(
    {
        "agg_over_asof_join",
        "agg_over_range_join",
    }
)

#: What distribution genuinely cannot do, with the reason. A standing invitation to delete a
#: line: every entry is a gap, not a decision.
UNROUTED: dict[str, str] = {
    # No equality to co-partition on and no `by` group to hash, so the shuffle every other
    # join uses is unavailable; the result needs one global order over the `on` key.
    "asof_keyless": "no `by` keys: nothing to co-partition, and it needs one global order",
    # `row_number() OVER (ORDER BY x)` and running aggregates need one global row order.
    # `dist/window_stream.py` already implements the algorithm that would lift this —
    # range-partition on the order key, compute per bucket, then apply each bucket's constant
    # prefix offset — but only the single-node streaming/spill path uses it today.
    "window_global_ordered": "one global row order; the ordered-bucket-offset algorithm "
    "exists in dist/window_stream.py but is not wired into the dispatcher",
    # A UDF pipeline feeding a *join*. `_stage_map_prefix` lands a map prefix on shared
    # scratch so the breaker above it sees an ordinary splittable source, but it walks a
    # single-input chain: a join has two operands, and substituting one of them is a
    # different rewrite. The single-input breakers (sort / distinct / window / limit) are
    # routed by that staging and are deliberately absent from this list.
    "map_then_join": "a join has two operands; the map-prefix stage rewrites single-input "
    "chains only",
}

SHAPES["agg_over_asof_join"] = lambda: (
    _l().join_asof(_r(), left_on="t", right_on="rt", by="k").agg(n=c("k").count())
)
SHAPES["agg_over_range_join"] = lambda: (
    _l().cross_join(_r()).filter((c("v") >= c("lo")) & (c("v") <= c("hi"))).agg(n=c("k").count())
)
SHAPES["asof_keyless"] = lambda: _l().join_asof(_r(), left_on="t", right_on="rt")
SHAPES["window_global_ordered"] = lambda: _l().with_columns(r=bt.row_number().over(order_by="t"))


def _mapped() -> bt.Dataset:
    """A `map_batches` pipeline — the prefix of any batch-inference job."""
    return _l().map_batches(lambda b: b)


SHAPES["map_only"] = _mapped
SHAPES["map_then_agg"] = lambda: _mapped().group_by("k").agg(s=c("v").sum())
SHAPES["map_then_sort"] = lambda: _mapped().sort("v")
SHAPES["map_then_distinct"] = lambda: _mapped().select("k").distinct()
SHAPES["map_then_join"] = lambda: _mapped().join(_r(), on="k")
SHAPES["map_then_window"] = lambda: _mapped().with_columns(s=c("v").sum().over(partition_by="g"))
SHAPES["map_then_limit"] = lambda: _mapped().limit(3)


class _FakeSplit:
    """One readable slice. Anything that is not a `WholeSourceSplit` counts as a real split."""

    def __init__(self, batches: list[pa.RecordBatch], schema: pa.Schema) -> None:
        self._batches, self._schema = batches, schema

    def schema(self) -> pa.Schema:
        return self._schema

    def read(self, projection: list[str] | None = None) -> list[pa.RecordBatch]:
        return [b.select(projection) for b in self._batches] if projection else self._batches

    def iter_batches(self):
        yield from self._batches

    def row_count(self) -> int:
        return sum(b.num_rows for b in self._batches)

    def identity(self) -> str:
        return f"fake:{id(self)}"


class _SplittableSource:
    """A source that reports real per-chunk splits, so the dispatcher treats it as
    distributed data.

    This is the whole point of the file. Over a non-splittable source `_unsupported` runs the
    plan on one node and returns the right answer, so a missing route cannot be observed at
    all; over a splittable one it raises, which is what turns a gap into a test failure.
    """

    resident = False

    def __init__(self, table: pa.Table) -> None:
        self._table = table

    def schema(self) -> pa.Schema:
        return self._table.schema

    def splits(self) -> list[object]:
        halves = [self._table.slice(0, 1), self._table.slice(1)]
        return [_FakeSplit(h.to_batches(), self._table.schema) for h in halves]

    def row_count(self) -> int:
        return self._table.num_rows

    def batches(self) -> list[pa.RecordBatch]:
        return self._table.to_batches()


#: Every executor entry point `_dispatch` can route to, as (module path, attribute). Stubbing
#: all of them is what lets the real dispatcher run with no Ray: whichever one it reaches
#: records the route and returns an empty table.
_ROUTES: tuple[tuple[str, str], ...] = (
    ("batcher.dist.executors.map", "_distributed_map"),
    ("batcher.dist.executors.map", "_distributed_map_aggregate"),
    ("batcher.dist.streaming", "stream_distributed_pipeline"),
    ("batcher.dist.executors.aggregate", "_distributed_aggregate"),
    ("batcher.dist.executors.distinct", "_distributed_distinct"),
    ("batcher.dist.executors.sort", "_distributed_sort"),
    ("batcher.dist.executors.window", "_distributed_window"),
    ("batcher.dist.executors.union", "_distributed_union"),
    ("batcher.dist.executors.join", "_distributed_join"),
    ("batcher.dist.executors.join", "_distributed_join_aggregate"),
    ("batcher.dist.executor", "_distributed_asof"),
    ("batcher.dist.executor", "_distributed_range_join"),
    ("batcher.dist.executor", "_staged_aggregate_over_join"),
    ("batcher.dist.executor", "_single_node"),
    ("batcher.dist.flight_aggregate", "execute_aggregate_flight"),
    ("batcher.dist.flight_join", "execute_join_flight"),
    ("batcher.dist.flight_sort", "execute_sort_flight"),
    ("batcher.dist.flight_sort", "execute_topn_flight"),
    ("batcher.dist.flight_window", "execute_window_flight"),
)


@pytest.fixture
def route(monkeypatch):
    """Route `plan` through the real `_dispatch`, returning the executor it chose.

    Returns the executor's name, or ``None`` when the plan reached `_unsupported` — which on
    splittable data is a `PlanError`, i.e. a query a user cannot run.
    """
    import importlib

    from batcher.dist import executor as dex

    taken: list[str] = []

    def stub(name):
        def fn(*_a, **_k):
            taken.append(name)
            return pa.table({})

        return fn

    for mod_path, attr in _ROUTES:
        mod = importlib.import_module(mod_path)
        monkeypatch.setattr(mod, attr, stub(attr), raising=False)
    # `executor` binds `_single_node` at import, so the module attribute must be patched too.
    monkeypatch.setattr(dex, "_single_node", stub("_single_node"), raising=False)
    monkeypatch.setattr(dex, "_require_shared_scratch", lambda _op: None)

    def go(ds):
        from batcher._internal.errors import PlanError

        plan = optimize_logical(ds._plan)
        sources = [_SplittableSource(LEFT), _SplittableSource(RIGHT)]
        taken.clear()
        try:
            dex._dispatch(plan, sources, 2, "disk")
        except PlanError:
            return None
        return taken[0] if taken else None

    return go


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_every_shape_is_routed_staged_or_recorded_as_unrouted(name, route):
    """A shape with none of the three is a query that raises `PlanError` on real data."""
    from batcher.dist.executors.plan_analysis import requires_staging

    plan = optimize_logical(SHAPES[name]()._plan)
    taken = route(SHAPES[name]())
    staged = requires_staging(plan)
    if name in UNROUTED:
        # Only the one-shot route is asserted here. `requires_staging` is not proof of a
        # staged route: it answers True for a `map_batches` prefix under a join or a window
        # (a `MapBatches` reads as a breaker to `_has_breaker`), but staging cannot decompose
        # a UDF prefix — the staged sub-plan is the same refused shape — so those still raise.
        # Which of the listed shapes genuinely run is settled by executing them over a real
        # splittable source, not by re-deriving it from the same predicates.
        assert taken is None, (
            f"{name!r} is listed in UNROUTED but the dispatcher now routes it "
            f"({taken}) — delete the entry"
        )
        return
    assert taken is not None or staged, (
        f"{name!r} has no distributed route and is not in UNROUTED: on splittable data it "
        "raises rather than runs. Add the route, or record it with its reason."
    )


@pytest.mark.parametrize("name", sorted(STAGED))
def test_the_staged_shapes_really_do_require_staging(name):
    """Staging is these shapes' only distributed path, so the gate must be told to turn it on.

    `resolve_adaptive` forces adaptivity on when `requires_staging` is true, and skips it
    otherwise. Answering "no" for a shape `_dispatch` cannot route does not make it run
    one-shot; it makes it raise.
    """
    from batcher.dist.executors.plan_analysis import requires_staging

    assert requires_staging(optimize_logical(SHAPES[name]()._plan))


def test_the_staging_predicate_mirrors_the_dispatchers_real_routes():
    """`_dispatcher_handles_aggregate_input` must not claim a route `_dispatch` lacks.

    It decides whether `requires_staging` says "no need", so a claim wider than the truth does
    not add a route — it removes the fallback. The dispatcher's fused aggregate-over-join paths
    (`_fusable_join_aggregate`, `_aggregate_over_join`) both test `isinstance(j, Join)`, and
    nothing else, so an ASOF or range join must answer False here.
    """
    from batcher.dist.executors.plan_analysis import _dispatcher_handles_aggregate_input
    from batcher.plan.logical import AsofJoin, Join, RangeJoin

    def inner(name):
        node = optimize_logical(SHAPES[name]()._plan)
        while not isinstance(node, (Join, AsofJoin, RangeJoin)) and hasattr(node, "input"):
            node = node.input
        return node

    assert _dispatcher_handles_aggregate_input(inner("join")) is True
    assert _dispatcher_handles_aggregate_input(inner("asof_join")) is False
    assert _dispatcher_handles_aggregate_input(inner("range_join")) is False


def test_a_reduced_union_maps_into_one_shuffle_instead_of_the_driver(route):
    """`union(...).group_by(...)` and both set operators reduce through the aggregate shuffle.

    `_distributed_union` runs each branch to a driver table and concatenates there, so routing
    a reduced union to it moves both inputs whole through one node to answer a query that
    reduces them. The branches share one bucket space instead.
    """
    for name in ("grouped_union", "intersect", "except", "union_distinct"):
        taken = route(SHAPES[name]())
        assert taken in {"_distributed_aggregate", "_distributed_union"}, (name, taken)
        if taken == "_distributed_union":
            # The union executor is still the entry point for UNION (distinct); what matters
            # is that it hands the branches to the shuffle rather than deduplicating a
            # driver-side concatenation. Asserted directly in the differential set-op tests.
            assert name == "union_distinct"


def test_union_branches_must_agree_on_type_before_they_share_a_shuffle():
    """An `Int64` branch against a `Float64` one is refused, not silently split.

    `Union` promotes the pair; independent mappers would skip the promotion and hash `1` and
    `1.0` to different reducers, returning one group as two — invisible single-node.
    """
    from batcher.dist.executors.plan_analysis import shuffle_branches

    ints = pa.table({"k": pa.array([1, 2], pa.int64()), "v": pa.array([1.0, 2.0])})
    floats = pa.table({"k": pa.array([1.0, 3.0], pa.float64()), "v": pa.array([3.0, 4.0])})
    assert shuffle_branches(bt.from_arrow(ints).union(bt.from_arrow(floats))._plan) is None
    assert shuffle_branches(bt.from_arrow(ints).union(bt.from_arrow(ints))._plan) is not None


def test_the_fixture_source_is_actually_splittable():
    """Without this the whole file is vacuous: a non-splittable source makes every missing
    route fall back to one node and pass."""
    from batcher.dist.executor import _is_splittable_source

    assert _is_splittable_source(_SplittableSource(LEFT))
    assert SchemaRef.from_arrow(LEFT.schema) is not None
