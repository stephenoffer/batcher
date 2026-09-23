"""A `map_batches` pipeline reaches Kyber, Carbonite and Core, and which part of each.

`.claude/rules/architecture.md` describes `api` as the one conductor: "Kyber optimizes ->
Carbonite checks feasibility -> Core executes -> metadata flows back". A plan holding a
`map_batches` does not take the relational route -- `api.executors.select` hands it to
`UdfExecutor`, which calls `core.udf.execute.execute_with_udfs` -- and it used to reach
only Core. An audit measured `ResourceManager.validate` called once for a relational
collect and **zero** times for a UDF collect, with `explain()` reporting
``memory_budget_bytes: 0`` and an empty ``carbonite_summary``.

What each subsystem can do here is bounded by one fact: `MapBatches` has no engine IR, so
`kyber.optimize` raises `NotImplementedError` and there is no `PhysicalPlan`. That rules
out the *optimizer* and rules out `ResourceManager.validate`, which takes one. It does not
rule out either subsystem:

* **Kyber** prunes each source's columns (`required_columns_per_source`) and estimates
  cardinality off the logical tree (`StatsEstimator.estimate`), which needs no lowering --
  the `Scan` under a UDF reports its exact row count. `MapBatches` itself is left
  unestimated on purpose: the estimator *will* answer for it by applying a default
  selectivity to an opaque Python function, and publishing that would make `est_error`
  compare a measurement against a guess.
* **Carbonite** admits the query (`admit`) and judges the projected input against the
  memory envelope (`input_exceeds_budget`), the two entry points needing no plan. This
  matters most on exactly this shape: `execute_with_udfs` returns a list, so peak memory
  is the whole output.
* **Core** executes and measures every stage.

These tests pin that split. The controls matter more than usual, because most assertions
here are about a *presence* that a broken measurement would report as absent.
"""

from __future__ import annotations

import json

import pytest

import batcher as bt

pytestmark = pytest.mark.unit


@pytest.fixture
def frame():
    return bt.from_pydict({"x": list(range(2000)), "k": [i % 7 for i in range(2000)]})


def _carbonite_calls(query) -> dict[str, int]:
    """How often each plan-free Carbonite entry point runs while `query` collects."""
    from batcher import carbonite

    counts = {"admit": 0, "input_exceeds_budget": 0, "validate": 0}
    originals = {name: getattr(carbonite.ResourceManager, name) for name in counts}

    def counting(name):
        real = originals[name]

        def spy(self, *args, **kwargs):
            counts[name] += 1
            return real(self, *args, **kwargs)

        return spy

    for name in counts:
        setattr(carbonite.ResourceManager, name, counting(name))
    try:
        query().collect()
    finally:
        for name, real in originals.items():
            setattr(carbonite.ResourceManager, name, real)
    return counts


def test_a_relational_plan_is_admitted_through_validate(frame):
    """The control for the next test, and the contrast it is measured against.

    Without it, "the UDF path calls `admit`" would be equally true of a counter that
    increments on everything, and "it does not call `validate`" of one that never sees a
    call at all.
    """
    counts = _carbonite_calls(lambda: frame.filter(bt.col("x") > 10))
    assert counts["validate"] == 1


def test_a_udf_plan_is_admitted_without_a_physical_plan(frame):
    """Carbonite reaches the UDF path through the entry points that need no plan.

    `validate` stays at zero and that is correct rather than a gap: it takes a
    `PhysicalPlan`, and a plan holding a `MapBatches` has none.
    """
    counts = _carbonite_calls(lambda: frame.filter(bt.col("x") > 10).map_batches(lambda b: b))
    assert counts["admit"] == 1, counts
    assert counts["input_exceeds_budget"] == 1, counts
    assert counts["validate"] == 0, counts


def test_the_optimizer_still_refuses_a_udf_plan(frame):
    """Why the optimizer is absent, asserted rather than assumed.

    This is the fact the whole split rests on. If it ever stops holding, the UDF path
    should take the relational route instead of the one these tests describe.
    """
    from batcher import kyber

    query = frame.filter(bt.col("x") > 10).map_batches(lambda b: b)
    with pytest.raises(NotImplementedError, match="not lowered to the engine IR"):
        kyber.optimize(query._plan, sources=query._sources)

    plain = frame.filter(bt.col("x") > 10)
    assert kyber.optimize(plain._plan, sources=plain._sources) is not None


def test_the_udf_profile_reports_carbonites_reading(frame):
    """`explain()` on an ML pipeline names the subsystem that judged the query."""
    udf = json.loads(
        frame.filter(bt.col("x") > 10).map_batches(lambda b: b).explain(analyze=True, format="json")
    )
    assert udf["carbonite_summary"] == "feasible"
    assert udf["memory_budget_bytes"] > 0
    admission = [d for d in udf["decisions"] if d["subsystem"] == "carbonite"]
    assert len(admission) == 1, udf["decisions"]
    assert admission[0]["detail"]["feasible"] is True
    # Recorded, so a reader can tell this reading from the relational path's `validate`.
    assert admission[0]["detail"]["validated"] is False


def test_kyber_estimates_every_operator_but_the_opaque_one(frame):
    """The exact-where-it-can-be, absent-where-it-cannot split, both halves asserted."""
    import math

    udf = json.loads(
        frame.filter(bt.col("x") > 10).map_batches(lambda b: b).explain(analyze=True, format="json")
    )
    by_kind = {op["kind"]: op for op in udf["ops"]}
    assert set(by_kind) >= {"Scan", "Filter", "MapBatches"}, list(by_kind)

    # The scan's row count is exact, which is the point: it is the operator whose estimate
    # was lost for no reason when the whole tree went unestimated.
    assert by_kind["Scan"]["est_rows"] == 2000
    assert by_kind["Filter"]["est_rows"] is not None
    assert not math.isnan(by_kind["Filter"]["est_rows"])

    # And the deliberate decline, which is the half a "more estimates is better" change
    # would quietly undo.
    assert by_kind["MapBatches"]["est_rows"] is None


def test_core_measures_every_stage_of_a_udf_pipeline(frame):
    udf = json.loads(
        frame.filter(bt.col("x") > 10).map_batches(lambda b: b).explain(analyze=True, format="json")
    )
    assert udf["ops"], "the UDF profile reported no operators at all"
    assert all(op["rows_out"] is not None for op in udf["ops"])
    assert udf["measured"] is True


#: (label, with the pass-through UDF, the same pipeline without it). The UDF is the
#: identity, so the second is the oracle for the first.
_EQUIVALENT_SHAPES = [
    (
        "udf-above-filter",
        lambda f: f.filter(bt.col("x") > 10).map_batches(lambda b: b),
        lambda f: f.filter(bt.col("x") > 10),
    ),
    (
        "udf-below-filter",
        lambda f: f.map_batches(lambda b: b).filter(bt.col("x") > 10),
        lambda f: f.filter(bt.col("x") > 10),
    ),
    (
        "udf-below-aggregate",
        lambda f: f.map_batches(lambda b: b).group_by("k").agg(n=bt.col("x").sum()),
        lambda f: f.group_by("k").agg(n=bt.col("x").sum()),
    ),
]


@pytest.mark.parametrize(
    ("with_udf", "without_udf"),
    [pytest.param(a, b, id=label) for label, a, b in _EQUIVALENT_SHAPES],
)
def test_admitting_the_pipeline_does_not_change_its_result(frame, with_udf, without_udf):
    """Correctness before anything else: admission is a resource concern, not a rewrite.

    A pass-through `map_batches` is the identity, so the same pipeline without it is the
    oracle. Bracketing execution with `ResourceManager.admit()` must not move a row, and
    the aggregate shape is here because it is the one where a resource decision could
    plausibly change partitioning.
    """
    got = with_udf(frame).to_pydict()
    want = without_udf(frame).to_pydict()
    assert got, "the UDF pipeline returned nothing, so the comparison would be vacuous"
    assert sorted(got) == sorted(want)
    for column in want:
        assert sorted(map(repr, got[column])) == sorted(map(repr, want[column])), column


# --- the admission grant, not just the slot -------------------------------------------


def test_the_admission_grant_narrows_the_engine_pool():
    """Holding a slot is half of admission; running inside it is the other half.

    `ResourceManager.admit()` yields an `ExecutionGrant` whose `workers` is the rayon pool
    width the query should request. The relational path applies it through
    `run._with_grant`. The UDF path takes a slot too, and it runs relational operators on
    the engine between its Python stages -- so discarding the grant would admit a query
    against a narrowed pool and then ask for a full-width one, which is exactly the
    oversubscription the grant prevents.
    """
    from batcher.api.orchestration.run import narrowed_to_grant
    from batcher.config import active_config

    ambient = active_config().execution.parallelism

    class Granted:
        workers = 3

    with narrowed_to_grant(Granted()):
        assert active_config().execution.parallelism == 3
    assert active_config().execution.parallelism == ambient, "the narrowing must not leak"


def test_an_unbounded_grant_leaves_the_config_alone():
    """The default path, and the control for the test above.

    `execution.max_concurrent_queries` is 0 by default, which grants `workers=0` meaning
    unbounded. An unconfigured deployment must execute exactly as it did before admission
    reached this path, so the config is untouched rather than set to 0.
    """
    from batcher.api.orchestration.run import narrowed_to_grant
    from batcher.config import active_config

    ambient = active_config().execution.parallelism

    class Unbounded:
        workers = 0

    with narrowed_to_grant(Unbounded()):
        assert active_config().execution.parallelism == ambient
    assert active_config().execution.parallelism == ambient


def test_a_throttled_udf_query_returns_the_same_rows(frame):
    """Admission is a resource concern; throttling must not change the answer."""
    import dataclasses

    from batcher.config import active_config, set_config

    build = lambda: frame.filter(bt.col("x") > 10).map_batches(lambda b: b)  # noqa: E731
    unthrottled = build().to_pydict()

    config = active_config()
    set_config(
        dataclasses.replace(
            config,
            execution=dataclasses.replace(config.execution, max_concurrent_queries=2),
        )
    )
    try:
        throttled = build().to_pydict()
    finally:
        set_config(config)
    assert throttled == unthrottled
    assert throttled == frame.filter(bt.col("x") > 10).to_pydict()
