"""Learned per-family memory sizing — Carbonite consumes measured `m_peak_bytes`.

The headline gap these cover: `OperatorFeedback.m_peak_bytes` is recorded to the hub
but was never consumed — admission / spill / reservation / per-task grants all sized
from the PLAN estimate alone. `carbonite.memory.learned.LearnedMemoryModel` fits a
per-operator-family bytes-per-input-row figure from those measured peaks, and the
sizing decisions blend the plan estimate toward it. Each test proves both halves of the
contract: the tuned parameter moves when the hub has measured stats, and a cold store
falls back to the exact plan-only behavior. The result-invariance test proves the
lever these decisions drive (in-memory vs the out-of-core spill route) yields a
byte-identical result either way.
"""

from __future__ import annotations

import pyarrow as pa

import batcher as bt
from batcher import Config, col, config_context
from batcher.carbonite import ResourceManager
from batcher.carbonite.base import ResourceContext
from batcher.carbonite.memory.learned import (
    LearnedMemoryModel,
    _canonical_kind,
    _fit,
    _memory_basis_rows,
    _upper_quantile,
    learned_memory_model,
)
from batcher.carbonite.policies import BudgetingAdmission, DefaultSchedulingPolicy
from batcher.config import active_config
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend
from batcher.plan.feedback import OperatorFeedback
from batcher.plan.ids import OpId
from batcher.plan.physical import PhysicalOp, PhysicalPlan, PlanProperties
from batcher.plan.resource import ResourceBounds

# Default per-row assumption the plan estimate uses; a measured wider footprint scales up.
_ROW_BYTES = active_config().optimizer.row_bytes


def _hub() -> MetadataHub:
    return MetadataHub(InProcessBackend())


def _seed(
    hub: MetadataHub, kind: str, bytes_per_row: float, *, n: int = 30, rows: int = 1000
) -> None:
    """Record `n` measured peaks for `kind` so its learned bytes-per-row == `bytes_per_row`."""
    for _ in range(n):
        hub.record(
            OperatorFeedback(
                op_id=OpId(1),
                kind=kind,
                n_actual=rows // 10,
                t_op_ms=1.0,
                m_peak_bytes=int(bytes_per_row * rows),
                selectivity=0.1,
                batch_size=16384,
                n_input=rows,
            )
        )


def _plan(kind: str, m_max_bytes: int) -> PhysicalPlan:
    op = PhysicalOp(
        op_id=OpId(1),
        kind=kind,
        backend="native",
        algorithm="",
        bounds=ResourceBounds(m_max_bytes=m_max_bytes, c_max_credits=0, n_max_parallelism=0),
        inputs=(),
    )
    return PhysicalPlan(ir={}, output_schema=None, ops=(op,))


def _plan_with_est_rows(kind: str, m_max_bytes: int, est_rows: float) -> PhysicalPlan:
    """`_plan`, but carrying the `est_rows` Kyber's `annotate_ops` stamps on every op."""
    op = PhysicalOp(
        op_id=OpId(1),
        kind=kind,
        backend="native",
        algorithm="",
        bounds=ResourceBounds(m_max_bytes=m_max_bytes, c_max_credits=0, n_max_parallelism=0),
        inputs=(),
        properties=PlanProperties(est_rows=est_rows),
    )
    return PhysicalPlan(ir={}, output_schema=None, ops=(op,))


# --- the learned model itself ------------------------------------------------


def test_bytes_per_row_learned_from_measured_peaks():
    hub = _hub()
    _seed(hub, "aggregate", bytes_per_row=256.0)
    model = learned_memory_model(hub)
    # Plan-op kind "Aggregate" (LogicalPlan class name) resolves the native "aggregate" stats.
    assert model.bytes_per_row("Aggregate") == 256.0
    assert model.max_bytes_per_row() == 256.0


def test_peak_rss_high_water_dominates_the_arrow_estimate():
    # When the measured process RSS high-water exceeds the Arrow-size estimate (scratch,
    # fragmentation, off-pool buffers), the learned model fits against the RSS — never
    # under-sizing the true footprint.
    hub = _hub()
    rows = 1000
    for _ in range(30):
        hub.record(
            OperatorFeedback(
                op_id=OpId(1),
                kind="aggregate",
                n_actual=rows // 10,
                t_op_ms=1.0,
                m_peak_bytes=100 * rows,  # Arrow estimate: 100 B/row
                peak_rss_bytes=250 * rows,  # measured RSS high-water: 250 B/row (the truth)
                selectivity=0.1,
                batch_size=16384,
                n_input=rows,
            )
        )
    assert learned_memory_model(hub).bytes_per_row("Aggregate") == 250.0


def test_learned_spill_volume_is_fit_and_predicted():
    # Measured spill volume per row is learned for families that spilled, then multiplied by
    # a new plan's estimated rows to predict its spill volume (sizes spill partitions).
    hub = _hub()
    rows = 1000
    for _ in range(30):
        hub.record(
            OperatorFeedback(
                op_id=OpId(1),
                kind="aggregate",
                n_actual=rows // 10,
                t_op_ms=1.0,
                m_peak_bytes=64 * rows,
                spill_bytes=200 * rows,  # 200 B/row actually spilled
                selectivity=0.1,
                batch_size=16384,
                n_input=rows,
            )
        )
    model = learned_memory_model(hub)
    assert model.spill_bytes_per_row("Aggregate") == 200.0
    # A cold family (never spilled) predicts nothing.
    assert model.spill_bytes_per_row("Filter") is None
    # predicted_spill scales with the plan op's estimated rows. With no `est_rows` on the op
    # (the NaN default) it falls back to recovering them as peak ÷ row_bytes.
    assert model.predicted_spill_bytes(_plan("Aggregate", 64 * 5000).ops) > 0
    assert model.predicted_spill_bytes(_plan("Filter", 64 * 5000).ops) == 0


def test_predicted_spill_uses_kybers_row_count_not_a_flat_width_inversion():
    """A wide-row plan must not have its row count recovered with the flat `row_bytes`.

    Kyber sizes `m_max_bytes = rows * row_width(node)` — a *byte-true* width that is far
    above the flat `optimizer.row_bytes` default for blob/embedding columns. Dividing that
    bound back out by the flat constant therefore overstates the row count by exactly
    `row_width / row_bytes`, and the prediction feeds `recommend_spill_partitions`, so a
    wide query over-shards its out-of-core phase. Kyber already publishes the true count at
    `properties.est_rows`; this pins that it is the figure used.
    """
    hub = _hub()
    rows = 1000
    for _ in range(30):
        hub.record(
            OperatorFeedback(
                op_id=OpId(1),
                kind="aggregate",
                n_actual=rows // 10,
                t_op_ms=1.0,
                m_peak_bytes=64 * rows,
                spill_bytes=200 * rows,  # 200 B/row actually spilled
                selectivity=0.1,
                batch_size=16384,
                n_input=rows,
            )
        )
    model = learned_memory_model(hub)
    assert model.spill_bytes_per_row("Aggregate") == 200.0

    # 5,000 rows of 640 B each (10x the flat 64 B default) → a 3.2 MB bound.
    est_rows, width = 5_000.0, 640
    plan = _plan_with_est_rows("Aggregate", int(est_rows * width), est_rows)
    # The truth: 5,000 rows x 200 B/row. The flat-inversion bug would report 10x this,
    # because 3.2 MB / 64 B recovers 50,000 rows rather than 5,000.
    assert model.predicted_spill_bytes(plan.ops) == int(200.0 * est_rows)

    # An operator Kyber could not size carries the `unknown_rows` sentinel, not a count;
    # multiplying a per-row figure by ~1e12 would swamp the total, so it contributes nothing.
    unknown = active_config().optimizer.cardinality.unknown_rows
    sized = _plan_with_est_rows("Aggregate", 64 * 5_000, unknown)
    assert model.predicted_spill_bytes(sized.ops) == 0


def test_cold_store_is_pass_through():
    model = learned_memory_model(_hub())  # no feedback recorded
    assert model.bytes_per_row("Aggregate") is None
    assert model.max_bytes_per_row() is None
    # A pass-through model returns the plan estimate untouched.
    assert model.blend_peak("Aggregate", 1_000_000) == 1_000_000
    assert learned_memory_model(None).blend_peak("Aggregate", 777) == 777


def test_blend_scales_toward_measured_and_clamps():
    hub = _hub()
    _seed(hub, "aggregate", bytes_per_row=_ROW_BYTES * 4)  # 4x the assumed width
    model = learned_memory_model(hub)
    # measured peak = 4x plan; alpha=0.5 → halfway between plan (1x) and measured (4x) = 2.5x.
    assert model.blend_peak("Aggregate", 1_000_000) == 2_500_000
    # An unsized op (plan estimate 0) has nothing to rescale — stays 0.
    assert model.blend_peak("Aggregate", 0) == 0


def test_min_samples_floor():
    hub = _hub()
    _seed(hub, "aggregate", bytes_per_row=256.0, n=3)  # below the sample floor
    assert learned_memory_model(hub).bytes_per_row("Aggregate") is None


# --- decisions that consume the model ---------------------------------------


def test_estimated_bytes_blends_learned_peak():
    hub = _hub()
    _seed(hub, "aggregate", bytes_per_row=_ROW_BYTES * 4)
    plan = _plan("Aggregate", 1_000_000)
    assert ResourceManager().estimated_bytes(plan) == 1_000_000  # cold: plan only
    assert ResourceManager(hub=hub).estimated_bytes(plan) == 2_500_000  # warm: blended up


def test_should_spill_uses_learned_peak():
    hub = _hub()
    _seed(hub, "aggregate", bytes_per_row=_ROW_BYTES * 8)  # 8x → blended 4.5x
    # A plan the cold manager keeps in memory (estimate under budget) spills once the
    # learned peak (much larger) is blended in — Carbonite protects from a real OOM.
    cfg = Config().replace(memory=active_config().memory)
    with config_context(cfg):
        rm_cold = ResourceManager()
        budget = rm_cold._hard_budget()
        # size the plan so cold estimate fits but the 4.5x blend does not
        plan = _plan("Aggregate", int(budget * 0.5))
        assert rm_cold.should_spill(plan) is False
        assert ResourceManager(hub=hub).should_spill(plan) is True


def test_should_spill_cold_never_spills_unsized():
    assert ResourceManager(hub=_hub()).should_spill(_plan("Aggregate", 0)) is False


def test_admission_blends_learned_peak():
    hub = _hub()
    _seed(hub, "aggregate", bytes_per_row=_ROW_BYTES * 4)
    model = learned_memory_model(hub)
    # Envelope = 3 MB. Plan estimate 1 MB fits; the 2.5 MB blended peak still fits →
    # feasible. A plan whose blend crosses the envelope becomes infeasible.
    ctx_cold = ResourceContext(config=active_config(), envelope_bytes=None)
    ctx_warm = ResourceContext(config=active_config(), envelope_bytes=None, memory_model=model)
    adm = BudgetingAdmission(available_bytes=3_000_000, soft_limit=1.0)
    plan = _plan("Aggregate", 1_500_000)  # cold fits (1.5<3); blended 3.75M exceeds 3M
    assert adm.validate(plan, ctx_cold).feasible is True
    assert adm.validate(plan, ctx_warm).feasible is False


def test_scheduling_envelope_memory_from_learned_peak():
    hub = _hub()
    _seed(hub, "aggregate", bytes_per_row=_ROW_BYTES * 4)
    model = learned_memory_model(hub)
    plan = _plan("Aggregate", 4_000_000)
    pol = DefaultSchedulingPolicy()
    cold = pol.envelope(
        plan,
        ResourceContext(config=active_config()),
        requested_workers=1,
        available_bytes=1 << 40,
    )
    warm = pol.envelope(
        plan,
        ResourceContext(config=active_config(), memory_model=model),
        requested_workers=1,
        available_bytes=1 << 40,
    )
    # The per-task grant tracks the blended (larger) peak, so a distributed worker is
    # right-sized to what the family really used rather than the plan guess.
    assert warm.memory_bytes > cold.memory_bytes


# --- result-invariance -------------------------------------------------------


def _rows(tbl: pa.Table) -> list[tuple]:
    return sorted(tuple(r.values()) for r in tbl.to_pylist())


def test_spill_route_is_result_invariant():
    # The lever learned should_spill selects between (in-memory vs out-of-core) must
    # yield a byte-identical result — that is what makes learned spill sizing safe.
    t = pa.table({"k": [i % 11 for i in range(6000)], "v": list(range(6000))})

    def q(**kw):
        return (
            bt.from_arrow(t).group_by("k").agg(s=col("v").sum(), n=col("v").count()).collect(**kw)
        )

    in_memory = q()
    spilled = q(spill=True, num_partitions=8)
    assert _rows(in_memory) == _rows(spilled)


def test_kyber_publishes_the_row_width_it_sized_with():
    """`PlanProperties.row_size` was declared and never populated or read by anything.

    `m_max_bytes == est_rows * row_width` is a lossy product: a consumer needing one factor
    back cannot divide by the flat `optimizer.row_bytes` default without being wrong by
    `row_width / row_bytes`. Carbonite's spill sizing did exactly that. Kyber already
    computes the byte-true width in `annotate_ops` — it just discarded it, leaving the
    contract slot empty. Publishing it makes the envelope self-describing.
    """
    from batcher.kyber import optimize

    ds = bt.from_pydict({"k": [1, 2, 3] * 100, "v": list(range(300))})
    ds = ds.group_by("k").agg(s=col("v").sum())
    phys = optimize(ds._plan, sources=ds._sources)

    sized = [op for op in phys.ops if op.bounds.m_max_bytes > 0]
    assert sized, "expected at least one sized operator"
    for op in sized:
        width = op.properties.row_size
        assert width == width and width > 0, f"{op.kind} published no row_size"

    # For a pipeline breaker the envelope *is* `rows * width`, so the published width is
    # exactly the factor a consumer needs to recover the row count from the byte bound.
    # (A streaming operator is sized to one morsel in flight instead — `min(morsel_rows *
    # width, morsel_bytes)` — so the product identity is deliberately breaker-only.)
    breakers = [op for op in sized if op.kind == "Aggregate"]
    assert breakers, "expected the group-by to be annotated as a breaker"
    for op in breakers:
        assert op.bounds.m_max_bytes == int(op.properties.est_rows * op.properties.row_size)


def test_recovering_rows_from_bytes_uses_the_published_width():
    """The byte-bound fallback divides by `row_size`, not the flat `row_bytes` default."""
    model = LearnedMemoryModel(
        _bytes_per_row={},
        _alpha=0.5,
        _clamp=4.0,
        _row_bytes=64,  # the flat default the old code always divided by
        _spill_per_row={"aggregate": 100.0},
        _unknown_rows=1e12,
    )
    rows, width = 5_000.0, 640  # 10x the flat default
    op = PhysicalOp(
        op_id=OpId(1),
        kind="Aggregate",
        backend="native",
        algorithm="",
        bounds=ResourceBounds(m_max_bytes=int(rows * width), c_max_credits=0, n_max_parallelism=0),
        inputs=(),
        # No `est_rows` (NaN) — force the byte-bound recovery path.
        properties=PlanProperties(row_size=float(width)),
    )
    plan = PhysicalPlan(ir={}, output_schema=None, ops=(op,))
    # Recovers 5,000 rows, not the 50,000 a flat-64 inversion would report.
    assert model.predicted_spill_bytes(plan.ops) == int(100.0 * rows)


def _from_scratch(hub: MetadataHub, cfg) -> tuple[dict[str, float], dict[str, float]]:
    """The fit derived in one pass over every bucket — the oracle for the incremental one."""
    opt = cfg.optimizer
    min_samples = max(1, opt.cost_calibration_min_samples)
    bpr: dict[str, float] = {}
    spr: dict[str, float] = {}
    for kind, rows in hub.op_stats_by_kind().items():
        footprints: list[float] = []
        spills: list[float] = []
        for r in rows:
            peak = max(
                float(r.get("m_peak_bytes", 0) or 0.0),
                float(r.get("peak_rss_bytes", 0) or 0.0),
            )
            basis = _memory_basis_rows(r)
            if peak > 0.0 and basis > 0.0:
                footprints.append(peak / basis)
            spill = float(r.get("spill_bytes", 0) or 0.0)
            if spill > 0.0 and basis > 0.0:
                spills.append(spill / basis)
        canon = _canonical_kind(kind)
        if len(footprints) >= min_samples:
            bpr[canon] = _upper_quantile(footprints)
        if len(spills) >= min_samples:
            spr[canon] = _upper_quantile(spills)
    return bpr, spr


def test_incremental_sample_derivation_matches_a_full_pass():
    """Extending the cached samples fits exactly what re-deriving the bucket would.

    The derivation is incremental because it was the dominant term in a refit — O(bucket)
    float work to absorb O(`_REFIT_AFTER`) new rows, which amortized to hundreds of
    microseconds on *every* query. That is only a safe trade if the samples are identical,
    so this drives the fit through the growth *and* the hub's front-trim (where the cached
    prefix is no longer a prefix and the derivation must start over).
    """
    hub = _hub()
    cfg = active_config()
    seen_trim = False
    previous = 0
    for round_ in range(24):
        _seed(hub, "aggregate", bytes_per_row=8.0 + round_, n=500, rows=1000)
        bucket = len(hub.op_stats_by_kind()["aggregate"])
        seen_trim = seen_trim or bucket < previous
        previous = bucket
        model = _fit(hub, cfg)
        want_bpr, want_spr = _from_scratch(hub, cfg)
        assert model._bytes_per_row == want_bpr, f"round {round_}: bytes-per-row diverged"
        assert model._spill_per_row == want_spr, f"round {round_}: spill-per-row diverged"
    assert seen_trim, "expected the hub to trim a bucket, exercising the re-derivation path"


def test_a_trimmed_bucket_does_not_reuse_the_stale_prefix():
    """A bucket trimmed from the front re-derives rather than extending the old samples.

    The cached prefix is keyed on the bucket's first row *object*. If that check were by
    `id()` a freed row's address could be reused and the stale samples silently accepted,
    which would pin the fit to measurements the hub has already discarded.
    """
    hub = _hub()
    cfg = active_config()
    _seed(hub, "aggregate", bytes_per_row=4.0, n=40, rows=1000)
    assert _fit(hub, cfg)._bytes_per_row["aggregate"] == 4.0

    # Replace the bucket wholesale: same length, entirely different rows and objects.
    bucket = hub.op_stats_by_kind()["aggregate"]
    bucket[:] = [dict(r, m_peak_bytes=9_000) for r in bucket]
    assert _fit(hub, cfg)._bytes_per_row["aggregate"] == 9.0


def test_concurrent_refits_do_not_double_count_the_shared_prefix():
    """Two threads refitting one hub agree with a single-threaded fit.

    `execution.max_concurrent_queries` runs several queries per process, so the incremental
    derivation must not extend the cached sample lists in place: both threads would append to
    the same object and count every new row twice, inflating the learned footprint and with it
    every reservation sized from it.
    """
    import threading

    hub = _hub()
    cfg = active_config()
    _seed(hub, "aggregate", bytes_per_row=6.0, n=200, rows=1000)
    _fit(hub, cfg)  # prime the cache so the threads race on the *reuse* path
    _seed(hub, "aggregate", bytes_per_row=6.0, n=50, rows=1000)

    results: list[float] = []
    lock = threading.Lock()

    def refit():
        model = _fit(hub, cfg)
        with lock:
            results.append(model._bytes_per_row["aggregate"])

    threads = [threading.Thread(target=refit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    want, _ = _from_scratch(hub, cfg)
    assert results, "expected every thread to produce a fit"
    assert set(results) == {want["aggregate"]}, (
        f"concurrent refits diverged from the single-threaded fit "
        f"{want['aggregate']}: {set(results)}"
    )


def test_spill_volume_is_predicted_against_input_rows_not_output_rows():
    """The coefficient is fitted per *input* row, so it must be applied to input rows.

    `_derive_samples` divides a measured `spill_bytes` by `_memory_basis_rows` — the
    operator's input rows. Multiplying that back by `est_rows`, which `annotate_ops`
    stamps with the *output* cardinality, disagrees by exactly the operator's
    selectivity. A 10x-reducing aggregate is the shape that spills most, and it is the
    shape the mismatch under-predicts most.
    """
    model = LearnedMemoryModel(
        _bytes_per_row={},
        _alpha=0.5,
        _clamp=4.0,
        _row_bytes=64,
        _spill_per_row={"aggregate": 200.0},
        _unknown_rows=1e12,
    )
    in_rows, out_rows = 50_000.0, 5_000.0
    scan = PhysicalOp(
        op_id=OpId(0),
        kind="Scan",
        backend="native",
        algorithm="",
        bounds=ResourceBounds(m_max_bytes=0, c_max_credits=0, n_max_parallelism=0),
        inputs=(),
        properties=PlanProperties(est_rows=in_rows),
    )
    agg = PhysicalOp(
        op_id=OpId(1),
        kind="Aggregate",
        backend="native",
        algorithm="",
        bounds=ResourceBounds(m_max_bytes=0, c_max_credits=0, n_max_parallelism=0),
        inputs=(OpId(0),),
        properties=PlanProperties(est_rows=out_rows),
    )
    plan = PhysicalPlan(ir={}, output_schema=None, ops=(scan, agg))
    assert model.predicted_spill_bytes(plan.ops) == int(200.0 * in_rows), (
        "the aggregate consumes 50,000 rows and emits 5,000; a per-input-row coefficient "
        "applied to the output count under-predicts the spill by the reduction ratio"
    )


def test_a_leaf_operator_still_predicts_against_its_own_estimate():
    """A scan has no child to read input rows from, and its own estimate is the right basis."""
    model = LearnedMemoryModel(
        _bytes_per_row={},
        _alpha=0.5,
        _clamp=4.0,
        _row_bytes=64,
        _spill_per_row={"scan": 8.0},
        _unknown_rows=1e12,
    )
    plan = _plan_with_est_rows("Scan", 64 * 1_000, 1_000.0)
    assert model.predicted_spill_bytes(plan.ops) == int(8.0 * 1_000)


def test_blend_peak_uses_the_input_row_basis_when_the_plan_supplies_it():
    """The learned footprint is per input row; blending must not rescale it by selectivity.

    A 10x-reducing aggregate whose family measured 256 B per input row really holds
    ``256 x input_rows``. Recovering a row count from ``plan_estimate / row_size`` recovers
    the *output* count, so the measurement arrives ten times too small and the blend pulls
    the estimate down instead of up.
    """
    model = LearnedMemoryModel(
        _bytes_per_row={"aggregate": 256.0},
        _alpha=1.0,  # take the measurement whole, so the basis is what is under test
        _clamp=1_000.0,  # wide enough that the clamp is not what decides
        _row_bytes=64,
        _spill_per_row={},
        _unknown_rows=1e12,
    )
    in_rows, out_rows, width = 50_000.0, 5_000.0, 64.0
    planned = int(out_rows * width)
    assert model.blend_peak("Aggregate", planned, width, input_rows=in_rows) == int(256.0 * in_rows)
    # Without the basis the older recovery stands, and it reads the output count.
    assert model.blend_peak("Aggregate", planned, width) == int(256.0 * out_rows)
