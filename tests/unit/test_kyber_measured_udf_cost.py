"""Kyber prices a `map_batches` by what Core measured its `fn` to cost.

The loop was half-built: `core.udf.strategy` times every `fn` on a sample to size its
batches, and Kyber's cost model priced *every* CPU `map_batches` at the trivial-column-map
rate regardless. A stage running a hundred microseconds a row was therefore the cheapest node
in its plan, so the optimizer had no reason to push a selective filter below it — the exact
rewrite the accelerator factor beside it exists to produce.

These tests pin the properties that make closing it safe: cold is unchanged, the measurement
is spent, and it is bounded at both ends.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa
import pytest

from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.column_tables import UDF_ROW_SECONDS_KEY
from batcher.kyber.cost.model import CostModel
from batcher.metadata.udf_stats import udf_cost_key
from batcher.plan.logical import MapBatches, Scan
from batcher.plan.schema import SchemaRef

pytestmark = pytest.mark.unit


def cheap_udf(batch):  # module-level, so its identity is stable across runs
    return batch


def expensive_udf(batch):
    return batch


def _plan(fn) -> MapBatches:
    scan = Scan(0, SchemaRef.from_arrow(pa.schema([("a", pa.int64())])))
    return MapBatches(scan, fn=fn)


def _cpu_cost(plan: MapBatches, learned: dict | None = None) -> float:
    return CostModel(CardinalityEstimator(sources=[], learned=learned)).op_cost(plan).cpu


def test_an_unmeasured_udf_is_priced_exactly_as_before() -> None:
    """A cold hub, a lambda, a first run: nothing measured, nothing changed."""
    assert _cpu_cost(_plan(cheap_udf), learned=None) == _cpu_cost(_plan(cheap_udf), learned={})


def test_a_measured_expensive_udf_costs_more_than_an_unmeasured_one() -> None:
    """The point of the loop: a stage Core timed as slow stops being the cheapest node.

    1 us/row is 20x the trivial-map prior, and the cost rises by exactly that — the model
    spends the measurement rather than merely noticing it.
    """
    learned = {UDF_ROW_SECONDS_KEY: {udf_cost_key(expensive_udf): 1e-6}}

    measured = _cpu_cost(_plan(expensive_udf), learned)
    unmeasured = _cpu_cost(_plan(expensive_udf), None)

    assert measured == pytest.approx(unmeasured * 20.0)


def test_a_udf_measured_cheaper_than_the_prior_is_not_priced_below_it() -> None:
    """An opaque Python call the engine cannot fuse must not undercut a projection on the
    strength of one sample, so the measurement only ever raises the cost."""
    learned = {UDF_ROW_SECONDS_KEY: {udf_cost_key(cheap_udf): 1e-12}}

    assert _cpu_cost(_plan(cheap_udf), learned) == _cpu_cost(_plan(cheap_udf), None)


def test_a_pathological_measurement_cannot_dominate_the_plan() -> None:
    """A `fn` timed on a thrashing machine, or one whose first call paid an import, is capped
    rather than allowed to swamp every other operator."""
    absurd = {UDF_ROW_SECONDS_KEY: {udf_cost_key(expensive_udf): 1e3}}  # 1000 s/row
    plausible = {UDF_ROW_SECONDS_KEY: {udf_cost_key(expensive_udf): 5e-6}}

    assert _cpu_cost(_plan(expensive_udf), absurd) == _cpu_cost(_plan(expensive_udf), plausible)


def test_an_accelerator_stage_keeps_its_own_factor() -> None:
    """The GPU branch is untouched — a measured CPU cost must not replace it."""
    gpu = dataclasses.replace(_plan(expensive_udf), num_gpus=1.0)
    learned = {UDF_ROW_SECONDS_KEY: {udf_cost_key(expensive_udf): 5e-8}}

    assert _cpu_cost(gpu, learned) == _cpu_cost(gpu, None)


def test_core_and_kyber_spell_the_udf_identity_the_same_way() -> None:
    """The measurement is worthless if the writer and the reader disagree, and they nearly
    did: Core keys a lambda by its defining line, Kyber's cardinality identity does not."""
    from batcher.core.udf import strategy as strat

    assert strat._fn_probe_key(expensive_udf) == udf_cost_key(expensive_udf)


def test_the_measurement_survives_a_round_trip_through_the_hub() -> None:
    """Core records, Kyber loads: the two halves must meet in `metadata.udf_stats`."""
    from batcher.kyber.learning import load_learned_stats
    from batcher.metadata.backends.in_process import InProcessBackend
    from batcher.metadata.hub import MetadataHub
    from batcher.metadata.udf_stats import record_udf_row_seconds

    hub = MetadataHub(InProcessBackend())
    record_udf_row_seconds(hub, udf_cost_key(expensive_udf), 5e-5)

    table = load_learned_stats(hub).get(UDF_ROW_SECONDS_KEY) or {}

    assert table.get(udf_cost_key(expensive_udf)) == pytest.approx(5e-5)
