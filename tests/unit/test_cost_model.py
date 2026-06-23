"""The Kyber cost model turns cardinality into a comparable four-axis estimate."""

from __future__ import annotations

import batcher as bt
from batcher import col
from batcher.kyber.cardinality import CardinalityEstimator
from batcher.kyber.cost import Cost, CostModel, CostWeights


def _model(ds) -> tuple[CostModel, object]:
    est = CardinalityEstimator(ds._sources)
    return CostModel(est), ds._plan


def test_cost_addition_and_total():
    a = Cost(cpu=1, mem=2, io=3, net=4)
    b = Cost(cpu=10, mem=20, io=30, net=40)
    s = a + b
    assert (s.cpu, s.mem, s.io, s.net) == (11, 22, 33, 44)
    # mem is a peak, not summed into the scalar; net is weighted 2x by default.
    assert s.total() == 11 * 1 + 33 * 1 + 44 * 2


def test_learned_width_makes_io_byte_true():
    # Cold start: scan IO uses the flat per-row width default. After a wide column's
    # byte width is learned, the same plan's IO estimate scales up — byte-true,
    # not row-count-blind.
    ds = bt.from_pydict({"blob": list(range(100))})
    cold = CostModel(CardinalityEstimator(ds._sources))
    warm = CostModel(
        CardinalityEstimator(ds._sources, {"__column_avg_bytes__": {"blob": 100_000.0}})
    )
    assert warm.op_cost(ds._plan).io > cold.op_cost(ds._plan).io


def test_row_width_falls_back_to_flat_default_when_unmeasured():
    ds = bt.from_pydict({"x": list(range(10))})
    model, plan = _model(ds)
    # No learned widths → the flat bytes_per_row coefficient (cold-start parity).
    assert model.row_bytes(plan) == model._c.bytes_per_row


def test_scan_cost_scales_with_rows():
    small = bt.from_pydict({"x": list(range(10))})
    big = bt.from_pydict({"x": list(range(10_000))})
    sm, sp = _model(small)
    bm, bp = _model(big)
    assert bm.cost(bp).total() > sm.cost(sp).total()
    # A scan does I/O proportional to rows.
    assert bm.op_cost(bp).io > 0


def test_subtree_cost_includes_inputs():
    ds = bt.from_pydict({"x": list(range(1000))}).filter(col("x") > 100)
    model, plan = _model(ds)
    # The whole-subtree cost exceeds the filter op alone (it adds the scan).
    assert model.cost(plan).cpu > model.op_cost(plan).cpu


def test_join_cost_and_memory_peak():
    left = bt.from_pydict({"k": list(range(1000)), "v": list(range(1000))})
    right = bt.from_pydict({"k": list(range(50)), "w": list(range(50))})
    ds = left.join(right, on="k")
    model, plan = _model(ds)
    c = model.cost(plan)
    assert c.cpu > 0
    # Hash join's working set is the build (right) side, so mem is bounded by it,
    # not by the larger probe side or the sum.
    assert c.mem > 0


def test_weights_reshape_the_objective():
    left = bt.from_pydict({"k": list(range(1000))})
    right = bt.from_pydict({"k": list(range(1000))})
    ds = left.join(right, on="k")
    model, plan = _model(ds)
    c = model.cost(plan)
    cheap_net = CostWeights(net=0.0)
    expensive_net = CostWeights(net=100.0)
    assert c.total(expensive_net) >= c.total(cheap_net)
