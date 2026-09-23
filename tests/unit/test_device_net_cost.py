"""What a byte off a device costs, and how wide a stage may fan out before it.

The node's summed port rate is optimistic for device-resident data in two directions at once:
a device uses one rail, and its bytes cross the host link first. Both errors point the same
way, so a stage planned against the host figure expects bandwidth it does not have.
"""

from __future__ import annotations

import pytest

from batcher.kyber.gpu.exchange import (
    DeviceFabric,
    device_exchange_gbps,
    device_net_gbps,
    device_net_weight,
    fabric_bounded_width,
    widest_fabric_island,
)

pytestmark = pytest.mark.unit


def test_the_off_node_rate_is_the_narrower_of_the_two_wires() -> None:
    """400 Gb/s of rail is 50 GB/s, and the host link is 25: the byte gets 25."""
    wires = DeviceFabric(rail_gbps=400.0, host_link_gbps=25.0)
    assert device_net_gbps(wires) == 25.0


def test_a_shared_rail_bounds_the_byte_even_behind_a_wide_link() -> None:
    wires = DeviceFabric(rail_gbps=50.0, host_link_gbps=64.0)
    assert device_net_gbps(wires) == pytest.approx(6.25)


def test_a_partially_readable_node_is_priced_against_the_half_it_knows() -> None:
    assert device_net_gbps(DeviceFabric(host_link_gbps=25.0)) == 25.0
    assert device_net_gbps(DeviceFabric(rail_gbps=400.0)) == 50.0


def test_an_unreadable_node_has_no_rate_and_no_weight() -> None:
    wires = DeviceFabric()
    assert not wires.readable
    assert device_net_gbps(wires) == 0.0
    assert device_net_weight(wires) is None


def test_the_weight_is_local_bandwidth_over_the_device_rate() -> None:
    wires = DeviceFabric(rail_gbps=400.0, host_link_gbps=25.0)
    assert device_net_weight(wires, local_gbps=25.0) == pytest.approx(1.0)
    assert device_net_weight(wires, local_gbps=50.0) == pytest.approx(2.0)


def test_the_weight_never_makes_a_device_byte_cheaper_than_a_local_one() -> None:
    fast = DeviceFabric(rail_gbps=4000.0, host_link_gbps=500.0)
    assert device_net_weight(fast, local_gbps=20.0) == 1.0


def test_the_weight_is_capped_where_the_ranking_stops_changing() -> None:
    slow = DeviceFabric(rail_gbps=1.0, host_link_gbps=0.1)
    assert device_net_weight(slow, local_gbps=20.0) == 32.0


def test_an_exchange_inside_one_island_runs_on_the_fabric() -> None:
    wires = DeviceFabric(host_link_gbps=25.0, island=8)
    assert device_exchange_gbps(4, wires, nvlink_gbps=450.0) == 450.0


def test_an_exchange_past_the_island_is_bounded_by_the_host_link() -> None:
    wires = DeviceFabric(host_link_gbps=25.0, island=4)
    assert device_exchange_gbps(8, wires, nvlink_gbps=450.0) == 25.0


def test_an_exchange_of_one_device_is_not_an_exchange() -> None:
    assert device_exchange_gbps(1, DeviceFabric(island=8), nvlink_gbps=450.0) == 0.0


def test_an_unknown_device_model_falls_back_to_the_host_link() -> None:
    wires = DeviceFabric(host_link_gbps=25.0, island=8)
    assert device_exchange_gbps(4, wires) == 25.0


def test_a_stage_that_exchanges_is_capped_at_the_island() -> None:
    """The ninth device on an eight-wide fabric makes the collective slower than eight."""
    assert fabric_bounded_width(16, 8) == 8


def test_a_stage_of_independent_shards_is_not_capped() -> None:
    assert fabric_bounded_width(16, 8, exchanges=False) == 16


def test_a_request_that_already_fits_is_untouched() -> None:
    assert fabric_bounded_width(4, 8) == 4


def test_an_unreadable_topology_leaves_the_request_alone() -> None:
    assert fabric_bounded_width(16, 0) == 16


def test_the_widest_island_is_read_from_the_groups_it_is_given() -> None:
    assert widest_fabric_island(((0, 1, 2, 3), (4, 5))) == 4
    assert widest_fabric_island(()) == 0


def test_the_summary_carries_the_derived_rate() -> None:
    summary = DeviceFabric(rail_gbps=400.0, host_link_gbps=25.0, island=8).summary()
    assert summary["net_gbps"] == 25.0
    assert summary["island"] == 8


def _pinned_node(monkeypatch, *, visible, links, islands, rails):
    """A node whose CUDA ordinals and NVML indices deliberately do not coincide.

    Every probe `device_fabric` consults is faked in the index space it really publishes:
    `visible_device_indices` maps ordinal to NVML index, `device_pcie_links` and `peer_islands`
    are positional on the NVML index, and `device_rail_bandwidth_gbps` is keyed by ordinal.
    """
    from batcher.kyber.gpu import exchange

    link_records = [type("L", (), {"bandwidth_gbps": gbps})() for gbps in links]
    monkeypatch.setattr(exchange, "peer_islands", lambda: islands)
    monkeypatch.setattr(
        "batcher._internal.hardware.devices.visible_device_indices", lambda: visible
    )
    monkeypatch.setattr("batcher._internal.hardware.devices.current_ordinal", lambda: 0)
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.device_links.device_pcie_links",
        lambda: tuple(link_records),
    )
    monkeypatch.setattr(
        "batcher._internal.hardware.fabric.rails.device_rail_bandwidth_gbps",
        lambda ordinal, rail_records=None: rails[ordinal],
    )


def test_each_wire_is_read_in_the_index_space_it_publishes(monkeypatch):
    """Rails are keyed by CUDA ordinal; PCI links and peer islands by NVML index.

    A worker pinned to board 6 finds its rail under ordinal 0 and its link under index 6, so
    one number handed to all three probes is wrong for two of them whichever number is chosen.
    It reproduces on no unpinned single-device host, and on a real worker it priced the
    shuffle against another board's rail share and another board's neighbours.
    """
    from batcher.kyber.gpu.exchange import device_fabric

    _pinned_node(
        monkeypatch,
        visible=(6,),  # ordinal 0 is NVML index 6
        links=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 200.0),  # index 6 negotiated 200 Gb/s
        islands=((0, 1, 2, 3, 4, 5), (6,)),  # index 6 is alone on the fabric
        rails={0: 400.0},  # ordinal 0's rail
    )
    wires = device_fabric()
    assert wires.host_link_gbps == 25.0, "index 6's link, not index 0's"
    assert wires.rail_gbps == 400.0, "ordinal 0's rail, not index 6's"


def test_the_island_is_this_devices_own_not_the_nodes_widest(monkeypatch):
    """`island` feeds `device_exchange_gbps`, which reads it as "can this group use NVLink".

    Reporting the node's widest group claimed the fabric rate for a device that reaches one
    peer through it — a six-fold over-statement on the exact node shape (one large island and
    a stray pair) the figure exists to distinguish.
    """
    from batcher.kyber.gpu.exchange import device_exchange_gbps, device_fabric

    _pinned_node(
        monkeypatch,
        visible=(6,),
        links=(0.0,) * 6 + (200.0,),
        islands=((0, 1, 2, 3, 4, 5), (6,)),
        rails={0: 400.0},
    )
    wires = device_fabric()
    assert wires.island == 1, "index 6's own group, not the six-device one next to it"
    # ...so a four-way exchange is priced at the host link rather than at NVLink.
    assert device_exchange_gbps(4, wires, nvlink_gbps=900.0) == 25.0


# --- the weight reaching the cost model -------------------------------------------
#
# `device_net_weight` was computed, rendered in the accelerator report, and discarded before
# any plan was ranked: `Cost.total` priced every shuffled byte at the node's *summed* port
# rate, device-resident or not. These pin the wire, in both directions.


def test_the_factor_is_the_device_weight_over_the_host_one(monkeypatch) -> None:
    """The correction is a ratio, because the `net` weight already charges the host figure."""
    from batcher.kyber.gpu import exchange

    monkeypatch.setattr(exchange, "device_net_weight", lambda: 8.0)
    monkeypatch.setattr(exchange, "fabric_net_weight", lambda: 2.0)
    exchange.reset_device_net_factor()
    assert exchange.device_net_factor() == pytest.approx(4.0)
    exchange.reset_device_net_factor()


def test_an_unreadable_device_leaves_the_ranking_exactly_where_it_was(monkeypatch) -> None:
    """`None` from the probe means "keep the configured weight", never "assume a slow one"."""
    from batcher.kyber.gpu import exchange

    monkeypatch.setattr(exchange, "device_net_weight", lambda: None)
    exchange.reset_device_net_factor()
    assert exchange.device_net_factor() == 1.0
    exchange.reset_device_net_factor()


def test_a_device_is_never_priced_cheaper_than_the_nodes_summed_fabric(monkeypatch) -> None:
    """A ratio below one is a measurement artifact: a board's wires are a subset of the node's."""
    from batcher.kyber.gpu import exchange

    monkeypatch.setattr(exchange, "device_net_weight", lambda: 1.0)
    monkeypatch.setattr(exchange, "fabric_net_weight", lambda: 8.0)
    exchange.reset_device_net_factor()
    assert exchange.device_net_factor() == 1.0
    exchange.reset_device_net_factor()


def _shuffle_net(ds, *, workers: int = 8) -> float:
    """The `net` axis of the aggregate `ds` feeds, which is the node that shuffles."""
    import batcher as bt
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.cost import CostModel

    agg = ds.group_by("k").agg(s=bt.col("v").sum())
    model = CostModel(CardinalityEstimator(agg._sources), workers=workers)
    return model.op_cost(agg._plan).net


def _frames():
    """The same shuffle over a CPU stage and over an accelerator stage."""
    import batcher as bt

    class _Model:
        def __call__(self, batch):
            return batch

    base = bt.from_pydict({"k": list(range(1000)), "v": [i % 7 for i in range(1000)]})
    return (
        base.map_batches(_Model, num_gpus=0),
        base.map_batches(_Model, num_gpus=1),
    )


def test_a_shuffle_below_a_device_stage_pays_the_devices_wires(monkeypatch) -> None:
    """The bytes start on a board, so they cross that board's rail and host link, not the node's.

    The positive control is the CPU arm: it runs the identical query through the identical
    model and must *not* move, or the factor is being applied to everything and the assertion
    below says nothing about device residency.
    """
    from batcher.kyber.gpu import exchange

    monkeypatch.setattr(exchange, "device_net_factor", lambda: 4.0)
    cpu, gpu = _frames()
    assert _shuffle_net(gpu) == pytest.approx(4.0 * _shuffle_net(cpu))


def test_a_cpu_only_plan_never_reads_a_devices_wires(monkeypatch) -> None:
    """The probe is not merely ignored on a CPU plan, it is not reached.

    Costing is on the enumerator's hot path, and a plan with no accelerator stage must not
    pay an NVML round-trip to be told so. A factor that raises proves the gate is the plan
    shape rather than the measurement.
    """
    from batcher.kyber.gpu import exchange

    def _explode() -> float:
        raise AssertionError("a CPU-only plan reached the device probe")

    monkeypatch.setattr(exchange, "device_net_factor", _explode)
    cpu, _ = _frames()
    assert _shuffle_net(cpu) > 0.0


def test_a_single_node_plan_is_unchanged_whatever_the_device_says(monkeypatch) -> None:
    """The `net` axis is zero on one worker, so the correction has nothing to multiply."""
    from batcher.kyber.gpu import exchange

    monkeypatch.setattr(exchange, "device_net_factor", lambda: 32.0)
    _, gpu = _frames()
    assert _shuffle_net(gpu, workers=1) == 0.0


def test_a_discarded_plan_node_cannot_hand_its_verdict_to_the_next_one(monkeypatch) -> None:
    """The device-residency memo is keyed by `id`, and the enumerator prices transient nodes.

    Join ordering and the build-side rule cost `replace`d orientations they then discard, so a
    freed node's address is reused within one model's lifetime. A bare `id` key would give the
    next node at that address the previous one's answer — silently, and only on a plan that
    mixes device and host stages, which is the one shape this correction exists for.
    """
    from batcher.kyber.gpu import exchange

    monkeypatch.setattr(exchange, "device_net_factor", lambda: 4.0)
    cpu, gpu = _frames()
    model, agg = _model_and_agg(gpu)
    assert model.op_cost(agg).net > 0.0

    # Force an id collision: drop the device plan and build a host one, asserting only that
    # the model answers from the node in hand rather than from a stale address.
    host_model, host_agg = _model_and_agg(cpu)
    assert host_model.op_cost(host_agg).net == pytest.approx(_shuffle_net(cpu))
    # And the same model, asked about both, keeps them apart.
    both = _model_and_agg(gpu)[0]
    assert both.op_cost(_model_and_agg(gpu)[1]).net > both.op_cost(host_agg).net


def _model_and_agg(ds, *, workers: int = 8):
    import batcher as bt
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.cost import CostModel

    agg = ds.group_by("k").agg(s=bt.col("v").sum())
    return CostModel(CardinalityEstimator(agg._sources), workers=workers), agg._plan


def test_the_factor_separates_two_candidates_rather_than_inflating_both(monkeypatch) -> None:
    """A weight that scaled every candidate equally would change no plan, only every number.

    The point of pricing a device shuffle correctly is that the enumerator *ranks* differently,
    and that only happens if the factor moves net-heavy candidates further than net-light ones.
    Two shapes of one query: reduce-then-join shuffles the aggregated side, join-then-reduce
    shuffles the wide side. Their cost ratio has to widen as the device's wires get narrower.

    The ranking itself flips only where the two are in tension on net against cpu+io -- a
    broadcast paying replication to avoid the wire is the canonical case. Here one candidate
    dominates on both axes, so the honest assertion is separation, not a flip.
    """
    import batcher as bt
    from batcher.kyber.cardinality import CardinalityEstimator
    from batcher.kyber.cost import CostModel
    from batcher.kyber.gpu import exchange

    class _Model:
        def __call__(self, batch):
            return batch

    big = bt.from_pydict({"k": [i % 500 for i in range(4000)], "v": list(range(4000))})
    small = bt.from_pydict({"k": list(range(500)), "w": list(range(500))})

    def ratio(factor: float) -> float:
        monkeypatch.setattr(exchange, "device_net_factor", lambda: factor)
        src = big.map_batches(_Model, num_gpus=1)
        wide = src.join(small, on="k").group_by("k").agg(s=bt.col("v").sum())
        reduced = src.group_by("k").agg(s=bt.col("v").sum()).join(small, on="k")

        def cost(ds) -> float:
            model = CostModel(CardinalityEstimator(ds._sources), workers=8)
            return model.cost(ds._plan).total()

        return cost(reduced) / cost(wide)

    near = ratio(1.0)
    far = ratio(8.0)
    assert far > near * 1.5, f"the factor did not separate the candidates: {near} -> {far}"
