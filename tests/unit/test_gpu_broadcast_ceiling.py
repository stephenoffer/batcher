"""The device fan-out's replication ceiling has to be a device's memory, not a CPU's cache.

`kyber.gpu.shape.broadcast_join` decides whether a join's build side is small enough to give
every device a copy. It asked `adaptive_build_side` with no threshold, which resolves to
`resolved_broadcast_max_bytes(l3_cache_bytes=0, workers=1)` — the historical **4 MiB** fallback,
which is a share of one CPU's last-level cache on one node.

Applied to a 15 GB board that is wrong by three orders of magnitude, and it declined the fan-out
for joins that fit a device many times over. Measured on a six-T4 fleet at TPC-H sf10, q4 and
q12 have build sides of roughly 240 MB: both were refused, both then ran the entire join on a
single device, at 8.7 s and 8.5 s against CPU-engine answers of 0.33 s and 1.28 s.

Over-estimating the ceiling is bounded — a probe shard that does not fit its share is
subdivided and rerun on the device — while under-estimating has no ladder at all: the fan-out
simply never runs. That asymmetry is why the number is worth getting right rather than merely
safe.
"""

from __future__ import annotations

import pytest

from batcher.config import DistributedConfig

pytestmark = pytest.mark.unit


# --- the budget --------------------------------------------------------------


def test_the_budget_is_a_share_of_a_devices_usable_memory():
    """Not of its nameplate: the replicated side sits beside the shard, the hash table built
    over it, and the CUDA context."""
    dc = DistributedConfig()
    whole = 16.0 * 1e9
    budget = dc.device_replication_bytes(16.0)
    assert 0 < budget < whole * dc.gpu_tree_broadcast_fraction


def test_the_budget_scales_with_the_device():
    dc = DistributedConfig()
    assert dc.device_replication_bytes(40.0) > dc.device_replication_bytes(16.0)


def test_a_zero_fraction_forbids_replication_entirely():
    import dataclasses

    dc = dataclasses.replace(DistributedConfig(), gpu_tree_broadcast_fraction=0.0)
    assert dc.device_replication_bytes(16.0) == 0.0


def test_the_fraction_is_capped_below_the_whole_device():
    """A deployment asking for the whole board would leave nothing for the shard being joined."""
    import dataclasses

    dc = dataclasses.replace(DistributedConfig(), gpu_tree_broadcast_fraction=5.0)
    assert dc.device_replication_bytes(16.0) < 16.0 * 1e9


def test_a_device_beats_a_cache_by_orders_of_magnitude():
    """The point of the change, stated as the comparison that was being made wrongly."""
    from batcher.config import OptimizerConfig

    cpu_ceiling = OptimizerConfig().resolved_broadcast_max_bytes(0)
    assert DistributedConfig().device_replication_bytes(16.0) > 100 * cpu_ceiling


# --- what the router does with it --------------------------------------------


def test_the_device_ceiling_reaches_the_broadcast_decision(monkeypatch):
    """The wiring, not the arithmetic: `broadcast_join` must pass what it was given."""
    from batcher.kyber.gpu import shape

    seen = {}

    def _spy(plan, est, *args, broadcast_max_bytes=None, **kwargs):
        seen["ceiling"] = broadcast_max_bytes
        return plan, []

    # `broadcast_join` imports the name inside the function, so the module that *defines* it is
    # the only target that binds. Patching `shape` as well reads as belt-and-braces and is worse
    # than nothing: it needs `raising=False` (the attribute does not exist), so it silently
    # invents one, and a future refactor that moved the import to module level would find the
    # test still green against a spy nothing calls.
    monkeypatch.setattr("batcher.kyber.rules.selection.adaptive_build_side", _spy)

    import batcher as bt

    left = bt.from_pydict({"k": [1, 2], "a": [1, 2]})
    right = bt.from_pydict({"k": [1, 2], "b": [3, 4]})
    joined = left.join(right, on="k")
    shape.broadcast_join(joined._plan, joined._sources, None, device_bytes=4.5e9)
    assert seen.get("ceiling") == int(4.5e9)


def test_no_device_ceiling_keeps_the_previous_behaviour(monkeypatch):
    """A caller that cannot say what device this runs on must change nothing."""
    from batcher.kyber.gpu import shape

    seen = {}

    def _spy(plan, est, *args, broadcast_max_bytes=None, **kwargs):
        seen["ceiling"] = broadcast_max_bytes
        return plan, []

    monkeypatch.setattr("batcher.kyber.rules.selection.adaptive_build_side", _spy)

    import batcher as bt

    left = bt.from_pydict({"k": [1, 2], "a": [1, 2]})
    right = bt.from_pydict({"k": [1, 2], "b": [3, 4]})
    joined = left.join(right, on="k")
    shape.broadcast_join(joined._plan, joined._sources, None)
    assert seen.get("ceiling") is None


# --- fitting is not the same as paying ---------------------------------------


class _Decision:
    def __init__(self, left_rows, right_rows, broadcast=True, swapped=False):
        self.left_rows = left_rows
        self.right_rows = right_rows
        self.broadcast = broadcast
        self.swapped = swapped


def test_a_build_side_larger_than_the_probe_does_not_pay_to_replicate():
    """TPC-H q4 at sf10 in its clearest form: `orders SEMI lineitem`, 573 K probe rows against
    30 M build rows. It fits a T4 many times over, so a fit-only rule broadcasts it — and six
    devices then each read thirty million rows to answer a query whose entire probe side is half
    a million, dividing almost nothing. Measured: **23.1 s**, against 0.35 s for the CPU engine
    and 11.4 s for the same join left on one device."""
    from batcher.kyber.gpu.shape import _replicating_pays

    assert _replicating_pays(_Decision(left_rows=573_000, right_rows=30_000_000), 6) is False


def test_a_small_dimension_against_a_large_fact_pays():
    """The shape the fan-out exists for: replicate 25 nations, split 60 M line items."""
    from batcher.kyber.gpu.shape import _replicating_pays

    assert _replicating_pays(_Decision(left_rows=60_000_000, right_rows=25), 6) is True


def test_the_verdict_does_not_move_with_the_fleet_width():
    """Both terms scale with the device count, so the comparison is per device and the width
    drops out. A rule that *did* scale with it — `probe > build x devices` — is the aggregate
    fleet-seconds rule, and it is the wrong objective: the devices read concurrently, so it
    refused TPC-H q14 (0.48 GB replicated against a 1.68 GB probe) and gave up a **12.4x**
    speedup to save bytes nobody was waiting on."""
    from batcher.kyber.gpu.shape import _replicating_pays

    decision = _Decision(left_rows=1_000_000, right_rows=100_000)
    assert _replicating_pays(decision, 4) is True
    assert _replicating_pays(decision, 32) is True


def test_a_probe_side_the_fan_out_barely_divides_is_refused():
    """The fan-out must divide more than it replicates, or N devices do N times one device's
    work and finish at the same time."""
    from batcher.kyber.gpu.shape import _replicating_pays

    assert _replicating_pays(_Decision(left_rows=100_000, right_rows=1_000_000), 6) is False


def test_unreadable_sizes_keep_the_broadcast():
    """The ceiling has already established it fits; refusing on an unreadable estimate would
    put the join back on one device, which is the outcome being avoided."""
    from batcher.kyber.gpu.shape import _replicating_pays

    assert _replicating_pays(_Decision(left_rows=0, right_rows=0), 6) is True


def test_a_single_device_fleet_skips_the_ratio_test(monkeypatch):
    """With one device there is no split to buy, and nothing to weigh the replication against."""
    from batcher.kyber.gpu import shape

    monkeypatch.setattr(
        "batcher.kyber.rules.selection.adaptive_build_side",
        lambda plan, est, **kw: (plan, [_Decision(left_rows=1, right_rows=10_000_000)]),
    )
    import batcher as bt

    left = bt.from_pydict({"k": [1, 2], "a": [1, 2]})
    right = bt.from_pydict({"k": [1, 2], "b": [3, 4]})
    joined = left.join(right, on="k")
    assert shape.broadcast_join(joined._plan, joined._sources, None, 4.5e9, 1) is True
    assert shape.broadcast_join(joined._plan, joined._sources, None, 4.5e9, 6) is False
