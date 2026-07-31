"""The neutral fleet-shape contract: locality shares, derived counts, and their degradations.

The shares are the load-bearing part — every tiered cost decision multiplies through them — so
they are tested for the algebraic property that makes them safe to use (they partition the
exchange) as well as for the specific fleets whose distinction they exist to draw.
"""

from __future__ import annotations

import pytest

from batcher.plan.resource import ClusterShape, HardwareProfile, LocalityShares, NodeShape

pytestmark = pytest.mark.unit


def _dense(nodes: int = 4, gpus: int = 8, domain: int = 8, rack: str = "r1") -> ClusterShape:
    """`nodes` hosts of `gpus` devices each, all in one rack."""
    return ClusterShape(
        nodes=tuple(
            NodeShape(
                node_id=f"n{i}",
                cpu_cores=96,
                memory_bytes=1 << 40,
                gpus=gpus,
                accelerator_type="NVIDIA_H100",
                gpu_memory_bytes=80 * (1 << 30),
                nvlink_domain=domain,
                rack=rack,
                fabric_gbps=400.0,
            )
            for i in range(nodes)
        )
    )


def _sparse(nodes: int = 32) -> ClusterShape:
    """`nodes` hosts of one device each, unlabelled — the fleet with no locality to exploit."""
    return ClusterShape(
        nodes=tuple(
            NodeShape(
                node_id=f"n{i}",
                cpu_cores=12,
                gpus=1,
                accelerator_type="NVIDIA_H100",
                gpu_memory_bytes=80 * (1 << 30),
                nvlink_domain=1,
                fabric_gbps=400.0,
            )
            for i in range(nodes)
        )
    )


def _total(shares: LocalityShares) -> float:
    return (
        shares.local
        + shares.intra_domain
        + shares.intra_node
        + shares.intra_rack
        + shares.cross_rack
    )


@pytest.mark.parametrize("workers", [1, 2, 3, 7, 8, 32, 64, 1000])
@pytest.mark.parametrize("unit", ["cpu", "gpu"])
@pytest.mark.parametrize("shape", [_dense(), _sparse(), _dense(nodes=1), ClusterShape()])
def test_shares_partition_the_exchange(shape, workers, unit):
    """The five shares always sum to one and none is negative, for any fleet and any width.

    They are consumed as a probability partition — a weighted average over them is a price —
    so a set that does not sum to one silently rescales every `net` cost that uses it.
    """
    shares = shape.locality_shares(workers, unit=unit)
    assert _total(shares) == pytest.approx(1.0)
    assert (
        min(
            shares.local,
            shares.intra_domain,
            shares.intra_node,
            shares.intra_rack,
            shares.cross_rack,
        )
        >= -1e-12
    )


def test_dense_and_sparse_fleets_are_distinguished():
    """The distinction the whole contract exists for: same devices, different placement.

    Thirty-two devices on four hosts and thirty-two on thirty-two report an identical device
    count and worker count. Before the shape existed both were priced as one flat pool.
    """
    dense = _dense().locality_shares(32, unit="gpu")
    sparse = _sparse().locality_shares(32, unit="gpu")
    assert dense.on_node == pytest.approx(0.25)
    assert sparse.on_node == pytest.approx(1.0 / 32)
    assert dense.off_node < sparse.off_node


def test_nvlink_domain_narrower_than_the_node_splits_the_on_host_share():
    """A node whose fabric is narrower than its device count has two on-host tiers.

    Eight devices in two four-device domains exchange half their on-host traffic over NVLink
    and half over the host bus, and those differ by more than an order of magnitude.
    """
    shares = _dense(nodes=4, gpus=8, domain=4).locality_shares(32, unit="gpu")
    assert shares.intra_domain > 0.0
    assert shares.intra_node > 0.0
    assert shares.intra_domain + shares.intra_node + shares.local == pytest.approx(0.25)


def test_cpu_workers_never_claim_the_device_fabric():
    """A relational shuffle's on-host share is the node tier, never the coherent-fabric tier.

    A CPU worker copies through host memory whatever wires the accelerators together. Pricing
    its intra-node traffic at NVLink would discount a relational shuffle by ~90x on exactly the
    dense nodes where a relational fleet is most likely to be co-scheduled with inference.
    """
    shares = _dense().locality_shares(32, unit="cpu")
    assert shares.intra_domain == 0.0
    assert shares.intra_node == pytest.approx(0.25 - 1.0 / 32)


def test_unknown_shape_reproduces_the_flat_model():
    """An unreadable fleet charges everything that moves to the network, exactly as before."""
    shares = ClusterShape().locality_shares(32)
    assert shares.local == pytest.approx(1.0 / 32)
    assert shares.cross_rack == pytest.approx(1.0 - 1.0 / 32)
    assert shares.intra_domain == 0.0 and shares.intra_node == 0.0


def test_unlabelled_nodes_are_not_assumed_adjacent():
    """Two nodes that never claimed a rack are not evidence of sharing one.

    The optimistic reading would under-charge every cross-host byte on the common case (an
    unlabelled fleet), which is the direction that produces a shuffle nobody budgeted for.
    """
    shares = _sparse().locality_shares(32, unit="gpu")
    assert shares.intra_rack == 0.0
    assert shares.cross_rack > 0.9


def test_placement_is_even_and_capacity_bounded():
    """Workers spread evenly while capacity allows, which is what the engine asks Ray for.

    Spreading is the least local arrangement, so a share derived from it never over-states how
    much of an exchange stays home — and over-stating locality is the failure that under-charges
    a shuffle nobody then budgets for.
    """
    lopsided = ClusterShape(
        nodes=(
            NodeShape(node_id="big", cpu_cores=64),
            *(NodeShape(node_id=f"s{i}", cpu_cores=1) for i in range(7)),
        )
    )
    assert lopsided.locality_shares(8, unit="cpu").on_node == pytest.approx(1.0 / 8)
    even = ClusterShape(nodes=tuple(NodeShape(node_id=f"n{i}", cpu_cores=8) for i in range(8)))
    assert even.locality_shares(8, unit="cpu").on_node == pytest.approx(1.0 / 8)


def test_a_small_node_stops_taking_workers_once_it_is_full():
    """Capacity bounds the spread: a one-core node holds one worker, not its even share.

    Without the bound a fleet of one big node and one tiny one would report half its exchange
    landing on the tiny node, which is neither where the work goes nor where the bytes do.
    """
    fleet = ClusterShape(
        nodes=(NodeShape(node_id="big", cpu_cores=16), NodeShape(node_id="small", cpu_cores=1))
    )
    shares = fleet.locality_shares(9, unit="cpu")
    # Eight on the big node and one on the small: 8^2 + 1^2 over 9^2.
    assert shares.on_node == pytest.approx(65 / 81)


def test_more_workers_than_capacity_oversubscribes_evenly():
    """An over-subscribed grant is dealt round the fleet, not piled onto its largest node."""
    fleet = ClusterShape(nodes=tuple(NodeShape(node_id=f"n{i}", gpus=8) for i in range(4)))
    assert fleet.locality_shares(64, unit="gpu").on_node == pytest.approx(0.25)


def test_fewer_workers_than_nodes_still_partitions():
    """A fleet larger than the exchange places one worker per node, largest first."""
    shares = _sparse(nodes=32).locality_shares(4, unit="gpu")
    assert _total(shares) == pytest.approx(1.0)
    assert shares.local == pytest.approx(0.25)


def test_derived_counts():
    """The scalar derivations a sizing decision reads."""
    dense = _dense()
    assert dense.total_gpus == 32
    assert dense.max_gpus_per_node == 8
    assert dense.largest_nvlink_domain == 8
    assert dense.homogeneous_gpus is True
    assert dense.device_models == ("NVIDIA_H100",)
    assert dense.racks == 1
    assert dense.binding_gpu_memory_bytes == 80 * (1 << 30)
    assert dense.aggregate_gpu_memory_bytes == 32 * 80 * (1 << 30)


def test_binding_figures_take_the_weakest_node():
    """A mixed fleet's binding VRAM and fabric are the smallest, never the mean.

    A shard sized to the largest device out-of-memories on every other one, and an exchange
    finishes when its slowest participant does.
    """
    mixed = ClusterShape(
        nodes=(
            NodeShape(
                node_id="a",
                cpu_cores=8,
                gpus=8,
                accelerator_type="NVIDIA_H100",
                gpu_memory_bytes=80 << 30,
                fabric_gbps=400.0,
            ),
            NodeShape(
                node_id="b",
                cpu_cores=8,
                gpus=4,
                accelerator_type="NVIDIA_A100",
                gpu_memory_bytes=40 << 30,
                fabric_gbps=100.0,
            ),
        )
    )
    assert mixed.binding_gpu_memory_bytes == 40 << 30
    assert mixed.fabric_gbps == pytest.approx(100.0)
    assert mixed.homogeneous_gpus is False
    assert mixed.min_gpus_per_node == 4


def test_unhealthy_devices_are_subtracted_from_the_schedulable_count_only():
    """A quarantined device leaves `gpus` alone and leaves `healthy_gpus`.

    A fan-out is sized against the devices that exist and scheduled onto the ones that work,
    and collapsing the two loses the ability to say which of them bound the plan.
    """
    node = NodeShape(gpus=8, unhealthy_gpus=3)
    assert node.gpus == 8 and node.healthy_gpus == 5
    assert NodeShape(gpus=2, unhealthy_gpus=9).healthy_gpus == 0


def test_node_local_domain_is_capped_by_the_devices_present():
    """An eight-way part on a two-device host has a domain of two, not eight."""
    assert NodeShape(gpus=2, nvlink_domain=8).local_domain == 2
    assert NodeShape(gpus=8, nvlink_domain=0).local_domain == 8  # unknown: no narrower than node
    assert NodeShape(gpus=8, nvlink_domain=4).domains == 2


def test_per_device_egress_divides_the_node_rate():
    """Eight devices sharing a 400 Gb/s node have 50 Gb/s each, not 400."""
    node = NodeShape(gpus=8, fabric_gbps=400.0)
    assert node.per_device_egress_gbps == pytest.approx(50.0)
    assert NodeShape(gpus=8).per_device_egress_gbps == 0.0  # unmeasured is not "no bandwidth"


def test_hardware_profile_defaults_preserve_the_flat_answers():
    """A profile with no shape reports the pre-existing single-pool figures."""
    flat = HardwareProfile(gpu_count=8)
    assert flat.gpus_per_node == 8  # every device assumed local, as before
    assert flat.gpu_node_count == 1
    assert flat.nvlink_domain == 8
    assert HardwareProfile().gpu_node_count == 0


def test_hardware_profile_reads_the_shape_when_it_has_one():
    """With a shape, density and domain come from the fleet rather than the device total."""
    hardware = HardwareProfile.for_cluster(
        cpu_cores=96, memory_bytes=1 << 40, worker_count=32, gpu_count=32, cluster=_sparse()
    )
    assert hardware.gpu_count == 32
    assert hardware.gpus_per_node == 1
    assert hardware.gpu_node_count == 32
    assert hardware.nvlink_domain == 1


def test_local_profile_describes_one_node():
    """A single-node profile carries a one-node shape, so one arithmetic serves both cases."""
    profile = HardwareProfile.local()
    assert profile.cluster.node_count == 1
    assert profile.cluster.total_cores == profile.cpu_cores
    assert profile.cluster.locality_shares(1).local == pytest.approx(1.0)


def test_summary_is_json_safe_and_marks_unknown():
    """The decision-log record distinguishes an empty fleet from an unread one."""
    assert ClusterShape().summary()["known"] is False
    summary = _dense().summary()
    assert summary["known"] is True and summary["gpus"] == 32
    assert summary["largest_nvlink_domain"] == 8


def test_exchange_width_is_capacity_not_node_count():
    """The width a shuffle actually fans out to, which `worker_count` is not.

    `HardwareProfile.worker_count` is the node count. Pricing tier shares against it reports no
    intra-node traffic at all — every non-local byte charged to the network — on exactly the
    dense fleets the tier model exists for.
    """
    dense = _dense(nodes=4, gpus=8)
    assert dense.node_count == 4
    assert dense.exchange_width("cpu") == 4 * 96
    assert dense.exchange_width("gpu") == 32
    assert ClusterShape().exchange_width() == 1  # unknown fleet: caller keeps its own width


def test_the_intra_node_tier_only_appears_at_the_real_width():
    """The defect the width fix exists for, pinned at both widths."""
    dense = _dense(nodes=4, gpus=8)
    assert dense.locality_shares(dense.node_count, unit="cpu").intra_node == 0.0
    assert dense.locality_shares(dense.exchange_width("cpu"), unit="cpu").intra_node > 0.2
