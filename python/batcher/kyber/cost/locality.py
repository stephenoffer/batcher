"""What a shuffled byte costs *depending on where it lands* — the interconnect tiers.

`shuffle` computes how many bytes an operator moves; `fabric` prices one such byte against the
node's NIC. Between them sits an assumption neither states: that every byte of an exchange
crosses the same wire. On the fleets Batcher targets that assumption is wrong by more than an
order of magnitude, and wrong in a way that changes which plan wins.

A hash exchange across thirty-two workers sends each row to one of thirty-two buckets. On four
eight-device hosts, a quarter of those buckets are on the producer's own host and an eighth are
inside its own NVLink domain, where a byte moves at ~450 GB/s against the ~5 GB/s of a 40 Gb/s
NIC — a factor of ninety. On thirty-two one-device hosts, essentially none of it stays home. The
two fleets report an identical `gpu_count` and an identical `worker_count`, and the flat model
prices their shuffles identically. So Kyber ranked a repartition against a broadcast, and a
co-partitioned aggregate against a re-partitioned one, using a network cost that was up to four
times too high on exactly the dense multi-GPU nodes the fan-out exists to use.

## What this module does

`locality_factor` collapses the fleet's tier *shares* (from `ClusterShape`, which knows the
structure) and the tiers' *prices* (from device and fabric bandwidth, which this module knows)
into a single multiplier on the flat byte count. The `net` axis therefore keeps carrying bytes —
**cross-rack-equivalent** bytes — and `Cost.total` keeps pricing them with the one `net` weight
`fabric` derives. Nothing downstream of the axis changes.

## Why every unknown resolves to 1.0

A tier whose bandwidth cannot be read is priced at the cross-rack rate, so it is charged exactly
what the flat model charged. An unreadable fleet, a single-node run, an unlabelled device: all
of them produce a factor of 1.0 and a ranking bit-for-bit identical to the one before this
existed. The discount is only ever applied against a measured or specified rate, and it can only
ever make an exchange *cheaper* — which is the safe direction, because the failure mode of
over-charging locality is a pessimistic plan and the failure mode of under-charging it is a
shuffle nobody budgeted for.
"""

from __future__ import annotations

from dataclasses import dataclass

from batcher.kyber.cost.fabric import REFERENCE_LOCAL_GBPS, measured_fabric_gbps

__all__ = [
    "TierPrices",
    "locality_factor",
    "locality_summary",
    "plan_locality_factor",
    "tier_prices",
]

#: Ratio floor for any tier's price against a cross-rack byte. A local exchange is cheaper than
#: a network one; it is not *free*. Even an on-package copy pays a serialization into Arrow, a
#: credit round, and a fragment the flat model never charged separately, and a factor below this
#: would let the enumerator believe an intra-node shuffle costs nothing at all — which is how a
#: plan comes to shuffle three times where it could shuffle once.
_MIN_TIER_RATIO = 0.02


@dataclass(frozen=True, slots=True)
class TierPrices:
    """What one byte costs on each interconnect tier, relative to a cross-rack byte.

    Ratios, not absolute costs, so they compose with whatever `net` weight `fabric` derives:
    the weight says what a cross-rack byte is worth against a local one, and these say what a
    byte on each cheaper tier is worth against that. Every field is in `(0, 1]` — no tier is
    more expensive than the network — and every unknown rate resolves to `1.0`, which charges
    the flat rate the model charged before.

    Attributes:
        intra_domain: A byte on the coherent device fabric (NVLink).
        intra_node: A byte crossing the host bus but not the network (PCIe, host memory).
        intra_rack: A byte on the rack's own fabric.
        cross_rack: Always `1.0`; the unit the others are quoted in.
        basis: One line naming where the rates came from, for the decision log.
    """

    intra_domain: float = 1.0
    intra_node: float = 1.0
    intra_rack: float = 1.0
    cross_rack: float = 1.0
    basis: str = "unmeasured: every tier charged at the network rate"

    @property
    def measured(self) -> bool:
        """Whether any tier was actually discounted below the network rate."""
        return min(self.intra_domain, self.intra_node, self.intra_rack) < 1.0


def _ratio(fabric_gbps: float, tier_gbps: float) -> float:
    """A tier's price against a cross-rack byte, from the two links' rates.

    A byte's cost is inversely proportional to the rate that carries it, so the ratio is
    `fabric / tier`. `1.0` whenever either rate is unknown or the tier is no faster than the
    network — never a discount taken on a number nobody has.

    Both figures are nameplate rates, and no efficiency factor is applied to either. One would
    not cancel: the cross-rack tier is fixed at `1.0` by definition, so a single constant here
    would silently re-scale every other tier against the network rather than leaving the
    comparison alone. A real exchange reaches a different fraction of its line rate on each
    tier — RDMA far more of its than TCP does — so the honest form is a *per-tier* measurement,
    and until one exists nothing is applied.
    """
    if fabric_gbps <= 0.0 or tier_gbps <= 0.0:
        return 1.0
    return min(1.0, max(_MIN_TIER_RATIO, fabric_gbps / tier_gbps))


def tier_prices(
    accelerator_type: str = "",
    *,
    fabric_gbps: float | None = None,
    host_gbps: float = REFERENCE_LOCAL_GBPS,
) -> TierPrices:
    """Price each interconnect tier against a cross-rack byte, from measured link rates.

    Args:
        accelerator_type: The fleet's device model, `""` on a mixed or unlabelled fleet. Names
            the coherent fabric's rate and the host link's; without it the device tier is
            charged at the host-memory rate, which is the honest floor for a copy that has to
            be staged through the host anyway.
        fabric_gbps: The node's off-node rate in *gigabits* per second, or `None` to measure it.
            `0.0` or an unmeasurable fabric leaves every tier at `1.0`.
        host_gbps: Effective host memory bandwidth in *gigabytes* per second — what a byte
            moved between two workers on one host actually achieves, since it is a copy through
            DRAM whatever bus nominally carries it.

    Returns:
        The tier prices. Every one is `1.0` when the fabric rate is unknown, which reproduces
        the flat model exactly.
    """
    fabric = measured_fabric_gbps() if fabric_gbps is None else fabric_gbps
    if fabric <= 0.0:
        return TierPrices()

    # Everything is converted to gigabits per second so the ratios are dimensionless. The
    # fabric figure is already in gigabits (a NIC's unit); the device and host figures are in
    # gigabytes (a memory bus's unit), which is exactly the mismatch that makes an unconverted
    # comparison look plausible and be wrong by eight.
    host_gbits = max(0.0, host_gbps) * 8.0
    node_ratio = _ratio(fabric, host_gbits)

    domain_gbits = 0.0
    if accelerator_type:
        from batcher._internal.device_specs import device_nvlink_gbps

        domain_gbits = max(0.0, device_nvlink_gbps(accelerator_type)) * 8.0
    # A device fleet with no published fabric rate falls back to the host tier rather than to
    # the network tier: two devices on one host exchange through host memory at worst, never
    # over the NIC, so charging them the network rate over-states the cost of the one placement
    # the fan-out is trying to prefer.
    domain_ratio = _ratio(fabric, domain_gbits) if domain_gbits > 0.0 else node_ratio

    # The rack tier is charged at the network rate. A rack's own fabric is genuinely faster
    # than a spine crossing on most builds, and by how much is a property of a specific
    # deployment's oversubscription ratio that this process cannot read. Discounting it on a
    # guess would put a fabricated figure under a join order; leaving it at 1.0 costs only the
    # locality Batcher cannot prove.
    return TierPrices(
        intra_domain=min(domain_ratio, node_ratio),
        intra_node=node_ratio,
        intra_rack=1.0,
        cross_rack=1.0,
        basis=(
            f"fabric {fabric:.0f}Gb/s vs host {host_gbits:.0f}Gb/s"
            + (f" vs {accelerator_type} fabric {domain_gbits:.0f}Gb/s" if domain_gbits else "")
        ),
    )


def locality_factor(
    hardware,
    workers: int,
    *,
    unit: str = "cpu",
    prices: TierPrices | None = None,
) -> float:
    """The multiplier turning flat shuffle bytes into cross-rack-equivalent bytes.

    Args:
        hardware: The `HardwareProfile` being planned against; its `cluster` supplies the tier
            shares. A profile with no cluster shape yields `1.0`.
        workers: Workers the exchange runs across.
        unit: `"cpu"` for a relational shuffle, `"gpu"` for a device fan-out. The two place
            workers differently — cores versus devices — and on a fleet whose CPU-only nodes
            outnumber its accelerator nodes the resulting shares are nothing alike.
        prices: Tier prices, or `None` to derive them from the fleet's device and fabric rates.

    Returns:
        A factor in `(0, 1]`. Exactly `1.0` for a single worker, an unknown fleet shape, or an
        unmeasurable fabric — in each case the flat byte count stands unchanged.
    """
    if hardware is None or workers <= 1:
        return 1.0
    cluster = getattr(hardware, "cluster", None)
    if cluster is None or not cluster.known:
        return 1.0
    tiers = prices or tier_prices(
        getattr(hardware, "accelerator_type", "") or "",
        fabric_gbps=cluster.fabric_gbps or None,
    )
    if not tiers.measured:
        return 1.0
    shares = cluster.locality_shares(workers, unit=unit)
    # The `local` share is charged nothing: it never leaves the worker that produced it, which
    # is the one case where "free" is not an approximation. Renormalized against the flat
    # model's own local discount so the factor is a pure re-pricing of the bytes that DO move
    # and not a second discount on the ones that do not — the flat `shuffle_bytes` has already
    # removed the local share, and taking it out twice would under-count every exchange.
    moving = 1.0 - shares.local
    if moving <= 0.0:
        return 1.0
    weighted = shares.weighted(
        local=0.0,
        intra_domain=tiers.intra_domain,
        intra_node=tiers.intra_node,
        intra_rack=tiers.intra_rack,
        cross_rack=tiers.cross_rack,
    )
    return min(1.0, max(_MIN_TIER_RATIO, weighted / moving))


def locality_summary(hardware, workers: int, *, unit: str = "cpu") -> dict:
    """What the locality model did to this plan's `net` axis, for the decision log.

    Args:
        hardware: The `HardwareProfile` being planned against.
        workers: Workers the exchange runs across.
        unit: `"cpu"` or `"gpu"`, as `locality_factor`.

    Returns:
        The tier shares, the tier prices, and the resulting factor. A reader can tell from this
        whether a plan was ranked against a measured topology or against the flat default,
        which is otherwise invisible in the chosen plan.
    """
    cluster = getattr(hardware, "cluster", None)
    if cluster is None or not cluster.known or workers <= 1:
        return {"factor": 1.0, "basis": "no cluster shape: flat network pricing"}
    tiers = tier_prices(
        getattr(hardware, "accelerator_type", "") or "",
        fabric_gbps=cluster.fabric_gbps or None,
    )
    shares = cluster.locality_shares(workers, unit=unit)
    return {
        "factor": locality_factor(hardware, workers, unit=unit, prices=tiers),
        "basis": tiers.basis,
        "unit": unit,
        "workers": workers,
        "shares": {
            "local": shares.local,
            "intra_domain": shares.intra_domain,
            "intra_node": shares.intra_node,
            "intra_rack": shares.intra_rack,
            "cross_rack": shares.cross_rack,
        },
        "prices": {
            "intra_domain": tiers.intra_domain,
            "intra_node": tiers.intra_node,
            "intra_rack": tiers.intra_rack,
        },
    }


def plan_locality_factor(hardware, workers: int) -> float:
    """The fleet's interconnect-tier discount on shuffled bytes, `1.0` when it cannot be read.

    Priced against the fleet's **schedulable capacity**, not against `worker_count`. Those are
    different numbers: `worker_count` is the node count, and the tier shares are far more
    sensitive to the difference than the volume term is. At one worker per node an exchange has
    no intra-node tier — every non-local byte is charged to the network — so a four-node,
    ninety-six-core fleet reported no on-host traffic at all and took no discount, which is the
    exact fleet the tier model exists to serve. Its real fan-out is closer to its core count,
    and that is what a shuffle across it is actually split over.

    Falls back to `workers` on an unknown shape, and over-states the width rather than
    under-stating it when the true fan-out is smaller — the safe direction, since a wider
    exchange keeps less of itself local and is therefore charged more.

    Isolated and defensive because it is the one term in the cost model derived from a *live*
    topology read: a cluster that went away mid-plan, a profile from an older release without a
    shape, or a device model nothing recognizes must all leave the ranking exactly where it was
    rather than fail an optimize.

    Args:
        hardware: The `HardwareProfile` being planned against, or `None`.
        workers: The plan's worker count, used as a floor on the exchange width.

    Returns:
        The multiplier for the `net` axis, `1.0` when nothing can be derived.
    """
    if hardware is None or workers <= 1:
        return 1.0
    try:
        cluster = getattr(hardware, "cluster", None)
        width = workers
        if cluster is not None and cluster.known:
            width = max(workers, cluster.exchange_width("cpu"))
        return locality_factor(hardware, width)
    except Exception as exc:  # pragma: no cover - cost must never break a plan
        from batcher._internal.logging import note_suppressed

        note_suppressed("kyber", "price the fleet's interconnect tiers", exc)
        return 1.0
