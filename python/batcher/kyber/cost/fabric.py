"""What a shuffled byte is worth on *this* cluster's fabric.

`CostWeights.net` has been the constant 2.0: a shuffled byte costs twice a local one. That is
a reasonable figure for a general-purpose cloud VM on a 25 Gb/s Ethernet NIC, and it is wrong
by more than an order of magnitude in both directions on the machines Batcher actually runs on.

The physics is a ratio of two bandwidths. A local byte moves at host memory bandwidth; a
shuffled byte moves at whatever the node's NICs sustain. On a GPU node with eight 400 Gb/s
InfiniBand ports the fabric carries 400 GB/s, comfortably *past* what a host achieves against
its own memory, so a shuffled byte costs about what a local one does and a plan that avoids a
shuffle by doing extra local work is the wrong plan. On a 10 Gb/s VM the same byte costs
sixteen times a local one, and the optimizer should contort to avoid it. One constant cannot
be right on both, and the failure is silent: the plan is correct, it is simply not the plan the
cluster wanted.

RDMA is read first and Ethernet second, because a node that has RDMA shuffles over it and the
management NIC beside it is not what a batch will take. Reading Ethernet at all is what makes
this useful on the commodity tier of rented GPU capacity, which has no RDMA and where the
measurement previously returned zero.

**The default is preserved wherever the fabric is unreadable, and wherever the operator set the
weight.** Precedence is explicit override, then measurement, then the library default — the
same order the resilience profiles use. A node with no RDMA NIC, a container without
`/sys/class/infiniband`, and a laptop all keep the 2.0 that every existing plan was ranked
against, so nothing about single-node behavior moves.

**The measurement is the *driver's* fabric, not the workers'.** Optimization happens where the
plan is built, so on a cluster whose head node is provisioned differently from its GPU nodes
this reads the wrong NIC — the same shape of gap `accelerators` documents for device memory.
It errs safe in the usual direction: a head node on ordinary Ethernet keeps a weight at or
above the default and merely under-uses a fast fabric it could not see. An operator whose head
and workers differ the other way sets `net` explicitly, which outranks this entirely.
"""

from __future__ import annotations

import functools

from batcher.config import CostWeights

__all__ = [
    "REFERENCE_LOCAL_GBPS",
    "fabric_adjusted_weights",
    "fabric_net_weight",
    "measured_fabric_gbps",
    "net_weight_summary",
    "reset_fabric_weight",
]

#: Effective host memory bandwidth, in gigabytes per second, that a shuffled byte is priced
#: against. Effective rather than nameplate: a scan does not reach a DRAM controller's peak,
#: and this is the same figure `kyber.gpu.energy` prices the CPU side of a device decision
#: against, so the two halves of the cost model do not disagree about what a host is worth.
REFERENCE_LOCAL_GBPS = 20.0

#: A shuffled byte is never *cheaper* than a local one, whatever the fabric measures. Even on
#: a fabric faster than host memory the byte still costs a serialization, a credit round, and a
#: fragment the local path does not pay, and letting the weight fall below 1.0 would make the
#: enumerator prefer shuffling data it already had — a plan that is worse in a way the cost
#: model would then be unable to see.
_MIN_NET_WEIGHT = 1.0

#: The ceiling on the derived weight. A very slow fabric is genuinely expensive, and past this
#: point the ranking no longer changes — every plan that shuffles has already lost to every
#: plan that does not — while the arithmetic starts to swamp the other axes in a way that makes
#: the total unreadable in a decision log.
_MAX_NET_WEIGHT = 32.0


def measured_fabric_gbps() -> float:
    """This node's off-node bandwidth in Gb/s, from RDMA where it exists and Ethernet where it
    does not.

    RDMA first because a node that has it uses it: the shuffle runs over Arrow Flight on the
    fast fabric, and the Ethernet management interface beside it is not what a batch will
    take. Ethernet is the answer for the far larger share of rented GPU capacity that has no
    RDMA at all, where the alternative was reporting zero and falling back to a constant.

    One caveat the caller should know and this function will not paper over: an Ethernet line
    rate is further from achievable bulk throughput than an RDMA port rate is, because TCP
    pays a stack the fabric does not. No discount is applied for it. Inventing an efficiency
    factor here would be a fabricated figure, and it would be invisible to a caller that had
    already applied one of its own.

    Returns:
        Summed active rate in gigabits per second, `0.0` when neither is readable.
    """
    from batcher._internal.hardware.fabric import ethernet_bandwidth_gbps, fabric_bandwidth_gbps

    return fabric_bandwidth_gbps() or ethernet_bandwidth_gbps()


def fabric_net_weight(
    fabric_gbps: float | None = None,
    local_gbps: float = REFERENCE_LOCAL_GBPS,
) -> float | None:
    """What one shuffled byte costs relative to one local byte, from measured bandwidth.

    Args:
        fabric_gbps: The node's aggregate active fabric rate in *gigabits* per second, or
            `None` to measure it. Only active ports count: a cabled-but-down port carries
            nothing, and counting it would price the shuffle against a link that does not exist.
        local_gbps: Effective host memory bandwidth in gigabytes per second, the denominator
            a local byte is priced at.

    Returns:
        The weight, clamped to `[1.0, 32.0]`, or `None` when the fabric could not be measured.
        `None` is the signal to keep the configured weight — an unreadable NIC is not evidence
        of a slow one.
    """
    measured = fabric_gbps
    if measured is None:
        from batcher.config import active_config

        # A declared rate outranks the probe, because the deployment that needs to declare one
        # is exactly the deployment the probe cannot serve: a pod without the host's `/sys`
        # has a real fabric it simply cannot see, and measuring there returns zero.
        declared = active_config().accelerator.fabric_gbps
        measured = declared if declared > 0.0 else measured_fabric_gbps()
    if measured is None or measured <= 0.0 or local_gbps <= 0.0:
        return None
    fabric_bytes_per_s = measured / 8.0  # Gb/s on the wire, GB/s in the cost model
    return min(_MAX_NET_WEIGHT, max(_MIN_NET_WEIGHT, local_gbps / fabric_bytes_per_s))


@functools.lru_cache(maxsize=1)
def _measured_weight() -> float | None:
    """The derived weight, computed once.

    Memoized because `Cost.total` is called per candidate plan during enumeration, and the
    underlying probe answers a question that cannot change under a running process. The probe
    it reads is itself memoized, so this saves the arithmetic rather than the syscall, which is
    still worth doing on a path that runs thousands of times per query.
    """
    return fabric_net_weight()


def fabric_adjusted_weights(weights: CostWeights) -> CostWeights:
    """`weights` with its `net` axis priced against the measured fabric, where that applies.

    Args:
        weights: The configured weights.

    Returns:
        The same object when the operator has set `net` to anything other than the library
        default, when the fabric is unreadable, or when the measurement lands back on the
        default. Otherwise a copy with the measured weight. Returning the input unchanged in
        the common case matters: it keeps the identity a caller may be relying on, and it makes
        "nothing was derived" visible rather than inferred.
    """
    if weights.net != CostWeights().net:
        return weights  # an explicit operator choice outranks a measurement
    measured = _measured_weight()
    if measured is None or measured == weights.net:
        return weights
    return CostWeights(cpu=weights.cpu, io=weights.io, net=measured)


def net_weight_summary() -> dict:
    """What the fabric measurement did to the cost model, for the decision log.

    Returns:
        The measured fabric rate, the derived weight, and the weight actually in force. A
        reader can tell from this whether a plan was ranked against a measured fabric or
        against the default, which is otherwise invisible in the chosen plan.
    """
    from batcher.config import active_config

    configured = active_config().optimizer.cost_weights
    return {
        "fabric_gbps": measured_fabric_gbps(),
        "derived_net_weight": _measured_weight(),
        "net_weight": fabric_adjusted_weights(configured).net,
    }


def reset_fabric_weight() -> None:
    """Forget the memoized weight, so the next call re-measures the fabric.

    A name currently bound to a test stand-in has no cache to clear and is skipped, matching
    the contract `reset_hardware_probes` holds to.
    """
    clear = getattr(_measured_weight, "cache_clear", None)
    if clear is not None:
        clear()
