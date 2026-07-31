"""What the optimizer made of the fleet — the half of a report the hardware does not say.

The rest of the accelerator report describes what the hardware *is*. This section describes
what Kyber concluded from it, which an operator cannot otherwise recover: the same query
produces different plans on a dense multi-GPU fleet and a wide single-GPU one, and a reader
comparing two clusters' timings has no way to tell which they are looking at.

Its own module rather than another arm of `report`, which is at its size limit and is the
section every future addition here would otherwise be appended to.
"""

from __future__ import annotations

__all__ = ["add_planning"]


def add_planning(report: dict) -> None:
    """Add the fleet shape Kyber plans against, and what it did to the cost of a shuffle.

    Omitted entirely when the topology is unreadable — a single-node run, or a driver with no
    cluster — because every field would then restate the flat defaults and say nothing.

    The width priced against is the fleet's schedulable capacity, not `worker_count` (which is
    the node count) — the same figure the cost model uses, so the report cannot disagree with
    the ranking it describes. Both units are reported because they are genuinely different
    answers on one fleet: a relational shuffle is placed against cores and a device fan-out
    against devices.

    Args:
        report: The report being assembled, mutated in place.

    Returns:
        None.
    """
    from batcher.api.orchestration.sizing import distributed_hardware

    hardware = distributed_hardware()
    if hardware is None or not hardware.cluster.known:
        return
    from batcher.kyber.cost.locality import locality_summary

    relational = locality_summary(hardware, hardware.cluster.exchange_width("cpu"), unit="cpu")
    planning: dict = {
        "cluster": hardware.cluster.summary(),
        "shuffle_cost_factor": round(relational["factor"], 3),
        "shuffle_basis": relational["basis"],
    }
    if hardware.gpu_count > 0:
        device = locality_summary(hardware, hardware.cluster.exchange_width("gpu"), unit="gpu")
        planning["device_exchange_cost_factor"] = round(device["factor"], 3)
    report["planning"] = planning
