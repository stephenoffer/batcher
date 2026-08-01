"""How many devices a working set that *fits* one device should still be spread across.

The memory question — does this fit on a GPU — and the throughput question — how many GPUs
should run it — are different questions, and the routing answered only the first. A plan whose
working set fit one device was dispatched to one device, whatever the fleet had: on a
four-device cluster a 32M-row group-by ran on one T4 while three sat idle, and the sharded
fan-out built for exactly that work was never reached, because it is only entered when the data
is too *large* for a single device.

Fitting is a floor, not a target. The stage is mergeable, so a shard's partial and the
concatenation of its shards' partials are the same value — splitting is a scheduling decision
with no semantic content, and the only reason not to split is that a piece can be too small to
be worth its own dispatch.

That is the bound this module applies, and it is the same one the shard planner already uses:
`gpu_min_shard_bytes`, below which the Ray dispatch delivering a shard costs more than the
shard's own compute. A relation worth two of those is worth two devices; one worth less than
two stays where it is. `dist.gpu.shards.plan_shard_count` then cuts the shards themselves under
the same floor, so the two never disagree about what a worthwhile piece is.
"""

from __future__ import annotations

__all__ = ["devices_worth_using"]


def devices_worth_using(ws_gb: float, gpu_count: int) -> int:
    """How many of the cluster's devices to spread a fitting working set across.

    Args:
        ws_gb: The estimated working set, in gigabytes. `0` or less means unknown, which keeps
            the single-device answer — spreading on a guess is how a small query pays for four
            dispatches to do one dispatch's work.
        gpu_count: The live cluster's device count.

    Returns:
        The device count to use, at least 1 and never more than the cluster has.

    Examples:
        .. doctest::

            >>> from batcher.kyber.gpu.spread import devices_worth_using
            >>> devices_worth_using(0.8, 4)  # six shard-floors' worth, four devices
            4
            >>> devices_worth_using(0.1, 4)  # smaller than one floor: one device
            1
    """
    from batcher.config import active_config

    devices = int(gpu_count)
    if ws_gb <= 0 or devices <= 1:
        return 1
    floor_gb = max(1, int(active_config().distributed.gpu_min_shard_bytes)) / 1e9
    # Floor division, not ceiling: a relation must be worth *two whole* pieces before it is cut
    # in two, or a query barely over one floor is split into a full shard and a sliver.
    worth = int(ws_gb // floor_gb)
    return max(1, min(devices, worth))
