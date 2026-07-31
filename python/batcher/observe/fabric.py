"""The node's wires as flat metric rows, for a dashboard that watches a fleet rather than a run.

`bt.accelerators()` answers "what is this node" for a person reading it once. A metrics sink
answers "which of my two hundred nodes is wrong" continuously, and the two wiring conditions
worth alerting on are invisible to every other gauge: devices piled onto one rail, and a node
whose devices cannot copy to each other at all. Both leave the ports `ACTIVE`, the error
counters at zero, and the nameplate bandwidth intact.

Keys are prefixed `fabric.` so they group beside the existing `energy.`, `query.`, and
`shuffle.` families. A figure the node cannot answer for is *absent* rather than zero, because
a gauge that reports zero rails is indistinguishable from a fleet-wide outage in the one view
that is supposed to detect one.

Neutral: `observe` imports no subsystem, so this reads the layer-0 probes directly.
"""

from __future__ import annotations

__all__ = ["fabric_metrics"]


def fabric_metrics() -> dict[str, float]:
    """This node's wiring as metric rows.

    Returns:
        Metric name to value: `fabric.rails`, `fabric.rails_loaded`, `fabric.rail_imbalance`,
        `fabric.rail_gbps`, `fabric.devices`, `fabric.islands`, `fabric.largest_island`,
        `fabric.fabric_pairs`, and `fabric.staged_pairs`. Empty on a host with no readable
        fabric and no accelerators, which is every CPU node and every container without the
        host's `/sys` tree.
    """
    from batcher._internal.hardware.fabric.p2p import peer_summary
    from batcher._internal.hardware.fabric.rails import rail_summary

    out: dict[str, float] = {}
    rails = rail_summary()
    if rails.get("rails"):
        out["fabric.rails"] = float(rails["rails"])
        out["fabric.rails_loaded"] = float(rails["loaded_rails"])
        out["fabric.rail_imbalance"] = float(rails["imbalance"])
        out["fabric.rail_gbps"] = float(rails["total_gbps"])
    peers = peer_summary()
    if peers.get("devices"):
        out["fabric.devices"] = float(peers["devices"])
        out["fabric.islands"] = float(len(peers["islands"]))
        out["fabric.largest_island"] = float(peers["largest_island"])
        out["fabric.fabric_pairs"] = float(peers["fabric_pairs"])
        out["fabric.staged_pairs"] = float(peers["staged_pairs"])
    return out
