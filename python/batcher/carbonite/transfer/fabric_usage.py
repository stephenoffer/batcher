"""What the node's RDMA fabric carried while a shuffle was running.

A cross-node stage priced against a 400 Gb/s fabric and running at a tenth of that is almost
always not on the fabric at all — the address the workers advertised resolved to the
management NIC — and nothing else a shuffle measures can tell the two apart. Fetch counts,
locality ratios and credit windows all read the same whether the bytes crossed InfiniBand or
a 25 Gb/s Ethernet port, because none of them knows which wire it was.

The port counters do. A baseline when the session opens and a delta when its statistics are
read is the whole measurement, and it costs a few small file reads on an RDMA node and one
empty directory listing everywhere else.

**Node-wide, not the session's share.** The counters belong to the port, so anything else
running on the node is in the figure. That asymmetry is the useful one: a *low* reading is
decisive — the fabric moved nothing, so this shuffle certainly did not use it — while a high
one is merely consistent with the shuffle having used it.
"""

from __future__ import annotations

import time

from batcher._internal.logging import note_suppressed

__all__ = ["fabric_baseline", "fabric_usage"]


def fabric_baseline() -> tuple[tuple, float]:
    """`(counters, started_at)` for the node's fabric, or `((), 0.0)` when there is none.

    Read once per session rather than per fetch: the counters are monotonic, so a baseline
    plus one later reading is the entire measurement.

    Returns:
        The sample and the monotonic time it was taken at. An empty sample means this node
        has no readable fabric, and every later call short-circuits on it.
    """
    from batcher._internal.hardware.fabric import port_counters

    try:
        sample = port_counters()
    except Exception as exc:  # pragma: no cover - a diagnostic must never fail a shuffle
        note_suppressed("carbonite", "read the fabric counters", exc)
        return ((), 0.0)
    return (sample, time.monotonic()) if sample else ((), 0.0)


def fabric_usage(baseline: tuple, started_at: float) -> dict[str, float]:
    """Observed against capable fabric throughput since `baseline` was taken.

    Args:
        baseline: The sample from `fabric_baseline`.
        started_at: The monotonic time that sample was taken at.

    Returns:
        Observed and capable rates in Gb/s and their ratio, or an empty mapping on a node with
        no readable fabric — which is most nodes, and where this must cost nothing and say
        nothing. Any failure in the probe also reports nothing rather than raising: this is a
        diagnostic, and a diagnostic that can fail a shuffle is worse than no diagnostic.
    """
    if not baseline:
        return {}
    from batcher._internal.hardware.fabric import (
        fabric_bandwidth_gbps,
        port_counters,
        throughput_delta,
    )

    try:
        elapsed = time.monotonic() - started_at
        observed = sum(throughput_delta(baseline, port_counters(), elapsed).values())
        capable = fabric_bandwidth_gbps()
    except Exception as exc:  # pragma: no cover - a diagnostic must never fail a shuffle
        note_suppressed("carbonite", "sample the fabric counters", exc)
        return {}
    out = {"fabric_gbps_observed": round(observed, 2), "fabric_gbps_capable": capable}
    if capable > 0:
        out["fabric_utilization"] = round(min(1.0, observed / capable), 4)
    return out
