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

__all__ = ["fabric_baseline", "fabric_usage", "rail_usage"]


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


def rail_usage(baseline: tuple, started_at: float) -> dict:
    """The same measurement per rail, which is where a node-wide ratio hides the finding.

    A node whose eight devices all landed on one NIC saturates that port and leaves seven
    idle. Summed, that reads as 12% fabric utilization — indistinguishable from a shuffle
    that is simply slow, and the fix suggested by the summed figure (more concurrency) makes
    it worse. Per rail it is unmistakable: one port at capacity and seven at nothing.

    Args:
        baseline: The sample from `fabric_baseline`.
        started_at: The monotonic time that sample was taken at.

    Returns:
        `rails` (port key to observed Gb/s), `busiest_gbps`, `idle_rails` (ports that carried
        nothing), and `spread` (busiest over mean, `1.0` when every rail carried the same and
        higher as the traffic concentrates). Empty on a node with fewer than two rails, where
        there is no spread to report, and on one with no readable fabric.
    """
    if not baseline:
        return {}
    from batcher._internal.hardware.fabric import port_counters, throughput_delta

    try:
        elapsed = time.monotonic() - started_at
        per_port = throughput_delta(baseline, port_counters(), elapsed)
    except Exception as exc:  # pragma: no cover - a diagnostic must never fail a shuffle
        note_suppressed("carbonite", "sample the fabric counters per rail", exc)
        return {}
    # The rail set comes from the *baseline*, which lists every active port, not from the
    # delta, which omits a port that moved nothing. Reading the rails off the delta inverts
    # the whole measurement: on the node this exists to catch — one rail carrying everything
    # and seven idle — the delta holds a single entry, and the function would report a
    # perfectly balanced one-rail node instead of a seven-rail imbalance.
    keys = sorted({port.key for port in baseline})
    if len(keys) < 2:
        return {}
    rates = [per_port.get(key, 0.0) for key in keys]
    per_port = dict(zip(keys, rates, strict=True))
    busiest = max(rates)
    mean = sum(rates) / len(rates)
    return {
        "rails": {key: round(rate, 2) for key, rate in sorted(per_port.items())},
        "busiest_gbps": round(busiest, 2),
        "idle_rails": sum(1 for rate in rates if rate <= 0.0),
        # Busiest over mean rather than a variance: it reads directly as "this rail carried N
        # times its share", which is the sentence an operator acts on.
        "spread": round(busiest / mean, 2) if mean > 0 else 0.0,
    }
