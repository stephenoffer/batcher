"""Cluster CPU-utilization sampler for the distributed comparison benchmark.

Pins exactly one lightweight actor per live Ray node (via
``NodeAffinitySchedulingStrategy``) and has each sample ``psutil.cpu_percent`` of
its whole node on a fixed cadence. The driver starts sampling, runs a query, stops,
and reads back per-node busy% — so a benchmark can report not just wall time but how
much of the cluster each engine actually used. This is measurement only; it never
touches the engines under test.
"""

from __future__ import annotations

import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=0)
class _NodeProbe:
    """One per node: samples whole-node CPU% on a cadence into an in-memory buffer."""

    def __init__(self) -> None:
        import psutil

        self._psutil = psutil
        self._samples: list[float] = []
        self._running = False

    def node_id(self) -> str:
        return ray.get_runtime_context().get_node_id()

    def start(self, interval_s: float) -> None:
        import threading

        self._samples = []
        self._running = True

        def _loop() -> None:
            # Prime the first reading (psutil's first call returns 0.0).
            self._psutil.cpu_percent(interval=None)
            while self._running:
                time.sleep(interval_s)
                self._samples.append(self._psutil.cpu_percent(interval=None))

        threading.Thread(target=_loop, daemon=True).start()

    def stop(self) -> list[float]:
        self._running = False
        return self._samples


class ClusterMonitor:
    """Starts one CPU probe per node and aggregates busy% over a sampling window."""

    def __init__(self, interval_s: float = 0.25) -> None:
        self._interval = interval_s
        self._probes: list = []
        node_ids = [n["NodeID"] for n in ray.nodes() if n.get("Alive")]
        for nid in node_ids:
            probe = _NodeProbe.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(nid, soft=False)
            ).remote()
            self._probes.append(probe)
        # Force placement before timing so probe spin-up isn't charged to a query.
        ray.get([p.node_id.remote() for p in self._probes])

    def start(self) -> None:
        ray.get([p.start.remote(self._interval) for p in self._probes])

    def stop(self) -> dict[str, float]:
        """Stop sampling and return aggregate utilization across the cluster.

        ``mean_busy_pct`` averages every per-node sample (the headline number),
        ``peak_busy_pct`` is the highest single node-sample, and ``active_nodes`` is
        how many nodes had any sample exceed 5% busy (work actually reached them).
        """
        per_node = ray.get([p.stop.remote() for p in self._probes])
        flat = [s for node in per_node for s in node]
        node_means = [sum(node) / len(node) for node in per_node if node]
        active = sum(1 for m in node_means if m > 5.0)
        return {
            "mean_busy_pct": sum(flat) / len(flat) if flat else 0.0,
            "peak_busy_pct": max(flat) if flat else 0.0,
            "active_nodes": float(active),
            "total_nodes": float(len(self._probes)),
        }

    def shutdown(self) -> None:
        for p in self._probes:
            ray.kill(p)
        self._probes = []
