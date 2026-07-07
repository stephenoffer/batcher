"""Cluster GPU-utilization sampler for the distributed GPU-inference benchmark.

Pins one lightweight actor per live Ray node (via ``NodeAffinitySchedulingStrategy``)
and has each sample every GPU's utilization on that node through NVML on a fixed
cadence. The driver starts sampling, runs a GPU pipeline, stops, and reads back the
mean/peak GPU busy% and how many GPUs did real work — so a benchmark can report not
just images/sec but whether the cluster's GPUs were actually saturated (the number
that separates "the model is fast" from "the data pipeline kept the GPU fed").

Measurement only; ``num_gpus=0`` so a probe never reserves a device under test — NVML
reads a node's counters without owning the GPU. It is the GPU twin of
``cluster_util.ClusterMonitor``.
"""

from __future__ import annotations

import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=0)
class _GpuProbe:
    """One per node: samples every local GPU's util% on a cadence via NVML."""

    def __init__(self) -> None:
        self._samples: list[float] = []  # per-tick mean-over-local-GPUs busy%
        self._running = False
        self._ngpu = 0

    def node_id(self) -> str:
        return ray.get_runtime_context().get_node_id()

    def _read(self):
        """Mean and max util% across this node's GPUs now, or ``None`` without NVML."""
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                n = pynvml.nvmlDeviceGetCount()
                if n == 0:
                    return None
                self._ngpu = n
                utils = [
                    float(
                        pynvml.nvmlDeviceGetUtilizationRates(
                            pynvml.nvmlDeviceGetHandleByIndex(i)
                        ).gpu
                    )
                    for i in range(n)
                ]
                return sum(utils) / n, max(utils)
            finally:
                pynvml.nvmlShutdown()
        except Exception:
            return None

    def start(self, interval_s: float) -> None:
        import threading

        self._samples = []
        self._peaks: list[float] = []
        self._running = True

        def _loop() -> None:
            while self._running:
                r = self._read()
                if r is not None:
                    self._samples.append(r[0])
                    self._peaks.append(r[1])
                time.sleep(interval_s)

        threading.Thread(target=_loop, daemon=True).start()

    def stop(self) -> tuple[list[float], list[float], int]:
        self._running = False
        return self._samples, getattr(self, "_peaks", []), self._ngpu


class GpuMonitor:
    """Starts one GPU probe per node and aggregates GPU busy% over a sampling window."""

    def __init__(self, interval_s: float = 0.25) -> None:
        self._interval = interval_s
        self._probes: list = []
        node_ids = [n["NodeID"] for n in ray.nodes() if n.get("Alive")]
        for nid in node_ids:
            probe = _GpuProbe.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(nid, soft=False)
            ).remote()
            self._probes.append(probe)
        ray.get([p.node_id.remote() for p in self._probes])  # force placement pre-timing

    def start(self) -> None:
        ray.get([p.start.remote(self._interval) for p in self._probes])

    def stop(self) -> dict[str, float]:
        """Stop sampling and return aggregate GPU utilization across the cluster.

        ``mean_gpu_pct`` averages every per-node mean-GPU sample (the headline: how busy
        the average GPU was), ``peak_gpu_pct`` is the highest single-GPU reading, and
        ``active_gpu_nodes`` counts nodes whose GPUs exceeded 5% (work reached the GPU).
        """
        per_node = ray.get([p.stop.remote() for p in self._probes])
        means = [s for samples, _peaks, _n in per_node for s in samples]
        peaks = [s for _samples, pk, _n in per_node for s in pk]
        node_active = sum(
            1 for samples, _pk, _n in per_node if samples and (sum(samples) / len(samples)) > 5.0
        )
        gpu_nodes = sum(1 for _s, _pk, n in per_node if n > 0)
        return {
            "mean_gpu_pct": sum(means) / len(means) if means else 0.0,
            "peak_gpu_pct": max(peaks) if peaks else 0.0,
            "active_gpu_nodes": float(node_active),
            "total_gpu_nodes": float(gpu_nodes),
        }

    def shutdown(self) -> None:
        for p in self._probes:
            ray.kill(p)
        self._probes = []
