"""Per-node CPU, memory and GPU sampling for a cluster benchmark.

`cluster_util.ClusterMonitor` samples CPU and `gpu_util.GpuMonitor` samples GPUs; neither
samples **memory**, which is half of the question "is the cluster actually being used". One
probe per node reads all three on one cadence, so the three series describe the same instants
rather than three differently-skewed windows.

Reserves nothing: `num_cpus=0` and no device, so the measurement cannot displace the workload
it is measuring.
"""

from __future__ import annotations

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=0)
class _Probe:
    def __init__(self) -> None:
        self._cpu: list[float] = []
        self._mem: list[float] = []
        self._gpu: list[float] = []
        self._running = False

    def node_id(self) -> str:
        return ray.get_runtime_context().get_node_id()

    def _gpu_now(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            if not n:
                return None
            vals = []
            for i in range(n):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                vals.append(float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu))
            return sum(vals) / len(vals)
        except Exception:
            return None

    def start(self, interval_s: float) -> None:
        import threading

        import psutil

        self._cpu, self._mem, self._gpu = [], [], []
        self._running = True

        def loop():
            psutil.cpu_percent(interval=None)  # prime; the first call always reads 0.0
            while self._running:
                import time as _t

                _t.sleep(interval_s)
                self._cpu.append(psutil.cpu_percent(interval=None))
                self._mem.append(float(psutil.virtual_memory().percent))
                g = self._gpu_now()
                if g is not None:
                    self._gpu.append(g)

        threading.Thread(target=loop, daemon=True).start()

    def stop(self):
        self._running = False
        return {
            "cpu": self._cpu,
            "mem": self._mem,
            "gpu": self._gpu,
            "cores": __import__("os").cpu_count() or 1,
        }


class ClusterUtil:
    """Start/stop sampling across every live node; report core-weighted aggregates."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self._interval = interval_s
        self._probes = []
        for node in ray.nodes():
            if not node.get("Alive"):
                continue
            nid = node["NodeID"]
            self._probes.append(
                _Probe.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(nid, soft=False)
                ).remote()
            )

    def start(self) -> None:
        ray.get([p.start.remote(self._interval) for p in self._probes])

    def stop(self) -> dict:
        """Aggregate. CPU is **core-weighted**: an unweighted node mean flatters a mixed
        fleet, where an 8-core node at 100% and a 16-core node at 10% average to 55% while
        only a quarter of the cores are busy."""
        out = ray.get([p.stop.remote() for p in self._probes])
        tot_cores = sum(o["cores"] for o in out) or 1
        cpu_w = sum((sum(o["cpu"]) / len(o["cpu"]) if o["cpu"] else 0.0) * o["cores"] for o in out)
        mems = [s for o in out for s in o["mem"]]
        gpus = [s for o in out for s in o["gpu"]]
        gpu_nodes = [o for o in out if o["gpu"]]
        return {
            "cpu_mean_pct": cpu_w / tot_cores,
            "cpu_peak_pct": max((max(o["cpu"]) for o in out if o["cpu"]), default=0.0),
            "mem_mean_pct": sum(mems) / len(mems) if mems else 0.0,
            "mem_peak_pct": max(mems) if mems else 0.0,
            "gpu_mean_pct": sum(gpus) / len(gpus) if gpus else 0.0,
            "gpu_peak_pct": max(gpus) if gpus else 0.0,
            "gpu_nodes": len(gpu_nodes),
            "nodes": len(out),
            "cores": tot_cores,
        }

    def shutdown(self) -> None:
        for p in self._probes:
            ray.kill(p)
        self._probes = []


def fmt(u: dict) -> str:
    return (
        f"cpu {u['cpu_mean_pct']:.0f}%/{u['cpu_peak_pct']:.0f}%peak  "
        f"mem {u['mem_mean_pct']:.0f}%/{u['mem_peak_pct']:.0f}%peak  "
        f"gpu {u['gpu_mean_pct']:.0f}%/{u['gpu_peak_pct']:.0f}%peak  "
        f"({u['nodes']}n {u['cores']}c, {u['gpu_nodes']} gpu nodes)"
    )
