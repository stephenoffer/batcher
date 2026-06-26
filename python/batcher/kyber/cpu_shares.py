"""Adaptive per-task CPU share — turn measured CPU utilization into a `num_cpus`.

Kyber annotates every operator with the CPU share one distributed task running it
should request (`ResourceBounds.c_cpu_shares` → Carbonite's `SchedulingEnvelope`).
The cold-start value is a static per-kind prior (a breaker wants a full core; a
CPU-light streaming op wants a fraction). Once a workload has run, Core has recorded
each operator's measured **CPU utilization** (CPU-time / (wall x threads), in
[0, 1]) into the hub's `op_stats`; this module reads that history and lets the
measured utilization override the prior — the CPU twin of the GPU-utilization loop
in `ml/gpu.py`. A CPU-bound family nears 1.0 (ask for a whole core); an IO-bound one
stays low (pack several per core). Pure: reads the hub, returns shares; decides
nothing. Best-effort — any failure falls back to the static prior.

The signal is deliberately coarse and known to be imperfect; it sizes a *scheduling
hint* (`num_cpus`), never a result, so error costs throughput, not correctness:

* It is a **mean** over the operator's run. A bursty op (e.g. IO-decode that spikes)
  has a low mean but high peaks; packing by the mean can cause CFS contention and
  hurt tail latency. The `cpu_share_min` floor bounds, but does not eliminate, this.
* Utilization is measured **per core across the single-node run** (`cpu_ns / (wall x
  threads)`). An op that parallelizes poorly (or is tiny) reads artificially low, and
  the thread count is the configured/host core count, which can differ from rayon's
  actual pool (`RAYON_NUM_THREADS`, cgroup limits). Single-node per-core utilization
  also only *approximates* a distributed task's per-partition utilization (an
  aggregate's sequential combine tail biases it low).
* Keying is **per operator family** (`scan`, `filter`, `aggregate`, ...), so a
  regex-heavy filter is averaged with cheap filters; it learns a family prior, not a
  per-query value. Sub-millisecond ops fall below CPU-timer granularity and read 0
  (unmeasured → static prior).
* The share is clamped to `[cpu_share_min, cpus_per_task]`; since utilization is in
  [0, 1], a configured `cpus_per_task > 1` acts as a ceiling the measurement caps at
  ~1 core (it cannot express a partition needing more than one fully-busy core).
"""

from __future__ import annotations

import weakref

from batcher.config import Config, active_config
from batcher.metadata import MetadataHub

__all__ = ["class_ir_tag", "load_cpu_utilization", "recommend_num_cpus"]

# Python `LogicalPlan` class name → native `ExecMetrics` `kind` tag, so a learned
# utilization (keyed by the IR tag the engine reports) can be looked up for a node
# Kyber annotates (keyed by class name). Mirrors the tag set in `calibration._KIND_COEFF`;
# `Join` lowers to the `hash_join` operator. An unmapped node has no learned share and
# keeps its static prior.
_CLASS_TO_TAG: dict[str, str] = {
    "Scan": "scan",
    "Filter": "filter",
    "Project": "project",
    "Limit": "limit",
    "Sample": "sample",
    "RowId": "row_id",
    "Union": "union",
    "Unnest": "unnest",
    "Unpivot": "unpivot",
    "Aggregate": "aggregate",
    "Sort": "sort",
    "Distinct": "distinct",
    "Join": "hash_join",
    "AsofJoin": "asof_join",
    "Window": "window",
}

# Per-hub memo of the learned utilization map, keyed weakly by the hub (a dropped hub
# evicts its entry). Value is `(hub.version, min_samples, util_by_tag)`: reused while
# the hub has absorbed no new feedback, so planning doesn't re-scan the whole op_stats
# history on every optimize (the calibration loop caches the same way).
_CPU_CACHE: weakref.WeakKeyDictionary[MetadataHub, tuple[int, int, dict[str, float]]] = (
    weakref.WeakKeyDictionary()
)


def class_ir_tag(class_name: str) -> str | None:
    """The native metrics `kind` tag for a `LogicalPlan` class name, if measured."""
    return _CLASS_TO_TAG.get(class_name)


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def load_cpu_utilization(hub: MetadataHub | None, config: Config | None = None) -> dict[str, float]:
    """Median measured CPU utilization per operator `kind` tag from the hub.

    Returns an empty map when there is no hub, no measured CPU data, or no family
    with enough samples — so a cold metadata store leaves the static priors in place.
    Only positive utilizations count (0.0 is the "unmeasured" sentinel from an engine
    that reports no CPU time).
    """
    if hub is None:
        return {}
    cfg = config or active_config()
    min_samples = max(1, cfg.optimizer.cost_calibration_min_samples)
    version = hub.version
    cached = _CPU_CACHE.get(hub)
    if cached is not None and cached[0] == version and cached[1] == min_samples:
        return cached[2]
    try:
        out: dict[str, float] = {}
        for tag, rows in hub.op_stats_by_kind().items():
            utils = [u for r in rows if (u := float(r.get("cpu_utilization", 0.0) or 0.0)) > 0.0]
            if len(utils) >= min_samples:
                out[tag] = _median(utils)
    except Exception:  # pragma: no cover - planning must never break on bad feedback
        out = {}
    _CPU_CACHE[hub] = (version, min_samples, out)
    return out


def recommend_num_cpus(util: float | None, base: float, config: Config | None = None) -> float:
    """Adapt a per-task CPU share from measured utilization, falling back to `base`.

    With no measurement (`util` is None or non-positive) the static prior `base`
    stands. Otherwise the share tracks utilization directly, clamped to
    ``[cpu_share_min, cpus_per_task]`` so an IO-bound op never asks for an
    unschedulable sliver and a CPU-bound one never exceeds a whole core.
    """
    if util is None or util <= 0.0:
        return base
    cfg = config or active_config()
    return max(cfg.execution.cpu_share_min, min(cfg.execution.cpus_per_task, util))
