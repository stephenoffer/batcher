"""The autoscaler request lifecycle: scale a cluster up for a query, reclaim after.

`ray.autoscaler.sdk.request_resources` sets a *sticky* floor — the autoscaler holds that
capacity until told otherwise — so an unmanaged request pins a cluster scaled-up forever
after one big query. This module owns the process-wide high-water floor across in-flight
query scopes and drops it back to zero when the last one ends.

Split from `scaling` (which measures and clamps against the live topology) because this is
the *write* side of the same concern: it asks the autoscaler for capacity, where `scaling`
observes what arrived.
"""

from __future__ import annotations

import contextlib
import math
import threading

# --- Autoscaler request lifecycle (scale up for a query, reclaim after) -------------
# `request_resources` sets a *sticky* floor: the autoscaler keeps that many cores until
# told otherwise. Left unmanaged, one big query pins the cluster scaled-up forever. We
# track a process-wide high-water floor across in-flight query scopes and reset it to 0
# the moment the last one ends, so the autoscaler reclaims the now-idle nodes. A
# running query's nodes are *busy* (tasks / persistent-fleet actors), so they are never
# reclaimed mid-query regardless of the floor — the floor only drives scale-*up* and
# keeps a node from being reclaimed in the brief gap before it picks up work.
_autoscale_lock = threading.Lock()
_autoscale_active = 0
_autoscale_floor = 0
_autoscale_gpu_floor = 0
# Custom-accelerator floor (TPU / neuron_cores / HPU / an operator's own resource), max
# per resource name across in-flight scopes — the same high-water rule as the CPU/GPU floors.
_autoscale_resources: tuple[tuple[str, float], ...] = ()


def _apply_autoscale_floor(
    cpus: int, gpus: int = 0, resources: tuple[tuple[str, float], ...] = ()
) -> None:
    with contextlib.suppress(Exception):
        from ray.autoscaler.sdk import request_resources

        if resources:
            # A custom accelerator needs bundles naming *that* resource: a `{"GPU": 1}`
            # bundle asks for GPU nodes, which a TPU cluster has none of, so the query would
            # wait out the autoscale window and then run on whatever was already up.
            request_resources(num_cpus=cpus, bundles=[{n: a} for n, a in resources])
        elif gpus > 0:
            # A GPU floor needs GPU *bundles* — `request_resources(num_cpus=)` alone never
            # triggers GPU-node scale-up, so a GPU query would hang or fall back to CPU
            # nodes it can't run on. One `{"GPU": 1}` bundle per requested GPU asks the
            # autoscaler for that many GPUs; the CPU floor rides alongside for the
            # relational stages. (Whole-GPU bundles — fractional packing is a scheduling
            # concern, not an autoscale-shape one.)
            request_resources(num_cpus=cpus, bundles=[{"GPU": 1}] * gpus)
        else:
            request_resources(num_cpus=cpus)


def request_autoscale(
    target_cpus: int,
    target_gpus: float = 0.0,
    target_resources: tuple[tuple[str, float], ...] = (),
) -> None:
    """Register a query scope wanting `target_cpus` cores (and `target_gpus` GPUs); maintain
    the high-water floor.

    The autoscaler is asked for the max over every in-flight scope, so concurrent
    queries compose and one scope never lowers the floor a live sibling still needs. A
    GPU query (`target_gpus > 0`) also lifts a GPU floor so the autoscaler provisions GPU
    nodes — not just cores, and `target_resources` does the same for an accelerator Ray
    names rather than counts as a GPU (TPU, neuron_cores, HPU). Balanced by exactly one
    `release_autoscale` at the scope's teardown.
    """
    global _autoscale_active, _autoscale_floor, _autoscale_gpu_floor, _autoscale_resources
    with _autoscale_lock:
        _autoscale_active += 1
        _autoscale_floor = max(_autoscale_floor, target_cpus)
        _autoscale_gpu_floor = max(_autoscale_gpu_floor, math.ceil(target_gpus))
        merged = dict(_autoscale_resources)
        for name, amount in target_resources:
            merged[name] = max(merged.get(name, 0.0), amount)
        _autoscale_resources = tuple(sorted(merged.items()))
        _apply_autoscale_floor(_autoscale_floor, _autoscale_gpu_floor, _autoscale_resources)


def release_autoscale() -> None:
    """End one query scope; when the last one ends, drop the autoscaler floor (CPU and GPU)
    to 0 so it can reclaim the idle nodes the query scaled up (instead of pinning them
    forever)."""
    global _autoscale_active, _autoscale_floor, _autoscale_gpu_floor, _autoscale_resources
    with _autoscale_lock:
        _autoscale_active -= 1
        if _autoscale_active <= 0:
            _autoscale_active = 0
            _autoscale_floor = 0
            _autoscale_gpu_floor = 0
            _autoscale_resources = ()
            _apply_autoscale_floor(0, 0, ())
