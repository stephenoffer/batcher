"""Ray lifecycle, scheduling envelope, autoscaling, and fault policies for the
distributed executor.

Façade over four responsibility modules: `lifecycle` (Ray init + remote-task
wrapping + single-node fallback), `scheduling` (the metadata-driven envelope and
placement groups), `scaling` (cluster topology + autoscaler request lifecycle), and
`policies` (config-driven fault-tolerance / recovery / skew). Import names from here;
the split is an implementation detail.
"""

from __future__ import annotations

from .lifecycle import (
    _ensure_ray,
    _rmtree,
    _single_node,
    _wrap_tasks,
    engine_config_json,
    resolve_transport,
)
from .policies import (
    actor_fault_options,
    draining_workers,
    fault_options,
    gather_map_results,
    recovery_policy,
    runtime_bloom_join,
    skew_join_salt,
    speculation_policy,
)
from .scaling import (
    clamp_workers,
    cluster_topology,
    release_autoscale,
    request_autoscale,
)
from .scheduling import (
    create_worker_placement,
    current_envelope,
    placement_actor_options,
    release_placement,
    reset_scheduling_envelope,
    set_scheduling_envelope,
    task_options,
)

__all__ = [
    "_ensure_ray",
    "_rmtree",
    "_single_node",
    "_wrap_tasks",
    "actor_fault_options",
    "clamp_workers",
    "cluster_topology",
    "create_worker_placement",
    "current_envelope",
    "draining_workers",
    "engine_config_json",
    "fault_options",
    "gather_map_results",
    "placement_actor_options",
    "recovery_policy",
    "release_autoscale",
    "release_placement",
    "request_autoscale",
    "reset_scheduling_envelope",
    "resolve_transport",
    "runtime_bloom_join",
    "set_scheduling_envelope",
    "skew_join_salt",
    "speculation_policy",
    "task_options",
]
