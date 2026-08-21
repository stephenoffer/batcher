"""Ray lifecycle, scheduling envelope, autoscaling, and fault policies for the
distributed executor.

Façade over four responsibility modules: `lifecycle` (Ray init + remote-task
wrapping + single-node fallback), `scheduling` (the metadata-driven envelope and
placement groups), `scaling` (cluster topology + autoscaler request lifecycle), and
`policies` (config-driven fault-tolerance / recovery / skew). Import names from here;
the split is an implementation detail.
"""

from __future__ import annotations

from .autoscale_request import release_autoscale, request_autoscale
from .lifecycle import (
    _ensure_ray,
    _rmtree,
    _single_node,
    _wrap_tasks,
    engine_config_json,
    resolve_transport,
)
from .metering import (
    drain_worker_metrics,
    execute_metered,
    record_worker_metrics,
)
from .policies import (
    actor_fault_options,
    blame_host_for_reduce_failure,
    draining_workers,
    fault_options,
    gather_map_results,
    is_recoverable_task_failure,
    map_barrier,
    recovery_policy,
    runtime_bloom_join,
    skew_join_salt,
    speculation_policy,
)
from .readiness import await_autoscale
from .reduce import gather_in_windows, run_bucket_reduce
from .reducers import buckets_for_envelope, map_partitions, shuffle_partitions
from .scaling import (
    alive_node_count,
    clamp_workers,
    cluster_topology,
    node_class_selector,
    topology_scope,
    worker_node_memory_bytes,
)
from .scheduling import (
    create_worker_placement,
    current_envelope,
    fleet_actor_options,
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
    "alive_node_count",
    "await_autoscale",
    "blame_host_for_reduce_failure",
    "buckets_for_envelope",
    "clamp_workers",
    "cluster_topology",
    "create_worker_placement",
    "current_envelope",
    "drain_worker_metrics",
    "draining_workers",
    "engine_config_json",
    "execute_metered",
    "fault_options",
    "fleet_actor_options",
    "gather_in_windows",
    "gather_map_results",
    "is_recoverable_task_failure",
    "map_barrier",
    "map_partitions",
    "node_class_selector",
    "placement_actor_options",
    "record_worker_metrics",
    "recovery_policy",
    "release_autoscale",
    "release_placement",
    "request_autoscale",
    "reset_scheduling_envelope",
    "resolve_transport",
    "run_bucket_reduce",
    "runtime_bloom_join",
    "set_scheduling_envelope",
    "shuffle_partitions",
    "skew_join_salt",
    "speculation_policy",
    "task_options",
    "topology_scope",
    "worker_node_memory_bytes",
]
