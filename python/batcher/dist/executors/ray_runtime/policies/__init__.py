"""Config-driven fault-tolerance, recovery, and skew policies for the distributed executor.

Three responsibilities that grew together and are now separated: `_faults` holds the pure
``active_config()`` → policy/option builders and the error classification they rely on,
`_drain` answers which workers sit on a node that is going away, and `_barrier` is the
map-stage gather loop that turns a preempted partition into a resubmission rather than a
failed stage.

Import names from here; the split is an implementation detail. The module holds no Ray
lifecycle state, so it imports nothing from the rest of the package.
"""

from __future__ import annotations

from ._barrier import _pending_window, gather_map_results, map_barrier
from ._drain import draining_workers
from ._faults import (
    _DEFAULT_PENDING_WINDOW,
    _is_fatal_ray_error,
    _is_transient_udf_error,
    actor_fault_options,
    fault_options,
    is_recoverable_task_failure,
    recovery_policy,
    runtime_bloom_join,
    skew_join_salt,
    speculation_policy,
)

# The leading `_`-prefixed names are private to the package but re-exported deliberately:
# the error-classification helpers decide whether a failure is retried or surfaced, and the
# submit-ahead window is a scheduling bound with its own sizing rules. Both are unit-tested
# directly rather than through a whole map stage, so the façade has to name them.
__all__ = [
    "_DEFAULT_PENDING_WINDOW",
    "_is_fatal_ray_error",
    "_is_transient_udf_error",
    "_pending_window",
    "actor_fault_options",
    "draining_workers",
    "fault_options",
    "gather_map_results",
    "is_recoverable_task_failure",
    "map_barrier",
    "recovery_policy",
    "runtime_bloom_join",
    "skew_join_salt",
    "speculation_policy",
]
