"""Which custom resource names a cluster's node *classes*.

`heterogeneous_node_isolation` keeps a CPU fleet off accelerator nodes by requiring a custom
resource only accelerator-free nodes advertise, so a CPU shuffle cannot hold an idle GPU
node's cores from an inference stage. **Which** resource that is depends on who built the
cluster, and selecting on a name nothing carries matches nothing — the same defect the zone
selector already fixed by reading `zone_label` off the node rather than assuming a key (a
Kubernetes fleet labels `topology.kubernetes.io/zone`, a plain Ray one
`ray.io/availability-zone`).

Separate from `scaling.py` and taking node records as an argument rather than reading Ray:
the question "which of these names describes a class" is a pure one over resource dicts, and
keeping it pure is what lets it be tested against a synthetic fleet with no monkeypatching,
and what keeps it from importing the module that will call it.
"""

from __future__ import annotations

from batcher._internal.accelerators import accelerator_units

__all__ = ["cpu_only_marker_resource"]

#: Resources that can never mark a node *class*: the capacity counters every node carries,
#: and Ray's per-node identity resources (`node:<ip>`, placement-group `bundle_group_*`),
#: which name one node or one reservation rather than a property a respawned node re-earns.
_CAPACITY_RESOURCES = frozenset({"CPU", "GPU", "memory", "object_store_memory", "bundle"})
_IDENTITY_PREFIXES = ("node:", "bundle_group", "accelerator_type:")


def _class_markers(resources: dict) -> set[str]:
    """The resource names on one node that could describe a node *class* rather than a node."""
    return {
        name
        for name in resources
        if name not in _CAPACITY_RESOURCES and not name.startswith(_IDENTITY_PREFIXES)
    }


def _names_cpu_only(name: str) -> bool:
    """Whether a resource name *says* it marks an accelerator-free node.

    Deliberately a vocabulary test rather than "any label the CPU-only nodes happen to share".
    On a fleet with one CPU instance type, the instance-type label
    (`anyscale/node-group:16cpu-32gb`) is carried by exactly those nodes and by no accelerator
    node, so a purely structural rule picks it — and it is a different fact, which stops
    matching the moment a second CPU instance type appears. A coincidence that survives one
    cluster shape is worse than no marker, because the selector it produces still schedules.
    """
    low = name.lower()
    return "cpu_only" in low or "cpu-only" in low or low.rsplit("/", 1)[-1] == "cpu_node"


def _is_accelerator(resources: dict) -> bool:
    """Whether a node's resource dict describes an accelerator node (GPU or TPU/NPU/etc.)."""
    return resources.get("GPU", 0.0) > 0 or accelerator_units(resources) > 0


def cpu_only_marker_resource(nodes: list[dict], configured: str) -> str:
    """The resource marking `nodes`' accelerator-free machines, else `configured`.

    A candidate must be present on **every** accelerator-free node, absent from **every**
    accelerator node, and name the property rather than merely correlate with it
    (`_names_cpu_only`). `configured` — the deploy-time `distributed.cpu_node_resource`
    convention an operator applies by hand — wins outright when it qualifies, and is the
    fallback otherwise, which keeps this strictly additive: a cluster that opted into the gate
    without labelling anything behaves exactly as it did before, and a managed fleet that has
    already labelled itself under its own name (`anyscale/cpu_only:true`) now works without
    anyone relabelling it.

    Args:
        nodes: Worker-eligible node records, each with a `"Resources"` dict.
        configured: The configured marker name, used as the fallback.

    Returns:
        The resource name to select on. Never empty — the caller's own gate has already
        established that the cluster is heterogeneous and can host the fleet.
    """
    on_cpu_only: list[set[str]] = []
    on_accelerators: set[str] = set()
    saw_accelerator = False
    for node in nodes:
        resources = node.get("Resources") or {}
        markers = _class_markers(resources)
        if _is_accelerator(resources):
            saw_accelerator = True
            on_accelerators |= markers
        else:
            on_cpu_only.append(markers)
    if not saw_accelerator or not on_cpu_only:
        return configured  # homogeneous either way; the caller's gate has the say
    shared = set.intersection(*on_cpu_only) - on_accelerators
    if configured in shared:
        return configured
    qualifying = sorted(name for name in shared if _names_cpu_only(name))
    return qualifying[0] if qualifying else configured
