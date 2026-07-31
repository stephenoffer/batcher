"""What launched this process — Slurm, PBS, LSF, Kubernetes, Ray, or nothing.

A GPU cluster is scheduled by one of two families, and neither is Ray. Batch schedulers run
most of the research and HPC-adjacent capacity — Slurm most visibly, with PBS/OpenPBS and LSF
behind it on older and vendor-supplied clusters; Kubernetes runs most of the rest. Ray sits
*inside* an allocation one of them made. That matters because the outer scheduler already
knows the shape of the job, and the shape is otherwise expensive or impossible to discover:

* **The node list.** Slurm hands the job its allocated nodes in `SLURM_JOB_NODELIST`. That is
  the multi-node topology, available before Ray has started, in an environment variable.
* **Devices per node.** `SLURM_GPUS_ON_NODE` is the allocation's per-node device count, which
  is what a stage should size against — not the node's physical device count, which is what a
  local probe reports and which is wrong the moment two jobs share a node.
* **Rank and locality.** `SLURM_PROCID` and `SLURM_LOCALID` say which worker this is and how
  many share its node, which is what decides whether a collective stays on NVLink.
* **The pod's node.** Under Kubernetes the useful identity is the *node* the pod landed on,
  because that is what carries the topology labels and what a co-location decision is about.
* **A host file rather than a variable.** PBS and LSF write their node lists to a file, one
  line per task *slot* rather than per node, so the distinct names in order are the allocation
  and the repetition is the task layout.

Read from environment variables, which every one of these schedulers exports into the process.
Nothing here shells out to `scontrol` or calls an API server: both are slow, both can fail
closed, and neither is needed for the facts above.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

__all__ = [
    "SchedulerJob",
    "expand_nodelist",
    "scheduler_job",
    "scheduler_kind",
]

#: A bracketed range group in a Slurm hostlist: `node[001-004,007]`. Matched as a whole so the
#: prefix and the group stay together; nested brackets do not occur in the format.
_HOSTLIST_RE = re.compile(r"([^,\[\]]+)\[([^\]]+)\]|([^,\[\]]+)")

#: The Kubernetes API server's in-cluster address, injected into every pod. Its presence is the
#: definition of "running under Kubernetes" and is what the client libraries themselves check.
_K8S_MARKER = "KUBERNETES_SERVICE_HOST"


def _int_env(name: str, default: int = 0) -> int:
    """An environment variable as an int, or `default` when absent or unparseable."""
    try:
        return int(os.environ.get(name, "").strip())
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class SchedulerJob:
    """The job this process belongs to, as its scheduler describes it.

    Attributes:
        kind: `"slurm"`, `"pbs"`, `"lsf"`, `"kubernetes"`, `"ray"`, or `"none"`.
        job_id: The scheduler's job identifier, `""` when there is none.
        nodes: Node names in the allocation, in the scheduler's own order. Empty under a
            scheduler that does not publish the list.
        gpus_per_node: Devices *this allocation* was given per node, `0` when unpublished.
            Distinct from the node's physical device count, which is what a local probe sees
            and which over-counts whenever a node is shared.
        cpus_per_node: Cores the allocation was given per node, `0` when unpublished.
        tasks: Total tasks in the job, `0` when unpublished.
        rank: This process's global task index, `0` when unpublished or single-task.
        local_rank: This process's index among the tasks on its own node.
        node_name: The node this process is on, `""` when unpublished.
        partition: The scheduler's queue or partition — a Slurm partition, a PBS or LSF
            queue, a Kubernetes namespace — `""` when unpublished.
    """

    kind: str = "none"
    job_id: str = ""
    nodes: tuple[str, ...] = field(default_factory=tuple)
    gpus_per_node: int = 0
    cpus_per_node: int = 0
    tasks: int = 0
    rank: int = 0
    local_rank: int = 0
    node_name: str = ""
    partition: str = ""

    @property
    def multi_node(self) -> bool:
        """Whether the allocation spans more than one node."""
        return len(self.nodes) > 1

    @property
    def total_gpus(self) -> int:
        """Devices across the whole allocation, `0` when either factor is unpublished.

        The number a distributed stage's parallelism should be sized against, and the one an
        operator means by "how big is this job".
        """
        return self.gpus_per_node * len(self.nodes)


def _expand_group(prefix: str, spec: str) -> list[str]:
    """Expand one bracketed hostlist group, preserving the zero padding Slurm uses.

    `node[001-003,007]` yields `node001 node002 node003 node007`. Padding is taken from the
    literal width of the lower bound, because that is what the node is actually named.
    """
    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, _, high = part.partition("-")
            try:
                start, end = int(low), int(high)
            except ValueError:
                out.append(f"{prefix}{part}")
                continue
            width = len(low)
            # A range whose bounds are inverted is a malformed list, not an empty allocation;
            # it is passed through literally rather than silently yielding no nodes.
            if end < start:
                out.append(f"{prefix}{part}")
                continue
            out.extend(f"{prefix}{n:0{width}d}" for n in range(start, end + 1))
        else:
            out.append(f"{prefix}{part}")
    return out


def expand_nodelist(spec: str) -> tuple[str, ...]:
    """Expand a Slurm hostlist into individual node names.

    Slurm compresses an allocation's node list into a range notation, so a 64-node job arrives
    as one short string. Expanding it here is what turns `SLURM_JOB_NODELIST` into a topology
    the scheduler can reason about, without shelling out to `scontrol show hostnames` — which
    costs a subprocess per query and is unavailable in a container that has the environment but
    not the Slurm client.

    Args:
        spec: The hostlist, as `"gpu-[001-004,007],login01"`.

    Returns:
        Node names in the order the list gives them, deduplicated. Empty for an empty or
        unparseable spec, which callers read as "no node list published".
    """
    if not spec or not spec.strip():
        return ()
    out: list[str] = []
    for match in _HOSTLIST_RE.finditer(spec.strip()):
        prefix, group, plain = match.group(1), match.group(2), match.group(3)
        if plain is not None:
            name = plain.strip()
            if name:
                out.append(name)
        elif prefix is not None and group is not None:
            out.extend(_expand_group(prefix.strip(), group))
    return tuple(dict.fromkeys(out))


def scheduler_kind() -> str:
    """What launched this process.

    Ordered outermost-first: a Ray worker inside a Slurm allocation reports `"slurm"`, because
    the allocation is what bounds the job and what will end it. A caller that specifically
    wants to know whether Ray is up asks Ray.

    Returns:
        `"slurm"`, `"pbs"`, `"lsf"`, `"kubernetes"`, `"ray"`, or `"none"`.
    """
    if os.environ.get("SLURM_JOB_ID", "").strip():
        return "slurm"
    if os.environ.get("PBS_JOBID", "").strip():
        return "pbs"
    if os.environ.get("LSB_JOBID", "").strip():
        return "lsf"
    if os.environ.get(_K8S_MARKER, "").strip():
        return "kubernetes"
    if os.environ.get("RAY_ADDRESS", "").strip() or os.environ.get("RAY_NODE_IP_ADDRESS", ""):
        return "ray"
    return "none"


def _slurm_job() -> SchedulerJob:
    """The allocation this process belongs to, from Slurm's environment."""
    nodes = expand_nodelist(os.environ.get("SLURM_JOB_NODELIST", ""))
    return SchedulerJob(
        kind="slurm",
        job_id=os.environ.get("SLURM_JOB_ID", "").strip(),
        nodes=nodes,
        # `SLURM_GPUS_ON_NODE` is the allocation's grant; `SLURM_JOB_GPUS` is the device id
        # list, whose length is the same number and which older Slurm sets instead.
        gpus_per_node=_int_env("SLURM_GPUS_ON_NODE")
        or len([g for g in os.environ.get("SLURM_JOB_GPUS", "").split(",") if g.strip()]),
        cpus_per_node=_int_env("SLURM_CPUS_ON_NODE"),
        tasks=_int_env("SLURM_NTASKS"),
        rank=_int_env("SLURM_PROCID"),
        local_rank=_int_env("SLURM_LOCALID"),
        node_name=os.environ.get("SLURMD_NODENAME", "").strip(),
        partition=os.environ.get("SLURM_JOB_PARTITION", "").strip(),
    )


def _visible_devices() -> tuple[str, ...]:
    """Device ids the container runtime exposed, dropping the runtime's own sentinels.

    `NVIDIA_VISIBLE_DEVICES` carries `"void"` for a container that asked for no devices and
    `"none"` for one that gets the driver without any, and counting either as a device would
    size a stage onto hardware the container cannot see.
    """
    raw = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
    ids = [d.strip() for d in raw.split(",")]
    return tuple(d for d in ids if d and d.lower() not in {"void", "none"})


def _kubernetes_job() -> SchedulerJob:
    """The pod this process runs in, from the downward-API convention.

    Kubernetes injects nothing about the pod by default: `POD_NAME`, `POD_NAMESPACE`, and
    `NODE_NAME` are set by the manifest through the downward API. A pod whose manifest sets
    none of them reports the marker-derived kind and empty identity, which is honest — the
    fields are absent, not zero.
    """
    node = ""
    for var in ("NODE_NAME", "KUBERNETES_NODE_NAME", "SPEC_NODE_NAME"):
        node = os.environ.get(var, "").strip()
        if node:
            break
    return SchedulerJob(
        kind="kubernetes",
        job_id=os.environ.get("POD_NAME", "").strip() or os.environ.get("JOB_NAME", "").strip(),
        nodes=(node,) if node else (),
        node_name=node,
        partition=os.environ.get("POD_NAMESPACE", "").strip(),
        # A device-plugin allocation is exposed to the container as the visible device list,
        # which is the only per-pod device count Kubernetes publishes into the process.
        gpus_per_node=len(_visible_devices()),
    )


def scheduler_job() -> SchedulerJob:
    """The job this process belongs to.

    Read live rather than memoized: a process re-execed into a different allocation is rare,
    and the read is a handful of environment lookups, so caching would buy nothing and cost a
    stale answer.

    Returns:
        The job. An unscheduled process (a laptop, a notebook) reports `kind="none"` with
        empty fields, which every caller treats as "decide it yourself, as before".
    """
    kind = scheduler_kind()
    if kind == "slurm":
        return _slurm_job()
    if kind == "pbs":
        return _pbs_job()
    if kind == "lsf":
        return _lsf_job()
    if kind == "kubernetes":
        return _kubernetes_job()
    if kind == "ray":
        return SchedulerJob(kind="ray", node_name=os.environ.get("NODE_NAME", "").strip())
    return SchedulerJob()


def _nodes_from_file(path: str) -> tuple[str, ...]:
    """Node names from a scheduler's host file, deduplicated in first-seen order.

    PBS writes one line per *task slot*, so a four-node job with eight tasks each lists every
    node eight times. The distinct names in order are the allocation; the repetition is the
    task layout, which `tasks` already carries.
    """
    try:
        with open(path) as f:
            names = [line.strip() for line in f if line.strip()]
    except OSError:
        return ()
    return tuple(dict.fromkeys(names))


def _pbs_job() -> SchedulerJob:
    """The allocation this process belongs to, from PBS/OpenPBS.

    PBS puts its node list in a *file* rather than an environment variable, which is the one
    structural difference from Slurm worth handling: `PBS_NODEFILE` names it, and the file is
    one line per task slot rather than one per node.
    """
    nodefile = os.environ.get("PBS_NODEFILE", "").strip()
    nodes = _nodes_from_file(nodefile) if nodefile else ()
    return SchedulerJob(
        kind="pbs",
        job_id=os.environ.get("PBS_JOBID", "").strip(),
        nodes=nodes,
        gpus_per_node=_int_env("PBS_NGPUS"),
        cpus_per_node=_int_env("PBS_NCPUS"),
        tasks=_int_env("PBS_NP"),
        node_name=os.environ.get("PBS_NODENUM", "").strip() or (nodes[0] if nodes else ""),
        partition=os.environ.get("PBS_QUEUE", "").strip(),
    )


def _lsf_job() -> SchedulerJob:
    """The allocation this process belongs to, from LSF.

    LSF lists its hosts inline in `LSB_HOSTS`, space-separated and repeated per slot, or in a
    file named by `LSB_DJOB_HOSTFILE` on a large job where the variable would overflow. Both
    are read, the file first, because that is the one that stays correct at scale.
    """
    hostfile = os.environ.get("LSB_DJOB_HOSTFILE", "").strip()
    nodes = _nodes_from_file(hostfile) if hostfile else ()
    if not nodes:
        nodes = tuple(dict.fromkeys(os.environ.get("LSB_HOSTS", "").split()))
    return SchedulerJob(
        kind="lsf",
        job_id=os.environ.get("LSB_JOBID", "").strip(),
        nodes=nodes,
        gpus_per_node=len(_visible_devices()),
        tasks=_int_env("LSB_DJOB_NUMPROC"),
        node_name=os.environ.get("HOSTNAME", "").strip() or (nodes[0] if nodes else ""),
        partition=os.environ.get("LSB_QUEUE", "").strip(),
    )
