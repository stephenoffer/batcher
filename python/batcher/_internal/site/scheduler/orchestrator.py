"""The container orchestrators and cluster managers, and Ray inside whatever they started.

Kubernetes runs most GPU capacity that is not on a batch scheduler, and the rest of this
module is what the remaining share runs on: Nomad, YARN, Databricks, and Ray itself.

They share a property the batch schedulers do not, and it is what shapes every reader here.
**None of them publishes the job's shape by default.** Kubernetes injects nothing about the
pod unless a manifest asks through the downward API; YARN names the container but not the
application's width; Ray tells a worker its own address and nothing about its peers. So each
reader reports the identity it can see and leaves the rest empty, and the *shape* comes from
`launcher_ranks` — the `torchrun`/MPI/PMI variables that whatever started the workers set.
That fallback is the difference between a four-pod job knowing it is four pods and every pod
believing it is alone.
"""

from __future__ import annotations

from batcher._internal.site.scheduler.job import (
    SchedulerJob,
    env_int,
    env_str,
    first_env,
    launcher_ranks,
    visible_device_count,
)

__all__ = [
    "databricks_job",
    "kubernetes_job",
    "nomad_job",
    "ray_job",
    "yarn_job",
]

#: The Kubernetes API server's in-cluster address, injected into every pod. Its presence is the
#: definition of "running under Kubernetes" and is what the client libraries themselves check.
K8S_MARKER = "KUBERNETES_SERVICE_HOST"

#: Where the node this process runs on is published, most conventional first. Kubernetes sets
#: none of these itself: they come from a manifest's downward API, from Volcano, or from a
#: platform that fills them in. All three spellings are in wide use.
_K8S_NODE_VARS = ("NODE_NAME", "KUBERNETES_NODE_NAME", "SPEC_NODE_NAME")


def _derived_nodes(ranks_tasks: int, ranks_local: int) -> int:
    """Nodes implied by a launcher's world size and per-node process count, or `0`.

    A `torchrun`-style job publishes `WORLD_SIZE` and `LOCAL_WORLD_SIZE`, whose quotient is the
    node count. It is the only width an orchestrated job usually knows, and without it a
    sixteen-process job across two nodes reads as single-node — which turns off every
    cross-node decision the engine makes.
    """
    if ranks_tasks > 0 and ranks_local > 0 and ranks_tasks % ranks_local == 0:
        return ranks_tasks // ranks_local
    return 0


def kubernetes_job() -> SchedulerJob:
    """The pod this process runs in, from the downward-API convention and the launcher.

    Kubernetes injects nothing about the pod by default: `POD_NAME`, `POD_NAMESPACE`, and
    `NODE_NAME` are set by the manifest through the downward API. A pod whose manifest sets
    none of them reports the marker-derived kind and empty identity, which is honest — the
    fields are absent, not zero.
    """
    node = first_env(_K8S_NODE_VARS)
    ranks = launcher_ranks()
    # Volcano and the Kubeflow operators number their pods themselves; where they have, that
    # index is more trustworthy than a launcher variable a user may have set by hand.
    rank = env_int("VC_TASK_INDEX") or env_int("VK_TASK_INDEX") or ranks.rank
    return SchedulerJob(
        kind="kubernetes",
        job_id=first_env(("POD_NAME", "JOB_NAME")),
        nodes=(node,) if node else (),
        num_nodes=_derived_nodes(ranks.tasks, ranks.local_size),
        node_name=node,
        partition=env_str("POD_NAMESPACE"),
        # A device-plugin allocation is exposed to the container as the visible device list,
        # which is the only per-pod device count Kubernetes publishes into the process.
        gpus_per_node=visible_device_count(),
        tasks=ranks.tasks,
        tasks_per_node=ranks.local_size,
        rank=rank,
        local_rank=ranks.local_rank,
        local_size=ranks.local_size,
        array_index=env_str("JOB_COMPLETION_INDEX"),
    )


def ray_job() -> SchedulerJob:
    """The Ray runtime this process belongs to, when nothing outer scheduled it.

    Reported only when Ray is the outermost thing: a Ray worker inside a Slurm allocation or a
    Kubernetes pod reports that instead, because the allocation is what bounds the job and
    what will end it. A caller that specifically wants to know whether Ray is up asks Ray.
    """
    ranks = launcher_ranks()
    return SchedulerJob(
        kind="ray",
        job_id=env_str("RAY_JOB_ID"),
        node_name=first_env(("NODE_NAME", "RAY_NODE_IP_ADDRESS")),
        gpus_per_node=visible_device_count(),
        tasks=ranks.tasks,
        rank=ranks.rank,
        local_rank=ranks.local_rank,
        local_size=ranks.local_size,
    )


def nomad_job() -> SchedulerJob:
    """The allocation this process runs in, from Nomad.

    Nomad schedules a good share of on-premises capacity that never adopted Kubernetes, and it
    is unusual in publishing its resource grant directly: `NOMAD_CPU_LIMIT` is in MHz rather
    than cores, so it is carried as identity rather than converted into a core count that
    would be a guess about clock speed.
    """
    ranks = launcher_ranks()
    return SchedulerJob(
        kind="nomad",
        job_id=first_env(("NOMAD_JOB_ID", "NOMAD_JOB_NAME")),
        node_name=first_env(("NOMAD_CLIENT_INTERFACE", "HOSTNAME")),
        partition=env_str("NOMAD_NAMESPACE"),
        gpus_per_node=visible_device_count(),
        tasks=ranks.tasks,
        rank=env_int("NOMAD_ALLOC_INDEX") or ranks.rank,
        local_rank=ranks.local_rank,
        local_size=ranks.local_size,
        array_index=env_str("NOMAD_ALLOC_INDEX"),
    )


def yarn_job() -> SchedulerJob:
    """The container this process runs in, from YARN.

    YARN is what schedules an on-premises Hadoop or Spark estate, which is exactly the estate
    a Batcher migration lands in. It publishes the container id and the node manager's host;
    the application id is the container id's own middle field, so it is derived rather than
    looked up.
    """
    container = env_str("CONTAINER_ID")
    ranks = launcher_ranks()
    return SchedulerJob(
        kind="yarn",
        job_id=_yarn_application_id(container) or container,
        node_name=env_str("NM_HOST"),
        gpus_per_node=visible_device_count(),
        tasks=ranks.tasks,
        rank=ranks.rank,
        local_rank=ranks.local_rank,
        local_size=ranks.local_size,
    )


def _yarn_application_id(container_id: str) -> str:
    """The application id inside a YARN container id, or `""`.

    `container_e17_1699999999999_0042_01_000003` names application
    `application_1699999999999_0042`. The epoch field (`e17`) is optional and is what a naive
    positional split gets wrong, so the cluster timestamp is found by shape instead.
    """
    parts = container_id.split("_")
    for i, part in enumerate(parts):
        if len(part) >= 13 and part.isdigit() and i + 1 < len(parts):
            return f"application_{part}_{parts[i + 1]}"
    return ""


def databricks_job() -> SchedulerJob:
    """The cluster this process runs on, from a Databricks runtime.

    Reported for the same reason as YARN: a migration onto Batcher frequently runs first
    *inside* the platform it is migrating from, and a run that cannot name where it is cannot
    explain a defaults difference between there and anywhere else.
    """
    return SchedulerJob(
        kind="databricks",
        job_id=first_env(("DB_CLUSTER_ID", "DB_CLUSTER_NAME")),
        node_name=env_str("DB_DRIVER_IP")
        if env_str("DB_IS_DRIVER") == "TRUE"
        else env_str("HOSTNAME"),
        gpus_per_node=visible_device_count(),
    )
