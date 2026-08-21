"""The managed job services — where a job is submitted rather than a cluster stood up.

AWS Batch, SageMaker, Vertex AI, Azure ML and SkyPilot. A job here is described by the service
rather than by a scheduler daemon on the node, and the description arrives in two forms: a few
flat variables, or a JSON blob naming every worker in the job.

The JSON form is worth the parse. Vertex AI's `CLUSTER_SPEC` and SageMaker's `SM_HOSTS` are the
only place those platforms publish the *peer list* — the thing a shuffle needs and a launcher
variable cannot supply — and both are small documents already sitting in the environment.

Nothing here calls the platform's API. A control-plane round trip to learn the shape of the
job the process is already inside is a hang waiting for a misconfigured egress rule.
"""

from __future__ import annotations

import json

from batcher._internal.logging import note_suppressed
from batcher._internal.site.scheduler.job import (
    SchedulerJob,
    env_int,
    env_str,
    first_env,
    launcher_ranks,
    visible_device_count,
)

__all__ = [
    "aws_batch_job",
    "azureml_job",
    "sagemaker_job",
    "skypilot_job",
    "vertex_job",
]


def _json_env(name: str) -> object:
    """One environment variable parsed as JSON, or `None` when absent or malformed.

    Malformed is reported through the suppression log rather than raised: the platform owns
    this variable's format, and a parse failure means a shape changed under us — worth a line
    in the diagnostic log, never worth failing a query that has nothing to do with it.
    """
    raw = env_str(name)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError as exc:
        note_suppressed("site", f"parse {name} as JSON", exc)
        return None


def aws_batch_job() -> SchedulerJob:
    """The job this process belongs to, from AWS Batch.

    Batch is the one managed service here that publishes a real multi-node shape without a
    JSON document: a multi-node parallel job gets its node index and the job's node count as
    plain variables, and an array job gets its index the same way.
    """
    ranks = launcher_ranks()
    return SchedulerJob(
        kind="aws_batch",
        job_id=env_str("AWS_BATCH_JOB_ID"),
        num_nodes=env_int("AWS_BATCH_JOB_NUM_NODES"),
        gpus_per_node=visible_device_count(),
        tasks=env_int("AWS_BATCH_JOB_NUM_NODES") or ranks.tasks,
        rank=env_int("AWS_BATCH_JOB_NODE_INDEX") or ranks.rank,
        local_rank=ranks.local_rank,
        local_size=ranks.local_size,
        node_name=env_str("HOSTNAME"),
        partition=env_str("AWS_BATCH_JQ_NAME"),
        array_index=env_str("AWS_BATCH_JOB_ARRAY_INDEX"),
    )


def sagemaker_job() -> SchedulerJob:
    """The training job this process belongs to, from SageMaker.

    SageMaker names every host in the job (`SM_HOSTS`) and which one this is
    (`SM_CURRENT_HOST`), so the rank is the position of one in the other rather than something
    a launcher had to supply. It also publishes the per-host device and core counts directly,
    which almost nothing else here does.
    """
    hosts = _json_env("SM_HOSTS")
    nodes = tuple(str(h) for h in hosts) if isinstance(hosts, list) else ()
    current = env_str("SM_CURRENT_HOST")
    ranks = launcher_ranks()
    return SchedulerJob(
        kind="sagemaker",
        job_id=env_str("TRAINING_JOB_NAME"),
        nodes=nodes,
        gpus_per_node=env_int("SM_NUM_GPUS") or visible_device_count(),
        cpus_per_node=env_int("SM_NUM_CPUS"),
        tasks=len(nodes) or ranks.tasks,
        rank=nodes.index(current) if current in nodes else ranks.rank,
        local_rank=ranks.local_rank,
        local_size=ranks.local_size,
        node_name=current,
    )


def vertex_job() -> SchedulerJob:
    """The custom training job this process belongs to, from Vertex AI.

    Vertex publishes `CLUSTER_SPEC`, a JSON document naming every worker pool and this
    process's place in one. Its `cluster` map is pool name to `host:port` list, so the job's
    node list is those lists concatenated and the rank is this task's offset within them.
    """
    spec = _json_env("CLUSTER_SPEC")
    ranks = launcher_ranks()
    if not isinstance(spec, dict):
        return SchedulerJob(
            kind="vertex",
            job_id=env_str("CLOUD_ML_JOB_ID"),
            gpus_per_node=visible_device_count(),
            tasks=ranks.tasks,
            rank=ranks.rank,
            local_rank=ranks.local_rank,
            local_size=ranks.local_size,
        )
    cluster = spec.get("cluster") or {}
    task = spec.get("task") or {}
    pools = [str(name) for name in cluster] if isinstance(cluster, dict) else []
    nodes: list[str] = []
    rank = 0
    for pool in pools:
        members = cluster.get(pool) or []
        if not isinstance(members, list):
            continue
        if isinstance(task, dict) and task.get("type") == pool:
            rank = len(nodes) + int(task.get("index") or 0)
        nodes.extend(str(m).rsplit(":", 1)[0] for m in members)
    return SchedulerJob(
        kind="vertex",
        job_id=env_str("CLOUD_ML_JOB_ID") or str(spec.get("job") or ""),
        nodes=tuple(nodes),
        gpus_per_node=visible_device_count(),
        tasks=len(nodes) or ranks.tasks,
        rank=rank or ranks.rank,
        local_rank=ranks.local_rank,
        local_size=ranks.local_size,
        node_name=nodes[rank] if rank < len(nodes) else "",
    )


def azureml_job() -> SchedulerJob:
    """The run this process belongs to, from Azure ML.

    Azure ML publishes the run's identity and leaves the job's shape to whatever it launched —
    `mpirun` for an MPI distribution, `torchrun` for a PyTorch one — so the shape comes from
    the launcher variables and only the identity is read here. Reporting an identity with an
    empty shape is the honest answer; inventing a node count from a run id is not.
    """
    ranks = launcher_ranks()
    return SchedulerJob(
        kind="azureml",
        job_id=env_str("AZUREML_RUN_ID"),
        gpus_per_node=visible_device_count(),
        tasks=ranks.tasks,
        rank=ranks.rank,
        local_rank=ranks.local_rank,
        local_size=ranks.local_size,
        node_name=env_str("HOSTNAME"),
        partition=env_str("AZUREML_ARM_WORKSPACE_NAME"),
    )


def skypilot_job() -> SchedulerJob:
    """The task this process belongs to, from SkyPilot.

    SkyPilot is how a job reaches a neocloud without being written against it, so it is the
    layer that knows the cluster's shape when the platform underneath publishes nothing:
    node rank, node count, the peer IP list and the per-node device count are all exported.
    Read before Kubernetes, because a SkyPilot task on a Kubernetes backend carries both and
    only one of them describes the job.
    """
    ips = tuple(ip.strip() for ip in env_str("SKYPILOT_NODE_IPS").split() if ip.strip())
    rank = env_int("SKYPILOT_NODE_RANK")
    gpus = env_int("SKYPILOT_NUM_GPUS_PER_NODE")
    ranks = launcher_ranks()
    return SchedulerJob(
        kind="skypilot",
        job_id=first_env(("SKYPILOT_TASK_ID", "SKYPILOT_CLUSTER_NAME")),
        nodes=ips,
        num_nodes=env_int("SKYPILOT_NUM_NODES"),
        gpus_per_node=gpus or visible_device_count(),
        tasks=env_int("SKYPILOT_NUM_NODES") or ranks.tasks,
        rank=rank or ranks.rank,
        local_rank=ranks.local_rank,
        local_size=ranks.local_size,
        node_name=ips[rank] if rank < len(ips) else "",
    )
