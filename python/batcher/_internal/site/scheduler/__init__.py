"""What launched this process, and the shape of the job it belongs to.

A GPU cluster is scheduled by one of three families, and none of them is Ray. **Batch
schedulers** run most research and HPC-adjacent capacity — Slurm most visibly, with PBS/OpenPBS,
LSF, Grid Engine, Flux and HTCondor behind it. **Container orchestrators** run most of the
rest: Kubernetes, and Nomad, YARN or Databricks where an estate never moved to it. **Managed job
services** — AWS Batch, SageMaker, Vertex AI, Azure ML, SkyPilot — run the share that is
submitted rather than provisioned. Ray sits *inside* an allocation one of them made.

That matters because the outer scheduler already knows the shape of the job, and the shape is
otherwise expensive or impossible to discover:

* **The node list.** Slurm hands the job its allocated nodes in `SLURM_JOB_NODELIST`, Vertex AI
  in `CLUSTER_SPEC`, SageMaker in `SM_HOSTS`. That is the multi-node topology, available before
  Ray has started, already in the environment.
* **Devices per node.** `SLURM_GPUS_ON_NODE` is the allocation's per-node device count, which is
  what a stage should size against — not the node's physical device count, which is what a local
  probe reports and which is wrong the moment two jobs share a node.
* **Cores per task.** A shared node is the norm under Grid Engine, LSF and PBS. `allocated_cpus`
  is the bound that keeps eight co-tenant tasks from each fanning out to the whole node.
* **Rank and locality.** Which worker this is and how many share its node decides whether a
  collective stays on NVLink. Slurm and Flux publish it; everywhere else it comes from the
  launcher — `torchrun`, `mpirun`, or a PMI-speaking equivalent — and reading those is the
  difference between a four-node job knowing it is four nodes and every node believing it is
  alone.

Read from environment variables and, where a scheduler writes one, its own host file. Nothing
here shells out to `scontrol` or calls an API server: both are slow, both can fail closed, and
neither is needed for the facts above.

The module layout follows the three families: `batch`, `orchestrator` and `managed` hold the
readers, `job` the record they fill in and the parsers they share, `hostlist` the node-list
formats, and `detect` the ordered table that picks between them.
"""

from __future__ import annotations

from batcher._internal.site.scheduler.detect import (
    SCHEDULERS,
    SCRATCH_DIR_VARS,
    allocated_cpus,
    scheduler_job,
    scheduler_kind,
    scheduler_memory_bytes,
    scheduler_scratch_dir,
)
from batcher._internal.site.scheduler.hostlist import (
    expand_nodelist,
    nodes_from_file,
    nodes_from_pe_hostfile,
)
from batcher._internal.site.scheduler.job import (
    VISIBLE_DEVICE_COUNT_ENVS,
    LauncherRanks,
    SchedulerJob,
    launcher_ranks,
    visible_device_count,
)

__all__ = [
    "SCHEDULERS",
    "SCRATCH_DIR_VARS",
    "VISIBLE_DEVICE_COUNT_ENVS",
    "LauncherRanks",
    "SchedulerJob",
    "allocated_cpus",
    "expand_nodelist",
    "launcher_ranks",
    "nodes_from_file",
    "nodes_from_pe_hostfile",
    "scheduler_job",
    "scheduler_kind",
    "scheduler_memory_bytes",
    "scheduler_scratch_dir",
    "visible_device_count",
]
