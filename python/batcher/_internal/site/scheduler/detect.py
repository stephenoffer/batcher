"""Which scheduler this is, and the two facts a caller wants before the whole job record.

The table below is the whole of detection: each entry names a scheduler, the environment
markers that identify it, and the reader that fills in its job record. Adding a platform is an
entry and a function, and the order of the table is the policy.

**That order is outermost-first, and it is the only judgement call here.** A Ray worker inside
a Slurm allocation reports `slurm`, because the allocation is what bounds the job and what will
end it. A SkyPilot task on a Kubernetes backend reports `skypilot`, because that is the layer
that knows the job spans four nodes while Kubernetes only knows this pod. The rule: the
scheduler that knows the *widest* true shape wins, and where two know the same shape, the one
that can end the job wins.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from batcher._internal.site.scheduler import batch, managed, orchestrator
from batcher._internal.site.scheduler.job import SchedulerJob, env_int, env_str, run_length_min

__all__ = [
    "CPU_GRANT_VARS",
    "GATED_CPU_GRANT_VARS",
    "SCHEDULERS",
    "allocated_cpus",
    "scheduler_job",
    "scheduler_kind",
    "scheduler_memory_bytes",
    "scheduler_scratch_dir",
]

#: The override, checked before every marker. A platform this module has not seen is named
#: here, and it is also how a test pins the answer. Mirrors `BATCHER_PROVIDER`.
_SCHEDULER_OVERRIDE = "BATCHER_SCHEDULER"


@dataclass(frozen=True, slots=True)
class _Scheduler:
    """One scheduler's markers and reader.

    Attributes:
        name: The identifier this scheduler is reported under.
        markers: Environment variables whose presence identifies it. Any one is sufficient.
        require: Variables that must *also* be present, any one of them. Used only where a
            marker is too generic to stand alone — Grid Engine's `JOB_ID` and YARN's
            `CONTAINER_ID` are both names other software sets.
        read: Builds the job record. Called only after the markers matched.
    """

    name: str
    markers: tuple[str, ...]
    read: Callable[[], SchedulerJob]
    require: tuple[str, ...] = ()


#: Every scheduler, outermost first. See the module docstring for why the order is what it is.
SCHEDULERS: tuple[_Scheduler, ...] = (
    _Scheduler("slurm", ("SLURM_JOB_ID",), batch.slurm_job),
    _Scheduler("pbs", ("PBS_JOBID",), batch.pbs_job),
    _Scheduler("lsf", ("LSB_JOBID",), batch.lsf_job),
    # Grid Engine's `JOB_ID` is far too generic to stand alone — CI systems set it — so a
    # Grid Engine-specific variable has to be present as well.
    _Scheduler("sge", ("SGE_TASK_ID", "PE_HOSTFILE", "SGE_O_WORKDIR"), batch.sge_job, ("JOB_ID",)),
    _Scheduler("flux", ("FLUX_JOB_ID",), batch.flux_job),
    _Scheduler(
        "htcondor", ("_CONDOR_SLOT", "_CONDOR_SCRATCH_DIR", "_CONDOR_JOB_AD"), batch.condor_job
    ),
    _Scheduler("aws_batch", ("AWS_BATCH_JOB_ID",), managed.aws_batch_job),
    _Scheduler("sagemaker", ("SM_TRAINING_ENV", "SM_CURRENT_HOST"), managed.sagemaker_job),
    _Scheduler("vertex", ("CLOUD_ML_JOB_ID", "CLUSTER_SPEC"), managed.vertex_job),
    _Scheduler("azureml", ("AZUREML_RUN_ID",), managed.azureml_job),
    _Scheduler(
        "databricks", ("DATABRICKS_RUNTIME_VERSION", "DB_CLUSTER_ID"), orchestrator.databricks_job
    ),
    # Before Kubernetes: a SkyPilot task on a Kubernetes backend carries both, and only
    # SkyPilot knows the task spans several nodes.
    _Scheduler("skypilot", ("SKYPILOT_TASK_ID", "SKYPILOT_NODE_RANK"), managed.skypilot_job),
    _Scheduler("nomad", ("NOMAD_ALLOC_ID",), orchestrator.nomad_job),
    _Scheduler(
        "yarn",
        ("CONTAINER_ID",),
        orchestrator.yarn_job,
        ("NM_HOST", "HADOOP_YARN_HOME", "LOCAL_DIRS"),
    ),
    _Scheduler("kubernetes", (orchestrator.K8S_MARKER,), orchestrator.kubernetes_job),
    _Scheduler("ray", ("RAY_ADDRESS", "RAY_NODE_IP_ADDRESS"), orchestrator.ray_job),
)


def _match() -> _Scheduler | None:
    """The first scheduler in the table whose markers are present, or `None`."""
    for spec in SCHEDULERS:
        if not any(env_str(m) for m in spec.markers):
            continue
        if spec.require and not any(env_str(r) for r in spec.require):
            continue
        return spec
    return None


def scheduler_kind() -> str:
    """What launched this process.

    Returns:
        The scheduler's identifier — `"slurm"`, `"pbs"`, `"lsf"`, `"sge"`, `"flux"`,
        `"htcondor"`, `"aws_batch"`, `"sagemaker"`, `"vertex"`, `"azureml"`, `"databricks"`,
        `"skypilot"`, `"nomad"`, `"yarn"`, `"kubernetes"`, `"ray"` — the value of
        `BATCHER_SCHEDULER` when set, or `"none"`.
    """
    override = env_str(_SCHEDULER_OVERRIDE).lower()
    if override:
        return override
    spec = _match()
    return spec.name if spec else "none"


def scheduler_job() -> SchedulerJob:
    """The job this process belongs to.

    Read live rather than memoized: a process re-execed into a different allocation is rare,
    and the read is a handful of environment lookups, so caching would buy nothing and cost a
    stale answer.

    Returns:
        The job. An unscheduled process (a laptop, a notebook) reports `kind="none"` with
        empty fields, which every caller treats as "decide it yourself, as before". A
        `BATCHER_SCHEDULER` override that names no known scheduler reports that kind with an
        empty record, so an operator can label a platform without claiming a shape for it.
    """
    override = env_str(_SCHEDULER_OVERRIDE).lower()
    spec = _match()
    if override:
        return spec.read() if spec and spec.name == override else SchedulerJob(kind=override)
    return spec.read() if spec else SchedulerJob()


#: Where each scheduler publishes its core grant. Environment only: this is read wherever a
#: thread pool is sized, so it must not cost a file read, and every entry here is a variable
#: rather than a host file.
#:
#: `SLURM_CPUS_PER_TASK` is set when the job asked with `--cpus-per-task`; `SLURM_CPUS_ON_NODE`
#: is the node's whole share of the allocation. `PBS_NCPUS` is PBS' submitted request and
#: `NSLOTS` is Grid Engine's slot grant. Each name here belongs to exactly one scheduler, so
#: its presence is enough.
CPU_GRANT_VARS: tuple[str, ...] = (
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "PBS_NCPUS",
    "NSLOTS",
)

#: Grant variables whose *name* belongs to nobody in particular, with the marker that makes
#: one this scheduler's. `NCPUS` is what PBS sets on the execution host and is the sharper of
#: its two figures — but it is also a name unrelated tooling sets, and this bound narrows
#: every thread pool in the process, so it is believed only inside a PBS job.
GATED_CPU_GRANT_VARS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("NCPUS", ("PBS_JOBID", "PBS_NODEFILE")),
)

#: Schedulers whose per-task grant is not in a plain environment variable, so reading it costs
#: the job record — and a file with it.
#:
#: Both would otherwise be bounded in the *dangerous* direction. PBS publishes `NCPUS`, which is
#: the node's grant, while the per-slot layout is only in the node file: eight tasks on a
#: 96-core node each read 96 and fan out twelve-fold over the node the scheduler placed them on.
#: HTCondor publishes nothing at all about cores; `RequestCpus` is in the ClassAd it writes
#: beside the job, and a pool without cgroup confinement enforces nothing, so a four-core slot
#: reads as the whole machine.
#:
#: Gated on these two schedulers' own markers rather than on `scheduler_kind`, which walks the
#: whole table: measured at 29 us against 4 environment lookups, on a bound that is read
#: wherever a thread pool is sized and therefore several times per query. Asking the question
#: the expensive way made `available_cpu_count` ten times slower on a laptop, where the answer
#: is always "no scheduler".
#:
#: A marker present while some *outer* scheduler won detection is harmless: the record then
#: describes that scheduler, whose own grant variable is already in the list above, so the
#: minimum is unchanged.
_JOB_RECORD_CPU_MARKERS: tuple[str, ...] = (
    "PBS_JOBID",
    "PBS_NODEFILE",
    "_CONDOR_JOB_AD",
    "_CONDOR_SLOT",
)


def allocated_cpus() -> int | None:
    """Cores this process's scheduler allocation granted it on this node, or `None`.

    The bound a container's cgroup does not supply. A container is confined by cgroups, which
    the affinity mask and the CFS quota already report. **A batch allocation is not**, unless
    the site configured cgroup confinement — and plenty of HPC sites do not. There the affinity
    mask reports every core on a shared node, so sizing to it fans a job allocated 8 cores out
    to 128 threads: it oversubscribes the node, steals from the co-tenants the scheduler placed
    there, and at a site with enforcement is exactly what gets the job killed.

    Slurm and Grid Engine publish the grant as a variable; LSF publishes a host-to-slots
    breakdown, whose smallest entry is taken. PBS and HTCondor publish only a *node* figure or
    nothing at all, so for those two — and only those two — the job record is read, which costs
    one file. Every other scheduler in this package runs its work in a container, where the
    cgroup bound is the real one and this correctly reports nothing.

    A variable whose *name* belongs to nobody in particular — `NCPUS` — is read only inside a
    job of the scheduler that sets it. This bound narrows every thread pool in the process, so
    a stray environment variable from unrelated tooling must not be able to halve a query's
    parallelism on a machine with no scheduler at all.

    **The smallest present grant wins, and detection is deliberately not consulted.** A job
    submitted through a compatibility wrapper carries two schedulers' variables at once, and
    the weakest bound is the one that keeps the co-tenants whole — the same reason
    `run_length_min` takes the minimum within a single variable.

    Returns:
        The core grant, or `None` when no scheduler published one — which callers read as "no
        bound from here", never as zero cores.
    """
    names = [
        *CPU_GRANT_VARS,
        *(var for var, markers in GATED_CPU_GRANT_VARS if any(env_str(m) for m in markers)),
    ]
    # The absent case first and cheaply. This runs several times per query on every machine,
    # and almost all of them have no scheduler at all — parsing an empty string five times is
    # most of what the call would otherwise cost there.
    bounds = [n for var in names if (raw := env_str(var)) and (n := run_length_min(raw) or 0) > 0]
    lsf = _lsf_min_slots()
    if lsf:
        bounds.append(lsf)
    if any(env_str(marker) for marker in _JOB_RECORD_CPU_MARKERS):
        share = scheduler_job().cpus_per_task
        if share > 0:
            bounds.append(share)
    return min(bounds) if bounds else None


def _lsf_min_slots() -> int | None:
    """This host's slot count in `LSB_MCPU_HOSTS`, or the smallest in it, or `None`.

    LSF's `LSB_DJOB_NUMPROC` is the *job-wide* slot total, so using it as a per-node bound
    over-counts by the number of hosts — the opposite of what a bound is for. `LSB_MCPU_HOSTS`
    is the per-host breakdown (`"hostA 8 hostB 8"`).

    This host's own entry is preferred, and matching it costs one `gethostname` — a string the
    kernel already holds, read only when the variable is present, so it is free off LSF. The
    minimum is the fallback for a name the breakdown spells differently, and it is the *safe*
    fallback rather than an equivalent one: on a job whose hosts were granted 16 and 8 slots it
    bounds the 16-slot host to 8, which under-parallelizes where the alternative would
    oversubscribe.
    """
    parts = env_str("LSB_MCPU_HOSTS").split()
    counts = [int(c) for c in parts[1::2] if c.isdigit() and int(c) > 0]
    if counts:
        import socket

        host = env_str("HOSTNAME") or socket.gethostname()
        short = host.split(".", 1)[0]
        for name, count in zip(parts[::2], parts[1::2], strict=False):
            if (name == host or name.split(".", 1)[0] == short) and count.isdigit():
                return int(count) or None
        return min(counts)
    # A single-host job publishes no breakdown, and there the job-wide total *is* this host's.
    hosts = {h for h in env_str("LSB_HOSTS").split() if h}
    if len(hosts) == 1:
        return env_int("LSB_DJOB_NUMPROC") or None
    return None


#: Where each scheduler puts the per-job scratch directory it creates and then removes, most
#: specific first. This is the answer an HPC site actually intends: the directory is on the
#: execute node's own disk, it is private to the job, and the scheduler cleans it up — none of
#: which is true of a shared mount that merely happens to be fast.
#:
#: `_CONDOR_SCRATCH_DIR` is stronger still. Under HTCondor it is the *only* directory a job is
#: guaranteed to be able to write to, so spilling anywhere else on a Condor pool is not a
#: slower choice but a failing one.
#:
#: Deliberately **not** `TMPDIR`. Every scheduler sets it, but it means only "somewhere
#: temporary", and on a container it is the root overlay this whole module exists to avoid. It
#: is already the last-resort fallback through `tempfile.gettempdir()`; promoting it to a
#: node-local hint would rank it above a measured NVMe, which is the exact inversion the
#: scratch probe was written to fix.
SCRATCH_DIR_VARS: tuple[str, ...] = (
    "_CONDOR_SCRATCH_DIR",
    "SLURM_TMPDIR",
    "PBS_JOBFS",
)


def scheduler_scratch_dir() -> str:
    """The per-job scratch directory this process's scheduler created, or `""`.

    Returns:
        The first directory named by `SCRATCH_DIR_VARS` that exists, or `""`. Existence only:
        whether it is *usable* — writable, on a real local block device — is `site.scratch`'s
        question, and it asks it of this path the same way it asks it of every other candidate.
    """
    for var in SCRATCH_DIR_VARS:
        path = env_str(var)
        if path and os.path.isdir(path):
            return path
    return ""


#: Where Slurm publishes the allocation's memory grant, in mebibytes. Slurm is the only
#: scheduler here that publishes one into the environment at all: PBS, LSF and Grid Engine
#: enforce theirs through a cgroup or an address-space rlimit, both of which the process can
#: read directly and `site.container.address_space_limit_bytes` already does.
#:
#: `SLURM_MEM_PER_NODE` is the whole node's grant; `SLURM_MEM_PER_CPU` is per allocated core
#: and has to be multiplied by the core grant to become one.
_SLURM_MEM_PER_NODE = "SLURM_MEM_PER_NODE"
_SLURM_MEM_PER_CPU = "SLURM_MEM_PER_CPU"

#: One mebibyte, the unit Slurm publishes these in.
_MIB = 1 << 20


def scheduler_memory_bytes() -> int | None:
    """Memory this process's allocation was granted on this node, or `None`.

    The memory half of `allocated_cpus`, and it fails the same way when unread. A batch
    allocation is not a cgroup unless the site configured confinement, so on an HPC cluster
    without it a job granted 16 GiB on a 512 GiB node *sees* 512 GiB — and every sizing
    decision, from the hash-table budget to the spill threshold, is made against memory the
    scheduler will kill the job for touching.

    Returns:
        The grant in bytes, or `None` when the scheduler published none — which callers read
        as "no bound from here", never as zero memory.
    """
    per_node = env_int(_SLURM_MEM_PER_NODE)
    if per_node > 0:
        return per_node * _MIB
    per_cpu = env_int(_SLURM_MEM_PER_CPU)
    cores = allocated_cpus()
    if per_cpu > 0 and cores:
        return per_cpu * cores * _MIB
    return None
