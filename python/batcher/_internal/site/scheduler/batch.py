"""The batch schedulers an HPC or research GPU cluster runs under.

Slurm, PBS/OpenPBS, LSF, Grid Engine, Flux and HTCondor. Between them they schedule most of
the capacity that is not Kubernetes, and they publish more about a job's shape than anything
else in this package: the node list, the per-node device and core grants, and this process's
rank within the job all arrive in the environment before any of it could be probed.

Two facts these readers exist to preserve, both of which cost real throughput when lost:

* **The grant is not the hardware.** A node's physical device and core counts are what a local
  probe reports, and they are wrong the moment two jobs share a node. Every figure here is
  what the *allocation* was given.
* **A shared node is the norm, not the exception.** Grid Engine and LSF place several tasks
  per node by default, and PBS and Slurm do it whenever a job asks for fewer resources than a
  node holds. `cpus_per_task` and `local_size` are what let a per-process budget be one share
  rather than the whole node.
"""

from __future__ import annotations

import socket

from batcher._internal.site.scheduler.hostlist import (
    expand_nodelist,
    nodes_from_file,
    nodes_from_pe_hostfile,
)
from batcher._internal.site.scheduler.job import (
    SchedulerJob,
    env_int,
    env_str,
    first_env,
    launcher_ranks,
    parse_id_list,
    run_length_min,
    visible_device_count,
)

__all__ = [
    "condor_job",
    "flux_job",
    "lsf_job",
    "pbs_job",
    "sge_job",
    "slurm_job",
]


def _this_host(nodes: tuple[str, ...]) -> str:
    """The name of the node this process is on, preferring one the allocation also names.

    A local call, never a lookup: `gethostname` reads a string the kernel already holds. The
    allocation's own spelling is preferred where it matches, because a scheduler's node names
    and a host's FQDN frequently differ by a domain suffix and downstream code compares them.
    """
    host = env_str("HOSTNAME") or socket.gethostname()
    short = host.split(".", 1)[0]
    for name in nodes:
        if name == host or name.split(".", 1)[0] == short:
            return name
    return host


def _gpu_grant(names: tuple[str, ...]) -> int:
    """The allocation's per-node device count from the first variable that names one.

    Falls through to the container's visibility list, which is what LSF, Grid Engine and
    HTCondor leave the count in. A value such as Slurm's `gpu:8` is read as its trailing
    count, since the resource name carries no number.
    """
    for name in names:
        raw = env_str(name)
        if not raw:
            continue
        tail = raw.rsplit(":", 1)[-1]
        if tail.isdigit():
            return int(tail)
        count = parse_id_list(raw)
        if count > 0:
            return count
    return visible_device_count()


# --- Slurm ----------------------------------------------------------------------------------


def slurm_job() -> SchedulerJob:
    """The allocation this process belongs to, from Slurm's environment."""
    nodes = expand_nodelist(
        first_env(("SLURM_JOB_NODELIST", "SLURM_NODELIST", "SLURM_STEP_NODELIST"))
    )
    cpus_on_node = run_length_min(env_str("SLURM_CPUS_ON_NODE")) or 0
    return SchedulerJob(
        kind="slurm",
        job_id=env_str("SLURM_JOB_ID"),
        nodes=nodes,
        # A job step that ran without a node list still knows how wide it is, and a
        # single-node reading of a 64-node allocation is the failure this guards.
        num_nodes=env_int("SLURM_JOB_NUM_NODES") or env_int("SLURM_NNODES"),
        # `SLURM_GPUS_ON_NODE` is the allocation's grant; `SLURM_GPUS_PER_NODE` is the request
        # in `gpu:8` form, and `SLURM_JOB_GPUS` is the device id list, which older Slurm sets
        # instead. All three are the allocation rather than the node's hardware.
        gpus_per_node=_gpu_grant(("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE", "SLURM_JOB_GPUS")),
        cpus_per_node=cpus_on_node,
        cpus_per_task=env_int("SLURM_CPUS_PER_TASK"),
        tasks=env_int("SLURM_NTASKS"),
        tasks_per_node=env_int("SLURM_NTASKS_PER_NODE"),
        rank=env_int("SLURM_PROCID"),
        local_rank=env_int("SLURM_LOCALID"),
        # Slurm publishes no per-node task count directly on the step, so the request is the
        # best available answer and the launcher's is the fallback under `srun --overlap`.
        local_size=env_int("SLURM_NTASKS_PER_NODE") or launcher_ranks().local_size,
        node_name=env_str("SLURMD_NODENAME") or _this_host(nodes),
        partition=env_str("SLURM_JOB_PARTITION"),
        array_index=env_str("SLURM_ARRAY_TASK_ID"),
    )


# --- PBS / OpenPBS / Torque -----------------------------------------------------------------


def pbs_job() -> SchedulerJob:
    """The allocation this process belongs to, from PBS/OpenPBS.

    PBS puts its node list in a *file* rather than an environment variable, which is the one
    structural difference from Slurm worth handling: `PBS_NODEFILE` names it, and the file is
    one line per task slot rather than one per node.
    """
    nodefile = env_str("PBS_NODEFILE")
    lines = _slot_lines(nodefile) if nodefile else ()
    nodes = tuple(dict.fromkeys(lines))
    slots = env_int("PBS_NUM_PPN") or _slots_here(nodes, lines)
    cpus = env_int("NCPUS") or env_int("PBS_NCPUS")
    return SchedulerJob(
        kind="pbs",
        job_id=env_str("PBS_JOBID"),
        nodes=nodes,
        num_nodes=env_int("PBS_NUM_NODES"),
        gpus_per_node=env_int("PBS_NGPUS") or visible_device_count(),
        # `NCPUS` is what PBS sets on the *execution* host; `PBS_NCPUS` is the request. Both
        # are the grant rather than the node, and the exec-host value is the sharper one.
        cpus_per_node=cpus,
        # PBS publishes no per-task core grant, so it is the node's share divided by the task
        # layout the node file describes. That division is the whole point of reading the file:
        # eight tasks on a 96-core node get 12 cores each, and sizing all eight to 96 is what
        # oversubscribes the node twelve-fold.
        cpus_per_task=cpus // slots if cpus and slots > 1 else 0,
        tasks=env_int("PBS_NP"),
        tasks_per_node=slots,
        rank=env_int("PBS_VNODENUM") or launcher_ranks().rank,
        local_rank=launcher_ranks().local_rank,
        local_size=slots or launcher_ranks().local_size,
        # `PBS_NODENUM` is this node's *index* in the allocation, not its name, and reading it
        # as one put the string "0" where every caller expected a hostname.
        node_name=_this_host(nodes),
        partition=env_str("PBS_QUEUE"),
        array_index=env_str("PBS_ARRAY_INDEX"),
    )


def _slot_lines(path: str) -> tuple[str, ...]:
    """Every line of a scheduler's host file, in order, `()` when it cannot be read.

    One read serves both questions the file answers — which nodes the allocation holds, and
    how many slots this one got — because `scheduler_job` is called per diagnostic and reading
    it twice was two opens for one file.
    """
    try:
        with open(path) as f:
            return tuple(line.strip() for line in f if line.strip())
    except OSError:
        return ()


def _slots_here(nodes: tuple[str, ...], lines: tuple[str, ...]) -> int:
    """Task slots this node was given, counted as its repetitions in the host file.

    PBS' one-line-per-slot format is the only place the per-node task layout appears, and
    without it a node running eight tasks looks like a node running one.
    """
    if not lines:
        return 0
    host = _this_host(nodes)
    short = host.split(".", 1)[0]
    return sum(1 for line in lines if line == host or line.split(".", 1)[0] == short)


# --- LSF ------------------------------------------------------------------------------------


def lsf_job() -> SchedulerJob:
    """The allocation this process belongs to, from LSF.

    LSF lists its hosts inline in `LSB_HOSTS`, space-separated and repeated per slot, or in a
    file named by `LSB_DJOB_HOSTFILE` on a large job where the variable would overflow. Both
    are read, the file first, because that is the one that stays correct at scale.
    """
    hostfile = env_str("LSB_DJOB_HOSTFILE")
    slot_hosts = tuple(env_str("LSB_HOSTS").split())
    nodes = nodes_from_file(hostfile) if hostfile else ()
    if not nodes:
        nodes = tuple(dict.fromkeys(slot_hosts))
    node_name = _this_host(nodes)
    ranks = launcher_ranks()
    return SchedulerJob(
        kind="lsf",
        job_id=env_str("LSB_JOBID"),
        nodes=nodes,
        gpus_per_node=visible_device_count(),
        cpus_per_node=_lsf_slots_here(node_name, slot_hosts),
        tasks=env_int("LSB_DJOB_NUMPROC"),
        tasks_per_node=_lsf_slots_here(node_name, slot_hosts),
        rank=ranks.rank,
        local_rank=ranks.local_rank,
        local_size=ranks.local_size or _lsf_slots_here(node_name, slot_hosts),
        node_name=node_name,
        partition=env_str("LSB_QUEUE"),
        array_index=env_str("LSB_JOBINDEX"),
    )


def _lsf_slots_here(node_name: str, slot_hosts: tuple[str, ...]) -> int:
    """Slots LSF gave this host, from `LSB_MCPU_HOSTS` or the repetitions in `LSB_HOSTS`.

    `LSB_MCPU_HOSTS` is the authoritative form — `"host1 4 host2 8"` — and is what a job
    spanning hosts with different slot counts is described by. The inline list is the
    fallback, where the repetition count is the same number.
    """
    mcpu = env_str("LSB_MCPU_HOSTS").split()
    short = node_name.split(".", 1)[0]
    for host, count in zip(mcpu[::2], mcpu[1::2], strict=False):
        if count.isdigit() and (host == node_name or host.split(".", 1)[0] == short):
            return int(count)
    hits = sum(1 for h in slot_hosts if h == node_name or h.split(".", 1)[0] == short)
    return hits


# --- Grid Engine (SGE / UGE / Altair Grid Engine) -------------------------------------------


def sge_job() -> SchedulerJob:
    """The allocation this process belongs to, from Grid Engine.

    Grid Engine's parallel-environment host file is the one that is not a bare host list:
    `hostname nslots queue processors` per line, so the per-node slot count is a column. A
    serial job has no host file at all and is described entirely by `NSLOTS`.
    """
    pe_hostfile = env_str("PE_HOSTFILE")
    nodes, slots = nodes_from_pe_hostfile(pe_hostfile) if pe_hostfile else ((), 0)
    ranks = launcher_ranks()
    node_name = env_str("HOSTNAME") or _this_host(nodes)
    return SchedulerJob(
        kind="sge",
        job_id=env_str("JOB_ID"),
        nodes=nodes,
        gpus_per_node=visible_device_count(),
        # `NSLOTS` is this task's slot grant on this node; the host file's summed column is
        # the job-wide total and is carried as `tasks`.
        cpus_per_node=env_int("NSLOTS"),
        cpus_per_task=env_int("NSLOTS"),
        tasks=slots or env_int("NSLOTS"),
        rank=ranks.rank,
        local_rank=ranks.local_rank,
        local_size=ranks.local_size,
        node_name=node_name,
        partition=env_str("QUEUE"),
        array_index=env_str("SGE_TASK_ID"),
    )


# --- Flux -----------------------------------------------------------------------------------


def flux_job() -> SchedulerJob:
    """The allocation this process belongs to, from Flux.

    Flux is the scheduler on several of the largest GPU systems, and it is the one that
    numbers its own tasks: `FLUX_TASK_RANK` and `FLUX_TASK_LOCAL_ID` are published directly
    rather than left to a launcher.
    """
    nodes = expand_nodelist(env_str("FLUX_JOB_NODELIST"))
    size = env_int("FLUX_JOB_SIZE")
    nnodes = env_int("FLUX_JOB_NNODES")
    return SchedulerJob(
        kind="flux",
        job_id=env_str("FLUX_JOB_ID"),
        nodes=nodes,
        num_nodes=nnodes,
        gpus_per_node=visible_device_count(),
        tasks=size,
        tasks_per_node=size // nnodes if size and nnodes else 0,
        rank=env_int("FLUX_TASK_RANK"),
        local_rank=env_int("FLUX_TASK_LOCAL_ID"),
        local_size=size // nnodes if size and nnodes else 0,
        node_name=_this_host(nodes),
    )


# --- HTCondor -------------------------------------------------------------------------------

#: ClassAd attributes worth lifting out of the ad files HTCondor writes beside a job. Read
#: from the file because HTCondor publishes almost nothing else into the environment: the slot
#: name is there, the job's own identity is not.
_CONDOR_JOB_ATTRS = ("ClusterId", "ProcId", "RequestCpus", "RequestGpus")


def condor_job() -> SchedulerJob:
    """The slot this process runs in, from HTCondor.

    HTCondor is the scheduler for high-throughput fleets — grid capacity, opportunistic
    cycles, and most of what a university runs — and its jobs are the most likely of any here
    to be preempted, so knowing the shape matters even though a slot is usually one node.

    The identity comes out of the job ad rather than the environment, since HTCondor writes
    the ad file beside the job and exports only the slot name. That is a local file read, the
    same cost PBS and LSF already pay for their host files.
    """
    ad = _read_classad(env_str("_CONDOR_JOB_AD"))
    cluster, proc = ad.get("ClusterId", ""), ad.get("ProcId", "")
    gpus = parse_id_list(env_str("_CONDOR_ASSIGNED_GPUS"))
    return SchedulerJob(
        kind="htcondor",
        job_id=f"{cluster}.{proc}" if cluster else cluster,
        gpus_per_node=(gpus if gpus > 0 else 0)
        or _int_attr(ad, "RequestGpus")
        or visible_device_count(),
        cpus_per_node=_int_attr(ad, "RequestCpus"),
        # A slot is what this process was given, so the request *is* the per-task grant. It is
        # the only core figure HTCondor publishes anywhere, and a pool without cgroup
        # confinement enforces nothing — so without it a four-core slot sizes to the machine.
        cpus_per_task=_int_attr(ad, "RequestCpus"),
        rank=env_int("_CONDOR_PROCNO") or launcher_ranks().rank,
        tasks=env_int("_CONDOR_NPROCS") or launcher_ranks().tasks,
        node_name=_this_host(()),
        partition=env_str("_CONDOR_SLOT"),
        array_index=proc,
    )


def _int_attr(ad: dict[str, str], name: str) -> int:
    """One ClassAd attribute as a non-negative int, or `0` when absent or not a number."""
    raw = ad.get(name, "")
    return int(raw) if raw.isdigit() else 0


def _read_classad(path: str) -> dict[str, str]:
    """The `Attribute = value` pairs this reader wants out of an HTCondor ClassAd file.

    Only the handful in `_CONDOR_JOB_ATTRS`, and only their literal text: a job ad is large
    and its value grammar is an expression language, so parsing it properly here would be a
    dependency rather than a lookup. Quotes are stripped, since string values carry them.
    """
    if not path:
        return {}
    out: dict[str, str] = {}
    try:
        with open(path) as f:
            for line in f:
                key, sep, value = line.partition("=")
                if not sep:
                    continue
                name = key.strip()
                if name in _CONDOR_JOB_ATTRS:
                    out[name] = value.strip().strip('"')
    except OSError:
        return out
    return out
