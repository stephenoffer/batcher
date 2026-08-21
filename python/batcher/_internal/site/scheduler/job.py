"""The record every scheduler reader fills in, and the environment primitives they share.

One dataclass and a handful of parsers, kept apart from the readers so that adding a
scheduler is a table entry plus a function rather than a change to the shape everything
downstream consumes.

Two of the parsers are worth reading before writing a new reader:

* `visible_device_count` is how a *container* publishes its device grant, and it is the only
  per-process device count several schedulers expose at all. It is deliberately vendor-plural
  — a Gaudi pod pinned to two of eight accelerators says so in `HABANA_VISIBLE_MODULES`, a
  Trainium one in `NEURON_RT_VISIBLE_CORES`, and reading only the NVIDIA variable reports the
  whole node to both.
* `launcher_ranks` is the fallback for rank, task count and local rank. Slurm and Flux publish
  their own; every other platform in this package leaves them to whatever launched the
  processes, which is `torchrun`, `mpirun`, or a `PMI`-speaking equivalent. Those three
  vocabularies cover essentially every distributed launch, and without them a Kubernetes
  `PyTorchJob` or an AWS Batch multi-node job reports rank 0 for every one of its workers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from batcher._internal.accelerators import VISIBLE_DEVICE_ENVS

__all__ = [
    "GPU_COUNT_SENTINELS",
    "VISIBLE_DEVICE_COUNT_ENVS",
    "LauncherRanks",
    "SchedulerJob",
    "env_int",
    "env_str",
    "first_env",
    "launcher_ranks",
    "parse_id_list",
    "run_length_min",
    "visible_device_count",
]

#: Values a container runtime writes to mean "not a device list". `all` is the NVIDIA
#: toolkit's default and the reason this parser exists: counting it as one identifier reported
#: a single GPU on every eight-GPU pod that had not been narrowed, which is the common case,
#: and every stage sized against it ran at an eighth of the node.
GPU_COUNT_SENTINELS = frozenset({"all", "void", "none", "-1"})

#: Vendors whose devices `accelerators.VISIBLE_DEVICE_ENVS` does not cover, because that list
#: exists to renumber an *NVML- or AMD-probed* device list and these have no such probe.
#:
#: Plural by vendor on purpose. Batcher runs on NVIDIA, AMD, Intel Gaudi and XPU, AWS Neuron
#: and Cloud TPU, and each publishes its grant under its own name; a reader that knows only the
#: NVIDIA spelling reports the whole node to every other one of them.
_OTHER_VENDOR_VISIBLE_ENVS: tuple[str, ...] = (
    "HABANA_VISIBLE_MODULES",  # Intel Gaudi, module ids
    "HABANA_VISIBLE_DEVICES",  # Intel Gaudi, device ids
    "NEURON_RT_VISIBLE_CORES",  # AWS Trainium / Inferentia
    "TPU_VISIBLE_DEVICES",  # Cloud TPU
    "ZE_AFFINITY_MASK",  # Intel oneAPI Level Zero (XPU)
)

#: Variables that publish *how many* accelerators this process may use, most authoritative
#: first: a framework-level pin (what Slurm, Ray and a Kubernetes device plugin narrow) before
#: the container runtime's coarser injection list.
#:
#: The NVIDIA and AMD half is spliced from `accelerators.VISIBLE_DEVICE_ENVS` rather than
#: restated, because that module says exactly why: "a vendor variable added to one copy and not
#: the others is a silent disagreement about which devices a process owns". This list answers a
#: different question — how many, across every vendor — but it must not answer it from a
#: staler vocabulary than the one that renumbers them.
VISIBLE_DEVICE_COUNT_ENVS: tuple[str, ...] = (
    *VISIBLE_DEVICE_ENVS,
    *_OTHER_VENDOR_VISIBLE_ENVS,
    "NVIDIA_VISIBLE_DEVICES",  # the container runtime's injection list
)


def env_str(name: str) -> str:
    """One environment variable, stripped, or `""` when unset."""
    return os.environ.get(name, "").strip()


def first_env(names: tuple[str, ...]) -> str:
    """The first non-empty value among `names`, or `""`."""
    for name in names:
        value = env_str(name)
        if value:
            return value
    return ""


def env_int(name: str, default: int = 0) -> int:
    """An environment variable as an int, or `default` when absent or unparseable.

    A leading-token parse, so Slurm's `SLURM_NTASKS_PER_NODE="8(x2)"` and PBS' decorated
    counts yield the number rather than the default. A value with no leading digits is
    `default`, which every caller reads as "unpublished".
    """
    raw = env_str(name)
    if not raw:
        return default
    head = raw.split("(", 1)[0].split(",", 1)[0].strip()
    try:
        return int(head)
    except ValueError:
        return default


def run_length_min(raw: str) -> int | None:
    """The smallest count in a scheduler's run-length list, or `None` if unparseable.

    A heterogeneous allocation publishes its per-node grants as a run-length list:
    Slurm's `SLURM_CPUS_ON_NODE="4(x2),8"` means two nodes granted 4 cores and one granted 8.
    Which entry describes *this* node is not derivable from the variable alone, so the minimum
    is taken, for the reason every binding figure here takes the weakest: under-parallelizing
    costs throughput, while over-parallelizing on the node that got the small grant is what
    oversubscribes a shared node and gets the job killed at a site with enforcement.

    Args:
        raw: The variable's value.

    Returns:
        The smallest positive count, or `None` when the shape is not recognized — no bound
        beats a wrong one.
    """
    counts: list[int] = []
    for part in raw.split(","):
        head = part.strip().split("(", 1)[0].strip()
        if not head.isdigit():
            return None
        value = int(head)
        if value > 0:
            counts.append(value)
    return min(counts) if counts else None


def parse_id_list(raw: str) -> int:
    """How many identifiers a device-visibility list names, or `-1` for "not a list".

    Handles the three spellings these variables actually use: a comma list (`0,1,2`), an
    inclusive range (`0-3`, which `NEURON_RT_VISIBLE_CORES` writes), and Level Zero's
    sub-device dotted form (`0.1`), where each entry is still one addressable device.

    Returns:
        The count, `0` for an explicitly empty grant, or `-1` when the value is a sentinel
        such as `all` and therefore says nothing about how many devices there are.
    """
    value = raw.strip()
    if value.lower() in GPU_COUNT_SENTINELS:
        return -1
    if not value:
        return 0
    total = 0
    for token in value.split(","):
        part = token.strip()
        if not part or part.lower() in GPU_COUNT_SENTINELS:
            continue
        low, sep, high = part.partition("-")
        if sep and low.strip().isdigit() and high.strip().isdigit():
            start, end = int(low), int(high)
            if end >= start:
                total += end - start + 1
                continue
        total += 1
    return total


def visible_device_count() -> int:
    """Accelerators this process may use, as the environment publishes it.

    The only per-process device count several schedulers expose: a Kubernetes device plugin,
    an LSF GPU allocation and an HTCondor GPU assignment all reach the process as a
    visibility list rather than as a count.

    Returns:
        The number of devices named, or `0` when nothing definite is published — which
        includes the container runtime's `all`, since that is the *absence* of a narrowing
        rather than a grant of one device.
    """
    for name in VISIBLE_DEVICE_COUNT_ENVS:
        raw = os.environ.get(name)
        if raw is None:
            continue
        count = parse_id_list(raw)
        if count >= 0:
            return count
    return 0


@dataclass(frozen=True, slots=True)
class LauncherRanks:
    """Rank, world size and node-local placement, as the process launcher published them.

    Attributes:
        rank: Global index of this process, `0` when unpublished.
        tasks: Processes in the job, `0` when unpublished.
        local_rank: Index of this process among those on its own node, `0` when unpublished.
        local_size: Processes on this node, `0` when unpublished.
    """

    rank: int = 0
    tasks: int = 0
    local_rank: int = 0
    local_size: int = 0


#: The three launcher vocabularies, in the order they are trusted. `torchrun`'s plain names
#: come first because they are what a Kubernetes `PyTorchJob`, a Vertex AI worker pool and a
#: SageMaker job all set; Open MPI and the PMI family follow and cover `mpirun`, Intel MPI,
#: MPICH and the MPI Operator.
_RANK_VARS = ("RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK")
_TASKS_VARS = ("WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE")
_LOCAL_RANK_VARS = ("LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID")
_LOCAL_SIZE_VARS = ("LOCAL_WORLD_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE", "MPI_LOCALNRANKS")


def _first_int(names: tuple[str, ...]) -> int:
    """The first parseable int among `names`, or `0`."""
    for name in names:
        if env_str(name):
            return env_int(name)
    return 0


def launcher_ranks() -> LauncherRanks:
    """What `torchrun`, `mpirun` or a PMI launcher says about this process's place in the job.

    The fallback for every scheduler that hands its workers off to a launcher instead of
    numbering them itself, which is most of them outside HPC. Without it a four-node
    Kubernetes job reports rank 0 four times, and anything that shards work by rank does the
    same quarter of it on every node.

    Returns:
        The ranks. Every field is `0` when nothing published it, which callers read as
        "single process" — the same answer as before this existed.
    """
    return LauncherRanks(
        rank=_first_int(_RANK_VARS),
        tasks=_first_int(_TASKS_VARS),
        local_rank=_first_int(_LOCAL_RANK_VARS),
        local_size=_first_int(_LOCAL_SIZE_VARS),
    )


@dataclass(frozen=True, slots=True)
class SchedulerJob:
    """The job this process belongs to, as its scheduler describes it.

    Attributes:
        kind: The scheduler's identifier, or `"none"` when nothing scheduled this process.
        job_id: The scheduler's job identifier, `""` when there is none.
        nodes: Node names in the allocation, in the scheduler's own order. Empty under a
            scheduler that does not publish the list.
        num_nodes: Nodes in the allocation when the scheduler publishes a *count* but not a
            list, `0` otherwise. Read through `node_count`, never directly.
        gpus_per_node: Devices *this allocation* was given per node, `0` when unpublished.
            Distinct from the node's physical device count, which is what a local probe sees
            and which over-counts whenever a node is shared.
        cpus_per_node: Cores the allocation was given per node, `0` when unpublished.
        cpus_per_task: Cores granted to *this process*, `0` when unpublished. The figure a
            thread pool should be sized against on a node running several tasks.
        tasks: Total tasks in the job, `0` when unpublished.
        tasks_per_node: Tasks placed on each node, `0` when unpublished.
        rank: This process's global task index, `0` when unpublished or single-task.
        local_rank: This process's index among the tasks on its own node.
        local_size: Tasks sharing this node, `0` when unpublished. What says whether this
            process owns the node's memory and devices or one share of them.
        node_name: The node this process is on, `""` when unpublished.
        partition: The scheduler's queue or partition — a Slurm partition, a PBS or LSF
            queue, a Kubernetes namespace — `""` when unpublished.
        array_index: This task's index within an array job, `""` when not an array job. The
            one identifier that distinguishes otherwise identical sibling tasks, so it is
            what a scratch directory or a checkpoint prefix should be keyed on.
    """

    kind: str = "none"
    job_id: str = ""
    nodes: tuple[str, ...] = field(default_factory=tuple)
    num_nodes: int = 0
    gpus_per_node: int = 0
    cpus_per_node: int = 0
    cpus_per_task: int = 0
    tasks: int = 0
    tasks_per_node: int = 0
    rank: int = 0
    local_rank: int = 0
    local_size: int = 0
    node_name: str = ""
    partition: str = ""
    array_index: str = ""

    @property
    def node_count(self) -> int:
        """Nodes in the allocation, from whichever of the list and the count is larger.

        A scheduler that publishes only a count is not a single-node job, and reading
        `len(nodes)` treated it as one — AWS Batch, SkyPilot and Azure ML all land there.
        The *larger* rather than a preference between them, because a node list can be
        partial while a count cannot: a Kubernetes pod names the one node it landed on and
        learns from its launcher that the job spans sixteen, and taking the list would have
        called that job single-node and turned off every cross-node decision the engine makes.
        """
        return max(len(self.nodes), self.num_nodes)

    @property
    def multi_node(self) -> bool:
        """Whether the allocation spans more than one node."""
        return self.node_count > 1

    @property
    def total_gpus(self) -> int:
        """Devices across the whole allocation, `0` when either factor is unpublished.

        The number a distributed stage's parallelism should be sized against, and the one an
        operator means by "how big is this job".
        """
        return self.gpus_per_node * self.node_count

    @property
    def shares_node(self) -> bool:
        """Whether other tasks of this job run on this node too.

        The question behind every per-process budget: a process that owns its node may size
        against the node's memory and cores, and one of eight siblings may not.
        """
        return self.local_size > 1 or self.tasks_per_node > 1
