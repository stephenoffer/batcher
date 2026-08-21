# Batch schedulers and job services

This page covers running Batcher under the thing that allocated your hardware: a batch
scheduler such as Slurm, PBS, LSF, Grid Engine, Flux or HTCondor; a container orchestrator such
as Kubernetes, Nomad or YARN; or a managed job service such as AWS Batch, SageMaker, Vertex AI,
Azure ML or SkyPilot.

There is nothing to configure. Batcher reads what the scheduler already exported into the
process and sizes itself against the *allocation* rather than against the machine. What this
page explains is which facts it reads, why reading them matters, and what to check when a run
looks like it is using a fraction of what you paid for.

## Why the allocation is not the machine

A local probe answers "what hardware is attached to this node". A scheduler answers "what
hardware is this job allowed to use". They are the same number only when the job owns the whole
node, and they diverge silently:

- A Grid Engine job granted 8 of a node's 128 cores sees all 128 in its CPU affinity mask,
  because most sites do not enable cgroup confinement. Sizing thread pools to the mask
  oversubscribes the node sixteen-fold and steals from the co-tenants the scheduler placed
  there.
- A Kubernetes pod granted 2 of a node's 8 GPUs sees 8 through the driver. Sizing an inference
  stage to 8 puts four times the working set on two devices.
- A Slurm allocation of 64 nodes whose job step did not publish a node list reads as one node,
  which turns off every cross-node decision the engine makes.

Batcher reads the grant instead. The core bound is the smallest of the affinity mask, the
cgroup CPU quota, and the scheduler's own grant; the device count is the allocation's, not the
node's; the width of the job comes from the scheduler's node list or node count.

Memory works the same way and matters more, because exceeding it is fatal rather than merely
rude. The ceiling is the smallest of the host's RAM less any reserved hugepages, every cgroup
cap in the ancestry (including the `memory.high` throttle threshold, not just `memory.max`),
the scheduler's memory grant, and `RLIMIT_AS`. That last one is how Grid Engine enforces
`h_vmem`, LSF `-M` and PBS `pvmem`, and it binds hardest of all: overshooting a cgroup gets the
process OOM-killed, while overshooting an address-space limit makes the allocator return NULL
and the query die of `MemoryError` inside a kernel that had no chance to spill instead.

The planner and the admission controller read that one ceiling, so a query that Batcher
predicted would fit is a query the node can actually hold.

## What Batcher reads

Ask, on any node:

```python
import batcher as bt

report = bt.accelerators()
print(sorted(report.get("site", {})))
```

The `site` block names the platform, the scheduler, and the shape the scheduler gave the job.
It is absent entirely on a laptop and in CI, where every field would be empty: a report that
says "provider: unknown, scheduler: none" has told you nothing and cost you a line. Within it,
keys are present only when the scheduler actually published them, so an absent `nodes` key
means "nobody said", not "one node".

| Key | Meaning |
|---|---|
| `provider` | The cloud or platform, from environment markers and then firmware. `unknown` leaves every default where it was. |
| `scheduler` | What launched this process, or `none`. |
| `job_id`, `partition`, `array_index` | The scheduler's own identifiers. `array_index` is what distinguishes otherwise identical sibling tasks, so key a scratch directory or a checkpoint prefix on it. |
| `nodes` | Nodes in the allocation. |
| `gpus_per_node`, `cpus_per_task` | The grant, not the hardware. |
| `tasks`, `tasks_per_node`, `rank` | The job's width and this process's place in it. |

The full list of variables read per scheduler is in
{doc}`/configuration/environment`.

## Ranks come from the launcher

Slurm and Flux number their own tasks. Every other scheduler here hands that job to whatever
started the processes, so Batcher reads the launcher's vocabulary instead: `RANK` /
`WORLD_SIZE` / `LOCAL_RANK` / `LOCAL_WORLD_SIZE` from `torchrun`, `OMPI_COMM_WORLD_*` from Open
MPI, and `PMI_RANK` / `PMI_SIZE` from the PMI family.

This matters more than it looks. Without a rank, every worker in a four-node job believes it is
worker zero, and anything that shards by rank does the same quarter of the work four times
while three quarters is never touched.

## Accelerator grants are read per vendor

The device grant is read from whichever vendor's visibility variable is set, so a pinned pod
reports its share rather than the node's:

| Vendor | Variable |
|---|---|
| NVIDIA | `CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES` |
| AMD | `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES` |
| Intel Gaudi | `HABANA_VISIBLE_MODULES`, `HABANA_VISIBLE_DEVICES` |
| Intel XPU | `ZE_AFFINITY_MASK` |
| AWS Trainium, Inferentia | `NEURON_RT_VISIBLE_CORES` |
| Cloud TPU | `TPU_VISIBLE_DEVICES` |

The NVIDIA container toolkit's `NVIDIA_VISIBLE_DEVICES=all` is its way of saying "inject
everything", which is the *absence* of a narrowing. Batcher reads it as "not narrowed", not as
one device.

## Slurm

A plain `sbatch` script needs no changes:

```bash
# docs: skip
#!/bin/bash
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --time=02:00:00

python my_pipeline.py
```

Two things happen automatically. The core bound becomes 12 rather than the node's core count,
so four co-tenant tasks do not each fan out to the whole node. And `SLURM_JOB_END_TIME` becomes
a drain deadline: workers migrate published shuffle output to a survivor before the allocation
is killed, instead of losing the stage mid-write.

### Ray does not span the allocation by itself

`srun -N 4 python job.py` gives the job four nodes. A bare `ray.init()` starts a *local*,
single-node Ray on whichever node the script landed on. The job then runs, returns the right
answer, and uses a quarter of the hardware it was billed for.

Batcher cannot fix this from inside the process, so it says so: when the allocation is wider
than the Ray cluster — including the case where no cluster is running at all — it logs a
warning naming both counts, once per process. Bringing Ray up across the allocation is the
launcher's job. Pass `distributed=False` if the single node is what you meant, and the notice
stops.

```bash
# docs: skip
# On the first node of the allocation
ray start --head --port=6379
# On the rest
srun --nodes=3 --ntasks-per-node=1 ray start --address="$HEAD:6379"

export RAY_ADDRESS="$HEAD:6379"
python my_pipeline.py
```

## PBS, LSF and Grid Engine

These place several tasks per node by default, and none of them confines a task to its share.
Batcher reads the layout so a per-process budget is one share rather than the whole node:

- **PBS** writes one line per task *slot* to `PBS_NODEFILE`. The distinct names are the
  allocation; the repetition count for this host is the task layout, and the node's core grant
  divided by it is the per-task budget.
- **LSF** publishes a per-host slot breakdown in `LSB_MCPU_HOSTS`. Its job-wide total
  (`LSB_DJOB_NUMPROC`) is deliberately *not* used as a per-node bound, since it over-counts by
  the number of hosts.
- **Grid Engine** writes `hostname nslots queue processors` per line to `PE_HOSTFILE`, so the
  slot count is a column rather than a repeat count.

### Where a spill lands

A GPU or HPC node ships with terabytes of local NVMe, and almost none of it is at `/tmp`. On a
container `/tmp` is an overlay on the root filesystem, so a spill that defaults there dies of
`ENOSPC` beside seven unused terabytes.

Batcher prefers, in order: the directory you named in `memory.spill_dir`; the per-job scratch
directory the scheduler created (`_CONDOR_SCRATCH_DIR`, `SLURM_TMPDIR`, `PBS_JOBFS`); the best
measured node-local volume; the system temp directory. Nothing is chosen from its name alone —
a candidate has to exist, be writable (tested by writing, since `os.access` passes on a
read-only mount), be on a real block device rather than tmpfs or an overlay, and have room.

The scheduler's own directory is preferred because a spill should not outlive its job. It is
also the only directory an HTCondor job is guaranteed to be able to write to.

### Draining before the wall-clock limit

Only Slurm publishes the moment an allocation ends. The others publish a wall-clock *limit*,
which Batcher cannot convert to an instant on its own. Export the lease and the same drain path
turns on:

```bash
# docs: skip
# A two-hour PBS job, in the job script before the run
export BATCHER_DEADLINE_SECONDS=$((2 * 3600))
python my_pipeline.py
```

## Kubernetes

Kubernetes injects nothing about the pod unless the manifest asks. Batcher reports the kind and
whatever identity it can see, which is honest but thin. Adding the downward API is a few lines
and makes placement decisions and diagnostics legible:

```yaml
# docs: skip
env:
  - name: NODE_NAME
    valueFrom: { fieldRef: { fieldPath: spec.nodeName } }
  - name: POD_NAME
    valueFrom: { fieldRef: { fieldPath: metadata.name } }
  - name: POD_NAMESPACE
    valueFrom: { fieldRef: { fieldPath: metadata.namespace } }
```

Volcano and the Kubeflow operators number their pods themselves (`VC_TASK_INDEX`), and Batcher
prefers that index over a launcher variable when both are present.

Three container defaults cost a GPU job real throughput, and Batcher reports each one with the
flag that fixes it. See {doc}`/configuration/environment` for how to read them, and raise
`/dev/shm` (`--shm-size` or a `Memory`-medium `emptyDir`), `memlock` (`--ulimit memlock=-1`),
and `nofile` (`--ulimit nofile=65536`).

## Managed job services

AWS Batch, SageMaker, Vertex AI, Azure ML and SkyPilot each publish the job's shape in their
own form, and Batcher reads all five. Two are worth knowing about because they are the only
place those platforms name the job's *peers*:

- **Vertex AI** publishes `CLUSTER_SPEC`, a JSON document mapping each worker pool to its
  member addresses plus this task's own pool and index.
- **SageMaker** publishes `SM_HOSTS` and `SM_CURRENT_HOST`, so a host's rank is its position in
  the list.

**SkyPilot is read before the platform it provisioned.** A SkyPilot task on a Kubernetes backend
carries both sets of markers, and only SkyPilot knows the task spans four nodes; Kubernetes
knows only about this pod.

## A scheduler Batcher doesn't know

Name it, and the run is labeled without Batcher claiming to know a shape it cannot see:

```python
import os

os.environ["BATCHER_SCHEDULER"] = "housescheduler"
os.environ["BATCHER_PROVIDER"] = "housecloud"
```

Export the pieces it *can* use alongside, and each one turns on the behavior that depends on it:
`BATCHER_DEADLINE_SECONDS` for the drain path, `CUDA_VISIBLE_DEVICES` (or your vendor's
equivalent) for the device grant, and `RANK`/`WORLD_SIZE` for the job's width.

## Requirements and limitations

- Detection is **local reads only** — environment variables, the scheduler's own host files, and
  the firmware's description of the machine in `/sys/class/dmi/id`. Batcher never calls a
  metadata service or a scheduler API, because a network round trip on a control-plane path
  becomes a multi-second hang the moment a firewall blackholes it.
- An unrecognized environment reports `unknown` and `none`, and every default stays exactly
  where it was. Batcher does not guess a shape from a partial signal.
- A scheduler that publishes no per-task grant gets no bound from this path. The cgroup quota
  and affinity mask still apply, which is the real bound for anything running in a container.
- Reading the allocation does not *create* it. Batcher will not start Ray across your nodes,
  request more of them, or extend a lease.

## See also

- {doc}`ray`: how a distributed run actually moves data, and why it bypasses the object store.
- {doc}`/configuration/environment`: every variable read, per scheduler.
- {doc}`/architecture/fault-tolerance`: what draining does with the deadline this page reads.
