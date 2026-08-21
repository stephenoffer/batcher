# Environment variables

Batcher reads two environment-driven config layers once when the package is
imported: `BATCHER_*` variables and an optional JSON file at `BATCHER_CONFIG_FILE`.
Both overlay onto the built-in defaults and can be reproduced explicitly with
{py:meth}`Config.from_env <batcher.Config.from_env>` and {py:meth}`Config.from_file <batcher.Config.from_file>`.

## BATCHER_ variables

Each variable maps to one config field by path:
`BATCHER_<SECTION>_<FIELD>`. Nested sub-sections compose by appending another
segment. Names are uppercased section and field names.

| Variable | Field |
|----------|-------|
| `BATCHER_EXECUTION_PARALLELISM` | `config.execution.parallelism` |
| `BATCHER_EXECUTION_MORSEL_ROWS` | `config.execution.morsel_rows` |
| `BATCHER_MEMORY_SOFT_LIMIT` | `config.memory.soft_limit` |
| `BATCHER_MEMORY_MAX_MEMORY_BYTES` | `config.memory.max_memory_bytes` |
| `BATCHER_FLOW_CONTROL_DEFAULT_CREDITS` | `config.flow_control.default_credits` |
| `BATCHER_OPTIMIZER_REOPTIMIZE_ERROR` | `config.optimizer.reoptimize_error` |
| `BATCHER_OPTIMIZER_CARDINALITY_EQ_SELECTIVITY` | `config.optimizer.cardinality.eq_selectivity` |
| `BATCHER_PID_KP` | `config.pid.kp` |
| `BATCHER_METADATA_BACKEND` | `config.metadata.backend` |

Values are coerced to the field's type. Integers and floats are parsed directly. Booleans accept `1`, `true`, `yes`, or `on`, case-insensitively, as true.

```bash
# docs: skip
export BATCHER_EXECUTION_PARALLELISM=8
export BATCHER_MEMORY_SOFT_LIMIT=0.75
python my_pipeline.py
```

You can reproduce the overlay in code by passing an explicit environment mapping.
`Config.from_env` returns a new {py:class}`Config <batcher.Config>` and does not mutate its base.

```python
from batcher import Config

cfg = Config.from_env(
    {"BATCHER_EXECUTION_PARALLELISM": "8", "BATCHER_MEMORY_SOFT_LIMIT": "0.75"}
)
print((cfg.execution.parallelism, cfg.memory.soft_limit))
# (8, 0.75)
```

A nested sub-section field composes its path the same way:

```python
from batcher import Config

cfg = Config.from_env({"BATCHER_OPTIMIZER_CARDINALITY_EQ_SELECTIVITY": "0.05"})
print(cfg.optimizer.cardinality.eq_selectivity)
# 0.05
```

## BATCHER_CONFIG_FILE

Set `BATCHER_CONFIG_FILE` to the path of a JSON document whose structure mirrors the
section layout. It is overlaid below the `BATCHER_*` variables.

```bash
# docs: skip
export BATCHER_CONFIG_FILE=/etc/batcher/config.json
python my_pipeline.py
```

```json
{
  "execution": { "morsel_rows": 4096, "parallelism": 4 },
  "memory": { "soft_limit": 0.80 },
  "optimizer": { "cardinality": { "eq_selectivity": 0.05 } }
}
```

`Config.from_file` applies the same overlay programmatically:

```python
# docs: skip
from batcher import Config

cfg = Config.from_file("/etc/batcher/config.json")
```

## Environment detection

A second, smaller group of variables doesn't name a config field. Batcher reads them to
work out what kind of machine it's running on, then picks defaults to match. Each is read
from the environment only, never from a metadata service on a hot path, so detection
costs nothing and can't hang.

Set one of these when Batcher can't see a signal your deployment does have:

| Variable | Effect |
|---|---|
| `BATCHER_SPOT` | Truthy marks the node preemptible, selecting the `"spot"` resilience profile. |
| `BATCHER_AUTOSCALE` | Authoritative in both directions: truthy forces the bounded autoscale wait on, falsey forces it off even on a managed cluster. |
| `BATCHER_RAY_CLUSTER` | Any non-empty value attaches to the running Ray cluster instead of starting a local one. |
| `BATCHER_DEADLINE_EPOCH_S` | The Unix time this process will be killed at. Makes the job preemptible and starts the drain `distributed.drain_lead_s` seconds ahead. |
| `BATCHER_DEADLINE_SECONDS` | The same lease as a duration measured from when the process started, for a scheduler that publishes a wall-clock limit rather than an end time. Every scheduler except Slurm is in that group. |
| `BATCHER_PROVIDER` | The platform name, for a cloud whose markers Batcher doesn't recognize. Selects that platform's scratch and object-store defaults. |
| `BATCHER_SCHEDULER` | The scheduler name, for a launcher Batcher doesn't recognize. Labels the run without claiming to know the job's shape. |
| `BATCHER_METADATA_URI` | An fsspec URL for learned statistics, so cross-run learning survives a driver that moves between nodes. |
| `BATCHER_SHUFFLE_PORT_RANGE` | A `40000-40100` range the shuffle listeners must bind inside, to match a firewall rule. |
| `BATCHER_SHUFFLE_TOKEN` | The shared secret fencing shuffle tickets, injectable without a config file. |

Batcher also reads variables it doesn't own, exported by the scheduler that launched the
job. You don't set these; they're listed so you can tell what Batcher already knows:

| Variable | Read for |
|---|---|
| `SLURM_JOB_END_TIME` | When the allocation ends, so workers drain before the kill. Slurm's unlimited-job sentinel is ignored. |
| `SLURM_CPUS_PER_TASK`, `SLURM_CPUS_ON_NODE`, `PBS_NCPUS`, `NSLOTS`, `LSB_MCPU_HOSTS` | The cores this job was granted, capping fan-out on a node without cgroup confinement. The smallest grant present binds. `NCPUS` counts too, but only inside a PBS job: the name is one unrelated tooling also sets. |
| `SLURM_MEM_PER_NODE`, `SLURM_MEM_PER_CPU` | The memory this job was granted, capping the working-set budget on a node without cgroup confinement. |
| `RAY_CLUSTER_NAME`, `RAY_CLUSTER_NAMESPACE` | A KubeRay-operated cluster, which implies an autoscaler. |
| `ANYSCALE_SESSION_ID`, `ANYSCALE_ARTIFACT_STORAGE` | A managed control plane, and durable storage for learned statistics. |

The job's *shape* comes from the same place. Batcher reads whichever of these the scheduler
that launched the job exports, and uses them to size stages against the allocation rather than
against the node's physical hardware, which over-counts the moment two jobs share a node:

| Scheduler | Read from |
|---|---|
| Slurm | `SLURM_JOB_NODELIST`, `SLURM_JOB_NUM_NODES`, `SLURM_GPUS_ON_NODE`, `SLURM_GPUS_PER_NODE`, `SLURM_JOB_GPUS`, `SLURM_NTASKS`, `SLURM_NTASKS_PER_NODE`, `SLURM_PROCID`, `SLURM_LOCALID`, `SLURM_ARRAY_TASK_ID` |
| PBS, OpenPBS, Torque | `PBS_JOBID`, `PBS_NODEFILE`, `PBS_NGPUS`, `PBS_NP`, `PBS_NUM_PPN`, `PBS_QUEUE`, `PBS_ARRAY_INDEX` |
| LSF | `LSB_JOBID`, `LSB_DJOB_HOSTFILE`, `LSB_HOSTS`, `LSB_MCPU_HOSTS`, `LSB_QUEUE`, `LSB_JOBINDEX` |
| Grid Engine (SGE, UGE) | `JOB_ID`, `PE_HOSTFILE`, `NSLOTS`, `QUEUE`, `SGE_TASK_ID` |
| Flux | `FLUX_JOB_ID`, `FLUX_JOB_SIZE`, `FLUX_JOB_NNODES`, `FLUX_TASK_RANK`, `FLUX_TASK_LOCAL_ID` |
| HTCondor | `_CONDOR_SLOT`, `_CONDOR_JOB_AD`, `_CONDOR_ASSIGNED_GPUS`, `_CONDOR_PROCNO` |
| Kubernetes | `KUBERNETES_SERVICE_HOST`, `NODE_NAME`, `POD_NAME`, `POD_NAMESPACE`, `JOB_COMPLETION_INDEX`, `VC_TASK_INDEX` |
| AWS Batch | `AWS_BATCH_JOB_ID`, `AWS_BATCH_JOB_NUM_NODES`, `AWS_BATCH_JOB_NODE_INDEX`, `AWS_BATCH_JQ_NAME` |
| SageMaker | `SM_HOSTS`, `SM_CURRENT_HOST`, `SM_NUM_GPUS`, `SM_NUM_CPUS`, `TRAINING_JOB_NAME` |
| Vertex AI | `CLUSTER_SPEC`, `CLOUD_ML_JOB_ID` |
| Azure ML | `AZUREML_RUN_ID`, `AZUREML_ARM_WORKSPACE_NAME` |
| SkyPilot | `SKYPILOT_TASK_ID`, `SKYPILOT_NUM_NODES`, `SKYPILOT_NODE_RANK`, `SKYPILOT_NODE_IPS`, `SKYPILOT_NUM_GPUS_PER_NODE` |
| Nomad | `NOMAD_ALLOC_ID`, `NOMAD_JOB_ID`, `NOMAD_ALLOC_INDEX`, `NOMAD_NAMESPACE` |
| YARN | `CONTAINER_ID`, `NM_HOST` |
| Databricks | `DB_CLUSTER_ID`, `DB_IS_DRIVER`, `DATABRICKS_RUNTIME_VERSION` |
| Ray | `RAY_ADDRESS`, `RAY_JOB_ID`, `RAY_NODE_IP_ADDRESS` |

A spill prefers the per-job scratch directory the scheduler created, when there is one:
`_CONDOR_SCRATCH_DIR`, `SLURM_TMPDIR`, or `PBS_JOBFS`. Those are on the execute node's own
disk, private to the job, and removed when it ends, so a spill can't outlive the job and fill
a shared mount for the next tenant. The directory is still measured like any other candidate,
so one that turns out to be tmpfs or a network mount is refused. `TMPDIR` is deliberately not
in that list: every scheduler sets it, it means only "somewhere temporary", and on a container
it is the root overlay the scratch probe exists to avoid.

Where a scheduler leaves the job's rank and width to the process launcher, Batcher reads the
launcher instead: `RANK`, `WORLD_SIZE`, `LOCAL_RANK` and `LOCAL_WORLD_SIZE` from `torchrun`,
`OMPI_COMM_WORLD_*` from Open MPI, and `PMI_RANK`/`PMI_SIZE` from the PMI family. Without
them a four-node job reports rank 0 on every node.

Memory has one more bound that isn't an environment variable: `RLIMIT_AS`, the address-space
limit. That is how Grid Engine enforces `h_vmem`, LSF enforces `-M`, PBS enforces `pvmem`, and
what a plain `ulimit -v` sets, so on an HPC cluster it is frequently the only bound on a job's
memory. It binds harder than a cgroup does. Overshooting a cgroup gets the process OOM-killed;
overshooting an address-space limit makes the allocator return NULL and the query die of
`MemoryError` inside a kernel that had no chance to spill instead.

The accelerator grant is read from whichever vendor's visibility variable is set:
`CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`,
`HABANA_VISIBLE_MODULES`, `HABANA_VISIBLE_DEVICES`, `NEURON_RT_VISIBLE_CORES`,
`TPU_VISIBLE_DEVICES`, `ZE_AFFINITY_MASK`, or `NVIDIA_VISIBLE_DEVICES`. The container
runtime's `all` is read as "not narrowed", not as one device.

## Precedence

The two layers here sit in the middle of the resolution order, highest first:

1. {py:func}`config_context(...) <batcher.config_context>`.
1. {py:func}`set_config(...) <batcher.set_config>`.
1. `BATCHER_*` environment variables.
1. `BATCHER_CONFIG_FILE` JSON.
1. Built-in defaults.

So a `BATCHER_*` variable overrides a value set in `BATCHER_CONFIG_FILE`, and a
runtime `set_config` or `config_context` overrides both.

## See also

- {doc}`index`: the runtime entry points these variables are overridden by.
- {doc}`options`: every field a `BATCHER_*` variable can name, with its default.
- {doc}`profiles`: ready-made configurations for common machine shapes.
- {doc}`/user-guide/trust/secrets`: why credentials belong in `env:` and `file:` references
  rather than in a config value.
- {doc}`/integrations/compute/ray`: the variables a cluster reads, including the shuffle token.
