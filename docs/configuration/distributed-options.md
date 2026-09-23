# Distributed configuration options

This page is the field-by-field reference for `config.distributed`, the section that
governs how Batcher attaches to a Ray cluster, how the shuffle behaves on it, and how inference and GPU stages are placed. It is the largest section by some distance, so it has a page of its own, grouped by concern.

The rest of the sections are in {doc}`options`.

```python
import dataclasses
from batcher import Config

base = Config()
cfg = base.replace(distributed=dataclasses.replace(base.distributed, transport="flight"))
print(cfg.distributed.transport)
# flight
```

## Attaching to a cluster

How the engine attaches to a Ray cluster, shuffles across it, and stays correct through node and task failures. Ray schedules the work, and bulk shuffle data moves over Arrow Flight, bypassing the Ray object store. {doc}`profiles` has a fault-tolerant cluster recipe.

| Field | Default | Meaning |
|-------|---------|---------|
| `ray_address` | `None` | Ray cluster address. `None` attaches to a running cluster when `RAY_ADDRESS` is set, or when Batcher detects a managed control plane such as Anyscale. A distributed query on a managed workspace therefore fans out across the cluster with no configuration instead of stranding on a local single-node Ray. It falls back to a local start only when no cluster is reachable. Set an explicit address to override. |
| `namespace` | `"batcher"` | Ray namespace for batcher's shuffle actors, so they are isolatable. |
| `runtime_env` | `None` | `runtime_env` dict shipped to workers so `batcher` + its native extension are present cluster-wide. |
| `transport` | `"auto"` | Shuffle transport. `"auto"` picks Flight on a multi-node cluster, disk on a single node / shared filesystem; `"flight"`/`"disk"` force it. |
| `shared_filesystem` | `False` | True when every worker shares a filesystem at the same path, so the disk shuffle is safe cluster-wide. |
| `dashboard` | `False` | Show the Ray dashboard. |
| `tls` | `ShuffleTlsConfig()` (off) | TLS/mTLS for the inter-node Arrow Flight shuffle. Sub-section below. |
| `adaptive_credits` | `True` | AIMD shuffle credits: the window grows and shrinks per remote fetch from observed memory backpressure instead of holding the static grant. Flow control only, so the merged output is unchanged. `False` pins the static `default_credits` window. |
| `runtime_bloom_join` | `"auto"` | Build a bloom from the join build side and push it to the probe side to drop non-matching rows before they shuffle, cutting network volume for a selective fact-to-dimension join. `"auto"` engages only when Kyber estimates the probe is much larger than the build. `True` always engages it and `False` never does. Inner and semi joins only. |
| `shared_memory_transfer` | `True` | Same-node shared-memory shuffle: a mapper mirrors each bucket to a memory-mapped Arrow-IPC file, using Linux `/dev/shm` when available, that a same-node reducer reads via mmap with no gRPC. It's pressure-gated, so it's skipped when the node is tight on memory, and best-effort. A miss falls back to Flight, which is bit-identical. |
| `locality_aware_scheduling` | `True` | Host a reducer whose bucket concentrates on one node on that node, turning the bulk of its fetches into same-node hits. Result-preserving; pays off on a multi-node cluster with a skewed / co-partitioned shuffle. A single-node fleet resolves to "nothing to place" from the worker addresses alone, with no remote call. |
| `persistent_fleet` | `False` | Reserve one placement group and worker fleet for a whole adaptive multi-stage query, keeping each stage's intermediate partitioned on the workers instead of collecting to the driver. Removes per-stage placement churn and the driver funnel. |
| `distribute_min_rows` | `1000000` | Estimated input rows below which `distributed="auto"` stays single-node even on a cluster, because the Ray fan-out carries a fixed cost a small query never repays. A GPU stage always distributes. `0` always distributes on a cluster. An explicit `distributed=True` or `False` overrides it. |
| `mode` | `"auto"` | What `distributed="auto"` means for every terminal that doesn't pass `distributed=` itself, which includes `count()`, `min()`, `to_pydict()`, the `ds.meta` fallbacks, `ds.dq.validate()` and `ds.dq.fail()`. `"auto"` is the size- and topology-aware routing above. `"always"` forces the Ray path and starts a local Ray when none is running. `"never"` keeps every such terminal single-node. An explicit `distributed=True` or `False` overrides it. |
| `cluster_connect_timeout_s` | `30.0` | Retry window, with exponential backoff, for attaching to a cluster whose head isn't answering yet, as when a KubeRay driver pod starts before its head. A detected address that never answers falls back to local Ray. An address you set explicitly raises instead. `0` makes one attempt. |
| `trust_cluster_image` | `False` | Trust that every worker image already carries a compatible `batcher`. By default the driver ships its own package to a remote cluster when no `runtime_env` is given. Set `True` for a production image that bakes Batcher in. |
| `object_store_memory_bytes` | `None` | Object store size for a Ray that Batcher starts locally. `None` uses Ray's default. Ignored when attaching to an existing cluster. Bulk data bypasses the object store, so this bounds only control-plane metadata. |
| `reuse_session_fleet` | `True` | Keep one health-checked shuffle fleet across separate distributed queries in a session, so the second query skips the actor, placement-group, and Flight-server spawn. Disabled while a `persistent_fleet` query owns a fleet. |
| `session_fleet_idle_s` | `30.0` | Seconds an idle reused fleet lives before its cores return to the cluster. |
| `fleet_max_attempts` | `2` | Fresh-fleet attempts for a `persistent_fleet` query when a worker dies holding an already-materialized cross-stage intermediate. The deterministic query re-runs, so the result is unchanged. |
| `resilience` | `"default"` | Named fault-tolerance profile. `"default"` keeps the conservative budgets below; `"spot"` hardens them as a bundle (more restarts / recompute, keepalive on, one speculative backup) for a churning spot-node cluster. Explicit knobs override the profile. See {doc}`../architecture/fault-tolerance`. |

This section is the {py:class}`DistributedConfig <batcher.config.config.DistributedConfig>`
dataclass (the API reference lists every field). Construct one and swap it onto {py:class}`Config <batcher.Config>`. For example, to isolate a job's shuffle actors in their own Ray namespace:

```python
from batcher import Config
from batcher.config import DistributedConfig

cfg = Config().replace(distributed=DistributedConfig(namespace="nightly-etl"))
print(cfg.distributed.namespace)
# nightly-etl
```

### Pin terminals that take no distributed argument

`collect()` and `iter_batches()` accept `distributed=`, but many terminals don't. The scalar terminals such as `count()` and `min()`, `to_arrow()` and `to_pydict()`, the fallbacks behind `ds.meta`, and `ds.dq.validate()` and `ds.dq.fail()` all run with `distributed="auto"`. Set `mode` to decide for all of them at once. Scope it with `option_context` so it covers one block of work:

```python
import batcher as bt
from batcher.config import option_context

orders = bt.from_pydict({"order_id": [1, 2, 3], "amount": [10.0, -2.0, 7.5]})
with option_context("distributed.mode", "never"):
    report = orders.dq.positive("amount").validate()
    rows = orders.meta.shape()[0]
print(report.violations, rows)
# {'positive(amount)': 1} 3
```

Use `"always"` inside the block to run the same checks on the cluster. With no `num_workers` to go on, a distributed run places one worker per node, and each worker uses all of its node's cores. So on a single-node Ray, such as a laptop, `"always"` runs one Ray worker, and to split the work into several partitions there you pass `collect(distributed=True, num_workers=N)` on a lazy result. On a cluster, a data-quality gate over a large file-backed table is where this matters most: `"auto"` already distributes it once the input passes `distribute_min_rows`, and `"always"` makes that routing explicit. A result is identical either way, within the stated exceptions in {doc}`../architecture/deep-dives/distribution/index`.

### Fault tolerance

The first line of defense is Ray-level retries; beneath it, a lost shuffle worker's
output is recomputed from its (durable) source partition and re-fetched.

| Field | Default | Meaning |
|-------|---------|---------|
| `task_max_retries` | `2` | Application-error retries for a shuffle task, which is deterministic and recomputed from a durable source, so a rerun is safe. **Not a count of preemption retries:** any non-zero value gives Ray *unlimited* retries for system errors (preemption, worker loss), and `0` makes the task non-retryable so a single preemption kills it. See the note below. |
| `retry_on_transient` | `True` | Extend task retries to application exceptions (not only worker death). |
| `actor_max_restarts` | `1` | Respawn a crashed compute actor (the map/inference pool) this many times. |
| `actor_max_task_retries` | `1` | Rerun an in-flight actor call on the respawned actor this many times. |
| `recovery_max_attempts` | `3` | Recompute-and-retry rounds before a still-broken shuffle fails loudly. A larger/flakier cluster may want more. |
| `recovery_backoff_base_s` | `0.5` | Base of the exponential backoff slept between recovery rounds (`0` disables the sleep). |
| `flight_idle_timeout_s` | `60.0` | Max gap between batches in a shuffle fetch before the peer is treated as dead. Generous so a long GC pause is not misread as death; bounded so a truly dead peer is detected and recomputed. |
| `flight_keepalive_s` | `None` | HTTP/2 keepalive ping interval. `None`/`0` disables it; set it to detect a silently-dropped connection faster than the idle timeout. |
| `placement_timeout_s` | `60.0` | How long gang-scheduling waits for a worker placement group before falling back to default scheduling (a real cluster may need to autoscale up). |
| `speculation_max_backups` | `1` | Max concurrent speculative backup tasks at a shuffle barrier. One backup catches the single worst straggler without letting a uniformly slow stage spawn a backup per task. `0` disables straggler speculation, making the barrier a plain wait. |
| `speculation_straggler_factor` | `1.5` | Back up a task whose elapsed time exceeds this multiple of the median finished task's time. Batcher also learns this factor per operator family from measured task-time variance, so a stage that finishes uniformly raises its own bar. |
| `speculation_min_finished_frac` | `0.75` | Fraction of tasks that must finish before speculation starts. |
| `skew_join_salt` | `0` | How many reducers a hot join key's rows spread across. `0` does not mean off: salting engages on measured skew and sizes its own fan-out. A positive value forces the detection pre-pass and pins the fan-out; a negative value never salts. |
| `skew_join_fraction` | `0.10` | A value is "hot" when it exceeds this fraction of a side's rows. |
| `shuffle_token` | `None` | Shared secret authenticating Flight shuffle fetches. Also read from `BATCHER_SHUFFLE_TOKEN`. |
| `shuffle_replication` | `1` | Workers holding each mapper's published shuffle buckets. At `2` or more, a reducer whose mapper is gone fetches a byte-identical copy from another node instead of waiting for a recompute. Covers aggregate, join, sort, and window shuffles. The `spot` profile raises it to `2`. |
| `drain_lead_s` | `120.0` | How long before a known termination deadline a worker starts migrating its shuffle output to a survivor. Consulted only when a deadline is discoverable, from `SLURM_JOB_END_TIME` or `BATCHER_DEADLINE_EPOCH_S`. |
| `on_read_error` | `"error"` | Distributed scan policy for an unreadable file or row group. `"error"` fails the query. `"skip"` skips the failing split, keeps its healthy siblings, and records the skipped count on the worker. |
| `shuffle_port_range` | `None` | `(min, max)` the Flight shuffle listener may bind, instead of an OS-ephemeral port. Also read from `BATCHER_SHUFFLE_PORT_RANGE` (`"40000-40100"`). |

:::{important}
**Retry counts are a step function, not a dial.** Ray treats any non-zero `max_retries` as "this task is safe to rerun" and then retries *system* errors (spot preemption, worker crash, node loss) without decrementing the count. The number bounds only application errors. At `0` the task becomes non-retryable and a single preemption kills it permanently.

So `task_max_retries=2` means "unlimited preemption retries, two application retries", and lowering it to `0` to reduce retry noise silently removes every spot protection on that path. The same holds for `actor_max_restarts`. On a churning cluster set `resilience="spot"`, which raises both rather than leaving you to reason about the step.
:::

### Restricted networks

The defaults assume nodes can reach each other freely, which is true on a normal cloud VPC and often false on-premises. Two knobs cover firewalls, multi-homed hosts, and NAT.

`shuffle_port_range` confines the Flight listener to a range you can open in a firewall rule. By default each worker takes an ephemeral port, which never collides but obliges you to open the entire ephemeral range node-to-node. Make the range at least as wide as the number of workers that share a node. A worker that can't find a free port fails with an error naming the range rather than binding somewhere unreachable.

```bash
export BATCHER_SHUFFLE_PORT_RANGE=40000-40100
```

`BATCHER_ADVERTISE_HOST` overrides the address a worker advertises to its peers. Batcher uses the node IP Ray reports, which is correct almost everywhere. Set this when the address peers must dial differs, such as on a multi-homed host whose shuffle belongs on a second network interface, or on a NAT'd or VPC-peered network. Set it per node, in the pod spec or the node environment, because the right value differs on each one.

IPv6 works with no configuration. A worker whose advertised address is an IPv6 literal binds an IPv6 listener and advertises a bracketed authority such as `[fd00::1]:40001`, so an IPv6-only cluster needs no IPv4 address anywhere. On a dual-stack fabric Batcher prefers the IPv4 address, since it routes in more places; a link-local `fe80::` address is never advertised, because it is dialable only with the peer's own zone index appended.

### Cluster saturation and autoscaling

Out of the box the distributed engine fills the whole cluster with no tuning. It attaches to the running cluster, even on a managed workspace that exports no `RAY_ADDRESS`. It cuts each node into several workers, gives each an even share of that node's cores so morsel parallelism saturates every core, and scales the shuffle reducer count with the worker count. On an autoscaling-capable cluster it also asks the autoscaler for the cores a query wants, then waits a bounded time for the new nodes to arrive before sizing the fan-out. A big query therefore runs on the grown cluster instead of clamping to the pre-scale size and leaving the new capacity for the next job.

The autoscale wait auto-enables when Batcher detects an autoscaling cluster, meaning Anyscale, a spot node, or `BATCHER_AUTOSCALE=1`. It stays off on a fixed or single-node cluster, so the default needs no configuration. Override any of it explicitly with the fields below.

| Field | Default | Meaning |
|-------|---------|---------|
| `autoscale_wait_s` | `-1.0` (auto) | Seconds to wait for autoscaler-launched nodes before sizing the fan-out. `-1` resolves to a bounded wait on an autoscaling cluster and to `0`, meaning off, on a fixed one. `0` disables it even on an autoscaling cluster. A positive value caps the budget. |
| `autoscale_poll_s` | `5.0` | Poll interval while waiting for capacity to arrive. |
| `autoscale_startup_grace_s` | `12.0` | How long to wait for the first sign of growth before concluding the autoscaler won't act. The query is already runnable on current capacity, so this keeps an unsatisfiable request from costing the full `autoscale_stall_s`. Once growth appears, `autoscale_stall_s` governs. |
| `autoscale_stall_s` | `90.0` | Grace window. Give up the wait once capacity has been flat this long, which happens on a fixed cluster or when spot capacity can't be had, so the wait never blocks the whole budget on nodes that won't arrive. Any capacity gain resets it. Sized longer than a node's boot time. |
| `map_partition_multiplier` | `4` | Ceiling on map partitions per worker, which is the shuffle's task granularity. Above 1 a straggler holds a fraction of a node's share of the input rather than all of it, and a lost worker's partitions are re-dealt across survivors. Bounded by the splits the source actually has, so a small input makes fewer partitions rather than empty tasks. `1` pins one partition per worker. |
| `shuffle_partition_multiplier` | `4` | Ceiling on shuffle reducers per worker. Above the one-per-worker floor, more buckets lower each reducer's memory. They do not fix skew: a hash bucket is the unit a key can't be split below, which is what `skew_join_salt` is for. |
| `max_shuffle_partitions` | `2048` | Cap on both counts above, so they scale with the cluster while an all-to-all exchange stays bounded (not O(nodes^2)) at thousands of nodes. `0` disables the cap. |

Both multipliers are ceilings rather than targets, and both are pure scheduling: the mergeable algebra makes any partitioning produce the same rows, so raising or lowering either changes cost and never the answer.

### Where the fleet lands

Placement decides which nodes hold the shuffle fleet. It never changes the answer, so it is a cost and latency knob.

| Field | Default | Meaning |
|-------|---------|---------|
| `zone_aware_placement` | `True` | Reserve the fleet's placement-group bundles inside one availability zone when the cluster spans several and one of them can host the whole fleet. A shuffle sends nearly all its bytes worker to worker, and every cloud bills those bytes in both directions across a zone boundary; the bundles are interchangeable, so a fleet that fits in one zone is placed in one. A no-op on a single-zone cluster, on nodes with no zone label, and when no one zone has room. Set `False` when the zone spread is deliberate, bought for availability. |
| `capacity_aware_placement` | `False` | Place each stage on the capacity its failure model fits: recomputable work on spot, state-holding work on on-demand. A stateless map partition re-derives from its durable descriptor, so a reclamation costs a resubmission; a shuffle worker holds accumulated state, so a reclamation costs the stage. Batcher expresses that as a market-type `label_selector` with a `fallback_strategy`, so a fleet that wants on-demand and finds none runs on spot rather than pending. Off by default, and a no-op even when on unless the live fleet is genuinely mixed and labelled under one market-type key. |
| `task_events` | `"auto"` | Whether each task reports its lifecycle to Ray's Dashboard and State API. `"auto"` keeps the events for any stage at or below `task_events_fanout_cap` tasks and drops them above it; `"always"` and `"never"` pin the answer. An observability trade, not a scheduling one: the tasks run identically either way. |
| `task_events_fanout_cap` | `10000` | The stage width past which `"auto"` stops reporting. A hundred-thousand-partition stage puts a hundred thousand events onto a control plane every driver in the fleet shares, to fill a task table nobody can read at that size. |

The zone is chosen by free capacity, not nameplate, so the pin never lands the fleet where a co-tenant has already filled the zone. And it is applied to the bundles rather than the tasks: a group that can't form is abandoned at `placement_timeout_s` and the stage falls back to ordinary scheduling, where a pin on the tasks themselves would leave them pending indefinitely.

The worker fan-out itself isn't a `Config` field. Pass `num_workers=` to the terminal call, such as {py:meth}`ds.collect(num_workers=16) <batcher.Dataset.collect>`, to pin it and skip the wait. Leave it unset and Batcher auto-sizes the fan-out from the cluster's shape on a multi-node cluster, or to all cores on a single node.

The `BATCHER_AUTOSCALE` environment variable is authoritative in both directions. `1` forces the wait on and `0` forces it off, even on a managed cluster. The wait is pure scheduling, so the result is identical whether it waits or not.

### Shuffle transport

These fields shape how bytes move once a shuffle is running. None of them changes a result.

| Field | Default | Meaning |
|-------|---------|---------|
| `flight_compression` | `"lz4"` | Wire codec for shuffle batches: `"none"`, `"lz4"`, `"zstd"`, or `"auto"`. `"lz4"` is nearly free and gives up fast on incompressible data. `"zstd"` trades CPU for ratio. `"auto"` decides from the node's measured fabric rate, since on a very fast link the compressor becomes the ceiling. |
| `flight_connections_per_peer` | `4` | TCP connections a reducer stripes one peer's fetches across, because cloud NICs cap a single flow below line rate. The pool grows only under concurrent fetches to that peer. `1` uses one connection. |
| `prefer_fabric_interface` | `False` | Advertise each worker's fabric (such as InfiniBand) address for the shuffle instead of the address Ray knows it by. A worker with no fabric address keeps its Ray address, and `BATCHER_ADVERTISE_HOST` still wins. Turn it on only where fabric addresses route between every pair of workers. |

### Map tasks and submission

Stateless map and inference partitions are placed and submitted by the fields below. Placement never changes which rows a partition holds.

| Field | Default | Meaning |
|-------|---------|---------|
| `map_spread` | `"auto"` | Map-task placement. `"auto"` keeps Ray's SPREAD only where many sub-core tasks would stack on one node, and uses Ray's locality-aware DEFAULT otherwise. `"always"` and `"never"` force one. |
| `map_spread_node_cap` | `100` | Alive nodes above which `"auto"` prefers DEFAULT, because SPREAD's per-task scheduling cost grows with the node count. |
| `map_spread_pack_share` | `0.5` | Per-task CPU share below which `"auto"` treats packing as a risk and keeps SPREAD. |
| `max_pending_tasks` | `0` | Cap on concurrently submitted map and inference partition tasks. `0` derives it as `pending_window_factor` times the schedulable cores, so an ordinary fan-out submits everything and a 100,000-partition job stays bounded. |
| `pending_window_factor` | `4` | Multiplier on schedulable cores for the derived `max_pending_tasks` window. |
| `heterogeneous_node_isolation` | `False` | Hard-restrict a CPU fleet to nodes advertising the `cpu_node_resource` custom resource, keeping shuffle tasks off GPU nodes. Emitted only when those nodes can host the fleet. Label CPU-only nodes with `resources={"cpu_node": N}`. |
| `cpu_node_resource` | `"cpu_node"` | Custom resource name CPU-only nodes advertise for `heterogeneous_node_isolation`. |

### Inference stages

These fields govern `map_batches` model stages and `ds.ml.infer` on a cluster. Each is result-preserving, apart from the numeric precision `autocast_inference` changes.

| Field | Default | Meaning |
|-------|---------|---------|
| `stream_inference` | `True` | Split a linear `map_batches` chain at every resource-class boundary into per-stage actor pools that stream partitions over Arrow Flight, so a model runs while the stage below prepares the next partition. `False` runs the whole chain per partition in one actor. |
| `warm_inference_pools` | `True` | Keep inference actor pools warm across `collect()` calls in a session, so a model loads once per session. {py:func}`bt.release_cluster() <batcher.release_cluster>` hands them back early. |
| `warm_inference_idle_s` | `120.0` | Seconds an idle warm pool keeps its devices before its actors are killed. `0` keeps them for the whole session, which suits a dedicated inference process. |
| `map_inflight_depth` | `2` | Partitions an inference actor may have in flight, so the device stays fed while the next partition is dispatched. |
| `map_inflight_adaptive` | `True` | Let a prior run's measured low GPU utilization raise the in-flight depth, within a bound. A first run is unchanged. |
| `autocast_inference` | `True` | Run a GPU inference forward pass under `torch.autocast` in the device's fast half-precision type. No-op on CPU, without `torch`, or on failure. `False` forces full precision for bit-exact reproduction. |
| `torch_compile` | `True` | Apply `channels_last` and `torch.compile` to a convolutional vision model in the managed `ds.ml.infer` path. Text models stay eager. Set `False` for a job too small to repay the compile. |
| `gpu_activation_bytes_per_row` | `65536` | Estimated activation bytes per row, used only to seed an inference stage's first batch size from the VRAM left after the model. The throughput controller corrects it from measurements. |

### GPU backend

These fields tune `backend="gpu"` and `backend="auto"` for relational stages on cuDF. A plan the device can't run exactly is declined and runs on the CPU engine, so none of them can change a result.

| Field | Default | Meaning |
|-------|---------|---------|
| `gpu_min_rows` | `10000000` | Estimated rows below which `backend="auto"` stays on the CPU engine, because the fixed device overhead isn't amortized. |
| `gpu_memory_gb` | `0.0` | Usable memory of one GPU. `0.0` detects it. A positive value pins it, for a device the probe can't see or to under-commit a shared one. |
| `gpu_backend_cudf` | `True` | Ship cuDF to the GPU worker tasks, matching the driver's installed version. `False` uses the slower `torch` fallback. |
| `gpu_rapids_path` | `""` | A directory on shared storage holding a staged RAPIDS tree, put on the tasks' `PYTHONPATH` instead of installing cuDF per node. A path that doesn't exist is ignored. |
| `gpu_shadow_verify` | `False` | Re-run every GPU result on the CPU engine and report disagreement. It doubles the work, so use it for benchmark, staging, and device-tier development runs rather than production. |
| `gpu_admission_wait_s` | `30.0` | How long a GPU fan-out waits for a free device before the CPU engine answers. `0` waits indefinitely. |
| `gpu_shard_oversubscribe` | `4` | Shards cut per GPU, so each shard is bounded and a preempted shard's retry is cheap. |
| `gpu_min_shard_bytes` | `134217728` (128 MiB) | Smallest shard a fan-out cuts, so a small scan isn't split into tasks that are mostly dispatch cost. |
| `gpu_shard_expansion` | `2.0` | Multiple of a shard's input bytes the device must hold for it. |
| `gpu_pack_shards` | `True` | Request a fraction of a GPU per shard, derived from the largest shard's working set, so several shards share a device. `False` asks for one whole device per shard. |
| `gpu_task_fraction` | `0.0` | Pin the per-shard device share instead of deriving it. `1.0` forces whole devices for one job. |
| `gpu_max_tasks_per_device` | `4` | Ceiling on shards resident on one device. |
| `gpu_shard_subdivide` | `4` | Pieces a shard that didn't fit the device is divided into and rerun on the device. Exact, because the stage is mergeable. `1` sends an over-large shard straight to the CPU. |
| `gpu_shard_subdivide_rounds` | `3` | Further subdivision rounds before a shard goes to the CPU. |
| `gpu_shard_cpu_fallback` | `True` | Recompute a GPU shard that fails for another reason on the CPU engine, rather than re-running the whole query on the host. |
| `gpu_merge_wave` | `32` | Shard partials the driver folds together at once, so driver memory tracks the wave size rather than the shard count. `0` or `1` folds everything at once. |
| `gpu_frame_cache_fraction` | `0.25` | Share of a device's usable memory a GPU worker may keep as cached decoded shards between queries. `0.0` turns the cache off. |
| `gpu_tree_broadcast_fraction` | `0.35` | Share of one device's memory the replicated leaves of a multi-way plan tree may occupy before the fan-out is declined. |
| `gpu_worker_reuse` | `True` | Let one GPU worker process serve several shards, so each shard doesn't pay a cuDF import and allocator setup. Set `False` for a UDF that holds device memory. |
| `gpu_max_autoscale_devices` | `64` | Most devices one query may ask the autoscaler for. Reaching the cap runs the query in more waves on fewer devices. |

GPU relational execution is opt-in: `backend` defaults to `"cpu"`. See {doc}`accelerator` for device memory, energy, and health.

### distributed.tls

The shuffle carries query data straight between worker processes, including columns a governance policy has already decrypted or masked. On a network you don't fully control, encrypt it. `config.distributed.tls` is a
{py:class}`ShuffleTlsConfig <batcher.config.config.ShuffleTlsConfig>`: the fields are **paths**
to PEM material your platform already mounts on every worker (a Kubernetes secret volume,
cert-manager, a cloud private CA). Batcher reads them at worker start and issues no
certificates itself.

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `False` | Turn on TLS for the Flight shuffle. Off means a plaintext shuffle, which is the right default only on a trusted network. |
| `ca_cert_path` | `""` | The CA a peer's certificate must chain to. It's the trust root in both directions, because one cluster CA usually signs both server and client certificates. |
| `server_cert_path` | `""` | This node's server certificate, presented on its Flight port. |
| `server_key_path` | `""` | The private key for `server_cert_path`. |
| `client_cert_path` | `""` | This node's client certificate, presented under mTLS when fetching from a peer. Empty means outbound connections are server-auth only. |
| `client_key_path` | `""` | The private key for `client_cert_path`. Set together with the certificate or not at all. |
| `require_client_auth` | `False` | mTLS: verify a client certificate on every incoming fetch, so a process that can merely reach the port cannot pull shuffle data. |
| `server_name` | `"batcher-shuffle"` | The name checked against a peer certificate's SAN. Peers are dialed by address, so the certificate rarely matches the literal host. Set this to the name your certificates actually carry. |

A half-configured TLS setup fails at config time, not at the first fetch: enabling TLS
without `ca_cert_path`, without the server certificate/key pair, or with only one half of
the client pair raises {py:exc}`ConfigError <batcher.ConfigError>`. Each field also has a
`BATCHER_DISTRIBUTED_TLS_<FIELD>` env override, which is how a deployment injects paths
without shipping a config file.

```python
# docs: skip
from batcher import Config, set_config
from batcher.config import DistributedConfig, ShuffleTlsConfig

set_config(
    Config().replace(
        distributed=DistributedConfig(
            transport="flight",
            tls=ShuffleTlsConfig(
                enabled=True,
                ca_cert_path="/etc/batcher/ca.pem",
                server_cert_path="/etc/batcher/server.pem",
                server_key_path="/etc/batcher/server.key",
                client_cert_path="/etc/batcher/client.pem",
                client_key_path="/etc/batcher/client.key",
                require_client_auth=True,
            ),
        )
    )
)
```

Pair it with `shuffle_token` above: TLS proves *who* the peer is, the token proves it is
allowed to fetch this partition.


## See also

- {doc}`options`: every other configuration section.
- {doc}`/architecture/deep-dives/distribution/distributed-scheduling`: running a pipeline on a cluster.
- {doc}`fault-tolerance`: what happens when nodes and devices fail underneath a job.
