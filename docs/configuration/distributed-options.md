# Distributed configuration options

This page is the field-by-field reference for `config.distributed`, the section that
governs how Batcher attaches to a Ray cluster and how the shuffle behaves on it. It is a
page of its own because it is the largest section by some distance and the only one with
sub-sections: fault tolerance, restricted networks, autoscaling, and TLS are separate
concerns that happen to share a dataclass.

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

How the engine attaches to a Ray cluster, shuffles across it, and stays correct
through node and task failures. Ray is used for scheduling only; bulk shuffle data
moves over Arrow Flight, bypassing the Ray object store. See the **fault-tolerant
cluster recipe in {doc}`profiles`.

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

### Fault tolerance

The first line of defense is Ray-level retries; beneath it, a lost shuffle worker's
output is recomputed from its (durable) source partition and re-fetched.

| Field | Default | Meaning |
|-------|---------|---------|
| `task_max_retries` | `2` | Ray reruns a failed shuffle task this many times. Shuffle tasks are deterministic and recomputed from a durable source, so a rerun is safe. |
| `retry_on_transient` | `True` | Extend task retries to application exceptions (not just worker death). |
| `actor_max_restarts` | `1` | Respawn a crashed compute actor (the map/inference pool) this many times. |
| `actor_max_task_retries` | `1` | Rerun an in-flight actor call on the respawned actor this many times. |
| `recovery_max_attempts` | `3` | Recompute→retry rounds before a still-broken shuffle fails loudly. A larger/flakier cluster may want more. |
| `recovery_backoff_base_s` | `0.5` | Base of the exponential backoff slept between recovery rounds (`0` disables the sleep). |
| `flight_idle_timeout_s` | `60.0` | Max gap between batches in a shuffle fetch before the peer is treated as dead. Generous so a long GC pause is not misread as death; bounded so a truly dead peer is detected and recomputed. |
| `flight_keepalive_s` | `None` | HTTP/2 keepalive ping interval. `None`/`0` disables it; set it to detect a silently-dropped connection faster than the idle timeout. |
| `placement_timeout_s` | `60.0` | How long gang-scheduling waits for a worker placement group before falling back to default scheduling (a real cluster may need to autoscale up). |
| `speculation_max_backups` | `1` | Max concurrent speculative backup tasks at a shuffle barrier. One backup catches the single worst straggler without letting a uniformly slow stage spawn a backup per task. `0` disables straggler speculation, making the barrier a plain wait. |
| `speculation_straggler_factor` | `1.5` | Back up a task whose elapsed time exceeds this multiple of the median finished task's time. Batcher also learns this factor per operator family from measured task-time variance, so a stage that finishes uniformly raises its own bar. |
| `speculation_min_finished_frac` | `0.75` | Fraction of tasks that must finish before speculation starts. |
| `skew_join_salt` | `0` | Spread a hot join key's rows across this many reducers (`0` disables skew-aware salting). |
| `skew_join_fraction` | `0.10` | A value is "hot" when it exceeds this fraction of a side's rows. |
| `shuffle_token` | `None` | Shared secret authenticating Flight shuffle fetches. Also read from `BATCHER_SHUFFLE_TOKEN`. |
| `shuffle_port_range` | `None` | `(min, max)` the Flight shuffle listener may bind, instead of an OS-ephemeral port. Also read from `BATCHER_SHUFFLE_PORT_RANGE` (`"40000-40100"`). |

### Restricted networks

The defaults assume nodes can reach each other freely, which is true on a normal cloud VPC and often false on-premises. Two knobs cover firewalls, multi-homed hosts, and NAT.

`shuffle_port_range` confines the Flight listener to a range you can open in a firewall rule. By default each worker takes an ephemeral port, which never collides but obliges you to open the entire ephemeral range node-to-node. Make the range at least as wide as the number of workers that share a node. A worker that can't find a free port fails with an error naming the range rather than binding somewhere unreachable.

```bash
export BATCHER_SHUFFLE_PORT_RANGE=40000-40100
```

`BATCHER_ADVERTISE_HOST` overrides the address a worker advertises to its peers. Batcher uses the node IP Ray reports, which is correct almost everywhere. Set this when the address peers must dial differs, such as on a multi-homed host whose shuffle belongs on a second network interface, or on a NAT'd or VPC-peered network. Set it per node, in the pod spec or the node environment, because the right value differs on each one.

### Cluster saturation and autoscaling

Out of the box the distributed engine fills the whole cluster with no tuning. It attaches to the running cluster, even on a managed workspace that exports no `RAY_ADDRESS`. It fans out to one worker per node, gives each worker an even share of that node's cores so morsel parallelism saturates every core, and scales the shuffle reducer count with the worker count. On an autoscaling-capable cluster it also asks the autoscaler for the cores a query wants, then waits a bounded time for the new nodes to arrive before sizing the fan-out. A big query therefore runs on the grown cluster instead of clamping to the pre-scale size and leaving the new capacity for the next job.

The autoscale wait auto-enables when Batcher detects an autoscaling cluster, meaning Anyscale, a spot node, or `BATCHER_AUTOSCALE=1`. It stays off on a fixed or single-node cluster, so the default needs no configuration. Override any of it explicitly with the fields below.

| Field | Default | Meaning |
|-------|---------|---------|
| `autoscale_wait_s` | `-1.0` (auto) | Seconds to wait for autoscaler-launched nodes before sizing the fan-out. `-1` resolves to a bounded wait on an autoscaling cluster and to `0`, meaning off, on a fixed one. `0` disables it even on an autoscaling cluster. A positive value caps the budget. |
| `autoscale_poll_s` | `5.0` | Poll interval while waiting for capacity to arrive. |
| `autoscale_stall_s` | `90.0` | Grace window. Give up the wait once capacity has been flat this long, which happens on a fixed cluster or when spot capacity can't be had, so the wait never blocks the whole budget on nodes that won't arrive. Any capacity gain resets it. Sized longer than a node's boot time. |
| `max_shuffle_partitions` | `2048` | Cap on shuffle reducers, so the reducer count scales with the cluster but an all-to-all exchange stays bounded (not O(nodes²)) at thousands of nodes. `0` disables the cap. |

The worker fan-out itself isn't a `Config` field. Pass `num_workers=` to the terminal call, such as {py:meth}`ds.collect(num_workers=16) <batcher.Dataset.collect>`, to pin it and skip the wait. Leave it unset and Batcher auto-sizes to one worker per node on a multi-node cluster, or to all cores on a single node.

The `BATCHER_AUTOSCALE` environment variable is authoritative in both directions. `1` forces the wait on and `0` forces it off, even on a managed cluster. The wait is pure scheduling, so the result is identical whether it waits or not.

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
- {doc}`/user-guide/scale/distributed`: running a pipeline on a cluster.
- {doc}`fault-tolerance`: what happens when nodes and devices fail underneath a job.
