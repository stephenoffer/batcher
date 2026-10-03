# The shuffle over Arrow Flight

The *shuffle* is the all-to-all redistribution that puts every row with the same key on the same machine, which is what a distributed group-by or join needs before it can reduce. This page describes how Batcher moves those batches over Arrow Flight, how a reducer picks the cheapest source for each bucket, and when the disk shuffle runs instead.

Batcher's shuffle moves Arrow record batches worker to worker over Arrow Flight on gRPC. Ray schedules the workers and carries their addresses. It doesn't carry their data.

:::{important}
The data plane bypasses the Ray object store. What crosses Ray is addresses, tickets, paths, row counts, and a metrics string. Bulk Arrow batches move over Arrow Flight with credit-based backpressure, so they never pay the object store's serialization and memory cost.
:::

From the user's side, a shuffle is just a distributed terminal op. The `transport` argument picks the shuffle, and `"auto"` is right almost always:

```python
# docs: skip
import batcher as bt

ds = bt.from_pydict({"k": [i % 10 for i in range(100_000)], "v": list(range(100_000))})
out = ds.group_by("k").agg(s=bt.sum("v")).collect(distributed=True, num_workers=4, transport="auto")
print(out.num_rows)  # 10
```

![A shuffle runs on two separate channels between the same pair of workers. On the control plane, Ray schedules the tasks and actors and carries an address, a ticket, a file path, a row count and a metrics JSON string, and nothing else: the mapper's Flight address goes up through Ray and the reducer's ticket comes back down. On the data plane, the mappers' partition_batches publishes every bucket including the empty ones, and the Arrow record batches travel directly to the reducers over do_exchange on gRPC, LZ4 by default and credit-bounded, one credit being one batch slot with the producer blocking at zero. The reducers fold arrivals into a running partial in Rust through gather_combine, so the intermediate never crosses back into Python. Bulk batches never pass through the Ray object store, because routing them through it reintroduces the serialization the columnar design removes.](/_static/diagrams/shuffle_dataflow.svg)

```text
   MAPPERS                                                REDUCERS
   -------                                                --------
   worker 0 --partition_batches--> [b0][b1][b2][b3]
   worker 1 --partition_batches--> [b0][b1][b2][b3]       the reducer for bucket 1
   worker 2 --partition_batches--> [b0][b1][b2][b3]       fetches b1 from all three
                                        |                            |
        every bucket is published,      +----------------------------+
        including the empty ones                                     v
                                        +------------------------------------------+
                                        |  it picks the cheapest source available  |
                                        +------------------------------------------+
                                        |  same process  DIRECT_MEMORY   no copy   |
                                        |  same node     SHARED_MEMORY   ~ memcpy  |
                                        |  another node  NETWORK         Flight,   |
                                        |                                credited  |
                                        +--------------------+---------------------+
                                                             |
                                    folded into a running partial IN RUST
                                    (gather_combine): the intermediate never
                                    crosses back into Python
                                                             |
                                                             v
                                                     combine_finalize
```

## The three tiers

A reducer fetching a bucket has three possible sources, and it picks the cheapest without any configuration.

![Carbonite routes one shuffle partition by placement. The same Flight address means one process, so DIRECT_MEMORY reads from the local store with no serialization. The same node identity means one host, so SHARED_MEMORY reads the bucket back through Arrow IPC over a memory map. Anything else falls back to NETWORK over credit-bounded Arrow Flight.](/_static/diagrams/transfer_modes.svg)

| Source | Path | Cost |
|---|---|---|
| Same process | `DIRECT_MEMORY`: read from the local store | no copy, no socket |
| Same node, other process | `SHARED_MEMORY`: mmap a 64-byte-aligned Arrow IPC file | about a memcpy |
| Another node | `NETWORK`: credit-bounded Arrow Flight | one gRPC stream |

`carbonite/transfer/locality.py::select_mode` makes the choice from the peer's Flight address and node id. Two unknown addresses are `NETWORK`, the mode that is always correct and only ever slower. The same test runs in Rust inside the concurrent gather in [`crates/bc-py/src/shuffle/gather.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-py/src/shuffle/gather.rs), so same-host buckets are read from shared memory while cross-node buckets keep fanning out. The common GPU-cluster layout packs several workers per node, which makes most fetches same-node, cross-process.

:::{dropdown} The shared-memory mirror
The shared-memory file is a second copy of the bucket in tmpfs. `ShuffleSession._shm_mirror_ok()` skips it under memory pressure (`SPILL` or worse) and on a worker that shares its node with no other worker, where it has no possible reader. A skipped mirror is harmless: the shared-memory read misses and the fetch falls back to Flight with the same batches.

The mirror is charged to the worker's cgroup and stays until the plan is torn down (`clear_plan`). `ShuffleSession.stats()` reports it as `bytes_mirrored_shm`, and buckets the gate declined as `shm_mirrors_skipped`. The mmap read is zero-copy: `read_mmap_zero_copy` wraps the mapping as an Arrow `Buffer::from_custom_allocation`.
:::

## The Flight server

Each worker process hosts one Flight server. `bc_transport::ShuffleExchange::bind_tls` binds `0.0.0.0:0` over a shared `Arc<PartitionStore>` and advertises `{node_ip}:{port}`. Only `do_exchange` is on the production path; every other Flight method returns `unimplemented`. It serves one query's buckets to one query's reducers.

A published bucket is held in the worker's heap until a reducer fetches it, and nobody reserved it, so the store keeps a running byte total and gives memory back two ways, spilling the largest buckets first to local Arrow IPC:

- **A cap.** `carbonite/policies/flow_control.py::shuffle_store_cap` sets it per worker at a quarter of the memory envelope, never above `memory.hard_limit` of it.
- **Cooperative spilling.** When an operator can't get a reservation, the {doc}`buffer pool </architecture/deep-dives/memory/buffer-pool>` asks the store to yield first. Published output is finished work, so spilling it costs one re-read.

A spilled bucket returns the same batches, so spilling can't change an answer. The wire encodes with `FlightDataEncoderBuilder`, LZ4 by default under `distributed.flight_compression`:

```python
import batcher as bt

dist = bt.Config().distributed
print(dist.transport, dist.flight_compression, dist.shuffle_replication)  # auto lz4 1
```

:::{dropdown} `ShuffleTicket`: the wire address of one mapper-to-reducer edge
```rust
// crates/bc-transport/src/ticket.rs
pub struct ShuffleTicket {
    pub plan_id: u64,       // per-query fence
    pub stage_id: u32,      // shuffle stage within the plan
    pub src_partition: u32, // mapper id
    pub dst_partition: u32, // reducer / bucket id
    pub epoch: u32,         // re-execution fence
}
```

It serializes to `"{plan}/{stage}/{src}/{dst}/{epoch}"` in `flight_descriptor.path[0]` of the first `DoExchange` message. `path[1]` is an auth token, and `path[2]` is an optional `"shard/nshards"` selector for striping one bucket across several connections. `plan_id` is a 63-bit value from a uuid4, so a reused fleet actor never serves a prior query's leftovers; `epoch` does the same for a recompute after worker loss.

Mappers publish **every** bucket, including empty ones, so a failed fetch always means the worker is gone, never that the bucket was empty.
:::

## Fetching

The reducer's gather, `crates/bc-py/src/shuffle/gather.rs::drive`, works in four steps:

1. Buckets held by this worker, including a replica that lives here, are read straight from the local store.
1. The rest are grouped by peer, and each peer is pulled over a share of a fixed stream budget. Small buckets are packed into one stream by bytes, and one large bucket is striped across several.
1. Each remote fetch tries shared memory for a same-host peer, then Flight. With `distributed.shuffle_replication` above 1, replica addresses follow the primary, so a lost mapper is re-fetched rather than recomputed.
1. Arriving batches fold into a running partial *in Rust* (`gather_combine`) or concatenate (`gather_concat`), so the reducer never materializes every mapper's bucket as a Python object.

Holding the stream count fixed and packing by bytes makes the transfer rate width-independent: a cluster of `W` workers makes `W^2` buckets, and one stream per bucket would tie the rate to the cluster's width rather than the link. On one 25 Gbps link, the run recorded in [`benchmarks/BENCHMARK_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/BENCHMARK_RESULTS.md) (2026-08-29, [`benchmarks/cluster/carbonite/bucket_shape.py`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/cluster/carbonite/bucket_shape.py)) puts the uncompressed wire at 2,684 MiB/s, 86% of line rate, and LZ4 at 2.67x on that data.

Folding in Rust keeps the join reducer's intermediate out of Python. On TPC-H sf10 that intermediate is 3.75M rows and roughly 106 MB per reducer, and `execute_plan_aggregated` runs the join and folds the aggregate inside the engine.

Four fan-in knobs bound different things:

```python
fc = bt.Config().flow_control
print(fc.shuffle_fan_in, fc.shuffle_fetch_fan_in, fc.gather_streams, fc.gather_inflight_bytes >> 20)
# 8 32 48 768
```

| Knob | Default | Bounds |
|---|---|---|
| `flow_control.shuffle_fan_in` | 8 | the *combiner tree* for an aggregate: how many partials one node folds |
| `flow_control.shuffle_fetch_fan_in` | 32 | how many channels a flat gather fetches at once, which also divides the per-channel byte budget |
| `flow_control.gather_streams` | 48 | concurrent Flight streams one reducer runs across all its peers |
| `flow_control.gather_inflight_bytes` | 768 MiB | decoded bytes those streams may hold between them |

To hold less than a flat gather, don't gather flat: {py:meth}`iter_batches(distributed=True) <batcher.Dataset.iter_batches>` reads a breaker's published buckets one at a time, as {doc}`Distributed scheduling </architecture/deep-dives/distribution/distributed-scheduling>` describes.

## Scaling

A single reducer's inbound rate is bounded by its NIC; the scaling is in the aggregate all-to-all, where every node reduces at once. The mergeable `partial -> combine -> finalize` algebra and credit flow control keep per-node memory bounded however wide the cluster gets.

An exchange of `m` mappers and `r` reducers opens `m x r` streams, so the reducer count is sized from the data rather than the fleet. `aggregate_reducer_count` sizes an aggregate from its group count, and `row_shuffle_reducer_count` sizes a join, sort or window from its rows. Measured on a 64-worker fleet, a 64-group aggregate given one reducer per worker spent 302 ms moving a few kilobytes through 4,096 streams, against 91 ms through one. Measure a given cluster shape with [`benchmarks/cluster/carbonite/xnode.py`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/cluster/carbonite/xnode.py).

## The disk alternative

`distributed.transport` takes three settings, resolved by `resolve_transport` in `dist/executors/ray_runtime/lifecycle.py`:

- `"flight"` forces the network shuffle. `"auto"` chooses it whenever the cluster has more than one node.
- `"disk"` forces the Arrow IPC file shuffle, where only paths pass through Ray. `"auto"` chooses it on a single node, where there's no gRPC and the page cache does the work, and whenever `distributed.shared_filesystem` is set.

Before the disk shuffle runs on more than one node, `dist/shuffle_io.py::verify_shared_scratch` proves the mount: the driver writes a random token, a task on each other node reads it back, and a node that can't fails the query up front with a {py:exc}`ConfigError <batcher.ConfigError>` naming it.

:::{dropdown} Compressing what the disk shuffle writes
`shuffle_ipc_options` in `dist/shuffle_io.py` decides from the path alone. A scratch directory on a cluster-shared mount is a network filesystem where every byte crosses the wire twice, so it gets LZ4. A node-local scratch directory honors `memory.spill_compression` and stays uncompressed under `"auto"`. Deciding from the path rather than configuration means every node agrees, since a Ray worker's config is its own process default. An Arrow IPC message records its own codec, so the read side never needs to know.
:::

## Security

Two independent layers protect the shuffle, both off by default:

- A shuffle token, `distributed.shuffle_token` or `BATCHER_SHUFFLE_TOKEN`, checked in constant time against `path[1]` before any data is served. It's one shared secret for the fleet, authenticating membership rather than a principal or a query, and it is read when the fleet is spawned.
- `distributed.tls`, through {py:class}`ShuffleTlsConfig <batcher.config.config.ShuffleTlsConfig>`, with `require_client_auth` for mutual TLS.

`distributed.require_secure_shuffle=True` makes an unsecured fleet a startup failure: TLS off fails validation, and a missing token raises {py:exc}`ConfigError <batcher.ConfigError>` before any worker is spawned.

## Code map

| Concern | File |
|---|---|
| Flight server, handler, ticket, store | `crates/bc-transport/src/{exchange,handler,ticket,store}.rs` |
| Shared-memory mmap path | [`crates/bc-transport/src/shared.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-transport/src/shared.rs) |
| Concurrent gather and fold | [`crates/bc-py/src/shuffle/`](https://github.com/stephenoffer/batcher/tree/main/crates/bc-py/src/shuffle) |
| The Ray actor hosting a worker's server | [`python/batcher/dist/flight_worker.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/dist/flight_worker.py) |
| Per-operator shuffle driving | `python/batcher/dist/flight_{aggregate,join,sort,window}.py` |
| Session, mode selection, reducer placement | [`python/batcher/carbonite/transfer/`](https://github.com/stephenoffer/batcher/tree/main/python/batcher/carbonite/transfer) |
| Store cap and credit ceilings | [`python/batcher/carbonite/policies/flow_control.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/carbonite/policies/flow_control.py) |
| Replicating published buckets | [`python/batcher/dist/shuffle_replication.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/dist/shuffle_replication.py) |
| The disk shuffle | [`python/batcher/dist/shuffle_io.py`](https://github.com/stephenoffer/batcher/blob/main/python/batcher/dist/shuffle_io.py) |

## See also

- {doc}`Credit-based flow control </architecture/deep-dives/distribution/credit-flow-control>`: what stops a mapper flooding a reducer.
- {doc}`Distributed scheduling </architecture/deep-dives/distribution/distributed-scheduling>`: who runs where, and how many reducers there are.
- {doc}`Mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>`: why a bucket can be reduced independently.
- {doc}`Fault tolerance </architecture/fault-tolerance>`: what `epoch`, replicas and the missing-file path are for.
- {doc}`Carbonite </architecture/internals/carbonite>`: the transport knobs, and who owns them.
- {doc}`Ray integration </integrations/compute/ray>`: what Ray is actually doing in this picture.
- {doc}`Configuration options </configuration/options>`: every `distributed.*` and `flow_control.*` knob.
- {doc}`Scaling benchmarks </benchmarks/results/scaling>`: what distribution buys, measured.
