# The shuffle over Arrow Flight

The *shuffle* is the all-to-all redistribution that puts every row with the same key on the
same machine, which is what a distributed group-by or join needs before it can reduce. This
page describes how Batcher moves those batches over Arrow Flight, how a reducer picks the
cheapest source for each bucket, how buckets are addressed and fetched, and when the disk
shuffle runs instead.

The shuffle is where distributed engines go to die. It moves the most bytes, it's the
all-to-all that doesn't scale politely, and the obvious implementation, handing the batches
to your cluster framework's object store, reintroduces exactly the serialization cost the
columnar engine was built to avoid.

Batcher's shuffle moves Arrow record batches worker-to-worker over Arrow Flight on gRPC.
Ray schedules the workers and carries their addresses. It doesn't carry their data.

:::{important}
The data plane bypasses the Ray object store. Bulk Arrow batches move over Arrow Flight with
credit-based backpressure; what crosses Ray is addresses, tickets, paths, row counts, and a
metrics string. Routing bulk data through Ray objects reintroduces exactly the serialization
and OOM overhead the columnar design removes.
:::

```text
   MAPPERS                                                REDUCERS
   ───────                                                ────────
   worker 0 ──partition_batches──► [b0][b1][b2][b3]
   worker 1 ──partition_batches──► [b0][b1][b2][b3]       the reducer for bucket 1
   worker 2 ──partition_batches──► [b0][b1][b2][b3]       fetches b1 from all three
                                        │                            │
        every bucket is published,      └────────────────────────────┤
        including the empty ones                                     ▼
                                        ┌──────────────────────────────────────────┐
                                        │  it picks the cheapest source available  │
                                        ├──────────────────────────────────────────┤
                                        │  same process  DIRECT_MEMORY   no copy   │
                                        │  same node     SHARED_MEMORY   ≈ memcpy  │
                                        │  another node  NETWORK         Flight,   │
                                        │                                credited  │
                                        └────────────────────┬─────────────────────┘
                                                             │
                                    folded into a running partial IN RUST
                                    (gather_combine): the intermediate never
                                    crosses back into Python
                                                             │
                                                             ▼
                                                     combine_finalize
```

## The three tiers

A reducer fetching a bucket has three possible sources, and it picks the cheapest without
any configuration. The selector is pure, taking placement in and returning a mode, so the
whole decision is the two comparisons below.

![Carbonite routes one shuffle partition by placement. The same Flight address means one process, so DIRECT_MEMORY reads from the local store with no serialization. The same node identity means one host, so SHARED_MEMORY uses Arrow IPC over a memory map, which is selected today but not yet executed. Anything else falls back to NETWORK over credit-bounded Arrow Flight.](/_static/diagrams/transfer_modes.svg)

| Source | Path | Cost |
|---|---|---|
| Same process | `DIRECT_MEMORY`: read from the local store | no copy, no socket |
| Same node, other process | `SHARED_MEMORY`: mmap a 64-byte-aligned Arrow IPC file | ≈ a memcpy |
| Another node | `NETWORK`: credit-bounded Arrow Flight | one gRPC stream |

`carbonite/transfer/locality.py::select_mode` makes the choice from the peer's Flight
address and node id. A matching Flight address means the same process, so `DIRECT_MEMORY`.
Otherwise, two known and equal node identities mean the same host, so `SHARED_MEMORY`.
Everything else is `NETWORK`. The same test is duplicated in Rust inside the concurrent
gather in `crates/bc-py/src/shuffle.rs`, so a same-host bucket is read from shared memory
*inside* the parallel fetch rather than being serialized ahead of it. Cross-node buckets
keep fanning out while the local ones are memcpy'd.

The common GPU-cluster layout packs several worker actors onto each node, so most of a
reducer's fetches are same-node but cross-process, which is exactly the tier the
shared-memory path accelerates. To measure the gap on your own hardware, run
`benchmarks/cluster/carbonite/xnode.py`, which moves an identical partition set both ways
between a producer and a consumer actor and reports the delivered throughput.

## The Flight server

Each worker process hosts one Flight server. `bc_transport::ShuffleExchange::bind_tls`
starts it over a shared `Arc<PartitionStore>`, binding `0.0.0.0:0` and advertising
`{node_ip}:{port}`, the IP coming from `ray.util.get_node_ip_address()` in
`dist/flight_worker.py`.

`FlightHandler` implements `arrow_flight::FlightService`, and **only `do_exchange` is on
the production path**. `do_get` exists but is the un-credited fetch. It isn't reachable
from `bc-py` and isn't what a reducer calls. Everything else, including `handshake`,
`get_flight_info`, `do_put`, and `do_action`, returns `unimplemented`. This isn't a
general-purpose Flight endpoint. It serves one query's buckets to one query's reducers.

Batches live in memory, not on disk:

```rust
// crates/bc-transport/src/store.rs
pub(crate) struct PartitionStore { partitions: RwLock<HashMap<String, Partition>> }
pub(crate) struct Partition { batches: Arc<Vec<RecordBatch>>, gauge: Arc<InflightGauge> }
```

Arrow IPC appears in exactly two places in the whole transport: the shared-memory mmap
file, and the disk shuffle. The Flight wire path encodes with `FlightDataEncoderBuilder`,
using LZ4 by default under `distributed.flight_compression`.

## Addressing a bucket

:::{dropdown} `ShuffleTicket`: the wire address of one mapper→reducer edge
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

It serializes to `"{plan}/{stage}/{src}/{dst}/{epoch}"` and rides in
`flight_descriptor.path[0]` of the first `DoExchange` message. `path[1]` is an auth token;
`path[2]` is an optional `"shard/nshards"` selector for striping one bucket across
several connections.

`plan_id` is minted per query in `dist/flight_worker.py` as a 63-bit value from a uuid4, so
it fits the ticket field. It exists because a session fleet actor is reused across queries,
and a reducer must not be able to fetch a crashed prior query's leftovers. `epoch` does the
same for a recompute after worker loss.
:::

:::{note}
Mappers publish **every** bucket, including empty ones. That turns a failed fetch into an
unambiguous signal: it means the worker is gone, never that the bucket happened to be empty.
:::

## Fetching

The reducer's gather is `crates/bc-py/src/shuffle.rs::drive`:

1. Co-located buckets are read straight from the local store: no socket, no credit permit.
1. The rest are spawned into a `JoinSet` bounded by a semaphore of
   `flow_control.shuffle_fetch_fan_in` (32).
1. Each remote fetch tries shared memory for a same-host peer, then falls back to Flight.
1. Arriving batches are folded into a running partial *in Rust* (`gather_combine`) or
   concatenated (`gather_concat`), so the reducer never materializes every mapper's bucket
   as a Python object first.

Point 4 matters more than it reads. Folding in Rust is what keeps the join reducer's
intermediate out of Python. On TPC-H sf10 that intermediate is 3.75M rows and roughly 106
MB per reducer, which would otherwise be built as Python `RecordBatch` objects and handed
straight back into the engine for the partial aggregate. The `execute_plan_aggregated` FFI
entry runs the join and folds the aggregate inside the engine instead, so the intermediate
never crosses the boundary.

:::{warning}
Two fan-in numbers exist and they aren't the same thing.

| Knob | Default | Bounds |
|---|---|---|
| `flow_control.shuffle_fan_in` | 8 | the *combiner tree* depth for an aggregate: how many partials one node folds, so per-node fan-in stays bounded as the cluster grows |
| `flow_control.shuffle_fetch_fan_in` | 32 | a *flat* gather's fetch concurrency |

A flat gather holds all its data anyway, so throttling its fetch buys no memory. It only
serializes the network. Sharing one value of 8 between the two makes a 16-worker shuffle pull
its buckets in two half-idle waves.
:::

## Scaling

A single reducer's inbound rate is bounded by its NIC, so there's no headroom to win back
on one node once the fetch runs at line rate. The scaling is in the aggregate all-to-all,
where every node reduces at once. Aggregate shuffle throughput grows with the node count,
because the mergeable `partial → combine → finalize` algebra plus credit flow control keep
per-node memory bounded however wide the cluster gets, so adding nodes adds reducers rather
than adding contention. Measure it for a given cluster shape with
`benchmarks/cluster/carbonite/xnode.py`.

## The disk alternative

`distributed.transport` takes three settings. The default, `"auto"`, picks between the
other two.

Setting `"flight"` forces the network shuffle: one Flight server per worker process over a
shared `Arc<PartitionStore>`, serving credit-bounded `do_exchange` streams. `"auto"`
chooses it whenever the cluster has more than one node.

Setting `"disk"` forces the Arrow IPC file shuffle, where only paths pass through Ray. It's
safe only when every worker sees the same filesystem at the same path. `"auto"` chooses it
on a single node, and whenever `distributed.shared_filesystem` is set. On one node the disk
shuffle is the *better* choice, which is why it's the default there. There's no gRPC and no
server, and the page cache does the work. The work directory is driver-local, which is why
`"auto"` won't choose it across nodes.

`resolve_transport` in `dist/executors/ray_runtime/lifecycle.py` makes the call.

### Compressing what the disk shuffle writes

The Flight wire has always compressed its batches (`distributed.flight_compression`). The
disk shuffle now makes the same trade, and it makes it the way the spill store does, by
looking at where the bytes are going rather than at what is in them.

A scratch directory on a **cluster-shared mount** is a network filesystem. Every byte a
mapper writes crosses the wire twice, once out to the mount and once back to the reducer, so
a cheap codec pays there for the same reason it pays on the remote spill tier: LZ4 runs at
around a gigabyte per second per core, well ahead of any network mount, and it gives up
quickly on data that will not compress. A scratch directory on node-local disk is fast, so it
honors `memory.spill_compression` instead and stays uncompressed under that field's `"auto"`
default.

Nothing on the read side changes. An Arrow IPC message records its own codec, so a reducer
decompresses whatever it is handed, and a file written by an earlier build still reads.
`shuffle_ipc_options` in `dist/shuffle_io.py` makes the call, and it decides from the path
alone. That is deliberate rather than incidental: a Ray worker's `active_config()` is its own
process default, not the driver's, so a codec chosen from configuration on a worker would
silently disagree with the one the driver intended. The branch where compression matters
reads no configuration and therefore agrees on every node.

## The self-limiting shared-memory mirror

The shared-memory file is a second copy of the bucket, in tmpfs, on top of the in-memory
store Flight already serves from. That's a real memory cost, and on a churning spot node
where recompute transiently doubles live state it could be the cost that kills you.

So `ShuffleSession._shm_mirror_ok()` skips writing the mirror when the pressure monitor
reports `SPILL` or worse. The reducer's read then misses, `fetch_shared` returns
`Ok(None)`, and it falls back to Flight, which is bit-identical, so single-node equals
distributed regardless. The fast path steps aside rather than risking OOM.

The mmap read itself is genuinely zero-copy: `read_mmap_zero_copy` wraps the mapping as an
Arrow `Buffer::from_custom_allocation` so the decoded arrays point *into* it and the
mapping outlives the batches.

## Security

Two independent layers, both off by default.

A shuffle token, set as `distributed.shuffle_token` or through the `BATCHER_SHUFFLE_TOKEN`
environment variable, is checked constant-time against `path[1]` before any data is served.
Separately, `distributed.tls` enables TLS on the Flight channel through {py:class}`ShuffleTlsConfig <batcher.config.config.ShuffleTlsConfig>`,
and setting `require_client_auth` there turns that into mutual TLS.

## Code map

Each concern below has a single owning file, so the transport path this page describes can
be traced end to end:

| Concern | File |
|---|---|
| Flight server, handler, ticket, store | `crates/bc-transport/src/{exchange,handler,ticket,store}.rs` |
| Shared-memory mmap path | `crates/bc-transport/src/shared.rs` |
| Concurrent gather + fold | `crates/bc-py/src/shuffle.rs` |
| The Ray actor hosting a worker's server | `python/batcher/dist/flight_worker.py` |
| Per-operator shuffle driving | `python/batcher/dist/flight_{aggregate,join,sort,window}.py` |
| Session, mode selection, reducer placement | `python/batcher/carbonite/transfer/` |
| The disk shuffle | `python/batcher/dist/shuffle_io.py` |

## See also

- {doc}`Architecture </architecture/index>`: distribution as a scheduling concern, not a second semantics.
- {doc}`Carbonite </architecture/internals/carbonite>`: the transport knobs, and who owns them.
- {doc}`Fault tolerance </architecture/fault-tolerance>`: what `epoch` and the missing-file path are for.
- {doc}`Ray integration </integrations/compute/ray>`: what Ray is actually doing in this picture.
- {doc}`Configuration options </configuration/options>`: every `distributed.*` and `flow_control.*` knob.
- {doc}`Scaling benchmarks </benchmarks/results/scaling>`: what distribution buys, measured.
- {doc}`Credit-based flow control </architecture/deep-dives/distribution/credit-flow-control>`: what stops a mapper flooding a reducer.
- {doc}`Distributed scheduling </architecture/deep-dives/distribution/distributed-scheduling>`: who runs where.
- {doc}`Mergeable algebra </architecture/deep-dives/operators/mergeable-algebra>`: why a bucket can be reduced independently.
