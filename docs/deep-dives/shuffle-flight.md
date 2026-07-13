# The shuffle over Arrow Flight

A distributed group-by has to get every row with the same key onto the same machine. That
is the shuffle, and it is where distributed engines go to die: it moves the most bytes, it
is the all-to-all that does not scale politely, and the obvious implementation (hand the
batches to your cluster framework's object store) reintroduces exactly the serialization
cost the columnar engine was built to avoid.

Batcher's shuffle moves Arrow record batches worker-to-worker over Arrow Flight (gRPC).
Ray schedules the workers and carries their addresses. It does not carry their data.

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
                                    (gather_combine) — the intermediate never
                                    crosses back into Python
                                                             │
                                                             ▼
                                                     combine_finalize
```

## The three tiers

A reducer fetching a bucket has three possible sources, and it picks the cheapest without
any configuration.

| Source | Path | Cost |
|---|---|---|
| Same process | `DIRECT_MEMORY`: read from the local store | no copy, no socket |
| Same node, other process | `SHARED_MEMORY`: mmap a 64-byte-aligned Arrow IPC file | ≈ a memcpy |
| Another node | `NETWORK`: credit-bounded Arrow Flight | one gRPC stream |

`carbonite/transfer/locality.py::select_mode` makes the choice from the peer's Flight
address and node id. The same test is duplicated in Rust inside the concurrent gather
(`crates/bc-py/src/shuffle.rs`), so a same-host bucket is read from shared memory *inside*
the parallel fetch rather than being serialized ahead of it, so cross-node buckets keep
fanning out while the local ones are memcpy'd.

The measured gap is large. A single-node multi-actor gather (8 producers → 1 reducer) runs
at 33.6 GB/s through shared memory versus 4.5 GB/s over loopback Flight: 7.5× through the
full concurrent gather, roughly 23× point to point. That shape (several worker actors per
node) is the common GPU-cluster layout, so most of a reducer's fetches are same-node
cross-process.

## The Flight server

Each worker process hosts one Flight server. `bc_transport::ShuffleExchange::bind_tls`
starts it over a shared `Arc<PartitionStore>`, binding `0.0.0.0:0` and advertising
`{node_ip}:{port}`, the IP coming from `ray.util.get_node_ip_address()` in
`dist/flight_worker.py`.

`FlightHandler` implements `arrow_flight::FlightService`, and **only `do_exchange` is on
the production path**. `do_get` exists but is the un-credited fetch; it is not reachable
from `bc-py` and is not what a reducer calls. Everything else (`handshake`,
`get_flight_info`, `do_put`, `do_action`) returns `unimplemented`. This is not a
general-purpose Flight endpoint. It serves one query's buckets to one query's reducers.

Batches live in memory, not on disk:

```rust
// crates/bc-transport/src/store.rs
pub(crate) struct PartitionStore { partitions: RwLock<HashMap<String, Partition>> }
pub(crate) struct Partition { batches: Arc<Vec<RecordBatch>>, gauge: Arc<InflightGauge> }
```

Arrow IPC appears in exactly two places in the whole transport: the shared-memory mmap
file, and the disk shuffle. The Flight wire path encodes with `FlightDataEncoderBuilder`,
LZ4 by default (`distributed.flight_compression`).

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

`plan_id` is minted per query as a 63-bit uuid. It exists because a session fleet actor is
reused across queries, and a reducer must not be able to fetch a crashed prior query's
leftovers. `epoch` does the same for a recompute after worker loss.
:::

:::{note}
Mappers publish **every** bucket, including empty ones. That turns a failed fetch into an
unambiguous signal: it means the worker is gone, never that the bucket happened to be empty.
:::

## Fetching

The reducer's gather is `crates/bc-py/src/shuffle.rs::drive`:

1. Co-located buckets are read straight from the local store: no socket, no credit permit.
2. The rest are spawned into a `JoinSet` bounded by a semaphore of
   `flow_control.shuffle_fetch_fan_in` (32).
3. Each remote fetch tries shared memory for a same-host peer, then falls back to Flight.
4. Arriving batches are folded into a running partial *in Rust* (`gather_combine`) or
   concatenated (`gather_concat`), so the reducer never materializes every mapper's bucket
   as a Python object first.

Point 4 matters more than it reads. The join reducer used to round-trip 3.75M rows / ~106
MB of Python `RecordBatch` objects out of the engine and straight back into it for the
partial aggregate. The `execute_plan_aggregated` FFI entry now runs the join and folds the
aggregate inside the engine; the intermediate never crosses the boundary.

:::{warning}
Two fan-in numbers exist and they are not the same thing.

| Knob | Default | Bounds |
|---|---|---|
| `flow_control.shuffle_fan_in` | 8 | the *combiner tree* depth for an aggregate: how many partials one node folds, so per-node fan-in stays bounded as the cluster grows |
| `flow_control.shuffle_fetch_fan_in` | 32 | a *flat* gather's fetch concurrency |

A flat gather holds all its data anyway, so throttling its fetch buys no memory; it only
serializes the network. At the old shared value of 8, a 16-worker shuffle pulled its buckets in
two half-idle waves.
:::

## Scaling

A single reducer's inbound rate is bounded by its NIC, about 2.7 GB/s (~22 Gbps) on a T4
node, which is line rate. The scaling is in the aggregate all-to-all, where every node
reduces at once. Measured aggregate shuffle throughput: **2.0 → 6.9 → 15.2 GB/s at 2 → 4 →
8 nodes**. It grows with node count because the mergeable algebra plus credit flow control
keep per-node memory bounded however wide the cluster gets.

## The disk alternative

`distributed.transport` has three settings: `"auto"` (the default), `"flight"`, and `"disk"`.

::::{tab-set}
:::{tab-item} Flight
```text
distributed.transport = "flight"

  one Flight server per worker process, over a shared Arc<PartitionStore>
  credit-bounded do_exchange streams
  chosen by "auto" whenever the cluster has more than one node
```
:::

:::{tab-item} Disk
```text
distributed.transport = "disk"

  Arrow IPC stream files; only PATHS pass through Ray
  safe only when every worker sees the same filesystem at the same path
  chosen by "auto" on a single node, and whenever distributed.shared_filesystem is set
```
On one node the disk shuffle is the *better* choice, which is why it is the default there: no
gRPC, no server, and the page cache does the work. The work directory is driver-local, which is
why `"auto"` will not choose it across nodes.
:::
::::

`resolve_transport` in `dist/executors/ray_runtime/lifecycle.py` makes the call.

## The self-limiting shared-memory mirror

The shm file is a second copy of the bucket, in tmpfs, on top of the in-memory store Flight
already serves from. That is a real memory cost, and on a churning spot node where
recompute transiently doubles live state it could be the cost that kills you.

So `ShuffleSession._shm_mirror_ok()` skips writing the mirror when the pressure monitor
reports `SPILL` or worse. The reducer's shm read then misses, `fetch_shared` returns
`Ok(None)`, and it falls back to Flight, which is bit-identical, so single-node equals
distributed regardless. The fast path steps aside rather than risking OOM.

The mmap read itself is genuinely zero-copy: `read_mmap_zero_copy` wraps the mapping as an
Arrow `Buffer::from_custom_allocation` so the decoded arrays point *into* it and the
mapping outlives the batches.

## Security

Two independent layers, both off by default.

A shuffle token (`distributed.shuffle_token`, or `BATCHER_SHUFFLE_TOKEN`) is checked
constant-time against `path[1]` before any data is served. And `distributed.tls` enables
TLS or mTLS on the Flight channel (`ShuffleTlsConfig`, with `require_client_auth` for
mutual auth).

## Code map

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

:::{seealso}
- [Architecture](../architecture/index.md): distribution as a scheduling concern, not a second semantics
- [Carbonite](../internals/carbonite.md): the transport knobs, and who owns them
- [Fault tolerance](../architecture/fault-tolerance.md): what `epoch` and the missing-file path are for
- [Ray integration](../integrations/ray.md): what Ray is actually doing in this picture
- [Configuration options](../configuration/options.md): every `distributed.*` and `flow_control.*` knob
- [Scaling benchmarks](../benchmarks/scaling.md): the 2.0 → 6.9 → 15.2 GB/s figures above
- [Credit-based flow control](credit-flow-control.md): what stops a mapper flooding a reducer
- [Distributed scheduling](distributed-scheduling.md): who runs where
- [Mergeable algebra](mergeable-algebra.md): why a bucket can be reduced independently
:::
