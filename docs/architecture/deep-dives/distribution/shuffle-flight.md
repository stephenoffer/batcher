# The shuffle over Arrow Flight

The *shuffle* is the all-to-all redistribution that puts every row with the same key on the same machine, which is what a distributed group-by or join needs before it can reduce. This page describes how Batcher moves those batches over Arrow Flight, how a reducer picks the cheapest source for each bucket, how buckets are addressed and fetched, and when the disk shuffle runs instead.

The shuffle is where distributed engines go to die. It moves the most bytes, it's the all-to-all that doesn't scale politely, and the obvious implementation, handing the batches to the cluster framework's object store, reintroduces the serialization cost a columnar engine was built to avoid.

Batcher's shuffle moves Arrow record batches worker to worker over Arrow Flight on gRPC. Ray schedules the workers and carries their addresses. It doesn't carry their data.

:::{important}
The data plane bypasses the Ray object store. Bulk Arrow batches move over Arrow Flight with credit-based backpressure. What crosses Ray is addresses, tickets, paths, row counts, and a metrics string. Routing bulk data through Ray objects reintroduces the serialization and OOM overhead the columnar design removes.
:::

The two planes are physically separate channels between the same pair of workers, and that's the half of the note above a reader skims.

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

A reducer fetching a bucket has three possible sources, and it picks the cheapest without any configuration. The selector is pure, taking placement in and returning a mode, so the whole decision is two comparisons.

![Carbonite routes one shuffle partition by placement. The same Flight address means one process, so DIRECT_MEMORY reads from the local store with no serialization. The same node identity means one host, so SHARED_MEMORY reads the bucket back through Arrow IPC over a memory map. Anything else falls back to NETWORK over credit-bounded Arrow Flight.](/_static/diagrams/transfer_modes.svg)

The following table lists each tier with the path it takes and what it costs:

| Source | Path | Cost |
|---|---|---|
| Same process | `DIRECT_MEMORY`: read from the local store | no copy, no socket |
| Same node, other process | `SHARED_MEMORY`: mmap a 64-byte-aligned Arrow IPC file | about a memcpy |
| Another node | `NETWORK`: credit-bounded Arrow Flight | one gRPC stream |

`carbonite/transfer/locality.py::select_mode` makes the choice from the peer's Flight address and node id. A matching, non-empty Flight address means the same process, so `DIRECT_MEMORY`. Otherwise, two known and equal node identities mean the same host, so `SHARED_MEMORY`. Everything else is `NETWORK`, including two addresses that are both still unknown: an empty address is "not bound yet", not a match, and `NETWORK` is the mode that is always correct and only ever slower.

The same test is repeated in Rust inside the concurrent gather in [`crates/bc-py/src/shuffle/gather.rs`](https://github.com/stephenoffer/batcher/blob/main/crates/bc-py/src/shuffle/gather.rs), so a same-host bucket is read from shared memory *inside* the parallel fetch rather than being serialized ahead of it. Cross-node buckets keep fanning out while the local ones are copied.

The common GPU-cluster layout packs several worker actors onto each node, so most of a reducer's fetches are same-node but cross-process, which is the tier the shared-memory path accelerates. To measure the gap on your own hardware, run [`benchmarks/cluster/carbonite/xnode.py`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/cluster/carbonite/xnode.py), which moves an identical partition set both ways between a producer and a consumer actor and reports the delivered throughput.

## The Flight server

Each worker process hosts one Flight server. `bc_transport::ShuffleExchange::bind_tls` starts it over a shared `Arc<PartitionStore>`, binding `0.0.0.0:0` and advertising `{node_ip}:{port}`, with the IP coming from `ray.util.get_node_ip_address()` in `dist/flight_worker.py`.

`FlightHandler` implements `arrow_flight::FlightService`, and **only `do_exchange` is on the production path**. `do_get` is the un-credited fetch. It's kept for the crate's own round-trip tests, isn't reachable from `bc-py`, and isn't what a reducer calls. Everything else, including `handshake`, `get_flight_info`, `do_put`, and `do_action`, returns `unimplemented`. This isn't a general-purpose Flight endpoint. It serves one query's buckets to one query's reducers.

### Where a published bucket lives

A published bucket is held in the worker's heap until a reducer fetches it:

```rust
// crates/bc-transport/src/store.rs
enum Body { Memory(Arc<Vec<RecordBatch>>), Spilled(PathBuf) }
pub(crate) struct Partition { body: Body, gauge: Arc<InflightGauge>, nbytes: usize }
```

That memory is the largest thing Carbonite's buffer pool can't see on its own. An operator reserves memory before it allocates, but nobody reserves a published bucket. The mapper hands it over and it stays resident. With `workers` mappers each producing `workers` buckets, a node holds its whole share of the shuffle this way.

So the store keeps a running byte total and has two ways to give memory back, both spilling the largest buckets first to local disk as Arrow IPC:

- **A cap.** `carbonite/policies/flow_control.py::shuffle_store_cap` sets it per worker when the Flight server starts, at a quarter of the memory envelope and never above `memory.hard_limit` of it. Past the cap, a new registration spills buckets until the store is back under.
- **Cooperative spilling.** When an operator can't get a reservation from the `bc-resource` pool, the pool asks the store to yield bytes before refusing. Published output is the right thing to ask first: it's finished work waiting to be collected, so spilling it costs one re-read and stalls nobody.

A spilled bucket is read back from disk on fetch and isn't put back in the heap, because the store spilled it for want of memory. The round trip returns the same batches, so spilling can't change an answer.

Arrow IPC appears in three places in the transport: the shared-memory mmap file, a spilled bucket, and the disk shuffle. The Flight wire path encodes with `FlightDataEncoderBuilder`, compressing with LZ4 by default under `distributed.flight_compression`.

## Addressing a bucket

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

It serializes to `"{plan}/{stage}/{src}/{dst}/{epoch}"` and rides in `flight_descriptor.path[0]` of the first `DoExchange` message. `path[1]` is an auth token, and `path[2]` is an optional `"shard/nshards"` selector for striping one bucket across several connections.

`plan_id` is minted per query in `dist/flight_worker.py` as a 63-bit value from a uuid4, so it fits the ticket field. It exists because a session fleet actor is reused across queries, and a reducer must not be able to fetch a crashed prior query's leftovers. `epoch` does the same for a recompute after worker loss.
:::

:::{note}
Mappers publish **every** bucket, including empty ones. That turns a failed fetch into an unambiguous signal: the worker is gone, never that the bucket happened to be empty.
:::

## Fetching

The reducer's gather is `crates/bc-py/src/shuffle/gather.rs::drive`. It works in four steps:

1. Buckets held by this worker are read straight from the local store, with no socket and no credit permit. That includes a *replica* that happens to live here.
1. The remaining buckets are grouped by the peer that holds them, and each peer is pulled over a share of a fixed stream budget. Many small buckets are packed into one stream by bytes, and one large bucket is striped across several.
1. Each remote fetch tries shared memory for a same-host peer, then falls back to Flight. When `distributed.shuffle_replication` placed copies of a bucket on other workers, those addresses follow the primary in the candidate list, so a lost mapper is re-fetched rather than recomputed.
1. Arriving batches are folded into a running partial *in Rust* (`gather_combine`) or concatenated (`gather_concat`), so the reducer never materializes every mapper's bucket as a Python object first.

Step 2 exists because a hash shuffle cuts one bucket per reducer out of every mapper. A cluster of `W` workers makes `W^2` buckets, each smaller as the cluster grows, and one stream per bucket makes the transfer rate a function of the cluster's width rather than the link. Holding the stream count fixed and packing buckets by bytes makes the rate width-independent. The rationale recorded beside the defaults in `config/config.py` measured 1.4 GiB across one 25 Gbps link at 1,608 MiB/s with 4,096 buckets, against 7,470 MiB/s at the same total when the stream count landed right.

Step 4 matters more than it reads. Folding in Rust is what keeps the join reducer's intermediate out of Python. On TPC-H sf10 that intermediate is 3.75M rows and roughly 106 MB per reducer, which would otherwise be built as Python `RecordBatch` objects and handed straight back into the engine for the partial aggregate. The `execute_plan_aggregated` FFI entry runs the join and folds the aggregate inside the engine instead, so the intermediate never crosses the boundary.

:::{warning}
Several fan-in numbers exist, and they bound different things.

| Knob | Default | Bounds |
|---|---|---|
| `flow_control.shuffle_fan_in` | 8 | the *combiner tree* for an aggregate: how many partials one node folds, so per-node fan-in stays bounded as the cluster grows |
| `flow_control.shuffle_fetch_fan_in` | 32 | how many channels a flat gather fetches at once, which also divides the per-channel byte budget |
| `flow_control.gather_streams` | 48 | concurrent Flight streams one reducer runs across all its peers |
| `flow_control.gather_inflight_bytes` | 768 MiB | decoded bytes those streams may hold between them |

A flat gather holds all its data anyway, so throttling its fetch buys no memory. It only serializes the network. Sharing one value of 8 between the tree and the flat fetch once made a 16-worker shuffle pull its buckets in two half-idle waves.
:::

## Scaling

A single reducer's inbound rate is bounded by its NIC, so there's no headroom to win back on one node once the fetch runs at line rate. The scaling is in the aggregate all-to-all, where every node reduces at once. The mergeable `partial -> combine -> finalize` algebra and credit flow control keep per-node memory bounded however wide the cluster gets, so adding nodes adds reducers rather than contention. Measure it for a given cluster shape with [`benchmarks/cluster/carbonite/xnode.py`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/cluster/carbonite/xnode.py).

That holds while there's enough shuffled volume to divide. It doesn't license widening the exchange to match the cluster. An exchange of `m` mappers and `r` reducers opens `m x r` streams, so a reducer count taken from the node count makes the *coordination* quadratic in the cluster while the bytes stay fixed.

That's why the reducer count is sized from the data rather than the fleet. `aggregate_reducer_count` sizes an aggregate, whose exchanged volume is the group count, and `row_shuffle_reducer_count` sizes a join, sort or window, whose exchanged volume is the rows. Measured on a 64-worker fleet, a 64-group aggregate given one reducer per worker spent 302 ms moving a few kilobytes through 4,096 streams, against 91 ms through one. {doc}`Distributed scheduling </architecture/deep-dives/distribution/distributed-scheduling>` has the full table.

## The disk alternative

`distributed.transport` takes three settings. The default, `"auto"`, picks between the other two, and `resolve_transport` in `dist/executors/ray_runtime/lifecycle.py` makes the call.

Setting `"flight"` forces the network shuffle described above. `"auto"` chooses it whenever the cluster has more than one node.

Setting `"disk"` forces the Arrow IPC file shuffle, where only paths pass through Ray. It's safe only when every worker sees the same filesystem at the same path. `"auto"` chooses it on a single node, and whenever `distributed.shared_filesystem` is set. On one node the disk shuffle is the *better* choice: there's no gRPC and no server, and the page cache does the work. The work directory is driver-local, which is why `"auto"` won't choose it across nodes.

### Compressing what the disk shuffle writes

The Flight wire compresses its batches under `distributed.flight_compression`. The disk shuffle makes the same trade the way the spill store does, by looking at where the bytes are going rather than at what's in them.

A scratch directory on a **cluster-shared mount** is a network filesystem. Every byte a mapper writes crosses the wire twice, once out to the mount and once back to the reducer, so a cheap codec pays there for the same reason it pays on the remote spill tier. A scratch directory on node-local disk is fast, so it honors `memory.spill_compression` instead and stays uncompressed under that field's `"auto"` default.

Nothing on the read side changes. An Arrow IPC message records its own codec, so a reducer decompresses whatever it's handed, and a file written by an earlier build still reads.

`shuffle_ipc_options` in `dist/shuffle_io.py` makes the call from the path alone, on purpose. A Ray worker's `active_config()` is its own process default, not the driver's, so a codec chosen from configuration on a worker could silently disagree with the one the driver intended. The branch where compression matters reads no configuration and so agrees on every node.

## The self-limiting shared-memory mirror

The shared-memory file is a second copy of the bucket, in tmpfs, on top of the in-memory store Flight already serves from. That's a real memory cost, and on a churning spot node where recompute transiently doubles live state it could be the cost that kills the worker.

So `ShuffleSession._shm_mirror_ok()` skips writing the mirror in two cases. The first is memory pressure, when the pressure monitor reports `SPILL` or worse. The second is a worker that shares its node with no other worker process. There the mirror has no possible reader: a same-address fetch is served from the local store and every other fetch comes from another machine.

A skipped mirror is harmless. The reducer's shared-memory read misses, `fetch_shared` returns `Ok(None)`, and the fetch falls back to Flight, which carries the same batches. It costs a memcpy's worth of latency and changes nothing else.

The mmap read itself is zero-copy. `read_mmap_zero_copy` wraps the mapping as an Arrow `Buffer::from_custom_allocation`, so the decoded arrays point *into* it and the mapping outlives the batches.

## Security

Two independent layers protect the shuffle, and both are off by default.

A shuffle token, set as `distributed.shuffle_token` or through the `BATCHER_SHUFFLE_TOKEN` environment variable, is checked in constant time against `path[1]` before any data is served. Separately, `distributed.tls` enables TLS on the Flight channel through {py:class}`ShuffleTlsConfig <batcher.config.config.ShuffleTlsConfig>`, and setting `require_client_auth` there turns that into mutual TLS.

## Code map

Each concern below has a single owning file, so the transport path this page describes can be traced end to end:

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
