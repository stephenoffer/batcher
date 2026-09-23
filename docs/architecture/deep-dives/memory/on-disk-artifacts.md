# On-disk artifacts

This page catalogues everything Batcher writes to disk while a query runs, and the three properties every artifact has to have. Read it when you add a code path that writes bytes, or when you need to know what a running query has left on a shared machine.

## What counts as an artifact

An artifact is any file Batcher creates that holds rows, plan text, or measured statistics.
Whether it is temporary does not matter. A spilled partition is deleted
within seconds and still holds the query's actual data for as long as it exists. A scratch
volume on a Ray worker is a volume other tenants mount.

The result cache is on the list once it runs out of memory. `carbonite/cache.py` holds `pyarrow.Table` objects in process memory, and under the default `MEMORY_AND_DISK` storage level an entry its budget sheds is demoted to a disk tier rather than dropped.

## The catalogue

Every artifact below is Arrow unless the format column says otherwise. The two Arrow encodings
differ in one way that decides which is used. The **stream** format is append-only and read
front to back. The **file** format carries a footer of block offsets, so a reader can seek to
one batch or memory-map the whole thing.

| Artifact | Written by | Format | Lands in |
|---|---|---|---|
| Grace spill partition | `bc-runtime::agg::spill::DiskSpillStore` | IPC stream | `bc-spill-{pid}-{seq}/part-{i}.arrow` under the spill root |
| Tiered spill bucket | `carbonite/spill/writer.py::BucketWriter` | IPC stream | the spill directory, or `memory.spill_remote_uri` |
| Demoted cache result | `carbonite/cache_disk.py::DiskCacheTier`, through the tiered spill store | IPC stream | the spill scratch directory, up to `memory.result_cache_disk_max_bytes` |
| Disk shuffle bucket | `dist/shuffle_io.py::IpcWriter` | IPC stream | the shuffle scratch directory |
| Flight gather staging file | `bc-py::write_gather_file` | IPC stream | the reducer's work directory |
| Same-node shuffle bucket | `bc-transport::shared` | IPC file | `/dev/shm/batcher_shm/` |
| Process-pool UDF shard | `core/udf/processes.py` | IPC file | `/dev/shm`, or the temp directory |
| Streaming checkpoint state | `io/formats/streaming/checkpoint/state_store.py` | IPC file | `<checkpoint_location>/state/batch-{id}.arrow` |
| Cached remote file | `io/_file_cache.py::FileBytesCache` | the source file, verbatim | `batcher_file_cache/` under the scratch volume |
| Event log document | `api/terminal/event_log.py` | JSON | `~/.batcher/logs` |
| Learned statistics | `metadata/backends/sqlite.py` | SQLite | `~/.batcher` |

## The three properties

Every artifact above answers the same three questions, and a write site that answers only two
of them is the shape this list keeps regrowing in.

### Owner-only from the moment it exists

An artifact holds the query's own rows, and the paths above are shared. `/dev/shm` and `/tmp`
are world-writable, a cluster mount is shared between tenants, and a node scratch volume is
shared with whatever else the node is running. At the default umask a new file lands 0644. A
new directory lands 0755.

So the mode is set in the `open` call, never by a following `chmod`. A chmod leaves a window in
which the rows are world-readable, and a reader that wins that race gets everything. One helper does this on each side of the boundary:
`_internal/paths.py::open_private` and `private_dir` in the control plane, and
`bc_arrow::create_private_file` and `create_private_dir` in the data plane. Both live in the
lowest module their callers share. The alternative is a copy per subsystem, and copies drift.

For the same-node shuffle the property is sharper than confidentiality. `/dev/shm` is
writable, so at 0644 a local user could *plant* a well-formed bucket under a ticket a reducer
is about to fetch. A planted file that decodes cleanly is read as authoritative shuffle data
and silently changes the answer.

[`tests/integration/test_artifact_permissions.py`](https://github.com/stephenoffer/batcher/blob/main/tests/integration/test_artifact_permissions.py) drives each real writer and stats what landed.
It is written that way deliberately: the helpers are ten lines and obviously correct in
isolation, and what actually rots is a new write site that does not reach for them.

### Buffered

Arrow's IPC writer issues a separate `write` per message *and* per buffer within it. A batch
with `k` columns therefore costs on the order of `2k` syscalls, most of them a few KB of
validity or offset data. Written straight to a file that is one syscall per buffer, and a spilled bucket of
a few thousand morsels over a dozen columns is hundreds of thousands of syscalls for bytes that
coalesce into a handful of large writes.

Buffering is invisible to the reader, because the IPC bytes are identical either way, so it is
pure throughput. Two shapes exist, and the difference is how many writers are open at once. The
grace spill store holds one writer per partition, and the partition count grows with skew, so it budgets 32 MiB *in total* and divides it, never below 8 KiB or above 1 MiB per writer
(`bc-runtime::agg::spill::store::write_buf_capacity`). The shm publisher and the Flight gather write
exactly one file at a time, so each takes a fixed 1 MiB buffer that cannot multiply.

A buffered writer has one failure mode worth naming. Dropping it flushes the tail and discards
any error doing so. That publishes a truncated file, which reads back as a short bucket rather
than as a failure. Every buffered path here calls `into_inner` explicitly to surface that
error.

### Compressed by what the link costs

Compression trades a core against a device, and the device sets the exchange rate. So the
decision is made per path, not globally:

| Path | Under `"auto"` | Why |
|---|---|---|
| Rust grace spill | Zstd for a blob-bearing schema, otherwise none | On fast local disk, compressing numeric or string state costs more CPU than the I/O it saves. Only blob payloads win. |
| Flight gather staging | The same rule, through `SpillCodec::Auto` | Same rows, same disk, so the same answer. Restating the policy is how a blob-bearing gather goes out uncompressed beside a compressed spill. |
| Python local spill tier | None | The same measurement, on the same class of disk. |
| Python remote spill tier | LZ4 | Object storage is slow and priced by the byte, so a cheap codec always pays. |
| Disk shuffle on a shared mount | LZ4 | Every byte crosses the wire twice, once to the mount and once back to the reducer. |
| Disk shuffle on node-local scratch | None | Fast disk, so it honours `memory.spill_compression` and stays uncompressed under the default. |
| Same-node shm shuffle | None | The reader memory-maps the file and decodes zero-copy. Compressing it would force a copy of every buffer. |

No read path needs to know any of this. An Arrow IPC message records its own codec, so a reader
decompresses whatever it is handed, and a file written by an older build still reads. That is
also why the choice is result-invariant: it trades CPU for bytes and nothing else.

## Where the bytes go

Two questions decide where an artifact lands, and each has exactly one function answering it.

`_internal/site/scratch.py::spill_scratch_dir()` resolves *which disk this process spills to*. It takes the configured
`memory.spill_dir`, else the best measured node-local volume, else the system temp directory.
The hardware fingerprint that keys every learned spill threshold reads the same function, so a
learned threshold names the disk the spill actually landed on. When those two disagreed, the
fingerprint described a container's overlay while the spill went to the node's NVMe, and two
machine classes that behave nothing alike were merged into one.

`dist/shuffle_io.py::shared_scratch_root()` resolves *which directory every node can see*, which is a
different question. The disk shuffle passes only paths between Ray tasks, so a path has to
resolve on whichever node the reducer lands on. That prefers a cluster-shared mount and falls
back to node-local scratch only where no shared mount exists.

Configure both through `MemoryConfig`:

```python
import dataclasses

import batcher as bt

base = bt.Config()
tuned = base.replace(
    memory=dataclasses.replace(
        base.memory,
        spill_dir="/mnt/local_storage/batcher",
        spill_compression="zstd",
        spill_remote_uri="s3://my-bucket/batcher-spill",
    )
)
print(tuned.memory.spill_compression)
```

```text
zstd
```

## Cleanup

A spill directory has the fullest lifecycle of anything in the catalogue, and the two ends
a `Drop` cannot reach are what the pid in its name is for.

![What a spill puts on disk, and its whole lifecycle. Under the spill root, which is memory.spill_dir or else the node's local scratch, each store owns a directory bc-spill-{pid}-{seq} holding part-0.arrow through part-N.arrow, one Arrow IPC stream each, made owner-only where the filesystem allows. A partition is a hash bucket for the grace operators and a sorted run for the external sort, in the same file format. A writer opens on the first append, so a bucket that received no rows has no file at all, and the codec comes from the first batch's schema: ZSTD for blob-bearing columns, none for anything else. The ordinary lifecycle is created, appended with rows and bytes counted per file, then read back once in the merge phase before the partition is released, and the store removes its own directory on drop, on success, on an error and on a panic alike. Two ends fall outside that: a SIGKILL leaves the directory behind, because the OOM killer picks the spilling process, and it is reclaimed by an orphan sweep that removes only directories whose embedded pid is no longer a live process, which is what keeps a concurrently spilling sibling safe. A truncated IPC stream reads back as a shorter valid one, so every read is checked against the row count taken on the way in.](/_static/diagrams/spill_artifacts.svg)

Each artifact is removed by the thing that created it. `DiskSpillStore` has a `Drop` that
removes its directory. `dist/spill/buckets.py::spill_scratch` removes a work directory it allocated and leaves an
operator-configured one alone. The cache's disk tier is process-scoped scratch rather than a durable store, and `DiskCacheTier.clear()` deletes its directory. The file cache evicts least-recently-used entries to stay under
its byte budget.

A remote file larger than the whole file-cache budget is never admitted. Caching it would
evict every other entry and then, with nothing else left to drop, the entry itself. No single
file can push the budget over. The read goes straight to the object store
instead, and `file_cache_max_bytes` bounds the volume as stated rather than approximately.
Concurrent readers that miss the same file share one download rather than each fetching a
copy.

Two cases survive a crash by design. The streaming checkpoint is durable on purpose, and
`prune_state` bounds it by deleting snapshots older than the last commit. A spill directory
orphaned by a killed process is swept by name on the next run, which is what the pid in
`bc-spill-{pid}-{seq}` is for.

## See also

- {doc}`Spilling </architecture/deep-dives/memory/spilling>`: when a query starts writing these files at all.
- {doc}`The buffer pool </architecture/deep-dives/memory/buffer-pool>`: the reservation whose failure starts a spill.
- {doc}`The shuffle </architecture/deep-dives/distribution/shuffle-flight>`: what the shuffle buckets are for.
- {doc}`Streaming </user-guide/moving-data/streaming/index>`: the checkpoint artifact, from a user's point of view.
- {doc}`Carbonite </architecture/internals/carbonite>`: the subsystem that owns the spill decision.
- {doc}`Arrow and memory </architecture/deep-dives/memory/arrow-memory>`: the IPC format every artifact here shares.
