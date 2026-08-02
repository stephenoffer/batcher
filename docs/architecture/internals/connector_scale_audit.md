# Connector scale audit — TB/PB datasets, up to millions of files

Measured 2026-07-13. Numbers below are from a real many-files corpus on local NVMe
(`/tmp/manyfiles/n{1000,10000,50000}`, tiny Parquet files — the variable under test is the
*file count*, not the bytes). Object storage is strictly worse per file: a footer GET is
~40 ms against ~0.1 ms local, partially offset by the 64-way read pool.

**Verdict: no connector family currently reaches a million files.** The table formats
(Delta/Iceberg) have the right *architecture* — a manifest that answers "which files, how many
rows, what bounds" without touching the data — and then throw it away by building a Python
object per data file at plan time. Plain file sources have no manifest at all and pay
O(files) footer round trips, several times per query. The SQL/NoSQL connectors mostly cannot
be sharded at all, and three of them lose data silently the moment you shard them.

The pruning layer (`io/stats/file_skipping.py`, `io/stats/lakehouse_manifest.py`) is the
exception and is genuinely excellent: fully vectorized over the file dimension with pyarrow
compute, no Python per file, sound three-valued logic. Everything else should be built the way
it is.

---

## 1. The file-source spine — governs ~40 of the 57 connectors

Everything under `io/formats/{structured,semistructured,ml,multimodal,unstructured}` inherits
`io/base/source.py::FileSource`. Its per-query, driver-side cost is linear in the file count
with no ceiling.

Measured, per file:

| phase | ms/file | at 1M files | notes |
|---|---|---|---|
| `expand()` (listing) | 0.014 | ~14 s | materializes + **sorts** every path |
| `collect_source_stats()` | 0.34 | ~5.7 min | reads every footer |
| `row_count()` | 0.65 | ~11 min | reads every footer |
| `splits()` | 0.74 | ~12 min | reads every footer, one `Split` per file |
| `partition_descriptors()` (distributed) | 1.15 | **~19 min** | plans splits **twice** |

Peak driver memory in `splits()` grew 68 MB → 161 MB across 1k → 50k files (~3.2 KB/file),
which extrapolates to **~3.2 GB of `Split` objects at 1M files** — before a byte is read.

### S1. `splits()` emits one `Split` per file and never coalesces. **Fatal.**
`io/base/source.py:319`. 1M files = 1M objects, ~3.2 GB, all pickled and shipped. The
coalescing primitive already exists (`io/splits/parquet.py:222::pack_row_groups`); the
`target_size` argument is accepted and ignored by every lakehouse source
(`# noqa: ARG002` in all four).

### S2. The distributed planner plans the whole thing twice. **Fatal, and free to fix.**
`dist/executors/partition_io/_sources.py:81-86`:
```python
fine = plan_splits(source, predicate=predicate)
floor = max(1, workers) * max(1, _SCAN_PREFETCH)
if len(fine) <= floor:
    return fine
coalesced = plan_splits(source, target_size=_SPLIT_TARGET_BYTES, predicate=predicate)
```
Any dataset above the fan-out floor — i.e. *every* large one — runs a second complete plan,
re-reading every footer. **Measured 2.1x**: 28.4 s → 58.3 s at 50k files. And because the
lakehouse sources ignore `target_size`, the second plan returns the identical split list and
is thrown away.

### S3. The footer cache is 1,024 entries. **Fatal at scale.**
`io/splits/parquet.py:58` — `@lru_cache(maxsize=1024)`. At 1M files it thrashes completely, so
each of the three-to-four O(files) passes above re-reads every footer from scratch.

### S4. Listing is eager, sorted, and unpruned.
`io/filesystem.py:227::expand` returns a fully-materialized sorted list. No pagination, no
streaming, and partition pruning cannot be applied until after every key is in memory.

### S5. `iter_batches()` is not streaming — it materializes a whole file, times 16. **Fatal.**
The `_read_file` contract (`io/base/source.py:370`) returns `list[pa.RecordBatch]` — an entire
decoded file — and the default `_iter_file` (`:373`) merely yields from that list. **Only 2 of
the 18 file connectors override `_iter_file` at all** (Parquet, point cloud); for the other 16
the "streaming" API holds the whole file in memory.

Measured: `iter_batches()` over a single 225 MB CSV grew peak RSS by **498 MB — 2.2x the file
size**. A genuinely streaming reader holds one batch.

And the multi-file path defeats even the two that do stream (`io/base/source.py:222-225`):
```python
depth = min(len(files), max(2, min(available_cpu_count(), _ITER_READAHEAD_FILES)))  # up to 16
def _read(f: str) -> list[pa.RecordBatch]:
    return list(self._iter_file(f, self._file_proj(f, projection)))   # collapses the stream
```
`list()` around the generator turns Parquet's real row-group stream back into a whole-file
materialization, and up to **16 files are held concurrently**. Peak memory is therefore
`16 x decoded-file-size`, independent of batch size: 1 GB shards → ~35 GB per worker. This is
the constraint that caps *file size* at PB scale, as distinct from the file-*count* constraints
above. Avro compounds it further (`avro.py:99`: `data = fh.read()` — the whole compressed file,
before decoding).

### S6. Most formats cannot split within a file, and cannot cheaply count rows.
Of the 18 `FileSource` connectors, only 7 have sub-file split granularity (`_file_splits`:
Parquet/ORC/Arrow/Feather/IPC row-groups, CSV/JSON byte ranges) and only 7 have a cheap
`_file_row_count`. **Avro, TFRecord, WebDataset, Protobuf, XML, msgpack, logs, documents, Excel
and numpy have neither.** For those, max parallelism equals the file count (a PB in a few large
shards cannot be parallelized at all), and `row_count()` returns `None`, so `_balance`
(`partition_io/_sources.py:97`) weights every split as 1 and the optimizer plans blind.

---

## 2. Table formats — right architecture, defeated by a Python object per file

### Delta
- **`_snapshot.py:311-321`** — `_partition_expressions` builds a pyarrow `Expression` per file
  in a nested Python loop. Even the *unpartitioned* path allocates 1M `pds.scalar(True)`
  objects. **Fatal.**
- **`_snapshot.py:275-285`** — `_pruned_dataset` builds 1M `ParquetFileFragment`s + a 1M-entry
  dict + two more 1M-element lists. Multi-GB. **Fatal.**
- **`source.py:73`** — every *worker* rebuilds the **unpruned** index of all 1M files
  (`dataset_index(None)`) in order to look up one file. Each worker pays the full O(all files)
  construction. **Fatal.**
- **`_snapshot.py:189-194`** — `rows_by_path()` builds a 1M-key dict *after* pruning already
  selected 10 survivors. Trivial fix: take `num_records` from the already-filtered manifest.
- **`source.py:186`** — `read()` is `list(self.iter_batches(...))`; the streaming reader is
  right there and is immediately collected into driver RAM.
- **`stream.py:106`** — the change feed is `read_all()`-ed whole, and `splits()` returns one
  `WholeSourceSplit`. A restarted stream with a backlog OOMs the driver.

*Done right:* snapshot caching + version pinning (`_snapshot.py:324`), `row_count()`/
`statistics()` answered from the manifest with `pc.sum` (no data scan), and pruning correctly
applied *before* fragment construction.

### Iceberg
- **`source.py:62-68`** — `_table()` is never memoized: a catalog `load_table` (REST/Glue +
  `metadata.json` GET) on every call, and **two per split** on the worker
  (`source.py:344-346`). 1M files = 2M catalog round trips. **Fatal.**
- **`source.py:196` → `_manifest.py:48`** — `statistics()` calls
  `table.inspect.data_files()`, which fetches **every manifest of the snapshot** (no
  partition/predicate pruning) and builds a Python dict-of-dicts per file per column. At 1M
  files × 20 columns that is ~20M nested dicts, on the driver, at plan time, even for a query
  touching one partition. **Fatal.**
- **`source.py:265`** — `list(scan.plan_files())` materializes every `FileScanTask`, each
  carrying per-column bounds dicts. **Fatal.**
- **`source.py:369-381`** — `IcebergTableSplit.row_count()` and `identity()` reference
  `self._data_file_path`, which is **not in `__slots__` and never assigned**. `identity()`
  raises `AttributeError` unconditionally. **Bug, not just a scale issue.**
- **`source.py:123`** — `read()` uses `scan.to_arrow()`; pyiceberg's own docstring says "All
  rows will be loaded into memory at once."

*Done right:* the predicate *is* passed into `plan_files()`, so pyiceberg's manifest evaluator
skips whole manifests — the correct layer. `iter_batches` uses `to_arrow_batch_reader()`.

### Hudi
- **`hudi.py:290-291`** — a *single* log file anywhere in a merge-on-read table forces
  `[WholeSourceSplit(self)]` → the entire PB table through the driver. MoR is the common Hudi
  deployment. **Fatal.**
- **`hudi.py:118-123`** — every split rebuilds the whole Hudi timeline to read one base file.
  1M timeline loads across the cluster. **Fatal.**
- **`hudi.py:258-261`** — `statistics()` returns no `columns`, so Kyber has **no zone map at
  all** for Hudi. No file skipping is possible.
- **`hudi.py:199-204`** — a filter-translation failure silently falls back to a full unfiltered
  TB scan.

### Delta Sharing
- **`delta_sharing.py:103-122`** — one `json.loads` + one dict (3×ncols keys) per file, then
  `pa.Table.from_pylist` must infer a schema by scanning all 1M heterogeneous dicts. Not cached;
  rebuilt by `read()`, `iter_batches()` and `splits()` separately. **Fatal.**
- **`delta_sharing.py:247-252`** — `row_count()` opens a Parquet footer per split, and
  `_balance` calls it on every split → **1M serial pre-signed-URL GETs on the driver** — for a
  number that `_stats_manifest` already parsed and threw away. **Fatal, and free to fix.**
- **`delta_sharing.py:174,235`** — `schema()` calls `pq.read_table()` and discards the data.
  Should be `ParquetFile(...).schema_arrow`. Happens per worker, per file.

---

## 3. SQL / NoSQL / streaming

### The two cross-cutting defects

**Predicate pushdown is dropped on every distributed read.** `io/source/read.py:49` and
`dist/executors/scan_read.py:396` both sniff the signature and pass the predicate only if the
source opts in. **Only 5 of 57 sources do** — and they are exactly the lakehouse ones:

```
splits() ACCEPTS predicate : ['databricks', 'delta', 'delta_sharing', 'hudi', 'iceberg']
splits() DROPS   predicate : 52 sources
```

So on the only path that can read a TB, the `WHERE` never reaches the server; the full table
crosses the wire and the engine's `Filter` throws it away. BigQuery is the sharpest instance —
`_create_session(predicate)` already knows how to set `row_restriction` and `selected_fields`
server-side, and `splits()` (`bigquery.py:219`) simply doesn't pass them.

**`schema()` executes the query and materializes the whole result.** `api/session.py:157` calls
`source.schema()` eagerly at `bt.read.<connector>(...)`, before a single operator is declared.
In `adbc.py:59-69`, `odbc.py:49-62`, `clickhouse.py:44-52`, `connectorx.py:76`,
`databricks.py:58-71` and `bigquery.py:61-67`, `schema()` reduces to fetching the entire result
set and reading `.schema` off it. `bt.read.clickhouse("SELECT * FROM events")` OOMs the driver
at *construction*. Every one of these drivers has a metadata-only path (`LIMIT 0`, ADBC
`adbc_get_table_schema`, BigQuery's already-computed `session.arrow_schema`); none is used.

### Silent data loss — the three worst bugs found

**`couchbase.py:103-107` and `neo4j.py:105-109` — parallelism silently truncates the table.**
```python
def _enumerate_partitions(self) -> list[_Window]:
    segments = max(1, self._partition_spec.segments)
    if segments == 1:
        return [(0, 0)]                                  # unbounded — correct, serial
    return [(i * _WINDOW_ROWS, _WINDOW_ROWS) for i in range(segments)]
```
`_WINDOW_ROWS` is a constant 100,000. With `segments=8` this reads rows `[0, 800_000)` and
**silently discards every row past 800,000**. It is not a cover of the collection; it is a
fixed prefix. A billion-row collection returns 800k rows and no error. Turning on parallelism
— the thing you do to handle scale — is what loses the data. (Neo4j additionally emits
`SKIP`/`LIMIT` with **no `ORDER BY`** when `order_by` is unset, so even the truncated prefix is
nondeterministic and the windows overlap.)

**`kinesis.py:99-111`** caches the shard list once. A PB-scale stream reshards routinely; new
shards are never discovered → silent data loss for the process lifetime. When a shard closes,
`NextShardIterator` is `None` and the code keeps polling the stale iterator instead of
following the child shards.

**`kafka.py:136-138`** commits the offset at *poll* time, before `streaming_query.py:335`
processes and writes to the sink. A crash in between loses the batch — at-most-once, not the
exactly-once the docstring claims.

### The autoloader killer — measured

`seen_store.py:90-92`:
```python
cur = self._conn.execute("SELECT path FROM seen_files")   # the ENTIRE table
known = {row[0] for row in cur.fetchall()}
return [c for c in candidates if c not in known]
```
To ask whether **10 new files** are unseen, it loads every path ever seen into a Python set —
on every discovery pass:

| files already seen | `unseen(10 new)` | peak memory |
|---|---|---|
| 10,000 | 17 ms | 2 MB |
| 100,000 | 226 ms | 20 MB |
| 1,000,000 | **2,612 ms** | **185 MB** |

Perfectly linear. At 100M files ever seen: ~260 s and ~18 GB, per pass, forever. `path` is
already the `PRIMARY KEY` — `WHERE path IN (...)` over the candidate batch is an index probe at
O(candidates). **One-line fix.**

Alongside it: `autoloader.py:93` does a full unpaginated LIST every pass and applies the
`> max_seen` filter in Python *afterwards* (S3's `start_after` is never plumbed down);
`autoloader.py:111` marks files seen **before** they are read (at-most-once, and `splits()` is
not idempotent); `mark()` commits per file, so discovering 1M files means 1M fsyncs — that is
why seeding this table for the measurement above had to bypass the public API.

The `> max_seen` filter turned out to be worse than a missed pushdown: it is silent data loss
whenever names are not monotonic, which is the normal case (`part-00000-<uuid>.parquet`). See
"the autoloader's lexical filter" in section 7.

### Other hard walls

| connector | defect | severity |
|---|---|---|
| `connectorx.py:66-74` | one split; `cx.read_sql` merges all partitions into one driver-resident table | fatal |
| `odbc.py:49-62` | one split; `fetchallarrow()`; `fetcharrowbatches()` exists and is unused | fatal |
| `clickhouse.py:155` | streams correctly, but one split → one worker pulls the whole result | fatal |
| `databricks.py:73-80` | warehouse path collapses Cloud Fetch's N result files into one `fetchall_arrow()` | fatal |
| `nosql/base.py:132-142` | `PartitionSpec.segments` defaults to **1** — every NoSQL source except Cassandra is a single serial cursor out of the box | fatal-by-default |
| `redis.py:120-136` | each of N splits scans the **whole** keyspace and discards non-matching slots (N× the work, not 1/N); one RTT per key for `KEYSLOT`, another for `GET`; `_is_cluster` misdetects a plain client so the RTT path always fires | fatal |
| `mongo.py:189-200` | boundary discovery via `skip(offset)` — the server walks `offset` docs per boundary; quadratic. Shard/chunk ranges never consulted | fatal |
| `elasticsearch.py:138-165` | ES\|QL path runs the whole query for the schema, then again for the data, in one split | fatal |
| `eventhubs.py:104-113` | constructs a consumer **inside** the poll loop and re-seeks to `starting_position` ("-1" = start of stream) every poll — **never advances**; leaks an AMQP link per partition per poll | fatal |
| `pubsub.py:71` | one split for the whole subscription; `offset` derived from `hash()` of a string, which is per-process randomized and therefore unstable across workers | fatal |
| `pulsar.py:112-116` | up to 16,384 sequential blocking `receive()` calls per poll; `batch_receive()` unused | slow |
| `hbase.py:109-125` | region splits are correct, but projection is applied client-side *after* every column family crosses Thrift; no server-side filter | slow |

---

## 4. What is already right — build everything else this way

- **`io/stats/file_skipping.py`** — the whole predicate tree evaluated with `pyarrow.compute`
  over the file dimension. Zero Python per file. Sound three-valued logic (a missing statistic
  keeps the file; `None` means "keep everything"). This is the model.
- **`io/stats/lakehouse_manifest.py`** — `pc.sum`/`pc.min`/`pc.max` over manifest columns. Cost
  is O(columns), not O(files). One implementation serves Delta, Iceberg and Delta Sharing.
- **Snowflake** (`snowflake.py:101-109`) — `get_result_batches()`: one submission, one picklable
  `ResultBatch` per cloud-storage chunk, fetched independently. Textbook.
- **ADBC/FlightSQL** (`adbc.py:224-237`) — `adbc_execute_partitions`: real server-side
  partitions, shippable to workers.
- **BigQuery** (`bigquery.py:167-192`) — the Storage Read API with `selected_fields` and
  `row_restriction` is exactly right; only the wiring in `splits()` drops it.
- **Databricks lakehouse path** — vends Unity credentials, bypasses the warehouse, delegates to
  `DeltaSource.splits(target_size, predicate)`. The **only** connector that gets predicate-aware
  split pruning end to end.
- **Cassandra** (`cassandra.py:180-193`) — Murmur3 token ranges: a genuinely disjoint,
  exhaustive cover, 64 segments by default.
- **Streaming state is bounded on purpose** — watermark eviction + a hard
  `streaming_state_budget_bytes()` cap that fails loudly rather than OOMing; `prune_state` after
  every commit; `_progress` is a `deque(maxlen=100)`. The unbounded-set anti-pattern is *only*
  in `seen_store.unseen()`, and that is a query bug, not a state-model bug.

---

## 5. Fixed in this pass

Three of the findings were cheap enough to fix immediately, and the first was too dangerous
to leave.

**`couchbase.py` / `neo4j.py` — the partition cover. (Silent data loss.)** Both now build their
windows with `nosql/base.py::offset_windows`, which is a real cover: sized from an actual
`COUNT(*)` so it is balanced, terminated by an **unbounded tail** so it is exhaustive even if
rows land after the count, and degrading to a single serial window when no count can be had —
slow and right rather than fast and short. Neo4j additionally refuses to split at all when the
query has no `ORDER BY`, because `SKIP`/`LIMIT` over an undefined order is not a cover no matter
how it is sized. `tests/unit/test_offset_window_cover.py` (47 cases) pins exhaustiveness and
disjointness as arithmetic — no server needed, which is precisely why the bug survived.

**`seen_store.unseen()` — the autoloader wall.** Now an index probe on the `path` PRIMARY KEY
instead of `SELECT path FROM seen_files` into a Python set. Measured at 1,000,000 files already
seen, asking about 10 new ones: **2,612 ms / 185 MB → 0.22 ms / ~0 MB**, and flat rather than
linear (0.39 ms at 10k, 0.28 ms at 100k, 0.22 ms at 1M).

**`_scan_splits` double-planning.** The coalesced plan is now attempted *first*, so the large
dataset — the one whose plan is expensive — plans once, and only a small one (where a second
plan is cheap by definition) can pay for two. Same decision, inverted cost. Measured **2.1x →
1.0x**; distributed planning at 50,000 files: 58.3 s → 30.7 s.

## 6. Priority for what remains

1. **Thread `predicate`/projection through `splits()` and `Split.read()`** — the opt-in
   machinery already exists; 52 sources just have to opt in. BigQuery is two lines and buys the
   most. **Largely done since this audit:** 38 of 57 sources now accept a predicate. The
   multimodal sources were the remaining gap and now prune on the columns a *listing* already
   knows (`uri`/`size`/`mime`, `io/formats/multimodal/_pruning.py`) — exact rather than
   conservative, since those are the values themselves and not per-chunk bounds. Still
   dropping it: `binary`, `text`, `lance`, `hdf5`, `zarr` and the streaming sources.
2. **Make `iter_batches` actually stream** (S5) — **the spine is fixed.** The `list()` around
   the per-file generator is gone, replaced by `io/base/_readahead.py`: an order-preserving
   read-ahead bounded by *bytes* rather than by file count, because a file count says nothing
   about memory when one row can be a 200 MB video. Measured **324 MB → 69 MB (4.7x)** on a
   single 132 MB CSV, and now flat in file size. (The bound is per in-flight file, not global
   — a single shared budget deadlocks, since the consumer only releases credit by draining the
   *head* file.)

   Real `_iter_file` implementations landed for **CSV, ORC, Arrow IPC, Avro** and **text
   line-mode** (748 MB → 57 MB, 13x, on a 128 MB log). **Still whole-file:** xml, documents,
   webdataset, tfrecord, msgpack, protobuf, excel, numpy, hdf5, zarr, point-cloud ASCII-PLY,
   logs, binary and media. For those the byte bound still throttles *delivery*, but the decode
   itself materializes one whole file per in-flight file.
3. **Coalesce splits** (S1) and **raise/replace the 1,024-entry footer cache** (S3).
4. **Delta `_partition_expressions` / `_pruned_dataset` / worker-side unpruned index** — the
   three that keep Delta from being the answer at 1M files.
5. **Iceberg `_table()` memoization** and **`statistics()`** (stop calling
   `inspect.data_files()`; read the manifest through `lakehouse_manifest` like Delta does).
6. **`eventhubs`** (cannot make progress at all) and **`kinesis`** (silent loss on reshard).
7. **Metadata-only `schema()`** for the six SQL connectors that currently execute the query.

## 7. Added since this audit

Not gaps closed but capability that did not exist, all in the multimodal/unstructured path:

- **Per-file error tolerance.** The IO layer had **zero** `try`/`except` in its read spine, so
  one corrupt file aborted a 10,000-file read. `on_error="skip"` (file and media sources) drops
  the file, records it, and carries on; `corrupt_files()` is the audit trail, because a
  silently-partial read is worse than a loud failure.
- **Byte-aware blob batching.** `MediaSource`/`BinarySource` batched by a fixed `batch_files=64`
  and *discarded* the `target_size` argument to `splits()`. 64 videos was 12.8 GB in one batch.
  Now bounded by count **and** bytes, with one shared chunking definition so `splits()`,
  `read()` and `iter_batches()` cannot disagree — a disagreement would make a distributed read
  return different batches from a single-node one.
- **Media statistics.** A media source reported no `statistics()` at all and its splits carried
  no `rows`, so Kyber planned a directory of 200 MB videos exactly like one of 4 KB thumbnails.
  It now reports an exact row count, real `byte_size`, and an exact `size` zone map — all from
  the listing and a stat the batching already performs.
- **64-bit blob offsets.** Blob columns were `binary` (32-bit offsets), which overflows at 2 GB
  *per batch* — reachable with 64 x 32 MB files. Now `large_binary`. This required teaching the
  Rust image/audio kernels to accept both offset widths; they took `BinaryArray` only, so the
  fix would otherwise have broken `.image.decode()` on exactly the inputs it exists for.
- **Avro temporal/decimal writes.** The writer mapped every temporal and decimal Arrow type to
  Avro `string`, and fastavro rejects `date`/`datetime`/`Decimal` values against a string
  branch — so writing *any* such column raised. The reader had always mapped the logical types
  back correctly; this is the write side of that map.
- **Avro multi-branch unions.** `_arrow_type` mapped a union to its **first non-null branch**,
  so `["null", "long", "string"]` was advertised as `int64` — and the read then *raised* an
  opaque pyarrow conversion error, making a valid Avro file unreadable and pointing the reader
  at pyarrow rather than at the mapping. Multi-branch unions are how Avro spells an evolving or
  sum-typed field, so this is ordinary data, not a corner case. Now mapped to a struct with one
  nullable `memberN` per branch — the same choice Spark's Avro reader makes — while the far
  commoner `["null", T]` idiom stays exactly the nullable scalar it always was.
- **SQL `schema()` no longer runs the query.** Every relational connector (ADBC, ODBC,
  ClickHouse, ConnectorX, Databricks, Snowflake, BigQuery) answered `schema()` by executing the
  user's query in full and reading `.schema` off the materialized table. The column names of a
  billion-row join cost the billion-row join; and since the planner needs the schema *before* it
  executes, an ordinary `read(...).filter(...).collect()` submitted the whole query **twice** —
  on a per-query or per-byte-billed warehouse, a second real invoice for a discarded result.
  Snowflake additionally *downloaded a result chunk from cloud storage*, and BigQuery opened and
  drained stream 0 when `create_read_session` already returns the Arrow schema for free.
  Now a zero-row `WHERE 1 = 0` probe (BigQuery: the session's own `arrow_schema`).
  The probe result is *checked* (`probe_is_typed`) and falls back to the full read if a driver
  returns untyped empty columns — a cheap-but-wrong `schema()` would be the same broken contract
  the CSV and Avro fixes were about, reintroduced by the optimization meant to speed it up.
  Both connectors also indexed `splits()[0]` unguarded, so an **empty relation** — which still
  has columns — raised `IndexError` instead of reporting its schema.
- **Delta change feed lost data on a partial read.** `DeltaStreamSource.iter_batches`
  advanced `_cursor` to the latest version *before the first `yield`*, so
  `snapshot_position()` reported a window as consumed while its batches were still inside an
  unstarted generator. A consumer that checkpoints — which is the only reason
  `snapshot_position`/`seek` exist — and then failed mid-drain resumed past data it never
  saw. Measured: the consumer took one batch of 3 rows, and resuming from its own checkpoint
  returned **0** of the remaining 3. The cursor now advances after the drain, making the
  stream at-least-once: a failure replays the window, which is the correct failure mode for
  a change feed and the one checkpointing consumers already handle.
- **Delta `row_count()` raised on a log without `num_records`.** A writer is not obliged to
  record it, and `_snapshot` already guards the same column for its zone maps. Reading it
  unguarded turned a best-effort statistic — a number the planner can do without — into a
  `KeyError` that failed the query. Now degrades to `None`.
- **Delta Sharing discarded the server's zone maps.** `_stats_manifest` built its table with
  `pa.Table.from_pylist`, which takes its column set from the **first dict alone**. A shared
  table whose first file carries no `stats` lost every `min.`/`max.`/`null_count.` column,
  for the whole table, silently — so nothing was ever pruned. Rows are now normalized to the
  union of keys first. (Fail-safe in direction, total in cost.)
- **Delta Sharing re-fetched row counts it already had.** The split carried only the URL, so
  `row_count()` opened a **pre-signed URL per split** to read a footer whose `numRecords` the
  manifest had already parsed — serially, on the driver, on every planner balance. The split
  now carries `rows`, as the sibling `DeltaFileSplit` always has.
- **Iceberg loaded the same table twice per split.** `_scan()` calls `_table()`, and every
  caller of `_scan()` called `_table()` again first — so an ordinary split read was two
  catalog round trips plus two metadata-JSON fetches, and since `_source()` builds a fresh
  source per split, a 1,000-split scan cost 2,000 of them before a byte of data moved.
  `_table()` is now memoized on the source, which is the correct scope: a source is pinned
  to one identifier and optionally one snapshot, so the table it names cannot change
  underneath it within its own lifetime.
- **Parquet page index: written, now read.** Row-group pruning is coarse — a row group is
  ~1M rows, so a selective predicate still decoded one whole. The writer had always emitted
  the ColumnIndex/OffsetIndex precisely so a reader could do better, and nothing read it
  back. `bc-io::page_index` now turns a pushed predicate into a `RowSelection` over the
  surviving pages: per-page min/max from the ColumnIndex, page→row ranges from the
  OffsetIndex, `And` intersecting selections and `Or` unioning them. Measured on a 2M-row
  single-row-group file, `k < 100`: **32,768 rows decoded instead of 2,000,000 (61x less),
  67.1 ms → 3.8 ms (17.5x)**.

  Superset-safe at every step, since the engine keeps its `Filter`: an undecidable leaf
  contributes nothing, an `And` with an undecidable side keeps the other side, an `Or` with
  one is `None` outright, and `None` at the root reads the group whole exactly as before.

  **The first working build of this pruned nothing and every test passed.**
  `ParquetObjectReader::get_metadata` ignores `ArrowReaderOptions::with_page_index` and
  consults its own `preload_column_index`/`preload_offset_index` flags instead — so the
  option compiled, ran, loaded no index, and left every page surviving. Correct results, a
  green suite, a no-op feature. `test_pruning_actually_engages` asserts the decoder returns
  fewer rows than the group holds, which is the only assertion that can tell the difference.
- **The autoloader's lexical filter dropped files.** Discovery kept only the names sorting
  after the greatest already-seen name. That is sound only when files arrive under
  monotonically increasing names, and the writers that matter do not: `part-00000-<uuid>`
  means that once one file is processed, roughly half of every later arrival sorts below the
  maximum and is never offered to the seen-store at all. Silent, permanent, and invisible to
  every existing test, all of which used zero-padded sequential names. Removed — the listing
  is unchanged and `unseen` is already an index probe, so the filter was buying a fraction of
  a millisecond and costing rows.
- **Pulsar's drain was one client call per message.** `batch_receive()` — listed as "unused"
  above — now takes the whole buffered batch in one call, under a `batch_receive_policy` sized
  for a poll's budget with a 1 ms give-up, while the blocking wait for the first message stays
  a plain `receive` so an idle topic parks rather than spins.
- **Parquet bloom filters: now read.** Range pruning — footer stats and the page index — is
  only as good as the data's clustering. On a **high-cardinality unordered** column every
  page's `[min, max]` spans the domain, so an equality prunes nothing: measured on 2M random
  keys in one row group, `k == <value>` kept **2,000,000 rows after page pruning, a 1.00x
  reduction**. That is the shape blooms answer, and it is the common shape for join keys,
  user ids and hashes. `bc-io::bloom` consults the bloom for equality terms and skips a row
  group whose bloom proves the value absent.

  Safe because a bloom has **no false negatives** — a negative verdict is definitive, and a
  false positive merely decodes a group the engine's `Filter` then empties. The lattice is
  the dual of the page index's: a conjunction is empty if *either* side is, a disjunction
  only if *both* are.

  The one way it could lose rows is a **physical-type mismatch**: the bloom hashes the
  column's physical bytes, so probing an `i64` against an `INT32` bloom hashes different
  bytes and can answer "absent" for a value that is present. Every probe therefore requires
  an exact physical-type match and otherwise declines to prune.

  Cost is bounded: a syntactic `has_eq` pre-check means a predicate with no equality never
  triggers a fetch, and a column with no bloom costs no I/O at all (the offset is absent from
  the footer the reader already holds). Batcher's own writer emits no blooms — the pinned
  pyarrow has no option for it — so this is about reading Spark/Databricks/DuckDB-written
  lakes well. The test fixture is therefore written from Rust, which is also what makes the
  feature verifiable at all; an unverifiable pruner is how the page-index work first shipped
  as a silent no-op.
