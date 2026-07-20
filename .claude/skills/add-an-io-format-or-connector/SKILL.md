---
name: add-an-io-format-or-connector
description: Recipe to add a new IO format or connector to Batcher — the FileSource/FileSink template bases, self-registration into the SOURCES/SINKS registries, the split protocol that determines read parallelism, schema inference, statistics, predicate/projection pushdown, credentials, and error tolerance — while keeping the io/ layer neutral. Invoke when adding or changing a reader/writer for a file format, lakehouse table, database, or streaming source.
---

# Add an IO format or connector

A format is a named pairing of a read path and a write path over Arrow. Adding one is
**one new file in the right category that registers itself** — `io/base/source.py`,
`io/base/sink.py`, and the `api` layer do not change. The Template-Method bases already
own path/glob/filesystem resolution, schema caching, multi-file concatenation,
projection plumbing, streaming read-ahead, atomic writes, Hive partitioning, and split
generation, so a concrete format is a small subclass.

Read `.claude/rules/architecture.md` (the layer rule below is load-bearing) and
`docs/user-guide/custom-connectors.md` (the worked end-to-end example this skill is the
procedure for) first. `docs/internals/extending.md` §"Add an IO format" is the short form.

## The layer rule — `io/` is NEUTRAL

`io` sits at layer 2. It imports `plan`, `config`, `_internal` (including
`_internal.native` for the Rust readers) — and **no subsystem**. It must never import
`kyber`, `carbonite`, `core`, `governance`, or `api`. That neutrality is why every
layer above may depend on it, and it is enforced by `just lint-layers`. If a connector
seems to need an optimizer or executor decision, it doesn't: it exposes *facts*
(statistics, splits, pushdown support) and Kyber decides.

Corollary: a connector never touches a row in Python for query purposes. Deserializing
bytes a format only offers row-wise (Avro via `fastavro`) is unavoidable and happens at
*batch* granularity; per-row query logic is not.

## 1. Pick the category and the file

`io/formats/<category>/<fmt>.py`, one module per format. The categories exist so no
directory blows the ≤12-files limit (`.claude/rules/maintainability.md`):

`structured` (parquet, csv, orc, avro, arrow_ipc, lance, excel) · `semistructured`
(json, xml, msgpack, protobuf, logs) · `unstructured` (text, binary, documents) ·
`lakehouse` (delta, iceberg, hudi, delta_sharing) · `sql` (adbc, snowflake, bigquery,
clickhouse, databricks, odbc, connectorx) · `nosql` (mongo, dynamodb, cassandra, redis,
elasticsearch, hbase, neo4j, couchbase) · `streaming` (kafka, kinesis, pubsub, pulsar,
eventhubs, autoloader) · `ml` (tfrecord, webdataset, numpy, hdf5, zarr, tensor,
point_cloud, shards) · `multimodal` (images, audio, video, embeddings, blob).

A format that outgrows one module becomes a package (`parquet/`, `delta/`,
`iceberg/`) — never a flattened `fmt_source.py` / `fmt_sink.py` pair in the parent.

## 2. Subclass the template bases

```python
# io/formats/structured/myfmt.py
from batcher.io.base import FileSink, FileSource
from batcher.io.formats.base import SINKS, SOURCES

@SOURCES.register("myfmt")
class MyFmtSource(FileSource):
    suffix = ".myf"          # drives directory/glob expansion
    format_name = "myfmt"    # the registry key a split rebuilds through

    def _read_schema(self, fh): ...                 # abstract — no data scan if possible
    def _read_file(self, fh, projection): ...       # abstract — one file → list[RecordBatch]

@SINKS.register("myfmt")
class MyFmtSink(FileSink):
    suffix = ".myf"
    format_name = "myfmt"

    def _write_file(self, table, fh): ...           # abstract — whole table → open handle
```

Then import the module from the category `__init__.py` so the decorator runs.
`io/formats/__init__.py` imports every category, so `bt.read(path, format="myfmt")`
works with no explicit import. `SOURCES.names()` / `SINKS.names()` enumerate what is
registered; registering a duplicate name raises.

**Optional source overrides**, in rough order of value: `_iter_file` (true streaming —
without it `iter_batches` falls back to reading whole files), `_file_row_count` (a cheap
footer count feeds the optimizer), `_file_splits` (sub-file parallelism, see §3),
`_reader_kwargs` (**required** if `__init__` takes behavior-changing kwargs — a
`message_cls`, a `sheet`, a stride; without it a worker rebuilds the wrong reader),
`_read_by_path` (skip the Python handle when the reader does its own C++-side I/O),
`statistics` (§5).

**Optional sink overrides**: `_open_stream_writer` / `_write_batch` /
`_close_stream_writer` turn `write_stream` from buffer-one-table into truly bounded
memory; `commit(manifest, path)` is a no-op for file sinks and is where a transactional
lakehouse sink publishes its files atomically.

`write` is atomic (temp file + rename locally, a single PUT on object stores) and
`resume=True` skips an already-present — hence complete — file. `write_partitioned`
gives you Hive `col=value` directories via `_hive_partition`, which is vectorized,
handles NULL and NaN keys, and URL-encodes path segments. Don't reimplement any of it.

## 3. Splits — this is what determines parallelism

`Split` (`io/splits/base.py`) is the unit of distributed read parallelism. A split
carries **locators only** — never data — so it pickles cheaply to a worker that reads
its slice straight from storage. Splits mirror the `Source` read surface
(`schema`/`read`/`iter_batches`/`row_count`/`identity`) so a worker treats one exactly
like a source.

- `FileSplit(format_name, path, kwargs)` — one whole file, rebuilt on the worker via
  `SOURCES.get(format_name)(path, **kwargs)`. The default, and usually right.
- `RowGroupSplit` — a contiguous run of Parquet row-groups. The finest granularity;
  `parquet_row_group_splits` also *prunes* at plan time so a ruled-out row-group never
  becomes a task. `pack_row_groups` balances runs against `target_size`.
- `LineRangeSplit` — a newline-aligned byte range, so one huge NDJSON file fans out.
- `IpcFileSplit` — Arrow IPC record-batch ranges.
- `WholeSourceSplit(source)` — the fallback for a source that cannot subdivide. It holds
  the source object, so it is only as picklable as that source.

**The granularity you emit is the parallelism you get.** One `FileSplit` per file caps a
single-file dataset at one worker. If the format has an internal chunk index (row
groups, stripes, shards, record boundaries), express it — that is the whole difference
between a scan that scales and one that doesn't. Note that a schema-evolving read
(`schema_mode != "strict"`) deliberately degrades to a single `WholeSourceSplit`,
because a per-file split rebuilds a reader that knows nothing of the unified schema.

## 4. Schema inference and evolution

`_read_schema` must answer from metadata where the format has any. `FileSource.schema()`
caches it and, in `schema_mode="strict"` (the default), takes file 0's schema for all.
The `"union"`/`"latest"` modes read every file's schema concurrently and reconcile them
through `io/schema/` (`unify_schemas`, `normalize_batch`, `reconcile_batches`,
`schema_drift`). That is read-time only and it is shared — never solve schema drift
inside a format.

## 5. Statistics for the optimizer

A source may expose `statistics() -> SourceStatistics | None` (duck-typed via
`io.source.read.source_statistics`, deliberately not a Protocol method so older sources
still satisfy `Source`). Build it from the extraction helpers in `io/stats/`:
`parquet_statistics`/`orc_statistics` (footers), `manifest_statistics` (Delta/Iceberg
manifests), `numpy_statistics` (`.npy` headers), `catalog_row_count` (SQL system
catalogs), `parquet_row_group_bounds`/`RowGroupBounds` (zone maps for pruning).

These are O(1) control-plane metadata reads, never a per-row scan. Cardinality that is
merely *exact-looking* is worse than none: `identity()` is the key learned statistics
are stored under, and a source pinned to a subset of a directory's files (the `files=`
argument) gets its own digest-suffixed key precisely so the optimizer never sizes a
one-file read against the whole table's stats.

## 6. Pushdown

- **Projection** — `read`/`iter_batches` receive a `projection: list[str] | None`.
  Honor it in the reader if the format is columnar; ignoring it is correct but reads
  more than needed.
- **Predicate** — `_file_splits` receives Kyber's pushed filter as its IR dict.
  `io/predicate.py` translates that IR into whatever the backend speaks:
  `to_pyarrow_expression`, `to_sql_where`, `to_iceberg_expression`, `to_mongo_filter`,
  `to_native_predicate`. A format with no statistics ignores the predicate — the
  engine's `Filter` re-checks every row regardless, so **ignoring a predicate is always
  correct and merely slower; mis-translating one is a correctness bug.** Translate
  conservatively and return `None` for anything you cannot represent exactly.

## 7. Detection, credentials, filesystem, tolerance

- `io/detect.py` — add the extension to `_EXT_TO_FORMAT` (or a scheme to
  `_SCHEME_TO_FORMAT`, or a table-marker directory) so `bt.read(path)` infers the format
  and `DATA_SUFFIXES` finds the files. `format_for_extension` is its inverse.
- `io/filesystem.py` — `resolve_filesystem(path)` is the only way to touch storage;
  local, `s3://`, `gs://`, `az://`, and fsspec-backed schemes are one code path
  (`open`, `expand`, `exists`, `size`, `mkdirs`, `atomic_writer`).
- `io/credentials.py` — `vend_unity_credentials` for Unity Catalog vended credentials.
  Never read a secret into the plan; a key travels as an `env:NAME` / `file:PATH`
  reference resolved on the worker.
- `io/base/_tolerance.py` — `on_error="skip"` (vs `"raise"`) is already wired into both
  read paths via `ErrorPolicy`; skipped paths surface through `corrupt_files()`. Don't
  add per-format `try`/`except`.

## 8. Sources that aren't files, and large payloads

A non-file connector (a database, a stream, a table service) implements the `Source`
Protocol (`io/source/base.py`) directly instead of subclassing `FileSource` — the same
five methods plus `splits()`, registered the same way. A replayable streaming source may
also add `snapshot_position()` / `seek(position)` (duck-typed via `is_checkpointable`)
for exactly-once resume. Heavy or optional dependencies are imported **lazily inside
methods**, so `import batcher` stays cheap and the dependency stays optional; raise
`BackendError` with a `pip install 'batcher-engine[<extra>]'` hint when it is missing.

For multi-GB per-row payloads, don't carry bytes inline — every shuffle and spill buffer
pays for them. Read by reference and materialize late:
`io/formats/multimodal/blob.py` (`offload_blob_bytes` / `read_blob_bytes`, content-
addressed by SHA-256) is the pattern, and `read.video(materialize_bytes=False)` is it
in use.

## 9. Native Rust readers (`crates/bc-io`)

`bc-io` is a pure-Rust leaf crate (Parquet over `object_store` with concurrent column-
chunk fetches; Avro OCF → Arrow), reached from Python through `_internal.native`.
Reach for it only when **object-store read throughput is the measured bottleneck** — its
value is fetching projected column chunks concurrently instead of PyArrow's latency-bound
GET chain. The Python layer always keeps a PyArrow fallback for unsupported
schemes/features, and the two must be **byte-identical**. A new format starts pure
Python; add a native reader later, with a benchmark that justifies it.

## 10. Tests — the gate

- **Round-trip**, per format: write → read → identical Arrow table, across nulls, empty
  input, a single row, several batches, and type edges.
- **Splits**: `splits()` covers the source exactly once (concatenating every split's rows
  == a whole-source read); each split is picklable and reads correctly after a
  pickle round-trip — that is the distributed path in miniature.
- **Differential vs DuckDB** for anything relational the connector affects (pushdown
  especially): a predicate pushed into the reader must return exactly what the engine's
  own `Filter` would. See `tests/integration/test_predicate_pushdown_reader.py`.
- **Registry + detection**: `tests/unit/test_io_format_registry.py` for the name;
  `tests/integration/test_write_autodetect.py` for extension inference.
- Reference points: `tests/unit/test_io_stats.py`, `test_io_corrupt_file_tolerance.py`,
  `test_io_avro_streaming.py`, `test_source_statistics.py`;
  `tests/integration/test_io_and_api.py`, `test_schema_mode_and_range.py`,
  `test_write_modes.py`, `test_distributed_write.py`.

Then the gate: `just lint-py` → `just lint-layers` (the neutrality contract) →
`just lint-structure` (module ≤500 lines, ≤12 files/dir) → `just build` →
`just test-py`, plus `just test-rust` if you touched `bc-io`, and `just docs` if you
added public surface. See `/run-quality-gate`.

## Done checklist

- [ ] `io/formats/<category>/<fmt>.py`, self-registered into `SOURCES`/`SINKS`
- [ ] Category `__init__.py` imports it; heavy deps imported lazily inside methods
- [ ] Subclasses `FileSource`/`FileSink` (or implements `Source` directly)
- [ ] `_reader_kwargs` set if the reader needs more than a path
- [ ] Splits at the finest granularity the format supports; picklable, covering once
- [ ] Projection honored; predicate translated exactly or not at all
- [ ] `statistics()` from `io/stats/` where the format has metadata
- [ ] Extension/scheme in `io/detect.py`; storage only via `resolve_filesystem`
- [ ] `io/` imports no subsystem — `just lint-layers` green
- [ ] Round-trip, split-coverage, and differential tests; `/run-quality-gate` green

## See also

- `docs/user-guide/custom-connectors.md` (worked example, sources that aren't files,
  fetching bytes late), `docs/internals/extending.md`, `docs/user-guide/reading-data.md`,
  `docs/user-guide/writing-data.md`.
- Rules: `.claude/rules/architecture.md` (the import matrix),
  `.claude/rules/maintainability.md` (registry + family modules),
  `.claude/rules/testing.md`, `.claude/rules/performance.md`.
- Skills: `run-quality-gate`; `add-relational-operator` and `add-expression-or-function`
  (the other two extension points); `add-distributed-operator` (if the connector's
  splits need shuffle/transport work); `optimize-a-slow-query` (if a read is merely slow).
