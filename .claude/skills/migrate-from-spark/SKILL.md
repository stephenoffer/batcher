---
name: migrate-from-spark
description: Port a PySpark job to Batcher's public Python API — the DataFrame verb translation, the SparkSession/lazy/save-mode/UDF/shuffle concept shifts, and an ordered recipe that verifies the ported script returns the same rows as the original. Invoke when converting PySpark code to Batcher, or when asked how a Spark idiom is spelled here.
---

# Migrate a PySpark job to Batcher

Use this when you have working PySpark and want it running on Batcher. The DataFrame
vocabulary carries over almost verbatim; what changes is the *runtime model*, and that
is where ports go wrong. The name-by-name reference is generated from the migration registry
into `docs/getting-started/migration/spark/`; this skill is the porting procedure around it.

## Where the name mappings live

Every PySpark 4.2.0 public name has one row in the migration registry
(`python/batcher/_internal/migration/data/pyspark/`), rendered into
`docs/getting-started/migration/spark/`:

- `dataframe.md`: `DataFrame`, its `na`/`stat` helpers, `GroupedData`, `Window`, `WindowSpec`.
- `expressions.md`: `Column`.
- `functions-aggregates.md`, `functions-collections.md`, `functions-math-and-misc.md`,
  `functions-strings.md`, `functions-temporal.md`: `pyspark.sql.functions` by family.
- `io.md`: `DataFrameReader`/`Writer`/`WriterV2`, `DataStreamReader`/`Writer`.
- `session-and-sql.md`: `SparkSession`, `Catalog`, `RuntimeConfig`, UDF registration,
  `StreamingQuery`.
- `types-and-data-sources.md`: `pyspark.sql.types` and the Python data source API.

Each row has a status (`index.md` defines them). **Read the status before renaming a call.**
A `mismatch` row is the dangerous one: the Batcher spelling exists and returns a different
answer, such as `orderBy` putting nulls first in Spark and last here, or `DataFrame.write`
raising on an existing target in Spark and overwriting here. `param` means a Spark option has
no Batcher counterpart yet, and `gap` means nothing does. Do not restate a mapping in this
skill: fix the registry row and run `just migration-docs`.

Three idioms span several names and so have no single row:

| PySpark | Batcher | Note |
|---|---|---|
| `SparkSession.builder.getOrCreate()` | *(nothing)* | `import batcher as bt`; the engine is in-process |
| `spark.read.option("mode", "DROPMALFORMED")` | `bt.read.csv(p, on_bad_lines="skip")` | also on `read.json`; `FAILFAST` is the default |
| `MERGE INTO` | `ds.write.delta(uri, merge_on=["id"])` | one transactional call |

## Conceptual shifts that actually bite

- **The read `mode` option splits in two.** Spark's `mode` conflates "this file is
  unreadable" with "this record is malformed", which in Batcher are `on_error=` and
  `on_bad_lines=` respectively. `DROPMALFORMED` is `on_bad_lines="skip"`; `FAILFAST` is the
  default. `PERMISSIVE` has no equivalent, because there is no corrupt-record column — a
  record that parses but does not fit the inferred type is answered by `schema=`, not by
  parking its text in a string column. Passing `mode=` raises with that translation rather
  than being ignored.
- **No `SparkSession`, no cluster.** The engine runs in-process. Delete the session
  builder, the `spark.stop()`, and every `spark.conf.set(...)`; configuration lives in
  `bt.Config` / `bt.set_config` / `bt.config_context`. Going distributed is a *keyword*,
  not a different program: `ds.collect(distributed=True)` (default `"auto"`).
- **Lazy is the same idea, actions are not the same list.** Batcher terminals are
  `collect`, `to_arrow`, `to_pydict`, `to_pylist`, `to_pandas`, `to_polars`, `count`,
  `show`, `iter_batches`, `stats`, `write`. Everything else builds a plan. A Spark script
  that relied on `.cache()` before repeated actions should use `ds.cache()` — but usually
  the right port is to stop re-collecting at all.
- **`withColumn` in a loop is an anti-pattern here too, and worse in Spark's shape.**
  Collapse `for c in cols: df = df.withColumn(...)` into one `ds.with_columns(**exprs)`.
- **Save modes carry over, `partitionBy` becomes a keyword.**
  `ds.write(path, mode="overwrite"|"error"|"ignore"|"append", partition_by=[...])`.
  Spark's `partitionOverwriteMode="dynamic"` is a session conf with no Batcher equivalent;
  it is `mode="overwrite_partitions"` on the write itself (`"dynamic"` is accepted too).
  `append` is lakehouse sinks only (delta/iceberg). `resume=True` makes a re-run skip
  committed shards, and `max_rows_per_file=` bounds file size.
- **UDFs are batch-first.** There is no `pandas_udf` decorator and no per-row JVM
  round-trip. `ds.map_batches(fn)` hands `fn` a **`pyarrow.RecordBatch`** and expects one
  back; `@bt.udf` bundles a function with its options so it can be applied as
  `fn(ds)`. Prefer an `Expr` over any Python callback — expressions run in Rust and JIT.
- **Shuffle/partitioning is not yours to hand-tune.** There is no
  `spark.sql.shuffle.partitions`. Partition count is chosen from measured cardinalities
  by the optimizer; `ds.repartition(...)` controls *output* file layout (`num_files=`,
  `by=`, `target_size_mb=`), and `ds.shuffle(seed=...)` is a row shuffle, not a
  redistribution hint. **`repartition(by=...)` is not Spark's `repartition($"col")`**: it
  Hive-partitions the *output directory* (Spark's `partitionBy`), it does not co-locate
  rows by key. To get Spark's co-location before a partitioned write, use
  `write(..., sort_by=[the partition columns])`, which makes the plan reduce through a
  breaker so each key lands in one file. Ray, when used, schedules tasks only — batches move over Arrow
  Flight, never the object store.
- **`explain()` is a string, and `analyze=True` actually runs.** For "where did the time
  go", `ds.stats()` reports measured rows/time/bytes/spill per operator plus the
  bottleneck — Spark has no equivalent.
- **`bt.from_spark(sdf)` materializes to the driver.** It is a migration on-ramp for
  small frames. For anything large, have Spark write Parquet/Delta and `bt.read` it.

## Porting recipe

1. **Inventory the script.** List every source, every action, and every UDF. Anything
   touching the JVM directly (`sc.parallelize`, RDD ops, `df.rdd`) has no port — rewrite
   it as a DataFrame/expression pipeline first, in Spark, so you can diff against it.
2. **Run the codemod first.** `python -m batcher.migrate --from pyspark --to batcher <paths>`
   prints a diff and changes nothing until you add `--write`. The `pyspark` direction may not be
   implemented yet: the command then raises `ConfigError` naming the directions that are, and
   the `spark/index.md` page says the same. Port by hand from the generated pages in that case.
   Either way, run `python -m batcher.migrate --from batcher --to batcher <paths>` over any
   code that already calls Batcher, so no removed Batcher spelling survives the port.
3. **Delete the session.** Replace the builder with `import batcher as bt`, drop
   `spark.conf` calls, and keep a note of any that were load-bearing (shuffle partitions,
   broadcast thresholds — these become non-goals, not settings).
4. **Port sources.** `spark.read.X` → `bt.read.X`. Keep the paths identical so the two
   scripts read the same bytes.
5. **Port transforms top-down**, one verb at a time using the generated pages. Fold
   `withColumn` chains into single `with_columns` calls. Replace `Window.partitionBy(...)`
   with `.over(partition_by=..., order_by=...)`.
6. **Port UDFs last.** Each `pandas_udf`/`udf` becomes a `map_batches` function over a
   `pyarrow.RecordBatch`. Pass `input_columns=[...]` naming *every* column the function
   reads — projection pushdown prunes the scan to that list, so an omission is a
   correctness bug, not a perf nit. When unsure what a ported UDF touches, leave
   `input_columns=None` (the default), which keeps every column alive. If the UDF is
   pure column arithmetic, delete it and write an `Expr`.
7. **Port the sink.** `df.write.mode(m).format(f).save(p)` → `ds.write(p, mode=m)` or the
   typed `ds.write.parquet/delta/iceberg(...)`.
8. **Verify equivalence.** Run both scripts on the same input, dump each to Arrow, and
   compare **order-independently** unless the query ends in an explicit `sort`. The
   in-repo pattern is `tests/differential/conftest.py::assert_same` (multiset comparison,
   tolerant of int↔float and float rounding); `assert_same_ordered` is the version to use
   when order is part of the contract. Mirror it:

   ```python
   import batcher as bt

   batcher_rows = sorted(map(tuple, zip(*ported.to_pydict().values())))
   spark_rows = sorted(tuple(r) for r in spark_df.collect())  # doctest: +SKIP
   assert batcher_rows == spark_rows
   ```

9. **Check the plan, then the clock.** `print(ported.explain())` to confirm the pushdowns
   landed, then `ported.stats()` for the measured per-operator profile.

## Gotchas / do-not

- **Do not assume Spark's output order.** Spark's row order is incidental too, but a port
  that "matched yesterday" and fails today usually never had a `sort`. Add an explicit
  `ds.sort(...)` if order matters; otherwise compare as a multiset.
- **Do not port `df.rdd.map(lambda row: ...)` into `ds.map(...)`.** `ds.map` exists and is
  per-row Python — it is the slow path. Use `map_batches` or, better, an `Expr`.
- **Do not `collect()` a large result** to iterate it in Python. Use `iter_batches()`
  (bounded memory) or push the work into the plan. Python must not touch rows in the hot
  path.
- **Do not translate `spark.sql.shuffle.partitions` into `repartition(n)`.** They are not
  the same knob; you will just make more files.
- **Do not leave `bt.from_spark` in the final script.** If it survives the port, you are
  still paying for the Spark job you meant to delete.
- **Do not assume a `map_batches` `fn` runs once.** A preempted worker recomputes its
  partition, so an `fn` with an external side effect can apply it twice — make sinks
  idempotent (upsert on a stable key).
- **Do not hand-tune GPU/batch-size knobs first.** `ds.ml.infer` / `ds.ml.map_batches`
  adapt batch size and `num_gpus` from measurements; set them only when a measurement
  says to.

## Going back

`docs/getting-started/migration/spark/leaving-batcher.md` maps Batcher spellings to PySpark for
the rows where both engines compute the same thing. A Batcher call whose PySpark counterpart is
a `mismatch` is not listed there, so look it up on the forward pages before porting it back.
`python -m batcher.migrate --from batcher --to pyspark <paths>` is the reverse codemod, subject
to the same implemented-directions check as the forward one.

Batcher has `bt.from_spark` but no Spark exporter, so hand data back as files: write Parquet or
Delta with `ds.write.parquet(p)` / `ds.write.delta(uri)` and read it with `spark.read`. Pass
`mode=` explicitly on both sides, because the two engines' default save modes differ.

## `bt.sql(dialect="spark")` gives you Spark's *syntax*, not Spark's semantics

`bt.sql` reads Spark SQL when you ask it to, so a `spark.sql("...")` string usually ports
by changing the call. The dialect selects a **parser**. Where Spark and DuckDB genuinely
disagree on what a function *means*, the engine follows DuckDB, because DuckDB is the
oracle every differential test in the repo is written against. These are the differences
a port actually hits, each found by running Spark's own documented examples through
`bt.sql` (`docs/architecture/internals/parity/competitor_parity_census.md`):

| Expression | Spark | Batcher (= DuckDB) |
|---|---|---|
| `regexp_replace(s, p, r)` | replaces **every** match | replaces the **first**; use `regexp_replace_all` |
| `sort_array(a)` | nulls **first** | nulls **last** |
| `array_distinct(a)` | keeps a null element | drops nulls |
| `dayname(d)` / `monthname(d)` | abbreviated (`Wed`, `Feb`) | full (`Wednesday`, `February`) |
| `round(x)` | half **up** | half **away from zero** |
| `split(s, p)` | `p` is a **regex** | `p` is a **literal**; use `regexp_split_to_array` |
| `weekday(d)` | `0` is **Monday** | `0` is **Sunday** (DuckDB's `dayofweek`) |
| `to_binary(s, charset)` | the encoded **bytes** | refused — DuckDB's `to_binary(s)` is a `0`/`1` bit string, a different function under the same name |
| `element_at(a, i)` | 1-based | 1-based (`a[i]` is 0-based in Spark, 1-based in DuckDB — the parser handles it) |

Rewrite the first seven explicitly during the port; none of them raises, so each is a
result that quietly differs. `to_binary` is the exception and is deliberately so: it is
refused rather than answered, because the two functions share a name and nothing else. Verify the port the way the recipe above says: compare row
counts and a checksum against the original job's output, not the eyeball.

## See also

- `docs/getting-started/migration/spark/index.md` — every PySpark name, with its Batcher spelling, status, and what differs.
- `docs/getting-started/migration/differences.md` — laziness, write-mode defaults, integer overflow.
- `docs/user-guide/analyze/sql.md`, `docs/user-guide/transform/columns/udfs.md`, `docs/user-guide/analyze/window-functions.md`, `docs/user-guide/moving-data/writing-data.md`, `docs/user-guide/operate/tuning/explain-plans.md`.
- `docs/integrations/compute/ray.md` — how distribution actually works (scheduling only).
- Skills: `run-quality-gate` (if the port changes repo code).
