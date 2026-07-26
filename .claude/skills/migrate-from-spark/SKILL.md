---
name: migrate-from-spark
description: Port a PySpark job to Batcher's public Python API — the DataFrame verb translation, the SparkSession/lazy/save-mode/UDF/shuffle concept shifts, and an ordered recipe that verifies the ported script returns the same rows as the original. Invoke when converting PySpark code to Batcher, or when asked how a Spark idiom is spelled here.
---

# Migrate a PySpark job to Batcher

Use this when you have working PySpark and want it running on Batcher. The DataFrame
vocabulary carries over almost verbatim; what changes is the *runtime model*, and that
is where ports go wrong. Read `docs/migration/transforming.md` (the canonical mapping tables)
before extending anything here — this skill is the porting procedure, that page is the
reference.

## Translation table

| PySpark | Batcher | Note |
|---|---|---|
| `SparkSession.builder.getOrCreate()` | *(nothing)* | `import batcher as bt`; the engine is in-process |
| `spark.read.parquet(p)` | `bt.read.parquet(p)` | already lazy — no `scan_*`/`read_*` split |
| `spark.read.load(p)` | `bt.read(p)` | format inferred from the path |
| `spark.read.format("delta").load(p)` | `bt.read.delta(p)` | also `.iceberg`, `.csv`, `.json`, `.orc`, `.avro` |
| `spark.createDataFrame(rows)` | `bt.from_pylist(rows)` / `bt.from_pydict(d)` | |
| existing `DataFrame` | `bt.from_spark(sdf)` | on-ramp only — collects through Spark's Arrow bridge |
| `df.select(...)` | `ds.select(...)` | |
| `df.withColumn("c", e)` | `ds.with_columns(c=e)` | plural, kwargs; adds/replaces |
| `df.withColumnRenamed(a, b)` | `ds.rename({a: b})` | |
| `df.drop("c")` | `ds.drop("c")` | |
| `df.filter(...)` / `.where(...)` | `ds.filter(bt.col(...) > 1)` | one spelling |
| `df.groupBy("k").agg(...)` | `ds.group_by("k").agg(total=bt.col("v").sum())` | named kwargs become output columns |
| `F.avg("v")` | `bt.col("v").mean()` | `mean` is canonical; `avg` accepted |
| `F.collect_list("v")` | `bt.col("v").array_agg()` | |
| `F.countDistinct("v")` | `bt.col("v").n_unique()` | `bt.approx_n_unique` for the sketch |
| `df.orderBy("a")` / `.sort` | `ds.sort("a", descending=False)` | `nulls_first=` is explicit |
| `df.join(o, "k", "left")` | `ds.join(o, on="k", how="left")` | also `left_on=`/`right_on=` |
| `df.distinct()` | `ds.distinct()` | |
| `df.limit(n)` | `ds.limit(n)` / `ds.head(n)` | |
| `F.when(c, a).otherwise(b)` | `bt.when(c).then(a).otherwise(b)` | |
| `F.lit(x)` | `bt.lit(x)` | |
| `F.rank().over(Window.partitionBy(..).orderBy(..))` | `bt.rank().over(partition_by=.., order_by=..)` | no `Window` object |
| `df.unionByName(o)` | `ds.union(o)` | `ds.intersect`, `ds.except_` too |
| `df.repartition(n)` | `ds.repartition(n)` | file/partition count, not a forced shuffle |
| `spark.sql(q)` | `bt.sql(q, t=ds)` / `ds.sql(q)` | tables bound as kwargs |
| `spark.udf.register(...)` | `bt.register_function(name, fn)` | callable from `bt.sql` |
| `F.pandas_udf` | `@bt.udf` + `ds.map_batches(fn)` | Arrow batch in, Arrow batch out |
| `df.explain()` | `ds.explain()` (`analyze=True` to run it) | returns a `str` |
| `df.collect()` | `ds.collect()` (Arrow table) / `ds.to_pylist()` | |
| `df.count()` / `.show()` | `ds.count()` / `ds.show()` | |
| `df.toLocalIterator()` | `ds.iter_batches()` | streams Arrow batches |
| `df.write.mode("append").parquet(p)` | `ds.write(p, mode="append")` | Spark `SaveMode` parity |
| `MERGE INTO` | `ds.write.delta(uri, merge_on=["id"])` | one transactional call |

## Conceptual shifts that actually bite

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
  redistribution hint. Ray, when used, schedules tasks only — batches move over Arrow
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
2. **Delete the session.** Replace the builder with `import batcher as bt`, drop
   `spark.conf` calls, and keep a note of any that were load-bearing (shuffle partitions,
   broadcast thresholds — these become non-goals, not settings).
3. **Port sources.** `spark.read.X` → `bt.read.X`. Keep the paths identical so the two
   scripts read the same bytes.
4. **Port transforms top-down**, one verb at a time using the table above. Fold
   `withColumn` chains into single `with_columns` calls. Replace `Window.partitionBy(...)`
   with `.over(partition_by=..., order_by=...)`.
5. **Port UDFs last.** Each `pandas_udf`/`udf` becomes a `map_batches` function over a
   `pyarrow.RecordBatch`. Pass `input_columns=[...]` naming *every* column the function
   reads — projection pushdown prunes the scan to that list, so an omission is a
   correctness bug, not a perf nit. When unsure what a ported UDF touches, leave
   `input_columns=None` (the default), which keeps every column alive. If the UDF is
   pure column arithmetic, delete it and write an `Expr`.
6. **Port the sink.** `df.write.mode(m).format(f).save(p)` → `ds.write(p, mode=m)` or the
   typed `ds.write.parquet/delta/iceberg(...)`.
7. **Verify equivalence.** Run both scripts on the same input, dump each to Arrow, and
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

8. **Check the plan, then the clock.** `print(ported.explain())` to confirm the pushdowns
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

## `bt.sql(dialect="spark")` gives you Spark's *syntax*, not Spark's semantics

`bt.sql` reads Spark SQL when you ask it to, so a `spark.sql("...")` string usually ports
by changing the call. The dialect selects a **parser**. Where Spark and DuckDB genuinely
disagree on what a function *means*, the engine follows DuckDB, because DuckDB is the
oracle every differential test in the repo is written against. These are the differences
a port actually hits, each found by running Spark's own documented examples through
`bt.sql` (`docs/internals/competitor_parity_census.md`):

| Expression | Spark | Batcher (= DuckDB) |
|---|---|---|
| `regexp_replace(s, p, r)` | replaces **every** match | replaces the **first**; use `regexp_replace_all` |
| `sort_array(a)` | nulls **first** | nulls **last** |
| `array_distinct(a)` | keeps a null element | drops nulls |
| `dayname(d)` / `monthname(d)` | abbreviated (`Wed`, `Feb`) | full (`Wednesday`, `February`) |
| `round(x)` | half **up** | half **away from zero** |
| `split(s, p)` | `p` is a **regex** | `p` is a **literal**; use `regexp_split_to_array` |
| `element_at(a, i)` | 1-based | 1-based (`a[i]` is 0-based in Spark, 1-based in DuckDB — the parser handles it) |

Rewrite the first six explicitly during the port; none of them raises, so each is a
result that quietly differs. Verify the port the way the recipe above says: compare row
counts and a checksum against the original job's output, not the eyeball.

## See also

- `docs/migration/transforming.md` — the full pandas/Polars/PySpark mapping tables.
- `docs/user-guide/{sql,udfs,window-functions,writing-data,explain-plans}.md`.
- `docs/integrations/ray.md` — how distribution actually works (scheduling only).
- Skills: `migrate-from-ray-data` (the Ray Data port), `run-quality-gate` (if the port
  changes repo code).
