---
name: write-a-batcher-pipeline
description: Write correct, idiomatic relational Batcher pipelines with the public Python API — the lazy/immutable mental model, the read → transform → write skeleton, expressions and accessor namespaces, joins, group_by/agg, window functions, batch UDFs, streaming with iter_batches, and the idiom/anti-pattern checklist to self-review against. Invoke for general Batcher pipeline and Dataset authoring work.
---

# Write a Batcher pipeline

This is the default skill for *using* Batcher relationally — read → transform → write,
expressions, joins, aggregations, windows, batch UDFs. Read it before writing any
`Dataset` code of that shape. Everything below is the public API reachable from
`import batcher as bt`.

**Never invent an API.** But this skill does not enumerate the whole surface — `bt.__all__`
is 179 names and `dir(bt.Dataset)` is 130 methods, and what follows is the relational core
of both. If a name is not here, check before concluding it is missing:
`python -c "import batcher as bt; print(sorted(bt.__all__))"` and
`python -c "import batcher as bt; print(sorted(dir(bt.Dataset)))"`. Verify, never guess —
in either direction.

## Not covered here — route to

- **`write-a-streaming-pipeline`** — unbounded sources, `ds.with_watermark`,
  `join_stream`, `drop_duplicates_within_watermark`, `ds.is_streaming`.
- **`read-and-write-data`** — reader/writer options in depth, formats, cloud paths,
  partitioning, schema evolution, incremental reads.
- **`manage-a-lakehouse-table`** — Delta/Iceberg/Hudi/Lance table maintenance,
  `ds.merge`, `ds.scd` (`type1`/`type2`/`type3`, `apply_changes`), time travel.
- **`add-an-io-format-or-connector`** — adding a new source/sink to the `io` layer
  (engine work, not pipeline authoring).
- **`apply-governance-and-security`** — row filters, column masks, `bt.SecurityCatalog`,
  `ds.lineage`.
- **`validate-data-quality`** — the `ds.dq` accessor (`not_null`, `unique`, `in_range`,
  `accepted_values`, `matches`, `foreign_key`, `check`, `validate`, `fail`/`drop`/
  `quarantine`) and `ds.meta` assertions/facts.
- **`migrate-from-duckdb-sql`** — writing or porting SQL (`bt.sql`, `bt.Session`, `ds.sql`).
- **`debug-a-batcher-query`** — a query that raises, hangs, OOMs, or returns wrong rows.
- **`optimize-a-slow-query`** — a correct query that is too slow.
- **`run-a-distributed-job`** — taking a working pipeline onto a Ray cluster.
- **`build-an-ml-pipeline`** — `ds.ml` inference, embeddings, multimodal, data loaders,
  preprocessors.

## Mental model (get this right and the rest follows)

- **A `Dataset` is a lazy plan handle, not data.** `bt.read.parquet(...)` reads nothing.
  Every verb returns a **new** `Dataset`; nothing mutates in place.
- **Work happens only at a terminal op**: `collect`, `to_arrow`, `to_pydict`, `to_pylist`,
  `to_pandas`, `to_polars`, `iter_batches`, `count`, `show`, `stats`, `describe`, `write`.
  Then the optimizer rewrites the plan and the Rust engine runs it.
- **Expressions execute in Rust; Python callbacks do not.** `bt.col("x") * 2` compiles and
  JITs, and the optimizer can push it around. Prefer an `Expr` every time one exists.
- **Python must never touch a row**, and **build one plan, collect once** — splitting a
  query across several `collect()` calls throws away optimization and re-reads the source.

## The canonical skeleton

```python
import batcher as bt

orders = bt.read.parquet("s3://bucket/orders/")      # lazy
customers = bt.read.parquet("s3://bucket/customers/")

result = (
    orders
    .filter(bt.col("status") == "paid")              # filter early — it pushes down
    .join(customers, on="cust_id", how="inner")
    .with_columns(net=bt.col("amount") * 0.9)
    .group_by("region")
    .agg(revenue=bt.sum("net"), orders=bt.count())
    .sort("revenue", descending=True)
)

result.write("s3://bucket/out/", mode="overwrite")   # terminal
```

Nothing before `.write(...)` executes. Swap the last line for `result.to_pydict()` when
the answer is small, or `result.iter_batches()` when it is not.

## Expressions

`bt.col(name)` selects, `bt.lit(v)` is a constant, and operators build up from there.
Breadth lives on **accessor namespaces**, not on `Expr` itself:
`.str`, `.dt`, `.list`, `.struct`, `.json`, `.map`, `.image`, `.audio`, `.video`.

```python
ds = bt.from_pydict({"name": [" Ann ", "bob"], "tags": [["a", "b"], ["c"]]})

ds = ds.with_columns(
    clean=bt.col("name").str.strip().str.to_lowercase(),
    n_tags=bt.col("tags").list.len(),
    tier=bt.when(bt.col("name").str.len_chars() > 3).then("long").otherwise("short"),
)
```

Chain with `& | ~` (parenthesize each side — `&` binds tighter than a comparison).
Conditionals are `bt.when(cond).then(a).otherwise(b)` (or `bt.iff(c, a, b)`). Horizontal
folds across columns: `bt.sum_horizontal`, `bt.max_horizontal`, `bt.coalesce`,
`bt.all_horizontal`. Column *sets* come from selectors: `bt.numeric()`, `bt.string()`,
`bt.by_dtype(...)`, `bt.exclude(...)`.

### `select` vs `with_columns`

`select(*cols, **named)` chooses/derives the **entire** output schema — anything not named
is dropped. `with_columns(*exprs, **named)` **adds or replaces**, keeping everything else.

```python
ds.with_columns(net=bt.col("amount") * 0.9)         # all original columns + net
ds.select("order_id", net=bt.col("amount") * 0.9)   # exactly two columns
```

Use `select` as the last step to prune the output; the optimizer pushes that pruning back
into the scan.

## Joins

```python
ds.join(other, on="k", how="inner")                       # on / left_on / right_on
ds.join(other, left_on="cust_id", right_on="id", how="left", suffix="_c")
ds.cross_join(other)
quotes.join_asof(trades, on="ts", by="symbol", direction="backward")   # time-series
```

`how` is `"inner" | "left" | "right" | "full"` (`"outer"`) `| "semi" | "anti"` — semi/anti
add no right columns, they filter by existence. Overlapping non-key columns get `suffix`
(default `"_right"`). `join_asof` matches each left row to the nearest preceding
(`direction="backward"`, the default) or following right row; `by`/`left_by`/`right_by`
scope the match to a group key. Both sides should be sorted on the asof key.
Set ops: `ds.union(other)` (`distinct=True` to dedup), `ds.intersect`, `ds.except_`.

## group_by / agg

```python
(ds.group_by("region", "status")
   .agg(revenue=bt.sum("amount"), n=bt.count(), p95=bt.quantile("amount", 0.95)))
```

Keyword names become output column names — that is the one obvious spelling. `group_by`
also takes computed keys as kwargs (`ds.group_by(day=bt.col("ts").dt.truncate("day"))`),
and `group_by()` with no keys aggregates globally. Shorthand reducers on the `GroupBy`
(`.sum()`, `.mean()`, `.len()`, …) reduce every remaining column the same way.
**Aggregates cannot be nested**, but expressions over them are fine
(`avg_price=bt.col("price").sum() / bt.count()`). For huge cardinality prefer the sketch
aggregates — `bt.approx_n_unique`, `bt.approx_quantile`, `bt.approx_median`.

## Window functions

Bind a window function with `.over(partition_by=..., order_by=..., frame=...)`:

```python
ds.with_columns(
    rn=bt.row_number().over(partition_by=["region"], order_by=["amount"]),
    prev=bt.lag("amount").over(partition_by=["region"], order_by=["ts"]),
    running=bt.sum("amount").over(partition_by=["region"], order_by=["ts"]),
)
```

Ranking/navigation functions (`bt.row_number`, `bt.rank`, `bt.dense_rank`, `bt.ntile`,
`bt.percent_rank`, `bt.cume_dist`, `bt.lag`, `bt.lead`, `bt.first_value`, `bt.last_value`,
`bt.nth_value`) and any aggregate all accept `.over(...)`. `order_by` is required for the
ordered ones. `frame=(preceding, following)` is a row frame, `None` meaning unbounded — a
running total is `frame=(None, 0)`. `Dataset.window(partition_by=, order_by=, functions=
{...}, frame=)` is the bulk form for several window columns in one operator.

**Windows may not appear where SQL also forbids them**: inside `group_by().agg(...)`, in a
join key, or in a sort key. Compute the window in a `with_columns` step first, then
reference the resulting column.

## UDFs — batch-first, never per row

When no expression covers the logic, drop to a callback over **whole Arrow batches**:

```python
import pyarrow as pa
import pyarrow.compute as pc

def add_total(batch: pa.RecordBatch) -> pa.RecordBatch:
    total = pc.multiply(batch.column("price"), batch.column("qty"))
    return batch.append_column("total", total)

ds.map_batches(add_total, output_columns=["price", "qty", "total"])
```

`map_batches(fn, *, batch_size=, input_columns=, output_columns=, num_workers="auto",
num_gpus=0.0, batch_format="pyarrow", multiprocessing=False, max_errored_rows=0)` hands
`fn` a `pyarrow.RecordBatch` and expects one back. You are writing the *body* of a
columnar operator, not a row loop.

- **`output_columns` is required whenever `fn` changes the schema.** Omit it and later
  operators still believe the old schema — a `select` on the new column fails at plan time.
- **`input_columns` is a declaration to the optimizer, not a filter.** A column you *fail*
  to declare can be pruned out from under `fn` — a correctness bug, not a perf nit.
  Declare **every** column `fn` reads, or leave it `None` (the default) if unsure.
- **Pass a class, not a function, for expensive setup.** A function is re-created per
  batch; a class is instantiated **once per worker** — the highest-leverage line in the
  API for model loading or a GPU context (`ds.map_batches(Classifier, num_gpus=1)`).
- `@bt.udf(output_columns=[...])` bundles a function with its options; the call site
  becomes `my_udf(ds)`. `@bt.udf(per_row=True)` wraps a `fn(row) -> row` callback.
  `bt.register_function(name, fn)` makes one callable from `bt.sql`.
- `Dataset.map` / `flat_map` are **per-row Python** — the slow path. `ds.select(tok=
  bt.col("text").str.split(",")).explode("tok")` beats a `flat_map` by roughly 10x.
- A preempted worker **recomputes** its partition, so `fn` must be idempotent — move side
  effects into a sink or upsert on a stable key.

## Streaming, and when not to `collect()`

`collect()` materializes the whole result in memory as an Arrow table. For anything that
does not comfortably fit, stream it — a breaker-free pipeline is consumed as batches are
produced, under bounded memory:

```python
for batch in ds.iter_batches(batch_size=16_384):
    ...  # a pyarrow.RecordBatch — hand it to a consumer, don't loop its rows
```

Better still, don't bring it to Python at all: end the plan in `ds.write(...)`.
`collect(spill=True)` allows out-of-core execution and `collect(distributed=True)` fans the
same plan across Ray workers with an identical result — turn neither on by default, both
cost overhead that hurts small queries. `ds.cache()` materializes once when a plan
genuinely branches. An unbounded source (`bt.read.kafka`, `bt.from_batches(...,
bounded=False)`) cannot be collected at all — check `ds.is_streaming`.

## Reading and writing

```python
bt.read(path)                              # format inferred from the extension
bt.read.parquet / .csv / .json / .arrow / .orc / .avro / .text   # files (globs, s3://)
bt.read.delta / .iceberg / .hudi / .lance                        # lakehouse
bt.read.kafka / .kinesis / .files_incremental                    # streaming sources
bt.read.images / .audio / .video / .sql / .table                 # and more
bt.from_pydict / from_arrow / from_pandas / from_polars          # in-memory

ds.write(path, format="parquet", mode="overwrite", partition_by=["region"])
ds.write.parquet(path, compression="zstd")
ds.write.delta(uri, mode="append", merge_on=["id"])     # transactional upsert
ds.write.iceberg("db.table", mode="append")
```

`mode` is `"overwrite" | "append" | "error" | "ignore"`; `append` is lakehouse sinks only.
`resume=True` skips committed shards on a re-run, `max_rows_per_file=` bounds file size,
and `partition_by=` writes Hive-style dirs that enable partition pruning on read.

## Configuration and inspection

`Config` is a tree of frozen dataclasses (`execution`, `memory`, `flow_control`,
`optimizer`, `pid`, `metadata`). Scope it to a block rather than setting it globally:

```python
from batcher import Config, ExecutionConfig

with bt.config_context(Config(execution=ExecutionConfig(morsel_rows=8192))):
    result = ds.collect()
```

`bt.set_config(cfg)` is the process-wide form; `Config().replace(section=...)` derives one.
Reach for config only when a measurement says to — the defaults adapt.

To see what the engine will do: `ds.explain()` returns the optimized plan with row
estimates tagged `exact`/`default`/`learned` (`analyze=True` runs it), `ds.stats()` reports
measured per-operator rows/time/bytes/spill after a run, `ds.schema`/`ds.columns`/
`ds.dtypes` describe the output without executing, `ds.show(n)` previews.

## Self-check before returning code

- [ ] Exactly **one** terminal op, at the end — no `collect()` mid-build, no follow-up
      Python that belongs in the plan.
- [ ] `filter` early and a final `select` so predicates/projections push into the scan; a
      column wrapped in a function before comparison blocks pushdown.
- [ ] No Python loop over rows; no `Dataset.map`/`flat_map` where an `Expr` or
      `map_batches` works; no `to_pandas()` as an intermediate step.
- [ ] `with_columns` called once with several kwargs, never chained in a `for` loop.
- [ ] Every `map_batches` that changes the schema passes `output_columns`; `input_columns`
      names **every** column read (or is `None`); `fn` is idempotent.
- [ ] Large results go to `iter_batches`/`write`, not `collect`/`to_pylist`.
- [ ] Output order is irrelevant or pinned by an explicit `ds.sort(...)` — row order is
      otherwise incidental.
- [ ] `.over(...)` specifies `order_by` where needed; no window in an agg, join key, or
      sort key.
- [ ] Joins name their keys (`on=` or `left_on=`/`right_on=`) and `how=` is explicit.
- [ ] No hand-tuned parallelism — `repartition` shapes *output files*, not shuffle width;
      `distributed=`/`spill=` are opt-in, not defaults.
- [ ] Every API used actually exists — verified against `bt.__all__` / `dir(bt.Dataset)`,
      not remembered.

## See also

- `docs/getting-started/quickstart.md`; `docs/user-guide/{expressions,joins,aggregations,
  window-functions,udfs,reading-data,writing-data,streaming,best-practices,explain-plans,
  performance,troubleshooting}.md`.
- Skills: `migrate-from-duckdb-sql` (SQL workloads); `migrate-from-spark`,
  `migrate-from-polars-or-pandas`, `migrate-from-ray-data`, `migrate-from-daft` (porting an
  existing script); `run-a-distributed-job`; `optimize-a-slow-query`;
  `debug-a-batcher-query`; `build-an-ml-pipeline` (inference/embeddings/multimodal).
