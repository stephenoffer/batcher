# Migrating to Batcher

This page maps the operations you know from pandas, Polars, and PySpark onto their
Batcher equivalents, and shows how to move data in and out.

Batcher's surface is deliberately close to the libraries you already know, so most of
your vocabulary carries over. Absorb one concept before anything else: a `Dataset` is
*lazy*. Transformations such as `select`, `filter`, `group_by().agg()`, and `join`
build a plan and return a new `Dataset`, and nothing runs until a terminal operation
such as `collect`, `to_arrow`, `to_pandas`, `write`, `count`, or `iter_batches`. This
is the Polars `LazyFrame` model rather than the eager pandas one.

## Coming from

Each card names the single shift that matters most from that system.

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} {octicon}`table;1.1em` pandas
The one shift is eager to *lazy*. Operations build a plan and run on a terminal
call. `assign`, `groupby`, and `merge` become `with_columns`, `group_by().agg()`, and `join`.
:::

:::{grid-item-card} {octicon}`code;1.1em` Polars
You already know the `LazyFrame` model. Expressions, `group_by().agg()`, `.over(...)`,
and the typed accessors carry over almost verbatim.
:::

:::{grid-item-card} {octicon}`server;1.1em` PySpark
No `SparkSession` and no cluster to start, because it runs in-process. The DataFrame
verbs carry over, and so do the save modes and `MERGE INTO`.
:::
::::

## Reading and writing

Batcher gives you one callable namespace per direction. `bt.read(path)` infers the
format, and the typed methods such as `bt.read.parquet` and `bt.read.delta` are
explicit and discoverable. `ds.write` mirrors it.

```python
import batcher as bt

ds = bt.from_pydict({"city": ["NYC", "LA", "NYC"], "amount": [10, 20, 30]})
ds.write.parquet("/tmp/sales")          # ds.write("/tmp/sales") also infers parquet
back = bt.read.parquet("/tmp/sales")
print(sorted(back.to_pydict()["amount"]))
# [10, 20, 30]
```

The reader and writer spellings map across as follows.

| Task | pandas | Polars | PySpark | Batcher |
|------|--------|--------|---------|---------|
| Read Parquet | `pd.read_parquet(p)` | `pl.read_parquet(p)` | `spark.read.parquet(p)` | `bt.read.parquet(p)` |
| Scan Parquet (lazy) | n/a | `pl.scan_parquet(p)` | n/a | `bt.read.parquet(p)` |
| Read CSV | `pd.read_csv(p)` | `pl.read_csv(p)` | `spark.read.csv(p)` | `bt.read.csv(p)` |
| Read Delta | n/a | `pl.read_delta(p)` | `spark.read.format("delta").load(p)` | `bt.read.delta(p)` |
| Autodetect | n/a | n/a | `spark.read.load(p)` | `bt.read(p)` |
| Write Parquet | `df.to_parquet(p)` | `df.write_parquet(p)` | `df.write.parquet(p)` | `ds.write.parquet(p)` |
| Write Delta | n/a | `df.write_delta(p)` | `df.write.format("delta").save(p)` | `ds.write.delta(p)` |

Polars splits reading into eager `read_*` and lazy `scan_*`. Batcher doesn't need the
split, because every `bt.read.*` is already lazy. It returns a `Dataset` plan and does
no I/O until a terminal op, with the projection and predicate pushdown `scan_*` gives
you. There is one spelling per format, and it's the lazy one.

## Transforming

Transformations chain off a `Dataset` and return a new one, so a whole pipeline reads
as a single expression:

```python
import batcher as bt
from batcher import col

ds = bt.from_pydict({"city": ["NYC", "LA", "NYC"], "amount": [10, 20, 30]})
out = (
    ds.filter(col("amount") > 10)
    .with_columns(tax=col("amount") * 0.1)
    .group_by("city")
    .agg(total=col("amount").sum(), n=bt.count())
)
print(out.to_pydict())
# {'city': ['LA', 'NYC'], 'total': [20, 30], 'n': [1, 1]}
```

The transformation verbs map across as follows, ordered roughly by how often you reach
for them.

| Task | pandas | Polars | PySpark | Batcher |
|------|--------|--------|---------|---------|
| Select / project | `df[["a", "b"]]` | `df.select("a", "b")` | `df.select("a", "b")` | `ds.select("a", "b")` |
| Derive column | `df.assign(c=...)` | `df.with_columns(c=...)` | `df.withColumn("c", ...)` | `ds.with_columns(c=...)` |
| Filter rows | `df[df.a > 1]` | `df.filter(pl.col("a") > 1)` | `df.filter(df.a > 1)` | `ds.filter(col("a") > 1)` |
| Group + aggregate | `df.groupby("k").agg(...)` | `df.group_by("k").agg(...)` | `df.groupBy("k").agg(...)` | `ds.group_by("k").agg(...)` |
| Group + sum all | `df.groupby("k").sum()` | `df.group_by("k").sum()` | n/a | `ds.group_by("k").sum()` |
| Mean aggregate | `df.a.mean()` | `pl.col("a").mean()` | `F.avg("a")` | `col("a").mean()` |
| Sort | `df.sort_values("a")` | `df.sort("a")` | `df.orderBy("a")` | `ds.sort("a")` |
| Join | `df.merge(o, on="k")` | `df.join(o, on="k")` | `df.join(o, "k")` | `ds.join(o, on="k")` |
| ASOF join | `pd.merge_asof(...)` | `df.join_asof(...)` | `ASOF JOIN` | `ds.join_asof(o, on=..., by=...)` |
| Distinct | `df.drop_duplicates()` | `df.unique()` | `df.distinct()` | `ds.distinct()` |
| Limit | `df.head(n)` | `df.head(n)` | `df.limit(n)` | `ds.limit(n)` |
| Window rank | n/a | `pl.col(..).rank().over(..)` | `F.rank().over(Window...)` | `rank().over(partition_by=.., order_by=..)` |
| Window | n/a | `.over(...)` | `F....over(Window...)` | `ds.window(partition_by=..., functions=...)` |
| Collect list | `df.groupby(k)[c].agg(list)` | `pl.col(c).implode()` | `F.collect_list(c)` | `col(c).array_agg()` |
| First / last | `df.groupby(k).first()` | `pl.col(c).first()` | `F.first(c)` | `col(c).first(order_by=..)` |
| Column ref | `df["a"]` | `df["a"]` | `df["a"]` | `ds["a"]` |
| Row slice | `df[:n]` | `df[:n]` | n/a | `ds[:n]` |
| Fill nulls | `df.fillna(0)` | `df.fill_null(0)` | `df.fillna(0)` | `ds.fill_null(0)` |
| Drop nulls | `df.dropna()` | `df.drop_nulls()` | `df.dropna()` | `ds.drop_nulls()` |
| Cast | `df.astype({...})` | `df.cast({...})` | `df.withColumn(...)` | `ds.cast({...})` |
| Global agg | `df.sum()` | `df.select(...sum())` | `df.agg(...)` | `ds.agg(...)` |
| Explode list | `df.explode("c")` | `df.explode("c")` | `df.select(explode(...))` | `ds.explode("c")` |
| Unpivot / melt | `df.melt(...)` | `df.unpivot(...)` | `df.unpivot(...)` | `ds.unpivot(index=..., on=...)` |
| Sample rows | `df.sample(frac=f)` | `df.sample(fraction=f)` | `df.sample(fraction=f)` | `ds.sample(f, seed=...)` |
| Pivot / wide | `df.pivot_table(...)` | `df.pivot(...)` | `df.groupBy(i).pivot(c)` | `ds.pivot(index=..., on=..., values=...)` |
| Window expr | n/a | `e.over(...)` | `e.over(Window...)` | `agg.over(partition_by=...)` |

In aggregates and the `window()` function table, Batcher uses `mean` as the canonical
name, matching pandas and Polars. `avg` is accepted as a synonym, so SQL muscle memory
still works.

## Terminal operations

A terminal operation is what triggers the plan to run. These are the equivalents.

| Task | pandas | Polars | PySpark | Batcher |
|------|--------|--------|---------|---------|
| Materialize | (eager) | `df.collect()` | `df.collect()` | `ds.collect()` / `ds.to_arrow()` |
| Row count | `len(df)` | `df.height` | `df.count()` | `ds.count()` |
| Preview | `df.head()` | `df.head()` | `df.show()` | `ds.show()` |
| Summary stats | `df.describe()` | `df.describe()` | `df.summary()` | `ds.describe()` |
| Null counts | `df.isnull().sum()` | `df.null_count()` | n/a | `ds.null_count()` |
| Stream batches | n/a | n/a | `df.toLocalIterator()` | `ds.iter_batches()` |
| Explain plan | n/a | `df.explain()` | `df.explain()` | `ds.explain()` |
| Measured per-op stats | n/a | n/a | n/a | `ds.stats()` |

`ds.write(path, mode=...)` takes the Spark save modes: `overwrite`, the default,
`error`, `ignore`, and `append`, which lakehouse sinks accept. For Delta upserts,
`ds.write.delta(uri, merge_on=["id"])` runs a transactional `MERGE INTO` that updates
matched rows and inserts new ones. That's the Spark and Delta `MERGE` in one call.

## Moving data in and out

Every `from_*` constructor has a symmetric `to_*` exporter, so Batcher slots into an
existing pipeline without a copy where the framework's Arrow bridge allows it.

```python
# docs: skip
import pandas as pd
import batcher as bt

ds = bt.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))   # pandas  -> Batcher
pdf = ds.filter(bt.col("a") > 1).to_pandas()          # Batcher -> pandas
pl_df = ds.to_polars()                                # Batcher -> Polars
table = ds.to_arrow()                                 # Batcher -> pyarrow.Table
```

Each row pairs a source system with its constructor and, where one exists, its exporter.

| Source | In | Out |
|--------|----|----|
| Arrow | `bt.from_arrow(t)` | `ds.to_arrow()` |
| Python dict / rows | `bt.from_pydict(d)` / `bt.from_pylist(rows)` | `ds.to_pydict()` / `ds.to_pylist()` |
| pandas | `bt.from_pandas(df)` | `ds.to_pandas()` |
| Polars | `bt.from_polars(df)` | `ds.to_polars()` |
| NumPy | `bt.from_numpy(arr)` | n/a |
| Ray Data | `bt.from_ray_dataset(ds)` | n/a |
| Spark | `bt.from_spark(df)` | n/a |
| Dask | `bt.from_dask(ddf)` | n/a |
| HuggingFace | `bt.from_huggingface(ds)` | n/a |
| PyTorch | `bt.from_torch(ds)` | `ds.to_torch()` / `ds.to_torch_dataloader()` |
| TensorFlow | `bt.from_tf(ds)` | `ds.to_tf()` |

The `to_torch` and `to_tf` exporters yield a re-iterable dataset of per-batch tensor
dicts, so a multi-epoch training loop streams the query in bounded memory.

## Batch inference and ML pipelines

`ds.map_batches(fn)` runs a function over Arrow batches, and `ds.ml.infer(model)` and
`ds.ml.embed(model)` run a model. Pass a class instead of an instance and the model
loads once per worker, with `num_gpus=` and `concurrency=` for GPU actor pools. The
relational work around the model goes through the same optimizer (Kyber) and resource
manager (Carbonite) as any other query, so it's planned and sized for you rather than
executed as written.

The entry points below cover the common ML shapes.

| Task | Batcher | Note |
|------|---------|------|
| Map a model over batches | `ds.ml.map_batches(Model, ...)` | class = model loaded once per worker |
| Batch inference | `ds.ml.infer(model, num_gpus=, concurrency=)` | CPU readers feed GPU actors |
| Embeddings | `ds.ml.embed(model)` / `batcher.ml.embed(...)` | text or image to a vector column |
| LLM generation | `batcher.ml.llm_generate(..., engine=vllm_engine("..."))` | engine self-batches; no outer PID |
| Distributed training feed | `ds.ml.stream_loader(world_size=, rank=, ...)` | deterministic, balanced, resumable |
| Per-op metrics | `ds.stats()` | measured rows, time, bytes, and bottleneck |
| Bounded output files | `ds.write.parquet(max_rows_per_file=)` | honored even with `partition_by` |
| Resumable writes | `ds.write.parquet(resume=True)` | skips committed shards on re-run |

Settings other engines make you tune by hand are measured defaults here. Batch size
adapts toward throughput under a VRAM cap, `num_gpus` adapts to observed GPU
utilization, and there's no object-store proportion to set, because the data plane
bypasses it. For timings, run `python benchmarks/run.py`, which checks every result
against DuckDB and Polars before it reports a number.

`ds.stats()` answers "where is my time going". It runs the query and reports measured
rows, wall time, peak bytes, and spill per operator, plus which one was the bottleneck:

```python
import batcher as bt
from batcher import col

ds = bt.from_pydict({"city": ["NYC", "LA", "NYC", "SF"], "amount": [10, 20, 30, 40]})
stats = ds.filter(col("amount") > 15).group_by("city").agg(total=col("amount").sum()).stats()
print(stats.rows, stats.bottleneck is not None)
# 3 True
```

Batch writes are atomic and resumable, so a job killed by a spot preemption re-runs
without losing or duplicating data, and `max_rows_per_file` bounds each output file:

```python
import batcher as bt

ds = bt.from_pydict({"v": list(range(1000))})
ds.write.parquet("/tmp/bt_resume_demo", max_rows_per_file=400)            # 3 part files
ds.write.parquet("/tmp/bt_resume_demo", max_rows_per_file=400, resume=True)  # skips committed
print(bt.read.parquet("/tmp/bt_resume_demo").count())
# 1000
```

Feeding a distributed PyTorch trainer, whether DDP, FSDP, or DeepSpeed, uses
`stream_loader`. It gives every rank the same number of batches in a seed-reproducible
order that's independent of world size, so a job can resume on a differently sized
cluster with no repeated or skipped samples. Disable the framework's own sampler,
because `stream_loader` is the single shard authority.

```python
# docs: skip  (requires torch; shown for reference)
loader = ds.ml.stream_loader(batch_size=256, world_size=8, rank=0, epoch=0, seed=1)
for batch in loader:          # {column: torch.Tensor}, this rank's shard
    train_step(batch)
```

Offline LLM batch inference wraps any text-generation engine, such as vLLM behind the
`batcher-engine[vllm]` extra. The engine is built once per worker, and `template` and
`parse_json` handle prompt templating and structured-output parsing:

```python
# docs: skip  (requires a GPU + batcher-engine[vllm]; shown for reference)
from batcher.ml import llm_generate, vllm_engine

for out in llm_generate(
    ds.iter_batches(),
    vllm_engine("meta-llama/Llama-3.1-8B-Instruct", max_model_len=4096),
    prompt_column="question",
    template="Answer concisely. Q: {question}",
):
    ...
```

## Porting with a coding agent

Each source system above has an agent skill that turns these tables into a procedure:
`migrate-from-spark`, `migrate-from-polars-or-pandas`, `migrate-from-duckdb-sql`,
`migrate-from-ray-data`, and `migrate-from-daft`. Beyond the mappings, each carries the
concept shifts that silently produce wrong or slow results, and a recipe that finishes by
proving the ported script returns the same rows as the original. See {doc}`../agents/index`.

## Requirements and limitations

- `from_pandas`, `from_polars`, `from_spark`, `from_dask`, `from_ray_dataset`,
  `from_huggingface`, `from_torch`, and `from_tf` each need the source framework
  installed. Batcher doesn't depend on any of them.
- Several source systems have a constructor but no exporter. NumPy, Ray Data, Spark,
  Dask, and HuggingFace are one-way, so round-trip through `to_arrow` or `to_pandas`.
- `append` mode is accepted by lakehouse sinks only.
- `merge_on` is a `write.delta` parameter. It has no equivalent on a plain Parquet
  write.
- Distributed execution and the GPU actor pools need the optional `[ray]` extra.
- LLM generation needs a text-generation engine you install separately, such as
  `batcher-engine[vllm]`.

## See also

- {doc}`../agents/index`: the migration skills, with the failure modes and the
  verification procedure.
- {doc}`../user-guide/index`: the task-oriented guides for the API this page maps onto.
- {doc}`../architecture/overview`: why a `Dataset` is lazy, and what runs where.
