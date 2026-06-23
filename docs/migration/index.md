# Migrating to Batcher

Batcher's surface is deliberately close to the libraries you already know. If you
come from pandas, Polars, PySpark, or Ray Data, most of your vocabulary carries
over — the table below maps the common operations, and the round-trip adapters
let you move data in and out without copying.

The one concept to internalize: a `Dataset` is **lazy**. Transformations
(`select`, `filter`, `group_by().agg()`, `join`, …) build a plan and return a new
`Dataset`; nothing runs until a **terminal** operation (`collect`, `to_arrow`,
`to_pandas`, `write`, `count`, `iter_batches`). This is the Polars `LazyFrame`
model, not the eager pandas one.

## Reading and writing

One callable namespace per direction. `bt.read(path)` infers the format; the typed
methods (`bt.read.parquet`, `bt.read.delta`, …) are explicit and discoverable.
`ds.write` mirrors it.

```python
import batcher as bt

ds = bt.from_pydict({"city": ["NYC", "LA", "NYC"], "amount": [10, 20, 30]})
ds.write.parquet("/tmp/sales")          # ds.write("/tmp/sales") also infers parquet
back = bt.read.parquet("/tmp/sales")
print(sorted(back.to_pydict()["amount"]))
# [10, 20, 30]
```

| Task | pandas | Polars | PySpark | Batcher |
|------|--------|--------|---------|---------|
| Read Parquet | `pd.read_parquet(p)` | `pl.read_parquet(p)` | `spark.read.parquet(p)` | `bt.read.parquet(p)` |
| Read CSV | `pd.read_csv(p)` | `pl.read_csv(p)` | `spark.read.csv(p)` | `bt.read.csv(p)` |
| Read Delta | — | `pl.read_delta(p)` | `spark.read.format("delta").load(p)` | `bt.read.delta(p)` |
| Autodetect | — | — | `spark.read.load(p)` | `bt.read(p)` |
| Write Parquet | `df.to_parquet(p)` | `df.write_parquet(p)` | `df.write.parquet(p)` | `ds.write.parquet(p)` |
| Write Delta | — | `df.write_delta(p)` | `df.write.format("delta").save(p)` | `ds.write.delta(p)` |

## Transforming

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

| Task | pandas | Polars | PySpark | Batcher |
|------|--------|--------|---------|---------|
| Select / project | `df[["a", "b"]]` | `df.select("a", "b")` | `df.select("a", "b")` | `ds.select("a", "b")` |
| Derive column | `df.assign(c=...)` | `df.with_columns(c=...)` | `df.withColumn("c", ...)` | `ds.with_columns(c=...)` |
| Filter rows | `df[df.a > 1]` | `df.filter(pl.col("a") > 1)` | `df.filter(df.a > 1)` | `ds.filter(col("a") > 1)` |
| Group + aggregate | `df.groupby("k").sum()` | `df.group_by("k").agg(...)` | `df.groupBy("k").agg(...)` | `ds.group_by("k").agg(...)` |
| Mean aggregate | `df.a.mean()` | `pl.col("a").mean()` | `F.avg("a")` | `col("a").mean()` |
| Sort | `df.sort_values("a")` | `df.sort("a")` | `df.orderBy("a")` | `ds.sort("a")` |
| Join | `df.merge(o, on="k")` | `df.join(o, on="k")` | `df.join(o, "k")` | `ds.join(o, on="k")` |
| ASOF join | `pd.merge_asof(...)` | `df.join_asof(...)` | `ASOF JOIN` | `ds.join_asof(o, on=..., by=...)` |
| Distinct | `df.drop_duplicates()` | `df.unique()` | `df.distinct()` | `ds.distinct()` |
| Limit | `df.head(n)` | `df.head(n)` | `df.limit(n)` | `ds.limit(n)` |
| Window rank | — | `pl.col(..).rank().over(..)` | `F.rank().over(Window...)` | `rank().over(partition_by=.., order_by=..)` |
| Window | — | `.over(...)` | `F....over(Window...)` | `ds.window(partition_by=..., functions=...)` |
| Collect list | `df.groupby(k)[c].agg(list)` | `pl.col(c).implode()` | `F.collect_list(c)` | `col(c).array_agg()` |
| First / last | `df.groupby(k).first()` | `pl.col(c).first()` | `F.first(c)` | `col(c).first(order_by=..)` |
| Column ref | `df["a"]` | `df["a"]` | `df["a"]` | `ds["a"]` |
| Row slice | `df[:n]` | `df[:n]` | — | `ds[:n]` |
| Fill nulls | `df.fillna(0)` | `df.fill_null(0)` | `df.fillna(0)` | `ds.fill_null(0)` |
| Drop nulls | `df.dropna()` | `df.drop_nulls()` | `df.dropna()` | `ds.drop_nulls()` |
| Cast | `df.astype({...})` | `df.cast({...})` | `df.withColumn(...)` | `ds.cast({...})` |
| Global agg | `df.sum()` | `df.select(...sum())` | `df.agg(...)` | `ds.agg(...)` |
| Explode list | `df.explode("c")` | `df.explode("c")` | `df.select(explode(...))` | `ds.explode("c")` |
| Unpivot / melt | `df.melt(...)` | `df.unpivot(...)` | `df.unpivot(...)` | `ds.unpivot(index=..., on=...)` |
| Sample rows | `df.sample(frac=f)` | `df.sample(fraction=f)` | `df.sample(fraction=f)` | `ds.sample(f, seed=...)` |
| Pivot / wide | `df.pivot_table(...)` | `df.pivot(...)` | `df.groupBy(i).pivot(c)` | `ds.pivot(index=..., on=..., values=...)` |
| Window expr | — | `e.over(...)` | `e.over(Window...)` | `agg.over(partition_by=...)` |

In aggregates and the `window()` function table, Batcher uses `mean` as the
canonical name (matching pandas/Polars); `avg` is accepted as a synonym so SQL
muscle-memory still works.

## Terminal operations

| Task | pandas | Polars | PySpark | Batcher |
|------|--------|--------|---------|---------|
| Materialize | (eager) | `df.collect()` | `df.collect()` | `ds.collect()` / `ds.to_arrow()` |
| Row count | `len(df)` | `df.height` | `df.count()` | `ds.count()` |
| Preview | `df.head()` | `df.head()` | `df.show()` | `ds.show()` |
| Summary stats | `df.describe()` | `df.describe()` | `df.summary()` | `ds.describe()` |
| Null counts | `df.isnull().sum()` | `df.null_count()` | — | `ds.null_count()` |
| Stream batches | — | — | `df.toLocalIterator()` | `ds.iter_batches()` |
| Explain plan | — | `df.explain()` | `df.explain()` | `ds.explain()` |
| Measured per-op stats | — | — | — | `ds.stats()` |

`ds.write(path, mode=...)` takes the Spark save modes — `overwrite` (default),
`error`, `ignore`, and `append` (lakehouse sinks only). For Delta upserts,
`ds.write.delta(uri, merge_on=["id"])` runs a transactional `MERGE INTO` (matched
rows updated, new rows inserted) — the Spark/Delta `MERGE` in one call.

## Moving data in and out

Every `from_*` constructor has a symmetric `to_*` exporter, so Batcher slots into
an existing pipeline without a copy where the framework's Arrow bridge allows it.

```python
# docs: skip
import pandas as pd
import batcher as bt

ds = bt.from_pandas(pd.DataFrame({"a": [1, 2, 3]}))   # pandas  -> Batcher
pdf = ds.filter(bt.col("a") > 1).to_pandas()          # Batcher -> pandas
pl_df = ds.to_polars()                                # Batcher -> Polars
table = ds.to_arrow()                                 # Batcher -> pyarrow.Table
```

| Source | In | Out |
|--------|----|----|
| Arrow | `bt.from_arrow(t)` | `ds.to_arrow()` |
| pandas | `bt.from_pandas(df)` | `ds.to_pandas()` |
| Polars | `bt.from_polars(df)` | `ds.to_polars()` |
| NumPy | `bt.from_numpy(arr)` | — |
| Spark | `bt.from_spark(df)` | — |
| Dask | `bt.from_dask(ddf)` | — |
| HuggingFace | `bt.from_huggingface(ds)` | — |
| PyTorch | `bt.from_torch(ds)` | `ds.to_torch()` / `ds.to_torch_dataloader()` |
| TensorFlow | `bt.from_tf(ds)` | `ds.to_tf()` |

The `to_torch` / `to_tf` exporters yield a re-iterable dataset of per-batch tensor
dicts, so a multi-epoch training loop streams the query in bounded memory (the
counterpart to Ray Data's `iter_torch_batches`).

## Notes for Ray Data users

Batcher's `ds.map_batches(fn)` and the `ds.ml.infer(model)` / `ds.ml.embed(model)`
accessors mirror Ray Data's batch-mapping and inference APIs (pass a class to load
a model once per worker, with `num_gpus=`/`concurrency=` for GPU actor pools). The
difference is that Batcher puts a real optimizer (Kyber) and an adaptive resource
manager (Carbonite) underneath, so relational work around the model is planned and
sized for you rather than executed as-authored.

| Ray Data | Batcher | Note |
|----------|---------|------|
| `ds.map_batches(Model, ...)` | `ds.ml.map_batches(Model, ...)` | class = model loaded once per worker |
| (batch inference) | `ds.ml.infer(model, num_gpus=, concurrency=)` | CPU readers feed GPU actors |
| (embeddings) | `ds.ml.embed(model)` / `batcher.ml.embed(...)` | text/image → vector column |
| Ray Data LLM + vLLM | `batcher.ml.llm_generate(..., engine=vllm_engine("..."))` | engine self-batches; no outer PID |
| `ray.train` + `iter_torch_batches` | `ds.ml.stream_loader(world_size=, rank=, ...)` | deterministic, balanced, **resumable** |
| `ds.stats()` | `ds.stats()` | *measured* per-op metrics + bottleneck |
| `write_parquet(num_rows_per_output_file=)` | `ds.write.parquet(max_rows_per_file=)` | honored even with `partition_by` |
| (no resume) | `ds.write.parquet(resume=True)` | skips committed shards on re-run |

Several knobs Ray Data makes you set by hand are measured defaults here: batch size
adapts toward throughput under a VRAM cap, `num_gpus` adapts to observed utilization,
and there is no object-store proportion to tune (the data plane bypasses it). On the
sub-million-row queries where Ray Data's per-query overhead dominates, Batcher's
low-overhead engine runs the same filter/group-by/aggregate **markedly faster** —
see `benchmarks/` (run `python benchmarks/run.py --family ray-data`) for the
correctness-gated head-to-head.

`ds.stats()` is the answer to "where is my time going" — it runs the query and
reports each operator's measured rows, wall time, peak bytes, spill, and the
dominant (bottleneck) operator:

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

Feeding a distributed PyTorch (DDP/FSDP) or DeepSpeed trainer uses `stream_loader`,
which gives every rank the same number of batches in a seed-reproducible order that
is independent of world size — so a job can resume on a differently-sized cluster
with no repeated or skipped samples (disable the framework's own sampler; this is
the single shard authority):

```python
# docs: skip  (requires torch; shown for reference)
loader = ds.ml.stream_loader(batch_size=256, world_size=8, rank=0, epoch=0, seed=1)
for batch in loader:          # {column: torch.Tensor}, this rank's shard
    train_step(batch)
```

Offline LLM batch inference over millions of rows wraps any text-generation engine
(vLLM behind `batcher-engine[vllm]`), built once per worker, with prompt templating and
structured-output parsing:

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
