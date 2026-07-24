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

If your fingers already type `pd.read_csv`, keep typing it. Every common format also
has a top-level shorthand under the ecosystem-standard name, and each one is the same
lazy reader as its `bt.read.*` twin:

| Shorthand | Same as | Reads |
|-----------|---------|-------|
| `bt.read_csv(p)` | `bt.read.csv(p)` | CSV files, directories, globs |
| `bt.read_parquet(p)` | `bt.read.parquet(p)` | Parquet |
| `bt.read_json(p)` / `bt.read_ndjson(p)` | `bt.read.json(p)` | newline-delimited JSON |
| `bt.read_ipc(p)` | `bt.read.arrow(p)` | Arrow IPC / Feather |
| `bt.read_orc(p)` | `bt.read.orc(p)` | ORC |
| `bt.read_avro(p)` | `bt.read.avro(p)` | Avro |
| `bt.read_excel(p)` | `bt.read.excel(p)` | Excel workbooks |
| `bt.read_delta(p)` | `bt.read.delta(p)` | Delta Lake tables |
| `bt.read_iceberg(t)` | `bt.read.iceberg(t)` | Iceberg tables |
| `bt.read_database(q, uri=...)` | `bt.read.sql(q, uri=...)` | any SQL database |

For a source that isn't a file at all, `bt.read_table(name, ...)` constructs any
registered connector by name, which is the escape hatch behind all of the above.

## Getting data in from another library

Whatever object you're holding, there's a constructor for it. The names follow pandas
and Polars, so `from_dict` and `from_dicts` mean what they mean there:

| You have | Call |
|----------|------|
| A `{column: values}` dict | `bt.from_pydict(d)`, or `bt.from_dict(d)` |
| A list of row dicts | `bt.from_pylist(rows)`, or `bt.from_dicts(rows)` |
| A list of row tuples | `bt.from_records(rows, columns=[...])` |
| A generator or any iterable | `bt.from_iter(gen)` |
| A pandas or Polars frame | `bt.from_pandas(df)` / `bt.from_polars(df)` |
| A DuckDB relation or connection | `bt.from_duckdb(rel)` |
| An Arrow table, or anything Arrow-exporting | `bt.from_arrow(t)` |
| Something whose type you don't know | `bt.from_any(obj)` |

`bt.from_any` is the one to reach for in migration code and glue: it dispatches on the
type and routes to the right constructor, so a script that accepts "a frame" from a
caller doesn't have to branch. `bt.sql` uses it for every bound table, which is why
you can pass a pandas frame or a plain dict straight into a query:

```python
import batcher as bt

print(bt.sql("SELECT x * 2 AS y FROM t", t={"x": [1, 2, 3]}).to_pydict())
# {'y': [2, 4, 6]}
```

## Concatenating and generating

`bt.concat` means frame concatenation, exactly as `pd.concat` and `pl.concat` do, and
takes Polars' `how` vocabulary. The string-building `concat` keeps its own explicit
name, `bt.concat_str`:

```python
import batcher as bt

a = bt.from_pydict({"x": [1, 2]})
b = bt.from_pydict({"x": [3, 4]})
print(bt.concat([a, b]).to_pydict())
# {'x': [1, 2, 3, 4]}

wide = bt.concat([bt.from_pydict({"x": [1]}), bt.from_pydict({"y": ["a"]})], how="diagonal")
print(wide.to_pydict())
# {'x': [1, None], 'y': [None, 'a']}

ds = bt.from_pydict({"first": ["ada"], "last": ["lovelace"]})
print(ds.select(name=bt.concat_str(bt.col("first"), bt.lit(" "), bt.col("last"))).to_pydict())
# {'name': ['ada lovelace']}
```

`how` is `"vertical"` (the default), `"vertical_relaxed"` to deduplicate,
`"diagonal"` to stack over the union of the columns, or `"horizontal"` to place frames
side by side by row position.

`bt.range` follows `builtins.range`, single-argument form included, and `bt.date_range`
follows `pandas.date_range` and `polars.date_range`:

```python
import batcher as bt

print(bt.range(5).to_pydict())
# {'value': [0, 1, 2, 3, 4]}
print(bt.date_range("2024-01-01", periods=3, interval="1mo").count())
# 3
```

Pass `end=` or `periods=`, the stride as `interval=` (Polars) or `freq=` (pandas), and
`closed=` to drop an endpoint the way pandas' `inclusive=` does.

## Reporting a problem

`bt.show_versions()` prints the Batcher version, the compiled engine version, Python,
the platform, and which optional backends are installed. `bt.versions()` returns the
same information as a dict.

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

## Names and arguments that carry over unchanged

Batcher accepts the spelling you already type for a long list of operations, so a
ported script usually needs fewer edits than the tables above suggest. Each alias is a
real method that delegates to the Batcher primary, not a shim, so it returns the same
plan and the same result.

| You type | Batcher primary |
|---|---|
| `ds.to_dicts()` | `ds.to_pylist()` |
| `ds.to_dict()` | `ds.to_pydict()` |
| `ds.drop_duplicates()` | `ds.distinct()` |
| `ds.with_row_count()` | `ds.with_row_index()` |
| `ds.vstack(other)` / `ds.append(other)` | `ds.union(other)` |
| `ds.difference(other)` | `ds.except_(other)` |
| `ds.persist()` | `ds.cache()` |
| `ds.coalesce(n)` | `ds.repartition(n)` |
| `ds.transform(fn)` | `ds.pipe(fn)` |
| `ds.fillna(v)` / `ds.dropna()` | `ds.fill_null(v)` / `ds.drop_nulls()` |
| `ds.groupby(...)` / `ds.merge(...)` | `ds.group_by(...)` / `ds.join(...)` |
| `ds.sort_values(...)` / `ds.nlargest(...)` | `ds.sort(...)` / `ds.top_k(...)` |
| `gb.nunique()` / `gb.size()` | `gb.n_unique()` / `gb.len()` |
| `ds.query("x > 2")` | `ds.filter(bt.col("x") > 2)` |
| `ds.to_parquet(p)` / `ds.to_csv(p)` / `ds.to_json(p)` | `ds.write.parquet(p)` and friends |
| `ds.first()` / `ds.last()` / `ds.item()` | terminal row accessors |
| `ds.width` / `ds.height` / `ds.empty` | `len(ds.columns)` / `ds.count()` / `ds.is_empty()` |
| `ds.info()` / `ds.glimpse()` / `ds.memory_usage()` | schema-and-count summaries |
| `ds.iter_rows()` / `ds.iter_slices()` | `ds.iter_batches()` |
| `ds.lazy()` / `ds.copy()` | identity — a `Dataset` is already lazy and immutable |

Argument names carry over too. `ds.sort()` takes `by=` and `ascending=` alongside
`descending=`, and `na_position=` alongside `nulls_first=`. `ds.sample()` reads a
positional `int` as a row count and a `float` as a fraction, and accepts `frac=` and
`random_state=`. `ds.melt()` takes `id_vars=`, `value_vars=`, and `var_name=`.
`ds.select_dtypes()` accepts a Python type, a dtype name, or a list of either, and an
`exclude=` argument. `ds.rename()` accepts a function applied to every column name.

Two shorthands have no pandas equivalent but save the parenthesizing that `&`
otherwise needs. Several predicates are ANDed, and a keyword is an equality test:

```python
import batcher as bt

ds = bt.from_pydict({"status": ["paid", "open", "paid"], "amount": [10, 20, 30]})
print(ds.filter(bt.col("amount") > 5, status="paid").to_pydict())
# {'status': ['paid', 'paid'], 'amount': [10, 30]}
```

`ds.group_by(...).agg()` also takes the pandas dict spec, where a list of reducers
suffixes the output names the way pandas does when it flattens:

```python
print(ds.group_by("status").agg({"amount": ["min", "max"]}).sort("status").to_pydict())
# {'status': ['open', 'paid'], 'amount_min': [20, 10], 'amount_max': [20, 30]}
```

## What Batcher deliberately does not have

Some familiar APIs are absent by design rather than by omission, and knowing which is
which saves you looking for a workaround that doesn't exist. Batcher tells you at the
point of use: every one of these raises an `AttributeError` naming the reason and the
replacement, so you can discover the mapping from a traceback instead of this table.

| Absent | Why | Instead |
|---|---|---|
| `df.set_index`, `df.reset_index`, `df.loc`, `df.iloc` | A relation is an unordered multiset with no row index, as in SQL. | `ds.filter(...)`, `ds.select(...)`, `ds.sort(...)`, `ds.with_row_index()` |
| `df.iterrows`, `df.itertuples`, `df.applymap` | Per-row Python never runs on the hot path. | `ds.iter_rows(named=True)` at the end of a pipeline; expressions or `ds.map_batches()` inside one |
| `df.apply` | Its per-row and per-column meanings don't survive a columnar engine. | `ds.with_columns(y=expr)` or `ds.map_batches(fn)` |
| `df.T`, `df.transpose` | Transposing needs a materialized, single-typed frame. | `ds.to_pandas().T`, or `ds.unpivot()` / `ds.pivot()` |
| `df.shift`, `df.diff`, `df.cumsum`, `df.rolling` | Each needs a row order the relation doesn't carry. | `ds.window(order_by=[...], functions={...})` |
| `df.resample` | Time bucketing is a grouping. | `ds.group_by(bucket=bt.col("t").dt.truncate("1h")).agg(...)` |
| Looping over a `GroupBy` | It materializes one frame per key in Python and caps the job at one machine. | `.agg(...)`, or `.window(partition_by=[...])` to keep every row |

Column attribute access (`df.amount`) is absent for a subtler reason: a column named
`filter` or `join` would shadow a method, which is a real source of pandas bugs. Use
`ds["amount"]` for the expression, or `bt.col("amount")` to build one.

## The error messages teach you the mapping

You don't have to memorize the tables above. Type the method you already know, and the
traceback tells you the Batcher spelling. This works at every level: on a `Dataset`, on
an expression, on a `GroupBy`, and on the `bt` package itself.

```python
import batcher as bt

demo = bt.from_pydict({"x": [1, 2, 3], "k": ["a", "b", "a"]})

# A pandas reshape on a Dataset:
try:
    demo.pivot_table
except AttributeError as exc:
    assert "ds.pivot" in str(exc)

# A Polars per-element UDF on an expression:
try:
    bt.col("x").map_elements
except AttributeError as exc:
    assert "map_batches" in str(exc)

# A pandas GroupBy transform:
try:
    demo.group_by("k").transform
except AttributeError as exc:
    assert "ds.window" in str(exc)

# A Polars top-level constructor:
try:
    bt.LazyFrame
except AttributeError as exc:
    assert "already lazy" in str(exc)

print("every wrong spelling names its Batcher replacement")
# every wrong spelling names its Batcher replacement
```

A near miss on a real method gets a `Did you mean ...?` suggestion instead, so a typo
such as `ds.filtr` or `bt.col("x").meen` points straight at `filter` and `mean`.

## Checking a port

`ds.equals(other)` compares *results*, not plans, so it answers the only question that
matters after a migration. Row order is ignored by default, because a relation is
unordered; pass `ordered=True` after a `sort` when the emitted order is part of the
contract.

```python
ported = ds.filter(status="paid")
expected = ds.filter(bt.col("status") == "paid")
print(ported.equals(expected))
# True
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
