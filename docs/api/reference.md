# Quick reference

A one-page map of the public API. Everything below is reachable from
`import batcher as bt`. The Dataset and Expr surfaces are deliberately small: there
is one obvious way to do each thing.

```python
import batcher as bt

ds = bt.from_pydict({"category": ["a", "b", "a"], "price": [10.0, 20.0, 30.0]})
print(ds.columns)
# ['category', 'price']
```

## Construction

| Call | Source |
| --- | --- |
| `bt.from_pydict(mapping)` | column-oriented dict |
| `bt.from_pylist(rows)` | row-oriented list of dicts (JSON records) |
| `bt.from_items(items)` | list of items, one row each (dicts → columns) |
| `bt.from_arrow(table_or_batches)` | pyarrow Table, RecordBatch, or batch list |
| `bt.from_batches(factory, schema)` | streaming source from a batch factory |
| `bt.from_pandas(df)` / `bt.from_polars(df)` / `bt.from_numpy(...)` | framework adapters |
| `bt.from_spark(...)` / `bt.from_dask(...)` / `bt.from_ray_dataset(...)` | distributed-frame adapters |
| `bt.from_torch(...)` / `bt.from_tf(...)` / `bt.from_huggingface(...)` | framework adapters |

## Readers

All readers take a local or cloud path and return a Dataset. `bt.read` infers the
format from the path; the rest are explicit.

| Call | Format |
| --- | --- |
| `bt.read(path, format=None, **opts)` | inferred |
| `bt.read.parquet`, `bt.read.csv`, `bt.read.json` | tabular files |
| `bt.read.table`, `bt.read.orc`, `bt.read.arrow`, `bt.read.avro` | tabular files |
| `bt.read.lance`, `bt.read.delta`, `bt.read.iceberg`, `bt.read.hudi` | lakehouse tables |
| `bt.read.images`, `bt.read.audio`, `bt.read.video` | multimodal |
| `bt.read.sql`, `bt.read.snowflake`, `bt.read.bigquery`, `bt.read.kafka` | external systems |

## Dataset transformations

Each returns a new lazy Dataset.

| Method | Effect |
| --- | --- |
| `.filter(expr)` | keep rows where the predicate is true |
| `.select(*names, **derived)` | choose or derive the full output |
| `.with_columns(**named)` / `.with_column(name, expr)` | add or replace columns |
| `.drop(*names)` | remove columns |
| `.rename({old: new})` | rename columns |
| `.sort(*by, descending=False, nulls_first=False)` | order rows |
| `.limit(n, offset=0)` / `.head(n=5)` | take a prefix |
| `.distinct()` | drop duplicate rows |
| `.union(*others, distinct=False)` | concatenate datasets |
| `.intersect(other)` / `.except_(other)` | set operations |
| `.join(other, on=None, left_on=None, right_on=None, how="inner", suffix="_right")` | join (`how`: inner, left, right, full, outer, semi, anti) |
| `.cross_join(other, suffix="_right")` | Cartesian product |
| `.join_asof(other, on, by=None, ...)` | nearest-key temporal join |
| `.window(partition_by=(), order_by=(), functions={...}, frame=None)` | window functions |
| `.group_by(*keys, **named) -> GroupBy` | start an aggregation |
| `.top_k(k, by, descending=True)` | k rows with the largest/smallest `by` |
| `.sample(fraction=None, n=None, seed=None)` | deterministic seeded row sample |
| `.cast(dtypes)` | cast one column (`"Int64"`) or many (`{col: dtype}`) |
| `.fill_null(value)` | replace nulls (scalar or `{col: value}`) |
| `.drop_nulls(subset=None)` | drop rows with nulls (optionally in `subset`) |
| `.map_batches(fn, ...)` | run a Python callable over whole Arrow batches |

### Reshaping

| Method | Effect |
| --- | --- |
| `.explode(column, alias=None)` | one row per element of a list column |
| `.with_row_index(name="index", offset=0)` | prepend a sequential row-index column (Polars) |
| `.with_random(name="random", seed=0, normal=False)` | add a reproducible seeded random column (uniform or standard normal) |
| `.unnest(*columns)` | lift struct fields into top-level columns |
| `.pivot(index=[...], on=, values=, aggregate="sum")` | long → wide |
| `.unpivot(on=[...], index=[...], ...)` | wide → long |
| `.value_counts(column, name="count")` | frequency of each distinct value |

## Dataset terminal operations

| Method | Returns |
| --- | --- |
| `.collect(distributed=False, num_workers=None, spill=False, num_partitions=16, adaptive=False, transport="disk")` | pyarrow Table |
| `.to_pydict()` | `dict[str, list]` |
| `.to_pylist()` | `list[dict]` |
| `.count()` | row count (`int`) |
| `.iter_batches(batch_size=None)` | iterator of RecordBatch |
| `.explain()` | optimized plan as text |
| `.show(limit=10)` | prints a preview |
| `.write(path, fmt=None, partition_by=None, distributed=False, num_workers=None, **kw)` | WriteManifest |
| `.write.parquet(path, compression="zstd", **kw)` | writes Parquet |
| `.write.csv(path, **kw)` / `.write.json(path, **kw)` | writes CSV / JSON |
| `.to_arrow()` | pyarrow Table (alias of `.collect()`) |
| `.to_pandas()` / `.to_polars()` | a pandas / Polars DataFrame |
| `.to_torch(columns=None, batch_size=None)` / `.to_tf(...)` | a Torch / TensorFlow dataset |
| `.to_torch_dataloader(...)` | a `torch.utils.data.DataLoader` |

### Introspection

These compute (or read) a small result and so are eager.

| Member | Returns |
| --- | --- |
| `.columns` | current schema names (property) |
| `.schema()` / `.dtypes()` | the Arrow schema / its column types |
| `.is_empty()` | whether the dataset has zero rows |
| `.is_streaming()` | whether the source is unbounded |
| `.describe(percentiles=(.25,.5,.75))` | summary statistics per column |
| `.null_count()` | null count per column |
| `.approx_quantile(column, q)` | a sketch-based quantile estimate |
| `.stats()` | the last run's measured `RunStats` |

```python
out = ds.filter(bt.col("price") >= 20).sort("price", descending=True)
print(out.to_pydict())
# {'category': ['a', 'b'], 'price': [30.0, 20.0]}
```

## GroupBy

`group_by(*keys)` returns a `GroupBy`; finalize with `.agg(**named_aggs)`. Each
keyword is the output column name. `group_by()` with no keys aggregates the whole
dataset.

```python
out = (
    ds.group_by("category")
    .agg(total=bt.col("price").sum(), n=bt.count())
    .sort("category")
)
print(out.to_pydict())
# {'category': ['a', 'b'], 'total': [40.0, 20.0], 'n': [2, 1]}
```

## Expression constructors

| Call | Meaning |
| --- | --- |
| `bt.col(name)` | reference a column |
| `bt.lit(value)` | a constant |
| `bt.when(c).then(v)...otherwise(d)` | SQL CASE |
| `bt.coalesce(*exprs)` | first non-null per row (also the SQL `IFNULL` case) |
| `bt.nullif(a, b)` | null when `a == b` |
| `bt.greatest(*exprs)` / `bt.least(*exprs)` | row-wise extreme |
| `bt.array(*exprs)` | build a list column |
| `bt.atan2(y, x)` | two-argument arctangent |
| `bt.count()` | COUNT(*) aggregate |
| `bt.iff(condition, if_true, if_false)` | `if_true` where `condition` is true, else `if_false` (DuckDB `IFF`) |
| `bt.nanvl(value, fallback)` | `value` unless it is NaN, then `fallback` (Spark `nanvl`) |
| `bt.concat(*exprs)` | concatenate values into one string |
| `bt.concat_ws(separator, *exprs)` | concatenate values with `separator` between them |
| `bt.format_string(format, *exprs)` | interpolate values into a `{}` template (Polars `format`) |
| `bt.log(base, value)` | logarithm of `value` in the given `base` (→ Float64) |
| `bt.gcd(a, b)` / `bt.lcm(a, b)` | greatest common divisor / least common multiple |
| `bt.hypot(a, b)` | Euclidean norm `sqrt(a² + b²)` |
| `bt.width_bucket(value, low, high, count)` | histogram bucket index over `[low, high]` |
| `bt.struct(**fields)` / `bt.named_struct(name, value, ...)` | build a struct column |
| `bt.sequence(start, stop, step=1)` | per-row integer list `[start..stop]` inclusive (DuckDB `generate_series`) |
| `bt.element()` | the current element inside `list.transform` / `list.filter` (Polars) |
| `bt.sum_horizontal(*exprs)` / `bt.mean_horizontal(*exprs)` | row-wise sum / mean across columns, ignoring nulls (Polars) |
| `bt.count_if(condition)` | count rows where `condition` is true (aggregate) |
| `bt.corr(x, y)` | Pearson correlation (aggregate) |
| `bt.covar_pop(x, y)` / `bt.covar_samp(x, y)` | population / sample covariance (aggregate) |
| `bt.nth_value(expr, n)` | the `n`-th value of the ordered partition (window) |
| `bt.current_timestamp()` | current timestamp, bound at plan-build time |
| `bt.current_date()` | today's date, bound at plan-build time |
| `bt.date_part(part, expr)` | extract a calendar field (`year`/`month`/`dow`/…) |
| `bt.date_add(expr, days)` | add a whole number of `days` to a date/time column (Spark `date_add`) |
| `bt.date_sub(expr, days)` | subtract a whole number of `days` from a date/time column (Spark `date_sub`) |

## Top-level helpers

| Call | Returns |
| --- | --- |
| `bt.date_range(start, end, *, interval_days=1, name="date")` | a one-column Dataset of dates (inclusive ISO `YYYY-MM-DD`) — the date-dimension generator |
| `bt.compact(path, *, target_size_mb=128.0, num_files=None, by=None, format=None, **opts)` | rewrite many small files at `path` into fewer larger ones in place; returns a `WriteManifest` |
| `bt.engine_version()` | the version reported by the compiled Rust engine (`str`) |

## Expression methods

- Operators: `+ - * / % **`; `== != > >= < <=`; `& | ~`
- Types and nulls: `.cast("Int64")`, `.is_null()`, `.is_not_null()`, `.is_in([...])`,
  `.between(low, high)`, `.fill_null(value)`, `.is_nan()`, `.is_not_nan()`,
  `.is_finite()`, `.is_infinite()`, `.clip(lower, upper)`
- Math: `.abs()`, `.round(digits)`, `.pow(e)`, `.sqrt()`, `.floor()`, `.ceil()`,
  `.ln()`, `.log10()`, `.log2()`, `.exp()`, `.sin()`, `.cos()`, `.tan()`, `.asin()`,
  `.acos()`, `.atan()`, `.sinh()`, `.cosh()`, `.tanh()`, `.cot()`, `.sign()`,
  `.trunc()`, `.cbrt()`, `.degrees()`, `.radians()`
- Aggregates (inside `.agg`): `.sum()`, `.min()`, `.max()`, `.mean()`, `.var()`,
  `.std()`, `.median()`, `.quantile(q)`, `.count()`, `.n_unique()`, `.mode()`,
  `.first()`, `.last()`, `.arg_min()`, `.arg_max()`, `.bool_and()`, `.bool_or()`,
  `.array_agg()`
- Approximate aggregates (sketch-backed, mergeable — for scale): `.approx_n_unique()`
  (HyperLogLog), `.approx_quantile(q)` / `.approx_median()` (KLL)

```python
out = ds.select(
    "category",
    label=bt.when(bt.col("price") >= 20).then(bt.lit("hi")).otherwise(bt.lit("lo")),
    rounded=(bt.col("price") / 3).round(2),
)
print(out.to_pydict())
# {'category': ['a', 'b', 'a'], 'label': ['lo', 'hi', 'hi'], 'rounded': [3.33, 6.67, 10.0]}
```

## Expression accessor namespaces

| Namespace | Covers |
| --- | --- |
| `.str` | casing, trim, search, slice, pad, encode (`upper`, `contains`, `like`, `ilike`, `substr`, `split`, `regexp_replace`, ...) |
| `.dt` | calendar parts (`year`, `month`, `day`, `hour`, `dayname`, `quarter`, `truncate`, ...) |
| `.list` | list reductions and reshaping (`len`, `sum`, `sort`, `get`, `join`, `contains`, ...) plus vector ops for retrieval/RAG (`cosine_similarity`, `cosine_distance`, `l2_distance`, `dot`, `normalize`) |
| `.struct` | `field(name)` |
| `.map` | Arrow `Map` columns: `keys()`, `values()`, `get(key)` |
| `.json` | `extract_string(path)` |
| `.image` | `decode()`, `to_tensor(width, height)`, `resize(width, height)` |
| `.audio` | native WAV/FLAC decode: `decode()` (metadata struct), `to_waveform()` (mono `List<Float32>`) |
| `.video` | native FFmpeg decode: `decode()` (metadata struct) — needs the `video` engine build feature |

## SQL

`bt.sql(query, table_name=ds_or_table, ...)` returns a Dataset. Each table named in
the query is bound by a keyword argument. The supported subset is SELECT, WHERE,
GROUP BY / HAVING, ORDER BY, LIMIT, INNER and LEFT JOIN, CASE, and CAST.

```python
out = bt.sql("SELECT category, SUM(price) AS total FROM t GROUP BY category ORDER BY category", t=ds)
print(out.to_pydict())
# {'category': ['a', 'b'], 'total': [40.0, 20.0]}
```

## ML accessor

The `.ml` accessor runs models over whole Arrow batches. A class callable loads its
model once per worker.

| Method | Use |
| --- | --- |
| `ds.ml.map_batches(fn, ...)` | arbitrary batch transform |
| `ds.ml.infer(model, ...)` | batched inference |
| `ds.ml.embed(model, ...)` | batched embeddings |

## Configuration

```python
from batcher import Config, set_config, config_context
```

`Config()` is a frozen dataclass of sections (`execution`, `memory`, `flow_control`,
`optimizer`, `pid`, `metadata`). Derive a modified Config and apply it process-wide
with `set_config(...)` or temporarily with `config_context(...)`. `Config.from_env`
and `Config.from_file` overlay `BATCHER_*` environment variables and a JSON file.
See the configuration page for the full pattern.
