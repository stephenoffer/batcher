# Quick reference

This page is a one-page map of the public API, for looking a name up fast. Everything below is reachable from `import batcher as bt`. The [area pages](index.md) explain the same surface with runnable examples, and the [complete reference](complete.md) renders every signature and docstring.

```python
import batcher as bt

ds = bt.from_pydict({"category": ["a", "b", "a"], "price": [10.0, 20.0, 30.0]})
print(ds.columns)
# ['category', 'price']
```

## Construction

Each of these builds a `Dataset` from data you already hold in the process:

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
| `bt.from_duckdb(rel)` | DuckDB relation, or a connection plus a query |
| `bt.from_dict(d)` / `bt.from_dicts(rows)` / `bt.from_records(rows, columns=...)` | pandas/Polars-spelled aliases |
| `bt.from_iter(iterable)` | any Python iterable or generator, one row per item |
| `bt.from_any(obj)` | dispatches on the type of whatever you hold |
| `bt.concat(frames, how="vertical")` | stack datasets (`vertical`/`vertical_relaxed`/`diagonal`/`horizontal`) |
| `bt.range(stop)` / `bt.date_range(start, end, interval=...)` | generated ranges |

## Readers

All readers take a local or cloud path and return a Dataset. `bt.read` infers the format from the path, and the rest are explicit.

| Call | Format |
| --- | --- |
| `bt.read(path, format=None, **opts)` | inferred |
| `bt.read.parquet`, `bt.read.csv`, `bt.read.json` | tabular files |
| `bt.read.table`, `bt.read.orc`, `bt.read.arrow`, `bt.read.avro` | tabular files |
| `bt.read.lance`, `bt.read.delta`, `bt.read.iceberg`, `bt.read.hudi` | lakehouse tables |
| `bt.read.images`, `bt.read.audio`, `bt.read.video` | multimodal |
| `bt.read.sql`, `bt.read.snowflake`, `bt.read.bigquery`, `bt.read.kafka` | external systems |

The pandas and Polars spellings work too, as top-level shorthands for the same lazy readers: `bt.read_csv`, `bt.read_parquet`, `bt.read_json`, `bt.read_ndjson`, `bt.read_ipc`, `bt.read_orc`, `bt.read_avro`, `bt.read_excel`, `bt.read_delta`, `bt.read_iceberg`, and `bt.read_database`. `bt.read_table(name, ...)` constructs any registered connector by name.

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
| `.tail(n=5)` | take a suffix (executes a `count` first) |
| `.gather_every(n, offset=0)` | keep every `n`-th row (downsample) |
| `.reverse()` | reverse the row order |
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
| `.fill_null(strategy=...)` | fill from a statistic (`"zero"`/`"mean"`/`"min"`/`"max"`) or carry a neighbour (`"forward"`/`"backward"`, which require `order_by`) |
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

Each of these executes the plan and returns a result or writes it out:

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
| `.to_numpy(columns=None)` | a `{column: numpy.ndarray}` dict (tensor columns → `(n, *shape)`) |
| `.to_jax(columns=None)` | a `{column: jax.Array}` dict, the JAX counterpart of `to_numpy` |
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
| `.corr_matrix(columns=None)` | pairwise Pearson correlation matrix over numeric columns (one scan) |
| `.cov_matrix(columns=None)` | pairwise sample covariance matrix over numeric columns (PCA/whitening input) |
| `.approx_quantile(column, q)` | a sketch-based quantile estimate |
| `.stats()` | the last run's measured `RunStats` |
| `__arrow_c_stream__()` | Arrow PyCapsule export, so `pl.DataFrame(ds)`, `duckdb.sql("... FROM ds")`, and `pa.table(ds)` consume a `Dataset` directly, lazily and zero-copy |

```python
out = ds.filter(bt.col("price") >= 20).sort("price", descending=True)
print(out.to_pydict())
# {'category': ['a', 'b'], 'price': [30.0, 20.0]}
```

## GroupBy

`group_by(*keys)` returns a `GroupBy`. Finalize it with `.agg(**named_aggs)`, where each keyword is the output column name. `group_by()` with no keys aggregates the whole dataset.

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

These build the `Expr` values every column operation takes:

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

## Column selectors

Each stands for every column matching a predicate, and expands against the input
schema wherever a column is expected. They produce a `Selector`.

| Call | Selects |
| --- | --- |
| `bt.all()` | every column |
| `bt.numeric()` / `bt.integer()` / `bt.floating()` | numeric / integer / float columns |
| `bt.string()` / `bt.boolean()` / `bt.temporal()` | string / boolean / date-time columns |
| `bt.exclude(*names)` | every column except the named ones |

`bt.by_dtype`, `bt.matches`, `bt.starts_with`, `bt.ends_with`, and `bt.contains` select by dtype or by name pattern. See the [complete reference](complete.md) for their signatures.

## Scalar, aggregate, and window functions

These are the top-level function forms. Rows marked `(aggregate)` belong inside `.agg(...)`, and rows marked `(window)` need a `.over(...)` binding.

| Call | Meaning |
| --- | --- |
| `bt.count()` | COUNT(*) aggregate |
| `bt.iff(condition, if_true, if_false)` | `if_true` where `condition` is true, else `if_false` (DuckDB `IFF`) |
| `bt.nanvl(value, fallback)` | `value` unless it is NaN, then `fallback` (Spark `nanvl`) |
| `bt.next_after(value, toward)` | the adjacent representable float, one ULP toward `toward` (DuckDB `nextafter`) |
| `bt.concat(*exprs)` | concatenate values into one string |
| `bt.concat_ws(separator, *exprs)` | concatenate values with `separator` between them |
| `bt.format_string(format, *exprs)` | interpolate values into a `{}` template (Polars `format`) |
| `bt.mask(e, show_first=0, show_last=0, char="X")` | redact a string, optionally revealing its ends |
| `bt.hmac_sha256(e, key)` | keyed, irreversible pseudonym that still joins |
| `bt.aes_encrypt(e, key)` / `bt.aes_decrypt(e, key)` | deterministic AES-256-GCM-SIV column encryption |
| `bt.log(base, value)` | logarithm of `value` in the given `base` (→ Float64) |
| `bt.gcd(a, b)` / `bt.lcm(a, b)` | greatest common divisor / least common multiple |
| `bt.hypot(a, b)` | Euclidean norm `sqrt(a² + b²)` |
| `bt.great_circle_distance(lat1, lon1, lat2, lon2, unit="km")` | haversine distance between two lat/lon points, in `km`/`m`/`mi`/`nm` |
| `bt.width_bucket(value, low, high, count)` | histogram bucket index over `[low, high]` |
| `bt.struct(**fields)` / `bt.named_struct(name, value, ...)` | build a struct column |
| `bt.sequence(start, stop, step=1)` | per-row integer list `[start..stop]` inclusive (DuckDB `generate_series`) |
| `bt.element()` | the current element inside `list.transform` / `list.filter` (Polars) |
| `bt.sum_horizontal(*exprs)` / `bt.mean_horizontal(*exprs)` | row-wise sum / mean across columns, ignoring nulls (Polars) |
| `bt.min_horizontal(*exprs)` / `bt.max_horizontal(*exprs)` | row-wise min / max across columns, ignoring nulls (Polars) |
| `bt.all_horizontal(*exprs)` / `bt.any_horizontal(*exprs)` | row-wise boolean AND / OR across columns (Polars) |
| `bt.hash_rows(*exprs, seed=0)` | deterministic 64-bit row digest (also `expr.hash(seed=0)`) |
| `bt.count_if(condition)` | count rows where `condition` is true (aggregate) |
| `bt.sum(x)` / `bt.mean(x)` / `bt.min(x)` / `bt.max(x)` / `bt.median(x)` / `bt.std(x)` / `bt.var(x)` / `bt.n_unique(x)` | the SQL-style column-aggregate shorthands for `col(x).<agg>()` |
| `bt.product(x)` / `bt.mode(x)` / `bt.skewness(x)` / `bt.kurtosis(x)` | product / most-frequent value / 3rd / 4th standardized moment (aggregate) |
| `bt.bool_and(x)` / `bt.bool_or(x)` | boolean AND / OR reduction of a group (aggregate) |
| `bt.bit_and(x)` / `bt.bit_or(x)` / `bt.bit_xor(x)` | bitwise AND / OR / XOR reduction of integers (aggregate) |
| `bt.array_agg(x)` | collect each group's values into a list (aggregate) |
| `bt.quantile(x, q)` / `bt.approx_quantile(x, q)` / `bt.approx_median(x)` | exact / sketch-based quantile / median (aggregate) |
| `bt.approx_n_unique(x)` / `bt.histogram(x)` | HyperLogLog distinct count / value→count map (aggregate) |
| `bt.corr(x, y)` | Pearson correlation (aggregate) |
| `bt.covar_pop(x, y)` / `bt.covar_samp(x, y)` | population / sample covariance (aggregate) |
| `bt.regr_slope(y, x)` / `bt.regr_intercept(y, x)` / `bt.regr_r2(y, x)` | least-squares slope / intercept / R² of `y` on `x` (aggregate) |
| `bt.regr_count(y, x)` / `bt.regr_avgx(y, x)` / `bt.regr_avgy(y, x)` | paired sample size and per-axis means (aggregate) |
| `bt.regr_sxx(y, x)` / `bt.regr_syy(y, x)` / `bt.regr_sxy(y, x)` | regression sums of squares / cross-products (aggregate) |
| `bt.var_pop(x)` / `bt.stddev_pop(x)` | population variance / standard deviation (aggregate; `var`/`std` are the sample forms) |
| `bt.geometric_mean(x)` / `bt.harmonic_mean(x)` / `bt.rms(x)` | geometric / harmonic / quadratic (root-mean-square) mean (aggregate) |
| `bt.cv(x)` / `bt.sem(x)` / `bt.midrange(x)` | coefficient of variation / standard error of the mean / midrange (aggregate) |
| `bt.weighted_mean(value, weight)` | mean of `value` weighted by `weight` (aggregate) |
| `bt.lag(expr, n=1)` / `bt.lead(expr, n=1)` | the value `n` rows before / after the current row (window) |
| `bt.first_value(expr)` / `bt.last_value(expr)` | the first / last value of the ordered partition (window) |
| `bt.nth_value(expr, n)` | the `n`-th value of the ordered partition (window) |
| `bt.current_timestamp()` | current timestamp, bound at plan-build time |
| `bt.current_date()` | today's date, bound at plan-build time |
| `bt.date_part(part, expr)` | extract a calendar field (`year`/`month`/`dow`/…) |
| `bt.date_add(expr, days)` | add a whole number of `days` to a date/time column (Spark `date_add`) |
| `bt.date_sub(expr, days)` | subtract a whole number of `days` from a date/time column (Spark `date_sub`) |
| `bt.make_date(year, month, day)` | build a Date from integer components; an impossible date is null |
| `bt.make_timestamp(year, month, day, hour=0, minute=0, second=0)` | build a Timestamp from components |
| `bt.from_epoch(expr, unit="s")` | read an integer epoch column as a Timestamp at a stated unit (`s`/`ms`/`us`/`ns`) |
| `bt.from_unix_date(expr)` | read an integer column of days since 1970-01-01 as a Date |

## Top-level helpers

These sit outside the `Dataset` and `Expr` surfaces:

| Call | Returns |
| --- | --- |
| `bt.date_range(start, end, *, interval_days=1, name="date")` | the date-dimension generator: a one-column Dataset of dates, inclusive, as ISO `YYYY-MM-DD` |
| `bt.compact(path, *, target_size_mb=128.0, num_files=None, by=None, format=None, **opts)` | rewrite many small files at `path` into fewer larger ones in place; returns a `WriteManifest` |
| `bt.engine_version()` | the version reported by the compiled Rust engine (`str`) |
| `bt.start_ui(*, port=4040, host="127.0.0.1", open_browser=False)` | start the web dashboard (queries, plan DAG, per-operator timings, live logs); returns its URL |
| `bt.stop_ui()` | stop the web dashboard; safe to call when none is running |
| `bt.ui_url()` | the running dashboard's URL, or `None` |

## Expression methods

- Operators: `+ - * / % **`; `== != > >= < <=`; `& | ~`
- Types and nulls: `.cast("Int64")`, `.try_cast("Int64")`, `.is_null()`,
  `.is_not_null()`, `.is_in([...])`, `.between(low, high)`, `.fill_null(value)`,
  `.fill_nan(value)`, `.eq_missing(other)`, `.is_nan()`, `.is_not_nan()`,
  `.is_finite()`, `.is_infinite()`, `.clip(lower, upper)`, `.alias(name)`
- Binning and gap-filling: `.cut(breaks, labels=None, left_closed=False)`,
  `.forward_fill()` and `.backward_fill()`. The two fills are window functions, so bind them with `.over(order_by=[...])`. An order is required.
- Math: `.abs()`, `.round(digits)`, `.pow(e)`, `.sqrt()`, `.floor()`, `.ceil()`,
  `.ln()`, `.log10()`, `.log2()`, `.exp()`, `.sin()`, `.cos()`, `.tan()`, `.asin()`,
  `.acos()`, `.atan()`, `.sinh()`, `.cosh()`, `.tanh()`, `.cot()`, `.sign()`,
  `.trunc()`, `.cbrt()`, `.degrees()`, `.radians()`, `.factorial()`
- Bitwise (integers): `.bitwise_and(o)`, `.bitwise_or(o)`, `.bitwise_xor(o)`,
  `.bitwise_left_shift(o)`, `.bitwise_right_shift(o)`, `.bit_count()`
- Aggregates (inside `.agg`): `.sum()`, `.min()`, `.max()`, `.mean()`, `.var()`,
  `.std()`, `.median()`, `.quantile(q)`, `.skewness()`, `.kurtosis()`, `.count()`,
  `.n_unique()` / `.count_distinct()`, `.mode()`, `.first()`, `.last()`,
  `.arg_min()`, `.arg_max()`, `.bool_and()`, `.bool_or()`,
  `.bit_and()` / `.bit_or()` / `.bit_xor()`, `.histogram()`, `.array_agg()`
- Approximate aggregates, sketch-backed and mergeable so they scale: `.approx_n_unique()` /
  `.approx_count_distinct()` (HyperLogLog), `.approx_quantile(q)` / `.approx_median()` (KLL)
- Cumulative & window analytics (bind with `.over(...)`): `.cum_sum()` / `.cum_min()` /
  `.cum_max()` / `.cum_count()`, `.rolling_sum(k)` / `.rolling_mean(k)` /
  `.rolling_min(k)` / `.rolling_max(k)` / `.rolling_count(k)`, `.diff(n=1)`,
  `.pct_change(n=1)`, `.shift(n)`, `.rank(method="min")`, `.is_duplicated()` /
  `.is_unique()`
- Full list, plus the `.str` / `.dt` / `.list` / `.struct` / `.json` / `.map` /
  `.image` / `.audio` / `.video` accessors: the [expressions API page](expressions.md).

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

Typed methods hang off an expression by namespace rather than crowding `Expr` itself:

| Namespace | Covers |
| --- | --- |
| `.str` | casing, trim, search, slice, pad, encode (`upper`, `contains`, `like`, `ilike`, `substr`, `split`, `regexp_replace`, ...) plus unstructured-text ingest: `strip_html()`, `chunk(size, overlap)`, `minhash(num_perm, ngram)` |
| `.dt` | calendar parts (`year`, `month`, `day`, `hour`, `dayname`, `quarter`, `truncate`, ...) |
| `.list` | list reductions and reshaping (`len`, `sum`, `sort`, `get`, `join`, `contains`, ...) plus vector ops for retrieval/RAG (`cosine_similarity`, `cosine_distance`, `l2_distance`, `dot`, `normalize`, `jaccard`) and the LSH blocking key `simhash(num_bits, seed=0)` |
| `.struct` | `field(name)` |
| `.map` | Arrow `Map` columns: `keys()`, `values()`, `get(key)` |
| `.json` | `extract_string(path)` |
| `.image` | `decode()`, `to_tensor(width, height)`, `to_tensor_f32(width, height, mean=, std=, channels_first=)`, `center_crop(width, height)`, `to_grayscale(width, height)`, `resize(width, height)` |
| `.audio` | native WAV/FLAC decode: `decode()` (metadata struct), `to_waveform()` (mono `List<Float32>`), `resample(rate)`, `mel_spectrogram(rate, n_fft=, hop_length=, n_mels=)` (speech-model mel power spectrogram, torchaudio-matching), `mfcc(rate, n_fft=, hop_length=, n_mels=, n_mfcc=)` (MFCC feature, torchaudio-matching) |
| `.video` | native FFmpeg decode: `decode()` returns a metadata struct, and needs the `video` engine build feature |

## SQL

`bt.sql(query, table_name=ds_or_table, ...)` returns a Dataset. Each table named in the query is bound by a keyword argument. The [SQL page](sql.md) lists the supported clauses and features in full.

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
| `ds.ml.generate(engine, ...)` | offline LLM text generation |
| `ds.ml.extract(engine, schema=...)` | LLM → **typed** columns (AI-powered ETL) |
| `ds.ml.classify(engine, labels=[...])` | zero-shot labelling, domain pinned to `labels` |
| `ds.ml.near_duplicates(col)` / `drop_near_duplicates(col)` | MinHash+LSH fuzzy dedup |
| `ds.ml.similarity_join(other, left_on=...)` | join two datasets on embedding similarity |

## Preprocessors

`batcher.ml.preprocessors` holds the scikit-learn-style `fit`/`transform` estimators. Fit on the training split and transform both. `Chain` composes them into one pipeline.

| Class | Learns |
| --- | --- |
| `StandardScaler` / `MinMaxScaler` / `MaxAbsScaler` / `RobustScaler` | per-column scaling statistics |
| `Normalizer` | stateless per-row vector normalization |
| `OneHotEncoder` / `MultiHotEncoder` / `LabelEncoder` / `OrdinalEncoder` | the category vocabulary |
| `KBinsDiscretizer` | bin edges (quantile or uniform) |
| `SimpleImputer` | the fill statistic (mean/median/most-frequent/constant) |
| `Tokenizer` / `Concatenator` | stateless text split / feature-vector assembly |
| `Chain` | each step, fit on the previous step's output |

See the [preprocessors guide](../ml/preprocessors/index.md) for the workflow and the
[ML API page](ml.md) for the per-class reference.

## Configuration

```python
from batcher import Config, set_config, config_context
```

`Config()` is a frozen dataclass of sections (`execution`, `memory`, `flow_control`,
`optimizer`, `pid`, `metadata`). Derive a modified Config and apply it process-wide
with `set_config(...)` or temporarily with `config_context(...)`. `Config.from_env`
and `Config.from_file` overlay `BATCHER_*` environment variables and a JSON file.
See the configuration page for the full pattern.

## See also

- {doc}`dataset`: every `Dataset` method, with its arguments and its return type.
- {doc}`expressions` and {doc}`expression-accessors`: the column language and the
  `.str` / `.dt` / `.list` / `.struct` / `.json` namespaces.
- {doc}`functions`: the free functions, grouped by family.
- {doc}`io`: readers, writers, save modes, and the format-specific options.
- {doc}`../user-guide/index`: the task-oriented guides behind these signatures.
- {doc}`../getting-started/quickstart`: the same surface as a five-minute walkthrough.
