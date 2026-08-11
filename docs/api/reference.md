# Quick reference

This page is a one-page map of the public API, for looking a name up fast. Everything below is reachable from `import batcher as bt`. The {doc}`area pages <index>` explain the same surface with runnable examples, and the {doc}`complete reference <complete>` renders every signature and docstring.

```python
import batcher as bt

ds = bt.from_pydict({"category": ["a", "b", "a"], "price": [10.0, 20.0, 30.0]})
print(ds.columns)
# ['category', 'price']
```

## Construction

Each of these builds a {py:class}`Dataset <batcher.Dataset>` from data you already hold in the process:

| Call | Source |
| --- | --- |
| {py:func}`bt.from_pydict(mapping) <batcher.from_pydict>` | column-oriented dict |
| {py:func}`bt.from_pylist(rows) <batcher.from_pylist>` | row-oriented list of dicts (JSON records) |
| {py:func}`bt.from_items(items) <batcher.from_items>` | list of items, one row each (dicts → columns) |
| {py:func}`bt.from_arrow(table_or_batches) <batcher.from_arrow>` | pyarrow Table, RecordBatch, or batch list |
| {py:func}`bt.from_batches(factory, schema) <batcher.from_batches>` | streaming source from a batch factory |
| {py:func}`bt.from_pandas(df) <batcher.from_pandas>` / {py:func}`bt.from_polars(df) <batcher.from_polars>` / {py:func}`bt.from_numpy(...) <batcher.from_numpy>` | framework adapters |
| {py:func}`bt.from_spark(...) <batcher.from_spark>` / {py:func}`bt.from_dask(...) <batcher.from_dask>` / {py:func}`bt.from_ray_dataset(...) <batcher.from_ray_dataset>` | distributed-frame adapters |
| {py:func}`bt.from_torch(...) <batcher.from_torch>` / {py:func}`bt.from_tf(...) <batcher.from_tf>` / {py:func}`bt.from_huggingface(...) <batcher.from_huggingface>` | framework adapters |
| {py:func}`bt.from_duckdb(rel) <batcher.from_duckdb>` | DuckDB relation, or a connection plus a query |
| {py:func}`bt.from_dict(d) <batcher.from_dict>` / {py:func}`bt.from_dicts(rows) <batcher.from_dicts>` / {py:func}`bt.from_records(rows, columns=...) <batcher.from_records>` | pandas/Polars-spelled aliases |
| {py:func}`bt.from_iter(iterable) <batcher.from_iter>` | any Python iterable or generator, one row per item |
| {py:func}`bt.from_any(obj) <batcher.from_any>` | dispatches on the type of whatever you hold |
| {py:func}`bt.concat(frames, how="vertical") <batcher.concat>` | stack datasets (`vertical`/`vertical_relaxed`/`diagonal`/`horizontal`) |
| {py:func}`bt.range(stop) <batcher.range>` / {py:func}`bt.date_range(start, end, interval=...) <batcher.date_range>` | generated ranges |

## Readers

All readers take a local or cloud path and return a Dataset. {py:obj}`bt.read <batcher.read>` infers the format from the path, and the rest are explicit.

| Call | Format |
| --- | --- |
| `bt.read(path, format=None, **opts)` | inferred |
| {py:meth}`bt.read.parquet <batcher.api.io_namespace.reader.Reader.parquet>`, {py:meth}`bt.read.csv <batcher.api.io_namespace.reader.Reader.csv>`, {py:meth}`bt.read.json <batcher.api.io_namespace.reader.Reader.json>` | tabular files |
| {py:meth}`bt.read.table <batcher.api.io_namespace.reader.Reader.table>`, {py:meth}`bt.read.orc <batcher.api.io_namespace.reader.Reader.orc>`, {py:meth}`bt.read.arrow <batcher.api.io_namespace.reader.Reader.arrow>`, {py:meth}`bt.read.avro <batcher.api.io_namespace.reader.Reader.avro>` | tabular files |
| {py:meth}`bt.read.lance <batcher.api.io_namespace.reader.Reader.lance>`, {py:meth}`bt.read.delta <batcher.api.io_namespace.reader.Reader.delta>`, {py:meth}`bt.read.iceberg <batcher.api.io_namespace.reader.Reader.iceberg>`, {py:meth}`bt.read.hudi <batcher.api.io_namespace.reader.Reader.hudi>` | lakehouse tables |
| {py:meth}`bt.read.images <batcher.api.io_namespace.reader.Reader.images>`, {py:meth}`bt.read.audio <batcher.api.io_namespace.reader.Reader.audio>`, {py:meth}`bt.read.video <batcher.api.io_namespace.reader.Reader.video>` | multimodal |
| {py:meth}`bt.read.sql <batcher.api.io_namespace.reader.Reader.sql>`, {py:meth}`bt.read.snowflake <batcher.api.io_namespace.reader.Reader.snowflake>`, {py:meth}`bt.read.bigquery <batcher.api.io_namespace.reader.Reader.bigquery>`, {py:meth}`bt.read.kafka <batcher.api.io_namespace.reader.Reader.kafka>` | external systems |

The pandas and Polars spellings work too, as top-level shorthands for the same lazy readers: {py:func}`bt.read_csv <batcher.read_csv>`, {py:func}`bt.read_parquet <batcher.read_parquet>`, {py:func}`bt.read_json <batcher.read_json>`, {py:func}`bt.read_ndjson <batcher.read_ndjson>`, {py:func}`bt.read_ipc <batcher.read_ipc>`, {py:func}`bt.read_orc <batcher.read_orc>`, {py:func}`bt.read_avro <batcher.read_avro>`, {py:func}`bt.read_excel <batcher.read_excel>`, {py:func}`bt.read_delta <batcher.read_delta>`, {py:func}`bt.read_iceberg <batcher.read_iceberg>`, and {py:func}`bt.read_database <batcher.read_database>`. {py:func}`bt.read_table(name, ...) <batcher.read_table>` constructs any registered connector by name.

## Dataset transformations

Each returns a new lazy Dataset.

| Method | Effect |
| --- | --- |
| `.filter(expr)` | keep rows where the predicate is true |
| {py:meth}`.select(*names, **derived) <batcher.Dataset.select>` | choose or derive the full output |
| {py:meth}`.with_columns(**named) <batcher.Dataset.with_columns>` / {py:meth}`.with_column(name, expr) <batcher.Dataset.with_column>` | add or replace columns |
| `.drop(*names)` | remove columns |
| `.rename({old: new})` | rename columns |
| `.sort(*by, descending=False, nulls_first=False)` | order rows |
| `.limit(n, offset=0)` / `.head(n=5)` | take a prefix |
| `.tail(n=5)` | take a suffix (executes a `count` first) |
| {py:meth}`.gather_every(n, offset=0) <batcher.Dataset.gather_every>` | keep every `n`-th row (downsample) |
| `.reverse()` | reverse the row order |
| {py:meth}`.distinct() <batcher.Dataset.distinct>` | drop duplicate rows |
| `.union(*others, distinct=False)` | concatenate datasets |
| `.intersect(other)` / `.except_(other)` | set operations |
| `.join(other, on=None, left_on=None, right_on=None, how="inner", suffix="_right")` | join (`how`: inner, left, right, full, outer, semi, anti) |
| {py:meth}`.cross_join(other, suffix="_right") <batcher.Dataset.cross_join>` | Cartesian product |
| {py:meth}`.join_asof(other, on, by=None, ...) <batcher.Dataset.join_asof>` | nearest-key temporal join |
| {py:meth}`.window(partition_by=(), order_by=(), functions={...}, frame=None) <batcher.Dataset.window>` | window functions |
| `.group_by(*keys, **named) -> GroupBy` | start an aggregation |
| `.top_k(k, by, descending=True)` | k rows with the largest/smallest `by` |
| {py:meth}`.sample(fraction=None, n=None, seed=None) <batcher.Dataset.sample>` | deterministic seeded row sample |
| `.cast(dtypes)` | cast one column (`"Int64"`) or many (`{col: dtype}`) |
| `.fill_null(value)` | replace nulls (scalar or `{col: value}`) |
| `.fill_null(strategy=...)` | fill from a statistic (`"zero"`/`"mean"`/`"min"`/`"max"`) or carry a neighbour (`"forward"`/`"backward"`, which require `order_by`) |
| `.drop_nulls(subset=None)` | drop rows with nulls (optionally in `subset`) |
| `.map_batches(fn, ...)` | run a Python callable over whole Arrow batches |

### Reshaping

These change a dataset's shape rather than its contents, turning rows into columns or the
reverse:

| Method | Effect |
| --- | --- |
| {py:meth}`.explode(column, alias=None) <batcher.Dataset.explode>` | one row per element of a list column |
| {py:meth}`.with_row_index(name="index", offset=0) <batcher.Dataset.with_row_index>` | prepend a sequential row-index column (Polars) |
| {py:meth}`.with_random(name="random", seed=0, normal=False) <batcher.Dataset.with_random>` | add a reproducible seeded random column (uniform or standard normal) |
| {py:meth}`.unnest(*columns) <batcher.Dataset.unnest>` | lift struct fields into top-level columns |
| {py:meth}`.pivot(index=[...], on=, values=, aggregate="sum") <batcher.Dataset.pivot>` | long → wide |
| {py:meth}`.unpivot(on=[...], index=[...], ...) <batcher.Dataset.unpivot>` | wide → long |
| {py:meth}`.value_counts(column, name="count") <batcher.Dataset.value_counts>` | frequency of each distinct value |

## Dataset terminal operations

Each of these executes the plan and returns a result or writes it out:

| Method | Returns |
| --- | --- |
| {py:meth}`.collect(distributed=False, num_workers=None, spill=False, num_partitions=16, adaptive=False, transport="disk") <batcher.Dataset.collect>` | pyarrow Table |
| {py:meth}`.to_pydict() <batcher.Dataset.to_pydict>` | `dict[str, list]` |
| {py:meth}`.to_pylist() <batcher.Dataset.to_pylist>` | `list[dict]` |
| `.count()` | row count (`int`) |
| {py:meth}`.iter_batches(batch_size=None) <batcher.Dataset.iter_batches>` | iterator of RecordBatch |
| `.explain()` | optimized plan as text |
| {py:meth}`.show(limit=10) <batcher.Dataset.show>` | prints a preview |
| {py:obj}`.write(path, fmt=None, partition_by=None, distributed=False, num_workers=None, **kw) <batcher.Dataset.write>` | WriteManifest |
| `.write.parquet(path, compression="zstd", **kw)` | writes Parquet |
| `.write.csv(path, **kw)` / `.write.json(path, **kw)` | writes CSV / JSON |
| {py:meth}`.to_arrow() <batcher.Dataset.to_arrow>` | pyarrow Table (alias of `.collect()`) |
| {py:meth}`.to_pandas() <batcher.Dataset.to_pandas>` / {py:meth}`.to_polars() <batcher.Dataset.to_polars>` | a pandas / Polars DataFrame |
| {py:meth}`.to_numpy(columns=None) <batcher.Dataset.to_numpy>` | a `{column: numpy.ndarray}` dict (tensor columns → `(n, *shape)`) |
| {py:meth}`.to_jax(columns=None) <batcher.Dataset.to_jax>` | a `{column: jax.Array}` dict, the JAX counterpart of {py:meth}`to_numpy <batcher.Dataset.to_numpy>` |
| `.to_torch(columns=None, batch_size=None)` / `.to_tf(...)` | a Torch / TensorFlow dataset |
| `.to_torch_dataloader(...)` | a `torch.utils.data.DataLoader` |
| {py:meth}`.to_ray_dataset() <batcher.Dataset.to_ray_dataset>` | a `ray.data.Dataset`, for a Ray Train / Tune / Serve stage |

### Introspection

These compute (or read) a small result and so are eager.

| Member | Returns |
| --- | --- |
| `.columns` | current schema names (property) |
| `.schema()` / `.dtypes()` | the Arrow schema / its column types |
| {py:meth}`.is_empty() <batcher.Dataset.is_empty>` | whether the dataset has zero rows |
| {py:obj}`.is_streaming() <batcher.Dataset.is_streaming>` | whether the source is unbounded |
| {py:meth}`.describe(percentiles=(.25,.5,.75)) <batcher.Dataset.describe>` | summary statistics per column |
| {py:meth}`.null_count() <batcher.Dataset.null_count>` | null count per column |
| {py:meth}`.corr_matrix(columns=None) <batcher.Dataset.corr_matrix>` | pairwise Pearson correlation matrix over numeric columns (one scan) |
| {py:meth}`.cov_matrix(columns=None) <batcher.Dataset.cov_matrix>` | pairwise sample covariance matrix over numeric columns (PCA/whitening input) |
| `.approx_quantile(column, q)` | a sketch-based quantile estimate |
| {py:meth}`.stats() <batcher.Dataset.stats>` | the last run's measured `RunStats` |
| `__arrow_c_stream__()` | Arrow PyCapsule export, so `pl.DataFrame(ds)`, `duckdb.sql("... FROM ds")`, and `pa.table(ds)` consume a `Dataset` directly, lazily and zero-copy |

```python
out = ds.filter(bt.col("price") >= 20).sort("price", descending=True)
print(out.to_pydict())
# {'category': ['a', 'b'], 'price': [30.0, 20.0]}
```

## GroupBy

`group_by(*keys)` returns a {py:class}`GroupBy <batcher.GroupBy>`. Finalize it with `.agg(**named_aggs)`, where each keyword is the output column name. `group_by()` with no keys aggregates the whole dataset.

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

These build the {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` values every column operation takes:

| Call | Meaning |
| --- | --- |
| {py:func}`bt.col(name) <batcher.col>` | reference a column |
| {py:func}`bt.lit(value) <batcher.lit>` | a constant |
| {py:func}`bt.when(c).then(v)...otherwise(d) <batcher.when>` | SQL CASE |
| {py:func}`bt.coalesce(*exprs) <batcher.coalesce>` | first non-null per row (also the SQL `IFNULL` case) |
| {py:func}`bt.nullif(a, b) <batcher.nullif>` | null when `a == b` |
| {py:func}`bt.greatest(*exprs) <batcher.greatest>` / {py:func}`bt.least(*exprs) <batcher.least>` | row-wise extreme |
| {py:func}`bt.array(*exprs) <batcher.array>` | build a list column |
| {py:func}`bt.atan2(y, x) <batcher.atan2>` | two-argument arctangent |

## Column selectors

Each stands for every column matching a predicate, and expands against the input
schema wherever a column is expected. They produce a {py:class}`Selector <batcher.plan.expr_ir.selectors.Selector>`.

| Call | Selects |
| --- | --- |
| {py:func}`bt.all() <batcher.all>` | every column |
| {py:func}`bt.numeric() <batcher.numeric>` / {py:func}`bt.integer() <batcher.integer>` / {py:func}`bt.floating() <batcher.floating>` | numeric / integer / float columns |
| {py:func}`bt.string() <batcher.string>` / {py:func}`bt.boolean() <batcher.boolean>` / {py:func}`bt.temporal() <batcher.temporal>` | string / boolean / date-time columns |
| {py:func}`bt.exclude(*names) <batcher.exclude>` | every column except the named ones |

{py:func}`bt.by_dtype <batcher.by_dtype>`, {py:func}`bt.matches <batcher.matches>`, {py:func}`bt.starts_with <batcher.starts_with>`, {py:func}`bt.ends_with <batcher.ends_with>`, and {py:func}`bt.contains <batcher.contains>` select by dtype or by name pattern. See the {doc}`complete reference <complete>` for their signatures.

## Scalar, aggregate, and window functions

These are the top-level function forms. Rows marked `(aggregate)` belong inside `.agg(...)`, and rows marked `(window)` need a `.over(...)` binding.

| Call | Meaning |
| --- | --- |
| {py:func}`bt.count() <batcher.count>` | COUNT(*) aggregate |
| {py:func}`bt.iff(condition, if_true, if_false) <batcher.iff>` | `if_true` where `condition` is true, else `if_false` (DuckDB `IFF`) |
| {py:func}`bt.nanvl(value, fallback) <batcher.nanvl>` | `value` unless it is NaN, then `fallback` (Spark `nanvl`) |
| {py:func}`bt.next_after(value, toward) <batcher.next_after>` | the adjacent representable float, one ULP toward `toward` (DuckDB `nextafter`) |
| `bt.concat(*exprs)` | concatenate values into one string |
| {py:func}`bt.concat_ws(separator, *exprs) <batcher.concat_ws>` | concatenate values with `separator` between them |
| {py:func}`bt.format_string(format, *exprs) <batcher.format_string>` | interpolate values into a `{}` template (Polars `format`) |
| {py:func}`bt.mask(e, show_first=0, show_last=0, char="X") <batcher.mask>` | redact a string, optionally revealing its ends |
| {py:func}`bt.hmac_sha256(e, key) <batcher.hmac_sha256>` | keyed, irreversible pseudonym that still joins |
| {py:func}`bt.aes_encrypt(e, key) <batcher.aes_encrypt>` / {py:func}`bt.aes_decrypt(e, key) <batcher.aes_decrypt>` | deterministic AES-256-GCM-SIV column encryption |
| {py:func}`bt.log(base, value) <batcher.log>` | logarithm of `value` in the given `base` (→ Float64) |
| {py:func}`bt.gcd(a, b) <batcher.gcd>` / {py:func}`bt.lcm(a, b) <batcher.lcm>` | greatest common divisor / least common multiple |
| {py:func}`bt.hypot(a, b) <batcher.hypot>` | Euclidean norm `sqrt(a² + b²)` |
| {py:func}`bt.great_circle_distance(lat1, lon1, lat2, lon2, unit="km") <batcher.great_circle_distance>` | haversine distance between two lat/lon points, in `km`/`m`/`mi`/`nm` |
| {py:func}`bt.width_bucket(value, low, high, count) <batcher.width_bucket>` | histogram bucket index over `[low, high]` |
| {py:func}`bt.struct(**fields) <batcher.struct>` / {py:func}`bt.named_struct(name, value, ...) <batcher.named_struct>` | build a struct column |
| {py:func}`bt.sequence(start, stop, step=1) <batcher.sequence>` | per-row integer list `[start..stop]` inclusive (DuckDB `generate_series`) |
| {py:func}`bt.element() <batcher.element>` | the current element inside `list.transform` / `list.filter` (Polars) |
| {py:func}`bt.sum_horizontal(*exprs) <batcher.sum_horizontal>` / {py:func}`bt.mean_horizontal(*exprs) <batcher.mean_horizontal>` | row-wise sum / mean across columns, ignoring nulls (Polars) |
| {py:func}`bt.min_horizontal(*exprs) <batcher.min_horizontal>` / {py:func}`bt.max_horizontal(*exprs) <batcher.max_horizontal>` | row-wise min / max across columns, ignoring nulls (Polars) |
| {py:func}`bt.all_horizontal(*exprs) <batcher.all_horizontal>` / {py:func}`bt.any_horizontal(*exprs) <batcher.any_horizontal>` | row-wise boolean AND / OR across columns (Polars) |
| {py:func}`bt.hash_rows(*exprs, seed=0) <batcher.hash_rows>` | deterministic 64-bit row digest (also `expr.hash(seed=0)`) |
| {py:func}`bt.count_if(condition) <batcher.count_if>` | count rows where `condition` is true (aggregate) |
| {py:func}`bt.sum(x) <batcher.sum>` / {py:func}`bt.mean(x) <batcher.mean>` / {py:func}`bt.min(x) <batcher.min>` / {py:func}`bt.max(x) <batcher.max>` / {py:func}`bt.median(x) <batcher.median>` / {py:func}`bt.std(x) <batcher.std>` / {py:func}`bt.var(x) <batcher.var>` / {py:func}`bt.n_unique(x) <batcher.n_unique>` | the SQL-style column-aggregate shorthands for `col(x).<agg>()` |
| {py:func}`bt.product(x) <batcher.product>` / {py:func}`bt.mode(x) <batcher.mode>` / {py:func}`bt.skewness(x) <batcher.skewness>` / {py:func}`bt.kurtosis(x) <batcher.kurtosis>` | product / most-frequent value / 3rd / 4th standardized moment (aggregate) |
| {py:func}`bt.bool_and(x) <batcher.bool_and>` / {py:func}`bt.bool_or(x) <batcher.bool_or>` | boolean AND / OR reduction of a group (aggregate) |
| {py:func}`bt.bit_and(x) <batcher.bit_and>` / {py:func}`bt.bit_or(x) <batcher.bit_or>` / {py:func}`bt.bit_xor(x) <batcher.bit_xor>` | bitwise AND / OR / XOR reduction of integers (aggregate) |
| {py:func}`bt.array_agg(x) <batcher.array_agg>` | collect each group's values into a list (aggregate) |
| {py:func}`bt.quantile(x, q) <batcher.quantile>` / {py:func}`bt.approx_quantile(x, q) <batcher.approx_quantile>` / {py:func}`bt.approx_median(x) <batcher.approx_median>` | exact / sketch-based quantile / median (aggregate) |
| {py:func}`bt.approx_n_unique(x) <batcher.approx_n_unique>` / {py:func}`bt.histogram(x) <batcher.histogram>` | HyperLogLog distinct count / value→count map (aggregate) |
| {py:func}`bt.corr(x, y) <batcher.corr>` | Pearson correlation (aggregate) |
| {py:func}`bt.covar_pop(x, y) <batcher.covar_pop>` / {py:func}`bt.covar_samp(x, y) <batcher.covar_samp>` | population / sample covariance (aggregate) |
| {py:func}`bt.regr_slope(y, x) <batcher.regr_slope>` / {py:func}`bt.regr_intercept(y, x) <batcher.regr_intercept>` / {py:func}`bt.regr_r2(y, x) <batcher.regr_r2>` | least-squares slope / intercept / R² of `y` on `x` (aggregate) |
| {py:func}`bt.regr_count(y, x) <batcher.regr_count>` / {py:func}`bt.regr_avgx(y, x) <batcher.regr_avgx>` / {py:func}`bt.regr_avgy(y, x) <batcher.regr_avgy>` | paired sample size and per-axis means (aggregate) |
| {py:func}`bt.regr_sxx(y, x) <batcher.regr_sxx>` / {py:func}`bt.regr_syy(y, x) <batcher.regr_syy>` / {py:func}`bt.regr_sxy(y, x) <batcher.regr_sxy>` | regression sums of squares / cross-products (aggregate) |
| {py:func}`bt.var_pop(x) <batcher.var_pop>` / {py:func}`bt.stddev_pop(x) <batcher.stddev_pop>` | population variance / standard deviation (aggregate; `var`/`std` are the sample forms) |
| {py:func}`bt.geometric_mean(x) <batcher.geometric_mean>` / {py:func}`bt.harmonic_mean(x) <batcher.harmonic_mean>` / {py:func}`bt.rms(x) <batcher.rms>` | geometric / harmonic / quadratic (root-mean-square) mean (aggregate) |
| {py:func}`bt.cv(x) <batcher.cv>` / {py:func}`bt.sem(x) <batcher.sem>` / {py:func}`bt.midrange(x) <batcher.midrange>` | coefficient of variation / standard error of the mean / midrange (aggregate) |
| {py:func}`bt.weighted_mean(value, weight) <batcher.weighted_mean>` | mean of `value` weighted by `weight` (aggregate) |
| {py:func}`bt.lag(expr, n=1) <batcher.lag>` / {py:func}`bt.lead(expr, n=1) <batcher.lead>` | the value `n` rows before / after the current row (window) |
| {py:func}`bt.first_value(expr) <batcher.first_value>` / {py:func}`bt.last_value(expr) <batcher.last_value>` | the first / last value of the ordered partition (window) |
| {py:func}`bt.nth_value(expr, n) <batcher.nth_value>` | the `n`-th value of the ordered partition (window) |
| {py:func}`bt.current_timestamp() <batcher.current_timestamp>` | current timestamp, bound at plan-build time |
| {py:func}`bt.current_date() <batcher.current_date>` | today's date, bound at plan-build time |
| {py:func}`bt.date_part(part, expr) <batcher.date_part>` | extract a calendar field such as `year`, `month`, or `dow` |
| {py:func}`bt.date_add(expr, days) <batcher.date_add>` | add a whole number of `days` to a date/time column (Spark {py:func}`date_add <batcher.date_add>`) |
| {py:func}`bt.date_sub(expr, days) <batcher.date_sub>` | subtract a whole number of `days` from a date/time column (Spark {py:func}`date_sub <batcher.date_sub>`) |
| {py:func}`bt.make_date(year, month, day) <batcher.make_date>` | build a Date from integer components; an impossible date is null |
| {py:func}`bt.make_timestamp(year, month, day, hour=0, minute=0, second=0) <batcher.make_timestamp>` | build a Timestamp from components |
| {py:func}`bt.from_epoch(expr, unit="s") <batcher.from_epoch>` | read an integer epoch column as a Timestamp at a stated unit (`s`/`ms`/`us`/`ns`) |
| {py:func}`bt.from_unix_date(expr) <batcher.from_unix_date>` | read an integer column of days since 1970-01-01 as a Date |

## Top-level helpers

These sit outside the `Dataset` and `Expr` surfaces:

| Call | Returns |
| --- | --- |
| `bt.date_range(start, end, *, interval_days=1, name="date")` | the date-dimension generator: a one-column Dataset of dates, inclusive, as ISO `YYYY-MM-DD` |
| {py:func}`bt.compact(path, *, target_size_mb=128.0, num_files=None, by=None, format=None, **opts) <batcher.compact>` | rewrite many small files at `path` into fewer larger ones in place; returns a {py:class}`WriteManifest <batcher.io.WriteManifest>` |
| {py:func}`bt.engine_version() <batcher.engine_version>` | the version reported by the compiled Rust engine (`str`) |
| {py:func}`bt.start_ui(*, port=4040, host="127.0.0.1", open_browser=False) <batcher.start_ui>` | start the web dashboard (queries, plan DAG, per-operator timings, live logs); returns its URL |
| {py:func}`bt.stop_ui() <batcher.stop_ui>` | stop the web dashboard; safe to call when none is running |
| {py:func}`bt.ui_url() <batcher.ui_url>` | the running dashboard's URL, or `None` |

## Expression methods

- Operators: `+ - * / % **`; `== != > >= < <=`; `& | ~`
- Types and nulls: `.cast("Int64")`, `.try_cast("Int64")`, `.is_null()`,
  `.is_not_null()`, `.is_in([...])`, `.between(low, high)`, `.fill_null(value)`,
  {py:meth}`.fill_nan(value) <batcher.plan.expr_ir.core.Expr.fill_nan>`, {py:meth}`.eq_missing(other) <batcher.plan.expr_ir.core.Expr.eq_missing>`, {py:meth}`.is_nan() <batcher.plan.expr_ir.core.Expr.is_nan>`, {py:meth}`.is_not_nan() <batcher.plan.expr_ir.core.Expr.is_not_nan>`,
  `.is_finite()`, `.is_infinite()`, `.clip(lower, upper)`, `.alias(name)`
- Binning and gap-filling: `.cut(breaks, labels=None, left_closed=False)`,
  {py:meth}`.forward_fill() <batcher.plan.expr_ir.core.Expr.forward_fill>` and {py:meth}`.backward_fill() <batcher.plan.expr_ir.core.Expr.backward_fill>`. The two fills are window functions, so bind them with {py:meth}`.over(order_by=[...]) <batcher.AggExpr.over>`. An order is required.
- Math: `.abs()`, `.round(digits)`, `.pow(e)`, `.sqrt()`, `.floor()`, `.ceil()`,
  `.ln()`, `.log10()`, `.log2()`, `.exp()`, `.sin()`, `.cos()`, `.tan()`, `.asin()`,
  `.acos()`, `.atan()`, `.sinh()`, `.cosh()`, `.tanh()`, `.cot()`, `.sign()`,
  `.trunc()`, `.cbrt()`, `.degrees()`, `.radians()`, `.factorial()`
- Bitwise (integers): {py:meth}`.bitwise_and(o) <batcher.plan.expr_ir.core.Expr.bitwise_and>`, {py:meth}`.bitwise_or(o) <batcher.plan.expr_ir.core.Expr.bitwise_or>`, {py:meth}`.bitwise_xor(o) <batcher.plan.expr_ir.core.Expr.bitwise_xor>`,
  {py:meth}`.bitwise_left_shift(o) <batcher.plan.expr_ir.core.Expr.bitwise_left_shift>`, {py:meth}`.bitwise_right_shift(o) <batcher.plan.expr_ir.core.Expr.bitwise_right_shift>`, {py:meth}`.bit_count() <batcher.plan.expr_ir.core.Expr.bit_count>`
- Aggregates (inside `.agg`): `.sum()`, `.min()`, `.max()`, `.mean()`, `.var()`,
  `.std()`, `.median()`, `.quantile(q)`, `.skewness()`, `.kurtosis()`, `.count()`,
  `.n_unique()` / `.count_distinct()`, `.mode()`, `.first()`, `.last()`,
  `.arg_min()`, `.arg_max()`, `.bool_and()`, `.bool_or()`,
  `.bit_and()` / `.bit_or()` / `.bit_xor()`, `.histogram()`, `.array_agg()`
- Approximate aggregates, sketch-backed and mergeable so they scale: `.approx_n_unique()` /
  `.approx_count_distinct()` (HyperLogLog), `.approx_quantile(q)` / `.approx_median()` (KLL)
- Cumulative & window analytics (bind with `.over(...)`): `.cum_sum()` / `.cum_min()` /
  {py:meth}`.cum_max() <batcher.plan.expr_ir.core.Expr.cum_max>` / {py:meth}`.cum_count() <batcher.plan.expr_ir.core.Expr.cum_count>`, {py:meth}`.rolling_sum(k) <batcher.plan.expr_ir.core.Expr.rolling_sum>` / {py:meth}`.rolling_mean(k) <batcher.plan.expr_ir.core.Expr.rolling_mean>` /
  `.rolling_min(k)` / `.rolling_max(k)` / `.rolling_count(k)`, `.diff(n=1)`,
  `.pct_change(n=1)`, `.shift(n)`, `.rank(method="min")`, `.is_duplicated()` /
  {py:meth}`.is_unique() <batcher.plan.expr_ir.core.Expr.is_unique>`
- Full list, plus the {py:class}`.str <batcher.plan.expr_ir.namespaces.strings._StrNamespace>` / {py:class}`.dt <batcher.plan.expr_ir.namespaces.temporal._DtNamespace>` / {py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>` / {py:class}`.struct <batcher.plan.expr_ir.namespaces.collections._StructNamespace>` / {py:class}`.json <batcher.plan.expr_ir.namespaces.collections._JsonNamespace>` / {py:class}`.map <batcher.plan.expr_ir.namespaces.collections._MapNamespace>` /
  {py:class}`.image <batcher.plan.expr_ir.image._ImageNamespace>` / {py:class}`.audio <batcher.plan.expr_ir.audio._AudioNamespace>` / {py:class}`.video <batcher.plan.expr_ir.video._VideoNamespace>` accessors: the {doc}`expressions API page </api/relational/expressions>`.

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
| {py:class}`.struct <batcher.plan.expr_ir.namespaces.collections._StructNamespace>` | {py:meth}`field(name) <batcher.plan.expr_ir.namespaces.collections._StructNamespace.field>` |
| `.map` | Arrow `Map` columns: `keys()`, `values()`, `get(key)` |
| {py:class}`.json <batcher.plan.expr_ir.namespaces.collections._JsonNamespace>` | {py:meth}`extract_string(path) <batcher.plan.expr_ir.namespaces.collections._JsonNamespace.extract_string>` |
| `.image` | `decode()`, `to_tensor(width, height)`, `to_tensor_f32(width, height, mean=, std=, channels_first=)`, `center_crop(width, height)`, `to_grayscale(width, height)`, `resize(width, height)` |
| `.audio` | native WAV/FLAC decode: `decode()` (metadata struct), `to_waveform()` (mono `List<Float32>`), `resample(rate)`, `trim_silence(threshold_db=-40)` (drop the leading and trailing quiet; a clip silent throughout trims to an empty list, which is how a silent recording is filtered out), `peak_normalize()` (scale so the loudest sample sits at full scale, the level-matching step before batching clips from different sources), `zero_crossing_rate()` (the voiced/unvoiced descriptor, in `[0, 1]`), `mel_spectrogram(rate, n_fft=, hop_length=, n_mels=)` (speech-model mel power spectrogram, torchaudio-matching), `mfcc(rate, n_fft=, hop_length=, n_mels=, n_mfcc=)` (MFCC feature, torchaudio-matching) |
| `.video` | native FFmpeg decode: `decode()` returns a metadata struct, and needs the `video` engine build feature |

## SQL

{py:func}`bt.sql(query, table_name=ds_or_table, ...) <batcher.sql>` returns a Dataset. Each table named in the query is bound by a keyword argument. The {doc}`SQL page </api/relational/sql>` lists the supported clauses and features in full.

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
| {py:meth}`ds.ml.map_batches(fn, ...) <batcher.api.dataset.ml.DatasetML.map_batches>` | arbitrary batch transform |
| {py:meth}`ds.ml.infer(model, ...) <batcher.api.dataset.ml.DatasetML.infer>` | batched inference |
| {py:meth}`ds.ml.embed(model, ...) <batcher.api.dataset.ml.DatasetML.embed>` | batched embeddings |
| {py:meth}`ds.ml.generate(engine, ...) <batcher.api.dataset.ml.DatasetML.generate>` | offline LLM text generation |
| {py:meth}`ds.ml.extract(engine, schema=...) <batcher.api.dataset.ml.DatasetML.extract>` | LLM → **typed** columns (AI-powered ETL) |
| {py:meth}`ds.ml.classify(engine, labels=[...]) <batcher.api.dataset.ml.DatasetML.classify>` | zero-shot labelling, domain pinned to `labels` |
| {py:meth}`ds.ml.near_duplicates(col) <batcher.api.dataset.ml.DatasetML.near_duplicates>` / {py:meth}`drop_near_duplicates(col) <batcher.api.dataset.ml.DatasetML.drop_near_duplicates>` | MinHash+LSH fuzzy dedup |
| {py:meth}`ds.ml.similarity_join(other, left_on=...) <batcher.api.dataset.ml.DatasetML.similarity_join>` | join two datasets on embedding similarity |

## Preprocessors

`batcher.ml.preprocessors` holds the scikit-learn-style `fit`/`transform` estimators. Fit on the training split and transform both. {py:class}`Chain <batcher.ml.preprocessors.Chain>` composes them into one pipeline.

| Class | Learns |
| --- | --- |
| {py:class}`StandardScaler <batcher.ml.preprocessors.StandardScaler>` / {py:class}`MinMaxScaler <batcher.ml.preprocessors.MinMaxScaler>` / {py:class}`MaxAbsScaler <batcher.ml.preprocessors.MaxAbsScaler>` / {py:class}`RobustScaler <batcher.ml.preprocessors.RobustScaler>` | per-column scaling statistics |
| {py:class}`Normalizer <batcher.ml.preprocessors.Normalizer>` | stateless per-row vector normalization |
| {py:class}`OneHotEncoder <batcher.ml.preprocessors.OneHotEncoder>` / {py:class}`MultiHotEncoder <batcher.ml.preprocessors.MultiHotEncoder>` / {py:class}`LabelEncoder <batcher.ml.preprocessors.LabelEncoder>` / {py:class}`OrdinalEncoder <batcher.ml.preprocessors.OrdinalEncoder>` | the category vocabulary |
| {py:class}`KBinsDiscretizer <batcher.ml.preprocessors.KBinsDiscretizer>` | bin edges (quantile or uniform) |
| {py:class}`SimpleImputer <batcher.ml.preprocessors.SimpleImputer>` | the fill statistic (mean/median/most-frequent/constant) |
| {py:class}`Tokenizer <batcher.ml.preprocessors.Tokenizer>` / {py:class}`Concatenator <batcher.ml.preprocessors.Concatenator>` | stateless text split / feature-vector assembly |
| `Chain` | each step, fit on the previous step's output |

See the {doc}`preprocessors guide </ml/preparing/preprocessors/index>` for the workflow and the
{doc}`ML API page </api/models/ml>` for the per-class reference.

## Configuration

```python
from batcher import Config, set_config, config_context
```

{py:class}`Config() <batcher.Config>` is a frozen dataclass of sections (`execution`, `memory`, `flow_control`,
`optimizer`, `pid`, `metadata`). Derive a modified Config and apply it process-wide
with {py:func}`set_config(...) <batcher.set_config>` or temporarily with {py:func}`config_context(...) <batcher.config_context>`. {py:meth}`Config.from_env <batcher.Config.from_env>`
and {py:meth}`Config.from_file <batcher.Config.from_file>` overlay `BATCHER_*` environment variables and a JSON file.
See the configuration page for the full pattern.

## See also

- {doc}`/api/relational/dataset`: every `Dataset` method, with its arguments and its return type.
- {doc}`/api/relational/expressions` and {doc}`/api/relational/expression-accessors`: the column language and the
  `.str` / `.dt` / `.list` / `.struct` / `.json` namespaces.
- {doc}`/api/relational/functions`: the free functions, grouped by family.
- {doc}`/api/relational/io`: readers, writers, save modes, and the format-specific options.
- {doc}`../user-guide/index`: the task-oriented guides behind these signatures.
- {doc}`../getting-started/quickstart`: the same surface as a five-minute walkthrough.
- {doc}`../cookbook/index`: 100 runnable recipes, when the signature is not enough.
