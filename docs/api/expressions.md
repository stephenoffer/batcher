# Expressions API

The expression API describes column computations that lower to the Rust data plane
and run vectorized over Arrow batches. This page is the reference for the
constructors, operators, methods, and accessor namespaces. For a guided tour with
runnable examples, see the [expressions user guide](../user-guide/expressions.md).

Blocks on this page share one namespace and run in order.

```python
import batcher as bt

ds = bt.from_pydict({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})
```

## Constructors

| Call | Meaning |
| --- | --- |
| `bt.col(name)` | reference an input column |
| `bt.lit(value)` | a constant value |
| `bt.when(c).then(v)...otherwise(d)` | SQL CASE |
| `bt.coalesce(*exprs)` | first non-null argument |
| `bt.nullif(a, b)` | null when `a == b` |
| `bt.greatest(*exprs)` / `bt.least(*exprs)` | row-wise max / min across columns |
| `bt.array(*exprs)` | build a list column from elements |
| `bt.atan2(y, x)` | two-argument arctangent |
| `bt.count()` | COUNT(*) aggregate |
| `bt.hash_rows(*exprs, seed=0)` | deterministic 64-bit row digest (also `expr.hash(seed=0)`) |

```python
out = ds.select(
    label=bt.when(bt.col("a") > 1).then(bt.lit("hi")).otherwise(bt.lit("lo")),
    best=bt.greatest(bt.col("a"), bt.lit(2)),
)
print(out.to_pydict())
# {'label': ['lo', 'hi', 'hi'], 'best': [2, 2, 3]}
```

`hash_rows` digests the row's **values**, typed: an integer from its bits, a float from
its canonicalized IEEE bits (so `-0.0` and `0.0` agree, and every NaN agrees), a string
from its UTF-8. It is order-sensitive, treats null as a positional value, and is stable
across partitions, runs, machines and versions — which is what lets it key a
reproducible split, a surrogate key, or a hash bucket. It is 3–10x faster than hashing
`cast(col, "string")`, and unlike that idiom it does not depend on how a float prints.

```python
keys = bt.from_pydict({"a": [1, 1, 2]})
print(keys.select(bucket=bt.col("a").hash().abs() % 10).to_pydict())
# {'bucket': [9, 9, 5]}
```

## Horizontal (row-wise) functions

These fold *across* columns within each row (the counterpart to aggregates, which
fold *down* a column). They mirror the Polars `*_horizontal` family.

| Call | Meaning |
| --- | --- |
| `bt.sum_horizontal(*exprs)` | row-wise sum, nulls treated as 0 |
| `bt.count_horizontal(*exprs)` | row-wise count of non-null values |
| `bt.product_horizontal(*exprs)` | row-wise product, nulls treated as 1 |
| `bt.reduce_horizontal(fn, *exprs)` / `bt.fold_horizontal(acc, fn, *exprs)` | fold columns row-wise with a binary `Expr` combiner (no seed / with seed) |
| `bt.mean_horizontal(*exprs)` | row-wise mean, ignoring nulls |
| `bt.min_horizontal(*exprs)` / `bt.max_horizontal(*exprs)` | row-wise min / max, ignoring nulls (the Polars-named `least` / `greatest`) |
| `bt.all_horizontal(*exprs)` / `bt.any_horizontal(*exprs)` | row-wise boolean AND / OR across predicate columns |

```python
flags = bt.from_pydict({"a": [1, None, 3], "b": [10, 20, None]})
out = flags.select(
    total=bt.sum_horizontal(bt.col("a"), bt.col("b")),
    lo=bt.min_horizontal(bt.col("a"), bt.col("b")),
    both_pos=bt.all_horizontal(bt.col("a") > 0, bt.col("b") > 0),
)
print(out.to_pydict())
# {'total': [11, 20, 3], 'lo': [1, 20, 3], 'both_pos': [True, None, None]}
```

## Operators

| Group | Operators |
| --- | --- |
| Arithmetic | `+` `-` `*` `/` `%` `**` (reflected forms work, e.g. `2 * bt.col("a")`) |
| Comparison | `==` `!=` `<` `<=` `>` `>=` |
| Boolean | `&` (and), `\|` (or), `~` (not) |

Parenthesize each side of a boolean combination, because `&` and `|` bind tighter
than comparison.

```python
out = ds.select(both=((bt.col("a") > 1) & (bt.col("b") < 30)))
print(out.to_pydict())
# {'both': [False, True, False]}
```

## Null handling

| Method | Description |
| --- | --- |
| `.is_null()` | true where null |
| `.is_not_null()` | true where not null |
| `.is_nan()` / `.is_not_nan()` | true where the float value is (not) NaN — distinct from null |
| `.is_finite()` / `.is_infinite()` | true where the float value is finite / ±infinity |
| `.fill_null(value)` | replace nulls with a value |
| `.forward_fill()` / `.backward_fill()` | carry the nearest non-null value along an ordered window (`.over(order_by=…)` required) |
| `.cut(breaks, labels=None, left_closed=False)` | bin a numeric column into labelled intervals |

```python
nulls = bt.from_pydict({"x": [1, None, 3]})
out = nulls.select(filled=bt.col("x").fill_null(0), missing=bt.col("x").is_null())
print(out.to_pydict())
# {'filled': [1, 0, 3], 'missing': [False, True, False]}
```

## Type, membership, and range

| Method | Description |
| --- | --- |
| `.cast(type)` | cast to an Arrow type named as a string (`"int64"`, `"float64"`, `"utf8"`, `"bool"`) |
| `.is_in([...])` | membership test |
| `.between(low, high, closed="both")` | range test; `closed` = `"both"`/`"left"`/`"right"`/`"none"` sets which bounds are inclusive |

```python
out = ds.select(
    as_float=bt.col("a").cast("float64"),
    in_set=bt.col("a").is_in([1, 3]),
    in_range=bt.col("b").between(15.0, 30.0),
)
print(out.to_pydict())
# {'as_float': [1.0, 2.0, 3.0], 'in_set': [True, False, True], 'in_range': [False, True, True]}
```

## Math methods

`.abs()`, `.round(digits)`, `.pow(e)`, `.sqrt()`, `.floor()`, `.ceil()`, `.ln()`,
`.log10()`, `.log2()`, `.exp()`, `.sin()`, `.cos()`, `.tan()`, `.asin()`, `.acos()`,
`.atan()`, `.sinh()`, `.cosh()`, `.tanh()`, `.cot()`, `.sign()`, `.trunc()`,
`.cbrt()`, `.degrees()`, `.radians()`, `.factorial()`, `.square()` (i.e. `x*x`),
`.log1p()` / `.expm1()` (accurate near zero), and the inverse-hyperbolics
`.asinh()` / `.acosh()` / `.atanh()` (→ Float64). Integer bitwise
ops (distinct from the boolean `&`/`|`): `.bitwise_and(o)`, `.bitwise_or(o)`,
`.bitwise_xor(o)`, `.bitwise_left_shift(o)`, `.bitwise_right_shift(o)`, and
`.bit_count()` (the number of set bits, i.e. population count → Int64).

```python
out = ds.select(root=bt.col("b").sqrt(), third=(bt.col("b") / 3).round(2))
print(out.to_pydict())
# {'root': [3.1622776601683795, 4.47213595499958, 5.477225575051661], 'third': [3.33, 6.67, 10.0]}
```

## Other core methods

| Method | Description |
| --- | --- |
| `.alias(name)` | bind an output name to a derived expression, for positional `select` |
| `.clip(lower=None, upper=None)` | clamp each value into `[lower, upper]` (either bound optional) |
| `.eq_missing(other)` | null-safe equality (SQL `IS NOT DISTINCT FROM`): two nulls compare equal, null vs non-null is false (never null) |
| `.try_cast(type)` | like `.cast` but unconvertible values become NULL instead of erroring (DuckDB `TRY_CAST`) — the safe-ingest spelling |
| `.approx_count_distinct()` | approximate `COUNT(DISTINCT)` via a HyperLogLog sketch (~2% error) |

## Aggregation methods

Used inside `group_by(...).agg(...)`: `.sum()`, `.min()`, `.max()`, `.mean()`,
`.var()`, `.std()`, `.median()`, `.quantile(q)`, `.skewness()` / `.kurtosis()`
(third / fourth standardized moment of each group; DuckDB `skewness` / `kurtosis`),
`.histogram()` (a
`Map<value, count>` of each group's values, DuckDB `histogram`), `.count()`, `.n_unique()`
(aliased `.count_distinct()`), `.mode()`, `.bool_and()`, `.bool_or()`,
`.bit_and()` / `.bit_or()` / `.bit_xor()` (bitwise reduction of the non-null
`Int64` values in each group), `.array_agg()` (collect each group's values into a
`List`; SQL `array_agg` /
Spark `collect_list`), `.arg_min(by=…)` / `.arg_max(by=…)` (the value at the
row with the extreme `by` key), and `.first(order_by=…)` / `.last(order_by=…)`
(the value at the first/last row in `order_by` order — a required argument, since
an arrival-order first/last would not be partition-independent). `bt.count()` is
the top-level `COUNT(*)`. Each of these returns an `AggExpr`, the aggregate type
that `group_by(...).agg(...)` and `.over(...)` consume; you rarely name it directly.

For heavy skew, the bounded-memory **approximate** variants keep one fixed-size
sketch per group instead of every value, so a hot key cannot OOM: `.approx_n_unique()`
(HLL, ~2% error) and `.approx_quantile(q)` / `.approx_median()` (DDSketch). They are
mergeable, so results are identical single-node and distributed.

```python
out = ds.group_by().agg(total=bt.col("a").sum(), avg=bt.col("b").mean(), rows=bt.count())
print(out.to_pydict())
# {'total': [6], 'avg': [20.0], 'rows': [3]}
```

## Window functions

Aggregates become windowed via `.over(...)`. The value functions `lag`, `lead`,
`first_value`, `last_value` and the ranking functions `row_number`, `rank`,
`dense_rank`, `percent_rank`, `cume_dist`, `ntile(n)` are top-level constructors
bound with `.over(...)`. The ranking functions take no input and require an
`order_by`:

```python
from batcher import dense_rank, first_value, lag, rank, row_number

w = bt.from_pydict({"g": [1, 1, 2], "t": [1, 2, 1], "v": [10, 20, 30]})
ranked = w.with_columns(
    running=bt.col("v").sum().over(partition_by=["g"], order_by=["t"]),
    prev=lag(bt.col("v"), 1).over(partition_by=["g"], order_by=["t"]),
    first=first_value(bt.col("v")).over(partition_by=["g"], order_by=["t"]),
    rn=row_number().over(partition_by=["g"], order_by=["t"]),
    rk=rank().over(partition_by=["g"], order_by=["t"]),
    dr=dense_rank().over(partition_by=["g"], order_by=["t"]),
)
print(ranked.sort("g", "t").to_pydict()["prev"])
# [None, 10, None]
```

Cumulative and shift shorthands (Polars-style) build the same window expressions:
`col("v").cum_sum()`, `.cum_min()`, `.cum_max()`, `.cum_count()` are running
aggregates in row order (pass `partition_by=`/`order_by=` for a grouped/ordered
running value), and `col("v").shift(n)` lags (positive `n`) or leads (negative `n`).

```python
c = bt.from_pydict({"x": [1, 2, 3, 4]})
print(c.with_columns(cs=bt.col("x").cum_sum(), prev=bt.col("x").shift(1)).to_pydict())
# {'x': [1, 2, 3, 4], 'cs': [1, 3, 6, 10], 'prev': [None, 1, 2, 3]}
```

A window expression composes with ordinary arithmetic and other windows — the engine
lifts it into a `Window` operator and rewrites the surrounding expression to read the
result (see [window functions](../user-guide/window-functions.md)). The shapes that
come up most have their own names:

| Method | Equivalent |
| --- | --- |
| `.diff(n=1)` | `x - lag(x, n)` |
| `.pct_change(n=1)` | `x / lag(x, n) - 1` |
| `.rank(method="min", descending=False)` | `RANK()` / `DENSE_RANK()` / `ROW_NUMBER()` over `x` |
| `.is_duplicated()` / `.is_unique()` | `count(1) OVER (PARTITION BY x)` vs 1 |
| `.rolling_sum(k)` / `.rolling_mean(k)` / `.rolling_min(k)` / `.rolling_max(k)` / `.rolling_count(k)` | `agg(x) OVER (ROWS BETWEEN k-1 PRECEDING AND CURRENT ROW)` |
| `.rolling_var(k, ddof=1)` / `.rolling_std(k, ddof=1)` | sample (or population, `ddof=0`) variance / stddev over the same trailing frame |

All of them take `partition_by=` / `order_by=`, and `.fill_nan(v)` replaces IEEE NaN
(which `.fill_null(v)` never touches, NaN being a value rather than a null).

The `rolling_*` family aggregates a fixed trailing frame. The leading rows of each
partition aggregate a *partial* frame, as SQL does; pass `min_periods=k` to make
those rows null instead (the Polars default).

```python
r = bt.from_pydict({"x": [1, 2, 3, 4]})
print(r.with_columns(m=bt.col("x").rolling_mean(2), s=bt.col("x").rolling_sum(2, min_periods=2)).to_pydict())
# {'x': [1, 2, 3, 4], 'm': [1.0, 1.5, 2.5, 3.5], 's': [None, 3, 5, 7]}
```

```python
d = bt.from_pydict({"x": [10, 15, 30]})
print(d.with_columns(chg=bt.col("x").diff(), pct=bt.col("x").pct_change()).to_pydict())
# {'x': [10, 15, 30], 'chg': [None, 5, 15], 'pct': [None, 0.5, 1.0]}
```

## Accessor namespaces

Breadth lives on accessor namespaces rather than on the expression itself.

| Namespace | Covers |
| --- | --- |
| `.str` | `upper`, `lower`, `trim(chars=None)`, `lstrip`/`rstrip(chars=None)`, `len`, `contains`, `starts_with`, `ends_with`, `like`, `ilike`, `substr`, `left`, `right`, `split`, `split_part(delim, n)`, `strip_html()` (markup → prose; drops `<script>`/`<style>` bodies and decodes entities), `chunk(size, overlap=0)` (RAG document splitter), `minhash(num_perm=128, ngram=5)` (fuzzy-dedup signature), `replace`, `regexp_replace`, `regexp_replace_all`, `regexp_extract`, `initcap`, `hex`, `base64`, `translate`, `zfill(width)` (zero-pad numeric strings), `contains_any([...])` (true if any literal substring is present), and more |
| `.dt` | `year`, `month`, `day`, `hour`, `minute`, `second`, `quarter`, `week`, `dayofweek`, `dayofyear`, `dayname`, `monthname`, `epoch`, `epoch_ms()` / `epoch_us()` / `epoch_ns()` (integer epoch at ms/µs/ns resolution), `iso_year`, `is_leap_year`, `days_in_month`, `truncate(unit)`, `strftime(fmt)`, `offset_by("1mo15d")`, `convert_timezone(from_tz, to_tz)` (DST-aware), and more |
| `.list` | `len`, `sum`, `min`, `max`, `mean`, `median`, `std`, `var`, `product`, `n_unique`, `l2_norm`, `normalize`, `sort`, `reverse`, `unique`, `flatten`, `get(i)` (negative ok), `first()`, `last()`, `slice`, `head(n)`, `contains(v)`, `position(v)`, `intersect(o)`, `difference(o)`, `union(o)`, `transform(element()-expr)`, `filter(element()-pred)`, `join(sep)`; vector ops `dot(o)`, `cosine_similarity(o)`, `cosine_distance(o)`, `l2_distance(o)`, `jaccard(o)` (agreement rate; the MinHash/SimHash similarity estimate), `simhash(num_bits=64, seed=0)` (random-hyperplane LSH signature — the blocking key for a vector similarity join) |
| `.struct` | `field(name)` |
| `.json` | `extract_string(path)` |
| `.map` | `get(key)`, `keys()`, `values()` — read a `Map`-typed column |
| `.image` | `decode()`, `to_tensor(width, height)`, `resize(width, height)` (re-encode to PNG bytes) |
| `.audio` | `decode()`, `to_waveform()` (decode to a mono PCM `List<Float>` signal), `resample(rate)` (decode + band-limited resample to `rate` Hz — the 16 kHz audio-ML preprocessing step) |
| `.video` | `decode()` |

### More `.str` methods

| Method | Description |
| --- | --- |
| `.lpad(width, fill=" ")` / `.rpad(width, fill=" ")` | pad to `width` characters with `fill` (cycled); truncate if longer |
| `.repeat(n)` | repeat the string `n` times (`n` ≤ 0 → empty) |
| `.normalize_whitespace()` | collapse every run of whitespace to a single space and trim the ends |
| `.position(pattern)` | 1-based index of `pattern`, 0 if absent (→ Int64) |
| `.regexp_matches(pattern)` | true where the regex matches anywhere (→ Bool) |
| `.ascii()` | Unicode codepoint of the first character, 0 if empty (→ Int64) |
| `.bit_length()` / `.octet_length()` | number of bits / UTF-8 bytes in the string (→ Int64) |
| `.from_base64()` | decode standard base64 to a UTF-8 string; null if invalid |
| `.unhex()` | decode pairs of hex digits to a UTF-8 string; null if invalid |
| `.md5()` / `.sha1()` / `.sha256()` | cryptographic digest as lowercase hex; null → null |
| `.crc32()` | CRC-32 (IEEE) checksum of the UTF-8 bytes (Spark `crc32`, → Int64) |
| `.hash64()` | deterministic FNV-1a 64-bit hash, stable across partitions/machines — surrogate-key building block (→ Int64) |
| `.xxhash64()` | fast non-cryptographic 64-bit xxHash; the standard bucketing/sharding hash (→ Int64) |
| `.substring_index(delimiter, count)` | substring before the `count`-th `delimiter` (Spark) |
| `.overlay(replacement, pos, length=None)` | replace `length` chars from 1-based `pos` (SQL `OVERLAY`) |
| `.regexp_extract_all(pattern)` | every regex match as a `List<Utf8>` (DuckDB `regexp_extract_all`) |
| `.regexp_count(pattern)` | number of non-overlapping regex matches (→ Int64) |
| `.levenshtein(target)` | edit distance to the constant `target` (DuckDB `levenshtein`, → Int64) |
| `.soundex()` | American Soundex phonetic code, a 4-character key (→ Utf8) |
| `.to_date(format="%Y-%m-%d")` | parse into a Date with a strftime format; unmatched → NULL (→ Date32) |
| `.to_datetime(format)` | parse into a Timestamp (DuckDB `try_strptime`); unmatched → NULL (→ Timestamp(us)) |

### More `.dt` methods

`.century()`, `.decade()`, `.isodow()` (ISO day of week), `.last_day()` (last day
of the month), `.millennium()` — each extracts the named field of a date/time
column (→ Int64).

### More `.json` methods

| Method | Description |
| --- | --- |
| `.extract_int(path)` | the integer value at JSON `path`; null if absent or non-integral (→ Int64) |
| `.extract_float(path)` | the numeric value at JSON `path` as a float; null if absent or non-numeric |
| `.extract_bool(path)` | the boolean value at JSON `path`; null if absent or non-boolean |

For retrieval / RAG, the vector ops score each row's embedding against a query
vector (a broadcast `array(...)` literal): `bt.col("emb").list.cosine_similarity(
bt.array(*[bt.lit(x) for x in query]))`.

```python
words = bt.from_pydict({"name": ["Ann", "bob"], "tags": [["x", "y"], ["z"]]})
out = words.select(
    upper=bt.col("name").str.upper(),
    n_tags=bt.col("tags").list.len(),
)
print(out.to_pydict())
# {'upper': ['ANN', 'BOB'], 'n_tags': [2, 1]}
```

## Compatibility spellings (Polars / pandas / SQL names)

For migration, many operations carry a second, framework-familiar name alongside the
SQL-style primary. These delegate to the primary spelling — same behavior, no new IR.

Trig / clip / range on `Expr`: `.arcsin()`, `.arccos()`, `.arctan()`, `.arcsinh()`,
`.arccosh()`, `.arctanh()` (NumPy/Polars names for `.asin()`…), `.clip_min(lo)` /
`.clip_max(hi)` (Polars, for `.clip(...)`), and `.is_between(lo, hi, closed="both")`
(Polars, for `.between(...)`). Top-level `bt.arctan2(y, x)` mirrors `bt.atan2`.

On `.str`: `.to_lowercase()` / `.to_uppercase()` / `.to_titlecase()` (Polars, for
`lower`/`upper`/`initcap`), `.pad_start(w, fill)` / `.pad_end(w, fill)` and pandas'
`.ljust(w, fill)` / `.rjust(w, fill)` (for `lpad`/`rpad`), `.count_matches(pattern)`
(for `regexp_count`), `.extract(pattern, group=1)` / `.extract_all(pattern)` /
`.replace_all(pattern, value)` (for the `regexp_*` methods), `.len_chars()` /
`.len_bytes()` (for `len`/`octet_length`), `.strip_chars(chars=None)` /
`.strip_chars_start(...)` / `.strip_chars_end(...)` (for `trim`/`lstrip`/`rstrip`), and
`.head(n)` / `.tail(n)` / `.slice(offset, length=None)` (for `left`/`right`/`substr`).

On `.dt`: `.weekday()` (for `isodow`), `.ordinal_day()` (for `dayofyear`),
`.to_string(fmt)` (for `strftime`), `.date()` / `.month_start()` (for `truncate(...)`),
`.month_end()` (for `last_day`), and the sub-second components `.millisecond()` /
`.microsecond()` / `.nanosecond()`.

On `.list`: `.set_union(o)` / `.set_intersection(o)` / `.set_difference(o)` (Polars
names for `union`/`intersect`/`difference`).

pandas string spellings: `.strip(chars=None)` (for `trim`), `.startswith(p)` /
`.endswith(p)` (for `starts_with`/`ends_with`), `.match(pattern)` (for
`regexp_matches`), `.title()` (for `initcap`), plus Python's `.removeprefix(p)` /
`.removesuffix(s)`. pandas datetime spellings: `.day_name()` / `.month_name()` (for
`dayname`/`monthname`), `.daysinmonth()` (for `days_in_month`), `.weekofyear()` (for
`week`), `.normalize()` and `.floor(unit)` (for `truncate`).

## Data science toolkit

Feature engineering, profiling, and text/calendar features as expressions, so a
fit-and-apply transform is one pass over Arrow with no Python state — and, being
ordinary window + arithmetic nodes, identical single-node and distributed.

**Scaling and encoding** (each takes `partition_by=` to fit per group):
`.zscore()` (standardize), `.minmax_scale()`, `.maxabs_scale()`, `.mean_center()`,
`.label_encode()` (0-based codes by sorted value), and `.hash_bucket(n, seed=0)` for
reproducible shard / split assignment.

**Activations and shape**: `.sigmoid()`, `.logit()`, `.relu()`, `.softplus()`, and
`.softmax()` (scores to a distribution summing to 1).

**Comparison and de-duplication**: `.abs_diff(other)`, plus
`.is_first_distinct(order_by)` / `.is_last_distinct(order_by)`, which mark one row per
distinct value (the `order_by` is required so the pick is partition-independent).

**Ratios and shares**: `.pct_of_total()`, `.cumulative_pct()` (the Pareto curve),
`.normalize_l1()`, `.rank_pct()` (percentile rank), and `.safe_divide(other)`, which
yields null rather than infinity when the divisor is zero.

**Expanding (cumulative) statistics**: `.expanding_mean()`, `.expanding_var()`,
`.expanding_std()` — the growing-frame counterparts of the `rolling_*` family.

**Value predicates**: `.is_positive()`, `.is_negative()`, `.is_zero()`, `.is_even()`,
`.is_odd()`, and `.is_outlier(threshold=3.0)` (the z-score rule, as a filterable
predicate).

**Calendar features** on `.dt`: `.is_weekend()` / `.is_weekday()`,
`.is_month_start()` / `.is_month_end()`, `.is_quarter_start()` / `.is_quarter_end()`,
`.is_year_start()` / `.is_year_end()`, `.quarter_start()`, `.year_start()`,
`.days_in_year()`, and `.week_of_month()`.

**Time deltas** on `.dt`: `.seconds_between(other)`, `.minutes_between(other)`,
`.hours_between(other)`, `.days_between(other)`, and `.weeks_between(other)` measure
elapsed fixed-width time between two timestamps; `.quarter_end()` and `.year_end()`
complete the period boundaries.

**Text features** on `.str`: `.word_count()`, `.digit_count()`, `.contains_all([...])`,
`.count_char(sub)`, `.is_alpha()`,
`.is_numeric()`, `.is_alnum()`, `.is_space()`, `.is_upper()`, `.is_lower()`,
`.capitalize()`, and `.remove_punctuation()`.

```python
feats = bt.from_pydict({"g": ["a", "a", "b", "b"], "v": [1.0, 3.0, 10.0, 20.0]})
out = feats.select(
    z=bt.col("v").zscore(["g"]).round(4),
    share=bt.col("v").pct_of_total(["g"]),
    bucket=bt.col("g").hash_bucket(2),
)
print(out.to_pydict())
# {'z': [-0.7071, 0.7071, -0.7071, 0.7071], 'share': [0.25, 0.75, 0.3333333333333333, 0.6666666666666666], 'bucket': [1, 1, 1, 1]}
```

Column-level profiling aggregates complete the toolkit: `bt.q1(x)` / `bt.q3(x)` /
`bt.iqr(x)` (robust spread), `bt.value_range(x)`, `bt.null_rate(x)` /
`bt.non_null_rate(x)` (completeness), and `bt.nunique_ratio(x)` (cardinality ratio —
near 1 marks an identifier, near 0 a categorical).

## AI data-pipeline toolkit

Curating a training corpus, scrubbing PII, and budgeting context windows are all
per-row scans, so they belong in the engine rather than a Python loop. These score a
whole corpus in one vectorized pass.

**Corpus quality heuristics** on `.str` — the character-class ratios and shape
statistics that Gopher / C4 / RefinedWeb-style filters threshold on to drop boilerplate
and machine-generated text: `.alpha_ratio()`, `.digit_ratio()`, `.uppercase_ratio()`,
`.lowercase_ratio()`, `.punctuation_ratio()`, `.whitespace_ratio()`,
`.non_ascii_ratio()`, `.alnum_ratio()`, plus `.non_ascii_count()`, `.line_count()`,
`.mean_line_length()`, `.avg_word_length()`, `.sentence_count()`, `.url_count()`, and
`.email_count()`.

Document-shape signals: `.paragraph_count()`, `.is_single_line()`,
`.ends_with_punctuation()` (catches truncated crawls),
`.has_repeated_punctuation()`, `.quote_count()`, `.paren_count()`,
`.digit_to_word_ratio()`, and the code detectors `.code_fence_count()` /
`.looks_like_code()` (route code out of a prose corpus, or keep only code).

More corpus signals: `.uppercase_word_count()` (shouting/headers),
`.long_word_count(n)`, `.symbol_to_word_ratio()` (markup and ASCII art),
`.hashtag_count()` / `.mention_count()` (social-media provenance), and
`.phone_count()`.

**Cleaning and PII scrubbing**: `.remove_urls()`, `.remove_emails()`,
`.remove_phones()`, `.has_phone()`, and the shape-preserving `.mask_emails(token)` /
`.mask_urls(token)` (preferred over deletion for training data),
`.remove_non_ascii()`, `.remove_digits()`, `.remove_html_tags()`, and the budget guards
`.truncate_chars(n)` / `.truncate_words(n)` (which never cut mid-word).

**Detection predicates** for filtering: `.has_url()`, `.has_email()`,
`.has_non_ascii()`, `.has_digits()`, `.has_html()`, `.is_ascii_only()`, `.is_blank()`,
`.starts_with_bullet()`, and `.looks_like_json()` (a cheap shape check before decoding
LLM structured output).

Counts and shape predicates: `.newline_count()`, `.tab_count()`, `.space_count()`,
`.word_char_ratio()`, `.avg_sentence_length()`, `.is_short(n)` / `.is_long(n)`,
`.is_question()`, `.is_exclamation()`, `.starts_with_capital()`, `.is_all_caps()`,
`.has_currency()`, `.is_url()`, and `.is_email()` (whole-string forms, stricter than
`has_url`/`has_email`).

Extraction into `List<Utf8>`: `.extract_urls()`, `.extract_emails()`,
`.extract_numbers()`, `.extract_hashtags()`, `.extract_mentions()`, plus the scalar
`.first_sentence()`, `.first_word()`, and `.last_word()`.

Normalization for dedup keys and prose corpora: `.slugify()`, `.remove_bullets()`,
`.remove_repeated_punctuation()`, `.remove_markdown_links()`, `.remove_code_blocks()`,
`.remove_stopwords(words)`, and `.truncate_sentences(n)`.

**Token budgeting**: `.estimate_tokens(chars_per_token=4.0)` and
`.fits_token_budget(budget)` — the tokenizer-free estimate used to size context windows
without paying to tokenize the corpus.

Embedding sanity and pooling on `.list`: `.dim()` (the embedding dimension),
`.is_zero_vector()` (the failed-encoder check), `.sum_squares()`, `.mean_pool()`, and
`.max_pool()`.

**Embedding helpers** on `.list`: `.magnitude()`, `.is_unit_norm(tol)` (assert the
normalization invariant held), `.euclidean_distance(o)`, and `.angular_distance(o)` — a
true metric, unlike `1 - cosine`, which nearest-neighbour indexes require.

```python
docs_ds = bt.from_pydict({"text": ["Real prose here, with sentences.", "AAA 111 &&& http://x.co"]})
scored = docs_ds.select(
    alpha=bt.col("text").str.alpha_ratio().round(3),
    toks=bt.col("text").str.estimate_tokens(),
    linky=bt.col("text").str.has_url(),
)
print(scored.to_pydict())
# {'alpha': [0.813, 0.435], 'toks': [8, 6], 'linky': [False, True]}
```

At the dataset level, `ds.shuffle(seed=)`, `ds.stratified_split(label, test_size)`
(preserves each class's proportion, value-hashed so it is identical distributed),
`ds.sample_per_group(by, n)`, `ds.class_balance(label)`, and `ds.class_weights(label)`
cover the train-set preparation steps.
