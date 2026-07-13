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
`.cbrt()`, `.degrees()`, `.radians()`, `.factorial()` (→ Float64). Integer bitwise
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
| `.str` | `upper`, `lower`, `trim(chars=None)`, `lstrip`/`rstrip(chars=None)`, `len`, `contains`, `starts_with`, `ends_with`, `like`, `ilike`, `substr`, `left`, `right`, `split`, `split_part(delim, n)`, `strip_html()` (markup → prose; drops `<script>`/`<style>` bodies and decodes entities), `chunk(size, overlap=0)` (RAG document splitter), `minhash(num_perm=128, ngram=5)` (fuzzy-dedup signature), `replace`, `regexp_replace`, `regexp_replace_all`, `regexp_extract`, `initcap`, `hex`, `base64`, `translate`, and more |
| `.dt` | `year`, `month`, `day`, `hour`, `minute`, `second`, `quarter`, `week`, `dayofweek`, `dayofyear`, `dayname`, `monthname`, `epoch`, `iso_year`, `is_leap_year`, `days_in_month`, `truncate(unit)`, `strftime(fmt)`, `offset_by("1mo15d")`, `convert_timezone(from_tz, to_tz)` (DST-aware), and more |
| `.list` | `len`, `sum`, `min`, `max`, `mean`, `median`, `std`, `var`, `product`, `n_unique`, `l2_norm`, `normalize`, `sort`, `reverse`, `unique`, `flatten`, `get(i)` (negative ok), `first()`, `last()`, `slice`, `contains(v)`, `position(v)`, `intersect(o)`, `difference(o)`, `union(o)`, `transform(element()-expr)`, `filter(element()-pred)`, `join(sep)`; vector ops `dot(o)`, `cosine_similarity(o)`, `cosine_distance(o)`, `l2_distance(o)`, `jaccard(o)` (agreement rate; the MinHash/SimHash similarity estimate), `simhash(num_bits=64, seed=0)` (random-hyperplane LSH signature — the blocking key for a vector similarity join) |
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
