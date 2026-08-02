# Expressions API

The expression API describes column computations that lower to the Rust data plane
and run vectorized over Arrow batches. This page is the reference for the
constructors, operators, and methods callable on an `Expr`. The accessor namespaces
(`.str`, `.dt`, `.list`, `.struct`, `.json`, `.map`, `.image`, `.audio`, `.video`) are
enumerated on {doc}`/api/relational/expression-accessors`. For a guided tour with runnable examples, see
the {doc}`expressions user guide </user-guide/transform/columns/expressions>`.

Blocks on this page share one namespace and run in order.

```python
import batcher as bt

ds = bt.from_pydict({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})
```

## Constructors

Every expression starts from one of these. `bt.col` names an input column, and the rest
build a value that has no column behind it:

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
across partitions, runs, machines and versions. That stability is what lets it key a reproducible split, a surrogate key, or a hash bucket. It's 3 to 10x faster than hashing `cast(col, "string")`, and unlike that idiom it doesn't depend on how a float prints.

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

Python's operators are overloaded on `Expr`, so an expression reads like ordinary
arithmetic. Each group below lowers to the same engine kernel a named method would:

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

Arithmetic propagates nulls, so these methods are how you test for a null and how you
replace one:

| Method | Description |
| --- | --- |
| `.is_null()` | true where null |
| `.is_not_null()` | true where not null |
| `.is_nan()` / `.is_not_nan()` | true where the float value is NaN, or is not NaN. NaN is distinct from null |
| `.is_finite()` / `.is_infinite()` | true where the float value is finite / ±infinity |
| `.fill_null(value)` | replace nulls with a value |
| `.forward_fill()` / `.backward_fill()` | carry the nearest non-null value along an ordered window (`.over(order_by=…)` required) |
| `.cut(breaks, labels=None, left_closed=False)` | bin a numeric column into labeled intervals |

```python
nulls = bt.from_pydict({"x": [1, None, 3]})
out = nulls.select(filled=bt.col("x").fill_null(0), missing=bt.col("x").is_null())
print(out.to_pydict())
# {'filled': [1, 0, 3], 'missing': [False, True, False]}
```

## Type, membership, and range

These convert a value's type or test it against a set or an interval, which is the work
most filters do before anything else happens:

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
`.asinh()` / `.acosh()` / `.atanh()` (→ Float64). The reciprocal trig pair
`.sec()` / `.csc()`, the gamma function `.gamma()` and its log `.lgamma()` (which stays
finite where `.gamma()` overflows, above about 171), and two rounding modes that are not
`.round()`: `.rint()` rounds half to *even*, `.even()` rounds *away from zero* to the
nearest even integer. Integer bitwise
ops (distinct from the boolean `&`/`|`): `.bitwise_and(o)`, `.bitwise_or(o)`,
`.bitwise_xor(o)`, `.bitwise_left_shift(o)`, `.bitwise_right_shift(o)`, and
`.bit_count()` (the number of set bits, i.e. population count → Int64).

```python
out = ds.select(root=bt.col("b").sqrt(), third=(bt.col("b") / 3).round(2))
print(out.to_pydict())
# {'root': [3.1622776601683795, 4.47213595499958, 5.477225575051661], 'third': [3.33, 6.67, 10.0]}
```

## Other core methods

The remaining methods on the base `Expr` name an output, sort within a window, or reach
the accessor namespaces:

| Method | Description |
| --- | --- |
| `.alias(name)` | bind an output name to a derived expression, for positional `select` |
| `.neg()` | arithmetic negation (the Polars spelling of the unary minus) |
| `.chr()` | the character at this Unicode code point (DuckDB/Spark `chr`) |
| `.to_base(radix)` | this integer written in base 2..36 (DuckDB `to_base`; `bin` is radix 2) |
| `.format_bytes(si=False)` | a byte count as human-readable text, such as `1.5 KiB`, or `1.5 kB` with `si=True` |
| `.clip(lower=None, upper=None)` | clamp each value into `[lower, upper]` (either bound optional) |
| `.eq_missing(other)` | null-safe equality (SQL `IS NOT DISTINCT FROM`): two nulls compare equal, null vs non-null is false (never null) |
| `.try_cast(type)` | the safe-ingest spelling of `.cast`: unconvertible values become NULL instead of erroring (DuckDB `TRY_CAST`) |
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
(the value at the first or last row in `order_by` order). `order_by` is required there, because an arrival-order first or last wouldn't be partition-independent. `bt.count()` is the top-level `COUNT(*)`. Each of these returns an `AggExpr`, the aggregate type that `group_by(...).agg(...)` and `.over(...)` consume. You rarely name it directly.

The **distribution** aggregates read a group's whole value list rather than a running
total: `.entropy()` (base-2 Shannon entropy of the value distribution, DuckDB `entropy`),
`.mad()` (median absolute deviation, a spread measure a single outlier cannot move),
`.kurtosis_pop()` (the population form of `.kurtosis()`), `.quantile_disc(q)` (the
quantile *element*, where `.quantile(q)` interpolates between two of them), `.top_k(k)`
(the `k` most frequent values as a list, DuckDB `approx_top_k`, computed exactly here),
`.kahan_sum()` (compensated summation, DuckDB `fsum` or `kahan_sum`) gives the same answer as
`.sum()` on a well-conditioned column and a materially better one when the addends differ
wildly in magnitude), and `.any_value()` (one value from the group, DuckDB `any_value` / `arbitrary`; the
engine resolves "unspecified" to the group minimum so a distributed run agrees with a
single-node one).

For heavy skew, the bounded-memory **approximate** variants keep one fixed-size
sketch per group instead of every value, so a hot key cannot OOM: `.approx_n_unique()`
(HLL, ~2% error) and `.approx_quantile(q)` / `.approx_median()` (DDSketch). They are
mergeable, so results are identical single-node and distributed.

An aggregate does not have to appear inside `group_by(...).agg(...)`. In a `select`
whose every item is an aggregate it means the whole-frame aggregation and returns one
row. Anywhere else, including `with_columns`, a mixed `select`, and a `filter` predicate, it means
the whole-frame aggregate **broadcast to every row**, which is `.over()` with no
partition:

```python
print(ds.select(total=bt.col("a").sum()).to_pydict())
# {'total': [6]}
print(ds.with_columns(share=bt.col("a") / bt.col("a").sum()).to_pydict()["share"])
# [0.16666666666666666, 0.3333333333333333, 0.5]
print(ds.filter(bt.col("a") > bt.col("a").mean()).to_pydict())
# {'a': [3], 'b': [30.0]}
```

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

A window expression composes with ordinary arithmetic and other windows. The engine lifts it into a `Window` operator and rewrites the surrounding expression to read the result, as described in {doc}`window functions </user-guide/analyze/window-functions>`. The shapes that come up most have their own names:

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

## Compatibility spellings

For migration, many operations carry a second, framework-familiar name alongside the
SQL-style primary. These delegate to the primary spelling, with the same behavior and no new IR.

Trig / clip / range on `Expr`: `.arcsin()`, `.arccos()`, `.arctan()`, `.arcsinh()`,
`.arccosh()`, `.arctanh()` (the NumPy and Polars names for `.asin()` and friends), `.clip_min(lo)` /
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

### pandas names on `Expr`

A script ported from pandas finds each operation under the name it already types. Every
one of these delegates to the primary, so the plan is identical:

| pandas spelling | Batcher primary |
|---|---|
| `.astype(dtype)` | `.cast(dtype)` |
| `.isna()`, `.isnull()` | `.is_null()` |
| `.notna()`, `.notnull()` | `.is_not_null()` |
| `.fillna(value)` | `.fill_null(value)` |
| `.isin(values)` | `.is_in(values)` |
| `.nunique()` | `.n_unique()` |
| `.rename(name)` | `.alias(name)` |
| `.skew()`, `.kurt()` | `.skewness()`, `.kurtosis()` |
| `.cumsum()`, `.cummax()`, `.cummin()`, `.cumcount()` | `.cum_sum()`, `.cum_max()`, `.cum_min()`, `.cum_count()` |
| `.prod()` | `.product()` |
| `.any()`, `.all()` | `.bool_or()`, `.bool_and()` |
| `.log()` | `.ln()` (numpy's natural-log convention) |

Cast type names are matched case-insensitively, so pandas' `.astype("Int64")` and SQL's
`.cast("BIGINT")` spelling both resolve to the canonical `int64`.

Each operator also has the pandas method form, for code that cannot emit an operator:
`.add(o)`, `.sub(o)`, `.mul(o)`, `.truediv(o)`, `.div(o)`, `.floordiv(o)`, `.mod(o)`,
`.eq(o)`, `.ne(o)`, `.lt(o)`, `.le(o)`, `.gt(o)`, `.ge(o)`. The boolean operators
likewise carry `.and_(o)`, `.or_(o)`, `.not_()`, and `.xor(o)`, because Python's `and`,
`or`, and `not` keywords cannot be overloaded.

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, None, 3], "y": [10, 20, 30]})
print(ds.select(
    filled=bt.col("x").fillna(0),
    missing=bt.col("x").isna(),
    total=bt.col("x").add(bt.col("y")),
).to_pydict())
# {'filled': [1, 0, 3], 'missing': [False, True, False], 'total': [11, None, 33]}
```

### Python `str` and numpy names on the accessors

On `.str`, the Python string predicates: `.isdigit()`, `.isalpha()`, `.isalnum()`, and
`.isspace()` (for `is_numeric`/`is_alpha`/`is_alnum`/`is_space`), plus Polars'
`.strip_prefix(p)` / `.strip_suffix(s)` (for `removeprefix`/`removesuffix`).

On `.dt`, the snake_case spellings `.day_of_week()`, `.day_of_year()`, and
`.week_of_year()` (for `dayofweek`/`dayofyear`/`weekofyear`).

On `.list`, `.lengths()` (the legacy Polars name for `len`), `.element_at(i)` (the
PySpark name for `get`), and `.argmin()` / `.argmax()` (the numpy names for
`arg_min`/`arg_max`).

Some names from other engines are deliberately **absent**, because they mean something different
here and a silently-wrong alias is worse than a missing one:

| Absent name | Why |
|---|---|
| `str.find`, `str.index` | `position` is 1-based and returns 0 when absent; pandas' `find` is 0-based and returns -1. |
| `str.substring` | `substr` is 1-based SQL. Use the 0-based `str.slice(offset, length)`. |
| `str.islower`, `str.isupper` | `is_lower`/`is_upper` are true for an uncased string such as `"123"`; Python's are false. |
| `str.count` | pandas' `count` is a regex count. Use `str.regexp_count(pattern)`. |
| `str.casefold` | Python's casefold is not lowercase for non-ASCII (`"ß"` folds to `"ss"`). |

## Data science toolkit

Feature engineering, profiling, and text/calendar features as expressions, so a
fit-and-apply transform is one pass over Arrow with no Python state. They're ordinary window and arithmetic nodes, so they're identical single-node and distributed.

**Scaling and encoding** (each takes `partition_by=` to fit per group):
`.zscore()` (standardize), `.minmax_scale()`, `.maxabs_scale()`, `.mean_center()`,
`.label_encode()` (0-based codes by sorted value), and `.hash_bucket(n, seed=0)` for
reproducible shard / split assignment.

**Activations and shape**: `.sigmoid()`, `.logit()`, `.relu()`, `.softplus()`, `.silu()`
(Swish, `x·sigmoid(x)`), `.gelu()` (the transformer default, tanh approximation), `.mish()`,
`.hardsigmoid()` / `.hardswish()` (the cheap piecewise-linear MobileNet variants),
`.leaky_relu(negative_slope=0.01)`, `.elu(alpha=1.0)`, `.hardtanh()`, `.softsign()`,
`.tanhshrink()`, and
`.softmax()` (scores to a distribution summing to 1). Each matches its `torch.nn.functional`
counterpart and runs in the data plane.

**Comparison and de-duplication**: `.abs_diff(other)`, plus
`.is_first_distinct(order_by)` / `.is_last_distinct(order_by)`, which mark one row per
distinct value (the `order_by` is required so the pick is partition-independent).

**Ratios and shares**: `.pct_of_total()`, `.cumulative_pct()` (the Pareto curve),
`.normalize_l1()`, `.rank_pct()` (percentile rank), and `.safe_divide(other)`, which
yields null rather than infinity when the divisor is zero.

**Expanding (cumulative) statistics**: `.expanding_mean()`, `.expanding_var()`,
`.expanding_std()`, the growing-frame counterparts of the `rolling_*` family.

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

**Weighted statistics** (when rows carry survey, recency, or size weights):
`bt.weighted_mean(x, w)`, `bt.weighted_var(x, w)`, `bt.weighted_std(x, w)`,
`bt.weighted_covariance(x, y, w)`, and `bt.weighted_correlation(x, y, w)` are each the
frequency-weighted form matching `numpy.average`.

Column-level profiling aggregates complete the toolkit: `bt.q1(x)` / `bt.q3(x)` /
`bt.iqr(x)` (robust spread), `bt.value_range(x)`, `bt.null_rate(x)` /
`bt.non_null_rate(x)` (completeness), and `bt.nunique_ratio(x)`, the cardinality ratio, where near 1 marks an identifier and near 0 a categorical.

## Model evaluation metrics

Every model-evaluation metric is an expression, so it belongs inside `agg()` and composes
with `group_by`. A per-segment report is the same query with a grouping added, at no extra
pass. All are checked against scikit-learn where it defines them.

They are top-level functions rather than `Expr` methods, so they are enumerated on
{doc}`/api/models/metrics` with their signatures and docstrings.

```python
scored = bt.from_pydict({"y": [1, 0, 1, 1, 0], "p": [1, 0, 0, 1, 1]})
print(scored.agg(
    f1=bt.f1_score("y", "p"),
    jaccard=bt.jaccard_score("y", "p"),
    informedness=bt.informedness("y", "p"),
).to_pydict())
```

The metrics that need a global ordering (ROC AUC, average precision) or return a table
(confusion matrix, calibration curve) are Dataset functions in `batcher.ml.metrics`, not
expressions. See {doc}`/ml/evaluation/evaluation`.

## See also

- {doc}`/api/relational/expression-accessors`: every method on every accessor namespace.
- {doc}`/api/relational/functions`: the top-level scalar, horizontal, aggregate, and window functions.
- {doc}`/api/models/metrics`: the scoring and statistical aggregates used inside `agg()`.
- {doc}`/user-guide/transform/columns/expressions`: the same language taught rather than tabulated.
- {doc}`/cookbook/expressions/index`: 34 runnable recipes for the methods on this page.
