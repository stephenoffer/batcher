# Expressions API

The expression API describes column computations that lower to the Rust data plane
and run vectorized over Arrow batches. This page is the reference for the
constructors, operators, and methods callable on an {py:class}`Expr <batcher.plan.expr_ir.core.Expr>`. The accessor namespaces
({py:class}`.str <batcher.plan.expr_ir.namespaces.strings._StrNamespace>`, {py:class}`.dt <batcher.plan.expr_ir.namespaces.temporal._DtNamespace>`, {py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>`, {py:class}`.struct <batcher.plan.expr_ir.namespaces.collections._StructNamespace>`, {py:class}`.json <batcher.plan.expr_ir.namespaces.collections._JsonNamespace>`, {py:class}`.map <batcher.plan.expr_ir.namespaces.collections._MapNamespace>`, {py:class}`.image <batcher.plan.expr_ir.image._ImageNamespace>`, {py:class}`.audio <batcher.plan.expr_ir.audio._AudioNamespace>`, {py:class}`.video <batcher.plan.expr_ir.video._VideoNamespace>`) are
enumerated on {doc}`/api/relational/expression-accessors`. For a guided tour with runnable examples, see
the {doc}`expressions user guide </user-guide/transform/columns/expressions>`.

Blocks on this page share one namespace and run in order.

```python
import batcher as bt

ds = bt.from_pydict({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})
```

## Constructors

Every expression starts from one of these. {py:func}`bt.col <batcher.col>` names an input column, and the rest
build a value that has no column behind it:

| Call | Meaning |
| --- | --- |
| `bt.col(name)` | reference an input column |
| {py:func}`bt.lit(value) <batcher.lit>` | a constant value |
| {py:func}`bt.when(c).then(v)...otherwise(d) <batcher.when>` | SQL CASE |
| {py:func}`bt.coalesce(*exprs) <batcher.coalesce>` | first non-null argument |
| {py:func}`bt.nullif(a, b) <batcher.nullif>` | null when `a == b` |
| {py:func}`bt.greatest(*exprs) <batcher.greatest>` / {py:func}`bt.least(*exprs) <batcher.least>` | row-wise max / min across columns |
| {py:func}`bt.array(*exprs) <batcher.array>` | build a list column from elements |
| {py:func}`bt.atan2(y, x) <batcher.atan2>` | two-argument arctangent |
| {py:func}`bt.count() <batcher.count>` | COUNT(*) aggregate |
| {py:func}`bt.hash_rows(*exprs, seed=0) <batcher.hash_rows>` | deterministic 64-bit row digest (also `expr.hash(seed=0)`) |

```python
out = ds.select(
    label=bt.when(bt.col("a") > 1).then(bt.lit("hi")).otherwise(bt.lit("lo")),
    best=bt.greatest(bt.col("a"), bt.lit(2)),
)
print(out.to_pydict())
# {'label': ['lo', 'hi', 'hi'], 'best': [2, 2, 3]}
```

{py:func}`hash_rows <batcher.hash_rows>` digests the row's **values**, typed: an integer from its bits, a float from
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
| {py:func}`bt.sum_horizontal(*exprs) <batcher.sum_horizontal>` | row-wise sum, nulls treated as 0 |
| {py:func}`bt.count_horizontal(*exprs) <batcher.count_horizontal>` | row-wise count of non-null values |
| {py:func}`bt.product_horizontal(*exprs) <batcher.product_horizontal>` | row-wise product, nulls treated as 1 |
| {py:func}`bt.reduce_horizontal(fn, *exprs) <batcher.reduce_horizontal>` / {py:func}`bt.fold_horizontal(acc, fn, *exprs) <batcher.fold_horizontal>` | fold columns row-wise with a binary `Expr` combiner (no seed / with seed) |
| {py:func}`bt.mean_horizontal(*exprs) <batcher.mean_horizontal>` | row-wise mean, ignoring nulls |
| {py:func}`bt.min_horizontal(*exprs) <batcher.min_horizontal>` / {py:func}`bt.max_horizontal(*exprs) <batcher.max_horizontal>` | row-wise min / max, ignoring nulls (the Polars-named `least` / `greatest`) |
| {py:func}`bt.all_horizontal(*exprs) <batcher.all_horizontal>` / {py:func}`bt.any_horizontal(*exprs) <batcher.any_horizontal>` | row-wise boolean AND / OR across predicate columns |

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
| {py:meth}`.is_null() <batcher.plan.expr_ir.core.Expr.is_null>` | true where null |
| {py:meth}`.is_not_null() <batcher.plan.expr_ir.core.Expr.is_not_null>` | true where not null |
| {py:meth}`.is_nan() <batcher.plan.expr_ir.core.Expr.is_nan>` / {py:meth}`.is_not_nan() <batcher.plan.expr_ir.core.Expr.is_not_nan>` | true where the float value is NaN, or is not NaN. NaN is distinct from null |
| {py:meth}`.is_finite() <batcher.plan.expr_ir.core.Expr.is_finite>` / {py:meth}`.is_infinite() <batcher.plan.expr_ir.core.Expr.is_infinite>` | true where the float value is finite / ±infinity |
| `.fill_null(value)` | replace nulls with a value |
| {py:meth}`.forward_fill() <batcher.plan.expr_ir.core.Expr.forward_fill>` / {py:meth}`.backward_fill() <batcher.plan.expr_ir.core.Expr.backward_fill>` | carry the nearest non-null value along an ordered window ({py:meth}`.over(order_by=…) <batcher.AggExpr.over>` required) |
| {py:meth}`.interpolate() <batcher.plan.expr_ir.core.Expr.interpolate>` | draw a straight line across an interior gap instead of holding the last value flat (`.over(order_by=…)` required) |
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
| {py:meth}`.is_in([...]) <batcher.plan.expr_ir.core.Expr.is_in>` | membership test |
| {py:meth}`.between(low, high, closed="both") <batcher.plan.expr_ir.core.Expr.between>` | range test; `closed` = `"both"`/`"left"`/`"right"`/`"none"` sets which bounds are inclusive |

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
{py:meth}`.asinh() <batcher.plan.expr_ir.core.Expr.asinh>` / {py:meth}`.acosh() <batcher.plan.expr_ir.core.Expr.acosh>` / {py:meth}`.atanh() <batcher.plan.expr_ir.core.Expr.atanh>` (→ Float64). The reciprocal trig pair
{py:meth}`.sec() <batcher.plan.expr_ir.core.Expr.sec>` / {py:meth}`.csc() <batcher.plan.expr_ir.core.Expr.csc>`, the gamma function {py:meth}`.gamma() <batcher.plan.expr_ir.core.Expr.gamma>` and its log {py:meth}`.lgamma() <batcher.plan.expr_ir.core.Expr.lgamma>` (which stays
finite where `.gamma()` overflows, above about 171), and two rounding modes that are not
`.round()`: `.rint()` rounds half to *even*, `.even()` rounds *away from zero* to the
nearest even integer. Integer bitwise
ops (distinct from the boolean `&`/`|`): {py:meth}`.bitwise_and(o) <batcher.plan.expr_ir.core.Expr.bitwise_and>`, {py:meth}`.bitwise_or(o) <batcher.plan.expr_ir.core.Expr.bitwise_or>`,
{py:meth}`.bitwise_xor(o) <batcher.plan.expr_ir.core.Expr.bitwise_xor>`, {py:meth}`.bitwise_left_shift(o) <batcher.plan.expr_ir.core.Expr.bitwise_left_shift>`, {py:meth}`.bitwise_right_shift(o) <batcher.plan.expr_ir.core.Expr.bitwise_right_shift>`, and
{py:meth}`.bit_count() <batcher.plan.expr_ir.core.Expr.bit_count>` (the number of set bits, i.e. population count → Int64).

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
| {py:meth}`.neg() <batcher.plan.expr_ir.core.Expr.neg>` | arithmetic negation (the Polars spelling of the unary minus) |
| `.chr()` | the character at this Unicode code point (DuckDB/Spark `chr`) |
| {py:meth}`.to_base(radix) <batcher.plan.expr_ir.core.Expr.to_base>` | this integer written in base 2..36 (DuckDB {py:meth}`to_base <batcher.plan.expr_ir.core.Expr.to_base>`; `bin` is radix 2) |
| {py:meth}`.format_bytes(si=False) <batcher.plan.expr_ir.core.Expr.format_bytes>` | a byte count as human-readable text, such as `1.5 KiB`, or `1.5 kB` with `si=True` |
| `.clip(lower=None, upper=None)` | clamp each value into `[lower, upper]` (either bound optional) |
| {py:meth}`.eq_missing(other) <batcher.plan.expr_ir.core.Expr.eq_missing>` | null-safe equality (SQL `IS NOT DISTINCT FROM`): two nulls compare equal, null vs non-null is false (never null) |
| `.try_cast(type)` | the safe-ingest spelling of `.cast`: unconvertible values become NULL instead of erroring (DuckDB `TRY_CAST`) |
| {py:meth}`.approx_count_distinct() <batcher.plan.expr_ir.core.Expr.approx_count_distinct>` | approximate `COUNT(DISTINCT)` via a HyperLogLog sketch (~2% error) |

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
(the value at the first or last row in `order_by` order). `order_by` is required there, because an arrival-order first or last wouldn't be partition-independent. `bt.count()` is the top-level `COUNT(*)`. Each of these returns an {py:class}`AggExpr <batcher.AggExpr>`, the aggregate type that {py:meth}`group_by(...).agg(...) <batcher.Dataset.group_by>` and {py:meth}`.over(...) <batcher.AggExpr.over>` consume. You rarely name it directly.

The **assembly-contiguity** aggregates measure how a set of lengths is distributed *by
base* rather than by item, which is what genome-assembly quality is judged on:
{py:meth}`.n50() <batcher.AggExpr>` (the length at which pieces at least that long hold half
the total), {py:meth}`.n90() <batcher.AggExpr>` (the same at 90%),
{py:meth}`.l50() <batcher.AggExpr>` (the *count* of pieces needed to reach half — N is a
length, L is a count), and {py:meth}`.aun() <batcher.AggExpr>` (the area under the Nx curve,
`sum(l²)/sum(l)`, which is threshold-free and therefore continuous where N50 steps). None is
a quantile of the same lengths: a median weighs every piece equally, so an assembly of one
10 Mb chromosome plus a thousand 500 bp fragments has a median of 500 and an N50 of 10 Mb.
All four are mergeable, so a value computed over a shuffle equals the single-node one. See
{doc}`/cookbook/expressions/genomics/index`.

The **distribution** aggregates read a group's whole value list rather than a running
total: `.entropy()` (base-2 Shannon entropy of the value distribution, DuckDB `entropy`),
{py:meth}`.mad() <batcher.plan.expr_ir.core.Expr.mad>` (median absolute deviation, a spread measure a single outlier cannot move),
`.kurtosis_pop()` (the population form of `.kurtosis()`), `.quantile_disc(q)` (the
quantile *element*, where `.quantile(q)` interpolates between two of them), `.top_k(k)`
(the `k` most frequent values as a list, DuckDB `approx_top_k`, computed exactly here),
{py:meth}`.kahan_sum() <batcher.plan.expr_ir.core.Expr.kahan_sum>` (compensated summation, DuckDB `fsum` or {py:meth}`kahan_sum <batcher.plan.expr_ir.core.Expr.kahan_sum>`) gives the same answer as
`.sum()` on a well-conditioned column and a materially better one when the addends differ
wildly in magnitude), and {py:meth}`.any_value() <batcher.plan.expr_ir.core.Expr.any_value>` (one value from the group, DuckDB {py:meth}`any_value <batcher.plan.expr_ir.core.Expr.any_value>` / `arbitrary`; the
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
{py:func}`dense_rank <batcher.dense_rank>`, {py:func}`percent_rank <batcher.percent_rank>`, {py:func}`cume_dist <batcher.cume_dist>`, {py:func}`ntile(n) <batcher.ntile>` are top-level constructors
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
`col("v").cum_sum()`, `.cum_min()`, `.cum_max()`, `.cum_count()` and
{py:meth}`.cum_prod() <batcher.plan.expr_ir.core.Expr.cum_prod>` are running
aggregates in row order (pass `partition_by=`/`order_by=` for a grouped/ordered
running value), and `col("v").shift(n)` lags (positive `n`) or leads (negative `n`).

```python
c = bt.from_pydict({"x": [1, 2, 3, 4]})
print(c.with_columns(cs=bt.col("x").cum_sum(), prev=bt.col("x").shift(1)).to_pydict())
# {'x': [1, 2, 3, 4], 'cs': [1, 3, 6, 10], 'prev': [None, 1, 2, 3]}
```

`.cum_prod()` returns `Float64` even for an integer input, because a running product
overflows an `Int64` far sooner than a running sum does and wrapping silently is the wrong
answer for a compounding factor. Nulls are skipped, as they are for the rest of the family.

```python
rates = bt.from_pydict({"fund": ["a", "a", "b", "b"], "r": [1.1, 1.2, 2.0, 0.5]})
print(rates.with_columns(growth=bt.col("r").cum_prod(partition_by="fund")).to_pydict()["growth"])
# [1.1, 1.32, 2.0, 1.0]
```

A window expression composes with ordinary arithmetic and other windows. The engine lifts it into a `Window` operator and rewrites the surrounding expression to read the result, as described in {doc}`window functions </user-guide/analyze/window-functions>`. The shapes that come up most have their own names:

| Method | Equivalent |
| --- | --- |
| `.diff(n=1)` | `x - lag(x, n)` |
| {py:meth}`.pct_change(n=1) <batcher.plan.expr_ir.core.Expr.pct_change>` | `x / lag(x, n) - 1` |
| `.rank(method="min", descending=False)` | `RANK()` / `DENSE_RANK()` / `ROW_NUMBER()` over `x` |
| `.is_duplicated()` / `.is_unique()` | `count(1) OVER (PARTITION BY x)` vs 1 |
| `.rolling_sum(k)` / `.rolling_mean(k)` / `.rolling_min(k)` / `.rolling_max(k)` / `.rolling_count(k)` | `agg(x) OVER (ROWS BETWEEN k-1 PRECEDING AND CURRENT ROW)` |
| {py:meth}`.rolling_var(k, ddof=1) <batcher.plan.expr_ir.core.Expr.rolling_var>` / {py:meth}`.rolling_std(k, ddof=1) <batcher.plan.expr_ir.core.Expr.rolling_std>` | sample (or population, `ddof=0`) variance / stddev over the same trailing frame |
| {py:meth}`.rolling_sum_by(by, w) <batcher.plan.expr_ir.core.Expr.rolling_sum_by>` / {py:meth}`.rolling_mean_by <batcher.plan.expr_ir.core.Expr.rolling_mean_by>` / {py:meth}`.rolling_min_by <batcher.plan.expr_ir.core.Expr.rolling_min_by>` / {py:meth}`.rolling_max_by <batcher.plan.expr_ir.core.Expr.rolling_max_by>` / {py:meth}`.rolling_count_by <batcher.plan.expr_ir.core.Expr.rolling_count_by>` | the same aggregates over a *time* window: `RANGE BETWEEN w PRECEDING AND CURRENT ROW` ordered by `by`, where `w` may be a duration such as `"5m"` |
| {py:meth}`.ewm_mean(…) <batcher.plan.expr_ir.core.Expr.ewm_mean>` / {py:meth}`.ewm_std(…) <batcher.plan.expr_ir.core.Expr.ewm_std>` / {py:meth}`.ewm_var(…) <batcher.plan.expr_ir.core.Expr.ewm_var>` | exponentially weighted moving statistics, decayed by `alpha` / `span` / `half_life` / `com` (`.over(order_by=…)` required) |
| {py:meth}`.ewm_mean_by(by, half_life) <batcher.plan.expr_ir.core.Expr.ewm_mean_by>` | the same smoother decayed by *elapsed* `by` rather than by row position, for an irregularly sampled series |
| {py:meth}`.rle_id() <batcher.plan.expr_ir.core.Expr.rle_id>` | 0-based index of the current run of equal values (`.over(order_by=…)` required) |
| {py:meth}`.peak_max(order_by=…) <batcher.plan.expr_ir.core.Expr.peak_max>` / {py:meth}`.peak_min(order_by=…) <batcher.plan.expr_ir.core.Expr.peak_min>` | true at a local extremum, strictly beyond both neighbours; an edge row is never one |

All of them take `partition_by=` / `order_by=`, and {py:meth}`.fill_nan(v) <batcher.plan.expr_ir.core.Expr.fill_nan>` replaces IEEE NaN (which `.fill_null(v)` never touches, NaN being a value rather than a null).

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

Trig / clip / range on `Expr`: {py:meth}`.arcsin() <batcher.plan.expr_ir.core.Expr.arcsin>`, {py:meth}`.arccos() <batcher.plan.expr_ir.core.Expr.arccos>`, {py:meth}`.arctan() <batcher.plan.expr_ir.core.Expr.arctan>`, {py:meth}`.arcsinh() <batcher.plan.expr_ir.core.Expr.arcsinh>`,
{py:meth}`.arccosh() <batcher.plan.expr_ir.core.Expr.arccosh>`, {py:meth}`.arctanh() <batcher.plan.expr_ir.core.Expr.arctanh>` (the NumPy and Polars names for {py:meth}`.asin() <batcher.plan.expr_ir.core.Expr.asin>` and friends), {py:meth}`.clip_min(lo) <batcher.plan.expr_ir.core.Expr.clip_min>` /
`.clip_max(hi)` (Polars, for `.clip(...)`), and `.is_between(lo, hi, closed="both")`
(Polars, for `.between(...)`). Top-level {py:func}`bt.arctan2(y, x) <batcher.arctan2>` mirrors `bt.atan2`.

On `.str`: `.to_lowercase()` / `.to_uppercase()` / `.to_titlecase()` (Polars, for
`lower`/`upper`/`initcap`), `.pad_start(w, fill)` / `.pad_end(w, fill)` and pandas'
`.ljust(w, fill)` / `.rjust(w, fill)` (for `lpad`/`rpad`), `.count_matches(pattern)`
(for `regexp_count`), `.extract(pattern, group=1)` / `.extract_all(pattern)` /
{py:meth}`.replace_all(pattern, value) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.replace_all>` (for the `regexp_*` methods), {py:meth}`.len_chars() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.len_chars>` /
`.len_bytes()` (for `len`/`octet_length`), `.strip_chars(chars=None)` /
`.strip_chars_start(...)` / `.strip_chars_end(...)` (for `trim`/`lstrip`/`rstrip`), and
`.head(n)` / `.tail(n)` / `.slice(offset, length=None)` (for `left`/`right`/`substr`).

On `.dt`: `.weekday()` (for `isodow`), `.ordinal_day()` (for `dayofyear`),
`.to_string(fmt)` (for `strftime`), `.date()` / `.month_start()` (for `truncate(...)`),
{py:meth}`.month_end() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.month_end>` (for {py:meth}`last_day <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.last_day>`), and the sub-second components {py:meth}`.millisecond() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.millisecond>` /
{py:meth}`.microsecond() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.microsecond>` / {py:meth}`.nanosecond() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.nanosecond>`.

On `.list`: {py:meth}`.set_union(o) <batcher.plan.expr_ir.namespaces.collections._ListNamespace.set_union>` / {py:meth}`.set_intersection(o) <batcher.plan.expr_ir.namespaces.collections._ListNamespace.set_intersection>` / {py:meth}`.set_difference(o) <batcher.plan.expr_ir.namespaces.collections._ListNamespace.set_difference>` (Polars
names for `union`/`intersect`/`difference`).

pandas string spellings: `.strip(chars=None)` (for `trim`), `.startswith(p)` /
`.endswith(p)` (for `starts_with`/`ends_with`), `.match(pattern)` (for
`regexp_matches`), `.title()` (for `initcap`), plus Python's `.removeprefix(p)` /
{py:meth}`.removesuffix(s) <batcher.plan.expr_ir.namespaces.strings._StrNamespace.removesuffix>`. pandas datetime spellings: {py:meth}`.day_name() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.day_name>` / {py:meth}`.month_name() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.month_name>` (for
`dayname`/`monthname`), `.daysinmonth()` (for `days_in_month`), `.weekofyear()` (for
`week`), `.normalize()` and {py:meth}`.floor(unit) <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.floor>` (for `truncate`), plus {py:meth}`.ceil(unit) <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.ceil>` (the next boundary, unless already on one) and {py:meth}`.round(unit) <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.round>` (the nearer one, half rounding up) over `second` through `year`, both measuring a calendar unit by real elapsed time so mid-February rounds to March.

### pandas names on `Expr`

A script ported from pandas finds each operation under the name it already types. Every
one of these delegates to the primary, so the plan is identical:

| pandas spelling | Batcher primary |
|---|---|
| `.astype(dtype)` | `.cast(dtype)` |
| `.isna()`, `.isnull()` | `.is_null()` |
| `.notna()`, `.notnull()` | `.is_not_null()` |
| `.fillna(value)` | `.fill_null(value)` |
| {py:meth}`.isin(values) <batcher.plan.expr_ir.core.Expr.isin>` | {py:meth}`.is_in(values) <batcher.plan.expr_ir.core.Expr.is_in>` |
| `.nunique()` | `.n_unique()` |
| `.rename(name)` | `.alias(name)` |
| `.skew()`, `.kurt()` | `.skewness()`, `.kurtosis()` |
| `.cumsum()`, `.cummax()`, `.cummin()`, `.cumcount()`, {py:meth}`.cumprod() <batcher.plan.expr_ir.compat.names.cumprod>` | `.cum_sum()`, `.cum_max()`, `.cum_min()`, `.cum_count()`, `.cum_prod()` |
| `.prod()` | `.product()` |
| `.any()`, `.all()` | `.bool_or()`, `.bool_and()` |
| `.log()` | `.ln()` (numpy's natural-log convention) |

Cast type names are matched case-insensitively, so pandas' `.astype("Int64")` and SQL's
`.cast("BIGINT")` spelling both resolve to the canonical `int64`.

Each operator also has the pandas method form, for code that cannot emit an operator:
`.add(o)`, `.sub(o)`, `.mul(o)`, `.truediv(o)`, `.div(o)`, `.floordiv(o)`, `.mod(o)`,
{py:meth}`.eq(o) <batcher.plan.expr_ir.core.Expr.eq>`, {py:meth}`.ne(o) <batcher.plan.expr_ir.core.Expr.ne>`, {py:meth}`.lt(o) <batcher.plan.expr_ir.core.Expr.lt>`, {py:meth}`.le(o) <batcher.plan.expr_ir.core.Expr.le>`, {py:meth}`.gt(o) <batcher.plan.expr_ir.core.Expr.gt>`, {py:meth}`.ge(o) <batcher.plan.expr_ir.core.Expr.ge>`. The boolean operators
likewise carry {py:meth}`.and_(o) <batcher.plan.expr_ir.core.Expr.and_>`, {py:meth}`.or_(o) <batcher.plan.expr_ir.core.Expr.or_>`, {py:meth}`.not_() <batcher.plan.expr_ir.core.Expr.not_>`, and {py:meth}`.xor(o) <batcher.plan.expr_ir.core.Expr.xor>`, because Python's `and`,
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

On `.str`, the Python string predicates: {py:meth}`.isdigit() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.isdigit>`, {py:meth}`.isalpha() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.isalpha>`, {py:meth}`.isalnum() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.isalnum>`, and
{py:meth}`.isspace() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.isspace>` (for {py:meth}`is_numeric <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_numeric>`/{py:meth}`is_alpha <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_alpha>`/{py:meth}`is_alnum <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_alnum>`/{py:meth}`is_space <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_space>`), plus Polars'
`.strip_prefix(p)` / `.strip_suffix(s)` (for `removeprefix`/`removesuffix`).

On `.dt`, the snake_case spellings {py:meth}`.day_of_week() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.day_of_week>`, {py:meth}`.day_of_year() <batcher.plan.expr_ir.namespaces.temporal._DtNamespace.day_of_year>`, and
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
| `str.islower`, `str.isupper` | {py:meth}`is_lower <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_lower>`/{py:meth}`is_upper <batcher.plan.expr_ir.namespaces.strings._StrNamespace.is_upper>` are true for an uncased string such as `"123"`; Python's are false. |
| `str.count` | pandas' `count` is a regex count. Use `str.regexp_count(pattern)`. |
| `str.casefold` | Python's casefold is not lowercase for non-ASCII (`"ß"` folds to `"ss"`). |

## Data science toolkit and evaluation metrics

The feature-engineering, profiling and model-evaluation expressions are tabulated
separately, in {doc}`/api/relational/expressions-datascience`.

## See also

- {doc}`/api/relational/expressions-datascience`: the feature-engineering and metric expressions.
- {doc}`/api/relational/expression-accessors`: every method on every accessor namespace.
- {doc}`/api/relational/functions`: the top-level scalar, horizontal, aggregate, and window functions.
- {doc}`/api/models/metrics`: the scoring and statistical aggregates used inside `agg()`.
- {doc}`/user-guide/transform/columns/expressions`: the same language taught rather than tabulated.
- {doc}`/cookbook/expressions/index`: 34 runnable recipes for the methods on this page.
