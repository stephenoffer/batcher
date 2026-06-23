# Expressions API

The expression API describes column computations that lower to the Rust data plane
and run vectorized over Arrow batches. This page is the reference for the
constructors, operators, methods, and accessor namespaces. For a guided tour with
runnable examples, see the expressions user guide.

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

```python
out = ds.select(
    label=bt.when(bt.col("a") > 1).then(bt.lit("hi")).otherwise(bt.lit("lo")),
    best=bt.greatest(bt.col("a"), bt.lit(2)),
)
print(out.to_pydict())
# {'label': ['lo', 'hi', 'hi'], 'best': [2, 2, 3]}
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
| `.fill_null(value)` | replace nulls with a value |

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
| `.between(low, high)` | inclusive range test |

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
`.cbrt()`, `.degrees()`, `.radians()`. Integer bitwise ops (distinct from the
boolean `&`/`|`): `.bitwise_and(o)`, `.bitwise_or(o)`, `.bitwise_xor(o)`,
`.bitwise_left_shift(o)`, `.bitwise_right_shift(o)`.

```python
out = ds.select(root=bt.col("b").sqrt(), third=(bt.col("b") / 3).round(2))
print(out.to_pydict())
# {'root': [3.1622776601683795, 4.47213595499958, 5.477225575051661], 'third': [3.33, 6.67, 10.0]}
```

## Aggregation methods

Used inside `group_by(...).agg(...)`: `.sum()`, `.min()`, `.max()`, `.mean()`,
`.var()`, `.std()`, `.median()`, `.quantile(q)`, `.count()`, `.n_unique()`
(aliased `.count_distinct()`), `.mode()`, `.bool_and()`, `.bool_or()`,
`.array_agg()` (collect each group's values into a `List`; SQL `array_agg` /
Spark `collect_list`), `.arg_min(by=…)` / `.arg_max(by=…)` (the value at the
row with the extreme `by` key), and `.first(order_by=…)` / `.last(order_by=…)`
(the value at the first/last row in `order_by` order — a required argument, since
an arrival-order first/last would not be partition-independent). `bt.count()` is
the top-level `COUNT(*)`.

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

## Accessor namespaces

Breadth lives on accessor namespaces rather than on the expression itself.

| Namespace | Covers |
| --- | --- |
| `.str` | `upper`, `lower`, `trim(chars=None)`, `lstrip`/`rstrip(chars=None)`, `len`, `contains`, `starts_with`, `ends_with`, `like`, `ilike`, `substr`, `left`, `right`, `split`, `split_part(delim, n)`, `replace`, `regexp_replace`, `regexp_replace_all`, `regexp_extract`, `initcap`, `hex`, `base64`, `translate`, and more |
| `.dt` | `year`, `month`, `day`, `hour`, `minute`, `second`, `quarter`, `week`, `dayofweek`, `dayofyear`, `dayname`, `monthname`, `epoch`, `iso_year`, `is_leap_year`, `days_in_month`, `truncate(unit)`, `strftime(fmt)`, `offset_by("1mo15d")`, and more |
| `.list` (alias `.arr`) | `len`, `sum`, `min`, `max`, `mean`, `median`, `std`, `var`, `product`, `n_unique`, `l2_norm`, `sort`, `reverse`, `unique`, `flatten`, `get(i)` (negative ok), `first`, `last`, `slice`, `contains(v)`, `join(sep)`; vector ops `dot(o)`, `cosine_similarity(o)`, `l2_distance(o)` |
| `.struct` | `field(name)` |
| `.json` | `extract_string(path)` |
| `.image` | `decode()`, `to_tensor(width, height)` |

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
