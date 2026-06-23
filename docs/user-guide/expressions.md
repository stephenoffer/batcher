# Expressions

Column work in Batcher is expressed with the `Expr` API, never with Python loops.
An expression is a small, typed description of a computation. It lowers to the
Rust data plane and runs over Arrow batches, so the same code is fast on three
rows or three billion.

Every example on this page runs against the engine. Blocks share one namespace
and execute in order.

```python
import batcher as bt

ds = bt.from_pydict(
    {
        "name": ["Ann", "bob", "CARL"],
        "price": [10.0, 20.0, 30.0],
        "qty": [1, 2, 3],
    }
)
```

## Columns and literals

`bt.col(name)` refers to an input column. `bt.lit(value)` is a constant. Both are
expressions, so they compose with operators and methods.

```python
out = ds.select(
    "name",
    revenue=bt.col("price") * bt.col("qty"),
    discounted=bt.col("price") * bt.lit(0.9),
)
print(out.to_pydict())
# {'name': ['Ann', 'bob', 'CARL'], 'revenue': [10.0, 40.0, 90.0], 'discounted': [9.0, 18.0, 27.0]}
```

## Arithmetic, comparison, and boolean operators

Arithmetic uses `+ - * / %` and `**` (power). Reflected forms work, so a literal
may lead: `2 * bt.col("x")`. Comparison uses `== != > >= < <=`. Boolean logic uses
`&` (and), `|` (or), and `~` (not); parenthesize each side because `&` binds
tighter than comparison.

```python
out = ds.select(
    "name",
    cheap=(bt.col("price") < 25),
    cheap_and_small=((bt.col("price") < 25) & (bt.col("qty") <= 1)),
    not_cheap=~(bt.col("price") < 25),
)
print(out.to_pydict())
# {'name': ['Ann', 'bob', 'CARL'], 'cheap': [True, True, False], 'cheap_and_small': [True, False, False], 'not_cheap': [False, False, True]}
```

## Conditionals: when / then / otherwise

`bt.when(cond).then(value)` builds a SQL `CASE`. Chain more `.when(...).then(...)`
clauses and close with `.otherwise(default)`.

```python
out = ds.select(
    "name",
    tier=bt.when(bt.col("price") >= 30)
    .then(bt.lit("high"))
    .when(bt.col("price") >= 15)
    .then(bt.lit("mid"))
    .otherwise(bt.lit("low")),
)
print(out.to_pydict())
# {'name': ['Ann', 'bob', 'CARL'], 'tier': ['low', 'mid', 'high']}
```

## Null handling

`bt.coalesce` returns the first non-null argument. `bt.nullif(a, b)` returns null
when `a == b`. `bt.greatest` and `bt.least` pick the extreme across columns. On a
single expression, `.fill_null(value)`, `.is_null()`, and `.is_not_null()` apply.

```python
nulls = bt.from_pydict({"a": [1, None, 3], "b": [9, 8, 7]})
out = nulls.select(
    first_present=bt.coalesce(bt.col("a"), bt.col("b")),
    filled=bt.col("a").fill_null(0),
    bigger=bt.greatest(bt.col("a").fill_null(0), bt.col("b")),
)
print(out.to_pydict())
# {'first_present': [1, 8, 3], 'filled': [1, 0, 3], 'bigger': [9, 8, 7]}
```

## Membership, ranges, and casts

```python
out = ds.select(
    "name",
    in_set=bt.col("qty").is_in([1, 3]),
    in_range=bt.col("price").between(15.0, 30.0),
    qty_f=bt.col("qty").cast("float64"),
)
print(out.to_pydict())
# {'name': ['Ann', 'bob', 'CARL'], 'in_set': [True, False, True], 'in_range': [False, True, True], 'qty_f': [1.0, 2.0, 3.0]}
```

`.cast` takes an Arrow type name as a string (for example `"int64"`, `"float64"`,
`"utf8"`).

## Math methods

Numeric expressions carry a full set of math methods, including `.abs()`,
`.round(digits)`, `.sqrt()`, `.pow(e)`, `.floor()`, `.ceil()`, `.ln()`,
`.log10()`, `.log2()`, `.exp()`, the trig family (`.sin()`, `.cos()`, `.tan()`,
`.asin()`, `.acos()`, `.atan()`, `.sinh()`, `.cosh()`, `.tanh()`, `.cot()`),
`.sign()`, `.trunc()`, `.cbrt()`, `.degrees()`, and `.radians()`.
`bt.atan2(y, x)` is a top-level two-argument form.

```python
nums = bt.from_pydict({"x": [1.0, 4.0, 9.0]})
out = nums.select(
    root=bt.col("x").sqrt(),
    third=(bt.col("x") / 3).round(2),
    squared=bt.col("x").pow(2),
)
print(out.to_pydict())
# {'root': [1.0, 2.0, 3.0], 'third': [0.33, 1.33, 3.0], 'squared': [1.0, 16.0, 81.0]}
```

## String accessor: .str

The `.str` namespace covers casing, trimming, search, slicing, padding, and
encoding. Search methods such as `contains`, `starts_with`, and `like` are
case-sensitive; use `ilike` for case-insensitive matching.

```python
out = ds.select(
    upper=bt.col("name").str.upper(),
    length=bt.col("name").str.len(),
    has_a=bt.col("name").str.ilike("%a%"),
    first_two=bt.col("name").str.left(2),
)
print(out.to_pydict())
# {'upper': ['ANN', 'BOB', 'CARL'], 'length': [3, 3, 4], 'has_a': [True, False, True], 'first_two': ['An', 'bo']}
```

Other `.str` methods include `lower`, `trim`, `lstrip`, `rstrip`, `reverse`,
`substr`, `right`, `repeat`, `lpad`, `rpad`, `position`, `split`, `regexp_matches`,
`regexp_replace`, `regexp_extract`, `replace`, `initcap`, `hex`, `base64`,
`from_base64`, `unhex`, and `translate`.

## Datetime accessor: .dt

The `.dt` namespace extracts calendar parts from timestamp columns.

```python
import datetime

events = bt.from_pydict(
    {"ts": [datetime.datetime(2024, 1, 15, 9, 30), datetime.datetime(2024, 6, 1, 18, 0)]}
)
out = events.select(
    year=bt.col("ts").dt.year(),
    month=bt.col("ts").dt.month(),
    day_name=bt.col("ts").dt.dayname(),
)
print(out.to_pydict())
# {'year': [2024, 2024], 'month': [1, 6], 'day_name': ['Monday', 'Saturday']}
```

Also available: `day`, `hour`, `minute`, `second`, `quarter`, `week`,
`dayofweek`, `dayofyear`, `epoch`, `monthname`, `isodow`, `century`, `decade`,
`millennium`, `last_day`, and `truncate(unit)`.

## List accessor: .list

The `.list` namespace (aliased `.arr`) reduces and reshapes list-typed columns.

```python
lists = bt.from_pydict({"tags": [["x", "y"], ["z"], ["a", "b", "c"]]})
out = lists.select(
    n=bt.col("tags").list.len(),
    joined=bt.col("tags").list.join("-"),
    first=bt.col("tags").list.get(0),
)
print(out.to_pydict())
# {'n': [2, 1, 3], 'joined': ['x-y', 'z', 'a-b-c'], 'first': ['x', 'z', 'a']}
```

Numeric lists support reductions: `sum`, `min`, `max`, `mean`, `median`, `std`,
`var`, `product`, `n_unique`, `arg_min`, `arg_max`. Structural methods include
`sort`, `reverse`, `unique`, `slice`, and `contains`.

## Struct accessor: .struct

`.struct.field(name)` pulls a field out of a struct column.

```python
import pyarrow as pa

points = bt.from_arrow(pa.table({"p": pa.array([{"x": 1, "y": 2}, {"x": 3, "y": 4}])}))
out = points.select(x=bt.col("p").struct.field("x"), y=bt.col("p").struct.field("y"))
print(out.to_pydict())
# {'x': [1, 3], 'y': [2, 4]}
```

## JSON accessor: .json

`.json.extract_string(path)` reads a string value from a JSON text column using a
JSONPath expression.

```python
docs = bt.from_pydict({"doc": ['{"a": {"b": "hi"}}', '{"a": {"b": "bye"}}']})
out = docs.select(value=bt.col("doc").json.extract_string("$.a.b"))
print(out.to_pydict())
# {'value': ['hi', 'bye']}
```

## Aggregate expressions

Aggregate methods such as `.sum()`, `.mean()`, `.min()`, `.max()`, `.median()`,
`.std()`, `.var()`, `.quantile(q)`, `.count()`, and `.n_unique()` are used inside
`group_by(...).agg(...)`. `bt.count()` is the top-level `COUNT(*)`.

```python
out = ds.group_by().agg(
    total=bt.col("price").sum(),
    avg_qty=bt.col("qty").mean(),
    rows=bt.count(),
)
print(out.to_pydict())
# {'total': [60.0], 'avg_qty': [2.0], 'rows': [3]}
```
