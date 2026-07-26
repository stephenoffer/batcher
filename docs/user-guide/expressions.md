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

{py:obj}`bt.col(name) <batcher.col>` refers to an input column. {py:obj}`bt.lit(value) <batcher.lit>` is a constant. Both are
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

{py:obj}`bt.when(cond).then(value) <batcher.when>` builds a SQL `CASE`. Chain more `.when(...).then(...)`
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

With exactly two branches, {py:obj}`bt.iff(cond, if_true, if_false) <batcher.iff>`
is the terse form of a single `when/then/otherwise`. It is the SQL `IF`/`IFF`.

```python
out = ds.select(
    "name",
    size=bt.iff(bt.col("price") >= 20, bt.lit("big"), bt.lit("small")),
)
print(out.to_pydict())
# {'name': ['Ann', 'bob', 'CARL'], 'size': ['small', 'big', 'big']}
```

## Null handling

{py:obj}`bt.coalesce <batcher.coalesce>` returns the first non-null argument. {py:obj}`bt.nullif(a, b) <batcher.nullif>` returns null
when `a == b`. {py:obj}`bt.greatest <batcher.greatest>` and {py:obj}`bt.least <batcher.least>` pick the extreme across columns. On a
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

A floating-point `NaN` is distinct from null. {py:obj}`bt.nanvl(value, fallback) <batcher.nanvl>`
(Spark's `nanvl`) substitutes `fallback` only where `value` is `NaN`. Real numbers
are left alone, and so are nulls.

```python
import math

floats = bt.from_pydict({"v": [1.0, math.nan, 3.0]})
out = floats.select(clean=bt.nanvl(bt.col("v"), bt.lit(0.0)))
print(out.to_pydict())
# {'clean': [1.0, 0.0, 3.0]}
```

## Row-wise (horizontal) reductions

Aggregates fold a column *down* to one value; the `*_horizontal` functions fold
*across* columns within each row. `sum_horizontal`/`mean_horizontal` combine numeric
columns (nulls treated as 0 / skipped), and `min_horizontal`/`max_horizontal` are the
Polars-named row-wise `least`/`greatest`. `all_horizontal`/`any_horizontal` reduce
many boolean columns into one, which is how you combine validation flags.
`count_horizontal` counts the non-null values in each row and `product_horizontal`
multiplies them (nulls treated as 1).

```python
checks = bt.from_pydict({"a": [1, 2, 3], "b": [4, 6, 6], "c": [7, 8, 9]})
out = checks.select(
    total=bt.sum_horizontal(bt.col("a"), bt.col("b"), bt.col("c")),
    smallest=bt.min_horizontal(bt.col("a"), bt.col("b"), bt.col("c")),
    filled=bt.count_horizontal(bt.col("a"), bt.col("b"), bt.col("c")),
    prod=bt.product_horizontal(bt.col("a"), bt.col("b"), bt.col("c")),
    all_even=bt.all_horizontal(bt.col("a") % 2 == 0, bt.col("b") % 2 == 0),
)
print(out.to_pydict())
# {'total': [12, 16, 18], 'smallest': [1, 2, 3], 'filled': [3, 3, 3], 'prod': [28, 96, 162], 'all_even': [False, True, False]}
```

When no named `*_horizontal` helper fits, `reduce_horizontal(fn, *exprs)` folds the
columns left-to-right with your own binary combiner, and `fold_horizontal(acc, fn,
*exprs)` does the same from an explicit seed. The combiner runs once at plan-build
time on `Expr` operands, never on a row, so the fold still lowers to pure Rust:

```python
cols = [bt.col("a"), bt.col("b"), bt.col("c")]
out = checks.select(
    manual_sum=bt.reduce_horizontal(lambda x, y: x + y, *cols),
    sum_sq=bt.fold_horizontal(bt.lit(0), lambda s, x: s + x * x, *cols),
)
print(out.to_pydict())
# {'manual_sum': [12, 16, 18], 'sum_sq': [66, 104, 126]}
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
{py:obj}`bt.atan2(y, x) <batcher.atan2>` is a top-level two-argument form.

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

A few math functions take two columns. {py:obj}`bt.gcd <batcher.gcd>` and
{py:obj}`bt.lcm <batcher.lcm>` are integer number-theory helpers;
{py:obj}`bt.hypot(a, b) <batcher.hypot>` is the Euclidean norm `sqrt(a^2 + b^2)`, a
top-level two-argument form as `atan2` is.

```python
pairs = bt.from_pydict({"a": [12.0, 15.0], "b": [18.0, 20.0], "x": [3.0, 5.0], "y": [4.0, 12.0]})
out = pairs.select(
    g=bt.gcd(bt.col("a"), bt.col("b")),
    l=bt.lcm(bt.col("a"), bt.col("b")),
    dist=bt.hypot(bt.col("x"), bt.col("y")),
)
print(out.to_pydict())
# {'g': [6.0, 5.0], 'l': [36.0, 60.0], 'dist': [5.0, 13.0]}
```

{py:obj}`bt.next_after(value, toward) <batcher.next_after>` is the two-argument
function to reach for when a comparison has to be *strict* in floating point. It returns
the adjacent representable double, one unit in the last place toward `toward`, which is
something no addition can express: for a large `value` there is no constant small enough
to change it and large enough to survive rounding.

```python
edge = bt.from_pydict({"limit": [1.0, 1e16]})
out = edge.select(
    just_above=bt.next_after(bt.col("limit"), bt.lit(float("inf"))),
    naive=bt.col("limit") + bt.lit(1e-12),
)
print(out.to_pydict())
# {'just_above': [1.0000000000000002, 1.0000000000000002e+16], 'naive': [1.000000000001, 1e+16]}
```

The `naive` column is the point: adding a small constant moved the small limit too far
and the large one not at all.

`hypot` measures a flat plane. For latitude and longitude,
{py:obj}`bt.great_circle_distance(lat1, lon1, lat2, lon2, unit="km") <batcher.great_circle_distance>`
measures the distance over the Earth's surface. It uses the haversine formula, which keeps
its precision for nearby points, and that is the case a proximity filter cares about.

```python
trips = bt.from_pydict(
    {"alat": [51.5074], "alon": [-0.1278], "blat": [48.8566], "blon": [2.3522]}
)
out = trips.select(
    km=bt.great_circle_distance(
        bt.col("alat"), bt.col("alon"), bt.col("blat"), bt.col("blon")
    )
)
print(out.to_pydict())
# {'km': [343.55653488088325]}
```

The `unit` argument takes `"km"`, `"m"`, `"mi"` (statute miles), or `"nm"` (nautical
miles).

{py:obj}`bt.width_bucket(value, low, high, count) <batcher.width_bucket>` assigns each
value to one of `count` equal-width histogram buckets spanning `[low, high)`. The
result is `1..count`, with `0` for values below the range and `count + 1` above it.
Reach for it to bin a continuous column without a chain of `when`s.

```python
scores = bt.from_pydict({"score": [5.0, 55.0, 95.0, 120.0]})
out = scores.select(bucket=bt.width_bucket(bt.col("score"), bt.lit(0.0), bt.lit(100.0), 4))
print(out.to_pydict())
# {'bucket': [1.0, 3.0, 4.0, 5.0]}
```

## Aggregate expressions

Aggregate methods such as `.sum()`, `.mean()`, `.min()`, `.max()`, `.median()`,
`.std()`, `.var()`, `.quantile(q)`, `.count()`, and `.n_unique()` are used inside
`group_by(...).agg(...)`. {py:obj}`bt.count() <batcher.count>` is the top-level `COUNT(*)`.

```python
out = ds.group_by().agg(
    total=bt.col("price").sum(),
    avg_qty=bt.col("qty").mean(),
    rows=bt.count(),
)
print(out.to_pydict())
# {'total': [60.0], 'avg_qty': [2.0], 'rows': [3]}
```

## Next steps

The rest of the expression language continues on two more pages:

- [Expression accessors](expression-accessors.md): the `.str`, `.dt`, `.list`,
  `.struct`, and `.json` namespaces, which hold the methods specific to one kind of
  column.
- [Expression recipes](expression-recipes.md): porting from pandas or Polars, feature
  engineering, and curating a text corpus.

Then, for reference and for where expressions are used:

- [Expressions API](../api/expressions.md) and
  [Expression accessors API](../api/expression-accessors.md): every `Expr` method and
  every accessor method, enumerated.
- [Aggregations](aggregations.md) and [Window functions](window-functions.md): where
  aggregate and windowed expressions are used.
- [SQL](sql.md): the same column language, spelled as SQL.
- [Transformations](transformations.md): where expressions are applied to a Dataset.

And for what happens to an expression after you write it:

- [Expression evaluation](../deep-dives/expression-evaluation.md): how a tree of `Expr`
  nodes becomes vectorized work over an Arrow batch.
- [JIT compilation](../deep-dives/jit-compilation.md): when the Cranelift tier compiles an
  arithmetic chain, and why it silently falls back rather than diverging.
