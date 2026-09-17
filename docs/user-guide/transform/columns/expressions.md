# Expressions

This page covers {py:class}`Expr <batcher.plan.expr_ir.core.Expr>`, the language every column computation in Batcher is written in. An expression is a small, typed description of a computation, not a Python function. It lowers to the Rust data plane and runs over whole Arrow batches, where the optimizer can push it into a scan and the JIT can compile it, so the same code is fast on three rows or three billion.

The blocks below build on each other in order.

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

{py:obj}`bt.col(name) <batcher.col>` refers to an input column. {py:obj}`bt.lit(value) <batcher.lit>` is a constant. Both are expressions, so they compose with operators and methods.

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

Arithmetic uses `+ - * / %` and `**` (power). Reflected forms work, so a literal may lead: `2 * bt.col("x")`. Comparison uses `== != > >= < <=`. Boolean logic uses `&` (and), `|` (or), and `~` (not). Parenthesize each side, because `&` binds tighter than comparison.

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

## Conditionals

{py:obj}`bt.when(cond).then(value) <batcher.when>` builds a SQL `CASE`. Chain more {py:func}`.when(...).then(...) <batcher.when>` clauses and close with `.otherwise(default)`.

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

With exactly two branches, {py:obj}`bt.iff(cond, if_true, if_false) <batcher.iff>` is the terse form of a single `when/then/otherwise`. It is the SQL `IF`/`IFF`.

```python
out = ds.select(
    "name",
    size=bt.iff(bt.col("price") >= 20, bt.lit("big"), bt.lit("small")),
)
print(out.to_pydict())
# {'name': ['Ann', 'bob', 'CARL'], 'size': ['small', 'big', 'big']}
```

## Null handling

{py:obj}`bt.coalesce <batcher.coalesce>` returns the first non-null argument. {py:obj}`bt.nullif(a, b) <batcher.nullif>` returns null when `a == b`. {py:obj}`bt.greatest <batcher.greatest>` and {py:obj}`bt.least <batcher.least>` pick the extreme across columns. On a single expression, `.fill_null(value)`, `.is_null()`, and `.is_not_null()` apply.

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

A column where *every* value is null carries Arrow's `null` type, which records no other type at all. That is an ordinary thing to have: a left join that matched nothing, a column of all `None`, an empty aggregation, or a batch of model generations the engine could not produce all give you one. Every expression treats it the way it treats a null value, so a function over it returns nulls rather than raising, and you do not need to special-case the column before parsing it:

```python
empty = bt.from_pydict({"note": [None, None]})
out = empty.select(
    shouted=bt.col("note").str.upper(),
    width=bt.col("note").str.len_chars(),
)
print(out.to_pydict())
# {'shouted': [None, None], 'width': [None, None]}
```

That holds across the string, list, math, temporal, map, and struct methods alike, and it matches what DuckDB returns. Use `.is_null()` or a `count()` if you need to *know* the column was empty, because the result on its own cannot tell you.

A floating-point `NaN` is distinct from null. {py:obj}`bt.nanvl(value, fallback) <batcher.nanvl>` (Spark's `nanvl`) substitutes `fallback` only where `value` is `NaN`. Real numbers are left alone, and so are nulls.

```python
import math

floats = bt.from_pydict({"v": [1.0, math.nan, 3.0]})
out = floats.select(clean=bt.nanvl(bt.col("v"), bt.lit(0.0)))
print(out.to_pydict())
# {'clean': [1.0, 0.0, 3.0]}
```

## Row-wise reductions

Aggregates fold a column *down* to one value, and the `*_horizontal` functions fold *across* columns within each row. {py:func}`sum_horizontal <batcher.sum_horizontal>` treats a null as 0 and {py:func}`mean_horizontal <batcher.mean_horizontal>` skips it, and the row-wise minimum and maximum are {py:obj}`bt.least <batcher.least>` and {py:obj}`bt.greatest <batcher.greatest>`. `all_horizontal`/`any_horizontal` reduce many boolean columns into one, which is how you combine validation flags. {py:func}`count_horizontal <batcher.count_horizontal>` counts the non-null values in each row, and {py:func}`product_horizontal <batcher.product_horizontal>` multiplies them, treating a null as 1.

```python
checks = bt.from_pydict({"a": [1, 2, 3], "b": [4, 6, 6], "c": [7, 8, 9]})
out = checks.select(
    total=bt.sum_horizontal(bt.col("a"), bt.col("b"), bt.col("c")),
    smallest=bt.least(bt.col("a"), bt.col("b"), bt.col("c")),
    filled=bt.count_horizontal(bt.col("a"), bt.col("b"), bt.col("c")),
    prod=bt.product_horizontal(bt.col("a"), bt.col("b"), bt.col("c")),
    all_even=bt.all_horizontal(bt.col("a") % 2 == 0, bt.col("b") % 2 == 0),
)
print(out.to_pydict())
# {'total': [12, 16, 18], 'smallest': [1, 2, 3], 'filled': [3, 3, 3], 'prod': [28, 96, 162], 'all_even': [False, True, False]}
```

When no named `*_horizontal` helper fits, {py:func}`reduce_horizontal(fn, *exprs) <batcher.reduce_horizontal>` folds the columns left-to-right with your own binary combiner, and `fold_horizontal(acc, fn, *exprs)` does the same from an explicit seed. The combiner runs once at plan-build time on `Expr` operands, never on a row, so the fold still lowers to pure Rust:

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

Set membership, an inclusive range test, and a cast all read as methods on the column they apply to.

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

`.cast` takes a type name as a string, such as `"int64"`, `"float64"`, or `"utf8"`. {doc}`The type system <type-system>` lists every name it accepts.

## Math methods

Numeric expressions carry a full set of math methods, including `.abs()`, `.round(digits)`, `.sqrt()`, `.floor()`, `.ceil()`, `.ln()`, `.log10()`, `.log2()`, `.exp()`, the trig family (`.sin()`, `.cos()`, `.tan()`, {py:meth}`.arcsin() <batcher.plan.expr_ir.core.Expr.arcsin>`, {py:meth}`.arccos() <batcher.plan.expr_ir.core.Expr.arccos>`, {py:meth}`.arctan() <batcher.plan.expr_ir.core.Expr.arctan>`, {py:meth}`.sinh() <batcher.plan.expr_ir.core.Expr.sinh>`, {py:meth}`.cosh() <batcher.plan.expr_ir.core.Expr.cosh>`, {py:meth}`.tanh() <batcher.plan.expr_ir.core.Expr.tanh>`, {py:meth}`.cot() <batcher.plan.expr_ir.core.Expr.cot>`), `.sign()`, `.trunc()`, `.cbrt()`, `.degrees()`, and `.radians()`. {py:obj}`bt.arctan2(y, x) <batcher.arctan2>` is a top-level two-argument form.

```python
nums = bt.from_pydict({"x": [1.0, 4.0, 9.0]})
out = nums.select(
    root=bt.col("x").sqrt(),
    third=(bt.col("x") / 3).round(2),
    squared=(bt.col("x") ** 2),
)
print(out.to_pydict())
# {'root': [1.0, 2.0, 3.0], 'third': [0.33, 1.33, 3.0], 'squared': [1.0, 16.0, 81.0]}
```

A few math functions take two columns. {py:obj}`bt.gcd <batcher.gcd>` and {py:obj}`bt.lcm <batcher.lcm>` are number-theory helpers that return integers, even from float columns as above. {py:obj}`bt.hypot(a, b) <batcher.hypot>` is the Euclidean norm `sqrt(a^2 + b^2)`, a top-level two-argument form as `atan2` is.

```python
pairs = bt.from_pydict({"a": [12.0, 15.0], "b": [18.0, 20.0], "x": [3.0, 5.0], "y": [4.0, 12.0]})
out = pairs.select(
    g=bt.gcd(bt.col("a"), bt.col("b")),
    l=bt.lcm(bt.col("a"), bt.col("b")),
    dist=bt.hypot(bt.col("x"), bt.col("y")),
)
print(out.to_pydict())
# {'g': [6, 5], 'l': [36, 60], 'dist': [5.0, 13.0]}
```

{py:obj}`bt.next_after(value, toward) <batcher.next_after>` is the two-argument function to reach for when a comparison has to be *strict* in floating point. It returns the adjacent representable double, one unit in the last place toward `toward`, which is something no addition can express. For a large `value` there is no constant small enough to change it and large enough to survive rounding.

```python
edge = bt.from_pydict({"limit": [1.0, 1e16]})
out = edge.select(
    just_above=bt.next_after(bt.col("limit"), bt.lit(float("inf"))),
    naive=bt.col("limit") + bt.lit(1e-12),
)
print(out.to_pydict())
# {'just_above': [1.0000000000000002, 1.0000000000000002e+16], 'naive': [1.000000000001, 1e+16]}
```

The `naive` column is the point. Adding a small constant moved the small limit too far and the large one not at all.

`hypot` measures a flat plane. For latitude and longitude, {py:obj}`bt.great_circle_distance(lat1, lon1, lat2, lon2, unit="km") <batcher.great_circle_distance>` measures the distance over the Earth's surface. It uses the haversine formula, which keeps its precision for nearby points, and that is the case a proximity filter cares about.

```python
trips = bt.from_pydict({"alat": [51.5074], "alon": [-0.1278], "blat": [48.8566], "blon": [2.3522]})
out = trips.select(
    km=bt.great_circle_distance(bt.col("alat"), bt.col("alon"), bt.col("blat"), bt.col("blon"))
)
print(out.to_pydict())
# {'km': [343.55653488088325]}
```

The `unit` argument takes `"km"`, `"m"`, `"mi"` for statute miles, or `"nm"` for nautical miles.

{py:obj}`bt.width_bucket(value, low, high, count) <batcher.width_bucket>` assigns each value to one of `count` equal-width histogram buckets spanning `[low, high)`. The result is `1..count`, with `0` for values below the range and `count + 1` above it. Reach for it to bin a continuous column without a chain of `when`s.

```python
scores = bt.from_pydict({"score": [5.0, 55.0, 95.0, 120.0]})
out = scores.select(bucket=bt.width_bucket(bt.col("score"), bt.lit(0.0), bt.lit(100.0), 4))
print(out.to_pydict())
# {'bucket': [1.0, 3.0, 4.0, 5.0]}
```

A few Spark functions have no DuckDB twin and are top-level functions too. {py:obj}`bt.pmod(a, b) <batcher.pmod>` is the positive remainder, where `%` keeps the dividend's sign. {py:obj}`bt.bit_get(value, position) <batcher.bit_get>` reads one bit of an integer, counting from the least significant. {py:obj}`bt.elt(index, *values) <batcher.elt>` picks the `index`-th of its arguments on each row, and is null when the index is out of range. A string argument to these is a column name.

```python
nums = bt.from_pydict({"n": [-10, 7, 2], "flags": [5, 2, 3]})
out = nums.select(
    rem=bt.col("n") % 3,
    pos=bt.pmod("n", 3),
    low_bit=bt.bit_get("flags", 0),
    label=bt.elt(bt.col("flags") - 1, bt.lit("low"), bt.lit("mid"), bt.lit("high")),
)
print(out.to_pydict())
# {'rem': [-1, 1, 2], 'pos': [2, 1, 2], 'low_bit': [1, 0, 1], 'label': [None, 'low', 'mid']}
```

{py:obj}`bt.pi() <batcher.pi>` and {py:obj}`bt.e() <batcher.e>` are the two constants, folded to a literal when the plan is built.

```python
print(nums.select(tau=bt.pi() * 2, e=bt.e()).limit(1).to_pydict())
# {'tau': [6.283185307179586], 'e': [2.718281828459045]}
```

## Aggregate expressions

Aggregate methods such as `.sum()`, `.mean()`, `.min()`, `.max()`, `.median()`, `.std()`, `.var()`, `.quantile(q)`, `.count()`, and `.count_distinct()` are used inside {py:meth}`group_by(...).agg(...) <batcher.Dataset.group_by>`. {py:obj}`bt.count() <batcher.count>` is the top-level `COUNT(*)`.

```python
out = ds.group_by().agg(
    total=bt.col("price").sum(),
    avg_qty=bt.col("qty").mean(),
    rows=bt.count(),
)
print(out.to_pydict())
# {'total': [60.0], 'avg_qty': [2.0], 'rows': [3]}
```

## Inspecting an expression

The `.meta` accessor answers questions about an expression's shape without running anything, which helps when a function receives expressions it did not build. {py:meth}`output_name <batcher.plan.expr_ir.namespaces.meta._MetaNamespace.output_name>` is the column name the expression would take in a `select`, {py:meth}`root_names <batcher.plan.expr_ir.namespaces.meta._MetaNamespace.root_names>` lists the columns it reads, {py:meth}`is_column <batcher.plan.expr_ir.namespaces.meta._MetaNamespace.is_column>` tells a bare column from a computation, and {py:meth}`has_multiple_outputs <batcher.plan.expr_ir.namespaces.meta._MetaNamespace.has_multiple_outputs>` is true for a selector that expands to several columns.

```python
revenue = (bt.col("price") * bt.col("qty")).alias("revenue")
print(revenue.meta.output_name(), revenue.meta.root_names())
# revenue ['price', 'qty']
print(bt.col("price").meta.is_column(), bt.numeric().meta.has_multiple_outputs())
# True True
```

{py:meth}`tree_format <batcher.plan.expr_ir.namespaces.meta._MetaNamespace.tree_format>` draws the tree the engine is handed, one node per line.

```python
revenue.meta.tree_format()
# alias(revenue)
# └─ binary(mul)
#    ├─ col(price)
#    └─ col(qty)
```

## See also

- {doc}`Expression accessors </user-guide/transform/columns/expression-accessors>`: the methods specific to one kind of column, under `.str`, `.dt`, `.list`, `.struct`, and `.json`.
- {doc}`Expression recipes </user-guide/transform/columns/expression-recipes>`: porting from pandas or Polars, feature engineering, and curating a text corpus.
- {doc}`The type system </user-guide/transform/columns/type-system>`: what `cast` accepts, and how nulls, NaN, and mixed types behave.
- {doc}`Expressions API </api/relational/expressions>` and {doc}`Expression accessors API </api/relational/expression-accessors>`: every `Expr` method and every accessor method, enumerated.
- {doc}`Aggregations </user-guide/analyze/aggregations>` and {doc}`Window functions </user-guide/analyze/window-functions>`: where aggregate and windowed expressions are used.
- {doc}`SQL </user-guide/analyze/sql>`: the same column language, spelled as SQL.
- {doc}`Expression evaluation </architecture/deep-dives/query/expression-evaluation>`: how a tree of `Expr` nodes becomes vectorized work over an Arrow batch.
- {doc}`JIT compilation </architecture/deep-dives/query/jit-compilation>`: when the Cranelift tier compiles an arithmetic chain, and why it falls back rather than diverging.
- {doc}`/cookbook/expressions/index`: runnable recipes for the expression API.
