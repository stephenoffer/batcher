# Expressions run in Rust

You describe column work in Python, and Rust does it. An {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` built from {py:obj}`bt.col(...) <batcher.col>` and {py:obj}`bt.lit(...) <batcher.lit>` is a *description* of a computation, not a loop. The engine evaluates it over whole Arrow batches with vectorized kernels, and compiles numeric filters and projections to native code with Cranelift.

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3, 4]})

total = bt.col("x") * bt.lit(10)
print(ds.select(scaled=total).to_pydict())
# {'scaled': [10, 20, 30, 40]}
```

Because the expression ships inside the plan, the optimizer can push a filter into a Parquet scan, drop a column nobody reads, or fold a constant before any data moves. A Python lambda is opaque to it.

The difference is easiest to see stage by stage, with the same multiply-by-ten written both ways:

![Two columns compared stage by stage. On the left, a Python for loop over rows becomes a Python function that is opaque to the plan, so the optimizer can't see inside it, and the Python interpreter calls it once per row, leaving nothing to push down, prune, or fold. On the right, bt.col("x") * bt.lit(10) builds an expression tree that is shipped in the plan, the optimizer can push filters, prune columns, and fold constants, and the work runs in Rust over whole Arrow batches with vectorized kernels and Cranelift for numeric work, so no part of it walks rows in Python.](/_static/diagrams/expression_vs_row_loop.svg)

## Conditionals and reuse

An expression is a value. Build it once and reuse it in `select`, `with_columns`, `filter`, or an aggregate:

```python
big = bt.col("x") > 2
print(ds.filter(big).to_pydict())
# {'x': [3, 4]}
print(ds.select(big=big).to_pydict())
# {'big': [False, False, True, True]}
```

Conditionals read like SQL's `CASE WHEN`:

```python
grades = bt.from_pydict({"score": [91, 72, 55]})
grade = bt.when(bt.col("score") >= 80).then(bt.lit("A")).otherwise(bt.lit("B"))
print(grades.select(grade=grade).to_pydict())
# {'grade': ['A', 'B', 'B']}
```

## Nulls, casts, and windows

Null handling and casts are methods on the expression:

```python
n = bt.from_pydict({"v": [1, None, 3]})
print(n.select(v=bt.col("v").fill_null(0), missing=bt.col("v").is_null()).to_pydict())
# {'v': [1, 0, 3], 'missing': [False, True, False]}
print(ds.select(f=bt.col("x").cast("float64")).to_pydict())
# {'f': [1.0, 2.0, 3.0, 4.0]}
```

Any aggregate becomes a window function with `.over(...)`, keeping every row:

```python
sales = bt.from_pydict({"g": ["a", "b", "a"], "v": [1, 2, 3]})
print(sales.with_columns(group_total=bt.col("v").sum().over(partition_by="g")).to_pydict())
# {'g': ['a', 'b', 'a'], 'v': [1, 2, 3], 'group_total': [4, 2, 4]}
```

## Accessors match the column type

Each column type has its own accessor namespace, so the vocabulary matches the data: `.str` for strings, `.dt` for dates and times, `.list` for arrays, and `.struct`, `.json`, and `.map` for nested data. Media columns get `.image`, `.audio`, and `.video`.

```python
import datetime

users = bt.from_pydict(
    {
        "email": ["Ann@Example.com"],
        "signup_ts": [datetime.datetime(2024, 3, 1)],
        "tags": [["ai", "data"]],
    }
)
print(
    users.select(
        email=bt.col("email").str.lower(),
        year=bt.col("signup_ts").dt.year(),
        likes_ai=bt.col("tags").list.contains("ai"),
    ).to_pydict()
)
# {'email': ['ann@example.com'], 'year': [2024], 'likes_ai': [True]}
```

## When you need your own Python

Some work has no expression, such as calling a model or a library you already have. {py:meth}`map_batches <batcher.Dataset.map_batches>` runs your function on a whole Arrow batch at a time, so the call overhead is paid once per batch rather than once per row:

```python
import pyarrow.compute as pc


def add_tax(batch):
    return batch.append_column("tax", pc.multiply(batch.column("x"), 0.5))


print(ds.map_batches(add_tax).to_pydict())
# {'x': [1, 2, 3, 4], 'tax': [0.5, 1.0, 1.5, 2.0]}
```

Reach for an expression first: it stays visible to the optimizer, and a function does not.

## See also

- {doc}`/user-guide/transform/columns/expressions`: the full expression surface, with nulls and casting.
- {doc}`/user-guide/transform/columns/expression-accessors`: the {py:class}`.str <batcher.plan.expr_ir.namespaces.strings._StrNamespace>`, {py:class}`.dt <batcher.plan.expr_ir.namespaces.temporal._DtNamespace>`, {py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>`, and {py:class}`.json <batcher.plan.expr_ir.namespaces.collections._JsonNamespace>` namespaces.
- {doc}`/user-guide/transform/columns/udfs`: batch functions and the `@bt.udf` decorator.
- {doc}`/api/relational/expressions`: every `Expr` method in one reference.
