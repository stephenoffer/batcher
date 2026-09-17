# Expressions run in Rust

You describe column work in Python, and Rust does it. An {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` built from {py:obj}`bt.col(...) <batcher.col>` and {py:obj}`bt.lit(...) <batcher.lit>` is a *description* of a computation, not a loop. When the plan runs, the Rust data plane evaluates it over whole Arrow batches with vectorized kernels. Numeric filters and projections go further: the engine compiles them to native code with Cranelift once per query shape, and falls back to the interpreter for anything the compiler doesn't cover. No part of it walks rows in Python.

```python
import batcher as bt

ds = bt.from_pydict({"x": [1, 2, 3, 4]})

total = bt.col("x") * bt.lit(10)
print(ds.select(scaled=total).to_pydict())
# {'scaled': [10, 20, 30, 40]}
```

Operators such as `+`, `==`, and `&`, methods such as `.sum()` and `.cast(...)`, and every accessor namespace build up the same expression tree. Python ships that tree to the engine as part of the plan, and the optimizer can reason about it: push a filter into a Parquet scan, drop a column nobody reads, or fold a constant before any data moves. None of that is possible with a Python lambda, which the optimizer can't see inside.

The difference is easiest to see stage by stage, with the same multiply-by-ten written both ways:

![Two columns compared stage by stage. On the left, a Python for loop over rows becomes a Python function that is opaque to the plan, so the optimizer can't see inside it, and the Python interpreter calls it once per row, leaving nothing to push down, prune, or fold. On the right, bt.col("x") * bt.lit(10) builds an expression tree that is shipped in the plan, the optimizer can push filters, prune columns, and fold constants, and the work runs in Rust over whole Arrow batches with vectorized kernels and Cranelift for numeric work, so no part of it walks rows in Python.](/_static/diagrams/expression_vs_row_loop.svg)

## Conditionals and reuse

An expression is a value, so you build it once and reuse it in `select`, `with_columns`, `filter`, or an aggregate. Conditionals read like SQL's `CASE WHEN`:

```python
grades = bt.from_pydict({"score": [91, 72, 55]})
grade = bt.when(bt.col("score") >= 80).then(bt.lit("A")).otherwise(bt.lit("B"))
print(grades.select(grade=grade).to_pydict())
# {'grade': ['A', 'B', 'B']}
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

Some work has no expression, such as calling a model or a library you already have. {py:meth}`map_batches <batcher.Dataset.map_batches>` runs your function on a whole batch at a time, as a PyArrow, pandas, or NumPy batch, so the per-call overhead is paid once per batch rather than once per row. Reach for it after checking the expression surface, because an expression stays visible to the optimizer and a function does not.

## See also

- {doc}`/user-guide/transform/columns/expressions`: the full expression surface, with nulls and casting.
- {doc}`/user-guide/transform/columns/expression-accessors`: the {py:class}`.str <batcher.plan.expr_ir.namespaces.strings._StrNamespace>`, {py:class}`.dt <batcher.plan.expr_ir.namespaces.temporal._DtNamespace>`, {py:class}`.list <batcher.plan.expr_ir.namespaces.collections._ListNamespace>`, and {py:class}`.json <batcher.plan.expr_ir.namespaces.collections._JsonNamespace>` namespaces.
- {doc}`/user-guide/transform/columns/udfs`: batch functions and the `@bt.udf` decorator.
- {doc}`/api/relational/expressions`: every `Expr` method in one reference.
