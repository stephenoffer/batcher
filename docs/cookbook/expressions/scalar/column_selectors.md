# Column selectors

A selector is an `Expr` leaf standing for *every* matching column, so "round every float" is one expression that keeps working when a column is added. Spelling out names is how a pipeline silently stops covering a new column.

The script selects columns by type family and by name pattern, takes everything or everything except, and combines selectors with `|`, `&`, `-`, and `~`. The payoff is a single `with_columns` that rounds every matched column in place and leaves the rest untouched.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/expressions/column_selectors.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/expressions/column_selectors.py
```


## Resolving a selector against a schema

A selector is lazy: {py:func}`bt.numeric() <batcher.numeric>` describes a rule, not a column list, and it expands when
the plan is built. {py:meth}`Selector.matched_columns <batcher.Selector>` runs that
expansion early, which is what you want when a caller must report or validate the choice
before any query runs.

```python
import batcher as bt

ds = bt.from_pydict({"a": [1], "b": [2.0], "label": ["x"]})
print(bt.starts_with("a").matched_columns(ds.columns, None))
```

A name-based selector needs only the column names, so `None` is an acceptable schema. A
type-based selector such as {py:func}`bt.numeric() <batcher.numeric>` needs the types too, and resolving one without
them matches nothing rather than raising.

## See also

- {doc}`/cookbook/expressions/scalar/aggregates`: counts, positions, quantiles, and approximations.
- {doc}`/cookbook/expressions/scalar/conditionals`: when/then/otherwise, and the SQL null helpers.
- {doc}`/user-guide/transform/columns/expressions`: what an expression is, and how it is evaluated.
- {doc}`/api/relational/expressions`: the complete {py:class}`Expr <batcher.plan.expr_ir.core.Expr>` reference.
