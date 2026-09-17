# Profiling a new table

The first thing to do with unfamiliar data is measure it, not query it. These are the one-liners that answer "what is in here" before you write a single business rule.

The script runs them in the order you would reach for them: `describe` for per-column statistics, null counts, exact and approximate distinct counts and quantiles, `value_counts` for a categorical, correlation and covariance matrices, `drop_constant_columns`, an emptiness check, and `glimpse` and `info` for a compact overview.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/dataset/profiling.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/profiling.py
```

## See also

- {doc}`/cookbook/dataset/cleaning/null_handling`: dropping, filling, and counting missing values.
- {doc}`/cookbook/dataset/inspecting/meta_columns`: bounds, uniqueness, nulls, and constancy.
- {doc}`/user-guide/transform/rows/transformations`: the full transformation surface these recipes draw on.
- {doc}`/api/relational/dataset`: every {py:class}`Dataset <batcher.Dataset>` method, in one reference table.
