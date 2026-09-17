# Column profiles

Reach for this before you write a data-quality rule. A guessed threshold is how a check ends up rejecting good rows, so ask the column what it holds and encode that answer as the rule.

The script profiles single columns through `ds.meta.col(...)`: bounds and midpoint, null fraction and completeness, uniqueness and duplicate count, constancy, and the low-cardinality and binary-valued hints that drive an encoder choice. It then uses the dataset-level null accounting and `is_key` to check a composite key on a table with no single unique column.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/dataset/meta_columns.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/meta_columns.py
```

## See also

- {doc}`/cookbook/dataset/cleaning/dq_contracts`: validate, fail, drop, or quarantine.
- {doc}`/cookbook/dataset/inspecting/meta_comparison`: asking about a join before running it, and reading approximate statistics.
- {doc}`/user-guide/transform/rows/transformations`: the full transformation surface these recipes draw on.
- {doc}`/api/relational/dataset`: every {py:class}`Dataset <batcher.Dataset>` method, in one reference table.
