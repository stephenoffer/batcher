# Column profiles

This is the accessor to reach for before writing a data-quality rule. Rather than guessing a threshold, ask the column what it actually contains, then encode the answer as a check.

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

- {doc}`/cookbook/dataset/verbs/joins`: join types, key spellings, and the as-of join for time series.
- {doc}`/cookbook/dataset/inspecting/meta_comparison`: asking about a join before running it, and reading approximate statistics.
- {doc}`/user-guide/transform/rows/transformations`: the full transformation surface these recipes draw on.
- {doc}`/api/relational/dataset`: every `Dataset` method, in one reference table.
