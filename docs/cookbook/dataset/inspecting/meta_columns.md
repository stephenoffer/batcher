# Column profiles

Reach for this before you write a data-quality rule. A guessed threshold is how a check ends up rejecting good rows. Ask the column what it holds, then encode that answer as the rule.

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
