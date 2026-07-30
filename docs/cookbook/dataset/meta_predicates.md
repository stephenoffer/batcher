# Cheap yes/no questions about the data, and the column-check shorthands

These short-circuit. ``any_match`` stops at the first matching row rather than counting them all, which makes "does this table contain any bad rows?" much cheaper than "how many bad rows does it contain?".

The whole script, executed on every test run:

```{literalinclude} ../../../examples/dataset/meta_predicates.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/meta_predicates.py
```

## See also

- {doc}`meta_comparison`: asking about a join before running it, and reading approximate statistics.
- {doc}`meta_schema`: asking about a dataset's shape without executing it.
- {doc}`../../user-guide/transformations`: the full transformation surface these recipes draw on.
- {doc}`../../api/dataset`: every `Dataset` method, in one reference table.
