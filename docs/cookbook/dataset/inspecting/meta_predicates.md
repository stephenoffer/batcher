# Cheap data checks

"Does this table contain any bad rows?" is a cheaper question than "how many bad rows does it contain?", and these accessors ask the cheap one. `any_match` and its siblings answer from metadata when the answer is provable and otherwise probe for a single row rather than counting them all.

The script asks existence questions with `any_match`, `all_match`, and `none_match`, uses `count_where` when the number matters, and reads the `check` shorthands as sentences such as "all amounts are between 1 and 100". It shows that sort order is tracked on the plan, and it ends with the gate these exist for: refuse to proceed when a contract is violated.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/dataset/meta_predicates.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/meta_predicates.py
```

## See also

- {doc}`/cookbook/dataset/inspecting/meta_comparison`: asking about a join before running it, and reading approximate statistics.
- {doc}`/cookbook/dataset/inspecting/meta_schema`: asking about a dataset's shape without executing it.
- {doc}`/user-guide/transform/rows/transformations`: the full transformation surface these recipes draw on.
- {doc}`/api/relational/dataset`: every {py:class}`Dataset <batcher.Dataset>` method, in one reference table.
