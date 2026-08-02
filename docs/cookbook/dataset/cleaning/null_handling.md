# Null handling

Null propagation is where a pipeline changes its answer without telling you. Decide per column whether a missing value means "unknown" (leave it), "zero" (fill it), or "this row is unusable" (drop it). Never let the default decide for you.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/dataset/null_handling.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/null_handling.py
```

## See also

- {doc}`/cookbook/dataset/inspecting/meta_schema`: asking about a dataset's shape without executing it.
- {doc}`/cookbook/dataset/inspecting/profiling`: profiling a table you have just been handed.
- {doc}`/user-guide/transform/rows/transformations`: the full transformation surface these recipes draw on.
- {doc}`/api/relational/dataset`: every `Dataset` method, in one reference table.
