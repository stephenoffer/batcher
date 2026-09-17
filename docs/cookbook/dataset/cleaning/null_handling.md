# Null handling

Null propagation is where a pipeline changes its answer without telling you. Decide per column whether a missing value means "unknown" (leave it), "zero" (fill it), or "this row is unusable" (drop it). Never let the default decide for you.

The script counts nulls first, then drops rows with any null and with a null in one named column, and fills with one value and per column. Its last assertion is the reason the page exists: a mean over four rows with two nulls is 20.0, and the same mean after `fill_null(0)` is 10.0.

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
- {doc}`/api/relational/dataset`: every {py:class}`Dataset <batcher.Dataset>` method, in one reference table.
