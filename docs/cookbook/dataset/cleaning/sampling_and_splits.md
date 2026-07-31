# Sampling and splits

Every one of these takes a seed, because an unseeded split is a split you cannot reproduce when the result looks wrong. ``stratified_split`` preserves class balance; a plain random split does not, and on an imbalanced problem that matters.

The whole script, executed on every test run:

```{literalinclude} ../../../../examples/dataset/sampling_and_splits.py
:language: python
:linenos:
```

Run it yourself:

```bash
python examples/dataset/sampling_and_splits.py
```

## See also

- {doc}`/cookbook/dataset/verbs/reshaping`: pivot, unpivot, explode, unnest, and set operations.
- {doc}`/cookbook/dataset/verbs/sql_interface`: SQL over the same engine, and mixing SQL with DataFrame verbs.
- {doc}`/user-guide/transform/transformations`: the full transformation surface these recipes draw on.
- {doc}`/api/relational/dataset`: every `Dataset` method, in one reference table.
