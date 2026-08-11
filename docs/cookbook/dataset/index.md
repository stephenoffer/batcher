# Dataset cookbook

This section holds 14 runnable recipes for the {py:class}`Dataset <batcher.Dataset>` verbs, grouped by what you are doing to the table.

Every page embeds a complete, self-contained script from the [`examples/dataset/`](https://github.com/batcher/batcher/tree/main/examples/dataset) directory. The scripts build their own in-memory data and assert on their own output, and `tests/docs/test_examples.py` runs all of them, so a page that stops matching the engine fails the suite instead of drifting.

| Group | Recipes | Covers |
|---|---|---|
| {doc}`/cookbook/dataset/verbs/index` | 5 | Joins, grouping, reshaping, and the two ways to get results out |
| {doc}`/cookbook/dataset/cleaning/index` | 4 | Deduplication, nulls, quality contracts, and reproducible splits |
| {doc}`/cookbook/dataset/inspecting/index` | 5 | The `meta` accessor, and profiling a table you were just handed |

## See also

- {doc}`/user-guide/index`: the task-oriented guide behind every verb here.
- {doc}`/api/relational/dataset`: the complete `Dataset` reference.
- {doc}`/cookbook/expressions/index`: the column language these verbs take as arguments.
- {doc}`/cookbook/data-engineering/index`: the same verbs assembled into complete pipelines.

```{toctree}
:hidden:

verbs/index
cleaning/index
inspecting/index
```
