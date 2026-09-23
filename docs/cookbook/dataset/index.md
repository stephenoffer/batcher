# Dataset cookbook

A {py:class}`Dataset <batcher.Dataset>` is a lazy plan, and its verbs are how you build one: join it, group it, reshape it, clean it, and ask it questions. These 14 recipes cover those verbs, grouped by what you are doing to the table, and each one is a complete script you can copy and run.

Every page embeds a script from the [`examples/dataset/`](https://github.com/stephenoffer/batcher/tree/main/examples/dataset) directory. The scripts build their own in-memory data and assert on their own output, and [`tests/docs/test_examples.py`](https://github.com/stephenoffer/batcher/blob/main/tests/docs/test_examples.py) runs all of them, so a recipe that stops matching the engine fails the suite instead of drifting.

If you are new to Batcher, start with {doc}`joins </cookbook/dataset/verbs/joins>` and {doc}`grouping </cookbook/dataset/verbs/grouping>`. If someone has just handed you a table, start with {doc}`profiling </cookbook/dataset/inspecting/profiling>`.

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
