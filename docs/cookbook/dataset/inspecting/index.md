# Inspecting a dataset

What a table already knows about itself. `ds.meta` asks Kyber whether the answer is provable from the plan, a Parquet footer or a recorded sketch, and runs the query itself when it isn't. Either route returns the same answer, so these are safe to branch on. The deliberate exception is `ds.meta.approx`, which never executes at all. It reads sketches, so it is approximate, and where the source carries none it returns `None` rather than guess.

Each page embeds a complete, self-contained script that builds its own in-memory data and asserts on its own output, so a page that stops matching the engine fails the suite instead of drifting.

| Recipe | What it shows |
|---|---|
| {doc}`/cookbook/dataset/inspecting/meta_schema` | A dataset's shape, without executing it |
| {doc}`/cookbook/dataset/inspecting/meta_columns` | Bounds, uniqueness, nulls, and constancy on one column |
| {doc}`/cookbook/dataset/inspecting/meta_predicates` | Cheap yes/no questions, and the column-check shorthands |
| {doc}`/cookbook/dataset/inspecting/meta_comparison` | Sizing a join before running it, and approximate statistics |
| {doc}`/cookbook/dataset/inspecting/profiling` | The first pass over a table you have just been handed |

```{toctree}
:hidden:

meta_schema
meta_columns
meta_predicates
meta_comparison
profiling
```
