# User guide

This section holds one guide per capability of the Dataset API, grouped by what you are doing with the data: transform it, analyze it, move it, trust it, or operate it.

Every example on every page runs as written, and the test suite executes them on each commit.

## In this section

| Group | Pages | Covers |
|---|---|---|
| {doc}`/user-guide/transform/index` | 10 | The expression language, and the operators that select, filter, order, and deduplicate |
| {doc}`/user-guide/analyze/index` | 6 | Grouping, joining, ranking, and the SQL front-end over the same engine |
| {doc}`/user-guide/moving-data/index` | 6 | Readers and writers, the storage layer underneath, and unbounded sources |
| {doc}`/user-guide/trust/index` | 4 | What counts as a valid row, and who may read which rows and columns |
| {doc}`/user-guide/operate/index` | 7 | Running the pipeline and understanding what it did |

## See also

- {doc}`../api/index`: the reference behind every method these guides use.
- {doc}`../cookbook/index`: the same operations as runnable recipes and complete pipelines.
- {doc}`../ml/index`: the model half of the pipeline, once the relational half is in place.
- {doc}`../configuration/index`: the tunables the performance and memory guides refer to.
- {doc}`/architecture/deep-dives/index`: why an operator behaves the way these pages describe.
- {doc}`../integrations/index`: connecting a specific source or sink.

```{toctree}
:hidden:

transform/index
analyze/index
moving-data/index
trust/index
operate/index
```
