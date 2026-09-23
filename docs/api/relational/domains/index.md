# Domain function libraries

This section is the reference for the three function libraries that answer questions the relational verbs have no spelling for: geometry on the Earth's surface, rigid-body motion in a robot's coordinate frames, and connectivity over an edge table.

All three are expressions, not a separate engine. A spatial predicate is an `Expr` like any other, so it composes with a filter, pushes into a scan, joins, and runs on the same Rust data plane as `+`. That is why a geospatial join is a join rather than a library call over a collected result, and why a graph algorithm sees a partitioned edge table rather than an in-memory adjacency list.

| Page | Covers |
|---|---|
| {doc}`geospatial` | Every `ST_*` function: geometry, predicates, measures, and grids |
| {doc}`spatial` | Rotations, poses, and coordinate frames for robotics and autonomous driving |
| {doc}`graph` | Graph analytics and graph-ML features over an edge table |

Each one has a guide behind it: {doc}`/user-guide/analyze/domains/geospatial`, {doc}`/user-guide/analyze/domains/robotics`, and {doc}`/user-guide/analyze/domains/graphs`.

## See also

- {doc}`/api/relational/functions`: the general scalar, aggregate, and window functions these sit beside.
- {doc}`/api/relational/sql`: every function here is callable from SQL under the same name.
- {doc}`/api/accessors/sequence`: the `.seq` namespace, the fourth domain library, which attaches to a column rather than standing alone.

```{toctree}
:hidden:

geospatial
spatial
graph
```
