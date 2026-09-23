# Domain analytics

This section covers the three analyses that need a vocabulary the relational verbs don't have: geometry on the Earth's surface, rigid-body motion between a robot's coordinate frames, and connectivity over an edge table.

None of them is a separate engine, and that is the reason they are here rather than in a library you install beside Batcher. A spatial predicate, a frame transform, and a PageRank iteration are all expressions and joins over ordinary Arrow columns, so they optimize, spill, and distribute exactly as a `group_by` does. A spatial join is a join. A graph algorithm reads an edge table that can live in Parquet, in a lakehouse table, or in a database.

| Guide | Covers |
| --- | --- |
| {doc}`geospatial` | Geometry, spatial joins, projections, and grid keys |
| {doc}`robotics` | Coordinate frames, poses, sensor alignment, and point clouds |
| {doc}`graphs` | PageRank, components, communities, and graph-ML features |

`ST_*` geometry runs natively in Rust over WKB, with no Python and no conversion step. On 2 million real map points it measured 1.35x to 12.9x faster than DuckDB's spatial extension over the same Arrow table.

## See also

- {doc}`/api/relational/domains/index`: the reference for every function these guides use.
- {doc}`../index`: the general analysis operators these build on.
- {doc}`/user-guide/moving-data/specialized-formats`: reading LiDAR sweeps, MCAP logs, and MDF4 measurements.
- {doc}`/cookbook/analytics/aggregates/geospatial-binning`: snapping coordinates to a grid, as a runnable recipe.

```{toctree}
:hidden:

geospatial
robotics
graphs
```
