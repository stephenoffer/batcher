# Domain analytics

This section covers the three analyses that need a vocabulary the relational verbs don't have: geometry on the Earth's surface, rigid-body motion between a robot's coordinate frames, and connectivity over an edge table.

None of them is a separate engine. A spatial predicate, a frame transform, and a PageRank iteration are expressions and joins over ordinary Arrow columns, so they optimize, spill, and distribute exactly as a `group_by` does.

`ST_*` geometry runs natively in Rust over WKB:

```python
import batcher as bt
import batcher.graph as bg

pts = bt.from_pydict({"name": ["a", "b"], "lon": [0.0, 3.0], "lat": [0.0, 4.0]})
geom = bt.st_point(bt.col("lon"), bt.col("lat"))
origin = bt.st_point(bt.lit(0.0), bt.lit(0.0))
print(pts.select("name", wkt=bt.st_as_text(geom), dist=bt.st_distance(geom, origin)).to_pydict())
# {'name': ['a', 'b'], 'wkt': ['POINT(0 0)', 'POINT(3 4)'], 'dist': [0.0, 5.0]}
```

Rotations and poses are columns too:

```python
heading = bt.from_pydict({"yaw": [1.5707963267948966]})
quat = heading.select(**bt.quat_from_euler(bt.lit(0.0), bt.lit(0.0), bt.col("yaw")))
print({k: [round(v, 4) for v in vs] for k, vs in quat.to_pydict().items()})
# {'qx': [0.0], 'qy': [0.0], 'qz': [0.7071], 'qw': [0.7071]}
```

And a graph algorithm reads an edge table that can live in Parquet, a lakehouse table, or a database:

```python
star = bg.Graph.from_edges(bt.from_pydict({"src": [1, 2, 3, 4], "dst": [0, 0, 0, 0]}))
top = bg.pagerank(star).sort("pagerank", descending=True).limit(1).to_pydict()
print(top["node"], round(top["pagerank"][0], 4))
# [0] 0.5238
```

| Guide | Covers |
| --- | --- |
| {doc}`geospatial` | Geometry, spatial joins, projections, and grid keys |
| {doc}`robotics` | Coordinate frames, poses, sensor alignment, and point clouds |
| {doc}`graphs` | PageRank, components, communities, and graph-ML features |

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
