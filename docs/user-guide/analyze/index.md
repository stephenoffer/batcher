# Analyze

Reduce many rows to an answer: grouping, joining, ranking, and the SQL front-end over the same engine.

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} {octicon}`graph;1.1em` Aggregations
:link: /user-guide/analyze/aggregations
:link-type: doc
Group and summarize; pivot; roll up.
:::

:::{grid-item-card} {octicon}`git-merge;1.1em` Joins
:link: /user-guide/analyze/joins
:link-type: doc
Inner, outer, semi, anti, as-of.
:::

:::{grid-item-card} {octicon}`clock;1.1em` Time series
:link: /user-guide/analyze/time-series
:link-type: doc
Bucketing, gap filling, smoothing, as-of alignment.
:::

:::{grid-item-card} {octicon}`versions;1.1em` Window functions
:link: /user-guide/analyze/window-functions
:link-type: doc
Ranking, running totals, lag and lead.
:::

:::{grid-item-card} {octicon}`table;1.1em` Pivoting
:link: /user-guide/analyze/pivoting
:link-type: doc
Long to wide and back.
:::

:::{grid-item-card} {octicon}`globe;1.1em` Geospatial
:link: /user-guide/analyze/geospatial
:link-type: doc
Geometry, spatial joins, projections, and grid keys.
:::

:::{grid-item-card} {octicon}`rocket;1.1em` Robotics and AV
:link: /user-guide/analyze/robotics
:link-type: doc
Coordinate frames, poses, sensor alignment, point clouds.
:::

:::{grid-item-card} {octicon}`share-android;1.1em` Graphs
:link: /user-guide/analyze/graphs
:link-type: doc
PageRank, components, communities, and graph-ML features.
:::

:::{grid-item-card} {octicon}`database;1.1em` SQL
:link: /user-guide/analyze/sql
:link-type: doc
Full SQL that lowers to the same engine.
:::

:::{grid-item-card} {octicon}`zap;1.1em` Metadata shortcuts
:link: /user-guide/analyze/metadata-shortcuts
:link-type: doc
Answer from the footer instead of the data, with {py:obj}`ds.meta <batcher.Dataset.meta>`.
:::
::::

```{toctree}
:hidden:

aggregations
joins
time-series
window-functions
pivoting
geospatial
robotics
graphs
sql
metadata-shortcuts
```
