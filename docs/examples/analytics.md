# Statistics, time series, geospatial, graph and robotics

This page covers the analytical scripts: summarizing distributions, working with time, and
the geospatial, graph and rigid-body surfaces.

## Statistics

Centre, spread, shape and extremes in one pass, and the order matters for interpretation. The
skew tells you whether to report the mean or the median, so compute both before deciding
which goes in the summary.

```python
import batcher as bt
from batcher import col

values = bt.from_pydict({"x": [1.0, 2.0, 2.0, 3.0, 40.0]})

summary = values.agg(
    mean=col("x").mean(),
    median=bt.median(col("x")),
    iqr=bt.iqr(col("x")),
    skew=bt.skewness(col("x")),
).to_pydict()

# Right-skewed, so the mean sits above the median and the median is the honest number.
assert summary["skew"][0] > 0
assert summary["mean"][0] > summary["median"][0]
```

Two habits the scripts enforce. A difference between groups is not a finding until you know
the spread and the sample size, so the standard error comes out of the same pass as the mean.
And with enough rows every difference is significant, so an effect size is what makes the
comparison informative.

## Time series

Resampling is a truncation used as a group key, and the one thing to watch is that periods
with no events are simply absent. Joining against a generated calendar is what turns a sparse
series into a dense one, and the rows the join adds are exactly the periods nothing happened
in.

```python
import batcher as bt
from batcher import col

events = bt.from_pydict(
    {
        "day": ["2024-01-01", "2024-01-01", "2024-01-03"],
        "amount": [10.0, 20.0, 5.0],
    }
).with_columns(day=col("day").str.to_date("%Y-%m-%d"))

daily = events.group_by("day").agg(total=col("amount").sum()).sort("day")

# Two days have events; the calendar between them spans three.
assert daily.count() == 2
```

Growth needs a previous value, and the first period has none. A null there is correct; a zero
is a lie that shows up as a spike on every chart.

## Geospatial

Longitude first, then latitude, which is what WKT, GeoJSON and PostGIS all use. Reversing it
puts the data in the wrong hemisphere without any error.

A spatial join cannot hash, so the pattern is to bound the candidates cheaply with an
envelope or a grid key and evaluate the exact predicate only on what survives. A geohash
turns the join into an equality join, and the precision is the trade: coarser cells mean more
candidates to check, finer cells mean more cells to enumerate.

## Graph

A graph is an edge table with two columns you have chosen to call source and target, which is
what makes graph analytics available to any dataset with a foreign key. Degree and connected
components cost one pass each and tell you whether an expensive algorithm will mean anything.

Materialize the edge list before running the iterative algorithms. They re-read the edges once
per iteration, so handing them a join-and-distinct plan re-executes that plan every time.

## Robotics and autonomous driving

A robotics log is measurements taken in different coordinate frames, one per sensor plus one
for the vehicle and one for the world. A quaternion is four numbers in `(x, y, z, w)` order,
scalar last, and a pose is a translation followed by that rotation. Reading a scalar-first
quaternion as a scalar-last one is a different, plausible rotation that nothing can detect,
so the component order is written at every call site.

Collapse a frame chain once per frame with {py:func}`se3_compose <batcher.se3_compose>` and
apply the single result per point. A sweep is a hundred thousand points, so the difference
between one transform per point and three is the whole cost of the job.

Interpolate a pose with {py:func}`pose_interpolate <batcher.pose_interpolate>` rather than
blending quaternion components and renormalizing. The second sweeps the angle at a
non-constant rate, which shows up as a lidar sweep that bends.

## Every script on this page

The table below lists the analytical scripts in path order.

<!-- library-table: statistics,timeseries_real,geospatial,graph,robotics -->
| Script | Shows |
| --- | --- |
| `examples/statistics/ab_test_inference.py` | A/B test statistics computed in the engine: effect size, t-statistic, and intervals |
| `examples/statistics/association.py` | How strongly does one column relate to another? |
| `examples/statistics/binomial_proportions.py` | Proportions and their uncertainty |
| `examples/statistics/comparing_groups.py` | Comparing a metric across groups, with the spread that says whether it means anything |
| `examples/statistics/correlation_matrix.py` | A correlation matrix over several columns at once |
| `examples/statistics/describing_a_real_column.py` | What to compute first when you meet a column |
| `examples/statistics/distribution_shape.py` | Is this column symmetric, skewed, or heavy-tailed? |
| `examples/statistics/distribution_tails.py` | Where the extreme values are, and how many there are |
| `examples/statistics/effect_sizes.py` | Effect size: how big a difference is, not just whether it exists |
| `examples/statistics/hypothesis_intervals.py` | A confidence interval around a difference between two groups |
| `examples/statistics/quantiles_and_histograms.py` | Quantiles, histograms, and the exact-versus-approximate trade |
| `examples/statistics/rank_correlation.py` | Rank correlation: agreement that survives a non-linear relationship |
| `examples/statistics/robust_dispersion.py` | Robust spread: quantile-based measures that one outlier cannot move |
| `examples/statistics/sampling_error.py` | How wrong a sample is, and how that shrinks with size |
| `examples/statistics/summary_statistics.py` | Summary aggregates beyond mean and stddev |
| `examples/statistics/weighted_statistics.py` | Statistics where the rows are not equally important |
| `examples/timeseries_real/cohort_retention.py` | A cohort table: grouping customers by when they first appeared |
| `examples/timeseries_real/forecast_baseline.py` | The baselines any forecast has to beat |
| `examples/timeseries_real/growth_rates.py` | Period-over-period change, and why the first period is null |
| `examples/timeseries_real/lead_lag_and_gaps.py` | Detecting gaps in an event series |
| `examples/timeseries_real/moving_windows_by_time.py` | Windows measured in days rather than in rows |
| `examples/timeseries_real/resampling.py` | Resampling a real order stream to daily, weekly and monthly grain |
| `examples/timeseries_real/seasonality.py` | Finding a weekly and monthly pattern in a real order series |
| `examples/geospatial/bounding_boxes.py` | Envelopes: the cheap bound that makes an exact predicate affordable |
| `examples/geospatial/geometry_basics.py` | Getting geometry in, reading it apart, and writing it back out |
| `examples/geospatial/grid_keys.py` | Turning positions into cell ids you can group, sort and join on |
| `examples/geospatial/grid_keys_as_join_keys.py` | Grid keys: turning a point into a joinable string |
| `examples/geospatial/linestrings_and_areas.py` | Lines and polygons: length, area, centroid, and the relationships between them |
| `examples/geospatial/measures_and_projections.py` | Measuring geometry, and getting the units right |
| `examples/geospatial/points_and_distance.py` | Building point geometry from coordinate columns, and measuring between them |
| `examples/geospatial/predicates_and_joins.py` | Spatial relationships, and the prefilter that makes a spatial join affordable |
| `examples/geospatial/spatial_joins_with_grids.py` | Making a spatial join hashable by joining on a grid cell first |
| `examples/geospatial/spatial_predicates.py` | Point-in-polygon and the other spatial predicates |
| `examples/geospatial/wkt_and_geojson.py` | Geometry codecs: WKT, WKB and GeoJSON |
| `examples/graph/analytics.py` | Graph analytics end to end: diagnose, rank, cluster, measure |
| `examples/graph/bipartite_projection.py` | Projecting a bipartite graph onto one of its sides |
| `examples/graph/degree_and_components.py` | The cheap graph diagnostics that decide what is worth running next |
| `examples/graph/graph_from_relational.py` | Building a graph out of an ordinary relational table |
| `examples/graph/graph_ml.py` | Graph ML: sampling a graph a model can read, and building features from it |
| `examples/graph/pagerank_and_centrality.py` | Ranking nodes by influence on a real bipartite graph |
| `examples/graph/shortest_paths.py` | Shortest paths on a small weighted graph |
| `examples/graph/triangles_and_clustering.py` | Local structure: triangles and the clustering coefficient |
| `examples/robotics/coordinate_frames.py` | Moving a sensor measurement between coordinate frames |
| `examples/robotics/point_clouds.py` | Putting a lidar sweep in world coordinates, then cutting it down |
| `examples/robotics/pose_interpolation.py` | Lining up sensors that sample at different rates |
| `examples/robotics/rotations.py` | Building, cleaning, composing and scoring rotations |
<!-- /library-table -->
