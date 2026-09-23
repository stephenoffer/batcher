# Geospatial

This page covers working with geometry in Batcher: getting it in, measuring it, joining on it, and turning positions into keys you can group by.

## What a geometry is here

A geometry column is WKB in a Binary column. Don't skip past that as an internal detail. GeoParquet, PostGIS, GeoPackage and DuckDB spatial all store WKB, so a geometry column round-trips through any of them with no conversion, and every operator, spill path and shuffle the engine already has moves it without a new physical type.

You do not have to build one to start. Every `st_*` function accepts a text column and parses it, detecting WKT, EWKT, GeoJSON and hex WKB by content:

```python
import batcher as bt
from batcher import col

parcels = bt.from_pydict(
    {
        "id": [1, 2, 3],
        "geom": [
            "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON((4 0, 8 0, 8 4, 4 4, 4 0))",
            "POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))",
        ],
    }
)
print(parcels.select("id", area=bt.st_area(col("geom"))).to_pydict())
# {'id': [1, 2, 3], 'area': [16.0, 16.0, 16.0]}
```

Most tables do not store geometry at all. They store two float columns. {py:func}`st_point <batcher.st_point>` is the bridge, and its argument order is x then y, which for geographic data means longitude first:

```python
sightings = bt.from_pydict(
    {
        "city": ["San Francisco", "London", "Sydney"],
        "lon": [-122.4194, -0.1278, 151.2093],
        "lat": [37.7749, 51.5074, -33.8688],
    }
)
located = sightings.with_columns(geom=bt.st_point(col("lon"), col("lat")))
print(located.select("city", wkt=bt.st_as_text(col("geom"))).to_pydict()["wkt"])
# ['POINT(-122.4194 37.7749)', 'POINT(-0.1278 51.5074)', 'POINT(151.2093 -33.8688)']
```

:::{warning}
WKT, GeoJSON and PostGIS all put longitude first, which is the opposite of the order latitude and longitude are usually spoken in. Reversing them puts Zurich in the Indian Ocean and raises no error, because both orderings are valid coordinates. {py:func}`st_flip_coordinates <batcher.st_flip_coordinates>` is the fix once you notice.
:::

## Check validity before you trust anything

Real geometry columns are full of invalid polygons: rings that cross themselves from a digitizing error, holes poking outside their shell, rings with two vertices. Every areal predicate produces nonsense on those and none of them complains.

{py:func}`st_is_valid_reason <batcher.st_is_valid_reason>` returns null for a valid geometry and a sentence for an invalid one, which makes finding the broken rows a single filter:

```python
mixed = bt.from_pydict(
    {
        "id": [1, 2],
        "geom": [
            "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON((0 0, 4 4, 4 0, 0 4, 0 0))",
        ],
    }
)
broken = mixed.filter(bt.st_is_valid_reason(col("geom")).is_not_null())
print(broken.select("id", why=bt.st_is_valid_reason(col("geom"))).to_pydict())
# {'id': [2], 'why': ['exterior ring self-intersects near (2, 2)']}
```

The rules are GEOS's, so `st_is_valid` agrees with PostGIS and DuckDB. A ring needs four positions after consecutive repeats are dropped, so `POLYGON((0 0, 0 0, 0 0, 0 0))` is invalid. A chain needs two distinct positions. Rings must not cross themselves, and holes must sit inside their shell without running along it or overlapping each other. {py:func}`st_is_simple <batcher.st_is_simple>` is the narrower question of whether each ring and chain avoids crossing itself, so a bowtie polygon is neither simple nor valid, while a hole poking out of its shell is simple and invalid.

A geometry that will not parse at all becomes null rather than raising the query. That is deliberate: one corrupt row in a hundred million should not abort a scan that is otherwise fine. The same rule covers a row whose values are outside a function's domain. A NaN longitude, a longitude of 200 or a latitude of 95 gives that row a null geohash, tile, S2 cell, UTM zone or geodesic measure, and interpolating along `LINESTRING EMPTY` gives a null point. DuckDB raises on some of those rows and PostGIS on most. A query *bug*, such as a geohash precision of 13, a negative zoom or an unsupported EPSG code, does raise, because it is wrong on every row rather than on one. A constant one is refused when the expression is built, and the message names the value you passed.

```python
fixes = bt.from_pydict({"lon": [-122.4194, float("nan"), 200.0], "lat": [37.7749, 10.0, 0.0]})
print(fixes.select(cell=bt.geohash_encode(col("lon"), col("lat"), 6)).to_pydict())
# {'cell': ['9q8yyk', None, None]}
```

## Degrees are not meters

{py:func}`st_area <batcher.st_area>`, {py:func}`st_length <batcher.st_length>` and {py:func}`st_distance <batcher.st_distance>` are *planar*, and this is the most important thing on the page. They treat coordinates as points on a flat plane and answer in whatever unit the coordinates are stated in. On EPSG:4326 that unit is degrees, and a degree is not a distance:

```python
pair = bt.from_pydict({"a": ["POINT(-122.4194 37.7749)"], "b": ["POINT(-0.1278 51.5074)"]})
print(
    pair.select(
        planar=bt.st_distance(col("a"), col("b")).round(2),
        km=(bt.st_distance_sphere(col("a"), col("b")) / 1000).round(0),
    ).to_pydict()
)
# {'planar': [123.06], 'km': [8616.0]}
```

The planar answer is not wrong. It answers a different question, and it is exactly what PostGIS's `geometry` type does. It exists because the planar metric is the one a bounding box can bound, which is what makes spatial joins affordable.

You have three options, in increasing order of how much they cost:

| You need | Use |
| --- | --- |
| To rank or filter by proximity | The planar functions, on lon/lat |
| A distance in meters, occasionally | {py:func}`st_distance_sphere <batcher.st_distance_sphere>` or {py:func}`st_distance_spheroid <batcher.st_distance_spheroid>` |
| Meters everywhere in a pipeline | {py:func}`st_transform <batcher.st_transform>` once, then the planar functions |

The two geodesic families differ in their model of the Earth. `_sphere` is haversine on one mean-radius sphere, which is cheap and about 0.5% off. Every `_spheroid` function, meaning {py:func}`st_distance_spheroid <batcher.st_distance_spheroid>`, {py:func}`st_length_spheroid <batcher.st_length_spheroid>`, {py:func}`st_perimeter_spheroid <batcher.st_perimeter_spheroid>` and {py:func}`st_area_spheroid <batcher.st_area_spheroid>`, is on the WGS 84 ellipsoid by Karney's algorithm. That is the algorithm GeographicLib, PROJ, PostGIS `geography` and DuckDB spatial use, and the differential tests hold these functions to DuckDB's answers at 1e-9 relative. It is defined for antipodal points, and a polygon that crosses the antimeridian is measured the short way across it:

```python
box = bt.from_pydict({"g": ["POLYGON((170 -10, -170 -10, -170 10, 170 10, 170 -10))"]})
print(box.select(km2=(bt.st_area_spheroid(col("g")) / 1e6).round(0)).to_pydict())
# {'km2': [4948480.0]}
```

The third is usually right. Project into the local UTM zone and every planar function afterwards answers in meters, accurate to better than a tenth of a percent:

```python
sf = bt.from_pydict({"lon": [-122.4194], "lat": [37.7749]})
utm = sf.select(epsg=bt.st_utm_epsg(col("lon"), col("lat")))
print(utm.to_pydict())
# {'epsg': [32610]}

projected = sf.select(
    m=bt.st_x(bt.st_transform(bt.st_point(col("lon"), col("lat")), 4326, 32610)).round(0)
)
print(projected.to_pydict())
# {'m': [551131.0]}
```

`st_transform` supports a deliberately small set of systems and refuses an unsupported EPSG code rather than silently returning the input: WGS 84 lon/lat (4326), Web Mercator (3857), the UTM zones (326xx and 327xx), and a cylindrical equal-area system (6933) for density comparisons across latitudes. Every one of them is on the WGS 84 datum, so converting between them loses nothing.

## Spatial joins, and the filter that makes them affordable

A spatial join is an ordinary join with a spatial predicate. The predicate is expensive: {py:func}`st_intersects <batcher.st_intersects>` decodes both geometries and walks their segments.

{py:func}`st_intersects_extent <batcher.st_intersects_extent>` compares four numbers instead, and it is exact in the negative direction: a false means the geometries certainly do not intersect. That makes it a sound prefilter, producing false positives the exact test then removes and never false negatives.

The example below runs both filters in that order, and each one removes pairs in only one direction:

![A spatial join drawn as a flow with two exits, using the example below. A cross join of 3 points and 2 regions produces 6 pairs. st_intersects_extent compares 4 numbers per pair: 4 pairs come back false and leave as certainly disjoint, which is never a false negative. They are point 3 with both regions, point 1 with east and point 2 with west. The 2 pairs that come back true go on to st_intersects, which decodes both geometries and walks their segments. Any false positive the box test let through would be removed there, and in this example there are none, so both pairs come back true: 2 hits, point 1 in west and point 2 in east. Faster still, materialize st_xmin, st_ymin, st_xmax and st_ymax once beside the geometry. They are plain Float64, so a range predicate on them pushes down to the scan and to Parquet statistics.](/_static/diagrams/spatial_join_prefilter.svg)

```python
regions = bt.from_pydict(
    {
        "region": ["west", "east"],
        "shape": [
            "POLYGON((0 0, 5 0, 5 10, 0 10, 0 0))",
            "POLYGON((5 0, 10 0, 10 10, 5 10, 5 0))",
        ],
    }
)
points = bt.from_pydict({"pid": [1, 2, 3], "at": ["POINT(1 1)", "POINT(7 3)", "POINT(20 20)"]})

hits = (
    points.join(regions, how="cross")
    .filter(bt.st_intersects_extent(col("shape"), col("at")))
    .filter(bt.st_intersects(col("shape"), col("at")))
    .select("pid", "region")
    .sort("pid")
)
print(hits.to_pydict())
# {'pid': [1, 2], 'region': ['west', 'east']}
```

:::{tip}
Better still, materialize the four bound columns once beside the geometry. {py:func}`st_xmin <batcher.st_xmin>`/{py:func}`st_ymin <batcher.st_ymin>`/{py:func}`st_xmax <batcher.st_xmax>`/{py:func}`st_ymax <batcher.st_ymax>` are plain Float64, so a range predicate on them pushes down to the scan and to Parquet statistics, which a geometry predicate cannot do.
:::

## `contains` is not `covers`

The two differ exactly on the boundary. A polygon *covers* a point sitting on its edge. It does not *contain* it, because `contains` also requires the point to meet the polygon's interior.

```python
edge = bt.from_pydict({"poly": ["POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))"], "pt": ["POINT(0 2)"]})
print(
    edge.select(
        covers=bt.st_covers(col("poly"), col("pt")),
        contains=bt.st_contains(col("poly"), col("pt")),
    ).to_pydict()
)
# {'covers': [True], 'contains': [False]}
```

If a spatial join is dropping rows that sit exactly on a border, and border cases are never rare in real data, this is usually why. {py:func}`st_covers <batcher.st_covers>` is also the cheaper of the two.

Two more pairs worth keeping straight:

- {py:func}`st_touches <batcher.st_touches>` means they meet but do not overlap, which is the adjacency predicate: neighboring parcels, bordering countries.
- {py:func}`st_overlaps <batcher.st_overlaps>` means they partially overlap. A polygon entirely inside another does not overlap it. It is contained by it.

## Grids: turning positions into group keys

Latitude and longitude are floats, so no two observations share a value and `GROUP BY lat, lon` returns the input. A grid function turns a position into a discrete cell id, and the engine then hashes, sorts, shuffles and joins that at full speed with no spatial index at all.

```python
pickups = bt.from_pydict(
    {
        "lon": [-122.4194, -122.4190, -0.1278],
        "lat": [37.7749, 37.7751, 51.5074],
    }
)
binned = pickups.with_columns(cell=bt.geohash_encode(col("lon"), col("lat"), 6))
print(binned.group_by("cell").agg(n=bt.count()).sort("cell").to_pydict())
# {'cell': ['9q8yyk', 'gcpvj0'], 'n': [2, 1]}
```

The four grids differ in what the id gives you beyond grouping:

| Function | Cell id | Also gives you |
| --- | --- | --- |
| {py:func}`geohash_encode <batcher.geohash_encode>` | base-32 string | prefix containment: `LIKE 'u09%'` is a region filter |
| {py:func}`st_quadkey <batcher.st_quadkey>`, {py:func}`st_tile_x <batcher.st_tile_x>`/{py:func}`st_tile_y <batcher.st_tile_y>` | tile address | the exact grid map tiles are served on |
| {py:func}`st_s2_cell <batcher.st_s2_cell>` | `Int64` Hilbert index | near-equal-area cells, and a region as a `BETWEEN` |
| {py:func}`st_hex_bin <batcher.st_hex_bin>` | packed `Int64` | six equidistant neighbors, for unbiased density |

Two of these have prefix structure, which is what makes a rollup across zoom levels a string operation rather than a recomputation. A geohash nests by character and a quadkey by digit:

```python
zoom = bt.from_pydict({"lon": [-122.4194], "lat": [37.7749]})
fine = bt.geohash_encode(col("lon"), col("lat"), 8)
print(zoom.select(fine=fine, coarse=fine.str.substr(1, 4)).to_pydict())
# {'fine': ['9q8yyk8y'], 'coarse': ['9q8y']}
```

`st_s2_cell` is the one to reach for when the grid has to be fair across latitudes. A degree grid's cells shrink to nothing at the poles, so a density comparison on one is comparing rectangles of different sizes. S2's cells are near-equal-area anywhere on Earth. Its ids also sort spatially, so sorting a table by one clusters neighbors onto the same pages, and rolling up to a coarser level is a bit mask:

```python
cells = bt.from_pydict({"lon": [-122.4194, -122.4190], "lat": [37.7749, 37.7751]})
fine_cell = bt.st_s2_cell(col("lon"), col("lat"), 15)
rolled = cells.select(parent=bt.st_s2_cell_parent(fine_cell, 8))
print(rolled.group_by("parent").agg(n=bt.count()).to_pydict()["n"])
# [2]
```

:::{note}
`st_hex_bin` is a planar hex grid, not H3. H3's cells live on an icosahedron and its indexes are not these, so do not join one against the other. Project with `st_transform` before binning, since the function bins whatever coordinates it is given.
:::

## Buffers

{py:func}`st_buffer <batcher.st_buffer>` returns every position within a distance of a geometry. A point grows into a regular polygon with `4 * quad_segs` sides, a chain into a capsule around each segment, and a polygon into itself plus a band around its rings. The parts of a multi-geometry are buffered separately and merged only where their buffers overlap. A negative radius erodes a polygon, and a radius of 0 returns it unchanged:

```python
shapes = bt.from_pydict(
    {"g": ["MULTIPOINT((0 0), (10 0))", "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))"]}
)
print(
    shapes.select(
        parts=bt.st_num_geometries(bt.st_buffer(col("g"), 1.0, 8)),
        grown=bt.st_area(bt.st_buffer(col("g"), 1.0, 8)).round(3),
        eroded=bt.st_area(bt.st_buffer(col("g"), -1.0, 8)),
    ).to_pydict()
)
# {'parts': [2, 1], 'grown': [6.243, 35.121], 'eroded': [0.0, 4.0]}
```

The only approximation is the circle drawn as a polygon, which every implementation makes. A point's buffer is GEOS's vertex for vertex. For a chain or a polygon, GEOS starts each round join at the segment's offset point where Batcher places its vertex disc at fixed angles, so the areas differ by a fraction of a percent at the default eight segments per quadrant. When the question is only "is this within X", {py:func}`st_dwithin <batcher.st_dwithin>` answers it exactly without building the polygon.

## Simplify before you shuffle

Vertex count drives the cost of every predicate, every byte written and every byte shuffled. {py:func}`st_simplify <batcher.st_simplify>` is usually the single biggest win available on a large geometry column, and {py:func}`st_hausdorff_distance <batcher.st_hausdorff_distance>` measures what the tolerance cost you:

```python
detailed = bt.from_pydict({"g": ["LINESTRING(0 0, 1 0.001, 2 0, 3 0.002, 4 0, 5 0)"]})
simple = bt.st_simplify(col("g"), 0.01)
print(
    detailed.select(
        before=bt.st_num_points(col("g")),
        after=bt.st_num_points(simple),
        error=bt.st_hausdorff_distance(col("g"), simple).round(4),
    ).to_pydict()
)
# {'before': [6], 'after': [2], 'error': [0.002]}
```

## What it costs

Every `ST_*` function is a scalar expression evaluated per row in Rust over the WKB buffer, with no Python in the loop and no conversion to a separate geometry type. [`benchmarks/geospatial.py`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/geospatial.py) runs the functions on 2 million real places from Overture Maps, in Batcher and in DuckDB's spatial extension over the same Arrow table, and checks each result for correctness before timing either engine. The ratio is DuckDB's time over Batcher's, so above 1.00x means Batcher is faster:

| Expression, 2M rows | Batcher | DuckDB spatial | Ratio |
| --- | --- | --- | --- |
| `st_x` of a constructed point | 13.7 ms | 46.4 ms | 3.39x |
| `st_intersects` against a box | 64.9 ms | 329.5 ms | 5.08x |
| `st_distance` to a point | 37.7 ms | 50.8 ms | 1.35x |
| `st_transform` to EPSG:3857 | 22.2 ms | 287.3 ms | 12.93x |
| `st_as_text` | 40.5 ms | 492.0 ms | 12.15x |
| `geohash_encode`, precision 8 | 17.9 ms | not available | |
| `st_s2_cell`, level 15 | 15.2 ms | not available | |

The run used a 96-core Xeon 8275CL with DuckDB 1.5.5 on a box shared with other work, so read the ratios rather than the absolute times. At 250,000 rows the same cases range from 0.89x to 6.43x, because fixed per-query overhead weighs more on a small input. The full record is in [`benchmarks/BENCHMARK_RESULTS.md`](https://github.com/stephenoffer/batcher/blob/main/benchmarks/BENCHMARK_RESULTS.md).

The ordering inside Batcher's column is the useful part. The grid encoders and accessors are cheap. A predicate costs several times more per row, because each row decodes two geometries and walks their segments, which is exactly why `st_intersects_extent` in front of `st_intersects` is worth the extra clause and why materializing the four bound columns is worth the storage.

## On a cluster

Every function on this page is a row-wise expression: it reads one row and writes one value, with no state carried between rows. So it distributes with any query the way arithmetic does, and `collect(distributed=True)` returns the rows a single node returns. A grid key computed on the workers feeds an ordinary distributed `group_by`, and a geometry column shuffles as the Binary column it is. [`tests/integration/test_geo_distributed.py`](https://github.com/stephenoffer/batcher/blob/main/tests/integration/test_geo_distributed.py) checks this on 80,000 rows over four Parquet files and four workers, with the NaN and off-globe rows that null row by row mixed in.

## Requirements and limitations

- {py:func}`st_buffer <batcher.st_buffer>` approximates a circle with `4 * quad_segs` chords, as every implementation does, and its areas for chains and polygons differ from GEOS's by a fraction of a percent. Where an eroded polygon narrows to a single point, GEOS writes one self-touching polygon and Batcher writes two polygons touching there, which is the same point set.
- The geodesic distances are measured vertex to vertex, not between the nearest points of two shapes. For point pairs, which is most proximity work, those coincide exactly. For extended geometries they over-report by at most a segment length, so they are an upper bound. Run {py:func}`st_segmentize <batcher.st_segmentize>` first when the answer must be tight.
- `st_transform` covers the four families of reference system listed above and rejects everything else by EPSG code. Reprojecting between datums such as NAD 27 or OSGB 36 needs a grid shift that is not built in.
- There is no public polygon overlay. `st_intersection`, `st_union` and `st_difference` do not exist, and neither do the aggregate forms `st_extent` and `st_union_agg`. `st_buffer` computes a union internally, but only of its own pieces. {py:func}`st_collect <batcher.st_collect>` concatenates without computing one, which is what you want before a single {py:func}`st_envelope <batcher.st_envelope>` or {py:func}`st_convex_hull <batcher.st_convex_hull>`.
- {py:func}`st_exterior_ring <batcher.st_exterior_ring>` and {py:func}`st_interior_ring_n <batcher.st_interior_ring_n>` answer only for a `POLYGON`, as in PostGIS and DuckDB. A multipolygon has no single exterior ring, so write `st_exterior_ring(st_geometry_n(g, 1))` for the first member's.
- An empty result is null in most places where DuckDB writes an empty geometry: the centroid or point on surface of an empty input, for example.
- A geometry with a NaN coordinate is treated as unparseable and yields null, because every predicate is a chain of comparisons and NaN makes all of them false in both directions.

## See also

- {doc}`/api/relational/domains/geospatial`: every `ST_*` function, grouped and enumerated.
- {doc}`/cookbook/analytics/aggregates/geospatial-binning`: snapping coordinates to a grid by hand, and why `floor` is the only correct way to do it.
- {doc}`/user-guide/analyze/joins`: the join mechanics a spatial join composes with.
- {doc}`/user-guide/analyze/aggregations`: the {py:meth}`group_by <batcher.Dataset.group_by>` a grid key feeds.
