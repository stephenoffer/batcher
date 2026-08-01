# Geospatial

This page covers working with geometry in Batcher: getting it in, measuring it, joining
on it, and turning positions into keys you can group by.

## What a geometry is here

A geometry column is **WKB in a Binary column**. That is not an internal detail you can
ignore, it is the reason geospatial work is fast: WKB is what GeoParquet, PostGIS,
GeoPackage and DuckDB spatial all store, so a geometry column round-trips through any of
them with no conversion, and every operator, spill path and shuffle the engine already
has moves it without a new physical type.

You do not have to build one to start. Every `st_*` function accepts a text column and
parses it, detecting WKT, EWKT, GeoJSON and hex WKB by content:

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

Most tables do not store geometry at all, they store two float columns. `st_point`
is the bridge, and its argument order is **x then y**, which for geographic data means
longitude first:

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
Longitude first is what WKT, GeoJSON and PostGIS all use, and it is the opposite of how
latitude and longitude are usually spoken. Reversing them puts Zurich in the Indian
Ocean and raises no error, because both orderings are valid coordinates.
`st_flip_coordinates` is the fix once you notice.
:::

## Check validity before you trust anything

Real geometry columns are full of invalid polygons: rings that cross themselves from a
digitizing error, holes poking outside their shell, rings with two vertices. Every areal
predicate produces nonsense on those and none of them complains.

`st_is_valid_reason` returns null for a valid geometry and a sentence for an invalid one,
which makes finding the broken rows a single filter:

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

A geometry that will not parse at all becomes null rather than raising the query. That
is deliberate: one corrupt row in a hundred million should not abort a scan that is
otherwise fine. A query *bug*, such as a negative radius or an unsupported EPSG code,
does raise, because it is wrong on every row rather than on one.

## Degrees are not metres

This is the single most important thing on the page.

`st_area`, `st_length` and `st_distance` are **planar**. They treat coordinates as
points on a flat plane and answer in whatever unit the coordinates are stated in. On
EPSG:4326 that unit is degrees, and a degree is not a distance:

```python
pair = bt.from_pydict(
    {"a": ["POINT(-122.4194 37.7749)"], "b": ["POINT(-0.1278 51.5074)"]}
)
print(
    pair.select(
        planar=bt.st_distance(col("a"), col("b")).round(2),
        km=(bt.st_distance_sphere(col("a"), col("b")) / 1000).round(0),
    ).to_pydict()
)
# {'planar': [123.06], 'km': [8616.0]}
```

The planar answer is not wrong, it is an answer to a different question, and it is
exactly what PostGIS's `geometry` type does. It exists because the planar metric is the
one a bounding box can bound, which is what makes spatial joins affordable.

You have three options, in increasing order of how much they cost:

| You need | Use |
| --- | --- |
| To rank or filter by proximity | The planar functions, on lon/lat |
| A distance in metres, occasionally | `st_distance_sphere` or `st_distance_spheroid` |
| Metres everywhere in a pipeline | `st_transform` once, then the planar functions |

The third is usually right. Project into the local UTM zone and every planar function
afterwards answers in metres, accurate to better than a tenth of a percent:

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

`st_transform` supports a deliberately small set of systems and **refuses an unsupported
EPSG code** rather than silently returning the input: WGS 84 lon/lat (4326), Web Mercator
(3857), the UTM zones (326xx and 327xx), and a cylindrical equal-area system (6933) for
density comparisons across latitudes. Every one of them is on the WGS 84 datum, so
converting between them loses nothing.

## Spatial joins, and the filter that makes them affordable

A spatial join is an ordinary join with a spatial predicate. The predicate is expensive:
`st_intersects` decodes both geometries and walks their segments.

`st_intersects_extent` compares four numbers instead, and it is **exact in the negative
direction**: a false means the geometries certainly do not intersect. That makes it a
sound prefilter, producing false positives the exact test then removes and never false
negatives.

```python
regions = bt.from_pydict(
    {"region": ["west", "east"], "shape": [
        "POLYGON((0 0, 5 0, 5 10, 0 10, 0 0))",
        "POLYGON((5 0, 10 0, 10 10, 5 10, 5 0))",
    ]}
)
points = bt.from_pydict({"pid": [1, 2, 3], "at": [
    "POINT(1 1)", "POINT(7 3)", "POINT(20 20)"]})

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
Better still, materialize the four bound columns once beside the geometry.
`st_xmin`/`st_ymin`/`st_xmax`/`st_ymax` are plain Float64, so a range predicate on them
pushes down to the scan and to Parquet statistics, which a geometry predicate cannot do.
:::

## `contains` is not `covers`

The two differ exactly on the boundary. A polygon *covers* a point sitting on its edge;
it does not *contain* it, because `contains` also requires the point to meet the
polygon's interior.

```python
edge = bt.from_pydict(
    {"poly": ["POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))"], "pt": ["POINT(0 2)"]}
)
print(
    edge.select(
        covers=bt.st_covers(col("poly"), col("pt")),
        contains=bt.st_contains(col("poly"), col("pt")),
    ).to_pydict()
)
# {'covers': [True], 'contains': [False]}
```

If a spatial join is dropping rows that sit exactly on a border, and border cases are
never rare in real data, this is usually why. `st_covers` is also the cheaper of the two.

Two more pairs worth keeping straight:

- `st_touches` means they meet but do not overlap, which is the adjacency predicate:
  neighbouring parcels, bordering countries.
- `st_overlaps` means they partially overlap. A polygon entirely inside another does not
  overlap it, it is contained by it.

## Grids: turning positions into group keys

Latitude and longitude are floats, so no two observations share a value and
`GROUP BY lat, lon` returns the input. A grid function turns a position into a discrete
cell id, and the engine then hashes, sorts, shuffles and joins that at full speed with
no spatial index at all.

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
| `geohash_encode` | base-32 string | prefix containment: `LIKE 'u09%'` is a region filter |
| `st_quadkey`, `st_tile_x`/`st_tile_y` | tile address | the exact grid map tiles are served on |
| `st_s2_cell` | `Int64` Hilbert index | near-equal-area cells, and a region as a `BETWEEN` |
| `st_hex_bin` | packed `Int64` | six equidistant neighbours, for unbiased density |

Two of these have prefix structure, which is what makes a rollup across zoom levels a
string operation rather than a recomputation. A geohash nests by character and a quadkey
by digit:

```python
zoom = bt.from_pydict({"lon": [-122.4194], "lat": [37.7749]})
fine = bt.geohash_encode(col("lon"), col("lat"), 8)
print(zoom.select(fine=fine, coarse=fine.str.substr(1, 4)).to_pydict())
# {'fine': ['9q8yyk8y'], 'coarse': ['9q8y']}
```

`st_s2_cell` is the one to reach for when the grid has to be fair across latitudes. A
degree grid's cells shrink to nothing at the poles, so a density comparison on one is
comparing rectangles of different sizes; S2's cells are near-equal-area anywhere on
Earth. Its ids also sort spatially, so sorting a table by one clusters neighbours onto
the same pages, and rolling up to a coarser level is a bit mask:

```python
cells = bt.from_pydict({"lon": [-122.4194, -122.4190], "lat": [37.7749, 37.7751]})
fine_cell = bt.st_s2_cell(col("lon"), col("lat"), 15)
rolled = cells.select(parent=bt.st_s2_cell_parent(fine_cell, 8))
print(rolled.group_by("parent").agg(n=bt.count()).to_pydict()["n"])
# [2]
```

:::{note}
`st_hex_bin` is a **planar** hex grid, not H3. H3's cells live on an icosahedron and its
indexes are not these, so do not join one against the other. Project with `st_transform`
before binning, since the function bins whatever coordinates it is given.
:::

## Simplify before you shuffle

Vertex count drives the cost of every predicate, every byte written and every byte
shuffled. `st_simplify` is usually the highest-leverage change you can make to a large
geometry column, and `st_hausdorff_distance` tells you what the tolerance cost you:

```python
detailed = bt.from_pydict(
    {"g": ["LINESTRING(0 0, 1 0.001, 2 0, 3 0.002, 4 0, 5 0)"]}
)
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

## Requirements and limitations

- **`st_buffer` is an approximation and over-estimates for a concave input.** It buffers
  each vertex and takes the convex hull, which is exact for a convex geometry up to the
  arc discretization and is the buffer of the hull otherwise. It is sound as a candidate
  filter and wrong as a number to report. When the question is "is this within X", use
  `st_dwithin`, which is exact and cheaper.
- **The geodesic distances are measured vertex to vertex**, not between the nearest
  points of two shapes. For point pairs, which is most proximity work, those coincide
  exactly. For extended geometries they over-report by at most a segment length, so they
  are an upper bound; run `st_segmentize` first when the answer must be tight.
- **`st_transform` covers four families of reference system**, listed above, and rejects
  everything else by EPSG code. Reprojecting between datums such as NAD 27 or OSGB 36
  needs a grid shift that is not built in.
- **There is no polygon overlay.** `st_intersection`, `st_union` and `st_difference` do
  not exist. `st_collect` concatenates without computing one, which is what you want
  before a single `st_envelope` or `st_convex_hull`.
- **A geometry with a NaN coordinate is treated as unparseable** and yields null, because
  every predicate is a chain of comparisons and NaN makes all of them false in both
  directions.

## See also

- {doc}`/api/relational/geospatial`: every `ST_*` function, grouped and enumerated.
- {doc}`/cookbook/analytics/geospatial-binning`: snapping coordinates to a grid by hand,
  and why `floor` is the only correct way to do it.
- {doc}`/user-guide/analyze/joins`: the join mechanics a spatial join composes with.
- {doc}`/user-guide/analyze/aggregations`: the `group_by` a grid key feeds.
