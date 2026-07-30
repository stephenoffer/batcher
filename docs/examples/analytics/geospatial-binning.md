# Geospatial binning

Where are the pickups concentrated? You have latitude and longitude as floats, which means
you have effectively no repeated values. Every point is unique, so `GROUP BY lat, lon` returns
the input. To count anything you have to snap the points to a grid first, and the snapping
function is where the bug hides.

## The data

Seven points: two in San Francisco within 20 metres of each other, one across the bay, two
in New York a few metres apart, one in London, one in Sydney.

```python
import batcher as bt
from batcher import col, concat_ws, lit

pickups = bt.from_pydict(
    {
        "id": [1, 2, 3, 4, 5, 6, 7],
        "lat": [37.7749, 37.7751, 37.8044, 40.7128, 40.7130, 51.5074, -33.8688],
        "lon": [-122.4194, -122.4190, -122.2712, -74.0060, -74.0059, -0.1278, 151.2093],
    }
)
CELL = 0.01  # degrees, roughly 1.1 km north-south
print(pickups.count())
# 7
```

## The trap: `round` and `cast` are not `floor`

:::{warning}
A bin number is a promise: bin `b` holds every point with `b * CELL <= x < (b+1) * CELL`.
`round` and `cast` both break that promise, each in its own direction, and both do it
silently.
:::

The obvious way to snap a coordinate to a 0.01 degree grid is to divide and turn the result into
an integer. Both of the obvious ways to turn it into an integer are wrong.

```python
edges = bt.from_pydict({"id": [1, 2, 3, 4], "lon": [-0.005, 0.005, -0.015, 0.015]})
print(
    edges.select(
        "lon",
        rounded=(col("lon") / lit(CELL)).round(0),
        cast=(col("lon") / lit(CELL)).cast("int64"),
        floored=(col("lon") / lit(CELL)).floor().cast("int64"),
    ).to_pydict()
)
# {'lon': [-0.005, 0.005, -0.015, 0.015], 'rounded': [-1.0, 1.0, -2.0, 2.0],
#  'cast': [0, 0, -2, 2], 'floored': [-1, 0, -2, 1]}
```

Check the promise against the middle column. A point at longitude -0.005 gets `cast` bin 0,
and bin 0 is supposed to span [0.000, 0.010). The point is not in it. `cast` rounds half-to-even
rather than truncating, so both -0.005 and +0.005 collapse into bin 0: one double-width cell
straddling the meridian, with the two cells either side of it half empty. `round` is off by the
same half-cell in a different direction. Its bin `b` covers `[(b - 0.5) * CELL, (b + 0.5) * CELL)`, so the
number you get back names a cell *centre* while you were reasoning about corners.

| Snap | Bin for -0.005 | Bin for +0.005 | Keeps `b * CELL <= x < (b+1) * CELL`? |
| --- | --- | --- | --- |
| `.round(0)` | -1.0 | 1.0 | No. Names the cell centre, not its corner. |
| `.cast("int64")` | 0 | 0 | No. Both signs collapse into one double-width cell. |
| `.floor().cast("int64")` | -1 | 0 | Yes, on both signs, everywhere. |

:::{tip}
Only `floor` satisfies the promise. Use it. The follow-on consequences of not using it are
the annoying kind: `bin * CELL` is supposed to be the cell's corner, so any join to a
reference grid, any bounding box, and any tile lookup built on a rounded bin is silently
offset by half a cell.
:::

## The grid

```python
binned = pickups.with_columns(
    lat_bin=(col("lat") / lit(CELL)).floor().cast("int64"),
    lon_bin=(col("lon") / lit(CELL)).floor().cast("int64"),
)
print(binned.select("id", "lat_bin", "lon_bin").to_pydict())
# {'id': [1, 2, 3, 4, 5, 6, 7], 'lat_bin': [3777, 3777, 3780, 4071, 4071, 5150, -3387],
#  'lon_bin': [-12242, -12242, -12228, -7401, -7401, -13, 15120]}
```

The two SF points share a cell, the two NY points share a cell, everything else is on its
own. That is the whole idea: a pair of integers per point, and integers group.

## Hot cells

Two integers is an awkward key, but it is a *groupable* one, and that is the entire trick.
The SQL tab spells the same snap-and-count with `FLOOR` and `CAST`:

::::{tab-set}
:::{tab-item} DataFrame
```python
hot = (
    binned.group_by("lat_bin", "lon_bin")
    .agg(pickups=bt.count())
    .sort("pickups", descending=True)
)
print(hot.to_pydict())
# {'lat_bin': [3777, 4071, 3780, 5150, -3387], 'lon_bin': [-12242, -7401, -12228, -13, 15120],
#  'pickups': [2, 2, 1, 1, 1]}
```
:::

:::{tab-item} SQL
```python
sql_hot = bt.sql(
    """
    SELECT CAST(FLOOR(lat / 0.01) AS BIGINT) AS lat_bin,
           CAST(FLOOR(lon / 0.01) AS BIGINT) AS lon_bin,
           COUNT(*) AS pickups
    FROM pickups
    GROUP BY CAST(FLOOR(lat / 0.01) AS BIGINT), CAST(FLOOR(lon / 0.01) AS BIGINT)
    ORDER BY pickups DESC, lat_bin
    """,
    pickups=pickups,
)
print(sql_hot.to_pydict())
# {'lat_bin': [3777, 4071, -3387, 3780, 5150],
#  'lon_bin': [-12242, -7401, 15120, -12228, -13], 'pickups': [2, 2, 1, 1, 1]}
```
:::
::::

Same cells, same counts. The tail order differs only because the SQL breaks the
three-way tie at one pickup on `lat_bin`, which the DataFrame `sort` leaves alone.

`concat_ws` folds the pair of integers into one string cell id, and `bin * CELL` recovers
the cell's south-west corner for plotting:

```python
labelled = binned.with_columns(
    cell=concat_ws(":", col("lat_bin"), col("lon_bin")),
    corner_lat=col("lat_bin") * lit(CELL),
    corner_lon=col("lon_bin") * lit(CELL),
)
print(labelled.select("id", "cell").to_pydict())
# {'id': [1, 2, 3, 4, 5, 6, 7],
#  'cell': ['3777:-12242', '3777:-12242', '3780:-12228', '4071:-7401', '4071:-7401',
#           '5150:-13', '-3387:15120']}
```

If you would rather have one integer than one string, interleave the two bins into a single
`Int64`. `lat_bin * 100_000 + lon_bin` works as long as the multiplier exceeds the longitude
range, and it keeps the key numeric for the hash shuffle.

:::{important}
Do not build the key by concatenating the *raw* floats as strings. Two coordinates that
should be the same point can print differently, and you get two cells where you meant one.
:::

:::{dropdown} Degrees are not distance
A 0.01 degree cell is about 1.1 km tall everywhere. Its *width* is 1.1 km at the equator, 0.87 km
in San Francisco, 0.69 km in London, and it collapses to nothing at the poles. So the cells
in this grid are not the same size, and a density comparison between London and Singapore
made on this grid is comparing rectangles of different areas. That is fine for "find the
busiest cell in one city" and wrong for "compare density across cities".

If the comparison is across latitudes, either scale the longitude bin by the cosine of the
latitude, or bin in a projected coordinate system (Web Mercator metres) before you snap.
Deciding which is a modelling question, not a query question. Do decide, though, rather than
discovering it in a chart.
:::

## Boxes, not bins

Filtering to a region is a plain range predicate, and it does not need the grid at all.
`between` reads better than four comparisons, and it is one expression rather than four:

```python
bay_area = pickups.filter(
    col("lat").between(37.7, 37.9) & col("lon").between(-122.5, -122.2)
)
print(bay_area.to_pydict()["id"])
# [1, 2, 3]
```

Push this *before* the binning when you can. The grid is a shuffle. The bounding box is a
filter, and Kyber pushes filters down to the scan, so the shuffle only ever sees the rows
you care about.

## See also

:::{seealso}
- {doc}`Time series rollups <time-series-rollups>`: `dt.truncate` is the same snap-to-a-grid
  move, one dimension and a calendar.
- {doc}`Basket analysis <basket-analysis>`: another recipe whose difficulty is entirely in
  choosing a key that groups.
- {doc}`Filtering <../../user-guide/filtering>`: `between`, and how predicates reach the scan.
- {doc}`Aggregations <../../user-guide/aggregations>`: the `group_by` the grid feeds.
- {doc}`Expressions API <../../api/expressions>`: `floor`, `concat_ws`, `width_bucket`.
- {doc}`Aggregation internals <../../deep-dives/aggregation-internals>`: what the hash shuffle
  on the cell key actually does.
:::
