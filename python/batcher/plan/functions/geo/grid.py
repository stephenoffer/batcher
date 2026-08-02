"""Turning a position into a discrete cell you can group, sort and join on.

Latitude and longitude are floats, so no two observations share a value and
`GROUP BY lat, lon` returns the input. A grid function turns a position into a cell id —
an ordinary string or integer — which the engine then hashes, sorts, shuffles and joins
at full speed with no spatial index at all. This is the cheapest useful geospatial thing
in the library and usually the right first move.

They differ in what the id gives you beyond grouping:

| Function | Cell id | Also gives you |
| --- | --- | --- |
| `geohash_encode` | base-32 string | prefix containment: `LIKE 'u09%'` is a region filter |
| `st_quadkey`, `st_tile_x`/`st_tile_y` | tile address | the exact grid map tiles are served on |
| `st_s2_cell` | `Int64` Hilbert index | near-equal-area cells, and a region as a `BETWEEN` |
| `st_hex_bin` | packed `Int64` | six equidistant neighbours, for unbiased density |

`st_s2_cell` deserves the attention: its ids sort spatially, so sorting a table by one
clusters nearby rows onto the same pages, and every descendant of a cell forms a single
contiguous integer range. That turns "everything in this region" into a range predicate
on an `Int64` column.

`st_hex_bin` is a **planar** hex grid, not H3. H3's cells live on an icosahedron and its
indexes are not these, so do not join one against the other. Project with
`st_transform` before binning.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.geo._build import geo_call, geometry, value

__all__ = [
    "geohash_decode_lat",
    "geohash_decode_lon",
    "geohash_encode",
    "st_geohash",
    "st_hex_bin",
    "st_hex_center_x",
    "st_hex_center_y",
    "st_quadkey",
    "st_s2_cell",
    "st_s2_cell_parent",
    "st_tile_x",
    "st_tile_y",
    "st_utm_epsg",
    "st_utm_zone",
]


def st_geohash(geom: Expr | str, precision: Expr | int) -> Expr:
    """The geohash cell of a lon/lat geometry's centroid.

    Reduces the geometry to its centroid, which is what a cell id of an extended
    shape can mean. For plain coordinate columns use `geohash_encode`, which skips
    building a geometry at all.

    Args:
        geom: A lon/lat geometry.
        precision: Characters of output, 1 to 12.

    Returns:
        The geohash of the geometry's centroid, or null for an empty geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(-122.4194 37.7749)']})
            >>> got = bt.st_geohash(bt.col("g"), 6)
            >>> ds.select(v=got).to_pydict()
            {'v': ['9q8yyk']}
    """
    return geo_call("st_geohash", geometry(geom), value(precision))


def geohash_encode(lon: Expr | float, lat: Expr | float, precision: Expr | int) -> Expr:
    """The geohash cell of a longitude and latitude.

    Hashes nest: a 6-character hash always extends the 5-character one for the same
    position, so a coarser rollup is `substr` rather than a recomputation.

    Prefix proximity is one-directional. Sharing a prefix means being close; being
    close does **not** mean sharing a prefix, because two positions either side of a
    cell boundary can differ in the first character. A proximity query has to cover the
    neighbouring cells too.

    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees.
        precision: Characters of output, 1 to 12.

    Returns:
        The geohash string, or null for a null coordinate.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'lon': [-122.4194], 'lat': [37.7749]})
            >>> got = bt.geohash_encode(bt.col("lon"), bt.col("lat"), 6)
            >>> ds.select(v=got).to_pydict()
            {'v': ['9q8yyk']}
    """
    return geo_call("geohash_encode", value(lon), value(lat), value(precision))


def geohash_decode_lon(geohash: Expr | str) -> Expr:
    """The longitude of a geohash cell's centre.

    Pairs with `geohash_decode_lat`. For the cell as a shape rather than a point, use
    `st_geom_from_geohash`.

    Args:
        geohash: A base-32 geohash string.

    Returns:
        The cell centre's longitude, or null when the string is not a geohash.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'h': ['9q8yyk']})
            >>> got = bt.geohash_decode_lon(bt.col("h")).round(4)
            >>> ds.select(v=got).to_pydict()
            {'v': [-122.4152]}
    """
    return geo_call("geohash_decode_lon", geometry(geohash))


def geohash_decode_lat(geohash: Expr | str) -> Expr:
    """The latitude of a geohash cell's centre.

    Pairs with `geohash_decode_lon`.

    Args:
        geohash: A base-32 geohash string.

    Returns:
        The cell centre's latitude, or null when the string is not a geohash.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'h': ['9q8yyk']})
            >>> got = bt.geohash_decode_lat(bt.col("h")).round(4)
            >>> ds.select(v=got).to_pydict()
            {'v': [37.7737]}
    """
    return geo_call("geohash_decode_lat", geometry(geohash))


def st_tile_x(lon: Expr | float, lat: Expr | float, zoom: Expr | int) -> Expr:
    """The slippy-map tile column containing a position.

    Pairs with `st_tile_y`. Latitudes past the Web Mercator limit are clamped rather
    than refused: a fix at 87 degrees north is a real observation, and the top tile is
    the honest answer for it.

    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees.
        zoom: The zoom level, 0 to 30.

    Returns:
        The tile column, or null for a null coordinate.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'lon': [-122.4194], 'lat': [37.7749]})
            >>> got = bt.st_tile_x(bt.col("lon"), bt.col("lat"), 12)
            >>> ds.select(v=got).to_pydict()
            {'v': [655]}
    """
    return geo_call("st_tile_x", value(lon), value(lat), value(zoom))


def st_tile_y(lon: Expr | float, lat: Expr | float, zoom: Expr | int) -> Expr:
    """The slippy-map tile row containing a position.

    **`y` increases southward.** Row 0 is the top of the map, near 85 degrees north.
    That is the slippy-map convention and it is the single most common source of
    off-by-a-hemisphere bugs in tile code.

    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees.
        zoom: The zoom level, 0 to 30.

    Returns:
        The tile row, or null for a null coordinate.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'lon': [-122.4194], 'lat': [37.7749]})
            >>> got = bt.st_tile_y(bt.col("lon"), bt.col("lat"), 12)
            >>> ds.select(v=got).to_pydict()
            {'v': [1583]}
    """
    return geo_call("st_tile_y", value(lon), value(lat), value(zoom))


def st_quadkey(lon: Expr | float, lat: Expr | float, zoom: Expr | int) -> Expr:
    """The Bing quadkey of the tile containing a position.

    The same tile as `st_tile_x`/`st_tile_y`, written as one base-4 string with one
    digit per zoom level. Like a geohash it has the prefix property, so a rollup across
    zoom levels is `substr` on a string column.

    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees.
        zoom: The zoom level, 0 to 30.

    Returns:
        The quadkey string, or null for a null coordinate.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'lon': [-122.4194], 'lat': [37.7749]})
            >>> got = bt.st_quadkey(bt.col("lon"), bt.col("lat"), 8)
            >>> ds.select(v=got).to_pydict()
            {'v': ['02301020']}
    """
    return geo_call("st_quadkey", value(lon), value(lat), value(zoom))


def st_s2_cell(lon: Expr | float, lat: Expr | float, level: Expr | int) -> Expr:
    """The S2 cell containing a position, as an Int64.

    Two properties make this the best general-purpose spatial key. Cells are
    **near-equal-area** anywhere on Earth, unlike a degree grid whose cells vanish at
    the poles, so a density comparison across latitudes is meaningful. And the id is a
    Hilbert index, so it **sorts spatially** — sorting by it clusters neighbours onto
    the same pages.

    Ids are byte-compatible with the reference implementation, so they join against
    S2 cells produced anywhere else.

    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees.
        level: The S2 level, 0 to 30.

    Returns:
        The S2 cell id, or null for a null coordinate.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'lon': [-122.4194], 'lat': [37.7749]})
            >>> got = bt.st_s2_cell(bt.col("lon"), bt.col("lat"), 12)
            >>> ds.select(v=got).to_pydict()
            {'v': [-9185794508988612608]}
    """
    return geo_call("st_s2_cell", value(lon), value(lat), value(level))


def st_s2_cell_parent(cell: Expr | int, level: Expr | int) -> Expr:
    """The ancestor of an S2 cell at a coarser level.

    The rollup operator: bin once at a fine level, then aggregate to any coarser one
    without revisiting the coordinates. A cell and all its descendants share an id
    prefix, which is why this is a bit mask rather than a recomputation.

    Args:
        cell: An S2 cell id.
        level: The coarser level to roll up to.

    Returns:
        The ancestor cell id, or null when the level is finer than the cell's own.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'lon': [-122.4194], 'lat': [37.7749]})
            >>> fine = bt.st_s2_cell(bt.col("lon"), bt.col("lat"), 12)
            >>> coarse = bt.st_s2_cell(bt.col("lon"), bt.col("lat"), 6)
            >>> got = bt.st_s2_cell_parent(fine, 6) == coarse
            >>> ds.select(v=got).to_pydict()
            {'v': [True]}
    """
    return geo_call("st_s2_cell_parent", value(cell), value(level))


def st_hex_bin(x: Expr | float, y: Expr | float, size: Expr | float) -> Expr:
    """The hexagonal cell containing a projected position.

    Hexagons beat squares for density because all six neighbours are equidistant,
    while a square's diagonal neighbour is 41% further — which biases any smoothing or
    adjacency logic along the axes.

    **Project first.** This bins whatever coordinates it is given, so feed it metres
    from `st_transform`, not degrees. And it is not H3: the ids are a planar axial
    address, not an icosahedral index.

    Args:
        x: The projected x ordinate.
        y: The projected y ordinate.
        size: The hexagon's centre-to-corner distance.

    Returns:
        A packed cell id, or null for a null coordinate.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'x': [100.0, 103.0], 'y': [50.0, 52.0]})
            >>> got = bt.st_hex_bin(bt.col("x"), bt.col("y"), 10.0)
            >>> ds.select(v=got).to_pydict()
            {'v': [2305843025319821311, 2305843025319821312]}
    """
    return geo_call("st_hex_bin", value(x), value(y), value(size))


def st_hex_center_x(cell: Expr | int, size: Expr | float) -> Expr:
    """The x ordinate of a hex cell's centre.

    Recovers a plottable position from a group key. The `size` must match the one
    `st_hex_bin` used, since the id is an axial address rather than a position.

    Args:
        cell: A packed hex cell id.
        size: The size the cell was binned at.

    Returns:
        The cell centre's x ordinate, or null for a null cell.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'x': [100.0], 'y': [50.0]})
            >>> cell = bt.st_hex_bin(bt.col("x"), bt.col("y"), 10.0)
            >>> got = bt.st_hex_center_x(cell, 10.0).round(4)
            >>> ds.select(v=got).to_pydict()
            {'v': [105.0]}
    """
    return geo_call("st_hex_center_x", value(cell), value(size))


def st_hex_center_y(cell: Expr | int, size: Expr | float) -> Expr:
    """The y ordinate of a hex cell's centre.

    Pairs with `st_hex_center_x`.

    Args:
        cell: A packed hex cell id.
        size: The size the cell was binned at.

    Returns:
        The cell centre's y ordinate, or null for a null cell.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'x': [100.0], 'y': [50.0]})
            >>> cell = bt.st_hex_bin(bt.col("x"), bt.col("y"), 10.0)
            >>> got = bt.st_hex_center_y(cell, 10.0).round(4)
            >>> ds.select(v=got).to_pydict()
            {'v': [43.3013]}
    """
    return geo_call("st_hex_center_y", value(cell), value(size))


def st_utm_zone(lon: Expr | float) -> Expr:
    """The UTM zone number covering a longitude.

    Use `st_utm_epsg` when the answer is going into `st_transform`; the bare zone
    number does not say which hemisphere.

    Args:
        lon: Longitude in degrees.

    Returns:
        The zone number from 1 to 60, or null for a null longitude.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'lon': [-122.4194]})
            >>> got = bt.st_utm_zone(bt.col("lon"))
            >>> ds.select(v=got).to_pydict()
            {'v': [10]}
    """
    return geo_call("st_utm_zone", value(lon))


def st_utm_epsg(lon: Expr | float, lat: Expr | float) -> Expr:
    """The EPSG code of the UTM zone covering a position.

    Feeds `st_transform` directly, which is the point: project into the local zone and
    every planar measurement afterwards is in metres, accurate to better than a tenth
    of a percent.

    Northern zones are 326xx and southern 327xx, so a dataset spanning the equator has
    no single code. That is a property of UTM, not a limitation here.

    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees.

    Returns:
        The EPSG code, or null for a null coordinate.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'lon': [-122.4194, 151.2093], 'lat': [37.7749, -33.8688]})
            >>> got = bt.st_utm_epsg(bt.col("lon"), bt.col("lat"))
            >>> ds.select(v=got).to_pydict()
            {'v': [32610, 32756]}
    """
    return geo_call("st_utm_epsg", value(lon), value(lat))
