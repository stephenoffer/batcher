"""Reading a geometry in, and writing one back out.

A geometry column is WKB in an Arrow `Binary` column — the same bytes GeoParquet,
PostGIS, GeoPackage and DuckDB spatial store, so a column round-trips through any of
them with no conversion pass. These are the functions that cross between that
representation and the text encodings people and other systems use.

Two things worth knowing before reaching for one of these:

**You usually do not need `st_geom_from_text`.** Every geospatial function accepts a
text column directly and parses it, detecting WKT, EWKT, GeoJSON and hex WKB by
content. Wrapping a column in a parse function is only worth it when the column is read
many times in one query, because it converts once instead of once per call.

**A geometry that will not parse becomes null, it does not raise.** One corrupt row in
a hundred million should not abort a scan, and `st_is_valid_reason` names every bad row
and why. A query bug — a negative radius, an unsupported EPSG code — does raise,
because that is wrong on every row rather than on one.
"""

from __future__ import annotations

from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.geo._build import geo_call, geometry

__all__ = [
    "st_as_binary",
    "st_as_ewkb",
    "st_as_ewkt",
    "st_as_geojson",
    "st_as_hex_wkb",
    "st_as_text",
    "st_geom_from_geohash",
    "st_geom_from_geojson",
    "st_geom_from_text",
    "st_geom_from_wkb",
]


def st_geom_from_text(text: Expr | str) -> Expr:
    """Parse a geometry from text, detecting the encoding by content.

    Accepts all four text encodings rather than one, because a geometry column in
    the wild is whichever the producer wrote and asking the user to know which is a
    wart. Detection is unambiguous: GeoJSON starts with `{`, hex WKB is all hex digits,
    everything else is WKT.

    Args:
        text: A geometry as WKT, EWKT, GeoJSON, or hex WKB.

    Returns:
        The geometry, or null when the text is not a geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {'t': ['POINT(1 2)', '{"type":"Point","coordinates":[3,4]}', 'not a
            ...     geometry']}
            ... )
            >>> got = bt.st_as_text(bt.st_geom_from_text(bt.col("t")))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POINT(1 2)', 'POINT(3 4)', None]}
    """
    return geo_call("st_geom_from_text", geometry(text))


def st_geom_from_wkb(wkb: Expr | str) -> Expr:
    """Validate a Binary column as geometry.

    The identity on a well-formed geometry column, which is what makes it useful: it
    turns 'this Binary column is supposed to hold WKB' into a checked claim, with the
    bad rows appearing as nulls.

    Args:
        wkb: A Binary column holding WKB or EWKB.

    Returns:
        The geometry, or null when the bytes are not a geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))']})
            >>> got = bt.st_geometry_type(bt.st_geom_from_wkb(bt.st_as_binary(bt.col("g"))))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POLYGON']}
    """
    return geo_call("st_geom_from_wkb", geometry(wkb))


def st_geom_from_geojson(json: Expr | str) -> Expr:
    """Parse a geometry from a GeoJSON object.

    A `Feature` is unwrapped to its geometry; its properties belong to the row, not
    to the geometry, so they are the IO layer's business. A multi-feature
    `FeatureCollection` is refused rather than collapsed, because it holds many rows.

    RFC 7946 fixes the coordinates as WGS 84 longitude-then-latitude, so the result
    carries SRID 4326 whether or not the document says so.

    Args:
        json: An RFC 7946 geometry or Feature object.

    Returns:
        The geometry, or null when the JSON is not a geometry.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'j': ['{"type":"Point","coordinates":[30,10]}']})
            >>> got = bt.st_srid(bt.st_geom_from_geojson(bt.col("j")))
            >>> ds.select(v=got).to_pydict()
            {'v': [4326]}
    """
    return geo_call("st_geom_from_geojson", geometry(json))


def st_geom_from_geohash(geohash: Expr | str) -> Expr:
    """The rectangle a geohash names.

    The inverse of `st_geohash`: a hash is a cell, and this is that cell as a
    geometry you can intersect against. Useful for turning a coarse prefix filter into
    an exact region test in the same query.

    Args:
        geohash: A base-32 geohash string.

    Returns:
        The cell as a rectangle, or null when the string is not a geohash.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'h': ['9q8yyk']})
            >>> got = bt.st_as_text(bt.st_centroid(bt.st_geom_from_geohash(bt.col("h"))))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POINT(-122.4151611328125 37.77374267578125)']}
    """
    return geo_call("st_geom_from_geohash", geometry(geohash))


def st_as_text(geom: Expr | str) -> Expr:
    """Render a geometry as WKT.

    The canonical rendering: uppercase keywords, one space after each, no trailing
    zeros. Stable enough to group by and to use as a golden value in a test.

    Args:
        geom: The geometry.

    Returns:
        The geometry as WKT, or null for a null input.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(2 2)']})
            >>> got = bt.st_as_text(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': ['POINT(2 2)']}
    """
    return geo_call("st_as_text", geometry(geom))


def st_as_ewkt(geom: Expr | str) -> Expr:
    """Render a geometry as WKT with its SRID.

    PostGIS's extended WKT: an `SRID=<n>;` prefix when the geometry carries one, and
    plain WKT when it does not. Use this rather than `st_as_text` when the CRS has to
    survive the trip out.

    Args:
        geom: The geometry.

    Returns:
        The geometry as EWKT, or null for a null input.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(2 2)']})
            >>> got = bt.st_as_ewkt(bt.st_set_srid(bt.col("g"), 4326))
            >>> ds.select(v=got).to_pydict()
            {'v': ['SRID=4326;POINT(2 2)']}
    """
    return geo_call("st_as_ewkt", geometry(geom))


def st_as_binary(geom: Expr | str) -> Expr:
    """Render a geometry as portable WKB, without an SRID.

    The maximally portable spelling, and what to write when the consumer is a system
    that does not understand PostGIS's EWKB extension. Use `st_as_ewkb` when the SRID
    must travel with the bytes.

    Args:
        geom: The geometry.

    Returns:
        The geometry as WKB bytes, or null for a null input.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(2 2)']})
            >>> got = bt.st_as_binary(bt.col("g")).str.len()
            >>> ds.select(v=got).to_pydict()
            {'v': [21]}
    """
    return geo_call("st_as_binary", geometry(geom))


def st_as_ewkb(geom: Expr | str) -> Expr:
    """Render a geometry as EWKB, carrying its SRID.

    The encoding a geometry column uses internally, so this is the identity for a
    column that already has one. It exists for writing out to a system that expects
    PostGIS bytes.

    Args:
        geom: The geometry.

    Returns:
        The geometry as EWKB bytes, or null for a null input.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(2 2)']})
            >>> got = bt.st_as_ewkb(bt.st_set_srid(bt.col("g"), 4326)).str.len()
            >>> ds.select(v=got).to_pydict()
            {'v': [None]}
    """
    return geo_call("st_as_ewkb", geometry(geom))


def st_as_hex_wkb(geom: Expr | str) -> Expr:
    """Render a geometry as hex-encoded EWKB.

    The spelling PostGIS's text protocol produces, and the one a geometry survives
    being pasted into a SQL client as. Round-trips through `st_geom_from_text`.

    Args:
        geom: The geometry.

    Returns:
        The geometry as a lowercase hex string, or null for a null input.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(1 2)']})
            >>> got = bt.st_as_hex_wkb(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': ['0101000000000000000000f03f0000000000000040']}
    """
    return geo_call("st_as_hex_wkb", geometry(geom))


def st_as_geojson(geom: Expr | str) -> Expr:
    """Render a geometry as a GeoJSON geometry object.

    The geometry object only, not a `Feature`: the properties of a feature are the
    row's other columns, so assembling a `FeatureCollection` is the GeoJSON sink's job
    and it has the whole batch to do it with.

    Args:
        geom: The geometry.

    Returns:
        The geometry as an RFC 7946 object, or null for a null input.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({'g': ['POINT(30 10)']})
            >>> got = bt.st_as_geojson(bt.col("g"))
            >>> ds.select(v=got).to_pydict()
            {'v': ['{"type":"Point","coordinates":[30,10]}']}
    """
    return geo_call("st_as_geojson", geometry(geom))
