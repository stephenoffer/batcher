"""Getting geometry in, reading it apart, and writing it back out.

A geometry column is WKB in a Binary column, but you almost never have to say so: every
`st_*` function parses a text column directly, detecting WKT, EWKT, GeoJSON and hex WKB
by content. This script walks the constructors, the accessors that read a geometry's
parts, and the codecs that render one for another system.

    python examples/geospatial/geometry_basics.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def build_geometry_from_coordinate_columns() -> None:
    """Most tables store two float columns, not geometry. `st_point` is the bridge."""
    stations = bt.from_pydict(
        {
            "name": ["Ferry Building", "Big Ben", "Opera House"],
            "lon": [-122.3937, -0.1246, 151.2153],
            "lat": [37.7955, 51.5007, -33.8568],
            "elevation": [3.0, 96.0, 65.0],
        }
    )
    located = stations.with_columns(
        # x then y: longitude first, which is what WKT, GeoJSON and PostGIS all use.
        flat=bt.st_point(col("lon"), col("lat")),
        # The 3D form carries an elevation through every structure-preserving function.
        solid=bt.st_point_z(col("lon"), col("lat"), col("elevation")),
    )
    print("--- constructed points ---")
    points = located.select(
        "name",
        wkt=bt.st_as_text(col("flat")),
        with_z=bt.st_as_text(col("solid")),
        dims=bt.st_coord_dim(col("solid")),
        has_z=bt.st_has_z(col("solid")),
        z=bt.st_z(col("solid")),
    ).to_pydict()
    print(points)
    # `st_point` takes x then y, so a swapped pair is a silently plausible wrong answer —
    # Big Ben would land in the Indian Ocean and nothing would raise. Pin the ordering.
    assert points["wkt"][1] == "POINT(-0.1246 51.5007)"
    assert points["with_z"][0] == "POINT Z(-122.3937 37.7955 3)"
    assert points["dims"] == [3, 3, 3] and points["has_z"] == [True, True, True]
    assert points["z"] == [3.0, 96.0, 65.0]

    # Lines and polygons are built from geometries rather than from ordinates.
    route = bt.from_pydict({"a": ["POINT(0 0)"], "b": ["POINT(3 4)"]})
    print("--- a line between two points ---")
    line = bt.st_make_line(col("a"), col("b"))
    segment = route.select(
        wkt=bt.st_as_text(line),
        length=bt.st_length(line),
        start=bt.st_as_text(bt.st_start_point(line)),
        end=bt.st_as_text(bt.st_end_point(line)),
        second=bt.st_as_text(bt.st_point_n(line, 2)),
    ).to_pydict()
    print(segment)
    assert segment["wkt"] == ["LINESTRING(0 0, 3 4)"]
    assert segment["length"] == [5.0], "the 3-4-5 triangle is the point of these coordinates"
    # `st_point_n` is 1-based, so the second point is the end point, not the one past it.
    assert segment["start"] == ["POINT(0 0)"] and segment["second"] == segment["end"]

    ring = bt.from_pydict({"r": ["LINESTRING(0 0, 6 0, 6 6, 0 6, 0 0)"]})
    print("--- a polygon from a closed chain ---")
    poly = bt.st_make_polygon(col("r"))
    square = ring.select(
        area=bt.st_area(poly),
        perimeter=bt.st_perimeter(poly),
        ring_wkt=bt.st_as_text(bt.st_exterior_ring(poly)),
    ).to_pydict()
    print(square)
    assert square["area"] == [36.0], "a 6x6 square"
    assert square["perimeter"] == [24.0]

    # An explicit rectangle, for a literal region filter.
    box = bt.from_pydict({"n": [1]})
    print("--- an explicit rectangle ---")
    envelope = box.select(wkt=bt.st_as_text(bt.st_make_envelope(0.0, 0.0, 2.0, 3.0))).to_pydict()
    print(envelope)
    # (xmin, ymin, xmax, ymax), wound counter-clockwise and closed back to the first corner.
    assert envelope["wkt"] == ["POLYGON((0 0, 2 0, 2 3, 0 3, 0 0))"]


def read_a_geometry_apart() -> None:
    """The accessors: type, dimension, counts, bounds, and members."""
    shapes = bt.from_pydict(
        {
            "g": [
                "POINT(2 3)",
                "LINESTRING(0 0, 4 3)",
                "POLYGON((0 0, 8 0, 8 8, 0 8, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))",
                "MULTIPOINT((0 0), (5 5), (9 1))",
            ]
        }
    )
    print("--- what is in this column ---")
    parts = shapes.select(
        kind=bt.st_geometry_type(col("g")),
        dim=bt.st_dimension(col("g")),
        points=bt.st_num_points(col("g")),
        members=bt.st_num_geometries(col("g")),
        holes=bt.st_num_interior_rings(col("g")),
        collection=bt.st_is_collection(col("g")),
    ).to_pydict()
    print(parts)
    assert parts["kind"] == ["POINT", "LINESTRING", "POLYGON", "MULTIPOINT"]
    assert parts["dim"] == [0, 1, 2, 0], "a MULTIPOINT is still zero-dimensional"
    # The donut's 10 points are its 5-point shell plus its 5-point hole.
    assert parts["points"] == [1, 2, 10, 3]
    assert parts["holes"] == [0, 0, 1, 0]
    # `num_geometries` counts members, which is 1 for anything that is not a collection.
    assert parts["members"] == [1, 1, 1, 3]
    assert parts["collection"] == [False, False, False, True]

    print("--- bounds: the four cheapest useful numbers ---")
    bounds = shapes.select(
        xmin=bt.st_xmin(col("g")),
        ymin=bt.st_ymin(col("g")),
        xmax=bt.st_xmax(col("g")),
        ymax=bt.st_ymax(col("g")),
    ).to_pydict()
    print(bounds)
    # A point's bounding box is the point: all four numbers collapse onto it.
    assert (
        (bounds["xmin"][0], bounds["ymin"][0])
        == (bounds["xmax"][0], bounds["ymax"][0])
        == (2.0, 3.0)
    )
    assert bounds["xmax"] == [2.0, 4.0, 8.0, 9.0] and bounds["ymax"] == [3.0, 3.0, 8.0, 5.0]

    donut = bt.from_pydict({"g": ["POLYGON((0 0, 8 0, 8 8, 0 8, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))"]})
    print("--- walking a polygon's rings and a collection's members ---")
    rings = donut.select(
        hole=bt.st_as_text(bt.st_interior_ring_n(col("g"), 1)),
        boundary=bt.st_geometry_type(bt.st_boundary(col("g"))),
    ).to_pydict()
    print(rings)
    assert rings["hole"] == ["LINESTRING(2 2, 4 2, 4 4, 2 4, 2 2)"]
    # A polygon with a hole has a two-part boundary: the shell and the hole.
    assert rings["boundary"] == ["MULTILINESTRING"]
    multi = bt.from_pydict({"g": ["MULTIPOINT((0 0), (5 5), (9 1))"]})
    members = multi.select(
        first=bt.st_as_text(bt.st_geometry_n(col("g"), 1)),
        third=bt.st_as_text(bt.st_geometry_n(col("g"), 3)),
    ).to_pydict()
    print(members)
    # 1-based, like `st_point_n`: member 1 is the first, not the second.
    assert members["first"] == ["POINT(0 0)"] and members["third"] == ["POINT(9 1)"]


def check_validity_before_trusting_anything() -> None:
    """An invalid polygon makes every areal predicate wrong, silently."""
    parcels = bt.from_pydict(
        {
            "id": [1, 2, 3, 4],
            "g": [
                "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))",
                # A bowtie: the ring crosses itself.
                "POLYGON((0 0, 4 4, 4 0, 0 4, 0 0))",
                # A hole outside its shell.
                "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0), (9 9, 11 9, 11 11, 9 9))",
                "not a geometry at all",
            ],
        }
    )
    print("--- validity, with a reason per broken row ---")
    checked = parcels.select(
        "id",
        valid=bt.st_is_valid(col("g")),
        why=bt.st_is_valid_reason(col("g")),
        empty=bt.st_is_empty(col("g")),
    ).to_pydict()
    print(checked)
    # The whole point of the example: the two broken polygons must be *caught*, not
    # silently accepted, and the unparseable row must be null rather than either.
    assert checked["valid"] == [True, False, False, None]
    assert checked["why"][0] is None, "a valid polygon has no complaint"
    assert "self-intersect" in checked["why"][1]
    assert "outside the exterior ring" in checked["why"][2]

    # Row 4 parses as nothing, so every function over it is null rather than an error.
    broken = parcels.filter(bt.st_geometry_type(col("g")).is_null())
    unparseable = broken.select("id").to_pydict()["id"]
    print("unparseable row ids:", unparseable)
    assert unparseable == [4], "row 4 alone is not a geometry, and it nulls rather than raising"

    chains = bt.from_pydict(
        {
            "g": [
                "LINESTRING(0 0, 4 0, 4 4, 0 0)",
                "LINESTRING(0 0, 4 4, 4 0, 0 4, 0 0)",
                "LINESTRING(0 0, 1 1)",
            ]
        }
    )
    print("--- closed, ring and simple are three different questions ---")
    chain_kinds = chains.select(
        closed=bt.st_is_closed(col("g")),
        ring=bt.st_is_ring(col("g")),
        simple=bt.st_is_simple(col("g")),
    ).to_pydict()
    print(chain_kinds)
    # Row 2 is closed but crosses itself, so it is not simple and therefore not a ring —
    # which is exactly why these are three questions and not one.
    assert chain_kinds["closed"] == [True, True, False]
    assert chain_kinds["ring"] == [True, False, False]
    assert chain_kinds["simple"] == [True, False, True]


def render_for_another_system() -> None:
    """The codecs, and the SRID that rides along with some of them."""
    one = bt.from_pydict({"g": ["POINT(30 10)"]})
    labelled = bt.st_set_srid(col("g"), 4326)
    print("--- one geometry, five renderings ---")
    rendered = one.select(
        wkt=bt.st_as_text(col("g")),
        ewkt=bt.st_as_ewkt(labelled),
        geojson=bt.st_as_geojson(col("g")),
        hex_wkb=bt.st_as_hex_wkb(col("g")),
        srid=bt.st_srid(labelled),
    ).to_pydict()
    print(rendered)
    assert rendered["wkt"] == ["POINT(30 10)"]
    # Only the E-prefixed encodings carry the SRID; plain WKT and GeoJSON drop it.
    assert rendered["ewkt"] == ["SRID=4326;POINT(30 10)"] and rendered["srid"] == [4326]
    assert rendered["geojson"] == ['{"type":"Point","coordinates":[30,10]}']

    # Every text encoding round-trips back to the same geometry.
    encoded = one.select(
        as_wkt=bt.st_as_text(col("g")),
        as_json=bt.st_as_geojson(col("g")),
        as_hex=bt.st_as_hex_wkb(col("g")),
        as_wkb=bt.st_as_binary(col("g")),
        as_ewkb=bt.st_as_ewkb(labelled),
    )
    print("--- and back again ---")
    decoded = encoded.select(
        from_wkt=bt.st_as_text(bt.st_geom_from_text(col("as_wkt"))),
        from_json=bt.st_as_text(bt.st_geom_from_geojson(col("as_json"))),
        from_hex=bt.st_as_text(bt.st_geom_from_text(col("as_hex"))),
        from_wkb=bt.st_as_text(bt.st_geom_from_wkb(col("as_wkb"))),
        from_ewkb=bt.st_srid(bt.st_geom_from_wkb(col("as_ewkb"))),
    ).to_pydict()
    print(decoded)
    # Every text and binary encoding is lossless for the geometry itself...
    for spelling in ("from_wkt", "from_json", "from_hex", "from_wkb"):
        assert decoded[spelling] == ["POINT(30 10)"], spelling
    # ...and EWKB alone also carries the SRID back.
    assert decoded["from_ewkb"] == [4326]


def main() -> None:
    build_geometry_from_coordinate_columns()
    read_a_geometry_apart()
    check_validity_before_trusting_anything()
    render_for_another_system()


if __name__ == "__main__":
    main()
