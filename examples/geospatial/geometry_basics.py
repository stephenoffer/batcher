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
    print(
        located.select(
            "name",
            wkt=bt.st_as_text(col("flat")),
            with_z=bt.st_as_text(col("solid")),
            dims=bt.st_coord_dim(col("solid")),
            has_z=bt.st_has_z(col("solid")),
            z=bt.st_z(col("solid")),
        ).to_pydict()
    )

    # Lines and polygons are built from geometries rather than from ordinates.
    route = bt.from_pydict({"a": ["POINT(0 0)"], "b": ["POINT(3 4)"]})
    print("--- a line between two points ---")
    line = bt.st_make_line(col("a"), col("b"))
    print(
        route.select(
            wkt=bt.st_as_text(line),
            length=bt.st_length(line),
            start=bt.st_as_text(bt.st_start_point(line)),
            end=bt.st_as_text(bt.st_end_point(line)),
            second=bt.st_as_text(bt.st_point_n(line, 2)),
        ).to_pydict()
    )

    ring = bt.from_pydict({"r": ["LINESTRING(0 0, 6 0, 6 6, 0 6, 0 0)"]})
    print("--- a polygon from a closed chain ---")
    poly = bt.st_make_polygon(col("r"))
    print(
        ring.select(
            area=bt.st_area(poly),
            perimeter=bt.st_perimeter(poly),
            ring_wkt=bt.st_as_text(bt.st_exterior_ring(poly)),
        ).to_pydict()
    )

    # An explicit rectangle, for a literal region filter.
    box = bt.from_pydict({"n": [1]})
    print("--- an explicit rectangle ---")
    print(box.select(wkt=bt.st_as_text(bt.st_make_envelope(0.0, 0.0, 2.0, 3.0))).to_pydict())


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
    print(
        shapes.select(
            kind=bt.st_geometry_type(col("g")),
            dim=bt.st_dimension(col("g")),
            points=bt.st_num_points(col("g")),
            members=bt.st_num_geometries(col("g")),
            holes=bt.st_num_interior_rings(col("g")),
            collection=bt.st_is_collection(col("g")),
        ).to_pydict()
    )

    print("--- bounds: the four cheapest useful numbers ---")
    print(
        shapes.select(
            xmin=bt.st_xmin(col("g")),
            ymin=bt.st_ymin(col("g")),
            xmax=bt.st_xmax(col("g")),
            ymax=bt.st_ymax(col("g")),
        ).to_pydict()
    )

    donut = bt.from_pydict({"g": ["POLYGON((0 0, 8 0, 8 8, 0 8, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))"]})
    print("--- walking a polygon's rings and a collection's members ---")
    print(
        donut.select(
            hole=bt.st_as_text(bt.st_interior_ring_n(col("g"), 1)),
            boundary=bt.st_geometry_type(bt.st_boundary(col("g"))),
        ).to_pydict()
    )
    multi = bt.from_pydict({"g": ["MULTIPOINT((0 0), (5 5), (9 1))"]})
    print(
        multi.select(
            first=bt.st_as_text(bt.st_geometry_n(col("g"), 1)),
            third=bt.st_as_text(bt.st_geometry_n(col("g"), 3)),
        ).to_pydict()
    )


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
    print(
        parcels.select(
            "id",
            valid=bt.st_is_valid(col("g")),
            why=bt.st_is_valid_reason(col("g")),
            empty=bt.st_is_empty(col("g")),
        ).to_pydict()
    )

    # Row 4 parses as nothing, so every function over it is null rather than an error.
    broken = parcels.filter(bt.st_geometry_type(col("g")).is_null())
    print("unparseable row ids:", broken.select("id").to_pydict()["id"])

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
    print(
        chains.select(
            closed=bt.st_is_closed(col("g")),
            ring=bt.st_is_ring(col("g")),
            simple=bt.st_is_simple(col("g")),
        ).to_pydict()
    )


def render_for_another_system() -> None:
    """The codecs, and the SRID that rides along with some of them."""
    one = bt.from_pydict({"g": ["POINT(30 10)"]})
    labelled = bt.st_set_srid(col("g"), 4326)
    print("--- one geometry, five renderings ---")
    print(
        one.select(
            wkt=bt.st_as_text(col("g")),
            ewkt=bt.st_as_ewkt(labelled),
            geojson=bt.st_as_geojson(col("g")),
            hex_wkb=bt.st_as_hex_wkb(col("g")),
            srid=bt.st_srid(labelled),
        ).to_pydict()
    )

    # Every text encoding round-trips back to the same geometry.
    encoded = one.select(
        as_wkt=bt.st_as_text(col("g")),
        as_json=bt.st_as_geojson(col("g")),
        as_hex=bt.st_as_hex_wkb(col("g")),
        as_wkb=bt.st_as_binary(col("g")),
        as_ewkb=bt.st_as_ewkb(labelled),
    )
    print("--- and back again ---")
    print(
        encoded.select(
            from_wkt=bt.st_as_text(bt.st_geom_from_text(col("as_wkt"))),
            from_json=bt.st_as_text(bt.st_geom_from_geojson(col("as_json"))),
            from_hex=bt.st_as_text(bt.st_geom_from_text(col("as_hex"))),
            from_wkb=bt.st_as_text(bt.st_geom_from_wkb(col("as_wkb"))),
            from_ewkb=bt.st_srid(bt.st_geom_from_wkb(col("as_ewkb"))),
        ).to_pydict()
    )


def main() -> None:
    build_geometry_from_coordinate_columns()
    read_a_geometry_apart()
    check_validity_before_trusting_anything()
    render_for_another_system()


if __name__ == "__main__":
    main()
