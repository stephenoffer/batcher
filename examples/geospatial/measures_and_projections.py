"""Measuring geometry, and getting the units right.

The planar functions answer in the coordinate system's own units. On EPSG:4326 that is
degrees, which is not a distance. This script shows the three ways to get metres, in
increasing order of what they cost, and the transforms and simplification that go with
them.

    python examples/geospatial/measures_and_projections.py
"""

from __future__ import annotations

import batcher as bt
from batcher import col


def planar_measurement_answers_in_coordinate_units() -> None:
    shapes = bt.from_pydict(
        {
            "g": [
                "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))",
                "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (4 4, 6 4, 6 6, 4 6, 4 4))",
                "LINESTRING(0 0, 3 4)",
                "POINT(1 1)",
            ]
        }
    )
    print("--- area, length and perimeter measure three different things ---")
    print(
        shapes.select(
            kind=bt.st_geometry_type(col("g")),
            # Holes are subtracted; a non-areal geometry has zero area, not null.
            area=bt.st_area(col("g")),
            # Chains only. A polygon reports zero, matching PostGIS.
            length=bt.st_length(col("g")),
            # Polygon boundaries only, holes included.
            perimeter=bt.st_perimeter(col("g")),
        ).to_pydict()
    )

    pairs = bt.from_pydict(
        {
            "a": ["POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))", "LINESTRING(0 0, 10 0)"],
            "b": ["POINT(6 2)", "LINESTRING(0 1, 10 1)"],
        }
    )
    print("--- the distance family ---")
    print(
        pairs.select(
            nearest=bt.st_distance(col("a"), col("b")),
            furthest=bt.st_max_distance(col("a"), col("b")),
            # How far apart two shapes are at their worst-matching point: the standard
            # measure of "are these the same shape".
            hausdorff=bt.st_hausdorff_distance(col("a"), col("b")),
        ).to_pydict()
    )

    bearings = bt.from_pydict(
        {
            "from": ["POINT(0 0)"] * 4,
            "to": ["POINT(0 1)", "POINT(1 0)", "POINT(0 -1)", "POINT(-1 0)"],
        }
    )
    print("--- azimuth: radians clockwise from north ---")
    print(bearings.select(rad=bt.st_azimuth(col("from"), col("to")).round(4)).to_pydict())


def geodesic_measurement_answers_in_metres() -> None:
    legs = bt.from_pydict(
        {
            "leg": ["SF to London", "London to Paris"],
            "a": ["POINT(-122.4194 37.7749)", "POINT(-0.1278 51.5074)"],
            "b": ["POINT(-0.1278 51.5074)", "POINT(2.3522 48.8566)"],
        }
    )
    print("--- planar degrees versus spherical and ellipsoidal metres ---")
    print(
        legs.select(
            "leg",
            degrees=bt.st_distance(col("a"), col("b")).round(2),
            # Haversine: about 0.5% accurate, cheap, no failure mode.
            sphere_km=(bt.st_distance_sphere(col("a"), col("b")) / 1000).round(1),
            # Vincenty on WGS 84: sub-millimetre, iterative, an order slower.
            spheroid_km=(bt.st_distance_spheroid(col("a"), col("b")) / 1000).round(1),
        ).to_pydict()
    )

    cells = bt.from_pydict(
        {
            "where": ["equator", "60 north"],
            "g": [
                "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
                "POLYGON((0 60, 1 60, 1 61, 0 61, 0 60))",
            ],
        }
    )
    print("--- one degree of ground is not one degree of area ---")
    print(
        cells.select(
            "where",
            square_degrees=bt.st_area(col("g")),
            square_km=(bt.st_area_spheroid(col("g")) / 1e6).round(0),
            perimeter_km=(bt.st_perimeter_spheroid(col("g")) / 1000).round(0),
        ).to_pydict()
    )

    routes = bt.from_pydict({"g": ["LINESTRING(0 0, 1 0, 1 1)"]})
    print("--- geodesic chain length ---")
    print(routes.select(km=(bt.st_length_spheroid(col("g")) / 1000).round(1)).to_pydict())


def project_once_then_measure_in_metres() -> None:
    """Usually the right answer for a whole pipeline."""
    sites = bt.from_pydict(
        {
            "site": ["San Francisco", "Sydney"],
            "lon": [-122.4194, 151.2093],
            "lat": [37.7749, -33.8688],
        }
    )
    print("--- pick the local zone, then project into it ---")
    zoned = sites.with_columns(
        zone=bt.st_utm_zone(col("lon")),
        epsg=bt.st_utm_epsg(col("lon"), col("lat")),
        wgs84=bt.st_point(col("lon"), col("lat")),
    )
    print(zoned.select("site", "zone", "epsg").to_pydict())

    # `st_transform` labels the result with the target SRID, so a later transform knows
    # where it is starting from.
    projected = zoned.select(
        "site",
        mercator=bt.st_as_text(bt.st_transform(col("wgs84"), 4326, 3857)),
        equal_area_srid=bt.st_srid(bt.st_transform(col("wgs84"), 4326, 6933)),
    )
    print(projected.to_pydict())

    # In a projected system the planar functions answer in metres, so a buffer, an area
    # and a distance are all directly meaningful.
    metres = bt.from_pydict({"g": ["POINT(551131 4180000)"]})
    print("--- planar functions in a metre-based CRS ---")
    print(
        metres.select(
            around_100m=bt.st_area(bt.st_buffer(col("g"), 100.0, 32)).round(0),
            box=bt.st_as_text(bt.st_expand(col("g"), 100.0, 100.0)),
        ).to_pydict()
    )


def reshape_before_shuffling() -> None:
    """Vertex count drives every cost; simplification is the biggest lever on it."""
    detailed = bt.from_pydict({"g": ["LINESTRING(0 0, 1 0.001, 2 0, 3 0.002, 4 0, 5 0.001, 6 0)"]})
    simple = bt.st_simplify(col("g"), 0.01)
    print("--- simplify, and measure what it cost you ---")
    print(
        detailed.select(
            before=bt.st_num_points(col("g")),
            after=bt.st_num_points(simple),
            error=bt.st_hausdorff_distance(col("g"), simple).round(4),
        ).to_pydict()
    )

    messy = bt.from_pydict({"g": ["LINESTRING(0 0, 0 0, 1.234 5.678, 1.234 5.678, 3 3)"]})
    print("--- snapping and thinning make a column compress and join ---")
    snapped = bt.st_snap_to_grid(col("g"), 0.5)
    print(
        messy.select(
            snapped=bt.st_as_text(snapped),
            thinned=bt.st_as_text(bt.st_remove_repeated_points(snapped, 0.0)),
        ).to_pydict()
    )

    long_leg = bt.from_pydict({"g": ["LINESTRING(0 0, 10 0)"]})
    print("--- densify before reprojecting a long segment ---")
    print(
        long_leg.select(
            before=bt.st_num_points(col("g")),
            after=bt.st_num_points(bt.st_segmentize(col("g"), 2.0)),
        ).to_pydict()
    )


def normalize_and_transform_shapes() -> None:
    """Affine transforms cannot invalidate a geometry; normalization fixes conventions."""
    one = bt.from_pydict({"g": ["POLYGON((0 0, 0 4, 4 4, 4 0, 0 0))"]})
    print("--- affine transforms preserve structure ---")
    print(
        one.select(
            moved=bt.st_as_text(bt.st_translate(col("g"), 10.0, 10.0)),
            scaled_area=bt.st_area(bt.st_scale(col("g"), 2.0, 3.0)),
            turned=bt.st_as_text(bt.st_rotate(col("g"), 3.141592653589793)),
            # Every one of the above is a special case of this.
            general=bt.st_as_text(bt.st_affine(col("g"), 1.0, 0.0, 0.0, 1.0, 5.0, 6.0)),
        ).to_pydict()
    )

    print("--- winding, dimension and direction ---")
    print(
        one.select(
            ccw_area=bt.st_area(bt.st_force_polygon_ccw(col("g"))),
            cw_area=bt.st_area(bt.st_force_polygon_cw(col("g"))),
            reversed_ring=bt.st_as_text(bt.st_reverse(col("g"))),
            flat=bt.st_as_text(bt.st_force_2d(bt.st_force_3d(col("g"), 7.0))),
            raised=bt.st_has_z(bt.st_force_3d(col("g"), 7.0)),
        ).to_pydict()
    )

    swapped = bt.from_pydict({"g": ["POINT(37.7749 -122.4194)"]})
    print("--- the fix for a lat/lon column loaded as lon/lat ---")
    print(swapped.select(fixed=bt.st_as_text(bt.st_flip_coordinates(col("g")))).to_pydict())

    print("--- derived shapes: hull, centroid, and a point that is really on the shape ---")
    crescent = bt.from_pydict(
        {"g": ["POLYGON((0 0, 10 0, 10 2, 2 2, 2 8, 10 8, 10 10, 0 10, 0 0))"]}
    )
    print(
        crescent.select(
            hull_area=bt.st_area(bt.st_convex_hull(col("g"))),
            own_area=bt.st_area(col("g")),
            centroid_inside=bt.st_contains(col("g"), bt.st_centroid(col("g"))),
            surface_inside=bt.st_contains(col("g"), bt.st_point_on_surface(col("g"))),
            envelope=bt.st_as_text(bt.st_envelope(col("g"))),
        ).to_pydict()
    )

    print("--- collect concatenates without computing an overlay ---")
    two = bt.from_pydict({"a": ["POINT(0 0)"], "b": ["POINT(4 4)"]})
    print(
        two.select(
            joined=bt.st_as_text(bt.st_collect(col("a"), col("b"))),
            span=bt.st_as_text(bt.st_envelope(bt.st_collect(col("a"), col("b")))),
        ).to_pydict()
    )


def positions_along_a_route() -> None:
    """Linear referencing: the vocabulary route and network data is described in."""
    road = bt.from_pydict({"g": ["LINESTRING(0 0, 10 0, 10 10)"], "fix": ["POINT(4 7)"]})
    print("--- interpolate and locate are exact inverses ---")
    print(
        road.select(
            halfway=bt.st_as_text(bt.st_line_interpolate_point(col("g"), 0.5)),
            where=bt.st_line_locate_point(col("g"), col("fix")).round(4),
            stretch=bt.st_as_text(bt.st_line_substring(col("g"), 0.25, 0.75)),
        ).to_pydict()
    )

    print("--- snapping a fix to the road, and drawing the gap ---")
    print(
        road.select(
            snapped=bt.st_as_text(bt.st_closest_point(col("g"), col("fix"))),
            gap=bt.st_as_text(bt.st_shortest_line(col("g"), col("fix"))),
        ).to_pydict()
    )

    origin = bt.from_pydict({"g": ["POINT(0 0)"]})
    print("--- travel a geodesic distance along a bearing ---")
    print(
        origin.select(
            north_111km=bt.st_as_text(bt.st_project(col("g"), 111195.0, 0.0)),
            east_111km=bt.st_as_text(bt.st_project(col("g"), 111195.0, 90.0)),
        ).to_pydict()
    )


def main() -> None:
    planar_measurement_answers_in_coordinate_units()
    geodesic_measurement_answers_in_metres()
    project_once_then_measure_in_metres()
    reshape_before_shuffling()
    normalize_and_transform_shapes()
    positions_along_a_route()


if __name__ == "__main__":
    main()
