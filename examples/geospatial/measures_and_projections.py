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
    measured = shapes.select(
        kind=bt.st_geometry_type(col("g")),
        # Holes are subtracted; a non-areal geometry has zero area, not null.
        area=bt.st_area(col("g")),
        # Chains only. A polygon reports zero, matching PostGIS.
        length=bt.st_length(col("g")),
        # Polygon boundaries only, holes included.
        perimeter=bt.st_perimeter(col("g")),
    ).to_pydict()
    print(measured)
    # Each comment above is a claim, so check it rather than trusting the printout.
    assert measured["area"][0] == 16.0, "a 4x4 square"
    assert measured["area"][1] == 96.0, "10x10 less a 2x2 hole — the hole IS subtracted"
    assert measured["length"][:2] == [0.0, 0.0], "a polygon has zero length, as in PostGIS"
    assert measured["length"][2] == 5.0, "the 3-4-5 chain"
    assert measured["perimeter"][1] == 48.0, "40 of shell plus 8 of hole — holes ARE included"
    assert measured["perimeter"][2] == 0.0 and measured["area"][3] == 0.0, "zero, never null"

    pairs = bt.from_pydict(
        {
            "a": ["POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))", "LINESTRING(0 0, 10 0)"],
            "b": ["POINT(6 2)", "LINESTRING(0 1, 10 1)"],
        }
    )
    print("--- the distance family ---")
    distances = pairs.select(
        nearest=bt.st_distance(col("a"), col("b")),
        furthest=bt.st_max_distance(col("a"), col("b")),
        # How far apart two shapes are at their worst-matching point: the standard
        # measure of "are these the same shape".
        hausdorff=bt.st_hausdorff_distance(col("a"), col("b")),
    ).to_pydict()
    print(distances)
    assert distances["nearest"] == [2.0, 1.0]
    # The three are genuinely different measures: for two parallel lines one unit apart,
    # nearest and Hausdorff agree at 1 while the furthest corner-to-corner span is ~10.
    assert distances["hausdorff"][1] == 1.0 and distances["furthest"][1] > 10.0
    for near, far in zip(distances["nearest"], distances["furthest"], strict=True):
        assert near <= far

    bearings = bt.from_pydict(
        {
            "from": ["POINT(0 0)"] * 4,
            "to": ["POINT(0 1)", "POINT(1 0)", "POINT(0 -1)", "POINT(-1 0)"],
        }
    )
    print("--- azimuth: radians clockwise from north ---")
    azimuths = bearings.select(rad=bt.st_azimuth(col("from"), col("to")).round(4)).to_pydict()
    print(azimuths)
    # North, east, south, west — clockwise from north, so east is pi/2 and not -pi/2. A
    # counter-clockwise or from-east convention would print equally plausible numbers.
    assert azimuths["rad"] == [0.0, 1.5708, 3.1416, 4.7124]


def geodesic_measurement_answers_in_metres() -> None:
    legs = bt.from_pydict(
        {
            "leg": ["SF to London", "London to Paris"],
            "a": ["POINT(-122.4194 37.7749)", "POINT(-0.1278 51.5074)"],
            "b": ["POINT(-0.1278 51.5074)", "POINT(2.3522 48.8566)"],
        }
    )
    print("--- planar degrees versus spherical and ellipsoidal metres ---")
    legs_out = legs.select(
        "leg",
        degrees=bt.st_distance(col("a"), col("b")).round(2),
        # Haversine: about 0.5% accurate, cheap, no failure mode.
        sphere_km=(bt.st_distance_sphere(col("a"), col("b")) / 1000).round(1),
        # Karney on WGS 84: nanometre-accurate, defined for antipodes, slower.
        spheroid_km=(bt.st_distance_spheroid(col("a"), col("b")) / 1000).round(1),
    ).to_pydict()
    print(legs_out)
    # SF to London is ~8,600 km and London to Paris ~344 km; the whole point of the section
    # is that the planar number is in degrees and means nothing as a distance.
    assert 8500 < legs_out["sphere_km"][0] < 8700
    assert 340 < legs_out["spheroid_km"][1] < 350
    # The sphere and the ellipsoid must agree to within the sphere's stated ~0.5%.
    for sphere, spheroid in zip(legs_out["sphere_km"], legs_out["spheroid_km"], strict=True):
        assert abs(sphere - spheroid) / spheroid < 0.005

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
    cells_out = cells.select(
        "where",
        square_degrees=bt.st_area(col("g")),
        square_km=(bt.st_area_spheroid(col("g")) / 1e6).round(0),
        perimeter_km=(bt.st_perimeter_spheroid(col("g")) / 1000).round(0),
    ).to_pydict()
    print(cells_out)
    # The section's entire claim: identical in degrees, and about half the ground area at
    # 60 north, because a degree of longitude shrinks with cos(latitude) and cos(60) = 0.5.
    assert cells_out["square_degrees"] == [1.0, 1.0]
    ratio = cells_out["square_km"][1] / cells_out["square_km"][0]
    assert 0.45 < ratio < 0.55, f"expected roughly cos(60)=0.5, measured {ratio:.3f}"

    routes = bt.from_pydict({"g": ["LINESTRING(0 0, 1 0, 1 1)"]})
    print("--- geodesic chain length ---")
    chain = routes.select(km=(bt.st_length_spheroid(col("g")) / 1000).round(1)).to_pydict()
    print(chain)
    # One degree of longitude along the equator (111.32 km on WGS 84) plus one of latitude
    # (110.57 km): the ellipsoid makes the two legs differ, which a sphere would not.
    assert chain["km"] == [221.9]


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
    zones = zoned.select("site", "zone", "epsg").to_pydict()
    print(zones)
    # Northern zones are 326xx and southern 327xx: Sydney is zone 56 *south*.
    assert zones["zone"] == [10, 56]
    assert zones["epsg"] == [32610, 32756]

    # `st_transform` labels the result with the target SRID, so a later transform knows
    # where it is starting from.
    projected = zoned.select(
        "site",
        mercator=bt.st_as_text(bt.st_transform(col("wgs84"), 4326, 3857)),
        equal_area_srid=bt.st_srid(bt.st_transform(col("wgs84"), 4326, 6933)),
    )
    projected_out = projected.to_pydict()
    print(projected_out)
    assert projected_out["equal_area_srid"] == [6933, 6933], "the target SRID is carried"
    # Web Mercator x is R * lon in radians: -122.4194 degrees is about -13.63 million m.
    sf_x = float(projected_out["mercator"][0].removeprefix("POINT(").split()[0])
    assert abs(sf_x + 13_627_665.27) < 0.01

    # In a projected system the planar functions answer in metres, so a buffer, an area
    # and a distance are all directly meaningful.
    metres = bt.from_pydict({"g": ["POINT(551131 4180000)"]})
    print("--- planar functions in a metre-based CRS ---")
    in_metres = metres.select(
        around_100m=bt.st_area(bt.st_buffer(col("g"), 100.0, 32)).round(0),
        box=bt.st_as_text(bt.st_expand(col("g"), 100.0, 100.0)),
    ).to_pydict()
    print(in_metres)
    # A 100 m buffer is a 128-sided polygon, just under pi * 100^2 = 31,416 square metres.
    assert in_metres["around_100m"] == [31403.0]
    assert in_metres["box"] == [
        "POLYGON((551031 4179900, 551231 4179900, 551231 4180100, 551031 4180100, 551031 4179900))"
    ]


def reshape_before_shuffling() -> None:
    """Vertex count drives every cost; simplification is the biggest lever on it."""
    detailed = bt.from_pydict({"g": ["LINESTRING(0 0, 1 0.001, 2 0, 3 0.002, 4 0, 5 0.001, 6 0)"]})
    simple = bt.st_simplify(col("g"), 0.01)
    print("--- simplify, and measure what it cost you ---")
    simplified = detailed.select(
        before=bt.st_num_points(col("g")),
        after=bt.st_num_points(simple),
        error=bt.st_hausdorff_distance(col("g"), simple).round(4),
    ).to_pydict()
    print(simplified)
    # Every wiggle is under the 0.01 tolerance, so only the endpoints survive, and the
    # price is the largest wiggle, 0.002.
    assert simplified == {"before": [7], "after": [2], "error": [0.002]}

    messy = bt.from_pydict({"g": ["LINESTRING(0 0, 0 0, 1.234 5.678, 1.234 5.678, 3 3)"]})
    print("--- snapping and thinning make a column compress and join ---")
    snapped = bt.st_snap_to_grid(col("g"), 0.5)
    tidied = messy.select(
        snapped=bt.st_as_text(snapped),
        thinned=bt.st_as_text(bt.st_remove_repeated_points(snapped, 0.0)),
    ).to_pydict()
    print(tidied)
    # Snapping makes the near-duplicates exact duplicates; thinning then removes them.
    assert tidied["snapped"] == ["LINESTRING(0 0, 0 0, 1 5.5, 1 5.5, 3 3)"]
    assert tidied["thinned"] == ["LINESTRING(0 0, 1 5.5, 3 3)"]

    long_leg = bt.from_pydict({"g": ["LINESTRING(0 0, 10 0)"]})
    print("--- densify before reprojecting a long segment ---")
    densified = long_leg.select(
        before=bt.st_num_points(col("g")),
        after=bt.st_num_points(bt.st_segmentize(col("g"), 2.0)),
    ).to_pydict()
    print(densified)
    # Ten units at no more than two per segment is five segments, six positions.
    assert densified == {"before": [2], "after": [6]}


def normalize_and_transform_shapes() -> None:
    """Affine transforms cannot invalidate a geometry; normalization fixes conventions."""
    one = bt.from_pydict({"g": ["POLYGON((0 0, 0 4, 4 4, 4 0, 0 0))"]})
    print("--- affine transforms preserve structure ---")
    affine = one.select(
        moved=bt.st_as_text(bt.st_translate(col("g"), 10.0, 10.0)),
        scaled_area=bt.st_area(bt.st_scale(col("g"), 2.0, 3.0)),
        turned_area=bt.st_area(bt.st_rotate(col("g"), 3.141592653589793)).round(9),
        turned_valid=bt.st_is_valid(bt.st_rotate(col("g"), 3.141592653589793)),
        # Every one of the above is a special case of this.
        general=bt.st_as_text(bt.st_affine(col("g"), 1.0, 0.0, 0.0, 1.0, 5.0, 6.0)),
    ).to_pydict()
    print(affine)
    assert affine["moved"] == ["POLYGON((10 10, 10 14, 14 14, 14 10, 10 10))"]
    assert affine["scaled_area"] == [96.0], "16 scaled by 2 x 3"
    assert affine["turned_area"] == [16.0] and affine["turned_valid"] == [True]
    assert affine["general"] == ["POLYGON((5 6, 5 10, 9 10, 9 6, 5 6))"]

    print("--- winding, dimension and direction ---")
    conventions = one.select(
        ccw_area=bt.st_area(bt.st_force_polygon_ccw(col("g"))),
        cw_area=bt.st_area(bt.st_force_polygon_cw(col("g"))),
        reversed_ring=bt.st_as_text(bt.st_reverse(col("g"))),
        flat=bt.st_as_text(bt.st_force_2d(bt.st_force_3d(col("g"), 7.0))),
        raised=bt.st_has_z(bt.st_force_3d(col("g"), 7.0)),
        # Force3D fills a missing z and never overwrites a measured one.
        kept_z=bt.st_z(bt.st_force_3d("POINT Z (1 2 3)", 7.0)),
    ).to_pydict()
    print(conventions)
    # Winding does not change area, only which way round the ring runs.
    assert conventions["ccw_area"] == conventions["cw_area"] == [16.0]
    assert conventions["reversed_ring"] == ["POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))"]
    assert conventions["flat"] == ["POLYGON((0 0, 0 4, 4 4, 4 0, 0 0))"]
    assert conventions["raised"] == [True]
    assert conventions["kept_z"] == [3.0]

    swapped = bt.from_pydict({"g": ["POINT(37.7749 -122.4194)"]})
    print("--- the fix for a lat/lon column loaded as lon/lat ---")
    flipped = swapped.select(fixed=bt.st_as_text(bt.st_flip_coordinates(col("g")))).to_pydict()
    print(flipped)
    assert flipped["fixed"] == ["POINT(-122.4194 37.7749)"]

    print("--- derived shapes: hull, centroid, and a point that is really on the shape ---")
    crescent = bt.from_pydict(
        {"g": ["POLYGON((0 0, 10 0, 10 2, 2 2, 2 8, 10 8, 10 10, 0 10, 0 0))"]}
    )
    derived = crescent.select(
        hull_area=bt.st_area(bt.st_convex_hull(col("g"))),
        own_area=bt.st_area(col("g")),
        centroid_inside=bt.st_contains(col("g"), bt.st_centroid(col("g"))),
        surface_inside=bt.st_contains(col("g"), bt.st_point_on_surface(col("g"))),
        envelope=bt.st_as_text(bt.st_envelope(col("g"))),
    ).to_pydict()
    print(derived)
    # The centroid of a C-shape falls in its mouth; the point on surface never does.
    assert derived["hull_area"] == [100.0] and derived["own_area"] == [52.0]
    assert derived["centroid_inside"] == [False]
    assert derived["surface_inside"] == [True]
    assert derived["envelope"] == ["POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))"]

    print("--- collect concatenates without computing an overlay ---")
    two = bt.from_pydict({"a": ["POINT(0 0)"], "b": ["POINT(4 4)"]})
    collected = two.select(
        joined=bt.st_as_text(bt.st_collect(col("a"), col("b"))),
        span=bt.st_as_text(bt.st_envelope(bt.st_collect(col("a"), col("b")))),
    ).to_pydict()
    print(collected)
    assert collected["joined"] == ["MULTIPOINT((0 0), (4 4))"]
    assert collected["span"] == ["POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))"]

    print("--- a buffer is a union of its parts, and a negative one erodes ---")
    shapes = bt.from_pydict(
        {
            "g": [
                "MULTIPOINT((0 0), (10 0))",
                "MULTIPOINT((0 0), (1 0))",
                "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))",
            ]
        }
    )
    grown = shapes.select(
        parts=bt.st_num_geometries(bt.st_buffer(col("g"), 1.0, 8)),
        area=bt.st_area(bt.st_buffer(col("g"), 1.0, 8)).round(3),
        eroded=bt.st_area(bt.st_buffer(col("g"), -1.0, 8)),
    ).to_pydict()
    print(grown)
    # Two points 10 apart stay two discs (2 x 3.121); two 1 apart merge into one shape
    # smaller than two discs; the 4 x 4 square eroded by 1 is the 2 x 2 square.
    assert grown["parts"] == [2, 1, 1]
    assert grown["area"][0] == 6.243
    assert 3.121 < grown["area"][1] < 6.243
    assert grown["eroded"] == [0.0, 0.0, 4.0]


def positions_along_a_route() -> None:
    """Linear referencing: the vocabulary route and network data is described in."""
    road = bt.from_pydict({"g": ["LINESTRING(0 0, 10 0, 10 10)"], "fix": ["POINT(4 7)"]})
    print("--- interpolate and locate are exact inverses ---")
    linear = road.select(
        halfway=bt.st_as_text(bt.st_line_interpolate_point(col("g"), 0.5)),
        where=bt.st_line_locate_point(col("g"), col("fix")).round(4),
        stretch=bt.st_as_text(bt.st_line_substring(col("g"), 0.25, 0.75)),
    ).to_pydict()
    print(linear)
    # The road is 20 units long, so the midpoint is exactly the corner at (10 0).
    assert linear["halfway"] == ["POINT(10 0)"]
    assert linear["stretch"] == ["LINESTRING(5 0, 10 0, 10 5)"]
    # "Exact inverses" is a claim, so round-trip it: interpolating at the fraction `locate`
    # returned must land back on the point `locate` was given.
    back = road.select(
        at=bt.st_as_text(bt.st_line_interpolate_point(col("g"), linear["where"][0]))
    ).to_pydict()["at"]
    assert back == ["POINT(10 7)"], f"locate/interpolate are not inverses: {back}"

    print("--- snapping a fix to the road, and drawing the gap ---")
    snapping = road.select(
        snapped=bt.st_as_text(bt.st_closest_point(col("g"), col("fix"))),
        gap=bt.st_as_text(bt.st_shortest_line(col("g"), col("fix"))),
    ).to_pydict()
    print(snapping)
    # The fix at (4 7) snaps sideways onto the vertical leg, and the gap runs from the
    # snapped point back to the fix — so the two answers must agree on where the road is.
    assert snapping["snapped"] == ["POINT(10 7)"]
    assert snapping["gap"] == ["LINESTRING(10 7, 4 7)"]

    origin = bt.from_pydict({"g": ["POINT(0 0)"]})
    print("--- travel a geodesic distance along a bearing ---")
    travelled = origin.select(
        north_111km=bt.st_as_text(bt.st_project(col("g"), 111195.0, 0.0)),
        east_111km=bt.st_as_text(bt.st_project(col("g"), 111195.0, 90.0)),
    ).to_pydict()
    print(travelled)
    # 111,195 m is one degree of great circle, so from the origin each bearing moves almost
    # exactly one degree along its own axis and essentially none along the other. A bearing
    # measured from the wrong reference would swap these two.
    north = travelled["north_111km"][0].removeprefix("POINT(").removesuffix(")").split()
    east = travelled["east_111km"][0].removeprefix("POINT(").removesuffix(")").split()
    assert abs(float(north[0])) < 1e-6 and abs(float(north[1]) - 1.0) < 1e-5
    assert abs(float(east[0]) - 1.0) < 1e-5 and abs(float(east[1])) < 1e-6


def main() -> None:
    planar_measurement_answers_in_coordinate_units()
    geodesic_measurement_answers_in_metres()
    project_once_then_measure_in_metres()
    reshape_before_shuffling()
    normalize_and_transform_shapes()
    positions_along_a_route()


if __name__ == "__main__":
    main()
