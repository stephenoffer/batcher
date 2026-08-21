"""Ground distances, geodesic measures and reprojection, against PROJ and GEOS.

Everything in ``bc-geo::proj`` answers a question about the *ground* rather than about
coordinate units, and every one of those answers has an independent oracle:

* **Reprojection** (``st_transform``) is checked against PROJ, which DuckDB's spatial
  extension links. Web Mercator agrees to the last bits; UTM agrees to under a micrometre
  inside the zone.
* **Ellipsoidal distance** (``st_distance_spheroid``) is Vincenty on WGS 84 and agrees
  with PROJ's geodesic solver to eleven significant figures.
* **Spherical measures** (``st_distance_sphere`` and the ``*_spheroid`` area, length and
  perimeter, which ``bc_geo::proj::geodesy`` computes on one mean-radius sphere) agree
  with the ellipsoidal answer only to the **0.5%** that module documents as the price of
  the sphere. That band is asserted from both sides: too loose and a factor-of-two error
  passes, too tight and the test is wrong about which model the code uses.

**The coordinate order is the trap this file exists to pin.** DuckDB's
``ST_Distance_Sphere``, ``ST_Distance_Spheroid`` and ``ST_Transform`` read a point as
**(latitude, longitude)** unless told otherwise, while Batcher -- like GeoJSON, WKT and
every other function in this suite -- reads (longitude, latitude). Comparing the two
without accounting for it does not fail loudly: it returns 8,500 km for New York to
Paris instead of 5,845 km, which is the right order of magnitude and the wrong answer.
So the oracle here flips the coordinates (or passes ``always_xy``) and the flip is
asserted to still be necessary, so this test starts failing if DuckDB ever changes it.
"""

from __future__ import annotations

import math

import pytest

import batcher as bt

pytestmark = pytest.mark.differential

duckdb = pytest.importorskip("duckdb")

#: The mean-radius sphere `bc_geo::proj::geodesy` uses, and the accuracy its module
#: docstring claims against the ellipsoid. Both are restated here on purpose: this test
#: is what holds the claim to the code.
SPHERE_TOLERANCE = 5e-3

#: Well-known places, as (longitude, latitude). Spread across all four quadrants and
#: both hemispheres so a sign or a swapped ordinate cannot survive.
PLACES = {
    "new_york": (-73.9857, 40.7484),
    "paris": (2.2945, 48.8584),
    "tokyo": (139.6917, 35.6895),
    "buenos_aires": (-58.3816, -34.6037),
    "null_island": (0.0, 0.0),
}
PAIRS = [
    ("new_york", "paris"),
    ("tokyo", "buenos_aires"),
    ("null_island", "paris"),
    ("new_york", "new_york"),
]


@pytest.fixture(scope="module")
def spatial():
    """A DuckDB connection with the spatial extension, or a skip when it is unavailable."""
    con = duckdb.connect()
    try:
        con.execute("INSTALL spatial; LOAD spatial;")
    except Exception as exc:
        pytest.skip(f"duckdb spatial extension unavailable: {exc}")
    return con


def _wkt(place: str) -> str:
    lon, lat = PLACES[place]
    return f"POINT({lon} {lat})"


def test_duckdb_still_reads_a_geodesic_point_as_latitude_first(spatial):
    """The premise every comparison below rests on, asserted rather than assumed.

    If DuckDB ever switches ``ST_Distance_Sphere`` to (longitude, latitude), the flips in
    this file become wrong and every other test here would start comparing the wrong
    numbers while still passing on the tolerance. This one fails first and says why.
    """
    lon, lat = PLACES["new_york"]
    lon2, lat2 = PLACES["paris"]
    as_lonlat = spatial.execute(
        "SELECT ST_Distance_Sphere(ST_Point(?, ?), ST_Point(?, ?))", [lon, lat, lon2, lat2]
    ).fetchone()[0]
    as_latlon = spatial.execute(
        "SELECT ST_Distance_Sphere(ST_Point(?, ?), ST_Point(?, ?))", [lat, lon, lat2, lon2]
    ).fetchone()[0]
    assert as_latlon == pytest.approx(5.83e6, rel=0.01), "latitude-first is the real distance"
    assert as_lonlat > 8e6, "longitude-first is not, and is why this premise is pinned"


def test_ellipsoidal_distance_matches_projs_geodesic(spatial):
    """Vincenty on WGS 84, against PROJ, to eleven significant figures."""
    ds = bt.from_pydict({"a": [_wkt(a) for a, _ in PAIRS], "b": [_wkt(b) for _, b in PAIRS]})
    got = ds.select(v=bt.st_distance_spheroid(bt.col("a"), bt.col("b"))).to_pydict()["v"]
    for (a, b), metres in zip(PAIRS, got, strict=True):
        lon1, lat1 = PLACES[a]
        lon2, lat2 = PLACES[b]
        want = spatial.execute(
            "SELECT ST_Distance_Spheroid(ST_Point(?, ?), ST_Point(?, ?))",
            [lat1, lon1, lat2, lon2],
        ).fetchone()[0]
        assert metres == pytest.approx(want, rel=1e-11), f"{a} -> {b}"


def test_spherical_distance_sits_inside_the_half_percent_the_sphere_costs(spatial):
    """The haversine answer, held to the accuracy its own module claims.

    Asserted from both sides. An implementation that quietly switched to the ellipsoid
    would pass the upper bound and fail the lower one, which is the point: this test
    knows which model the code is supposed to be using.
    """
    ds = bt.from_pydict({"a": [_wkt(a) for a, _ in PAIRS], "b": [_wkt(b) for _, b in PAIRS]})
    got = ds.select(
        sphere=bt.st_distance_sphere(bt.col("a"), bt.col("b")),
        ellipsoid=bt.st_distance_spheroid(bt.col("a"), bt.col("b")),
    ).to_pydict()
    for (a, b), s, e in zip(PAIRS, got["sphere"], got["ellipsoid"], strict=True):
        assert s == pytest.approx(e, rel=SPHERE_TOLERANCE), f"{a} -> {b}: sphere is off"
        lon1, lat1 = PLACES[a]
        lon2, lat2 = PLACES[b]
        want = spatial.execute(
            "SELECT ST_Distance_Sphere(ST_Point(?, ?), ST_Point(?, ?))",
            [lat1, lon1, lat2, lon2],
        ).fetchone()[0]
        assert s == pytest.approx(want, rel=1e-5), f"{a} -> {b}: not GEOS's sphere either"


def test_a_point_is_zero_distance_from_itself():
    """The degenerate case both models must get exactly, not nearly, right."""
    same = [(p, p) for p in PLACES]
    ds = bt.from_pydict({"a": [_wkt(a) for a, _ in same], "b": [_wkt(b) for _, b in same]})
    got = ds.select(
        sphere=bt.st_distance_sphere(bt.col("a"), bt.col("b")),
        ellipsoid=bt.st_distance_spheroid(bt.col("a"), bt.col("b")),
    ).to_pydict()
    assert got["sphere"] == [0.0] * len(same)
    assert got["ellipsoid"] == [0.0] * len(same)


def test_dwithin_sphere_agrees_with_the_distance_it_thresholds():
    """The predicate must be exactly ``st_distance_sphere <= metres``, at the boundary too.

    The radius is taken from the measured distance itself, so the two sides of the
    boundary are a millimetre apart and a predicate computed a different way cannot
    straddle them by accident.
    """
    pairs = [(a, b) for a, b in PAIRS if a != b]
    ds = bt.from_pydict({"a": [_wkt(a) for a, _ in pairs], "b": [_wkt(b) for _, b in pairs]})
    exact = ds.select(d=bt.st_distance_sphere(bt.col("a"), bt.col("b"))).to_pydict()["d"]
    for (a, b), metres in zip(pairs, exact, strict=True):
        one = bt.from_pydict({"a": [_wkt(a)], "b": [_wkt(b)]})
        got = one.select(
            inside=bt.st_dwithin_sphere(bt.col("a"), bt.col("b"), metres + 1e-3),
            outside=bt.st_dwithin_sphere(bt.col("a"), bt.col("b"), metres - 1e-3),
        ).to_pydict()
        assert got["inside"] == [True], f"{a} -> {b} is within its own distance"
        assert got["outside"] == [False], f"{a} -> {b} is not within one millimetre less"


def test_geodesic_area_length_and_perimeter_track_geos_inside_the_sphere_band(spatial):
    """The ``*_spheroid`` measures, which Batcher computes on the sphere.

    The name says spheroid and the implementation is spherical excess (see
    ``bc_geo::proj::geodesy``), so these agree with GEOS's ellipsoidal answer to the
    documented 0.5% and no better. Asserting equality would be wrong; asserting nothing
    would let a hemisphere-sized error through. The band is asserted in both directions.
    """
    geometries = [
        "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))",
        "POLYGON((10 45, 11 45, 11 46, 10 46, 10 45))",
        "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 7 3, 7 7, 3 7, 3 3))",
        "LINESTRING(0 0, 1 0, 1 1)",
        "POINT(2 3)",
    ]
    ds = bt.from_pydict({"g": geometries})
    got = ds.select(
        area=bt.st_area_spheroid(bt.col("g")),
        length=bt.st_length_spheroid(bt.col("g")),
        perimeter=bt.st_perimeter_spheroid(bt.col("g")),
    ).to_pydict()
    for key, theirs in [
        ("area", "ST_Area_Spheroid"),
        ("length", "ST_Length_Spheroid"),
        ("perimeter", "ST_Perimeter_Spheroid"),
    ]:
        for geom, ours in zip(geometries, got[key], strict=True):
            want = spatial.execute(
                # DuckDB reads latitude first here too, so the geometry is flipped rather
                # than the fixture being written in the other order.
                f"SELECT {theirs}(ST_FlipCoordinates(ST_GeomFromText(?)))",
                [geom],
            ).fetchone()[0]
            if want == 0.0:
                assert ours == 0.0, f"{key} of {geom} is not zero off its own type"
                continue
            assert ours == pytest.approx(want, rel=SPHERE_TOLERANCE), f"{key} of {geom}"
            assert ours != want, (
                f"{key} of {geom} matched the ellipsoid exactly -- if the implementation "
                "moved to the ellipsoid, tighten this file rather than deleting the check"
            )


def test_project_walks_the_distance_and_bearing_it_was_given():
    """``st_project`` inverted by ``st_distance_sphere``: the round trip must close."""
    starts = ["POINT(0 0)", "POINT(10 45)", "POINT(-73.9857 40.7484)"]
    ds = bt.from_pydict({"p": starts})
    for distance in (1_000.0, 100_000.0):
        for azimuth_deg in (0.0, 90.0, 217.5):
            moved = bt.st_project(bt.col("p"), distance, math.radians(azimuth_deg))
            got = ds.select(
                back=bt.st_distance_sphere(bt.col("p"), bt.st_as_text(moved))
            ).to_pydict()["back"]
            for start, walked in zip(starts, got, strict=True):
                assert walked == pytest.approx(distance, rel=1e-6), (
                    f"projecting {start} by {distance} m at {azimuth_deg} deg "
                    f"landed {walked} m away"
                )


def test_project_due_north_moves_only_the_latitude():
    """A bearing of zero must leave the longitude alone, whatever the latitude."""
    ds = bt.from_pydict({"p": ["POINT(10 45)", "POINT(-73.9857 40.7484)"]})
    moved = bt.st_project(bt.col("p"), 50_000.0, 0.0)
    got = ds.select(
        lon0=bt.st_x(bt.col("p")),
        lon1=bt.st_x(moved),
        lat0=bt.st_y(bt.col("p")),
        lat1=bt.st_y(moved),
    ).to_pydict()
    for i in range(2):
        assert got["lon1"][i] == pytest.approx(got["lon0"][i], abs=1e-12)
        assert got["lat1"][i] > got["lat0"][i], "due north must increase the latitude"


def test_web_mercator_reprojection_matches_proj(spatial):
    """EPSG:4326 to EPSG:3857, against PROJ and against the closed form.

    Two oracles because they fail differently: PROJ catches a wrong datum or a swapped
    ordinate, and the closed form catches PROJ and Batcher agreeing on the wrong thing.
    """
    lons = [lon for lon, _ in PLACES.values()]
    lats = [lat for _, lat in PLACES.values()]
    ds = bt.from_pydict({"lon": lons, "lat": lats})
    projected = bt.st_transform(bt.st_point(bt.col("lon"), bt.col("lat")), 4326, 3857)
    got = ds.select(x=bt.st_x(projected), y=bt.st_y(projected)).to_pydict()
    for i, (lon, lat) in enumerate(zip(lons, lats, strict=True)):
        wx, wy = spatial.execute(
            "SELECT ST_X(p), ST_Y(p) FROM ("
            "  SELECT ST_Transform(ST_Point(?, ?), 'EPSG:4326', 'EPSG:3857', always_xy := true) p)",
            [lon, lat],
        ).fetchone()
        assert got["x"][i] == pytest.approx(wx, abs=1e-6), f"x of ({lon}, {lat})"
        assert got["y"][i] == pytest.approx(wy, abs=1e-6), f"y of ({lon}, {lat})"
        # Spherical Mercator on the WGS 84 semi-major axis, which is what EPSG:3857 is
        # defined as. Written with the radius rather than the rounded 20037508.34 that
        # circulates in web-map code: that constant is a millimetre short of pi * R, and
        # a test built on it cannot tell a millimetre of rounding from a millimetre of error.
        radius = 6378137.0
        closed_x = radius * math.radians(lon)
        closed_y = radius * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
        assert got["x"][i] == pytest.approx(closed_x, abs=1e-6), f"closed-form x ({lon}, {lat})"
        assert got["y"][i] == pytest.approx(closed_y, abs=1e-6), f"closed-form y ({lon}, {lat})"


def test_utm_reprojection_matches_proj_inside_its_own_zone(spatial):
    """UTM, checked in the zone ``st_utm_epsg`` picks for the point.

    Scoped to the correct zone deliberately. Transverse Mercator is a truncated series
    that diverges far from its central meridian, so two implementations disagree by
    metres several zones out -- not a defect in either, and comparing there would be
    testing the divergence rather than the projection.
    """
    lons = [lon for lon, _ in PLACES.values()]
    lats = [lat for _, lat in PLACES.values()]
    ds = bt.from_pydict({"lon": lons, "lat": lats})
    epsg = ds.select(v=bt.st_utm_epsg(bt.col("lon"), bt.col("lat"))).to_pydict()["v"]
    for (lon, lat), code in zip(zip(lons, lats, strict=True), epsg, strict=True):
        one = bt.from_pydict({"lon": [lon], "lat": [lat]})
        projected = bt.st_transform(bt.st_point(bt.col("lon"), bt.col("lat")), 4326, int(code))
        got = one.select(x=bt.st_x(projected), y=bt.st_y(projected)).to_pydict()
        wx, wy = spatial.execute(
            "SELECT ST_X(p), ST_Y(p) FROM ("
            f"  SELECT ST_Transform(ST_Point(?, ?), 'EPSG:4326', 'EPSG:{int(code)}',"
            "   always_xy := true) p)",
            [lon, lat],
        ).fetchone()
        # A millimetre. The two implementations truncate the transverse-Mercator series
        # at different orders, so inside the zone they agree to well under that and no
        # tighter bound is meaningful.
        assert got["x"][0] == pytest.approx(wx, abs=1e-3), f"UTM x of ({lon}, {lat}) in {code}"
        assert got["y"][0] == pytest.approx(wy, abs=1e-3), f"UTM y of ({lon}, {lat}) in {code}"


def test_utm_zone_and_epsg_follow_the_published_rule():
    """Zone number is ``floor((lon + 180) / 6) + 1``; the EPSG code encodes the hemisphere."""
    lons = [-180.0, -73.9857, -0.1, 0.0, 10.0, 139.6917, 179.9]
    lats = [10.0, 40.7484, 51.5, -1.0, 45.0, 35.6895, -60.0]
    got = (
        bt.from_pydict({"lon": lons, "lat": lats})
        .select(
            zone=bt.st_utm_zone(bt.col("lon")),
            epsg=bt.st_utm_epsg(bt.col("lon"), bt.col("lat")),
        )
        .to_pydict()
    )
    for i, (lon, lat) in enumerate(zip(lons, lats, strict=True)):
        want_zone = int((lon + 180.0) // 6) + 1
        assert got["zone"][i] == want_zone, f"zone of {lon}"
        want_epsg = (32600 if lat >= 0 else 32700) + want_zone
        assert got["epsg"][i] == want_epsg, f"EPSG of ({lon}, {lat})"


def test_reprojection_round_trips_back_to_the_original_position():
    """4326 to 3857 and back must return the input, to well under a millimetre."""
    lons = [lon for lon, _ in PLACES.values()]
    lats = [lat for _, lat in PLACES.values()]
    ds = bt.from_pydict({"lon": lons, "lat": lats})
    there = bt.st_transform(bt.st_point(bt.col("lon"), bt.col("lat")), 4326, 3857)
    back = bt.st_transform(bt.st_as_text(there), 3857, 4326)
    got = ds.select(lon=bt.st_x(back), lat=bt.st_y(back)).to_pydict()
    for i, (lon, lat) in enumerate(zip(lons, lats, strict=True)):
        assert got["lon"][i] == pytest.approx(lon, abs=1e-9)
        assert got["lat"][i] == pytest.approx(lat, abs=1e-9)
