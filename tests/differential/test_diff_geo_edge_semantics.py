"""Edge semantics of the geospatial surface, against DuckDB spatial (GEOS + GeographicLib).

Every case here was wrong before the change that added this file, and each test says
what it used to return. They fall into five groups:

* **Row-local failure.** One NaN or off-globe coordinate used to raise for the whole
  column in every grid and UTM function; now that row alone is null. DuckDB raises on
  some of these rows and PostGIS on most. Nulling is this engine's rule for every
  row-local geometry failure (a malformed WKB already nulled), so the valid rows are
  compared with DuckDB and the bad ones are asserted null. A *parameter* out of range
  (a precision of -1) still raises, and the message names the value passed.
* **The ellipsoid.** The ``*_spheroid`` functions now use Karney's algorithm on WGS 84,
  the same one DuckDB links, so they are held to 1e-9 relative rather than to the 0.5%
  band a sphere needed. This includes the antimeridian box that used to be 17x too big
  and the antipodal pair that used to be null.
* **Buffers** are now unions of their parts, compared by area (to the difference between
  GEOS's arcs and a regular polygon's chords) and by part count.
* **Validity, simplicity and boundaries** on a battery of degenerate shapes.
* **Z** in centroids, envelopes and ``st_force_3d``.
"""

from __future__ import annotations

import math
import random

import pytest

import batcher as bt
from batcher._internal.errors import BatcherError, PlanError

pytestmark = pytest.mark.differential

duckdb = pytest.importorskip("duckdb")


@pytest.fixture(scope="module")
def spatial():
    """A DuckDB connection with the spatial extension, or a skip when it is unavailable."""
    con = duckdb.connect()
    try:
        con.execute("INSTALL spatial; LOAD spatial;")
    except Exception as exc:
        pytest.skip(f"duckdb spatial extension unavailable: {exc}")
    return con


def _one(spatial, sql: str, *binds: object) -> object:
    return spatial.execute(sql, list(binds)).fetchone()[0]


def _column(expr, **columns) -> list:
    return bt.from_pydict(columns).select(v=expr).to_pydict()["v"]


# --- row-local failure ---------------------------------------------------------------

LONS = [13.4, float("nan"), 200.0, None, -122.4194, 180.0, -180.0, 0.0]
LATS = [52.5, 0.0, 0.0, 1.0, 37.7749, 0.0, -90.0, 95.0]
#: Rows 1 (NaN), 2 (lon 200), 3 (null) and 7 (lat 95) have no cell.
BAD_ROWS = {1, 2, 3, 7}


@pytest.mark.parametrize(
    "build",
    [
        lambda lo, la: bt.geohash_encode(lo, la, 9),
        lambda lo, la: bt.st_geohash(bt.st_point(lo, la), 9),
        lambda lo, la: bt.st_quadkey(lo, la, 10),
        lambda lo, la: bt.st_tile_x(lo, la, 10),
        lambda lo, la: bt.st_tile_y(lo, la, 10),
        lambda lo, la: bt.st_s2_cell(lo, la, 10),
        lambda lo, la: bt.st_utm_epsg(lo, la),
    ],
    ids=[
        "geohash_encode",
        "st_geohash",
        "st_quadkey",
        "st_tile_x",
        "st_tile_y",
        "st_s2_cell",
        "st_utm_epsg",
    ],
)
def test_one_off_globe_position_nulls_only_its_own_row(build):
    """Used to raise ``ExecutionError`` for the whole column on the NaN or lon=200 row."""
    got = _column(build(bt.col("lon"), bt.col("lat")), lon=LONS, lat=LATS)
    for i, v in enumerate(got):
        if i in BAD_ROWS:
            assert v is None, f"row {i} ({LONS[i]}, {LATS[i]}) should be null, got {v!r}"
        else:
            assert v is not None, f"row {i} ({LONS[i]}, {LATS[i]}) lost its cell"


def test_utm_zone_nulls_only_the_bad_longitude():
    got = _column(bt.st_utm_zone(bt.col("lon")), lon=[13.4, float("nan"), 200.0, -180.0])
    assert got == [33, None, None, 1]


def test_valid_quadkeys_still_match_duckdb(spatial):
    """The positive control: nulling bad rows did not change a good one.

    Interior positions only. On the map's edges the two engines follow different, both
    defensible conventions that predate this file: DuckDB wraps longitude 180 onto the
    west edge (tile 0) and puts latitude -90 in the top row, where Batcher keeps 180 in
    the east column and -90 in the bottom row.
    """
    got = _column(bt.st_quadkey(bt.col("lon"), bt.col("lat"), 10), lon=LONS, lat=LATS)
    for i, (lon, lat) in enumerate(zip(LONS, LATS, strict=True)):
        if i in BAD_ROWS or abs(lon) == 180.0 or abs(lat) == 90.0:
            continue
        assert got[i] == _one(spatial, "SELECT ST_QuadKey(?, ?, 10)", lon, lat), (lon, lat)


def test_interpolating_along_an_empty_line_nulls_the_row():
    """Used to raise ``ExecutionError`` ("needs a line with at least two positions")."""
    got = _column(
        bt.st_as_text(bt.st_line_interpolate_point(bt.col("g"), 0.5)),
        g=["LINESTRING EMPTY", "LINESTRING(0 0, 10 0)"],
    )
    assert got == [None, "POINT(5 0)"]


def test_voxel_index_nulls_a_nan_point_and_a_zero_size():
    """Used to raise ``Can't cast NaN to Int64`` for the whole column."""
    ds = bt.from_pydict(
        {
            "x": [0.25, float("nan"), float("inf"), 1.5],
            "y": [0.0, 0.0, 0.0, -0.5],
            "z": [0.0, 0.0, 0.0, 0.0],
            "s": [1.0, 1.0, 1.0, 0.0],
        }
    )
    fixed = ds.select(**bt.voxel_index(("x", "y", "z"), 1.0)).to_pydict()
    assert fixed["ix"] == [0, None, None, 1]
    assert fixed["iy"] == [0, 0, 0, -1]
    per_row = ds.select(**bt.voxel_index(("x", "y", "z"), "s")).to_pydict()
    assert per_row["ix"] == [0, None, None, None], "a zero size has no cell"


@pytest.mark.parametrize("size", [0.0, -1.0, float("nan")])
def test_voxel_index_refuses_a_constant_size_that_is_not_positive(size):
    with pytest.raises(PlanError, match="voxel_index size"):
        bt.voxel_index(("x", "y", "z"), size)


@pytest.mark.parametrize(
    ("build", "bad"),
    [
        (lambda p: bt.geohash_encode(bt.col("lon"), bt.col("lat"), p), -1),
        (lambda p: bt.geohash_encode(bt.col("lon"), bt.col("lat"), p), 0),
        (lambda p: bt.st_geohash(bt.col("g"), p), 13),
        (lambda p: bt.st_quadkey(bt.col("lon"), bt.col("lat"), p), -1),
        (lambda p: bt.st_tile_x(bt.col("lon"), bt.col("lat"), p), 31),
        (lambda p: bt.st_s2_cell(bt.col("lon"), bt.col("lat"), p), -1),
        (lambda p: bt.st_s2_cell_parent(bt.col("c"), p), -1),
    ],
)
def test_a_constant_parameter_out_of_range_is_a_plan_error_naming_it(build, bad):
    """``st_quadkey(..., -1)`` used to return ``''`` and ``st_s2_cell(..., -1)`` level 0;
    ``geohash_encode(..., -1)`` said "got 0"."""
    with pytest.raises(PlanError, match=f"got {bad}"):
        build(bad)


def test_a_column_valued_parameter_out_of_range_fails_naming_the_value():
    ds = bt.from_pydict({"lon": [1.0], "lat": [2.0], "z": [-1]})
    with pytest.raises(BatcherError, match="got -1"):
        ds.select(v=bt.st_quadkey(bt.col("lon"), bt.col("lat"), bt.col("z"))).to_pydict()
    ok = ds.select(v=bt.st_quadkey(bt.col("lon"), bt.col("lat"), bt.col("z") + 4)).to_pydict()
    assert len(ok["v"][0]) == 3


# --- the ellipsoid -------------------------------------------------------------------


def _random_lonlat(rng: random.Random) -> tuple[float, float]:
    return rng.uniform(-180, 180), math.degrees(math.asin(rng.uniform(-1, 1)))


def _pairs() -> list[tuple[float, float, float, float]]:
    rng = random.Random(20260922)
    out = []
    for _ in range(40):
        (a, b), (c, d) = _random_lonlat(rng), _random_lonlat(rng)
        out.append((a, b, c, d))
    # Antipodal and near-antipodal, where Vincenty returned nothing.
    for lon, lat in [(0.0, 0.0), (10.0, 20.0), (-73.9, 40.7), (179.9, -0.1)]:
        anti = lon - 180.0 if lon > 0 else lon + 180.0
        near = anti - 0.3 if anti > 0 else anti + 0.3
        out.append((lon, lat, anti, -lat))
        out.append((lon, lat, near, -lat + 0.1))
    out += [(0.0, 90.0, 0.0, -90.0), (179.5, 0.0, -179.5, 0.0), (1.0, 2.0, 1.0, 2.0)]
    return out


def test_distance_spheroid_matches_geographiclib_everywhere(spatial):
    """Karney on WGS 84 against DuckDB's GeographicLib, 1e-9 relative, antipodes included.

    Before: Vincenty, null for the antipodal rows of this set.
    """
    pairs = _pairs()
    got = _column(
        bt.st_distance_spheroid(
            bt.st_point(bt.col("a"), bt.col("b")), bt.st_point(bt.col("c"), bt.col("d"))
        ),
        a=[p[0] for p in pairs],
        b=[p[1] for p in pairs],
        c=[p[2] for p in pairs],
        d=[p[3] for p in pairs],
    )
    for (lon1, lat1, lon2, lat2), ours in zip(pairs, got, strict=True):
        # DuckDB reads (latitude, longitude) here; see test_diff_geospatial_geodesy.py.
        want = _one(
            spatial,
            "SELECT ST_Distance_Spheroid(ST_Point(?, ?), ST_Point(?, ?))",
            lat1,
            lon1,
            lat2,
            lon2,
        )
        assert ours is not None, (lon1, lat1, lon2, lat2)
        assert ours == pytest.approx(want, rel=1e-9, abs=1e-6), (lon1, lat1, lon2, lat2)


MEASURED = [
    "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))",
    "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 2 4, 4 4, 4 2, 2 2))",
    "POLYGON((170 -10, -170 -10, -170 10, 170 10, 170 -10))",
    "POLYGON((-10 60, 30 60, 30 80, -10 80, -10 60))",
    "POLYGON((0 0, 1 1, 2 2, 0 0))",
    "POLYGON((-120 -50, -60 -50, -60 -20, -120 -20, -120 -50))",
    "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)), ((5 5, 6 5, 6 6, 5 6, 5 5)))",
    "LINESTRING(0 0, 3 4)",
    "LINESTRING(179 0, -179 0)",
    "LINESTRING(0 0, 1 1, 0 1, 1 0)",
    "MULTILINESTRING((0 0, 1 1), (1 1, 2 0))",
    "LINESTRING(-74 40.7, 2.35 48.85, 139.7 35.7)",
    "POINT(1 2)",
]


@pytest.mark.parametrize(
    ("ours", "theirs", "rel"),
    [
        (bt.st_length_spheroid, "ST_Length_Spheroid", 1e-9),
        (bt.st_perimeter_spheroid, "ST_Perimeter_Spheroid", 1e-9),
        (bt.st_area_spheroid, "ST_Area_Spheroid", 1e-6),
    ],
    ids=["length", "perimeter", "area"],
)
def test_spheroid_measures_match_geographiclib(spatial, ours, theirs, rel):
    """Length/perimeter used to sum haversine (0.3% off) and area was spherical excess.

    The antimeridian box was 8.37e13 m^2 against GeographicLib's 4.948e12.
    """
    got = _column(ours(bt.col("g")), g=MEASURED)
    for geom, value in zip(MEASURED, got, strict=True):
        want = _one(spatial, f"SELECT {theirs}(ST_FlipCoordinates(ST_GeomFromText(?)))", geom)
        assert value == pytest.approx(want, rel=rel, abs=1e-3), geom


def test_an_off_globe_vertex_nulls_the_spheroid_measure():
    got = _column(
        bt.st_length_spheroid(bt.col("g")), g=["LINESTRING(0 0, 200 0)", "LINESTRING(0 0, 1 0)"]
    )
    assert got[0] is None
    assert got[1] == pytest.approx(111_319.490_793, rel=1e-9)


# --- buffers -------------------------------------------------------------------------


def _star(seed: int, n: int) -> str:
    rng = random.Random(seed)
    pts = []
    for k in range(n):
        r = rng.uniform(3.0, 10.0)
        t = 2 * math.pi * k / n
        pts.append(f"{r * math.cos(t):.6f} {r * math.sin(t):.6f}")
    pts.append(pts[0])
    return f"POLYGON(({', '.join(pts)}))"


BUFFERS = [
    ("POINT(0 0)", 1.0),
    ("MULTIPOINT((0 0), (10 0))", 1.0),
    ("MULTIPOINT((0 0), (1 0))", 1.0),
    ("MULTIPOINT((0 0), (1 0), (0.5 0.8))", 1.0),
    ("LINESTRING(0 0, 10 0)", 1.0),
    ("LINESTRING(0 0, 10 0, 10 10)", 1.0),
    ("LINESTRING(0 0, 10 0, 0 1)", 0.5),
    ("LINESTRING(0 0, 5 5, 10 0, 15 5)", 2.0),
    ("MULTILINESTRING((0 0, 10 0), (0 5, 10 5))", 1.0),
    ("MULTILINESTRING((0 0, 10 0), (0 5, 10 5))", 3.0),
    ("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))", 1.0),
    ("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))", -1.0),
    ("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))", -3.0),
    ("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))", 0.0),
    ("POLYGON((0 0, 10 0, 10 10, 5 1, 0 10, 0 0))", 0.5),
    # The bridge under the notch is 1 wide, so -0.5 would pinch it to a single point:
    # the same point set GEOS writes as one self-touching polygon and this writes as two
    # touching ones. Either side of that knife edge the part count is unambiguous.
    ("POLYGON((0 0, 10 0, 10 10, 5 1, 0 10, 0 0))", -0.4),
    ("POLYGON((0 0, 10 0, 10 10, 5 1, 0 10, 0 0))", -0.6),
    ("POLYGON((0 0, 10 0, 10 10, 5 1, 0 10, 0 0))", 2.0),
    ("POLYGON((0 0, 6 0, 6 2, 2 2, 2 6, 0 6, 0 0))", -0.5),
    ("POLYGON((0 0, 6 0, 6 2, 2 2, 2 6, 0 6, 0 0))", 1.0),
    ("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (4 4, 6 4, 6 6, 4 6, 4 4))", 0.5),
    ("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (4 4, 6 4, 6 6, 4 6, 4 4))", 1.5),
    ("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (4 4, 6 4, 6 6, 4 6, 4 4))", -0.5),
    ("MULTIPOLYGON(((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))", 1.0),
    ("MULTIPOLYGON(((0 0, 2 0, 2 2, 0 2, 0 0)), ((3 0, 5 0, 5 2, 3 2, 3 0)))", 0.2),
    (_star(1, 24), 0.4),
    (_star(2, 40), -0.4),
    (_star(3, 60), 1.5),
    ("LINESTRING(0 0, 0 0)", 1.0),
    ("POINT(0 0)", 0.0),
    ("LINESTRING(0 0, 1 0)", -1.0),
]


@pytest.mark.parametrize("quad_segs", [2, 8])
def test_buffer_matches_geos_by_area_and_part_count(spatial, quad_segs):
    """The hull-based buffer gave 23.1 for two separate unit discs and 120.8 for a
    concave polygon GEOS buffers to 80.9, and returned an empty polygon for -1 on a
    4 x 4 square instead of the 2 x 2 one.

    Areas agree to the difference between GEOS's arcs, which start at each segment's
    offset point, and the vertex discs here, whose vertices sit at fixed angles; that is
    a fraction of the area between a circle and its inscribed polygon per vertex.
    """
    geoms = [g for g, _ in BUFFERS]
    radii = [r for _, r in BUFFERS]
    buffered = bt.st_buffer(bt.col("g"), bt.col("r"), quad_segs)
    got = (
        bt.from_pydict({"g": geoms, "r": radii})
        .select(
            a=bt.st_area(buffered), n=bt.st_num_geometries(buffered), e=bt.st_is_empty(buffered)
        )
        .to_pydict()
    )
    tol = 0.01 if quad_segs == 8 else 0.05
    for (geom, r), area, parts, empty in zip(BUFFERS, got["a"], got["n"], got["e"], strict=True):
        want_area, want_parts, want_empty = spatial.execute(
            "SELECT ST_Area(b), ST_NumGeometries(b), ST_IsEmpty(b) "
            "FROM (SELECT ST_Buffer(ST_GeomFromText(?), ?, ?) AS b)",
            [geom, r, quad_segs],
        ).fetchone()
        label = f"{geom} r={r} q={quad_segs}"
        assert empty == want_empty, label
        if want_empty:
            continue
        assert parts == want_parts, f"{label}: {parts} parts, GEOS has {want_parts}"
        assert area == pytest.approx(want_area, rel=tol), label


def test_a_negative_buffer_of_a_square_is_the_inset_square(spatial):
    """Used to be ``POLYGON EMPTY`` for every radius at or below zero."""
    g = "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))"
    got = _column(bt.st_as_text(bt.st_buffer(bt.col("g"), -1.0, 8)), g=[g])[0]
    assert _one(
        spatial,
        "SELECT ST_Equals(ST_GeomFromText(?), ST_GeomFromText(?))",
        got,
        "POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))",
    )


def test_a_zero_buffer_returns_the_polygon_and_empties_a_point():
    got = _column(
        bt.st_as_text(bt.st_buffer(bt.col("g"), 0.0, 8)),
        g=["POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))", "POINT(1 1)"],
    )
    assert got == ["POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))", "POLYGON EMPTY"]


# --- validity, simplicity, boundaries ------------------------------------------------

VALIDITY = [
    "POLYGON((0 0, 2 2, 2 0, 0 2, 0 0))",
    "POLYGON((0 0, 1 1, 2 2, 0 0))",
    "POLYGON((0 0, 0 0, 0 0, 0 0))",
    "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 2 1, 2 2, 1 1), (1 1, 2 1, 2 2, 1 1))",
    "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0), (0 0, 4 0, 4 4, 0 4, 0 0))",
    "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 1 1, 1 1, 1 1))",
    "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 5 1, 5 2, 1 2, 1 1))",
    "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 2 1, 2 2, 1 1), (2 2, 3 2, 3 3, 2 2))",
    "POLYGON((0 0, 4 0, 4 4, 2 0, 0 4, 0 0))",
    "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))",
    "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 0)), ((0 0, 1 0, 1 1, 0 0)))",
    "MULTIPOLYGON(((0 0, 2 0, 2 2, 0 2, 0 0)), ((1 1, 3 1, 3 3, 1 3, 1 1)))",
    "MULTIPOLYGON(((0 0, 2 2, 2 0, 0 2, 0 0)), ((5 5, 6 5, 6 6, 5 5)))",
    "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)), ((1 1, 2 1, 2 2, 1 2, 1 1)))",
    "LINESTRING(0 0, 0 0)",
    "LINESTRING(0 0, 0 0, 1 1)",
    "LINESTRING(0 0, 1 1, 0 1, 1 0)",
    "MULTILINESTRING((0 0, 0 0), (1 1, 2 2))",
    "MULTILINESTRING((0 0, 1 1), (1 1, 2 0))",
    "MULTIPOINT((0 0), (0 0))",
    "POINT(1 2)",
]


@pytest.mark.parametrize(
    ("ours", "theirs"), [(bt.st_is_valid, "ST_IsValid"), (bt.st_is_simple, "ST_IsSimple")]
)
def test_validity_and_simplicity_match_geos(spatial, ours, theirs):
    """``st_is_simple`` was true for every polygon, a bowtie included, and ``st_is_valid``
    was true for a four-times-repeated point, two identical holes and ``LINESTRING(0 0, 0 0)``."""
    got = _column(ours(bt.col("g")), g=VALIDITY)
    for geom, value in zip(VALIDITY, got, strict=True):
        assert value == _one(spatial, f"SELECT {theirs}(ST_GeomFromText(?))", geom), geom


BOUNDARIES = [
    "MULTILINESTRING((0 0, 1 1), (1 1, 2 0))",
    "MULTILINESTRING((0 0, 1 1), (1 1, 2 0), (1 1, 1 5))",
    "MULTILINESTRING((5 5, 1 1), (1 1, 2 0))",
    "MULTILINESTRING((0 0, 1 1), (1 1, 0 0))",
    "MULTILINESTRING((5 5, 6 6))",
    "LINESTRING(0 0, 3 4)",
]


def test_multiline_boundaries_follow_the_mod_2_rule(spatial):
    """``MULTILINESTRING((0 0, 1 1), (1 1, 2 0))`` used to have ``(1 1)`` in its boundary
    twice; the shared endpoint is interior."""
    got = _column(bt.st_as_text(bt.st_boundary(bt.col("g"))), g=BOUNDARIES)
    for geom, wkt in zip(BOUNDARIES, got, strict=True):
        want = _one(spatial, "SELECT ST_AsText(ST_Boundary(ST_GeomFromText(?)))", geom)
        same = _one(spatial, "SELECT ST_Equals(ST_GeomFromText(?), ST_GeomFromText(?))", wkt, want)
        both_empty = "EMPTY" in wkt and "EMPTY" in want
        assert same or both_empty, f"{geom}: {wkt} vs {want}"
        # GEOS lists a multi-chain's boundary in x-then-y order; so does this.
        ordered = _one(spatial, "SELECT ST_AsText(ST_GeomFromText(?))", wkt)
        assert ordered == want, geom


# --- Z -------------------------------------------------------------------------------

ZS = [
    "POINT Z (1 2 3)",
    "LINESTRING Z (0 0 0, 1 1 1)",
    "LINESTRING Z (0 0 0, 1 0 0, 1 1 3)",
    "MULTIPOINT Z ((0 0 1), (2 2 3))",
    "POLYGON Z ((0 0 0, 4 0 0, 4 4 4, 0 4 4, 0 0 0))",
    "POLYGON Z ((0 0 0, 4 0 0, 4 4 8, 0 0 0))",
]


def test_a_centroid_keeps_z_and_an_envelope_drops_it(spatial):
    """Centroid of ``POINT Z (1 2 3)`` came back at z = 0; the envelope of a 3D input
    wrote every corner at z = 0."""
    ds = bt.from_pydict({"g": ZS})
    got = ds.select(
        x=bt.st_x(bt.st_centroid(bt.col("g"))),
        y=bt.st_y(bt.st_centroid(bt.col("g"))),
        z=bt.st_z(bt.st_centroid(bt.col("g"))),
        env_has_z=bt.st_has_z(bt.st_envelope(bt.col("g"))),
    ).to_pydict()
    for i, geom in enumerate(ZS):
        wx, wy, wz, env_z = spatial.execute(
            "SELECT ST_X(c), ST_Y(c), ST_Z(c), ST_HasZ(ST_Envelope(g)) FROM "
            "(SELECT ST_Centroid(ST_GeomFromText(?)) AS c, ST_GeomFromText(?) AS g)",
            [geom, geom],
        ).fetchone()
        assert (got["x"][i], got["y"][i]) == (pytest.approx(wx), pytest.approx(wy)), geom
        assert got["z"][i] == pytest.approx(wz, abs=1e-12), geom
        assert got["env_has_z"][i] == env_z, geom


def test_force_3d_adds_z_and_never_overwrites_it(spatial):
    """``st_force_3d('POINT Z(0 0 1)', 5)`` used to overwrite the 1 with 5."""
    geoms = ["POINT Z (0 0 1)", "POINT(0 0)", "LINESTRING Z (0 0 1, 1 1 2)"]
    got = _column(bt.st_as_text(bt.st_force_3d(bt.col("g"), 5.0)), g=geoms)
    for geom, wkt in zip(geoms, got, strict=True):
        want = _one(spatial, "SELECT ST_AsText(ST_Force3DZ(ST_GeomFromText(?), 5))", geom)
        assert _one(spatial, "SELECT ST_AsText(ST_GeomFromText(?))", wkt) == want, geom


# --- accessors on the edges of their domain -------------------------------------------

POLY_WITH_HOLE = "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 2 4, 4 4, 4 2, 2 2))"


@pytest.mark.parametrize("n", [-3, 0, 1, 2])
def test_interior_ring_n_is_null_outside_one_to_the_hole_count(spatial, n):
    """``n`` of 0 and -3 used to return the first hole (the index was clamped to 1)."""
    got = _column(bt.st_as_text(bt.st_interior_ring_n(bt.col("g"), n)), g=[POLY_WITH_HOLE])[0]
    want = _one(
        spatial, "SELECT ST_AsText(ST_InteriorRingN(ST_GeomFromText(?), ?))", POLY_WITH_HOLE, n
    )
    if want is None:
        assert got is None
    else:
        assert _one(spatial, "SELECT ST_Equals(ST_GeomFromText(?), ST_GeomFromText(?))", got, want)


def test_exterior_ring_of_a_multipolygon_is_null(spatial):
    """Used to return the first member's shell."""
    g = "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)), ((5 5, 6 5, 6 6, 5 6, 5 5)))"
    assert _column(bt.st_exterior_ring(bt.col("g")), g=[g]) == [None]
    assert _one(spatial, "SELECT ST_ExteriorRing(ST_GeomFromText(?))", g) is None


@pytest.mark.parametrize(
    "geom",
    [
        "POINT EMPTY",
        "LINESTRING EMPTY",
        "POLYGON EMPTY",
        "MULTIPOINT EMPTY",
        "GEOMETRYCOLLECTION(POINT EMPTY)",
        "POINT(1 1)",
    ],
)
def test_num_geometries_of_an_empty_simple_geometry_is_zero(spatial, geom):
    """``POINT EMPTY`` used to report one member."""
    got = _column(bt.st_num_geometries(bt.col("g")), g=[geom])[0]
    assert got == _one(spatial, "SELECT ST_NumGeometries(ST_GeomFromText(?))", geom)


@pytest.mark.parametrize(
    ("geom", "tol"),
    [
        ("LINESTRING(0 0, 0 0)", 0.0),
        ("LINESTRING(0 0, 0 0, 0 0)", 0.0),
        ("LINESTRING(0 0, 5 5, 5.1 5)", 1.0),
        ("LINESTRING(0 0, 0.1 0, 0.2 0)", 1.0),
        ("LINESTRING(0 0, 1 1, 1 1, 2 2)", 0.0),
    ],
)
def test_remove_repeated_points_keeps_both_ends(spatial, geom, tol):
    """``LINESTRING(0 0, 0 0)`` used to thin to a one-position "line"."""
    got = _column(bt.st_as_text(bt.st_remove_repeated_points(bt.col("g"), tol)), g=[geom])[0]
    want = _one(
        spatial, "SELECT ST_AsText(ST_RemoveRepeatedPoints(ST_GeomFromText(?), ?))", geom, tol
    )
    assert _one(spatial, "SELECT ST_AsText(ST_GeomFromText(?))", got) == want


def test_an_unclosed_ring_is_measured_closed_and_reported_invalid(spatial):
    """``POLYGON((0 0, 1 0, 1 1))`` had an area of 0; DuckDB measures 0.5 and calls it
    invalid, and so does this now. PostGIS refuses to parse it; nulling it here would
    also hide it from ``st_is_valid_reason``, which is how the row gets found."""
    g = "POLYGON((0 0, 1 0, 1 1))"
    ds = bt.from_pydict({"g": [g]})
    got = ds.select(a=bt.st_area(bt.col("g")), ok=bt.st_is_valid(bt.col("g"))).to_pydict()
    assert got["a"] == [_one(spatial, "SELECT ST_Area(ST_GeomFromText(?))", g)]
    assert got["ok"] == [_one(spatial, "SELECT ST_IsValid(ST_GeomFromText(?))", g)]


def test_dwithin_values_match_geos(spatial):
    """``st_dwithin`` had no value test: the planar predicate, at and around the boundary."""
    cases = [
        ("POINT(0 0)", "POINT(3 4)", 5.0),
        ("POINT(0 0)", "POINT(3 4)", 4.999),
        ("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))", "POINT(5 1)", 3.0),
        ("POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))", "POINT(5 1)", 2.9),
        ("LINESTRING(0 0, 10 0)", "LINESTRING(5 1, 5 5)", 1.0),
        ("LINESTRING(0 0, 10 0)", "POINT(5 0)", 0.0),
        ("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))", "POINT(5 5)", 0.0),
        ("POINT EMPTY", "POINT(0 0)", 1.0),
    ]
    got = (
        bt.from_pydict(
            {"a": [c[0] for c in cases], "b": [c[1] for c in cases], "r": [c[2] for c in cases]}
        )
        .select(v=bt.st_dwithin(bt.col("a"), bt.col("b"), bt.col("r")))
        .to_pydict()["v"]
    )
    for (a, b, r), value in zip(cases, got, strict=True):
        want = _one(
            spatial, "SELECT ST_DWithin(ST_GeomFromText(?), ST_GeomFromText(?), ?)", a, b, r
        )
        assert value == want, (a, b, r)
