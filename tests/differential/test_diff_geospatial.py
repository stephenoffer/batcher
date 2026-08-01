"""The geospatial surface against DuckDB's spatial extension, the closest oracle there is.

DuckDB spatial is GEOS underneath, so it is an *independent* implementation of the same
OGC standard rather than a restatement of the same code. That makes it the right oracle
for this surface: a predicate that agrees with GEOS on adjacency, containment and the
boundary cases is a predicate that agrees with PostGIS, Sedona and every other spatial
system a user might be porting from.

Three families are deliberately not compared here, with the reason in each case:

* **Grids** (geohash, S2, tiles). DuckDB has no equivalent, and the reference values are
  pinned against the published implementations in `bc-geo`'s own tests instead.
* **Geodesic measures.** DuckDB's `ST_Distance_Sphere` uses a different Earth radius, so
  agreement would be to a tolerance that hides real error rather than exposing it. Those
  are pinned against published distances in `bc_geo::proj::geodesy`.
* **`st_buffer`.** Batcher's is a documented approximation and GEOS's is exact, so they
  are not supposed to agree.
"""

from __future__ import annotations

import pytest

import batcher as bt

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


#: Geometries chosen to hit the cases that separate the predicates from each other:
#: identical shapes, shapes sharing only an edge, shapes sharing only a corner, one
#: inside another, a point exactly on a boundary, a line crossing an area, and a
#: geometry with a hole.
GEOMETRIES = [
    "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))",
    "POLYGON((0 0, 0 4, 4 4, 4 0, 0 0))",
    "POLYGON((4 0, 8 0, 8 4, 4 4, 4 0))",
    "POLYGON((4 4, 8 4, 8 8, 4 8, 4 4))",
    "POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))",
    "POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))",
    "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 7 3, 7 7, 3 7, 3 3))",
    "POINT(2 2)",
    "POINT(0 2)",
    "POINT(4 4)",
    "POINT(50 50)",
    "LINESTRING(-1 2, 5 2)",
    "LINESTRING(1 1, 3 3)",
    "LINESTRING(0 0, 4 0)",
    "MULTIPOINT((0 0), (4 4))",
]

PAIRS = [(a, b) for a in GEOMETRIES for b in GEOMETRIES]


def _duck_scalar(con, sql: str, args: tuple) -> list:
    """Evaluate a DuckDB spatial expression once per argument row."""
    return [con.execute(sql, list(row)).fetchone()[0] for row in args]


PREDICATES = [
    ("st_intersects", "ST_Intersects"),
    ("st_disjoint", "ST_Disjoint"),
    ("st_contains", "ST_Contains"),
    ("st_within", "ST_Within"),
    ("st_covers", "ST_Covers"),
    ("st_covered_by", "ST_CoveredBy"),
    ("st_touches", "ST_Touches"),
    ("st_crosses", "ST_Crosses"),
    ("st_overlaps", "ST_Overlaps"),
    ("st_equals", "ST_Equals"),
]


@pytest.mark.parametrize(("ours", "theirs"), PREDICATES)
def test_predicate_matches_duckdb_over_every_geometry_pair(spatial, ours, theirs):
    """Every predicate, over all 225 ordered pairs of the fixture geometries."""
    ds = bt.from_pydict({"a": [p[0] for p in PAIRS], "b": [p[1] for p in PAIRS]})
    got = ds.select(v=getattr(bt, ours)(bt.col("a"), bt.col("b"))).to_pydict()["v"]
    want = _duck_scalar(
        spatial,
        f"SELECT {theirs}(ST_GeomFromText(?), ST_GeomFromText(?))",
        PAIRS,
    )
    mismatches = [(a, b, g, w) for (a, b), g, w in zip(PAIRS, got, want, strict=True) if g != w]
    assert not mismatches, (
        f"{ours} disagrees with {theirs} on {len(mismatches)} pair(s): {mismatches[:4]}"
    )


MEASURES = [
    ("st_area", "ST_Area"),
    ("st_length", "ST_Length"),
    ("st_perimeter", "ST_Perimeter"),
]


@pytest.mark.parametrize(("ours", "theirs"), MEASURES)
def test_planar_measure_matches_duckdb(spatial, ours, theirs):
    ds = bt.from_pydict({"g": GEOMETRIES})
    got = ds.select(v=getattr(bt, ours)(bt.col("g"))).to_pydict()["v"]
    want = _duck_scalar(spatial, f"SELECT {theirs}(ST_GeomFromText(?))", [(g,) for g in GEOMETRIES])
    for geom, g, w in zip(GEOMETRIES, got, want, strict=True):
        assert g == pytest.approx(w, abs=1e-9), f"{ours}({geom}): {g} vs {w}"


def test_distance_matches_duckdb_over_every_pair(spatial):
    ds = bt.from_pydict({"a": [p[0] for p in PAIRS], "b": [p[1] for p in PAIRS]})
    got = ds.select(v=bt.st_distance(bt.col("a"), bt.col("b"))).to_pydict()["v"]
    want = _duck_scalar(
        spatial, "SELECT ST_Distance(ST_GeomFromText(?), ST_GeomFromText(?))", PAIRS
    )
    for (a, b), g, w in zip(PAIRS, got, want, strict=True):
        assert g == pytest.approx(w, abs=1e-9), f"st_distance({a}, {b}): {g} vs {w}"


ACCESSORS = [
    ("st_geometry_type", "ST_GeometryType"),
    ("st_dimension", "ST_Dimension"),
    ("st_num_points", "ST_NPoints"),
    ("st_is_empty", "ST_IsEmpty"),
    ("st_x", "ST_X"),
    ("st_y", "ST_Y"),
]


@pytest.mark.parametrize(("ours", "theirs"), ACCESSORS)
def test_accessor_matches_duckdb(spatial, ours, theirs):
    """The accessors, including the ones that are null off their own type.

    `ST_X` of a polygon is the case worth having: both must answer null rather than
    reaching for the first vertex, and an implementation that returned a vertex would
    look right on every point in the fixture.
    """
    ds = bt.from_pydict({"g": GEOMETRIES})
    got = ds.select(v=getattr(bt, ours)(bt.col("g"))).to_pydict()["v"]
    want = []
    for geom in GEOMETRIES:
        try:
            want.append(
                spatial.execute(f"SELECT {theirs}(ST_GeomFromText(?))", [geom]).fetchone()[0]
            )
        except Exception:
            want.append(None)
    for geom, g, w in zip(GEOMETRIES, got, want, strict=True):
        if isinstance(w, str):
            # DuckDB prefixes the OGC name; Batcher returns the bare keyword.
            w = w.removeprefix("ST_").upper()
        assert g == w, f"{ours}({geom}): {g} vs {w}"


def test_centroid_matches_duckdb():
    """Compared as coordinates, since the two render WKT with different precision."""
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    areal = [g for g in GEOMETRIES if g.startswith(("POLYGON", "MULTIPOINT", "LINESTRING"))]
    ds = bt.from_pydict({"g": areal})
    centroid = bt.st_centroid(bt.col("g"))
    got = ds.select(x=bt.st_x(centroid), y=bt.st_y(centroid)).to_pydict()
    for i, geom in enumerate(areal):
        wx, wy = con.execute(
            "SELECT ST_X(ST_Centroid(ST_GeomFromText(?))), ST_Y(ST_Centroid(ST_GeomFromText(?)))",
            [geom, geom],
        ).fetchone()
        assert got["x"][i] == pytest.approx(wx, abs=1e-9), f"centroid x of {geom}"
        assert got["y"][i] == pytest.approx(wy, abs=1e-9), f"centroid y of {geom}"


def test_wkt_round_trip_agrees_with_duckdb_on_the_geometry_it_names(spatial):
    """Our WKT must parse in DuckDB to a geometry equal to what we started from.

    Stronger than string comparison, which the two spell differently, and it is the
    property that actually matters: a geometry written by Batcher has to be readable by
    everything else.
    """
    ds = bt.from_pydict({"g": GEOMETRIES})
    ours = ds.select(v=bt.st_as_text(bt.col("g"))).to_pydict()["v"]
    for original, rendered in zip(GEOMETRIES, ours, strict=True):
        same = spatial.execute(
            "SELECT ST_Equals(ST_GeomFromText(?), ST_GeomFromText(?))", [original, rendered]
        ).fetchone()[0]
        assert same, f"our WKT {rendered!r} is not the geometry {original!r}"


def test_wkb_round_trips_through_duckdb(spatial):
    """The storage encoding, checked against an independent WKB reader."""
    ds = bt.from_pydict({"g": GEOMETRIES})
    blobs = ds.select(v=bt.st_as_binary(bt.col("g"))).to_pydict()["v"]
    for original, blob in zip(GEOMETRIES, blobs, strict=True):
        same = spatial.execute(
            "SELECT ST_Equals(ST_GeomFromWKB(?), ST_GeomFromText(?))", [blob, original]
        ).fetchone()[0]
        assert same, f"our WKB does not decode to {original!r}"


def test_a_null_and_an_unparseable_geometry_both_yield_null():
    """Batcher's documented departure from PostGIS, pinned so it cannot drift.

    A geometry column with one corrupt row must not abort a hundred-million-row scan, so
    the row nulls. The two failure modes are deliberately different and both appear here:
    text that is *not a geometry* nulls every function over it, while a geometry that is
    well-formed but **invalid** (a two-position ring) still parses and still measures --
    it is `st_is_valid_reason` that names it, which is why validity is a separate check
    rather than something folded into parsing.
    """
    ds = bt.from_pydict({"g": ["POINT(1 2)", None, "not a geometry", "POLYGON((0 0, 1 1))"]})
    got = ds.select(
        area=bt.st_area(bt.col("g")),
        kind=bt.st_geometry_type(bt.col("g")),
        why=bt.st_is_valid_reason(bt.col("g")),
    ).to_pydict()
    assert got["area"] == [0.0, None, None, 0.0]
    assert got["kind"] == ["POINT", None, None, "POLYGON"]
    assert got["why"][0] is None, "a valid geometry has no reason"
    assert got["why"][2] is None, "unparseable text nulls rather than reporting a reason"
    assert "at least 4" in got["why"][3], "the degenerate ring is named, not silently dropped"
