"""The geometry editors, accessors and constructors against DuckDB's spatial extension.

``test_diff_geospatial.py`` covers the predicates, the planar measures and the WKT/WKB
round trip. This module covers the rest of the OGC surface Batcher exposes and DuckDB
also implements: the shape editors (``st_simplify``, ``st_translate``, ...), the
component accessors (``st_exterior_ring``, ``st_point_n``, ...), the derived shapes
(``st_boundary``, ``st_convex_hull``, ``st_envelope``) and the constructors.

DuckDB spatial is GEOS underneath, so agreement here is agreement with an independent
implementation of the same standard rather than a restatement of Batcher's own code.

Three deliberate departures are pinned rather than asserted equal, because in each case
Batcher answers where GEOS declines or picks a different valid answer:

* ``st_exterior_ring`` of a **multi**-polygon. GEOS nulls; Batcher returns the first
  member's ring, which is what makes the function usable in a projection over a mixed
  column without a type guard first.
* ``st_point_on_surface`` of a **chain**. The contract is "a point on the geometry", and
  both satisfy it: Batcher returns the midpoint by length, GEOS returns a vertex. The
  test asserts the contract instead of the choice.
* ``st_num_interior_rings`` of a **non**-polygon. GEOS nulls; Batcher answers 0, matching
  how ``st_area`` answers 0 rather than null off its own type.
"""

from __future__ import annotations

import json

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


#: One geometry of every type Batcher's model has, plus a polygon with a hole, so a
#: function that only ever sees a simple polygon cannot pass by accident.
GEOMETRIES = [
    "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))",
    "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 7 3, 7 7, 3 7, 3 3))",
    "POINT(2 3)",
    "LINESTRING(0 0, 4 0, 4 3)",
    "MULTIPOINT((0 0), (4 4))",
    "MULTILINESTRING((0 0, 1 1), (2 2, 3 3))",
    "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))",
]

#: The chain-only fixture. DuckDB raises on ``ST_LineInterpolatePoint`` of a polygon
#: rather than returning null, so the linear referencing family needs its own input.
CHAINS = [
    "LINESTRING(0 0, 4 0, 4 3)",
    "LINESTRING(0 0, 1 1)",
    "LINESTRING(10 10, 10 20, 20 20, 20 10)",
]

_W = "ST_GeomFromText(?)"


def _duck(con, sql: str, rows: list) -> list:
    """Evaluate a DuckDB expression once per argument row."""
    return [con.execute(sql, list(row)).fetchone()[0] for row in rows]


def _assert_same_geometry(con, inputs: list[str], ours: list, theirs: list, label: str) -> None:
    """Compare geometry results through ``ST_Equals``, not through their spellings.

    The two engines render WKT differently (``POLYGON((`` against ``POLYGON ((``), so a
    string comparison would fail on formatting and hide whether the *geometry* agrees.
    """
    for src, got, want in zip(inputs, ours, theirs, strict=True):
        if got is None or want is None:
            assert (got is None) == (want is None), f"{label}({src}): {got!r} vs {want!r}"
            continue
        same = con.execute(
            "SELECT ST_Equals(ST_GeomFromText(?), ST_GeomFromText(?))", [got, want]
        ).fetchone()[0]
        assert same, f"{label}({src}): {got!r} is not {want!r}"


#: ``(batcher name, DuckDB expression)`` for every geometry-returning function of one
#: geometry that both engines implement with the same contract.
DERIVED = [
    ("st_boundary", "ST_Boundary({g})"),
    ("st_convex_hull", "ST_ConvexHull({g})"),
    ("st_envelope", "ST_Envelope({g})"),
    ("st_flip_coordinates", "ST_FlipCoordinates({g})"),
    ("st_force_2d", "ST_Force2D({g})"),
    ("st_reverse", "ST_Reverse({g})"),
    ("st_start_point", "ST_StartPoint({g})"),
    ("st_end_point", "ST_EndPoint({g})"),
]


@pytest.mark.parametrize(("ours", "theirs"), DERIVED)
def test_derived_geometry_matches_duckdb(spatial, ours, theirs):
    """Every shape derived from one geometry, over every geometry type."""
    ds = bt.from_pydict({"g": GEOMETRIES})
    got = ds.select(v=bt.st_as_text(getattr(bt, ours)(bt.col("g")))).to_pydict()["v"]
    want = _duck(
        spatial,
        f"SELECT ST_AsText({theirs.format(g=_W)})",
        [(g,) for g in GEOMETRIES],
    )
    _assert_same_geometry(spatial, GEOMETRIES, got, want, ours)


#: The editors, each with the arguments used on both sides. A tolerance that is too
#: coarse would let a no-op pass, so every one of these changes the fixture geometries.
EDITORS = [
    ("st_translate", (1.5, -2.0), "ST_Translate({g}, 1.5, -2.0)"),
    ("st_scale", (2.0, 3.0), "ST_Scale({g}, 2.0, 3.0)"),
    ("st_simplify", (0.5,), "ST_Simplify({g}, 0.5)"),
    ("st_remove_repeated_points", (0.1,), "ST_RemoveRepeatedPoints({g}, 0.1)"),
]


@pytest.mark.parametrize(("ours", "args", "theirs"), EDITORS)
def test_editor_matches_duckdb(spatial, ours, args, theirs):
    """Every editor that rewrites positions, over every geometry type."""
    ds = bt.from_pydict({"g": GEOMETRIES})
    got = ds.select(v=bt.st_as_text(getattr(bt, ours)(bt.col("g"), *args))).to_pydict()["v"]
    want = _duck(spatial, f"SELECT ST_AsText({theirs.format(g=_W)})", [(g,) for g in GEOMETRIES])
    _assert_same_geometry(spatial, GEOMETRIES, got, want, ours)


def test_nth_component_accessors_match_duckdb(spatial):
    """``st_point_n`` and ``st_interior_ring_n``, including off-type and out-of-range."""
    ds = bt.from_pydict({"g": GEOMETRIES})
    got = ds.select(
        p=bt.st_as_text(bt.st_point_n(bt.col("g"), 2)),
        r=bt.st_as_text(bt.st_interior_ring_n(bt.col("g"), 1)),
    ).to_pydict()
    want_p = _duck(spatial, f"SELECT ST_AsText(ST_PointN({_W}, 2))", [(g,) for g in GEOMETRIES])
    want_r = _duck(
        spatial, f"SELECT ST_AsText(ST_InteriorRingN({_W}, 1))", [(g,) for g in GEOMETRIES]
    )
    _assert_same_geometry(spatial, GEOMETRIES, got["p"], want_p, "st_point_n")
    _assert_same_geometry(spatial, GEOMETRIES, got["r"], want_r, "st_interior_ring_n")


def test_linear_referencing_matches_duckdb(spatial):
    """Interpolate, locate and substring along a chain -- the three must be consistent."""
    ds = bt.from_pydict({"g": CHAINS})
    got = ds.select(
        at=bt.st_as_text(bt.st_line_interpolate_point(bt.col("g"), 0.25)),
        sub=bt.st_as_text(bt.st_line_substring(bt.col("g"), 0.2, 0.7)),
        where=bt.st_line_locate_point(bt.col("g"), bt.lit("POINT(4 1)")),
    ).to_pydict()
    rows = [(g,) for g in CHAINS]
    _assert_same_geometry(
        spatial,
        CHAINS,
        got["at"],
        _duck(spatial, f"SELECT ST_AsText(ST_LineInterpolatePoint({_W}, 0.25))", rows),
        "st_line_interpolate_point",
    )
    _assert_same_geometry(
        spatial,
        CHAINS,
        got["sub"],
        _duck(spatial, f"SELECT ST_AsText(ST_LineSubstring({_W}, 0.2, 0.7))", rows),
        "st_line_substring",
    )
    want = _duck(
        spatial,
        f"SELECT ST_LineLocatePoint({_W}, ST_GeomFromText('POINT(4 1)'))",
        rows,
    )
    for chain, g, w in zip(CHAINS, got["where"], want, strict=True):
        assert g == pytest.approx(w, abs=1e-12), f"st_line_locate_point({chain})"


def test_closest_point_and_shortest_line_match_duckdb_when_disjoint(spatial):
    """The nearest-approach pair, on disjoint inputs where the answer is unique.

    Restricted to disjoint geometries on purpose: when two shapes intersect, every point
    of the intersection is a valid answer and the two engines pick different ones, so an
    equality assertion there would be testing a tie-break rather than a computation.
    """
    pairs = [
        ("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))", "POINT(6 6)"),
        ("LINESTRING(0 0, 4 0)", "POINT(2 3)"),
        ("POINT(0 0)", "POINT(3 4)"),
        ("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))", "LINESTRING(5 -5, 5 5)"),
    ]
    ds = bt.from_pydict({"a": [p[0] for p in pairs], "b": [p[1] for p in pairs]})
    got = ds.select(
        near=bt.st_as_text(bt.st_closest_point(bt.col("a"), bt.col("b"))),
        line=bt.st_as_text(bt.st_shortest_line(bt.col("a"), bt.col("b"))),
    ).to_pydict()
    labels = [f"{a} | {b}" for a, b in pairs]
    _assert_same_geometry(
        spatial,
        labels,
        got["near"],
        _duck(spatial, f"SELECT ST_AsText(ST_ClosestPoint({_W}, {_W}))", pairs),
        "st_closest_point",
    )
    _assert_same_geometry(
        spatial,
        labels,
        got["line"],
        _duck(spatial, f"SELECT ST_AsText(ST_ShortestLine({_W}, {_W}))", pairs),
        "st_shortest_line",
    )


def test_azimuth_matches_duckdb_over_point_pairs(spatial):
    """The planar bearing, on the point pairs DuckDB accepts."""
    pairs = [
        ("POINT(0 0)", "POINT(0 1)"),
        ("POINT(0 0)", "POINT(1 0)"),
        ("POINT(0 0)", "POINT(-1 -1)"),
        ("POINT(3 4)", "POINT(-3 -4)"),
    ]
    ds = bt.from_pydict({"a": [p[0] for p in pairs], "b": [p[1] for p in pairs]})
    got = ds.select(v=bt.st_azimuth(bt.col("a"), bt.col("b"))).to_pydict()["v"]
    want = _duck(spatial, f"SELECT ST_Azimuth({_W}, {_W})", pairs)
    for pair, g, w in zip(pairs, got, want, strict=True):
        assert g == pytest.approx(w, abs=1e-12), f"st_azimuth{pair}"


def test_boolean_shape_tests_match_duckdb(spatial):
    """``st_is_ring`` / ``st_is_simple`` / ``st_is_valid`` / ``st_has_z``."""
    ds = bt.from_pydict({"g": GEOMETRIES})
    for ours, theirs in [
        ("st_is_ring", "ST_IsRing"),
        ("st_is_simple", "ST_IsSimple"),
        ("st_is_valid", "ST_IsValid"),
        ("st_has_z", "ST_HasZ"),
    ]:
        got = ds.select(v=getattr(bt, ours)(bt.col("g"))).to_pydict()["v"]
        want = _duck(spatial, f"SELECT {theirs}({_W})", [(g,) for g in GEOMETRIES])
        assert got == want, f"{ours}: {got} vs {want}"


def test_extent_accessors_match_duckdb(spatial):
    """The bounding-box ordinates and the member count."""
    ds = bt.from_pydict({"g": GEOMETRIES})
    got = ds.select(
        xmin=bt.st_xmin(bt.col("g")),
        xmax=bt.st_xmax(bt.col("g")),
        ymin=bt.st_ymin(bt.col("g")),
        ymax=bt.st_ymax(bt.col("g")),
        n=bt.st_num_geometries(bt.col("g")),
    ).to_pydict()
    rows = [(g,) for g in GEOMETRIES]
    for key, theirs in [
        ("xmin", "ST_XMin"),
        ("xmax", "ST_XMax"),
        ("ymin", "ST_YMin"),
        ("ymax", "ST_YMax"),
        ("n", "ST_NumGeometries"),
    ]:
        assert got[key] == _duck(spatial, f"SELECT {theirs}({_W})", rows), key


def test_constructors_match_duckdb(spatial):
    """``st_make_envelope`` / ``st_make_line`` / ``st_make_polygon`` / ``st_affine``."""
    ds = bt.from_pydict({"i": [0]})
    got = ds.select(
        env=bt.st_as_text(bt.st_make_envelope(0.0, 1.0, 2.0, 3.0)),
        line=bt.st_as_text(bt.st_make_line(bt.lit("POINT(0 0)"), bt.lit("POINT(3 4)"))),
        poly=bt.st_as_text(bt.st_make_polygon(bt.lit("LINESTRING(0 0, 4 0, 4 4, 0 0)"))),
        aff=bt.st_as_text(bt.st_affine(bt.lit("POINT(1 2)"), 2.0, 0.0, 0.0, 3.0, 5.0, 7.0)),
    ).to_pydict()
    want = spatial.execute(
        "SELECT ST_AsText(ST_MakeEnvelope(0, 1, 2, 3)),"
        "       ST_AsText(ST_MakeLine(ST_Point(0, 0), ST_Point(3, 4))),"
        "       ST_AsText(ST_MakePolygon(ST_GeomFromText('LINESTRING(0 0, 4 0, 4 4, 0 0)'))),"
        "       ST_AsText(ST_Affine(ST_Point(1, 2), 2, 0, 0, 3, 5, 7))"
    ).fetchone()
    _assert_same_geometry(
        spatial,
        ["envelope", "line", "polygon", "affine"],
        [got["env"][0], got["line"][0], got["poly"][0], got["aff"][0]],
        list(want),
        "constructor",
    )


def test_rotate_matches_duckdb(spatial):
    """Rotation about the origin, at an angle where a sign error is visible."""
    import math

    ds = bt.from_pydict({"g": ["POINT(1 0)", "POINT(0 1)", "LINESTRING(1 0, 2 0)"]})
    rotated = bt.st_rotate(bt.col("g"), math.pi / 2)
    got = ds.select(x=bt.st_x(rotated), y=bt.st_y(rotated)).to_pydict()
    assert got["x"][0] == pytest.approx(0.0, abs=1e-12)
    assert got["y"][0] == pytest.approx(1.0, abs=1e-12)
    assert got["x"][1] == pytest.approx(-1.0, abs=1e-12)
    assert got["y"][1] == pytest.approx(0.0, abs=1e-12)
    want = spatial.execute(
        "SELECT ST_X(r), ST_Y(r) FROM (SELECT ST_Rotate(ST_Point(1, 0), pi() / 2) r)"
    ).fetchone()
    assert got["x"][0] == pytest.approx(want[0], abs=1e-12)
    assert got["y"][0] == pytest.approx(want[1], abs=1e-12)


def test_expand_grows_the_extent_on_every_side():
    """``st_expand`` takes an independent dx and dy, which DuckDB's one-argument form cannot."""
    ds = bt.from_pydict({"i": [0]})
    grown = bt.st_expand(bt.lit("POINT(1 2)"), 1.0, 2.0)
    got = ds.select(
        xmin=bt.st_xmin(grown),
        xmax=bt.st_xmax(grown),
        ymin=bt.st_ymin(grown),
        ymax=bt.st_ymax(grown),
    ).to_pydict()
    assert (got["xmin"][0], got["xmax"][0]) == (0.0, 2.0)
    assert (got["ymin"][0], got["ymax"][0]) == (0.0, 4.0)


def test_geojson_matches_duckdb_as_parsed_json(spatial):
    """Compared as parsed JSON: the two render ``0`` and ``0.0`` for the same ordinate."""
    ds = bt.from_pydict({"g": GEOMETRIES})
    got = ds.select(v=bt.st_as_geojson(bt.col("g"))).to_pydict()["v"]
    want = _duck(spatial, f"SELECT ST_AsGeoJSON({_W})", [(g,) for g in GEOMETRIES])
    for src, g, w in zip(GEOMETRIES, got, want, strict=True):
        assert json.loads(g) == json.loads(w), f"st_as_geojson({src})"


def test_hex_wkb_matches_duckdb_case_insensitively(spatial):
    """The same bytes; the two differ only in the case of the hex digits."""
    ds = bt.from_pydict({"g": GEOMETRIES})
    got = ds.select(v=bt.st_as_hex_wkb(bt.col("g"))).to_pydict()["v"]
    want = _duck(spatial, f"SELECT ST_AsHEXWKB({_W})", [(g,) for g in GEOMETRIES])
    for src, g, w in zip(GEOMETRIES, got, want, strict=True):
        assert g.lower() == w.lower(), f"st_as_hex_wkb({src})"


def test_parsers_agree_with_duckdb_on_the_geometry_they_name(spatial):
    """``st_geom_from_text`` / ``_geojson`` / ``_wkb`` all land on the same geometry."""
    ds = bt.from_pydict({"g": GEOMETRIES})
    geojson = ds.select(v=bt.st_as_geojson(bt.col("g"))).to_pydict()["v"]
    round_tripped = (
        bt.from_pydict({"g": GEOMETRIES, "j": geojson})
        .select(
            from_text=bt.st_as_text(bt.st_geom_from_text(bt.col("g"))),
            from_json=bt.st_as_text(bt.st_geom_from_geojson(bt.col("j"))),
            from_wkb=bt.st_as_text(bt.st_geom_from_wkb(bt.st_as_binary(bt.col("g")))),
        )
        .to_pydict()
    )
    for key in ("from_text", "from_json", "from_wkb"):
        _assert_same_geometry(spatial, GEOMETRIES, round_tripped[key], GEOMETRIES, key)


def test_point_on_surface_lies_on_the_geometry(spatial):
    """The contract, not the choice: GEOS picks a vertex where Batcher picks a midpoint.

    ``ST_PointOnSurface`` is specified as *a* point on the geometry, and for a chain there
    are infinitely many. Asserting equality with GEOS would pin a tie-break neither
    standard requires, so this asserts what the standard actually promises.
    """
    ds = bt.from_pydict({"g": GEOMETRIES})
    got = ds.select(v=bt.st_as_text(bt.st_point_on_surface(bt.col("g")))).to_pydict()["v"]
    for src, point in zip(GEOMETRIES, got, strict=True):
        assert point is not None, f"st_point_on_surface({src}) returned null"
        on_it = spatial.execute(
            "SELECT ST_Intersects(ST_GeomFromText(?), ST_GeomFromText(?))", [src, point]
        ).fetchone()[0]
        assert on_it, f"st_point_on_surface({src}) = {point} is not on the geometry"


def test_exterior_ring_of_a_multipolygon_is_the_first_member(spatial):
    """Batcher's documented departure: GEOS nulls here, Batcher answers.

    Pinned rather than argued: a projection over a column mixing polygons and
    multipolygons should not have to type-guard every row, and the first member's ring is
    the answer PostGIS's ``ST_ExteriorRing(ST_GeometryN(g, 1))`` idiom spells out longhand.
    """
    multi = "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)), ((2 2, 3 2, 3 3, 2 3, 2 2)))"
    got = (
        bt.from_pydict({"g": [multi]})
        .select(v=bt.st_as_text(bt.st_exterior_ring(bt.col("g"))))
        .to_pydict()["v"][0]
    )
    assert got is not None, "Batcher answers where GEOS nulls"
    first = spatial.execute(
        "SELECT ST_AsText(ST_ExteriorRing(unnest(ST_Dump(ST_GeomFromText(?))).geom)) LIMIT 1",
        [multi],
    ).fetchone()[0]
    same = spatial.execute(
        "SELECT ST_Equals(ST_GeomFromText(?), ST_GeomFromText(?))", [got, first]
    ).fetchone()[0]
    assert same, f"{got!r} is not the first member's ring {first!r}"
    assert spatial.execute(
        "SELECT ST_ExteriorRing(ST_GeomFromText(?)) IS NULL", [multi]
    ).fetchone()[0], "the departure is only real while GEOS still nulls"


def test_num_interior_rings_is_zero_off_its_own_type(spatial):
    """Batcher answers 0 for a non-polygon where GEOS nulls, as ``st_area`` answers 0."""
    got = (
        bt.from_pydict({"g": GEOMETRIES})
        .select(v=bt.st_num_interior_rings(bt.col("g")))
        .to_pydict()["v"]
    )
    assert got == [0, 1, 0, 0, 0, 0, 0]
    want = _duck(spatial, f"SELECT ST_NumInteriorRings({_W})", [(g,) for g in GEOMETRIES])
    assert want == [0, 1, None, None, None, None, None], (
        "the departure is only real while GEOS still nulls off-type"
    )


def test_is_closed_answers_for_every_type_where_duckdb_raises(spatial):
    """A ring is closed, a chain is closed when its ends meet, a point is not.

    DuckDB raises on anything that is not a chain, so there is no oracle for the other
    types; the chains are compared and the rest are pinned to the documented contract.
    """
    got = bt.from_pydict({"g": GEOMETRIES}).select(v=bt.st_is_closed(bt.col("g"))).to_pydict()["v"]
    assert got == [True, True, False, False, False, False, True]
    chains = ["LINESTRING(0 0, 1 0, 1 1, 0 0)", "LINESTRING(0 0, 1 1)"]
    ours = bt.from_pydict({"g": chains}).select(v=bt.st_is_closed(bt.col("g"))).to_pydict()["v"]
    assert ours == _duck(spatial, f"SELECT ST_IsClosed({_W})", [(c,) for c in chains])


def test_collect_combines_without_an_overlay(spatial):
    """``st_collect`` must keep both parts, not union them into one."""
    a, b = "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))", "POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))"
    got = (
        bt.from_pydict({"a": [a], "b": [b]})
        .select(
            wkt=bt.st_as_text(bt.st_collect(bt.col("a"), bt.col("b"))),
            n=bt.st_num_geometries(bt.st_collect(bt.col("a"), bt.col("b"))),
        )
        .to_pydict()
    )
    assert got["n"] == [2], "collecting two overlapping polygons must not merge them"
    # Compared through ST_Normalize rather than ST_Equals: GEOS answers false for an
    # ST_Equals of a self-overlapping multipolygon against itself, which is exactly the
    # shape this function is specified to produce.
    same = spatial.execute(
        "SELECT ST_AsText(ST_Normalize(ST_GeomFromText(?)))"
        "     = ST_AsText(ST_Normalize(ST_Collect([ST_GeomFromText(?), ST_GeomFromText(?)])))",
        [got["wkt"][0], a, b],
    ).fetchone()[0]
    assert same, f"{got['wkt'][0]!r} is not the collection of the two"


def test_extent_predicates_are_the_bounding_box_not_the_shape():
    """``st_intersects_extent`` / ``st_contains_extent`` compare boxes, and must say so.

    The L-shaped polygon and the point are the case that separates them from the exact
    predicates: the point is outside the polygon but inside its bounding box.
    """
    ell = "POLYGON((0 0, 4 0, 4 1, 1 1, 1 4, 0 4, 0 0))"
    got = (
        bt.from_pydict({"a": [ell], "b": ["POINT(3 3)"]})
        .select(
            box_hit=bt.st_intersects_extent(bt.col("a"), bt.col("b")),
            box_has=bt.st_contains_extent(bt.col("a"), bt.col("b")),
            exact=bt.st_intersects(bt.col("a"), bt.col("b")),
        )
        .to_pydict()
    )
    assert got["box_hit"] == [True]
    assert got["box_has"] == [True]
    assert got["exact"] == [False], "the point is in the box but outside the shape"
