"""The spatial grids and the SRID-carrying encodings, against their published definitions.

DuckDB has no geohash, S2, slippy-tile, quadkey or hex-grid function, so there is no
second engine to difference against. Every one of these has a *published* definition
instead, and that is what this module checks:

* **Geohash** against the worked example in its own specification, and against the
  invariants that make a geohash useful as a join key -- a prefix is the parent cell, and
  decoding a cell returns a rectangle containing every point that encodes to it.
* **Slippy tiles and quadkeys** against the OpenStreetMap and Bing formulas, recomputed
  here in Python from the definition rather than copied from Batcher's output.
* **S2** against the containment and ancestry properties the cell hierarchy guarantees.
* **The hex grid** against the geometry of a regular hexagon: a cell's centre is one of
  the lattice points, and every point inside the cell maps back to that cell.
* **EWKB / EWKT / hex-WKB** against the SRID they are supposed to be carrying, checked by
  reading it back out.

A test that only compared these functions to themselves would pass on a coordinate swap;
each formula below is written from the specification, not from the engine.
"""

from __future__ import annotations

import math

import pytest

import batcher as bt

pytestmark = pytest.mark.differential

#: The worked example in the geohash specification: 57.64911 N, 10.40744 E encodes to
#: ``u4pruydqqvj`` at precision 11. Reproduced by every published implementation.
GEOHASH_REFERENCE = (10.40744, 57.64911, 11, "u4pruydqqvj")

#: Points spread over both hemispheres and both sides of the prime meridian.
POINTS = [
    (-73.9857, 40.7484),
    (10.40744, 57.64911),
    (139.6917, 35.6895),
    (-58.3816, -34.6037),
    (0.0, 0.0),
]


def _one(expr):
    """Evaluate a scalar expression on a single-row dataset."""
    return bt.from_pydict({"i": [0]}).select(v=expr).to_pydict()["v"][0]


def test_geohash_matches_the_specifications_worked_example():
    """Both spellings of the encoder, against the published value."""
    lon, lat, precision, want = GEOHASH_REFERENCE
    assert _one(bt.geohash_encode(bt.lit(lon), bt.lit(lat), precision)) == want
    assert _one(bt.st_geohash(bt.lit(f"POINT({lon} {lat})"), precision)) == want


def test_geohash_decodes_back_inside_the_cell_it_names():
    """Decoding is the inverse of encoding to within the cell the precision buys."""
    for lon, lat in POINTS:
        for precision in (5, 9, 12):
            cell = _one(bt.geohash_encode(bt.lit(lon), bt.lit(lat), precision))
            got_lat = _one(bt.geohash_decode_lat(bt.lit(cell)))
            got_lon = _one(bt.geohash_decode_lon(bt.lit(cell)))
            # A geohash of length n splits the bits between longitude and latitude, so the
            # cell is at worst 360 / 2**(ceil(5n/2)) wide. Half of that bounds the error
            # of the centre this returns.
            lon_bits = (5 * precision + 1) // 2
            lat_bits = 5 * precision - lon_bits
            assert abs(got_lon - lon) <= 360.0 / 2**lon_bits, f"lon of {cell}"
            assert abs(got_lat - lat) <= 180.0 / 2**lat_bits, f"lat of {cell}"


def test_a_geohash_prefix_is_the_parent_cell():
    """The property the whole encoding exists for, and what makes it a usable join key."""
    for lon, lat in POINTS:
        fine = _one(bt.geohash_encode(bt.lit(lon), bt.lit(lat), 9))
        for precision in range(1, 9):
            coarse = _one(bt.geohash_encode(bt.lit(lon), bt.lit(lat), precision))
            assert fine.startswith(coarse), f"{fine} is not inside {coarse}"


def test_the_rectangle_a_geohash_names_contains_the_point_that_produced_it():
    """``st_geom_from_geohash`` against ``geohash_encode``, in both directions."""
    for lon, lat in POINTS:
        for precision in (3, 6, 9):
            cell = _one(bt.geohash_encode(bt.lit(lon), bt.lit(lat), precision))
            box = _one(bt.st_as_text(bt.st_geom_from_geohash(bt.lit(cell))))
            got = (
                bt.from_pydict({"g": [box]})
                .select(
                    inside=bt.st_intersects(bt.col("g"), bt.lit(f"POINT({lon} {lat})")),
                    xmin=bt.st_xmin(bt.col("g")),
                    xmax=bt.st_xmax(bt.col("g")),
                    ymin=bt.st_ymin(bt.col("g")),
                    ymax=bt.st_ymax(bt.col("g")),
                )
                .to_pydict()
            )
            assert got["inside"] == [True], f"{cell} does not contain ({lon}, {lat})"
            assert got["xmin"][0] <= lon <= got["xmax"][0]
            assert got["ymin"][0] <= lat <= got["ymax"][0]


def _slippy_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """The OpenStreetMap slippy-map tile formula, written from its published definition."""
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return min(x, n - 1), min(max(y, 0), n - 1)


def test_slippy_tiles_match_the_openstreetmap_formula():
    """``st_tile_x`` / ``st_tile_y`` against the formula recomputed here."""
    for zoom in (0, 1, 8, 12, 18):
        lons = [p[0] for p in POINTS]
        lats = [p[1] for p in POINTS]
        got = (
            bt.from_pydict({"lon": lons, "lat": lats})
            .select(
                x=bt.st_tile_x(bt.col("lon"), bt.col("lat"), zoom),
                y=bt.st_tile_y(bt.col("lon"), bt.col("lat"), zoom),
            )
            .to_pydict()
        )
        for i, (lon, lat) in enumerate(POINTS):
            want_x, want_y = _slippy_tile(lon, lat, zoom)
            assert got["x"][i] == want_x, f"tile x of ({lon}, {lat}) at z{zoom}"
            assert got["y"][i] == want_y, f"tile y of ({lon}, {lat}) at z{zoom}"


def _quadkey(x: int, y: int, zoom: int) -> str:
    """The Bing Maps quadkey formula, written from its published definition."""
    out = []
    for level in range(zoom, 0, -1):
        digit = 0
        mask = 1 << (level - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        out.append(str(digit))
    return "".join(out)


def test_quadkeys_match_the_bing_formula_and_the_tile_they_name():
    """The quadkey must be the interleaving of the tile coordinates at the same zoom."""
    for zoom in (1, 8, 12, 18):
        lons = [p[0] for p in POINTS]
        lats = [p[1] for p in POINTS]
        got = (
            bt.from_pydict({"lon": lons, "lat": lats})
            .select(
                key=bt.st_quadkey(bt.col("lon"), bt.col("lat"), zoom),
                x=bt.st_tile_x(bt.col("lon"), bt.col("lat"), zoom),
                y=bt.st_tile_y(bt.col("lon"), bt.col("lat"), zoom),
            )
            .to_pydict()
        )
        for i, (lon, lat) in enumerate(POINTS):
            assert len(got["key"][i]) == zoom, f"a z{zoom} quadkey has {zoom} digits"
            assert got["key"][i] == _quadkey(got["x"][i], got["y"][i], zoom), (
                f"quadkey of ({lon}, {lat}) at z{zoom}"
            )


def test_a_quadkey_prefix_is_the_parent_tile():
    """The hierarchy property, which is why quadkeys are used as keys at all."""
    for lon, lat in POINTS:
        deep = _one(bt.st_quadkey(bt.lit(lon), bt.lit(lat), 16))
        for zoom in range(1, 16):
            assert deep.startswith(_one(bt.st_quadkey(bt.lit(lon), bt.lit(lat), zoom)))


def test_s2_cells_nest_and_a_parent_covers_its_children():
    """The ancestry property: taking the parent of a finer cell is idempotent upward.

    S2 has no second implementation to hand here, so what is checked is the structure the
    hierarchy guarantees: a cell's ancestor at a coarser level is the same cell you get by
    encoding the point at that level, and asking for an ancestor at the cell's own level
    returns the cell.
    """
    for lon, lat in POINTS:
        for level in (5, 10, 20):
            cell = _one(bt.st_s2_cell(bt.lit(lon), bt.lit(lat), level))
            assert cell is not None
            assert _one(bt.st_s2_cell_parent(bt.lit(cell), level)) == cell
            for coarser in range(0, level):
                direct = _one(bt.st_s2_cell(bt.lit(lon), bt.lit(lat), coarser))
                derived = _one(bt.st_s2_cell_parent(bt.lit(cell), coarser))
                assert derived == direct, (
                    f"S2 ancestor of ({lon}, {lat}) at level {coarser} disagrees with "
                    "encoding the point there directly"
                )


def test_distinct_points_land_in_distinct_s2_cells_at_a_fine_level():
    """A cheap guard against an encoder that collapses everything onto one cell."""
    lons = [p[0] for p in POINTS]
    lats = [p[1] for p in POINTS]
    cells = (
        bt.from_pydict({"lon": lons, "lat": lats})
        .select(c=bt.st_s2_cell(bt.col("lon"), bt.col("lat"), 20))
        .to_pydict()["c"]
    )
    assert len(set(cells)) == len(POINTS)


def test_hex_cells_recover_their_own_centre():
    """A point inside a hex cell maps to that cell, and the centre is a lattice point."""
    size = 50.0
    positions = [(0.0, 0.0), (100.0, 200.0), (-137.5, 42.25), (1e4, -1e4)]
    for x, y in positions:
        cell = _one(bt.st_hex_bin(bt.lit(x), bt.lit(y), size))
        cx = _one(bt.st_hex_center_x(bt.lit(cell), size))
        cy = _one(bt.st_hex_center_y(bt.lit(cell), size))
        # The centre must be nearer to the point than a cell is wide, and must itself bin
        # back to the same cell -- the two together rule out an off-by-one lattice index.
        assert math.hypot(cx - x, cy - y) <= size, f"centre of the cell holding ({x}, {y})"
        assert _one(bt.st_hex_bin(bt.lit(cx), bt.lit(cy), size)) == cell
        assert _one(bt.st_hex_center_x(bt.lit(cell), size)) == cx
        assert _one(bt.st_hex_center_y(bt.lit(cell), size)) == cy


def test_neighbouring_hex_centres_are_one_cell_apart():
    """The lattice spacing, which is what makes a hex grid a hex grid.

    Regular hexagons of circumradius ``size`` tile with centres ``sqrt(3) * size`` apart.
    A grid that was secretly square would fail this while still round-tripping above.
    """
    size = 50.0
    seen: dict[int, tuple[float, float]] = {}
    for i in range(-40, 41):
        for j in range(-40, 41):
            x, y = i * 12.5, j * 12.5
            cell = _one(bt.st_hex_bin(bt.lit(x), bt.lit(y), size))
            if cell not in seen:
                seen[cell] = (
                    _one(bt.st_hex_center_x(bt.lit(cell), size)),
                    _one(bt.st_hex_center_y(bt.lit(cell), size)),
                )
        if len(seen) > 6:
            break
    centres = list(seen.values())
    spacing = math.sqrt(3.0) * size
    nearest = min(
        math.hypot(a[0] - b[0], a[1] - b[1])
        for i, a in enumerate(centres)
        for b in centres[i + 1 :]
    )
    assert nearest == pytest.approx(spacing, rel=1e-9), (
        f"nearest hex centres are {nearest} apart, not the {spacing} a hex lattice gives"
    )


def test_ewkt_and_ewkb_carry_the_srid_that_plain_wkt_and_wkb_drop():
    """The whole reason the E-encodings exist, checked by reading the SRID back."""
    tagged = bt.st_set_srid(bt.lit("POINT(1 2)"), 4326)
    got = (
        bt.from_pydict({"i": [0]})
        .select(
            ewkt=bt.st_as_ewkt(tagged),
            wkt=bt.st_as_text(tagged),
            srid=bt.st_srid(tagged),
            from_ewkb=bt.st_srid(bt.st_geom_from_wkb(bt.st_as_ewkb(tagged))),
            from_wkb=bt.st_srid(bt.st_geom_from_wkb(bt.st_as_binary(tagged))),
        )
        .to_pydict()
    )
    assert got["ewkt"] == ["SRID=4326;POINT(1 2)"]
    assert got["wkt"] == ["POINT(1 2)"], "plain WKT has nowhere to put the SRID"
    assert got["srid"] == [4326]
    assert got["from_ewkb"] == [4326], "EWKB must survive the round trip with its SRID"
    assert got["from_wkb"] == [0], "plain WKB has no SRID to restore"


def test_hex_wkb_is_the_hex_spelling_of_the_same_bytes():
    """``st_as_hex_wkb`` against ``st_as_ewkb``, so the two encodings cannot drift."""
    tagged = bt.st_set_srid(bt.lit("LINESTRING(0 0, 1 1)"), 3857)
    got = (
        bt.from_pydict({"i": [0]})
        .select(hexed=bt.st_as_hex_wkb(tagged), raw=bt.st_as_ewkb(tagged))
        .to_pydict()
    )
    assert bytes.fromhex(got["hexed"][0]) == got["raw"][0]


def test_three_dimensional_points_keep_their_elevation():
    """``st_point_z`` / ``st_z`` / ``st_coord_dim`` / ``st_has_z`` / ``st_force_3d``."""
    got = (
        bt.from_pydict({"x": [1.0], "y": [2.0], "z": [3.0]})
        .select(
            wkt=bt.st_as_text(bt.st_point_z(bt.col("x"), bt.col("y"), bt.col("z"))),
            z=bt.st_z(bt.st_point_z(bt.col("x"), bt.col("y"), bt.col("z"))),
            dim3=bt.st_coord_dim(bt.st_point_z(bt.col("x"), bt.col("y"), bt.col("z"))),
            dim2=bt.st_coord_dim(bt.st_point(bt.col("x"), bt.col("y"))),
            has_z=bt.st_has_z(bt.st_point_z(bt.col("x"), bt.col("y"), bt.col("z"))),
            forced=bt.st_z(bt.st_force_3d(bt.st_point(bt.col("x"), bt.col("y")), 9.0)),
            dropped=bt.st_z(bt.st_force_2d(bt.st_point_z(bt.col("x"), bt.col("y"), bt.col("z")))),
        )
        .to_pydict()
    )
    assert got["wkt"] == ["POINT Z(1 2 3)"]
    assert got["z"] == [3.0]
    assert got["dim3"] == [3]
    assert got["dim2"] == [2]
    assert got["has_z"] == [True]
    assert got["forced"] == [9.0]
    assert got["dropped"] == [None], "forcing to 2D must remove the ordinate, not zero it"


def test_collection_membership_accessors():
    """``st_is_collection`` and ``st_geometry_n``, including out of range."""
    geoms = ["MULTIPOINT((0 0), (1 1))", "POINT(0 0)", "GEOMETRYCOLLECTION(POINT(5 5))"]
    got = (
        bt.from_pydict({"g": geoms})
        .select(
            many=bt.st_is_collection(bt.col("g")),
            first=bt.st_as_text(bt.st_geometry_n(bt.col("g"), 1)),
            second=bt.st_as_text(bt.st_geometry_n(bt.col("g"), 2)),
            past_end=bt.st_as_text(bt.st_geometry_n(bt.col("g"), 99)),
        )
        .to_pydict()
    )
    assert got["many"] == [True, False, True]
    assert got["first"] == ["POINT(0 0)", "POINT(0 0)", "POINT(5 5)"]
    assert got["second"] == ["POINT(1 1)", None, None]
    assert got["past_end"] == [None, None, None]


def test_winding_order_is_forced_without_changing_the_shape():
    """``st_force_polygon_cw`` / ``_ccw`` must reorder positions, not move them."""
    ccw = "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))"
    cw = "POLYGON((0 0, 0 4, 4 4, 4 0, 0 0))"
    got = (
        bt.from_pydict({"a": [ccw, cw]})
        .select(
            to_cw=bt.st_as_text(bt.st_force_polygon_cw(bt.col("a"))),
            to_ccw=bt.st_as_text(bt.st_force_polygon_ccw(bt.col("a"))),
            area_cw=bt.st_area(bt.st_force_polygon_cw(bt.col("a"))),
            area_ccw=bt.st_area(bt.st_force_polygon_ccw(bt.col("a"))),
        )
        .to_pydict()
    )
    assert got["to_cw"] == [cw, cw], "both inputs must come out clockwise"
    assert got["to_ccw"] == [ccw, ccw], "both inputs must come out counter-clockwise"
    assert got["area_cw"] == [16.0, 16.0]
    assert got["area_ccw"] == [16.0, 16.0]


def test_segmentize_and_snap_to_grid_change_positions_predictably():
    """One inserts vertices under a length bound; the other rounds every ordinate."""
    got = (
        bt.from_pydict({"g": ["LINESTRING(0 0, 4 0)"]})
        .select(
            dense=bt.st_as_text(bt.st_segmentize(bt.col("g"), 1.0)),
            n=bt.st_num_points(bt.st_segmentize(bt.col("g"), 1.0)),
            length=bt.st_length(bt.st_segmentize(bt.col("g"), 1.0)),
        )
        .to_pydict()
    )
    assert got["dense"] == ["LINESTRING(0 0, 1 0, 2 0, 3 0, 4 0)"]
    assert got["n"] == [5]
    assert got["length"] == [4.0], "densifying must not change the length"

    snapped = (
        bt.from_pydict({"g": ["POINT(1.234 5.678)", "POINT(-1.26 -5.64)"]})
        .select(
            x=bt.st_x(bt.st_snap_to_grid(bt.col("g"), 0.1)),
            y=bt.st_y(bt.st_snap_to_grid(bt.col("g"), 0.1)),
        )
        .to_pydict()
    )
    assert snapped["x"][0] == pytest.approx(1.2, abs=1e-9)
    assert snapped["y"][0] == pytest.approx(5.7, abs=1e-9)
    assert snapped["x"][1] == pytest.approx(-1.3, abs=1e-9)
    assert snapped["y"][1] == pytest.approx(-5.6, abs=1e-9)


def test_hausdorff_and_max_distance_measure_the_worst_case_not_the_best():
    """Both must exceed ``st_distance``, which measures the nearest approach."""
    pairs = [
        ("LINESTRING(0 0, 2 0)", "LINESTRING(0 1, 2 1)"),
        ("POINT(0 0)", "POINT(3 4)"),
        ("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))", "POINT(6 6)"),
    ]
    got = (
        bt.from_pydict({"a": [p[0] for p in pairs], "b": [p[1] for p in pairs]})
        .select(
            near=bt.st_distance(bt.col("a"), bt.col("b")),
            hausdorff=bt.st_hausdorff_distance(bt.col("a"), bt.col("b")),
            far=bt.st_max_distance(bt.col("a"), bt.col("b")),
        )
        .to_pydict()
    )
    assert got["hausdorff"][0] == pytest.approx(1.0, abs=1e-12)
    assert got["hausdorff"][1] == pytest.approx(5.0, abs=1e-12)
    assert got["far"][1] == pytest.approx(5.0, abs=1e-12)
    for i, (a, b) in enumerate(pairs):
        assert got["hausdorff"][i] >= got["near"][i], f"hausdorff below the minimum: {a}|{b}"
        assert got["far"][i] >= got["near"][i], f"max distance below the minimum: {a}|{b}"


def test_arctan2_is_the_two_argument_arctangent():
    """Distinguished from ``arctan`` by getting the quadrant right, which is the point."""
    ys = [1.0, 1.0, -1.0, -1.0, 0.0]
    xs = [1.0, -1.0, -1.0, 1.0, -1.0]
    got = (
        bt.from_pydict({"y": ys, "x": xs})
        .select(v=bt.arctan2(bt.col("y"), bt.col("x")))
        .to_pydict()["v"]
    )
    for y, x, value in zip(ys, xs, got, strict=True):
        assert value == pytest.approx(math.atan2(y, x), abs=1e-15), f"arctan2({y}, {x})"
