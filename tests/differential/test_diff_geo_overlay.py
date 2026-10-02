"""Polygon union, intersection and difference, against DuckDB spatial (GEOS).

The two engines are free to write the same point set with different vertex orders,
starting points and ring splits, so the comparison is on the point set itself: the area of
the symmetric difference between Batcher's result and GEOS's, computed by GEOS, must be
zero up to round-off, and the two areas must agree. That is a strictly stronger check than
comparing areas alone, which a result shifted sideways would pass.

The corpus covers the cases an overlay gets wrong: partial overlap, containment in both
directions, identical operands, disjoint operands (an empty result), a shared edge, a
shared vertex, a hole the other operand crosses, multipolygon operands, concave shapes,
and a set of pseudo-random convex polygons.
"""

from __future__ import annotations

import math
import random

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential

duckdb = pytest.importorskip("duckdb")

_OPS = {"union": bt.st_union, "intersection": bt.st_intersection, "difference": bt.st_difference}
_DUCK = {"union": "ST_Union", "intersection": "ST_Intersection", "difference": "ST_Difference"}


def _convex(rng: random.Random) -> str:
    cx, cy, r = rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(1, 6)
    angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(rng.randint(3, 9)))
    pts = [(cx + r * math.cos(t), cy + r * math.sin(t)) for t in angles]
    pts.append(pts[0])
    return "POLYGON((" + ", ".join(f"{x:.6f} {y:.6f}" for x, y in pts) + "))"


_SQ = "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))"
PAIRS = [
    (_SQ, "POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))"),
    (_SQ, "POLYGON((1 1, 2 1, 2 2, 1 2, 1 1))"),
    ("POLYGON((1 1, 2 1, 2 2, 1 2, 1 1))", _SQ),
    (_SQ, _SQ),
    (_SQ, "POLYGON((10 10, 11 10, 11 11, 10 11, 10 10))"),
    (_SQ, "POLYGON((4 0, 8 0, 8 4, 4 4, 4 0))"),
    (_SQ, "POLYGON((4 4, 8 4, 8 8, 4 8, 4 4))"),
    (_SQ, "POLYGON((4 1, 8 1, 8 3, 4 3, 4 1))"),
    (
        "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 7 3, 7 7, 3 7, 3 3))",
        "POLYGON((-1 4, 11 4, 11 6, -1 6, -1 4))",
    ),
    (
        "MULTIPOLYGON(((0 0, 2 0, 2 2, 0 2, 0 0)), ((5 5, 7 5, 7 7, 5 7, 5 5)))",
        "POLYGON((1 1, 6 1, 6 6, 1 6, 1 1))",
    ),
    (
        "POLYGON((0 0, 6 0, 6 6, 4 6, 4 2, 2 2, 2 6, 0 6, 0 0))",
        "POLYGON((-1 3, 7 3, 7 5, -1 5, -1 3))",
    ),
    (
        "POLYGON((0 0, 3 0, 3 1, 1 1, 1 2, 3 2, 3 3, 0 3, 0 0))",
        "POLYGON((2 -1, 4 -1, 4 4, 2 4, 2 -1))",
    ),
]
_RNG = random.Random(20260928)
PAIRS += [(_convex(_RNG), _convex(_RNG)) for _ in range(20)]


@pytest.fixture(scope="module")
def spatial():
    duck = duckdb.connect()
    try:
        duck.execute("INSTALL spatial; LOAD spatial;")
    except Exception as exc:  # pragma: no cover - offline / no extension available
        pytest.skip(f"duckdb spatial unavailable: {exc}")
    return duck


@pytest.mark.parametrize("op", sorted(_OPS))
def test_the_overlay_covers_the_same_point_set_as_geos(spatial, op):
    ds = bt.from_pydict({"a": [a for a, _ in PAIRS], "b": [b for _, b in PAIRS]})
    got = ds.select(v=bt.st_as_text(_OPS[op](col("a"), col("b")))).to_pydict()["v"]
    for (a, b), wkt in zip(PAIRS, got, strict=True):
        assert wkt is not None, (op, a, b)
        theirs = f"{_DUCK[op]}(ST_GeomFromText('{a}'), ST_GeomFromText('{b}'))"
        ours = f"ST_GeomFromText('{wkt}')"
        area_ours, area_theirs, sym = spatial.execute(
            f"SELECT ST_Area({ours}), ST_Area({theirs}), "
            f"ST_Area(ST_SymDifference({ours}, {theirs}))"
        ).fetchone()
        scale = max(1.0, area_theirs)
        assert area_ours == pytest.approx(area_theirs, rel=1e-9, abs=1e-9), (op, a, b, wkt)
        assert sym <= 1e-9 * scale, (op, a, b, wkt, sym)


def test_an_empty_result_is_the_empty_polygon_not_null():
    far = "POLYGON((10 10, 11 10, 11 11, 10 11, 10 10))"
    ds = bt.from_pydict({"a": [_SQ, _SQ], "b": [far, _SQ]})
    got = ds.select(
        i=bt.st_as_text(bt.st_intersection(col("a"), col("b"))),
        d=bt.st_as_text(bt.st_difference(col("a"), col("b"))),
    ).to_pydict()
    assert got == {
        "i": ["POLYGON EMPTY", "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))"],
        "d": [_SQ, "POLYGON EMPTY"],
    }


def test_a_null_a_line_or_an_invalid_operand_is_a_null_row():
    bowtie = "POLYGON((0 0, 4 4, 4 0, 0 4, 0 0))"
    ds = bt.from_pydict(
        {"a": [_SQ, _SQ, _SQ, None], "b": [_SQ, "LINESTRING(0 0, 1 1)", bowtie, _SQ]}
    )
    got = ds.select(v=bt.st_area(bt.st_union(col("a"), col("b")))).to_pydict()["v"]
    assert got == [16.0, None, None, None]


def test_the_srid_follows_the_first_operand():
    ds = bt.from_pydict({"a": [f"SRID=3857;{_SQ}"], "b": ["POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))"]})
    got = ds.select(s=bt.st_srid(bt.st_intersection(col("a"), col("b")))).to_pydict()
    assert got == {"s": [3857]}


def test_an_empty_input_and_streaming_agree_with_collect():
    schema = pa.schema([("a", pa.string()), ("b", pa.string())])
    empty = bt.from_pydict({"a": [], "b": []}, schema=schema)
    assert empty.select(v=bt.st_union(col("a"), col("b"))).to_pydict() == {"v": []}
    rows = PAIRS * 20
    ds = bt.from_pydict({"a": [a for a, _ in rows], "b": [b for _, b in rows]})
    q = ds.select(v=bt.st_area(bt.st_difference(col("a"), col("b"))))
    streamed = [v for b in q.iter_batches(batch_size=37) for v in b.column("v").to_pylist()]
    assert streamed == q.to_pydict()["v"]


def test_operands_in_different_reference_systems_are_refused():
    """PostGIS raises on mixed SRIDs; comparing Web Mercator metres to degrees is noise."""
    ds = bt.from_pydict({"a": ["SRID=4326;POINT(1 1)"], "b": ["SRID=3857;POINT(1 1)"]})
    for fn in (bt.st_intersects, bt.st_distance, bt.st_intersection):
        with pytest.raises(bt.ExecutionError, match="different reference systems"):
            ds.select(v=fn(col("a"), col("b"))).collect()
    # An unknown SRID (0, what a bare WKT literal carries) combines with anything.
    got = ds.select(v=bt.st_intersects(col("a"), "POINT(1 1)")).to_pydict()
    assert got == {"v": [True]}
