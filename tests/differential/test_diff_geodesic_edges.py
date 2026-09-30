"""Geodesic distance to a shape's *edges*, against a densified PROJ oracle.

`st_distance_sphere`, `st_distance_spheroid` and `st_dwithin_sphere` measure between the
nearest points of two shapes, taking each edge as the shorter great-circle arc between its
positions. They used to measure vertex to vertex only, which put a point one degree beside
the middle of a ten-degree edge 567 km away instead of 111 km, and made `st_dwithin_sphere`
miss it at any radius in between.

DuckDB's spheroid and sphere distances accept only points, so the oracle here is built
independently: every edge is densified along its great circle in NumPy (spherical linear
interpolation of unit vectors, 20,001 samples) and the distance is the minimum over the
samples, measured with PROJ's geodesic inverse (``pyproj.Geod``, GeographicLib's Karney
algorithm) for the ellipsoid and a NumPy haversine for the sphere. Near a minimum the
sampling error is quadratic in the sample spacing, well under a millimetre at these sizes.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pyarrow as pa
import pytest

import batcher as bt
from batcher import col

pytestmark = pytest.mark.differential

pyproj = pytest.importorskip("pyproj")

_GEOD = pyproj.Geod(ellps="WGS84")
_RADIUS = 6_371_008.8
_SAMPLES = 20_001


def _unit(lon: float, lat: float) -> np.ndarray:
    lo, la = math.radians(lon), math.radians(lat)
    return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])


def _densify(a: tuple[float, float], b: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Longitudes and latitudes along the shorter great-circle arc from `a` to `b`."""
    ua, ub = _unit(*a), _unit(*b)
    omega = math.atan2(np.linalg.norm(np.cross(ua, ub)), float(ua @ ub))
    t = np.linspace(0.0, 1.0, _SAMPLES)[:, None]
    pts = (np.sin((1 - t) * omega) * ua + np.sin(t * omega) * ub) / math.sin(omega)
    lon = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    lat = np.degrees(np.arctan2(pts[:, 2], np.hypot(pts[:, 0], pts[:, 1])))
    return lon, lat


def _haversine(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - lon1)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * _RADIUS * np.arcsin(np.sqrt(h))


def _oracle(point: tuple[float, float], chain: list[tuple[float, float]], spheroid: bool) -> float:
    """The distance from `point` to the nearest point of `chain`'s great-circle edges."""
    best = math.inf
    for a, b in itertools.pairwise(chain):
        lon, lat = _densify(a, b)
        if spheroid:
            _, _, d = _GEOD.inv(np.full_like(lon, point[0]), np.full_like(lat, point[1]), lon, lat)
        else:
            d = _haversine(point[0], point[1], lon, lat)
        best = min(best, float(np.min(d)))
    return best


def _vertex(point: tuple[float, float], v: tuple[float, float], spheroid: bool) -> float:
    if spheroid:
        return float(_GEOD.inv(point[0], point[1], v[0], v[1])[2])
    return float(_haversine(point[0], point[1], v[0], v[1]))


def _wkt_line(chain: list[tuple[float, float]]) -> str:
    return "LINESTRING(" + ", ".join(f"{x} {y}" for x, y in chain) + ")"


#: A point beside a chain, chosen so the nearest point is inside an edge rather than at a
#: vertex: mid-latitude, near a pole, across the antimeridian, and a long diagonal edge.
CASES = [
    ((5.0, 1.0), [(0.0, 0.0), (10.0, 0.0)]),
    ((-73.0, 42.0), [(-80.0, 40.0), (-60.0, 40.5), (-50.0, 30.0)]),
    ((30.0, 84.0), [(0.0, 85.0), (60.0, 85.0)]),
    ((180.0, 2.0), [(175.0, 0.0), (-175.0, 0.0)]),
    ((12.0, -33.0), [(0.0, -40.0), (30.0, -20.0)]),
]


@pytest.mark.parametrize(("point", "chain"), CASES)
@pytest.mark.parametrize("spheroid", [False, True])
def test_a_point_measures_to_the_nearest_point_of_an_edge(point, chain, spheroid):
    ds = bt.from_pydict({"p": [f"POINT({point[0]} {point[1]})"], "l": [_wkt_line(chain)]})
    fn = bt.st_distance_spheroid if spheroid else bt.st_distance_sphere
    got = ds.select(d=fn(col("p"), col("l"))).to_pydict()["d"][0]
    want = _oracle(point, chain, spheroid)
    assert got == pytest.approx(want, abs=0.01), (point, chain)
    # And the nearest point really is inside an edge, not a vertex, so this is the case
    # the old vertex-to-vertex answer got wrong.
    vertex = min(_vertex(point, v, spheroid) for v in chain)
    assert vertex > want * 1.01


def test_two_disjoint_chains_measure_between_their_edges():
    # Two meridian segments one degree apart with no vertex facing a vertex: the nearest
    # pair is an end of the short one against the interior of the long one.
    a, b = [(0.0, -3.0), (0.0, 3.0)], [(1.0, -1.0), (1.0, 1.0)]
    ds = bt.from_pydict({"a": [_wkt_line(a)], "b": [_wkt_line(b)]})
    got = ds.select(d=bt.st_distance_spheroid(col("a"), col("b"))).to_pydict()["d"][0]
    want = min(_oracle(v, a, True) for v in b)
    assert got == pytest.approx(want, abs=0.01)


def test_a_polygon_measures_to_its_boundary_and_is_zero_inside():
    square = "POLYGON((0 -5, 10 -5, 10 5, 0 5, 0 -5))"
    ds = bt.from_pydict({"p": ["POINT(10.5 0)", "POINT(5 0)"], "g": [square, square]})
    got = ds.select(d=bt.st_distance_spheroid(col("p"), col("g"))).to_pydict()["d"]
    ring = [(0.0, -5.0), (10.0, -5.0), (10.0, 5.0), (0.0, 5.0), (0.0, -5.0)]
    assert got[0] == pytest.approx(_oracle((10.5, 0.0), ring, True), abs=0.01)
    assert got[1] == 0.0


def test_crossing_edges_are_zero_apart():
    ds = bt.from_pydict({"a": ["LINESTRING(-1 5, 11 5)"], "b": ["LINESTRING(5 -1, 5 11)"]})
    got = ds.select(d=bt.st_distance_sphere(col("a"), col("b"))).to_pydict()["d"]
    assert got == [0.0]


def test_dwithin_sphere_now_sees_a_point_beside_an_edge():
    # 111 km from the edge, 567 km from its nearest vertex: inside a 200 km radius.
    ds = bt.from_pydict({"p": ["POINT(5 1)"], "l": ["LINESTRING(0 0, 10 0)"]})
    got = ds.select(v=bt.st_dwithin_sphere(col("p"), col("l"), 200_000.0)).to_pydict()
    assert got == {"v": [True]}


def test_points_nulls_and_empty_input_are_unchanged():
    ds = bt.from_pydict(
        {
            "a": ["POINT(0 0)", None, "POINT(0 0)"],
            "b": ["POINT(1 1)", "POINT(1 1)", "LINESTRING EMPTY"],
        }
    )
    got = ds.select(d=bt.st_distance_spheroid(col("a"), col("b"))).to_pydict()["d"]
    assert got[0] == pytest.approx(float(_GEOD.inv(0.0, 0.0, 1.0, 1.0)[2]), rel=1e-9)
    assert got[1:] == [None, None]
    empty = bt.from_pydict(
        {"a": [], "b": []}, schema=pa.schema([("a", pa.string()), ("b", pa.string())])
    )
    assert empty.select(d=bt.st_distance_sphere(col("a"), col("b"))).to_pydict() == {"d": []}


def test_streaming_agrees_with_collect():
    rows = [(f"POINT({p[0]} {p[1]})", _wkt_line(c)) for p, c in CASES] * 50
    ds = bt.from_pydict({"p": [r[0] for r in rows], "l": [r[1] for r in rows]})
    q = ds.select(d=bt.st_distance_spheroid(col("p"), col("l")))
    streamed = [v for b in q.iter_batches(batch_size=7) for v in b.column("d").to_pylist()]
    assert streamed == q.to_pydict()["d"]
