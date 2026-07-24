"""`str.regexp_split` against DuckDB, and `great_circle_distance` against known geodesics.

Two unrelated gaps closed in one change, tested together because neither needs a file of
its own.

`regexp_split` has a direct DuckDB counterpart (`regexp_split_to_array`), so it is compared
against it. `great_circle_distance` has none — DuckDB ships no haversine — so it is checked
against an independent implementation of the formula and against distances whose values are
fixed by geometry rather than by measurement (a point to itself, pole to pole, a quarter of
the equator).
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
from batcher import col
from batcher._internal.errors import PlanError

SPLIT_INPUTS = ["a1b22c", "no digits here", "", "9leading", "trailing7", None]


@pytest.mark.parametrize("pattern", ["[0-9]+", r"\s+", "[,;]", "x"])
def test_regexp_split_matches_duckdb(duck, pattern):
    t = pa.table({"s": SPLIT_INPUTS})
    duck.register("s", t)
    out = bt.from_arrow(t).select(r=col("s").str.regexp_split(pattern)).collect()
    assert_same(out, duck.sql(f"SELECT regexp_split_to_array(s, '{pattern}') r FROM s"))


def test_regexp_split_keeps_the_empty_pieces_a_separator_creates():
    # Leading and trailing separators produce empty pieces. Dropping them would be a
    # different function, and would make `len(split)` stop counting separators.
    t = pa.table({"s": ["  a  b "]})
    got = bt.from_arrow(t).select(r=col("s").str.regexp_split(r"\s+")).to_pydict()["r"]
    assert got == [["", "a", "b", ""]]


def test_regexp_split_on_a_literal_agrees_with_split():
    # A pattern with no metacharacters must behave exactly like the literal splitter.
    t = pa.table({"s": ["a-b-c", "", "no dash", None]})
    ds = bt.from_arrow(t)
    literal = ds.select(r=col("s").str.split("-")).to_pydict()["r"]
    regex = ds.select(r=col("s").str.regexp_split("-")).to_pydict()["r"]
    assert literal == regex


def test_regexp_split_feeds_the_list_namespace():
    t = pa.table({"s": ["a, b,c", "d"]})
    got = bt.from_arrow(t).select(n=col("s").str.regexp_split(r",\s*").list.len()).to_pydict()["n"]
    assert got == [3, 1]


# --- great_circle_distance ---------------------------------------------------------

# (name, lat1, lon1, lat2, lon2, expected km, tolerance km)
GEODESICS = [
    # Fixed by geometry, not by measurement: these hold for any sphere radius.
    ("a point to itself", 51.5, -0.13, 51.5, -0.13, 0.0, 1e-9),
    ("pole to pole", 90.0, 0.0, -90.0, 0.0, 20015.087, 0.1),
    ("a quarter of the equator", 0.0, 0.0, 0.0, 90.0, 10007.543, 0.1),
    ("antipodes across the equator", 0.0, 0.0, 0.0, 180.0, 20015.087, 0.1),
    # Published great-circle distances, to the kilometre.
    ("London to Paris", 51.5074, -0.1278, 48.8566, 2.3522, 343.5, 1.0),
    ("New York to Los Angeles", 40.7128, -74.0060, 34.0522, -118.2437, 3936.0, 5.0),
]


@pytest.mark.parametrize(
    ("name", "lat1", "lon1", "lat2", "lon2", "km", "tol"),
    GEODESICS,
    ids=[g[0] for g in GEODESICS],
)
def test_great_circle_distance_matches_known_geodesics(name, lat1, lon1, lat2, lon2, km, tol):
    t = pa.table({"a": [lat1], "b": [lon1], "c": [lat2], "d": [lon2]})
    got = (
        bt.from_arrow(t)
        .select(r=bt.great_circle_distance(col("a"), col("b"), col("c"), col("d")))
        .to_pydict()["r"][0]
    )
    assert abs(got - km) < tol, f"{name}: {got} km, expected {km} +/- {tol}"


def test_great_circle_distance_matches_an_independent_implementation():
    # An independent haversine in Python over a spread of coordinates, including the
    # near-zero separations where the law-of-cosines form loses its precision and this
    # formula is chosen not to.
    pairs = [
        (0.0, 0.0, 0.0, 0.0),
        (51.5074, -0.1278, 48.8566, 2.3522),
        (-33.8688, 151.2093, 35.6762, 139.6503),
        (12.0, 34.0, 12.000001, 34.000001),
        (89.9, 10.0, 89.9, -170.0),
        (-45.0, -179.9, -45.0, 179.9),
    ]
    radius = 6371.0088

    def haversine(lat1, lon1, lat2, lon2):
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    t = pa.table(
        {
            "a": [p[0] for p in pairs],
            "b": [p[1] for p in pairs],
            "c": [p[2] for p in pairs],
            "d": [p[3] for p in pairs],
        }
    )
    got = (
        bt.from_arrow(t)
        .select(r=bt.great_circle_distance(col("a"), col("b"), col("c"), col("d")))
        .to_pydict()["r"]
    )
    for value, pair in zip(got, pairs, strict=True):
        assert value == pytest.approx(haversine(*pair), rel=1e-9, abs=1e-9), pair


@pytest.mark.parametrize(
    ("unit", "per_km"), [("km", 1.0), ("m", 1000.0), ("mi", 1 / 1.609344), ("nm", 1 / 1.852)]
)
def test_units_are_a_pure_rescaling(unit, per_km):
    t = pa.table({"a": [51.5074], "b": [-0.1278], "c": [48.8566], "d": [2.3522]})
    ds = bt.from_arrow(t)
    km = ds.select(r=bt.great_circle_distance(col("a"), col("b"), col("c"), col("d"))).to_pydict()
    other = ds.select(
        r=bt.great_circle_distance(col("a"), col("b"), col("c"), col("d"), unit)
    ).to_pydict()
    assert other["r"][0] == pytest.approx(km["r"][0] * per_km, rel=1e-4)


def test_distance_is_symmetric_and_nulls_propagate():
    t = pa.table({"a": [51.5, None], "b": [-0.1, 0.0], "c": [48.9, 1.0], "d": [2.4, 2.0]})
    ds = bt.from_arrow(t)
    forward = ds.select(r=bt.great_circle_distance(col("a"), col("b"), col("c"), col("d")))
    backward = ds.select(r=bt.great_circle_distance(col("c"), col("d"), col("a"), col("b")))
    f, b = forward.to_pydict()["r"], backward.to_pydict()["r"]
    assert f[0] == pytest.approx(b[0])
    assert f[1] is None and b[1] is None


def test_unknown_unit_fails_at_plan_build():
    with pytest.raises(PlanError, match="unit must be one of"):
        bt.great_circle_distance(col("a"), col("b"), col("c"), col("d"), "furlongs")
