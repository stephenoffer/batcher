"""The geospatial and rigid-body functions on the distributed path, against single-node.

Every function in these two families is a row-wise expression: it reads one row's columns
and writes one value, with no state across rows. So it distributes with any query the way
`+` does, and the distributed answer must be the single-node answer row for row. What this
file adds over the per-function differential tests is the *combination* that has broken
things before: many such expressions in one projection, a filter on one of them, and a
`group_by` keyed on another, over a source that genuinely splits across workers — with
the bad rows (NaN, off the globe, degenerate shapes) that now null row by row mixed in, so
a worker that raised instead of nulling would fail the whole distributed query.

The comparison names `distributed=False` on one side and `distributed=True, num_workers=4`
on the other, and carries the control `.claude/rules/testing.md` asks for: a `LIMIT` over
an unordered `group_by`, which is *known* to pick different groups once more than one
worker is involved. If that control agrees, the run was one worker in disguise and the
equality below would prove nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import batcher as bt
from _ray_cluster import init_test_ray, shutdown_test_ray

pytestmark = pytest.mark.integration

pytest.importorskip("ray", reason="ray not installed")

_ROWS = 80_000
_FILES = 4
_WORKERS = 4


@pytest.fixture(scope="module", autouse=True)
def _ray_session():
    started = init_test_ray(_WORKERS)
    yield
    shutdown_test_ray(started)


@pytest.fixture(scope="module")
def tracks(cluster_scratch) -> str:
    """80,000 positions, poses and small shapes over four Parquet files.

    One row in every 997 has a NaN longitude and one in every 1,009 a longitude of 200:
    rows the grid and geodesic functions used to fail the whole column on.
    """
    rng = np.random.default_rng(20260922)
    lon = rng.uniform(-10.0, 30.0, _ROWS)
    lat = rng.uniform(35.0, 60.0, _ROWS)
    ids = np.arange(_ROWS, dtype="int64")
    lon[ids % 997 == 5] = np.nan
    lon[ids % 1009 == 7] = 200.0
    half = rng.uniform(0.001, 0.05, _ROWS)
    squares = [
        f"POLYGON(({x - h} {y - h}, {x + h} {y - h}, {x + h} {y + h}, {x - h} {y + h}, "
        f"{x - h} {y - h}))"
        if math.isfinite(x) and x <= 180.0
        else "POLYGON((0 0, 0 0, 0 0, 0 0))"
        for x, y, h in zip(lon, lat, half, strict=True)
    ]
    tracks_wkt = [
        f"LINESTRING({x} {y}, {x + 0.1} {y + 0.05}, {x + 0.2} {y})"
        if math.isfinite(x) and x <= 180.0
        else "LINESTRING EMPTY"
        for x, y in zip(lon, lat, strict=True)
    ]
    yaw = rng.uniform(-math.pi, math.pi, _ROWS)
    table = pa.table(
        {
            "id": ids,
            "lon": lon,
            "lat": lat,
            "shape": squares,
            "track": tracks_wkt,
            "qz": np.sin(yaw / 2),
            "qw": np.cos(yaw / 2),
            "px": rng.uniform(-50, 50, _ROWS),
            "py": rng.uniform(-50, 50, _ROWS),
            "pz": rng.uniform(-2, 5, _ROWS),
        }
    )
    directory = cluster_scratch("geo_distributed")
    per = _ROWS // _FILES
    for k in range(_FILES):
        pq.write_table(
            table.slice(k * per, per), directory / f"part{k}.parquet", row_group_size=5_000
        )
    return str(directory)


def _projection(ds: bt.Dataset) -> bt.Dataset:
    """Fifteen-odd geospatial and rigid-body functions in one projection."""
    lon, lat = bt.col("lon"), bt.col("lat")
    shape, track = bt.col("shape"), bt.col("track")
    here = bt.st_point(lon, lat)
    zero = bt.lit(0.0)
    rotation = (zero, zero, bt.col("qz"), bt.col("qw"))
    point = ("px", "py", "pz")
    return ds.select(
        "id",
        gh=bt.geohash_encode(lon, lat, 3),
        s2=bt.st_s2_cell(lon, lat, 10),
        quadkey=bt.st_quadkey(lon, lat, 9),
        tile_x=bt.st_tile_x(lon, lat, 9),
        utm=bt.st_utm_epsg(lon, lat),
        area_m2=bt.st_area_spheroid(shape),
        length_m=bt.st_length_spheroid(track),
        to_paris=bt.st_distance_spheroid(here, "POINT(2.3522 48.8566)"),
        to_paris_sphere=bt.st_distance_sphere(here, "POINT(2.3522 48.8566)"),
        buffered=bt.st_area(bt.st_buffer(track, 0.01, 4)),
        centroid_x=bt.st_x(bt.st_centroid(shape)),
        valid=bt.st_is_valid(shape),
        parts=bt.st_num_geometries(shape),
        along=bt.st_as_text(bt.st_line_interpolate_point(track, 0.5)),
        rx=bt.quat_rotate_x(*rotation, *point),
        **bt.voxel_index(point, 2.5),
    )


def _by_id(table: pa.Table) -> dict[str, list]:
    return table.sort_by("id").to_pydict()


def _same_value(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


@pytest.mark.integration
def test_a_row_wise_projection_is_identical_row_for_row(tracks):
    """Row-wise expressions have nothing to reassociate, so this is exact, floats too."""
    ds = _projection(bt.read.parquet(tracks))
    single = _by_id(ds.collect(distributed=False))
    dist = _by_id(ds.collect(distributed=True, num_workers=_WORKERS))
    assert len(single["id"]) == _ROWS
    assert single.keys() == dist.keys()
    for name in single:
        mismatches = [
            i
            for i, (a, b) in enumerate(zip(single[name], dist[name], strict=True))
            if not _same_value(a, b)
        ]
        assert not mismatches, f"{name}: first differing row {mismatches[:3]}"
    # The bad rows nulled themselves and nothing else.
    assert sum(g is None for g in single["gh"]) == sum(
        1 for i in range(_ROWS) if i % 997 == 5 or i % 1009 == 7
    )
    assert all(v is not None for v in single["area_m2"])


@pytest.mark.integration
def test_a_filtered_group_by_on_a_geohash_agrees(tracks):
    ds = (
        _projection(bt.read.parquet(tracks))
        .filter(bt.col("valid") & (bt.col("area_m2") > 1.0e6))
        .group_by("gh")
        .agg(
            n=bt.col("id").count(),
            area=bt.col("area_m2").sum(),
            longest=bt.col("length_m").max(),
            nearest=bt.col("to_paris").min(),
            cells=bt.col("s2").count_distinct(),
        )
    )
    single = ds.collect(distributed=False).sort_by("gh").to_pydict()
    dist = ds.collect(distributed=True, num_workers=_WORKERS).sort_by("gh").to_pydict()
    assert single["gh"] == dist["gh"]
    assert len(single["gh"]) > 20, "the fixture should spread over many geohash cells"
    assert single["n"] == dist["n"]
    assert single["cells"] == dist["cells"]
    assert single["longest"] == dist["longest"]
    assert single["nearest"] == dist["nearest"]
    # A float sum may reassociate across partitions; it may not move past the last bits.
    assert dist["area"] == pytest.approx(single["area"], rel=1e-12)


@pytest.mark.integration
def test_the_control_proves_more_than_one_worker_ran(tracks):
    """`LIMIT` over an unordered group_by keeps *some* groups; with several workers it
    keeps different ones than a single node does. Agreement here would mean the
    comparisons above ran on one worker and proved nothing."""
    base = _projection(bt.read.parquet(tracks)).group_by("gh").agg(n=bt.col("id").count())
    everything = {row["gh"] for row in base.collect(distributed=False).to_pylist()}
    differ = False
    for limit in (3, 5, 8):
        ds = base.limit(limit)
        single = {row["gh"] for row in ds.collect(distributed=False).to_pylist()}
        dist = {row["gh"] for row in ds.collect(distributed=True, num_workers=_WORKERS).to_pylist()}
        assert len(single) == len(dist) == limit
        assert dist <= everything, "a limit selects rows of the full result, never invents"
        differ = differ or single != dist
    assert differ, "every unordered LIMIT agreed: the distributed run was not distributed"
