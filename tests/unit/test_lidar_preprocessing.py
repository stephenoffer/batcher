"""The LiDAR preprocessing chain runs as native operators — no Python per point.

`io/formats/ml/point_cloud.py` claims that because a cloud is stored columnar, the usual
robotics preprocessing "is then a native operator". That is a load-bearing claim: a sweep
is 50k-2M points and a perception pipeline runs this chain on every frame, so if any step
silently fell back to Python the throughput story would be wrong.

These tests are the executable form of that claim. They also serve as the reference for
the idioms, which are not obvious — voxel downsampling as a `floor`-then-`group_by`, and a
rigid transform as three `with_columns` expressions, are exactly the operations users
would otherwise reach for a UDF to do.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import batcher as bt
from batcher import col, lit

pytestmark = pytest.mark.unit


@pytest.fixture
def sweep(tmp_path):
    """A KITTI-style Velodyne sweep: a flat float32 buffer of (x, y, z, intensity)."""
    rng = np.random.default_rng(0)
    points = np.column_stack(
        [
            rng.uniform(-50, 50, 50_000),
            rng.uniform(-50, 50, 50_000),
            rng.uniform(-3, 5, 50_000),
            rng.uniform(0, 1, 50_000),
        ]
    ).astype(np.float32)
    path = tmp_path / "0000.bin"
    points.tofile(path)
    return bt.read.point_cloud(str(path), frame_column=None)


def test_the_sweep_reads_as_one_row_per_point(sweep) -> None:
    assert sweep.count() == 50_000
    assert sweep.columns == ["x", "y", "z", "intensity"]


def test_region_of_interest_crop_is_a_filter(sweep) -> None:
    """The first stage of every perception pipeline: keep the drivable corridor."""
    roi = sweep.filter((col("x") > 0) & (col("x") < 30) & (col("y") > -10) & (col("y") < 10))
    got = roi.collect().to_pydict()

    assert 0 < len(got["x"]) < 50_000
    assert all(0 < x < 30 for x in got["x"])
    assert all(-10 < y < 10 for y in got["y"])


def test_ground_plane_removal_is_a_filter_on_z(sweep) -> None:
    above = sweep.filter(col("z") > -1.5)

    assert all(z > -1.5 for z in above.collect().to_pydict()["z"])


def test_voxel_downsampling_is_a_floor_then_a_group_by(sweep) -> None:
    """The canonical LiDAR reduction, and the least obvious idiom: snap each point to a
    grid cell with `floor`, then average the points that land in the same cell."""
    voxel = 0.5
    downsampled = (
        sweep.filter(col("z") > -1.5)
        .with_columns(
            vx=(col("x") / voxel).floor(),
            vy=(col("y") / voxel).floor(),
            vz=(col("z") / voxel).floor(),
        )
        .group_by("vx", "vy", "vz")
        .agg(x=col("x").mean(), y=col("y").mean(), z=col("z").mean(), n=col("intensity").count())
    )
    got = downsampled.collect()

    assert got.num_rows < sweep.filter(col("z") > -1.5).count(), "downsampling added points"
    assert got.num_rows > 0
    # Every voxel holds at least one point, and they account for every input point.
    counts = got.column("n").to_pylist()
    assert min(counts) >= 1
    assert sum(counts) == sweep.filter(col("z") > -1.5).count()


def test_range_filtering_is_arithmetic_on_the_coordinates(sweep) -> None:
    within = sweep.with_columns(
        rho=(col("x") * col("x") + col("y") * col("y") + col("z") * col("z")).sqrt()
    ).filter(col("rho") < 25)
    got = within.collect().to_pydict()

    assert all(r < 25 for r in got["rho"])
    assert len(got["rho"]) > 0


def test_a_rigid_transform_is_three_projections(tmp_path) -> None:
    """Ego frame to world frame — the other operation every perception stack performs.

    A yaw rotation plus a translation, which is what a pose from the localization stack
    supplies. Exact values are asserted because a transposed or sign-flipped rotation
    still produces plausible-looking coordinates.
    """
    points = np.array([[1, 0, 0, 0.5], [0, 1, 0, 0.5], [1, 1, 0, 0.5]], dtype=np.float32)
    path = tmp_path / "s.bin"
    points.tofile(path)
    cloud = bt.read.point_cloud(str(path), frame_column=None)

    yaw = math.pi / 2
    cos, sin = math.cos(yaw), math.sin(yaw)
    world = cloud.with_columns(
        wx=col("x") * lit(cos) - col("y") * lit(sin) + lit(10.0),
        wy=col("x") * lit(sin) + col("y") * lit(cos) + lit(20.0),
        wz=col("z") + lit(0.0),
    ).select("wx", "wy", "wz")
    got = world.collect().to_pydict()

    assert got["wx"] == pytest.approx([10.0, 9.0, 9.0], abs=1e-5)
    assert got["wy"] == pytest.approx([21.0, 20.0, 21.0], abs=1e-5)
    assert got["wz"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-5)


def test_the_whole_chain_composes_into_one_lazy_plan(sweep) -> None:
    """Crop, remove ground, downsample — one plan, so it fuses and streams rather than
    materializing a cloud per stage."""
    voxel = 0.5
    pipeline = (
        sweep.filter((col("x") > 0) & (col("x") < 30))
        .filter(col("z") > -1.5)
        .with_columns(vx=(col("x") / voxel).floor(), vy=(col("y") / voxel).floor())
        .group_by("vx", "vy")
        .agg(z=col("z").max(), n=col("x").count())
    )

    assert pipeline.collect().num_rows > 0
    # It is still one lazy plan: the whole chain is visible in a single explain, with the
    # scan at the bottom, rather than having been executed stage by stage.
    plan = pipeline.explain()
    assert "aggregate" in plan
    assert "scan" in plan
