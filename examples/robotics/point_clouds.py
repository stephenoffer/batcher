"""Putting a lidar sweep in world coordinates, then cutting it down.

Batcher reads a point cloud as one row per point, with `x`/`y`/`z` as ordinary columns.
That is what makes the whole of this file native relational work: the transform is a
projection, the range and height cuts are a filter, and the sector histogram is a
`group_by`. None of it materializes the sweep.

    python examples/robotics/point_clouds.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    # A sweep in the lidar's own frame. In a real pipeline this is `bt.read.point_cloud`
    # over a directory of `.bin`/`.pcd`/`.ply` files.
    sweep = bt.from_pydict(
        {
            "x": [10.0, 40.0, 0.4, -12.0, 25.0, 1.0],
            "y": [0.0, 5.0, 0.1, -3.0, -25.0, 0.0],
            "z": [0.2, 0.5, 0.0, 1.5, 0.8, -1.75],
            "intensity": [180, 96, 250, 40, 12, 210],
        }
    )
    point = ("x", "y", "z")

    # --- 1. Where each return is, in the sensor's own spherical coordinates -----
    # Range is the cheapest and most effective filter a point cloud has. Azimuth splits
    # the sweep into the sectors a rotating sensor actually measures in. Elevation
    # separates ground returns from vehicles and from overhead structure.
    described = sweep.with_columns(
        rng=bt.norm_3d(point),
        az=bt.azimuth_3d(point),
        el=bt.elevation_3d(point),
    )
    rows = described.select("rng", "az", "el").to_pydict()
    assert abs(rows["az"][0]) < 1e-12  # straight ahead
    assert abs(rows["az"][4] - math.atan2(-25.0, 25.0)) < 1e-12
    print("range / azimuth / elevation of the first three returns:")
    for i in range(3):
        print(f"  {rows['rng'][i]:7.2f} m  {rows['az'][i]:7.3f} rad  {rows['el'][i]:7.3f} rad")

    # --- 2. The usual first cuts ------------------------------------------------
    # Drop the vehicle itself, drop anything past the sensor's rated range, drop the
    # ground given a sensor mounted 1.8 m up.
    useful = described.filter((col("rng") > 2.0) & (col("rng") < 60.0) & (col("z") > -1.7))
    kept = useful.to_pydict()
    assert 0.4 not in kept["x"], "the near return is the vehicle"
    assert -1.75 not in kept["z"], "that return is the ground"
    print(f"\n{len(kept['x'])} of {sweep.count()} returns survive the first cuts")

    # --- 3. Into world coordinates ----------------------------------------------
    # The vehicle is at (100, 50) facing 90 degrees round from the X axis, and the lidar
    # is 1.2 m forward and 1.8 m up on it. `se3_compose` collapses that chain once.
    heading = math.pi / 2
    frames = bt.from_pydict(
        {
            "etx": [100.0], "ety": [50.0], "etz": [0.0],
            "eqx": [0.0], "eqy": [0.0],
            "eqz": [math.sin(heading / 2)], "eqw": [math.cos(heading / 2)],
            "ltx": [1.2], "lty": [0.0], "ltz": [1.8],
            "lqx": [0.0], "lqy": [0.0], "lqz": [0.0], "lqw": [1.0],
        }
    )  # fmt: skip
    world_from_ego = ("etx", "ety", "etz", "eqx", "eqy", "eqz", "eqw")
    ego_from_lidar = ("ltx", "lty", "ltz", "lqx", "lqy", "lqz", "lqw")
    posed = frames.with_columns(**bt.se3_compose(world_from_ego, ego_from_lidar, prefix="w_"))
    world_from_lidar = ("w_tx", "w_ty", "w_tz", "w_qx", "w_qy", "w_qz", "w_qw")

    # The pose is one row, so broadcasting it across the sweep is a cross join.
    located = useful.cross_join(posed.select(*world_from_lidar))
    in_world = located.select(
        "intensity",
        "x",
        "y",
        "z",
        # The component functions rather than the `se3_transform` helper, to show what
        # the helper is doing and because naming the outputs here is clearer.
        wx=bt.se3_transform_x(*world_from_lidar, "x", "y", "z"),
        wy=bt.se3_transform_y(*world_from_lidar, "x", "y", "z"),
        wz=bt.se3_transform_z(*world_from_lidar, "x", "y", "z"),
    )
    # Name the row rather than taking whichever comes back first: a filter does not
    # promise to preserve input order, and asserting on position instead of on identity
    # is how an order bug hides.
    ahead = in_world.filter(col("x") == 10.0).to_pydict()
    print(
        "\nthe return 10 m straight ahead, in world coordinates: "
        f"({ahead['wx'][0]:.2f}, {ahead['wy'][0]:.2f}, {ahead['wz'][0]:.2f})"
    )
    # Facing +Y, it lands 10 m further along Y than the mount and no further along X.
    assert abs(ahead["wx"][0] - 100.0) < 1e-9
    assert abs(ahead["wy"][0] - 61.2) < 1e-9

    # --- 4. And back again, which is the check worth making ---------------------
    # `se3_inverse` turns the pose around; `se3_inverse_transform_*` applies it directly.
    # Both must return the coordinates the sweep started with.
    round_tripped = in_world.cross_join(posed.select(*world_from_lidar)).select(
        "x",
        "y",
        "z",
        bx=bt.se3_inverse_transform_x(*world_from_lidar, "wx", "wy", "wz"),
        by=bt.se3_inverse_transform_y(*world_from_lidar, "wx", "wy", "wz"),
        bz=bt.se3_inverse_transform_z(*world_from_lidar, "wx", "wy", "wz"),
    )
    check = round_tripped.to_pydict()
    for axis in "xyz":
        for before, after in zip(check[axis], check[f"b{axis}"], strict=True):
            assert abs(before - after) < 1e-9, (axis, before, after)

    via_inverse = posed.with_columns(**bt.se3_inverse(world_from_lidar, prefix="i"))
    inverse_pose = ("itx", "ity", "itz", "iqx", "iqy", "iqz", "iqw")
    same = (
        in_world.cross_join(via_inverse.select(*inverse_pose))
        .select("x", **bt.se3_transform(inverse_pose, ("wx", "wy", "wz"), prefix="r"))
        .to_pydict()
    )
    for before, after in zip(same["x"], same["rx"], strict=True):
        assert abs(before - after) < 1e-9, (before, after)
    print("every return round trips back to its sensor-frame coordinates")

    # --- 5. How far apart are two returns? ---------------------------------------
    # Both points must already be in the same frame; nothing can detect it if they are
    # not, which is why this is worth saying out loud in a pipeline.
    origin = in_world.filter(col("x") == 10.0).select(ax=col("wx"), ay=col("wy"), az=col("wz"))
    spread = (
        in_world.cross_join(origin)
        .select(d=bt.distance_3d(("ax", "ay", "az"), ("wx", "wy", "wz")))
        .to_pydict()["d"]
    )
    # The reference point is in the set, so exactly one distance is zero.
    assert min(spread) < 1e-9
    print(f"furthest return from that one: {max(spread):.2f} m")

    # --- 6. Voxel downsampling, which is also an ordinary group_by ----------------
    # A sweep is far denser near the sensor than far from it. Binning into cubes and
    # keeping one point per occupied cube makes the density uniform and the cloud much
    # smaller. Because it is a `group_by`, it spills and distributes for free.
    thinned = (
        in_world.group_by(**bt.voxel_index(("wx", "wy", "wz"), 0.5))
        .agg(
            n=bt.count(),
            x=col("wx").mean(),
            y=col("wy").mean(),
            z=col("wz").mean(),
            intensity=col("intensity").mean(),
        )
        .to_pydict()
    )
    print(f"\n{sum(thinned['n'])} returns reduce to {len(thinned['n'])} voxels at 0.5 m")
    assert sum(thinned["n"]) == 4
    assert len(thinned["n"]) <= 4

    # --- 6. A sector histogram, which is an ordinary group_by ---------------------
    sectors = (
        described.filter(col("rng") > 2.0)
        .with_columns(sector=(bt.azimuth_3d(point) / (math.pi / 4)).floor())
        .group_by("sector")
        .agg(n=bt.count(), mean_range=col("rng").mean())
        .sort("sector")
        .to_pydict()
    )
    print("\nreturns per 45-degree sector:")
    for sector, n in zip(sectors["sector"], sectors["n"], strict=True):
        print(f"  sector {int(sector):>3}: {n}")
    assert sum(sectors["n"]) == 5


if __name__ == "__main__":
    main()
