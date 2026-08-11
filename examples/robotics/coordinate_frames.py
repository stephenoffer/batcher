"""Moving a sensor measurement between coordinate frames.

The one thing every robotics log needs and no relational engine gives you for free. The
lidar reports in the lidar's frame, the localizer reports where the vehicle was in the
world, and the calibration says where the lidar is bolted relative to the vehicle. Answer
almost any question and you have moved a point between two of those.

    python examples/robotics/coordinate_frames.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt


def main() -> None:
    # A vehicle at (100, 50) in the world, heading 90 degrees round from the X axis.
    heading = math.pi / 2
    world_from_ego = bt.from_pydict(
        {
            "etx": [100.0],
            "ety": [50.0],
            "etz": [0.0],
            # A yaw-only rotation: sin/cos of half the heading, about Z.
            "eqx": [0.0],
            "eqy": [0.0],
            "eqz": [math.sin(heading / 2)],
            "eqw": [math.cos(heading / 2)],
            # The lidar sits 1.2 m forward and 1.8 m up, mounted square.
            "ltx": [1.2],
            "lty": [0.0],
            "ltz": [1.8],
            "lqx": [0.0],
            "lqy": [0.0],
            "lqz": [0.0],
            "lqw": [1.0],
        }
    )

    ego = ("etx", "ety", "etz", "eqx", "eqy", "eqz", "eqw")
    lidar = ("ltx", "lty", "ltz", "lqx", "lqy", "lqz", "lqw")

    # Collapse the chain once per frame. Applying each pose to every point instead
    # would do the same work N times, where N is the number of returns in the sweep.
    chain = world_from_ego.with_columns(**bt.se3_compose(ego, lidar, prefix="w_"))
    posed = chain.select("w_tx", "w_ty", "w_tz", "w_qx", "w_qy", "w_qz", "w_qw").to_pydict()
    print("world_from_lidar translation:")
    print(f"  ({posed['w_tx'][0]:.2f}, {posed['w_ty'][0]:.2f}, {posed['w_tz'][0]:.2f})")

    # Facing +Y, so the sensor's forward offset of 1.2 m moves the mount in +Y.
    assert posed["w_tx"][0] == 100.0
    assert abs(posed["w_ty"][0] - 51.2) < 1e-9
    assert abs(posed["w_tz"][0] - 1.8) < 1e-9

    # Now the sweep itself: four returns, in the lidar's own coordinates.
    sweep = bt.from_pydict(
        {
            "x": [10.0, 0.0, -3.0, 45.0],
            "y": [0.0, 8.0, 0.0, -2.0],
            "z": [0.0, 0.0, 0.5, 1.0],
            "intensity": [180, 24, 200, 9],
        }
    )
    # The pose is one row; broadcasting it across the sweep is a cross join.
    world_pose = ("w_tx", "w_ty", "w_tz", "w_qx", "w_qy", "w_qz", "w_qw")
    in_world = sweep.cross_join(chain.select(*world_pose)).with_columns(
        **bt.se3_transform(world_pose, ("x", "y", "z"), prefix="world_")
    )

    got = in_world.select("world_x", "world_y", "world_z").to_pydict()
    print("\nreturns, in world coordinates:")
    for wx, wy, wz in zip(got["world_x"], got["world_y"], got["world_z"], strict=True):
        print(f"  ({wx:8.2f}, {wy:8.2f}, {wz:6.2f})")

    # The first return is 10 m straight ahead of a sensor facing +Y, so it lands
    # 10 m further along Y than the mount did.
    assert abs(got["world_x"][0] - 100.0) < 1e-9
    assert abs(got["world_y"][0] - 61.2) < 1e-9

    # And the inverse puts it back where it started, which is the check worth making
    # whenever a transform chain is written for the first time.
    back = (
        in_world.with_columns(
            **bt.se3_inverse_transform(
                world_pose, ("world_x", "world_y", "world_z"), prefix="back_"
            )
        )
        .select("x", "y", "z", "back_x", "back_y", "back_z")
        .to_pydict()
    )
    for axis in ("x", "y", "z"):
        for before, after in zip(back[axis], back[f"back_{axis}"], strict=True):
            assert abs(before - after) < 1e-9, (axis, before, after)
    print("\nround trip through the inverse pose returns the original coordinates")


if __name__ == "__main__":
    main()
