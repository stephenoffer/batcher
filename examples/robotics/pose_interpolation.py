"""Lining up sensors that sample at different rates.

Poses arrive at the localizer's rate and measurements arrive at each sensor's, so a
measurement almost never lands on a logged pose. `join_asof` finds the poses bracketing
each timestamp and `pose_interpolate` finds the pose between them. Interpolating the
quaternion components independently instead is the classic mistake: it sweeps the angle
at a non-constant rate, which shows up as a lidar sweep that visibly bends.

    python examples/robotics/pose_interpolation.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def _yaw_quaternion(angle: float) -> tuple[float, float, float, float]:
    """A rotation of `angle` radians about Z, as (x, y, z, w)."""
    return (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))


def main() -> None:
    # The localizer publishes at 10 Hz: a vehicle driving straight and turning left.
    stamps = [0, 100_000, 200_000, 300_000]
    yaws = [0.0, 0.2, 0.4, 0.6]
    poses = bt.from_pydict(
        {
            "pose_t": stamps,
            "tx": [0.0, 5.0, 10.0, 15.0],
            "ty": [0.0, 0.0, 1.0, 3.0],
            "tz": [0.0, 0.0, 0.0, 0.0],
            "qx": [_yaw_quaternion(y)[0] for y in yaws],
            "qy": [_yaw_quaternion(y)[1] for y in yaws],
            "qz": [_yaw_quaternion(y)[2] for y in yaws],
            "qw": [_yaw_quaternion(y)[3] for y in yaws],
        }
    )

    # The lidar publishes on its own clock, landing between the poses.
    sweeps = bt.from_pydict({"sweep_t": [50_000, 150_000, 250_000]})

    # Backward for the pose at or before each sweep, forward for the one after. Two
    # as-of joins rather than one is what gives you both ends of the bracket.
    #
    # `join_asof` consumes its right-hand key, so carry a *copy* of the timestamp under
    # another name. Without it the bracket's endpoints are gone by the time you need
    # them to compute the interpolation fraction.
    before = sweeps.join_asof(
        poses.with_columns(t0=col("pose_t")),
        left_on="sweep_t",
        right_on="pose_t",
        direction="backward",
    )
    after = sweeps.join_asof(
        poses.select(
            col("pose_t").alias("next_t"),
            col("pose_t").alias("t1"),
            col("tx").alias("ntx"),
            col("ty").alias("nty"),
            col("tz").alias("ntz"),
            col("qx").alias("nqx"),
            col("qy").alias("nqy"),
            col("qz").alias("nqz"),
            col("qw").alias("nqw"),
        ),
        left_on="sweep_t",
        right_on="next_t",
        direction="forward",
    )
    bracketed = before.join(after, on="sweep_t")

    frac = (col("sweep_t") - col("t0")) / (col("t1") - col("t0"))
    at_sweep = bracketed.with_columns(
        **bt.pose_interpolate(
            ("tx", "ty", "tz", "qx", "qy", "qz", "qw"),
            ("ntx", "nty", "ntz", "nqx", "nqy", "nqz", "nqw"),
            frac,
            prefix="at_",
        )
    )

    # Read the interpolated rotation back as a heading, which is what a person checks.
    described = at_sweep.select(
        "sweep_t",
        "at_tx",
        "at_ty",
        heading=bt.quat_to_yaw("at_qx", "at_qy", "at_qz", "at_qw"),
    ).sort("sweep_t")

    got = described.to_pydict()
    print("sweep_t      x       y    heading (rad)")
    for t, x, y, h in zip(got["sweep_t"], got["at_tx"], got["at_ty"], got["heading"], strict=True):
        print(f"{t:>8}  {x:6.2f}  {y:6.2f}   {h:.4f}")

    # Each sweep sits exactly halfway between two poses, so both the position and the
    # heading are the midpoint of their bracket.
    assert got["at_tx"] == [2.5, 7.5, 12.5]
    for got_heading, want in zip(got["heading"], [0.1, 0.3, 0.5], strict=True):
        assert abs(got_heading - want) < 1e-9, (got_heading, want)

    # Spherical interpolation sweeps the angle at a constant rate. That is the property
    # component-wise interpolation fails, and the reason to use this function at all.
    fine = bt.from_pydict({"f": [i / 8 for i in range(9)]})
    start, end = _yaw_quaternion(0.0), _yaw_quaternion(1.2)
    fine = fine.with_columns(
        **{f"a{n}": bt.lit(v) for n, v in zip("xyzw", start, strict=True)},
        **{f"b{n}": bt.lit(v) for n, v in zip("xyzw", end, strict=True)},
    )
    swept = fine.with_columns(
        **bt.quat_slerp(("ax", "ay", "az", "aw"), ("bx", "by", "bz", "bw"), "f")
    ).select("f", angle=bt.quat_angle("qx", "qy", "qz", "qw"))
    swept_rows = swept.sort("f").to_pydict()
    for f, angle in zip(swept_rows["f"], swept_rows["angle"], strict=True):
        assert abs(angle - 1.2 * f) < 1e-9, (f, angle)
    print("\nslerp sweeps the angle at a constant rate across the whole arc")


if __name__ == "__main__":
    main()
