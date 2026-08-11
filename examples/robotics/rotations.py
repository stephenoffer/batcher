"""Building, cleaning, composing and scoring rotations.

Four things a log makes you do before any of its orientations can be trusted. The
sources disagree: the IMU publishes Euler angles, the calibration file has a 3x3 matrix,
and the localizer's quaternions have drifted off unit length after a few thousand
multiplications. This gets them all into one comparable form and then measures how far
apart two of them are.

    python examples/robotics/rotations.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batcher as bt
from batcher import col


def main() -> None:
    # --- 1. From Euler angles, which is what an IMU publishes ------------------
    imu = bt.from_pydict(
        {
            "roll": [0.0, 0.1, -0.3],
            "pitch": [0.0, 0.0, 0.2],
            "yaw": [0.0, math.pi / 2, 1.0],
        }
    )
    # The whole-rotation helper is one call; the four component functions behind it are
    # there for when you want a single component in a filter.
    as_quat = imu.with_columns(**bt.quat_from_euler("roll", "pitch", "yaw"))
    one_at_a_time = imu.select(
        qx=bt.quat_from_euler_x("roll", "pitch", "yaw"),
        qy=bt.quat_from_euler_y("roll", "pitch", "yaw"),
        qz=bt.quat_from_euler_z("roll", "pitch", "yaw"),
        qw=bt.quat_from_euler_w("roll", "pitch", "yaw"),
    ).to_pydict()
    # The zero-angle row is the identity rotation, whichever way it is built.
    assert abs(one_at_a_time["qw"][0] - 1.0) < 1e-12
    assert all(abs(one_at_a_time[c][0]) < 1e-12 for c in ("qx", "qy", "qz"))

    # Reading them back is the round trip that proves the convention is what you think.
    back = as_quat.select(**bt.quat_to_euler(("qx", "qy", "qz", "qw")))
    got = back.to_pydict()
    for name, before in [("roll", imu.to_pydict()["roll"]), ("yaw", imu.to_pydict()["yaw"])]:
        for a, b in zip(before, got[name], strict=True):
            assert abs(a - b) < 1e-12, (name, a, b)
    print("euler -> quaternion -> euler round trips")

    # The individual angle readers are what a query filters on. A ground vehicle is
    # near-level almost always, so yaw is the whole of its orientation for most purposes.
    angles = as_quat.select(
        r=bt.quat_to_roll("qx", "qy", "qz", "qw"),
        p=bt.quat_to_pitch("qx", "qy", "qz", "qw"),
        y=bt.quat_to_yaw("qx", "qy", "qz", "qw"),
    ).to_pydict()
    assert abs(angles["y"][1] - math.pi / 2) < 1e-12
    assert abs(angles["r"][2] - (-0.3)) < 1e-12
    assert abs(angles["p"][2] - 0.2) < 1e-12

    # --- 2. From a rotation matrix, which is what a calibration file carries ----
    # A quarter turn about Z, written out row-major. Nine numbers need no convention
    # note, which is why calibration files use them.
    calib = bt.from_pydict(
        {
            "m00": [0.0], "m01": [-1.0], "m02": [0.0],
            "m10": [1.0], "m11": [0.0], "m12": [0.0],
            "m20": [0.0], "m21": [0.0], "m22": [1.0],
        }
    )  # fmt: skip
    m = ["m00", "m01", "m02", "m10", "m11", "m12", "m20", "m21", "m22"]
    from_matrix = calib.select(
        qx=bt.quat_from_rotmat_x(*m),
        qy=bt.quat_from_rotmat_y(*m),
        qz=bt.quat_from_rotmat_z(*m),
        qw=bt.quat_from_rotmat_w(*m),
    )
    yaw = from_matrix.select(y=bt.quat_to_yaw("qx", "qy", "qz", "qw")).to_pydict()["y"]
    assert abs(yaw[0] - math.pi / 2) < 1e-12
    print(f"calibration matrix reads back as yaw {yaw[0]:.4f} rad")

    # --- 3. Cleaning a drifted rotation column ---------------------------------
    # Three spellings of the same quarter turn about Z: unit, scaled by three, and
    # negated. Every function normalizes internally, so this is for storage and for
    # comparing raw components, not for the correctness of the arithmetic.
    s, c = math.sin(math.pi / 4), math.cos(math.pi / 4)
    drifted = bt.from_pydict(
        {
            "qx": [0.0, 0.0, 0.0],
            "qy": [0.0, 0.0, 0.0],
            "qz": [s, s * 3, -s],
            "qw": [c, c * 3, -c],
        }
    )
    q = ("qx", "qy", "qz", "qw")
    lengths = drifted.select(n=bt.quat_norm(*q)).to_pydict()["n"]
    assert abs(lengths[0] - 1.0) < 1e-12
    assert abs(lengths[1] - 3.0) < 1e-12
    print(f"quaternion lengths before cleaning: {[round(v, 3) for v in lengths]}")

    cleaned = drifted.with_columns(**bt.quat_normalize(q, prefix="u"))
    # The component functions are the same thing one column at a time.
    also = drifted.select(
        ux=bt.quat_normalize_x(*q),
        uy=bt.quat_normalize_y(*q),
        uz=bt.quat_normalize_z(*q),
        uw=bt.quat_normalize_w(*q),
    ).to_pydict()
    after = cleaned.select(n=bt.quat_norm("uqx", "uqy", "uqz", "uqw")).to_pydict()["n"]
    assert all(abs(v - 1.0) < 1e-12 for v in after)
    assert abs(also["uz"][1] - s) < 1e-12
    print("all rotations are unit length after normalizing")

    # The inverse turns each of them around. For a unit quaternion that is the conjugate,
    # but `quat_inverse` normalizes first so a drifted input cannot smuggle in a scale.
    inverted = drifted.with_columns(**bt.quat_inverse(q, prefix="i"))
    parts = drifted.select(
        ix=bt.quat_inverse_x(*q),
        iy=bt.quat_inverse_y(*q),
        iz=bt.quat_inverse_z(*q),
        iw=bt.quat_inverse_w(*q),
    ).to_pydict()
    assert abs(parts["iz"][1] + s) < 1e-12
    assert abs(parts["iw"][1] - c) < 1e-12

    # --- 4. Composing two rotations --------------------------------------------
    # Two quarter turns about Z make a half turn. `quat_multiply` applies its second
    # argument first, matching function composition.
    pair = drifted.select("qx", "qy", "qz", "qw").with_columns(
        **{f"b{n}": col(f"q{n}") for n in "xyzw"}
    )
    b = ("bx", "by", "bz", "bw")
    composed = pair.with_columns(**bt.quat_multiply(q, b, prefix="c"))
    components = pair.select(
        cx=bt.quat_multiply_x(*q, *b),
        cy=bt.quat_multiply_y(*q, *b),
        cz=bt.quat_multiply_z(*q, *b),
        cw=bt.quat_multiply_w(*q, *b),
    ).to_pydict()
    turn = composed.select(a=bt.quat_angle("cqx", "cqy", "cqz", "cqw")).to_pydict()["a"]
    assert abs(turn[0] - math.pi) < 1e-12
    assert abs(components["cw"][0]) < 1e-12
    print(f"two quarter turns compose to {turn[0]:.4f} rad")

    # --- 5. Scoring one rotation against another --------------------------------
    # The geodesic angle, which is the honest error metric. Row 2 is the *negated*
    # spelling of row 0's rotation, and scores zero — a component-wise difference would
    # score it as maximally wrong.
    scored = inverted.select("qx", "qy", "qz", "qw").with_columns(
        rx=bt.lit(0.0), ry=bt.lit(0.0), rz=bt.lit(s), rw=bt.lit(c)
    )
    err = scored.select(
        d=bt.quat_angular_distance("qx", "qy", "qz", "qw", "rx", "ry", "rz", "rw")
    ).to_pydict()["d"]
    print(f"angular error against the reference: {[round(v, 6) for v in err]}")
    assert abs(err[0]) < 1e-12
    assert abs(err[1]) < 1e-12
    assert abs(err[2]) < 1e-12

    # --- 6. Rotating a vector that is not a position -----------------------------
    # A velocity turns with the frame but does not move with it, so it takes the
    # rotation alone and never a pose. This is the distinction that quietly ruins a
    # pipeline if `se3_transform` is used for it: the translation would be added to a
    # speed.
    motion = bt.from_pydict(
        {
            # A vehicle yawed a quarter turn, moving at 12 m/s along its own forward axis.
            "qx": [0.0], "qy": [0.0], "qz": [s], "qw": [c],
            "vx": [12.0], "vy": [0.0], "vz": [0.0],
        }
    )  # fmt: skip
    mq, mv = ("qx", "qy", "qz", "qw"), ("vx", "vy", "vz")
    in_world = motion.select(
        wx=bt.quat_rotate_x(*mq, *mv),
        wy=bt.quat_rotate_y(*mq, *mv),
        wz=bt.quat_rotate_z(*mq, *mv),
    )
    world_velocity = in_world.to_pydict()
    # Yawed a quarter turn, "forward" points along world +Y.
    assert abs(world_velocity["wx"][0]) < 1e-12
    assert abs(world_velocity["wy"][0] - 12.0) < 1e-12
    print(
        "12 m/s forward reads as "
        f"({world_velocity['wx'][0]:.2f}, {world_velocity['wy'][0]:.2f}) in the world"
    )

    # And back into the body frame, which is how a world-frame wind or a tracked
    # object's velocity is read as forward/left/up relative to the vehicle.
    body = (
        in_world.cross_join(motion.select(*mq))
        .select(
            bx=bt.quat_inverse_rotate_x(*mq, "wx", "wy", "wz"),
            by=bt.quat_inverse_rotate_y(*mq, "wx", "wy", "wz"),
            bz=bt.quat_inverse_rotate_z(*mq, "wx", "wy", "wz"),
        )
        .to_pydict()
    )
    assert abs(body["bx"][0] - 12.0) < 1e-12
    assert abs(body["by"][0]) < 1e-12
    assert abs(body["bz"][0]) < 1e-12
    print("and rotates back to 12 m/s along the vehicle's own forward axis")

    # --- 7. Interpolating, one component at a time ------------------------------
    # `quat_slerp` returns all four; these are the same thing spelled out, which is what
    # you want when only one component feeds a downstream filter.
    ends = bt.from_pydict(
        {"ax": [0.0], "ay": [0.0], "az": [0.0], "aw": [1.0],
         "bx": [0.0], "by": [0.0], "bz": [s], "bw": [c], "t": [0.5]}
    )  # fmt: skip
    a4, b4 = ("ax", "ay", "az", "aw"), ("bx", "by", "bz", "bw")
    mid = ends.select(
        mx=bt.quat_slerp_x(*a4, *b4, "t"),
        my=bt.quat_slerp_y(*a4, *b4, "t"),
        mz=bt.quat_slerp_z(*a4, *b4, "t"),
        mw=bt.quat_slerp_w(*a4, *b4, "t"),
    )
    half = mid.select(a=bt.quat_angle("mx", "my", "mz", "mw")).to_pydict()["a"]
    assert abs(half[0] - math.pi / 4) < 1e-12
    print(f"halfway through a quarter turn is {half[0]:.4f} rad")


if __name__ == "__main__":
    main()
