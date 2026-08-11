"""The rigid-body function family: rotations, poses and coordinate frames.

A robotics or autonomous-driving log is measurements taken in different coordinate
frames — one per sensor, one for the vehicle, one for the world — and almost every
question worth asking of it is a question about moving a measurement between them. This
family is that arithmetic, as ordinary `Float64` expressions over the scalar columns a
log already stores.

Grouped by module: `quaternion` reads a rotation apart and builds one from Euler angles
or a matrix, `rotate` composes and interpolates rotations and applies one to a vector,
`pose` applies a full rigid transform to a point, `frames` wraps all of those as
whole-rotation and whole-pose helpers that return a mapping of column name to
expression, and `vectors` measures the distances and directions a point cloud is
filtered by.

The conventions — quaternions in ``(x, y, z, w)`` order with the scalar last, intrinsic
Z-Y-X Euler angles, right-handed active rotations, poses that rotate and then translate
— are the ROS and SciPy ones, and are stated in full in the `bc_spatial` Rust crate and
in :doc:`/user-guide/analyze/robotics`.

Each submodule's own ``__all__`` is the single curated list for its group, and this
façade splices them rather than restating fifty-seven names a fourth time (the two
higher façades, `plan/functions/__init__.py` and `api/functions.py`, splice this one).
"""

from __future__ import annotations

from batcher.plan.functions.spatial import frames, pose, quaternion, rotate, vectors
from batcher.plan.functions.spatial.frames import *  # noqa: F403
from batcher.plan.functions.spatial.pose import *  # noqa: F403
from batcher.plan.functions.spatial.quaternion import *  # noqa: F403
from batcher.plan.functions.spatial.rotate import *  # noqa: F403
from batcher.plan.functions.spatial.vectors import *  # noqa: F403

__all__ = sorted(
    [
        *frames.__all__,
        *pose.__all__,
        *quaternion.__all__,
        *rotate.__all__,
        *vectors.__all__,
    ]
)
