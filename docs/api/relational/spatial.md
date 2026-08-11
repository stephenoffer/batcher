# Rigid-body reference

Every rotation and pose function, grouped by what it does. This page is the lookup table.
To learn these rather than look them up, start with
{doc}`/user-guide/analyze/robotics`.

A rotation is four numbers and a pose is seven, and both live in ordinary `Float64`
columns rather than a composite type. That is what makes a coordinate-frame transform an
ordinary projection: the optimizer pushes it down, the executor spills it, and the
shuffle moves it, all without unpacking anything.

## Conventions

Every function on this page assumes the following, which are the ROS and SciPy
conventions:

| Question | Answer |
|---|---|
| Quaternion component order | `(x, y, z, w)`, scalar **last** |
| Handedness | Right-handed |
| What a rotation does | Moves the vector, not the axes |
| Euler sequence | Intrinsic Z-Y-X: yaw about Z, pitch about Y, roll about X |
| Pose layout | Translation `(tx, ty, tz)` first, then rotation `(qx, qy, qz, qw)` |
| Pose application | Rotate, then translate |

Scalar-last is the one that bites. ROS, SciPy and Eigen's storage order put `w` last;
nuScenes, the Waymo protos and Eigen's *constructor* put it first. Reading a quaternion
from one and passing it to the other is not an error anything can detect, it is a
different and entirely plausible rotation. Every function here takes the four components
as separate arguments so the order is written at the call site.

A quaternion whose components are all zero names no rotation. Every function returns
null for one rather than raising, so a single unrecoverable pose cannot end a scan over
the rest of the log. {py:func}`quat_norm <batcher.quat_norm>` is the exception, and is
how you find those rows.

```{eval-rst}
.. currentmodule:: batcher
```

## Whole rotations and whole poses

Start here. Each of these returns a mapping of column name to expression, so a whole
transform is one call splatted into
{py:meth}`with_columns <batcher.Dataset.with_columns>`. They compose the component
functions below and add no new engine surface.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   pose_interpolate
   quat_from_euler
   quat_inverse
   quat_multiply
   quat_normalize
   quat_slerp
   quat_to_euler
   se3_compose
   se3_inverse
   se3_inverse_transform
   se3_transform
```

## Applying a pose to a point

The component functions behind {py:func}`se3_transform <batcher.se3_transform>`. Reach
for these directly when you want one coordinate rather than three, which a filter
usually does.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   se3_inverse_transform_x
   se3_inverse_transform_y
   se3_inverse_transform_z
   se3_transform_x
   se3_transform_y
   se3_transform_z
```

## Rotating a vector

Rotation with no translation. The right choice for a velocity, an acceleration or a
surface normal, all of which turn with a frame but do not move with it. A position wants
the pose functions above.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   quat_inverse_rotate_x
   quat_inverse_rotate_y
   quat_inverse_rotate_z
   quat_rotate_x
   quat_rotate_y
   quat_rotate_z
```

## Reading a rotation

Turning a quaternion into a number you can filter, group or report on.
{py:func}`quat_to_yaw <batcher.quat_to_yaw>` is the one most queries want: a ground
vehicle is near-level almost always, so yaw is the whole of its orientation for most
purposes.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   quat_angle
   quat_angular_distance
   quat_norm
   quat_to_pitch
   quat_to_roll
   quat_to_yaw
```

## Building a rotation

From Euler angles, which is what an IMU or a map annotation publishes, or from a 3x3
matrix, which is what a calibration file usually carries. Matrix arguments are
row-major.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   quat_from_euler_w
   quat_from_euler_x
   quat_from_euler_y
   quat_from_euler_z
   quat_from_rotmat_w
   quat_from_rotmat_x
   quat_from_rotmat_y
   quat_from_rotmat_z
```

## Combining and interpolating rotations

{py:func}`quat_slerp_x <batcher.quat_slerp_x>` and its siblings are what make sensor
fusion work: poses arrive at the localizer's rate and measurements at each sensor's, so
almost every measurement needs the pose *between* two logged poses.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   quat_inverse_w
   quat_inverse_x
   quat_inverse_y
   quat_inverse_z
   quat_multiply_w
   quat_multiply_x
   quat_multiply_y
   quat_multiply_z
   quat_normalize_w
   quat_normalize_x
   quat_normalize_y
   quat_normalize_z
   quat_slerp_w
   quat_slerp_x
   quat_slerp_y
   quat_slerp_z
```

## Distances and directions

The measurements a point cloud is filtered and bucketed by, once its points are in
whichever frame the question is asked in.
{py:func}`voxel_index <batcher.voxel_index>` is the bucketing one: it turns
downsampling into an ordinary `group_by`. These are Cartesian and answer in the
coordinates' own units. A `lat`/`lon` column wants
{py:func}`st_distance_sphere <batcher.st_distance_sphere>` from
{doc}`/api/relational/geospatial` instead.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   azimuth_3d
   distance_3d
   elevation_3d
   norm_3d
   voxel_index
```

## See also

- {doc}`/user-guide/analyze/robotics` teaches these against a worked example.
- {doc}`/api/relational/geospatial` for geometry on the Earth's surface.
- {doc}`/api/relational/io` for the MCAP, MDF4 and point-cloud readers that produce
  these columns.
