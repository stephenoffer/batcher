# Robotics and autonomous driving

This page covers working with coordinate frames in a robotics or autonomous-driving log:
what a pose is, how to move a measurement between frames, how to line up sensors that
sample at different rates, and how to filter a point cloud once its points are where you
need them.

## What is a coordinate frame?

A robot log is a pile of measurements, and none of them are in the same coordinate
system. The lidar reports points in the lidar's frame, whose origin is the sensor and
whose axes point wherever the sensor is bolted. The camera reports in its own. The
localizer reports where the vehicle is in a world frame that does not move. The
calibration file says where each sensor sits relative to the vehicle.

A *frame* is one of those coordinate systems. A *pose* is where one frame sits inside
another: three numbers for the translation and four for the rotation. Naming a pose for
the two frames it relates, target first, is the convention ROS uses and the one this page
follows, so `world_from_lidar` is "where the lidar is, expressed in the world frame".

Almost every interesting question about a log is a question about moving a measurement
from one frame to another. Did this detection overlap that one? Where was the obstacle in
world coordinates? How far did the vehicle travel? Each is a rigid transform, and Batcher
spells those as ordinary expressions over the scalar columns a log already has.

## How rotations are represented

Batcher stores a rotation as a *quaternion*: four numbers in `(x, y, z, w)` order, with
the scalar part last. That is the ROS, SciPy and Eigen-storage order.

```{warning}
nuScenes, the Waymo protos and Eigen's constructor put the scalar part **first**. A
quaternion read from one of those and passed to a function here without reordering is
not an error anything can detect. It is a different, entirely plausible rotation, and it
will move your points somewhere wrong and quiet. Every function in this family takes the
four components as separate arguments precisely so that the order is written down at the
call site.
```

Quaternions are used rather than Euler angles because they compose and interpolate
without degenerating. Euler angles are for reading and reporting, and
{py:func}`quat_to_euler <batcher.quat_to_euler>` is there for that. They are not unique
at a pitch of plus or minus a quarter turn, which is *gimbal lock*, so a rotation that
round-trips through them near that pole does not come back as the angles it started as.

A rotation is a *unit* quaternion, but a logged one is only nearly unit after being
rounded to `float32` and multiplied a few thousand times. Every function here normalizes
its input first, so drift shows up as the rounding it is instead of a scale factor
smuggled into your coordinates. A quaternion of length zero carries no rotation at all,
and those rows come back null rather than raising.

## Moving a point between frames

{py:func}`se3_transform <batcher.se3_transform>` takes a pose and a point and returns
three named columns, so a whole transform is one call:

```python
import batcher as bt

sweep = bt.from_pydict(
    {
        # Where the lidar was, in the world frame, when this point was measured.
        "tx": [100.0, 100.0, 100.0],
        "ty": [50.0, 50.0, 50.0],
        "tz": [1.8, 1.8, 1.8],
        "qx": [0.0, 0.0, 0.0],
        "qy": [0.0, 0.0, 0.0],
        "qz": [0.0, 0.0, 0.0],
        "qw": [1.0, 1.0, 1.0],
        # The returns, in the lidar's own frame.
        "x": [10.0, 20.0, -5.0],
        "y": [0.0, 3.0, 0.0],
        "z": [0.0, 0.0, 1.0],
    }
)

world_from_lidar = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
in_world = sweep.with_columns(
    **bt.se3_transform(world_from_lidar, ("x", "y", "z"), prefix="world_")
)
print(in_world.select("world_x", "world_y", "world_z").to_pydict())
```

The transform rotates and then translates. The other order is a different transform and
gets a different, wrong answer for every point that is not at the origin.

Going the other way — a world-frame obstacle expressed relative to the vehicle — is
{py:func}`se3_inverse_transform <batcher.se3_inverse_transform>`:

```python
relative = in_world.with_columns(
    **bt.se3_inverse_transform(
        world_from_lidar, ("world_x", "world_y", "world_z"), prefix="back_"
    )
)
print(relative.select("back_x", "back_y", "back_z").to_pydict())
```

Use {py:func}`quat_rotate_x <batcher.quat_rotate_x>` and its siblings instead when the
thing you are moving is a velocity, an acceleration or a surface normal. Those turn with
a frame but do not move with it, so applying a translation to one is wrong.

## Chaining frames

A sensor is rarely posed directly in the world. The calibration says where the lidar is
relative to the vehicle, and the localizer says where the vehicle is in the world. Chain
them with {py:func}`se3_compose <batcher.se3_compose>`, which cancels the adjacent frame
names: `world_from_ego` composed with `ego_from_lidar` is `world_from_lidar`.

```python
import batcher as bt

frames = bt.from_pydict(
    {
        # world_from_ego: the vehicle, from the localizer.
        "etx": [100.0], "ety": [50.0], "etz": [0.0],
        "eqx": [0.0], "eqy": [0.0], "eqz": [0.0], "eqw": [1.0],
        # ego_from_lidar: the sensor mount, from the calibration file.
        "ltx": [1.2], "lty": [0.0], "ltz": [1.8],
        "lqx": [0.0], "lqy": [0.0], "lqz": [0.0], "lqw": [1.0],
    }
)

world_from_ego = ("etx", "ety", "etz", "eqx", "eqy", "eqz", "eqw")
ego_from_lidar = ("ltx", "lty", "ltz", "lqx", "lqy", "lqz", "lqw")
chained = frames.with_columns(
    **bt.se3_compose(world_from_ego, ego_from_lidar, prefix="w_")
)
print(chained.select("w_tx", "w_ty", "w_tz").to_pydict())
```

Compose once per frame and apply the single result per point. A sweep is a hundred
thousand points and a log is tens of thousands of sweeps, so collapsing the chain first
is the difference between one transform per point and three.

## Lining up sensors that sample at different rates

Poses arrive at the localizer's rate and measurements arrive at each sensor's, so a
measurement almost never lands exactly on a logged pose. The fix is to interpolate the
pose to the measurement's timestamp.

Interpolating the four quaternion components independently and renormalizing is the
obvious approach and it is wrong: it sweeps the angle at a non-constant rate, which shows
up as a lidar sweep that visibly bends.
{py:func}`pose_interpolate <batcher.pose_interpolate>` interpolates the rotation
spherically and the translation linearly, which is the standard treatment.

The fraction is a number you compute, not a timestamp:

```python
import batcher as bt

# Each row: a measurement, with the logged poses bracketing its timestamp already
# attached (see the note below on how to attach them).
rows = bt.from_pydict(
    {
        "t_meas": [1500.0],
        "t_before": [1000.0],
        "t_after": [2000.0],
        "ax": [0.0], "ay": [0.0], "az": [0.0],
        "bx": [10.0], "by": [0.0], "bz": [0.0],
        "qx": [0.0], "qy": [0.0], "qz": [0.0], "qw": [1.0],
    }
)

frac = (bt.col("t_meas") - bt.col("t_before")) / (bt.col("t_after") - bt.col("t_before"))
rot = ("qx", "qy", "qz", "qw")
at_measurement = rows.with_columns(
    **bt.pose_interpolate(("ax", "ay", "az", *rot), ("bx", "by", "bz", *rot), frac)
)
print(at_measurement.select("tx", "ty", "tz").to_pydict())
```

To attach the bracketing poses in the first place, join the measurements against the pose
table with {py:meth}`join_asof <batcher.Dataset.join_asof>` twice: once backward for the
pose at or before each timestamp, and once forward for the one after.

`t` is not clamped, so a measurement whose timestamp falls just past the last logged pose
extrapolates along the same arc rather than pinning to the endpoint. That is usually what
you want, and it is worth knowing you are doing it.

## Filtering a point cloud

Batcher reads a point cloud as one row per point, with `x`, `y`, `z` and whatever else
the format carries as ordinary columns. That is what makes the usual first cuts native
operators rather than a Python loop:

```python
import batcher as bt

cloud = bt.from_pydict(
    {
        "x": [1.0, 40.0, 0.3, 12.0],
        "y": [0.0, 5.0, 0.1, -3.0],
        "z": [-1.9, 0.5, 0.0, 2.0],
        "intensity": [12, 200, 3, 88],
    }
)

point = ("x", "y", "z")
useful = cloud.filter(
    # Drop the vehicle itself and anything past the sensor's rated range.
    (bt.norm_3d(point) > 2.0)
    & (bt.norm_3d(point) < 60.0)
    # Drop the ground, given a sensor mounted 1.8 m up.
    & (bt.col("z") > -1.7)
)
print(useful.select("x", "y", "z").to_pydict())
```

{py:func}`azimuth_3d <batcher.azimuth_3d>` and
{py:func}`elevation_3d <batcher.elevation_3d>` give the other two spherical coordinates,
which are how you split a sweep into the sectors a rotating sensor actually measures in
and how you separate ground returns from overhead structure.

Comparing a range against a threshold is faster if you square the threshold and compare
against the squared distance, skipping a square root per point. That is worth doing in an
inner loop over billions of points and not worth doing anywhere else.

## Downsampling a point cloud

A lidar sweep is far denser near the sensor than far from it, which biases anything you
compute over it and costs you memory for returns that add nothing. *Voxel downsampling*
fixes both: bin the points into cubes and keep one point per occupied cube.

{py:func}`voxel_index <batcher.voxel_index>` produces the bin coordinates, so the
downsample is an ordinary grouped aggregation:

```python
import batcher as bt
from batcher import col

cloud = bt.from_pydict(
    {
        "x": [0.01, 0.02, 0.03, 5.00, 5.01],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0],
        "z": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
)

thinned = cloud.group_by(**bt.voxel_index(("x", "y", "z"), 0.2)).agg(
    n=bt.count(),
    x=col("x").mean(),
    y=col("y").mean(),
    z=col("z").mean(),
)
print(sorted(thinned.to_pydict()["n"]))
```

Because it is a `group_by`, it spills when the cloud does not fit and distributes across
a cluster with no special handling. Nothing about it is point-cloud-specific machinery.

The bin index is a floor and an integer, and both matter. Truncating toward zero would
make the cube straddling the origin twice as wide as every other cube, which is a real
defect that is very hard to see in a rendered cloud. And grouping on a float key can
split one group in two.

## Scoring an orientation estimate

{py:func}`quat_angular_distance <batcher.quat_angular_distance>` is the honest error
metric between two rotations: the angle of the rotation taking one to the other. A
component-wise difference is not, because a quaternion and its negation are the same
rotation and would score as maximally far apart.

```python
import batcher as bt

runs = bt.from_pydict(
    {
        "ax": [0.0, 0.0], "ay": [0.0, 0.0], "az": [0.0, 0.0], "aw": [1.0, 1.0],
        # The second row is the same rotation spelled with every sign flipped.
        "bx": [0.0, 0.0], "by": [0.0, 0.0], "bz": [0.0, 0.0], "bw": [1.0, -1.0],
    }
)
err = runs.select(
    e=bt.quat_angular_distance("ax", "ay", "az", "aw", "bx", "by", "bz", "bw")
)
print(err.to_pydict())
```

Aggregate it like any other column to get a per-log or per-scenario score.

## Where this runs

Every function on this page is a `Float64` expression over `Float64` columns. Nothing
here introduces a composite type, which has three consequences worth stating:

- A coordinate transform is an ordinary projection, so the optimizer pushes it below a
  join, prunes it when its output is unused, and fuses it with neighbouring arithmetic.
- It spills and shuffles like arithmetic, so a transform over a cluster-scale log needs
  no special handling and produces the same rows single-node or distributed.
- It streams. A transform in an
  {py:meth}`iter_batches <batcher.Dataset.iter_batches>` pipeline never materializes the
  sweep.

The arithmetic itself runs in Rust, in the `bc-spatial` crate. The JIT declines this
family and falls back to the interpreter, which is deliberate: the kernels call
transcendental functions whose bit-for-bit agreement across two tiers is a claim nothing
currently proves, and the interpreter's loop is already tight.

## Requirements and limitations

- Euler angles use the intrinsic Z-Y-X sequence, with roll about X, pitch about Y and
  yaw about Z. There is no way to select a different sequence.
- {py:func}`quat_to_euler <batcher.quat_to_euler>` reports roll as zero at gimbal lock
  and folds the whole rotation into yaw. That keeps the function single-valued; it does
  not make the decomposition unique, because nothing can.
- {py:func}`quat_from_rotmat_x <batcher.quat_from_rotmat_x>` does not check that its nine
  arguments are a rotation. Feeding it something else produces a quaternion that is the
  nearest rotation in no particular sense.
- These are Cartesian and answer in the coordinates' own units. A `lat`/`lon` column
  wants {doc}`/user-guide/analyze/geospatial` instead.

## See also

- {doc}`/api/relational/spatial` enumerates every function on this page.
- {doc}`/user-guide/analyze/joins` for `join_asof`, which is how sensor streams are
  aligned before they are interpolated.
- {doc}`/api/relational/io` for the MCAP, MDF4 and point-cloud readers.
- {doc}`/user-guide/analyze/geospatial` for geometry on the Earth's surface.
