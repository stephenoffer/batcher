//! `bc-spatial` — rigid-body motion in three dimensions: rotations, poses, frames.
//!
//! A robotics or autonomous-driving log is a pile of measurements taken in different
//! coordinate frames. The lidar reports points in the sensor's frame, the camera in
//! its own, the localizer reports where the vehicle was in the world, and the
//! calibration file says where each sensor is bolted relative to the vehicle. Almost
//! every question worth asking — *did this detection overlap that one*, *where was the
//! obstacle in world coordinates*, *how far did the ego travel* — is a question about
//! moving a measurement from one frame into another. That move is a rigid transform,
//! and this crate is the arithmetic for it.
//!
//! This crate is a **near-leaf**: it depends on nothing in the workspace and on no
//! third-party code. It sits below `bc-expr`, which calls it per row to evaluate the
//! `Expr::Spatial` variant. Nothing here knows about Arrow, which is the same split
//! `bc-geo` uses and for the same reason — the array-level plumbing belongs in
//! `bc-expr`, so this crate stays a library of pure functions that unit-test without a
//! `RecordBatch` in sight.
//!
//! # Conventions, stated once and never varied
//!
//! Conventions are where rigid-body code goes wrong, because every one of them is a
//! defensible choice that silently disagrees with the next library's equally
//! defensible choice. Batcher picks the ones the robotics ecosystem already uses:
//!
//! | Question | This crate's answer | Who else says so |
//! |---|---|---|
//! | Quaternion component order | `(x, y, z, w)`, scalar **last** | ROS `geometry_msgs/Quaternion`, SciPy, Eigen's `coeffs()` |
//! | Handedness | right-handed | everyone |
//! | What a rotation does | **active**: it moves the vector, it does not relabel the axes | ROS `tf2`, SciPy |
//! | Euler angle sequence | intrinsic **Z-Y-X** — yaw about Z, then pitch about Y, then roll about X | ROS, aerospace |
//! | Euler naming | roll about X, pitch about Y, yaw about Z | ROS, aerospace |
//! | A pose | translation `(tx, ty, tz)` plus rotation quaternion, applied **rotate-then-translate** | ROS `geometry_msgs/Pose`, SE(3) convention |
//!
//! The scalar-last order is worth dwelling on because it is the one that bites. ROS,
//! SciPy and Eigen's storage order put `w` last; nuScenes, Waymo's proto and Eigen's
//! *constructor* put it first. A quaternion read out of one and into the other without
//! reordering is not an error anything can detect — it is a different, plausible
//! rotation. Batcher takes the components as four separate named arguments precisely so
//! the order is written at the call site rather than assumed.
//!
//! # Non-unit quaternions are normalized, not rejected
//!
//! A rotation is a *unit* quaternion, but a quaternion that has been logged, converted
//! to `float32`, interpolated, or multiplied a few thousand times is only nearly unit.
//! Every function here that needs a rotation normalizes its input first, so drift shows
//! up as the rounding it is rather than as a scale factor smuggled into the answer. A
//! quaternion of length zero carries no rotation, and every such function returns
//! `None` for it — which the expression layer surfaces as a null, the same as any other
//! undefined result.

pub mod quat;
pub mod rigid;
pub mod vec3;

pub use quat::{Euler, Quat};
pub use rigid::Pose;
pub use vec3::Vec3;
