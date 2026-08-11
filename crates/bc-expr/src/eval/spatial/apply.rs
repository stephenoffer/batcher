//! One row of a rigid-body function: numbers in, one number out.
//!
//! Split from `mod.rs` because it is a different job. Nothing here knows what an array
//! is; it reads a flat slice of `f64` laid out in the order the vocabulary documents and
//! calls into `bc_spatial`. That makes the argument-order convention — quaternions
//! scalar-last, poses translation-first — checkable in one place.

use bc_spatial::{Euler, Pose, Quat, Vec3};

use crate::SpatialFunc;

/// One row. `a` is exactly `func.arity()` long.
///
/// `None` is a null result — the only source of one is a quaternion with no rotation in
/// it. Split by argument shape rather than by function so each group reads its
/// arguments once, in the order the vocabulary documents them.
pub(super) fn apply(func: SpatialFunc, a: &[f64]) -> Option<f64> {
    use SpatialFunc::*;
    match func {
        // --- roll, pitch, yaw -----------------------------------------------
        QuatFromEulerX | QuatFromEulerY | QuatFromEulerZ | QuatFromEulerW => {
            let q = Quat::from_euler(Euler {
                roll: a[0],
                pitch: a[1],
                yaw: a[2],
            });
            Some(match func {
                QuatFromEulerX => q.x,
                QuatFromEulerY => q.y,
                QuatFromEulerZ => q.z,
                _ => q.w,
            })
        }

        // --- one quaternion --------------------------------------------------
        QuatNorm => Some(quat(a, 0).norm()),
        QuatNormalizeX | QuatNormalizeY | QuatNormalizeZ | QuatNormalizeW => {
            let q = quat(a, 0).normalize()?;
            Some(match func {
                QuatNormalizeX => q.x,
                QuatNormalizeY => q.y,
                QuatNormalizeZ => q.z,
                _ => q.w,
            })
        }
        QuatInverseX | QuatInverseY | QuatInverseZ | QuatInverseW => {
            let q = quat(a, 0).inverse()?;
            Some(match func {
                QuatInverseX => q.x,
                QuatInverseY => q.y,
                QuatInverseZ => q.z,
                _ => q.w,
            })
        }
        QuatAngle => quat(a, 0).angle(),
        QuatToRoll | QuatToPitch | QuatToYaw => {
            let e = quat(a, 0).to_euler()?;
            Some(match func {
                QuatToRoll => e.roll,
                QuatToPitch => e.pitch,
                _ => e.yaw,
            })
        }

        // --- a quaternion and a vector ---------------------------------------
        QuatRotateX | QuatRotateY | QuatRotateZ => component(quat(a, 0).rotate(vec3(a, 4))?, func),
        QuatInverseRotateX | QuatInverseRotateY | QuatInverseRotateZ => {
            component(quat(a, 0).inverse_rotate(vec3(a, 4))?, func)
        }

        // --- two quaternions --------------------------------------------------
        QuatMultiplyX | QuatMultiplyY | QuatMultiplyZ | QuatMultiplyW => {
            let q = quat(a, 0) * quat(a, 4);
            Some(match func {
                QuatMultiplyX => q.x,
                QuatMultiplyY => q.y,
                QuatMultiplyZ => q.z,
                _ => q.w,
            })
        }
        QuatAngularDistance => quat(a, 0).angular_distance(quat(a, 4)),

        // --- two quaternions and a parameter, or a matrix ---------------------
        QuatSlerpX | QuatSlerpY | QuatSlerpZ | QuatSlerpW => {
            let q = quat(a, 0).slerp(quat(a, 4), a[8])?;
            Some(match func {
                QuatSlerpX => q.x,
                QuatSlerpY => q.y,
                QuatSlerpZ => q.z,
                _ => q.w,
            })
        }
        QuatFromRotmatX | QuatFromRotmatY | QuatFromRotmatZ | QuatFromRotmatW => {
            let q =
                Quat::from_rotation_matrix(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
            Some(match func {
                QuatFromRotmatX => q.x,
                QuatFromRotmatY => q.y,
                QuatFromRotmatZ => q.z,
                _ => q.w,
            })
        }

        // --- a pose and a point -----------------------------------------------
        Se3TransformX | Se3TransformY | Se3TransformZ => {
            component(pose(a).transform(vec3(a, 7))?, func)
        }
        Se3InverseTransformX | Se3InverseTransformY | Se3InverseTransformZ => {
            component(pose(a).inverse_transform(vec3(a, 7))?, func)
        }
    }
}

/// The quaternion at `a[off..off + 4]`, in `(x, y, z, w)` order.
fn quat(a: &[f64], off: usize) -> Quat {
    Quat::new(a[off], a[off + 1], a[off + 2], a[off + 3])
}

/// The vector at `a[off..off + 3]`.
fn vec3(a: &[f64], off: usize) -> Vec3 {
    Vec3::new(a[off], a[off + 1], a[off + 2])
}

/// The pose at `a[0..7]` — translation first, then rotation.
fn pose(a: &[f64]) -> Pose {
    Pose::new(vec3(a, 0), quat(a, 3))
}

/// Pick the component a vector-valued function's name asks for.
fn component(v: Vec3, func: SpatialFunc) -> Option<f64> {
    use SpatialFunc::*;
    Some(match func {
        QuatRotateX | QuatInverseRotateX | Se3TransformX | Se3InverseTransformX => v.x,
        QuatRotateY | QuatInverseRotateY | Se3TransformY | Se3InverseTransformY => v.y,
        _ => v.z,
    })
}
