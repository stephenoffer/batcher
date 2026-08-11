//! Rotations as quaternions, and the conversions into and out of the other two
//! spellings a log is likely to use.
//!
//! Three representations of the same thing appear in robotics data, and each is here
//! because some upstream producer insists on it:
//!
//! * the **quaternion**, which is what this module stores and what every operation is
//!   defined on, because it composes and interpolates without degenerating;
//! * **Euler angles** — roll, pitch and yaw — which is what a human reads and what an
//!   IMU or a map annotation usually publishes;
//! * the **rotation matrix**, which is what a calibration file usually contains,
//!   because a nine-number matrix needs no convention note to be unambiguous.
//!
//! Euler angles are supported and are never used internally. They are not a faithful
//! representation: at pitch = ±90 degrees roll and yaw describe the same motion and the
//! decomposition stops being unique, which is *gimbal lock*. `Quat::to_euler` reports
//! that case as roll = 0 and folds the whole rotation into yaw rather than returning
//! one of the infinitely many equivalent splits, so the result is at least a function.
//! Round-tripping a rotation through Euler angles near that pole does not return the
//! angles you started with, and no implementation can make it.

use crate::vec3::Vec3;

/// A rotation, as a quaternion in `(x, y, z, w)` order with the scalar part last.
///
/// See the crate documentation for why that order and not the other one. A value of
/// this type is not required to be unit-length; every operation that needs a rotation
/// normalizes first.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    /// The `i` coefficient.
    pub x: f64,
    /// The `j` coefficient.
    pub y: f64,
    /// The `k` coefficient.
    pub z: f64,
    /// The scalar part.
    pub w: f64,
}

/// A rotation as three angles in radians, in the intrinsic Z-Y-X sequence.
///
/// Applied in the order they are named in that sequence: yaw about Z first, then pitch
/// about the new Y, then roll about the new X.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Euler {
    /// Rotation about the X axis, in radians.
    pub roll: f64,
    /// Rotation about the Y axis, in radians.
    pub pitch: f64,
    /// Rotation about the Z axis, in radians.
    pub yaw: f64,
}

/// Below this length a quaternion carries no recoverable rotation direction, so the
/// operations that need one return `None` instead of dividing by a denormal and
/// reporting whatever falls out. Chosen well above `f64::MIN_POSITIVE` so that squaring
/// the components during normalization cannot itself underflow to zero.
const MIN_NORM: f64 = 1e-150;

/// Above this dot product two unit quaternions are close enough that `slerp`'s
/// `sin(theta)` denominator loses its significant digits, and a straight-line
/// interpolation is both stable and — at this separation — indistinguishable from the
/// spherical one.
const SLERP_LINEAR_ABOVE: f64 = 1.0 - 1e-9;

impl Quat {
    /// The quaternion with the given components, in `(x, y, z, w)` order.
    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    /// The identity rotation.
    pub const IDENTITY: Self = Self::new(0.0, 0.0, 0.0, 1.0);

    /// The Euclidean length of the four components.
    ///
    /// A rotation has length one. How far a logged quaternion has drifted from that is
    /// worth measuring directly, which is why this is exposed rather than kept private
    /// to `normalize`.
    pub fn norm(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    /// The same rotation as a unit quaternion, or `None` when there is no rotation to
    /// recover because every component is zero.
    pub fn normalize(self) -> Option<Self> {
        let n = self.norm();
        // `is_finite` first so a NaN component is rejected by the check that is defined
        // for it, rather than by a comparison whose answer is false either way.
        if !n.is_finite() || n <= MIN_NORM {
            return None;
        }
        Some(Self::new(self.x / n, self.y / n, self.z / n, self.w / n))
    }

    /// The conjugate: the vector part negated, the scalar part kept.
    ///
    /// For a unit quaternion this is the inverse rotation. For a non-unit one it is
    /// not, which is why `inverse` exists separately.
    pub fn conjugate(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, self.w)
    }

    /// The inverse rotation, as a unit quaternion.
    ///
    /// Normalizes first, so this is the inverse of the *rotation* the input denotes
    /// rather than the multiplicative inverse of the input itself. Those differ by a
    /// scale factor that a rotation cannot express, and returning the scaled version
    /// would let a drifted input quietly shrink a point cloud.
    pub fn inverse(self) -> Option<Self> {
        self.normalize().map(Self::conjugate)
    }

    /// The vector `v` rotated by this rotation.
    ///
    /// Uses the two-cross-product form rather than the literal `q * v * q_conj`
    /// sandwich: it is the same result in about half the multiplications, which matters
    /// because this is the function a lidar sweep calls once per point.
    pub fn rotate(self, v: Vec3) -> Option<Vec3> {
        let q = self.normalize()?;
        let qv = Vec3::new(q.x, q.y, q.z);
        let t = qv.cross(v).scale(2.0);
        Some(v + t.scale(q.w) + qv.cross(t))
    }

    /// The vector `v` rotated by the *inverse* of this rotation.
    ///
    /// The direction a world-frame measurement travels to reach a sensor frame, and
    /// common enough to deserve its own name rather than an `inverse` call the caller
    /// has to remember to make.
    pub fn inverse_rotate(self, v: Vec3) -> Option<Vec3> {
        self.inverse()?.rotate(v)
    }

    /// The magnitude of this rotation, in radians, on `[0, pi]`.
    ///
    /// Computed from `atan2` of the vector and scalar parts rather than from
    /// `2 * acos(w)`, because `acos` loses roughly half its significant digits for the
    /// small angles that dominate real pose data.
    pub fn angle(self) -> Option<f64> {
        let q = self.normalize()?;
        let vec_norm = Vec3::new(q.x, q.y, q.z).norm();
        let angle = 2.0 * vec_norm.atan2(q.w.abs());
        Some(angle)
    }

    /// The angle of the rotation taking `self` to `other`, in radians, on `[0, pi]`.
    ///
    /// This is the geodesic distance on the rotation group and the honest way to say
    /// "how wrong was this orientation estimate". Component-wise differences are not,
    /// because `q` and `-q` are the same rotation.
    pub fn angular_distance(self, other: Self) -> Option<f64> {
        let a = self.normalize()?;
        let b = other.normalize()?;
        // Take the nearer of the two antipodal representations, then measure with the
        // chord-length form: stable everywhere, unlike `2 * acos(dot)` near zero.
        let dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        let b = if dot < 0.0 {
            Self::new(-b.x, -b.y, -b.z, -b.w)
        } else {
            b
        };
        let diff = (a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
        let sum = (a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
        let dn = (diff.0 * diff.0 + diff.1 * diff.1 + diff.2 * diff.2 + diff.3 * diff.3).sqrt();
        let sn = (sum.0 * sum.0 + sum.1 * sum.1 + sum.2 * sum.2 + sum.3 * sum.3).sqrt();
        // `atan2(|a-b|, |a+b|)` is half the angle *between the quaternions*, and a
        // rotation angle is twice a quaternion angle — hence four, not two. The factor
        // is the double cover, and getting it wrong halves every reported error.
        Some(4.0 * dn.atan2(sn))
    }

    /// Spherical linear interpolation from `self` at `t = 0` to `other` at `t = 1`.
    ///
    /// The operation that makes sensor fusion possible: poses arrive at the localizer's
    /// rate and measurements arrive at the sensor's, so almost every measurement needs
    /// the pose *between* two logged poses. Interpolating the four components
    /// independently and renormalizing is the tempting alternative and it is wrong —
    /// it sweeps the angle at a non-constant rate, which shows up as a lidar sweep that
    /// bends.
    ///
    /// `t` is not clamped. Outside `[0, 1]` this extrapolates along the same great
    /// circle, which is what you want when a measurement's timestamp falls just past
    /// the last pose and is a mistake worth being able to make deliberately.
    pub fn slerp(self, other: Self, t: f64) -> Option<Self> {
        let a = self.normalize()?;
        let b = other.normalize()?;
        let mut dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        // `q` and `-q` name the same rotation, so flip to whichever of the two is on
        // the near side. Without this the interpolation takes the long way round the
        // sphere for any pair more than a quarter turn apart.
        let b = if dot < 0.0 {
            dot = -dot;
            Self::new(-b.x, -b.y, -b.z, -b.w)
        } else {
            b
        };
        if dot > SLERP_LINEAR_ABOVE {
            let lerped = Self::new(
                a.x + t * (b.x - a.x),
                a.y + t * (b.y - a.y),
                a.z + t * (b.z - a.z),
                a.w + t * (b.w - a.w),
            );
            return lerped.normalize();
        }
        let theta = dot.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();
        let sa = ((1.0 - t) * theta).sin() / sin_theta;
        let sb = (t * theta).sin() / sin_theta;
        Some(Self::new(
            sa * a.x + sb * b.x,
            sa * a.y + sb * b.y,
            sa * a.z + sb * b.z,
            sa * a.w + sb * b.w,
        ))
    }

    /// The rotation these intrinsic Z-Y-X angles describe.
    pub fn from_euler(e: Euler) -> Self {
        let (sr, cr) = (e.roll * 0.5).sin_cos();
        let (sp, cp) = (e.pitch * 0.5).sin_cos();
        let (sy, cy) = (e.yaw * 0.5).sin_cos();
        Self::new(
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    }

    /// This rotation as intrinsic Z-Y-X angles in radians.
    ///
    /// At the gimbal-lock poles — pitch at plus or minus a quarter turn — roll and yaw
    /// are not separately determined. This reports roll as zero there and puts the
    /// whole remaining rotation into yaw, which keeps the function single-valued at the
    /// cost of not round-tripping the angles you may have started with.
    pub fn to_euler(self) -> Option<Euler> {
        let q = self.normalize()?;
        // The middle angle first: it is the one that decides whether the other two are
        // separable at all.
        let sin_pitch = 2.0 * (q.w * q.y - q.z * q.x);
        if sin_pitch.abs() >= 1.0 - 1e-12 {
            let pitch = std::f64::consts::FRAC_PI_2.copysign(sin_pitch);
            // Locked, only `roll - yaw` (at pitch = +90) or `roll + yaw` (at -90) is
            // determined, and `2 * atan2(x, w)` is exactly that combination. Reporting
            // roll as zero leaves the whole of it in yaw, with the sign the combination
            // dictates.
            let yaw = -sin_pitch.signum() * 2.0 * q.x.atan2(q.w);
            return Some(Euler {
                roll: 0.0,
                pitch,
                yaw,
            });
        }
        Some(Euler {
            roll: (2.0 * (q.w * q.x + q.y * q.z)).atan2(1.0 - 2.0 * (q.x * q.x + q.y * q.y)),
            pitch: sin_pitch.asin(),
            yaw: (2.0 * (q.w * q.z + q.x * q.y)).atan2(1.0 - 2.0 * (q.y * q.y + q.z * q.z)),
        })
    }

    /// The rotation this row-major 3x3 matrix describes.
    ///
    /// Uses Shepperd's method: pick the component the trace says is largest and derive
    /// the other three from it. The textbook single-branch formula divides by
    /// `sqrt(1 + trace)`, which vanishes for a half turn and loses precision well
    /// before that; a calibration file containing a 180-degree sensor mount is not
    /// unusual.
    ///
    /// A matrix that is not a rotation is not detected. Feeding one in produces a
    /// quaternion that is the nearest rotation in no particular sense.
    #[allow(clippy::too_many_arguments)]
    pub fn from_rotation_matrix(
        m00: f64,
        m01: f64,
        m02: f64,
        m10: f64,
        m11: f64,
        m12: f64,
        m20: f64,
        m21: f64,
        m22: f64,
    ) -> Self {
        let trace = m00 + m11 + m22;
        if trace > 0.0 {
            let s = 0.5 / (trace + 1.0).sqrt();
            Self::new((m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s, 0.25 / s)
        } else if m00 > m11 && m00 > m22 {
            let s = 2.0 * (1.0 + m00 - m11 - m22).sqrt();
            Self::new(0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
        } else if m11 > m22 {
            let s = 2.0 * (1.0 + m11 - m00 - m22).sqrt();
            Self::new((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
        } else {
            let s = 2.0 * (1.0 + m22 - m00 - m11).sqrt();
            Self::new((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)
        }
    }
}

impl std::ops::Mul for Quat {
    type Output = Self;

    /// The Hamilton product `self * other` — the rotation that applies `other` first,
    /// then `self`.
    ///
    /// The argument order is the one that composes like function application and like
    /// matrix multiplication, and it is the opposite of the order the frames read in.
    /// Going from a sensor frame to the world through the vehicle is
    /// `world_from_ego * ego_from_sensor`.
    fn mul(self, other: Self) -> Self {
        Self::new(
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }

    fn close_vec(a: Vec3, b: Vec3) {
        close(a.x, b.x);
        close(a.y, b.y);
        close(a.z, b.z);
    }

    /// A quaternion for a rotation of `angle` about a unit axis.
    fn axis_angle(ax: f64, ay: f64, az: f64, angle: f64) -> Quat {
        let (s, c) = (angle * 0.5).sin_cos();
        Quat::new(ax * s, ay * s, az * s, c)
    }

    #[test]
    fn identity_rotates_nothing() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        close_vec(Quat::IDENTITY.rotate(v).unwrap(), v);
    }

    #[test]
    fn quarter_turn_about_z_maps_x_to_y() {
        let q = axis_angle(0.0, 0.0, 1.0, std::f64::consts::FRAC_PI_2);
        close_vec(
            q.rotate(Vec3::new(1.0, 0.0, 0.0)).unwrap(),
            Vec3::new(0.0, 1.0, 0.0),
        );
    }

    #[test]
    fn rotation_preserves_length() {
        let q = axis_angle(0.267_261, 0.534_522, 0.801_784, 1.234);
        let v = Vec3::new(3.0, -4.0, 12.0);
        close(q.rotate(v).unwrap().norm(), v.norm());
    }

    #[test]
    fn inverse_rotate_undoes_rotate() {
        let q = axis_angle(0.267_261, 0.534_522, 0.801_784, 2.1);
        let v = Vec3::new(1.5, -0.25, 7.0);
        close_vec(q.inverse_rotate(q.rotate(v).unwrap()).unwrap(), v);
    }

    #[test]
    fn non_unit_input_does_not_scale_the_result() {
        // The whole reason every entry point normalizes: a drifted quaternion must
        // rotate, not rotate-and-scale.
        let q = axis_angle(0.0, 0.0, 1.0, 0.7);
        let drifted = Quat::new(q.x * 3.0, q.y * 3.0, q.z * 3.0, q.w * 3.0);
        let v = Vec3::new(2.0, 0.0, 0.0);
        close_vec(drifted.rotate(v).unwrap(), q.rotate(v).unwrap());
    }

    #[test]
    fn zero_quaternion_has_no_rotation() {
        let zero = Quat::new(0.0, 0.0, 0.0, 0.0);
        assert!(zero.normalize().is_none());
        assert!(zero.rotate(Vec3::new(1.0, 0.0, 0.0)).is_none());
        assert!(zero.angle().is_none());
        assert!(zero.to_euler().is_none());
    }

    #[test]
    fn non_finite_quaternion_has_no_rotation() {
        assert!(Quat::new(f64::NAN, 0.0, 0.0, 1.0).normalize().is_none());
        assert!(Quat::new(f64::INFINITY, 0.0, 0.0, 1.0)
            .normalize()
            .is_none());
    }

    #[test]
    fn multiply_composes_in_apply_order() {
        // Rotating by `a` then by `b` is `b * a`, matching function composition.
        let a = axis_angle(0.0, 0.0, 1.0, 0.5);
        let b = axis_angle(1.0, 0.0, 0.0, 0.9);
        let v = Vec3::new(1.0, 2.0, 3.0);
        let stepwise = b.rotate(a.rotate(v).unwrap()).unwrap();
        close_vec((b * a).rotate(v).unwrap(), stepwise);
    }

    #[test]
    fn multiply_by_inverse_is_identity() {
        let q = axis_angle(0.267_261, 0.534_522, 0.801_784, 1.7);
        let i = q * q.inverse().unwrap();
        close(i.angle().unwrap(), 0.0);
    }

    #[test]
    fn angle_recovers_the_axis_angle_magnitude() {
        for &a in &[0.0, 1e-8, 0.3, 1.0, 3.0, std::f64::consts::PI] {
            let q = axis_angle(0.0, 1.0, 0.0, a);
            close(q.angle().unwrap(), a);
        }
    }

    #[test]
    fn angle_is_the_same_for_a_negated_quaternion() {
        let q = axis_angle(0.0, 1.0, 0.0, 1.1);
        let neg = Quat::new(-q.x, -q.y, -q.z, -q.w);
        close(q.angle().unwrap(), neg.angle().unwrap());
    }

    #[test]
    fn angular_distance_is_zero_to_self_and_to_its_negation() {
        let q = axis_angle(0.267_261, 0.534_522, 0.801_784, 2.0);
        close(q.angular_distance(q).unwrap(), 0.0);
        let neg = Quat::new(-q.x, -q.y, -q.z, -q.w);
        close(q.angular_distance(neg).unwrap(), 0.0);
    }

    #[test]
    fn angular_distance_matches_the_composed_angle() {
        let a = axis_angle(0.0, 0.0, 1.0, 0.4);
        let b = axis_angle(0.0, 0.0, 1.0, 1.9);
        close(a.angular_distance(b).unwrap(), 1.5);
        close(b.angular_distance(a).unwrap(), 1.5);
    }

    #[test]
    fn angular_distance_never_exceeds_half_a_turn() {
        let a = Quat::IDENTITY;
        let b = axis_angle(0.0, 0.0, 1.0, 6.0);
        let d = a.angular_distance(b).unwrap();
        assert!(d <= std::f64::consts::PI + EPS, "{d}");
        close(d, 2.0 * std::f64::consts::PI - 6.0);
    }

    #[test]
    fn slerp_hits_both_endpoints() {
        let a = axis_angle(0.0, 0.0, 1.0, 0.2);
        let b = axis_angle(0.0, 1.0, 0.0, 1.3);
        close(a.slerp(b, 0.0).unwrap().angular_distance(a).unwrap(), 0.0);
        close(a.slerp(b, 1.0).unwrap().angular_distance(b).unwrap(), 0.0);
    }

    #[test]
    fn slerp_sweeps_the_angle_at_a_constant_rate() {
        // This is the property component-wise interpolation fails, and the reason a
        // naively-interpolated pose makes a lidar sweep bend.
        let a = Quat::IDENTITY;
        let b = axis_angle(0.0, 0.0, 1.0, 1.2);
        for i in 0..=10 {
            let t = f64::from(i) / 10.0;
            close(a.slerp(b, t).unwrap().angle().unwrap(), 1.2 * t);
        }
    }

    #[test]
    fn slerp_takes_the_short_way_round() {
        let a = Quat::IDENTITY;
        let far = axis_angle(0.0, 0.0, 1.0, 3.0);
        // The antipodal spelling of the same rotation must interpolate identically.
        let neg = Quat::new(-far.x, -far.y, -far.z, -far.w);
        let via_far = a.slerp(far, 0.5).unwrap();
        let via_neg = a.slerp(neg, 0.5).unwrap();
        close(via_far.angular_distance(via_neg).unwrap(), 0.0);
        close(via_far.angle().unwrap(), 1.5);
    }

    #[test]
    fn slerp_between_near_identical_rotations_stays_finite() {
        let a = axis_angle(0.0, 0.0, 1.0, 1.0);
        let b = axis_angle(0.0, 0.0, 1.0, 1.0 + 1e-12);
        let mid = a.slerp(b, 0.5).unwrap();
        assert!(mid.x.is_finite() && mid.w.is_finite());
        close(mid.angle().unwrap(), 1.0);
    }

    #[test]
    fn slerp_extrapolates_past_the_endpoints() {
        let a = Quat::IDENTITY;
        let b = axis_angle(0.0, 0.0, 1.0, 0.5);
        close(a.slerp(b, 2.0).unwrap().angle().unwrap(), 1.0);
    }

    #[test]
    fn euler_round_trips_away_from_the_poles() {
        for &(r, p, y) in &[
            (0.0, 0.0, 0.0),
            (0.3, -0.2, 1.1),
            (-2.9, 0.9, 2.5),
            (1.5, 1.4, -1.5),
        ] {
            let e = Euler {
                roll: r,
                pitch: p,
                yaw: y,
            };
            let back = Quat::from_euler(e).to_euler().unwrap();
            close(back.roll, r);
            close(back.pitch, p);
            close(back.yaw, y);
        }
    }

    #[test]
    fn euler_axes_are_named_the_ros_way() {
        // roll about X, pitch about Y, yaw about Z — checked by rotating a basis vector.
        let roll = Quat::from_euler(Euler {
            roll: std::f64::consts::FRAC_PI_2,
            pitch: 0.0,
            yaw: 0.0,
        });
        close_vec(
            roll.rotate(Vec3::new(0.0, 1.0, 0.0)).unwrap(),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let yaw = Quat::from_euler(Euler {
            roll: 0.0,
            pitch: 0.0,
            yaw: std::f64::consts::FRAC_PI_2,
        });
        close_vec(
            yaw.rotate(Vec3::new(1.0, 0.0, 0.0)).unwrap(),
            Vec3::new(0.0, 1.0, 0.0),
        );
    }

    #[test]
    fn gimbal_lock_reports_a_single_valued_answer() {
        let e = Euler {
            roll: 0.4,
            pitch: std::f64::consts::FRAC_PI_2,
            yaw: 0.9,
        };
        let got = Quat::from_euler(e).to_euler().unwrap();
        close(got.roll, 0.0);
        close(got.pitch, std::f64::consts::FRAC_PI_2);
        // Roll and yaw are the same motion here, so only their difference survives.
        close(got.yaw, 0.9 - 0.4);
        // And the rotation itself is unchanged, which is the property that matters.
        close(
            Quat::from_euler(got)
                .angular_distance(Quat::from_euler(e))
                .unwrap(),
            0.0,
        );
    }

    #[test]
    fn rotation_matrix_round_trips_through_every_shepperd_branch() {
        // One rotation per branch: small (trace positive), then a half turn about each
        // axis in turn, which is where the naive single-branch formula divides by zero.
        let cases = [
            axis_angle(0.267_261, 0.534_522, 0.801_784, 0.6),
            axis_angle(1.0, 0.0, 0.0, std::f64::consts::PI),
            axis_angle(0.0, 1.0, 0.0, std::f64::consts::PI),
            axis_angle(0.0, 0.0, 1.0, std::f64::consts::PI),
        ];
        for q in cases {
            // Build the matrix by rotating the basis vectors, so the test does not
            // restate the formula it is checking.
            let c0 = q.rotate(Vec3::new(1.0, 0.0, 0.0)).unwrap();
            let c1 = q.rotate(Vec3::new(0.0, 1.0, 0.0)).unwrap();
            let c2 = q.rotate(Vec3::new(0.0, 0.0, 1.0)).unwrap();
            let back =
                Quat::from_rotation_matrix(c0.x, c1.x, c2.x, c0.y, c1.y, c2.y, c0.z, c1.z, c2.z);
            close(back.angular_distance(q).unwrap(), 0.0);
        }
    }
}
