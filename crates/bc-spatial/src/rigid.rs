//! A pose — where a frame is, and which way it is facing — and the transforms between
//! frames that poses define.
//!
//! A `Pose` is an element of SE(3): a rotation and a translation, applied in that
//! order. Read `world_from_sensor` as "the pose of the sensor, expressed in the world
//! frame", and applying it to a point takes that point *out of* the sensor frame and
//! *into* the world frame. Naming a pose for the two frames it relates, target first,
//! is the ROS `tf2` convention and it is the reason a chain composes by cancelling
//! adjacent names: `world_from_ego * ego_from_lidar` is `world_from_lidar`.

use crate::quat::Quat;
use crate::vec3::Vec3;

/// A rigid transform: rotate, then translate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    /// Where the frame's origin sits in the parent frame.
    pub translation: Vec3,
    /// How the frame is oriented relative to the parent frame.
    pub rotation: Quat,
}

impl Pose {
    /// The pose with the given translation and rotation.
    pub const fn new(translation: Vec3, rotation: Quat) -> Self {
        Self {
            translation,
            rotation,
        }
    }

    /// The pose that moves nothing.
    pub const IDENTITY: Self = Self::new(Vec3::ZERO, Quat::IDENTITY);

    /// `point`, expressed in the parent frame.
    ///
    /// Rotation is applied before translation. The other order is a different transform
    /// and gets a different, wrong answer for every point off the origin.
    pub fn transform(self, point: Vec3) -> Option<Vec3> {
        Some(self.rotation.rotate(point)? + self.translation)
    }

    /// `point`, expressed in *this* frame, given its coordinates in the parent frame.
    ///
    /// The inverse of `transform`, and worth having directly: composing it out of
    /// `inverse` and `transform` costs an extra rotation and gets the subtract-then-
    /// rotate order wrong about half the time it is written by hand.
    pub fn inverse_transform(self, point: Vec3) -> Option<Vec3> {
        self.rotation.inverse_rotate(point - self.translation)
    }

    /// The pose that undoes this one.
    pub fn inverse(self) -> Option<Self> {
        let inv = self.rotation.inverse()?;
        Some(Self::new(inv.rotate(self.translation)?.scale(-1.0), inv))
    }

    /// `self * other` — the transform that applies `other` first, then `self`.
    ///
    /// Composes the way the frame names read: `world_from_ego.compose(ego_from_lidar)`
    /// is `world_from_lidar`.
    pub fn compose(self, other: Self) -> Option<Self> {
        Some(Self::new(
            self.rotation.rotate(other.translation)? + self.translation,
            self.rotation * other.rotation,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-9, "{a} != {b}");
    }

    fn close_vec(a: Vec3, b: Vec3) {
        close(a.x, b.x);
        close(a.y, b.y);
        close(a.z, b.z);
    }

    fn axis_angle(ax: f64, ay: f64, az: f64, angle: f64) -> Quat {
        let (s, c) = (angle * 0.5).sin_cos();
        Quat::new(ax * s, ay * s, az * s, c)
    }

    fn sample() -> Pose {
        Pose::new(
            Vec3::new(1.0, -2.0, 0.5),
            axis_angle(0.267_261, 0.534_522, 0.801_784, 1.1),
        )
    }

    #[test]
    fn identity_moves_nothing() {
        let v = Vec3::new(3.0, 4.0, 5.0);
        close_vec(Pose::IDENTITY.transform(v).unwrap(), v);
    }

    #[test]
    fn rotation_is_applied_before_translation() {
        // A quarter turn about Z then a shift along X. Applied the other way round the
        // origin point would land at (0, 1, 0) instead.
        let p = Pose::new(
            Vec3::new(1.0, 0.0, 0.0),
            axis_angle(0.0, 0.0, 1.0, std::f64::consts::FRAC_PI_2),
        );
        close_vec(
            p.transform(Vec3::new(1.0, 0.0, 0.0)).unwrap(),
            Vec3::new(1.0, 1.0, 0.0),
        );
    }

    #[test]
    fn inverse_transform_undoes_transform() {
        let p = sample();
        let v = Vec3::new(7.0, -1.5, 2.25);
        close_vec(p.inverse_transform(p.transform(v).unwrap()).unwrap(), v);
    }

    #[test]
    fn inverse_pose_agrees_with_inverse_transform() {
        let p = sample();
        let v = Vec3::new(0.25, 9.0, -3.0);
        close_vec(
            p.inverse().unwrap().transform(v).unwrap(),
            p.inverse_transform(v).unwrap(),
        );
    }

    #[test]
    fn compose_matches_applying_each_in_turn() {
        let a = sample();
        let b = Pose::new(Vec3::new(-0.5, 3.0, 1.0), axis_angle(0.0, 1.0, 0.0, -0.7));
        let v = Vec3::new(2.0, 2.0, 2.0);
        close_vec(
            a.compose(b).unwrap().transform(v).unwrap(),
            a.transform(b.transform(v).unwrap()).unwrap(),
        );
    }

    #[test]
    fn compose_with_inverse_is_identity() {
        let p = sample();
        let round = p.compose(p.inverse().unwrap()).unwrap();
        close_vec(round.translation, Vec3::ZERO);
        close(round.rotation.angle().unwrap(), 0.0);
    }

    #[test]
    fn compose_is_associative() {
        let a = sample();
        let b = Pose::new(Vec3::new(-0.5, 3.0, 1.0), axis_angle(0.0, 1.0, 0.0, -0.7));
        let c = Pose::new(Vec3::new(4.0, 0.0, -1.0), axis_angle(1.0, 0.0, 0.0, 2.2));
        let v = Vec3::new(1.0, -1.0, 0.5);
        let left = a.compose(b).unwrap().compose(c).unwrap();
        let right = a.compose(b.compose(c).unwrap()).unwrap();
        close_vec(left.transform(v).unwrap(), right.transform(v).unwrap());
    }

    #[test]
    fn a_pose_with_no_rotation_transforms_nothing() {
        let p = Pose::new(Vec3::new(1.0, 1.0, 1.0), Quat::new(0.0, 0.0, 0.0, 0.0));
        assert!(p.transform(Vec3::ZERO).is_none());
        assert!(p.inverse_transform(Vec3::ZERO).is_none());
        assert!(p.inverse().is_none());
    }
}
