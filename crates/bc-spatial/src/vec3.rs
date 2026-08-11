//! A point or displacement in three dimensions.
//!
//! Deliberately not a general vector type: it carries three `f64`s and the four
//! operations rigid-body math needs of it. Everything wider belongs to whatever
//! consumes it.

/// A three-dimensional vector — a point, a displacement, or an axis.
///
/// Addition and subtraction are the `std::ops` traits rather than inherent methods, so
/// `a + b` reads as the arithmetic it is and cannot be confused with `Add::add`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    /// The first component.
    pub x: f64,
    /// The second component.
    pub y: f64,
    /// The third component.
    pub z: f64,
}

impl Vec3 {
    /// The vector with the given components.
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// The zero vector.
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);

    /// Every component multiplied by `k`.
    pub fn scale(self, k: f64) -> Self {
        Self::new(self.x * k, self.y * k, self.z * k)
    }

    /// The dot product.
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// The cross product, `self x other`.
    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// The Euclidean length.
    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;

    /// Component-wise sum.
    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;

    /// Component-wise difference.
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_is_right_handed() {
        // x cross y == z is the definition of the handedness this whole crate assumes.
        let got = Vec3::new(1.0, 0.0, 0.0).cross(Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(got, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn cross_is_antisymmetric() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(-4.0, 5.0, 0.5);
        let ab = a.cross(b);
        let ba = b.cross(a);
        assert!((ab.x + ba.x).abs() < 1e-15);
        assert!((ab.y + ba.y).abs() < 1e-15);
        assert!((ab.z + ba.z).abs() < 1e-15);
    }

    #[test]
    fn cross_is_orthogonal_to_both() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(-4.0, 5.0, 0.5);
        let c = a.cross(b);
        assert!(c.dot(a).abs() < 1e-13);
        assert!(c.dot(b).abs() < 1e-13);
    }

    #[test]
    fn norm_matches_pythagoras() {
        assert!((Vec3::new(3.0, 4.0, 12.0).norm() - 13.0).abs() < 1e-15);
    }

    #[test]
    fn add_and_sub_round_trip() {
        let a = Vec3::new(1.5, -2.0, 0.25);
        let b = Vec3::new(-0.5, 7.0, 3.0);
        assert_eq!((a + b) - b, a);
    }

    #[test]
    fn scale_distributes_over_add() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!((a + b).scale(2.0), a.scale(2.0) + b.scale(2.0));
    }
}
