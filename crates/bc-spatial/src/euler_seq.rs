//! Euler angles in any of the twelve axis sequences, intrinsic or extrinsic.
//!
//! The crate's own convention is intrinsic Z-Y-X, which is what ROS and aerospace use
//! and what `Quat::from_euler` / `Quat::to_euler` implement. Other fields standardize on
//! other orders: biomechanics on Y-X-Z, orbital mechanics on the proper Euler angles
//! Z-X-Z, a camera rig on whatever its vendor chose. This module reads and writes all of
//! them, so a log in one of those conventions converts at the call site instead of
//! through a hand-written chain of axis rotations.
//!
//! # Naming a sequence
//!
//! A sequence is three axes, with no axis repeated back to back. Following SciPy, an
//! *uppercase* sequence (`"ZYX"`) is intrinsic, each rotation about the axes as the
//! previous rotations left them, and a *lowercase* one (`"xyz"`) is extrinsic, each
//! about the fixed axes. Intrinsic `"ZYX"` and extrinsic `"xyz"` are the same rotation
//! with the angles listed in reverse order, and both are the crate's default.
//!
//! Across the Arrow boundary a sequence travels as a number, because every argument of
//! the family is a `Float64`: the three axes as digits, `x` = 1, `y` = 2, `z` = 3, plus
//! 1000 for intrinsic. Intrinsic `"ZYX"` is 1321; extrinsic `"xyz"` is 123. The Python
//! wrappers accept only the string and write the number.
//!
//! # Reading angles back
//!
//! `to_euler_seq` is the direct method of Bernardes and Viollet (2022), the one SciPy's
//! `Rotation.as_euler` uses, so the two agree on the angles and on the gimbal-lock
//! convention: where the middle angle is 0 or a half turn (a quarter turn for a
//! sequence with three distinct axes), the first and third angles are not separately
//! determined, and the third is reported as zero. For intrinsic `"ZYX"` the third angle
//! is roll, which is the choice `Quat::to_euler` makes. Every angle is wrapped to
//! `[-pi, pi)`.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use crate::quat::Quat;

/// Below this distance from 0 or a half turn, the middle angle counts as gimbal lock.
/// SciPy's value, kept so the two report the same angles at the same inputs.
const LOCK_EPS: f64 = 1e-7;

/// An axis sequence: three axis indices (0 = x, 1 = y, 2 = z) and whether it is
/// intrinsic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EulerSeq {
    axes: [usize; 3],
    intrinsic: bool,
}

impl EulerSeq {
    /// The sequence a numeric code names, or `None` for a code that names none.
    ///
    /// See the module docs for the encoding. A code is refused rather than guessed at
    /// when it is not a whole number, uses a digit other than 1 to 3, or repeats an
    /// axis back to back.
    #[must_use]
    pub fn from_code(code: f64) -> Option<Self> {
        if !code.is_finite() || code.fract() != 0.0 || !(0.0..2000.0).contains(&code) {
            return None;
        }
        let code = code as u32;
        let intrinsic = code >= 1000;
        let digits = code % 1000;
        let axes = [digits / 100, (digits / 10) % 10, digits % 10];
        if axes.iter().any(|d| !(1..=3).contains(d)) || axes[0] == axes[1] || axes[1] == axes[2] {
            return None;
        }
        Some(Self {
            axes: axes.map(|d| (d - 1) as usize),
            intrinsic,
        })
    }
}

/// The rotation by `angle` radians about one coordinate axis.
fn elementary(axis: usize, angle: f64) -> Quat {
    let (s, c) = (angle * 0.5).sin_cos();
    let mut v = [0.0; 3];
    v[axis] = s;
    Quat::new(v[0], v[1], v[2], c)
}

/// Wrap an angle to `[-pi, pi)`.
fn wrap(a: f64) -> f64 {
    (a + PI).rem_euclid(TAU) - PI
}

impl Quat {
    /// The rotation these three angles describe, about the axes of `seq` in order.
    ///
    /// Intrinsic angles compose left to right (`q1 * q2 * q3`), extrinsic ones right to
    /// left (`q3 * q2 * q1`), which is the whole difference between the two.
    #[must_use]
    pub fn from_euler_seq(seq: EulerSeq, angles: [f64; 3]) -> Self {
        let mut q = elementary(seq.axes[0], angles[0]);
        for (&axis, &angle) in seq.axes.iter().zip(&angles).skip(1) {
            let e = elementary(axis, angle);
            q = if seq.intrinsic { q * e } else { e * q };
        }
        q
    }

    /// This rotation as angles about the axes of `seq`, in radians, in sequence order.
    ///
    /// `None` for a quaternion with no rotation in it.
    #[must_use]
    pub fn to_euler_seq(self, seq: EulerSeq) -> Option<[f64; 3]> {
        let q = self.normalize()?;
        let quat = [q.x, q.y, q.z, q.w];
        let extrinsic = !seq.intrinsic;
        let mut axes = seq.axes;
        if !extrinsic {
            axes.reverse();
        }
        let [i, j, mut k] = axes;
        let symmetric = i == k;
        if symmetric {
            k = 3 - i - j;
        }
        // +1 for an even permutation of (x, y, z), -1 for an odd one.
        let (si, sj, sk) = (i as i64, j as i64, k as i64);
        let sign = ((si - sj) * (sj - sk) * (sk - si) / 2) as f64;
        let (a, b, c, d) = if symmetric {
            (quat[3], quat[i], quat[j], quat[k] * sign)
        } else {
            (
                quat[3] - quat[j],
                quat[i] + quat[k] * sign,
                quat[j] + quat[3],
                quat[k] * sign - quat[i],
            )
        };
        let half_sum = b.atan2(a);
        let half_diff = d.atan2(c);
        let mut angles = [0.0; 3];
        angles[1] = 2.0 * c.hypot(d).atan2(a.hypot(b));
        let (first, third) = if extrinsic { (0, 2) } else { (2, 0) };
        let near_zero = angles[1].abs() <= LOCK_EPS;
        let near_half = (angles[1] - PI).abs() <= LOCK_EPS;
        let free = !(near_zero || near_half);
        angles[0] = if near_zero {
            2.0 * half_sum
        } else {
            2.0 * half_diff * if extrinsic { -1.0 } else { 1.0 }
        };
        if free {
            angles[first] = half_sum - half_diff;
        }
        let mut last = if free {
            half_sum + half_diff
        } else {
            angles[third]
        };
        if !symmetric {
            last *= sign;
            angles[1] -= FRAC_PI_2;
        }
        angles[third] = last;
        Some(angles.map(wrap))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quat::Euler;

    const ALL: [&str; 12] = [
        "xyz", "xzy", "yxz", "yzx", "zxy", "zyx", "xyx", "xzx", "yxy", "yzy", "zxz", "zyz",
    ];

    fn code(seq: &str) -> f64 {
        let digits: String = seq
            .to_ascii_lowercase()
            .chars()
            .map(|c| char::from(b'1' + (c as u8 - b'x')))
            .collect();
        let n: f64 = digits.parse().unwrap();
        if seq.chars().all(|c| c.is_ascii_uppercase()) {
            n + 1000.0
        } else {
            n
        }
    }

    fn seqs() -> impl Iterator<Item = EulerSeq> {
        ALL.iter()
            .flat_map(|s| [s.to_string(), s.to_ascii_uppercase()])
            .map(|s| EulerSeq::from_code(code(&s)).unwrap())
    }

    #[test]
    fn codes_decode_and_invalid_ones_are_refused() {
        assert_eq!(
            EulerSeq::from_code(1321.0),
            Some(EulerSeq {
                axes: [2, 1, 0],
                intrinsic: true
            })
        );
        for bad in [0.0, 112.0, 124.0, 1321.5, f64::NAN, -123.0, 2123.0, 1122.0] {
            assert!(EulerSeq::from_code(bad).is_none(), "{bad}");
        }
    }

    #[test]
    fn intrinsic_zyx_is_the_crate_convention() {
        let seq = EulerSeq::from_code(1321.0).unwrap();
        let e = Euler {
            roll: 0.3,
            pitch: -0.6,
            yaw: 2.2,
        };
        let q = Quat::from_euler(e);
        let built = Quat::from_euler_seq(seq, [e.yaw, e.pitch, e.roll]);
        assert!(built.angular_distance(q).unwrap() < 1e-12);
        let [yaw, pitch, roll] = q.to_euler_seq(seq).unwrap();
        assert!((yaw - e.yaw).abs() < 1e-12);
        assert!((pitch - e.pitch).abs() < 1e-12);
        assert!((roll - e.roll).abs() < 1e-12);
        // Extrinsic xyz is the same rotation with the angles reversed.
        let ext = EulerSeq::from_code(123.0).unwrap();
        let built = Quat::from_euler_seq(ext, [e.roll, e.pitch, e.yaw]);
        assert!(built.angular_distance(q).unwrap() < 1e-12);
    }

    #[test]
    fn every_sequence_round_trips_away_from_gimbal_lock() {
        for seq in seqs() {
            for angles in [[0.4, 0.9, -1.2], [-2.5, 1.3, 0.7], [3.0, 0.2, -3.0]] {
                let q = Quat::from_euler_seq(seq, angles);
                let back = q.to_euler_seq(seq).unwrap();
                let again = Quat::from_euler_seq(seq, back);
                assert!(
                    again.angular_distance(q).unwrap() < 1e-10,
                    "{seq:?} {angles:?} -> {back:?}"
                );
            }
        }
    }

    #[test]
    fn gimbal_lock_zeroes_the_third_angle_and_keeps_the_rotation() {
        for seq in seqs() {
            let middle = if seq.axes[0] == seq.axes[2] {
                0.0
            } else {
                FRAC_PI_2
            };
            let q = Quat::from_euler_seq(seq, [0.7, middle, 0.4]);
            let back = q.to_euler_seq(seq).unwrap();
            assert_eq!(back[2], 0.0, "{seq:?} {back:?}");
            let again = Quat::from_euler_seq(seq, back);
            assert!(
                again.angular_distance(q).unwrap() < 1e-9,
                "{seq:?} {back:?}"
            );
        }
    }

    #[test]
    fn a_zero_quaternion_has_no_angles() {
        let seq = EulerSeq::from_code(1321.0).unwrap();
        assert!(Quat::new(0.0, 0.0, 0.0, 0.0).to_euler_seq(seq).is_none());
    }
}
