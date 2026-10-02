//! The rotation nearest to a matrix that has drifted from being one.
//!
//! `Quat::from_rotation_matrix` refuses a matrix more than `ROTMAT_TOL` from orthonormal,
//! which is the right default: a zero matrix from a missing calibration or a scaled one
//! from a units mix-up is a bug to surface, not a rotation to guess at. But a matrix that
//! has been through a few thousand multiplications, or was written out to four digits,
//! drifts past any fixed tolerance while still plainly meaning a rotation. This module is
//! the explicit repair for that case, kept separate so the refusal stays the default.
//!
//! # The method
//!
//! The rotation `R` nearest to a matrix `M` in the Frobenius norm is the one maximizing
//! `trace(R^T M)`, and for `R = R(q)` that trace is the quadratic form `q^T K q` of a
//! symmetric 4x4 matrix `K` built from the entries of `M`. The maximizing unit
//! quaternion is `K`'s eigenvector for its largest eigenvalue (Bar-Itzhack 2000, the same
//! construction as Horn's absolute orientation). For a matrix with a positive
//! determinant that rotation is also the orthogonal factor of the polar decomposition,
//! which is what an SVD-based repair (`U V^T`, SciPy's `Rotation.from_matrix`) returns.
//!
//! The eigenvector comes from a cyclic Jacobi iteration. `K` is 4x4 and symmetric, so
//! Jacobi converges quadratically, needs no pivoting, and has no failure mode on a
//! repeated eigenvalue: it is the textbook method for exactly this size of problem.
//!
//! A matrix with a determinant that is zero or negative is still refused. The nearest
//! rotation to a reflection exists, but a reflection in a calibration file is a
//! handedness mistake that no amount of rounding produces, and repairing it silently
//! would move every point to a plausible wrong place.

use crate::quat::Quat;

/// Jacobi sweeps before giving up. A 4x4 symmetric matrix converges in well under ten;
/// the cap only bounds a pathological input.
const MAX_SWEEPS: usize = 64;

impl Quat {
    /// The rotation nearest to this row-major 3x3 matrix, or `None` when the matrix has
    /// a non-finite entry or a determinant that is not positive.
    ///
    /// For a matrix that already is a rotation this is the same rotation
    /// `from_rotation_matrix` returns, up to rounding and the sign of the quaternion.
    /// The returned quaternion is normalized and has `w >= 0`.
    #[must_use]
    pub fn nearest_to_matrix(m: [f64; 9]) -> Option<Self> {
        if m.iter().any(|v| !v.is_finite()) || determinant(m) <= 0.0 {
            return None;
        }
        let [m00, m01, m02, m10, m11, m12, m20, m21, m22] = m;
        // `q^T K q == trace(R(q)^T M)` with `q = (x, y, z, w)`.
        let k = [
            [m00 - m11 - m22, m01 + m10, m02 + m20, m21 - m12],
            [m01 + m10, m11 - m00 - m22, m12 + m21, m02 - m20],
            [m02 + m20, m12 + m21, m22 - m00 - m11, m10 - m01],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ];
        let (values, vectors) = jacobi(k);
        let top = (0..4).max_by(|&a, &b| values[a].total_cmp(&values[b]))?;
        let q = Self::new(
            vectors[0][top],
            vectors[1][top],
            vectors[2][top],
            vectors[3][top],
        )
        .normalize()?;
        Some(if q.w < 0.0 {
            Self::new(-q.x, -q.y, -q.z, -q.w)
        } else {
            q
        })
    }
}

/// The determinant of a row-major 3x3 matrix.
#[must_use]
pub fn determinant(m: [f64; 9]) -> f64 {
    let [m00, m01, m02, m10, m11, m12, m20, m21, m22] = m;
    m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20)
}

/// Eigenvalues and eigenvectors (as columns) of a symmetric 4x4 matrix.
fn jacobi(mut a: [[f64; 4]; 4]) -> ([f64; 4], [[f64; 4]; 4]) {
    let mut v = [[0.0; 4]; 4];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..MAX_SWEEPS {
        let off: f64 = (0..4)
            .flat_map(|p| (p + 1..4).map(move |q| (p, q)))
            .map(|(p, q)| a[p][q] * a[p][q])
            .sum();
        let scale: f64 = (0..4).map(|i| a[i][i] * a[i][i]).sum::<f64>() + off;
        if off <= f64::EPSILON * f64::EPSILON * scale || off == 0.0 {
            break;
        }
        for p in 0..3 {
            for q in p + 1..4 {
                if a[p][q] == 0.0 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let (kp, kq) = (row[p], row[q]);
                    row[p] = c * kp - s * kq;
                    row[q] = s * kp + c * kq;
                }
                let (row_p, row_q) = (a[p], a[q]);
                for k in 0..4 {
                    a[p][k] = c * row_p[k] - s * row_q[k];
                    a[q][k] = s * row_p[k] + c * row_q[k];
                }
                for row in v.iter_mut() {
                    let (kp, kq) = (row[p], row[q]);
                    row[p] = c * kp - s * kq;
                    row[q] = s * kp + c * kq;
                }
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2], a[3][3]], v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quat::Euler;

    /// The row-major rotation matrix of a unit quaternion.
    fn matrix(q: Quat) -> [f64; 9] {
        let Quat { x, y, z, w } = q.normalize().unwrap();
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ]
    }

    fn same_rotation(a: Quat, b: Quat) -> bool {
        a.angular_distance(b).unwrap() < 1e-9
    }

    #[test]
    fn an_exact_rotation_comes_back_unchanged() {
        for (roll, pitch, yaw) in [
            (0.0, 0.0, 0.0),
            (0.3, -0.7, 2.1),
            (std::f64::consts::PI, 0.0, 0.0),
            (0.0, 0.0, std::f64::consts::PI),
            (1.0, std::f64::consts::FRAC_PI_2, -0.4),
        ] {
            let q = Quat::from_euler(Euler { roll, pitch, yaw });
            let got = Quat::nearest_to_matrix(matrix(q)).unwrap();
            assert!(same_rotation(got, q), "{q:?} -> {got:?}");
            assert!(got.w >= 0.0);
        }
    }

    #[test]
    fn a_drifted_matrix_is_repaired_where_the_strict_reader_refuses_it() {
        let q = Quat::from_euler(Euler {
            roll: 0.2,
            pitch: 0.1,
            yaw: -1.3,
        });
        let mut m = matrix(q);
        // 1% drift on every entry: far past ROTMAT_TOL, plainly still this rotation.
        for (i, v) in m.iter_mut().enumerate() {
            *v += 0.01 * if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let [m00, m01, m02, m10, m11, m12, m20, m21, m22] = m;
        assert!(Quat::from_rotation_matrix(m00, m01, m02, m10, m11, m12, m20, m21, m22).is_none());
        let got = Quat::nearest_to_matrix(m).unwrap();
        assert!(got.angular_distance(q).unwrap() < 0.02);
        // Nearest in the Frobenius sense: no small perturbation of the answer is closer.
        let dist = |r: Quat| {
            matrix(r)
                .iter()
                .zip(m)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
        };
        let best = dist(got);
        for axis in 0..3 {
            for step in [-1e-4f64, 1e-4] {
                let half = step / 2.0;
                let mut e = [0.0; 3];
                e[axis] = half.sin();
                let nudge = Quat::new(e[0], e[1], e[2], half.cos());
                assert!(dist(nudge * got) >= best);
            }
        }
    }

    #[test]
    fn a_scaled_rotation_repairs_to_the_rotation() {
        let q = Quat::from_euler(Euler {
            roll: 0.5,
            pitch: 0.0,
            yaw: 0.0,
        });
        let m = matrix(q).map(|v| v * 2.0);
        assert!(same_rotation(Quat::nearest_to_matrix(m).unwrap(), q));
    }

    #[test]
    fn a_reflection_a_zero_and_a_nan_are_refused() {
        let mut reflect = matrix(Quat::IDENTITY);
        reflect[8] = -1.0;
        assert!(Quat::nearest_to_matrix(reflect).is_none());
        assert!(Quat::nearest_to_matrix([0.0; 9]).is_none());
        let mut nan = matrix(Quat::IDENTITY);
        nan[4] = f64::NAN;
        assert!(Quat::nearest_to_matrix(nan).is_none());
    }
}
