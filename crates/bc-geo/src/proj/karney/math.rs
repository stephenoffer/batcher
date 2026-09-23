//! The scalar helpers and series coefficients Karney's solver is written in terms of.
//!
//! Split from `mod.rs` only for size. Everything here is a direct transcription of
//! GeographicLib's `geomath.py` and the coefficient tables of `geodesic.py` (series
//! order 6), kept under the reference's names so the two can be compared line by line.

use super::N;

pub(super) const N_C3X: usize = N * (N - 1) / 2;
pub(super) const N_C4X: usize = N * (N + 1) / 2;
pub(super) const MAXIT1: u32 = 20;
pub(super) const MAXIT2: u32 = MAXIT1 + f64::MANTISSA_DIGITS + 10;

pub(super) fn tiny() -> f64 {
    f64::MIN_POSITIVE.sqrt()
}
pub(super) const TOL0: f64 = f64::EPSILON;
pub(super) const TOL1: f64 = 200.0 * TOL0;
pub(super) fn tol2() -> f64 {
    TOL0.sqrt()
}
pub(super) const TOLB: f64 = TOL0;
pub(super) fn xthresh() -> f64 {
    1000.0 * tol2()
}

pub(super) fn sq(x: f64) -> f64 {
    x * x
}

/// Horner evaluation of the degree-`n` polynomial whose coefficients start at `p[s]`.
pub(super) fn polyval(n: isize, p: &[f64], s: usize, x: f64) -> f64 {
    if n < 0 {
        return 0.0;
    }
    let mut y = p[s];
    for i in 1..=n as usize {
        y = y * x + p[s + i];
    }
    y
}

/// Error-free transformation of a sum: `u + v == s + t` exactly.
pub(super) fn sum(u: f64, v: f64) -> (f64, f64) {
    let s = u + v;
    let up = s - v;
    let vpp = s - up;
    let t = if s == 0.0 {
        s
    } else {
        0.0 - ((up - u) + (vpp - v))
    };
    (s, t)
}

/// IEEE remainder of `x / y`, in `[-y/2, y/2]`.
pub(super) fn remainder(x: f64, y: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    x - (x / y).round_ties_even() * y
}

pub(super) fn ang_round(x: f64) -> f64 {
    let z = 1.0 / 16.0;
    let mut y = x.abs();
    if y < z {
        y = z - (z - y);
    }
    y.copysign(x)
}

pub(super) fn ang_normalize(x: f64) -> f64 {
    let y = remainder(x, 360.0);
    if y.abs() == 180.0 {
        180f64.copysign(x)
    } else {
        y
    }
}

/// `y - x` reduced to `[-180, 180]`, with the rounding error as the second value.
pub(super) fn ang_diff(x: f64, y: f64) -> (f64, f64) {
    let (d, t) = sum(remainder(-x, 360.0), remainder(y, 360.0));
    let (mut d, t) = sum(remainder(d, 360.0), t);
    if d == 0.0 || d.abs() == 180.0 {
        d = d.copysign(if t == 0.0 { y - x } else { -t });
    }
    (d, t)
}

pub(super) fn quadrant(s: f64, c: f64, q: i64) -> (f64, f64) {
    match q.rem_euclid(4) {
        1 => (c, -s),
        2 => (-s, -c),
        3 => (-c, s),
        _ => (s, c),
    }
}

/// Sine and cosine of an angle in degrees, exact at multiples of 90.
pub(super) fn sincosd(x: f64) -> (f64, f64) {
    let mut r = if x.is_finite() { x % 360.0 } else { f64::NAN };
    let q = if r.is_nan() {
        0
    } else {
        (r / 90.0).round_ties_even() as i64
    };
    r = (r - 90.0 * q as f64).to_radians();
    let (s, c) = quadrant(r.sin(), r.cos(), q);
    let c = c + 0.0;
    let s = if s == 0.0 { s.copysign(x) } else { s };
    (s, c)
}

/// Sine and cosine of `x + t` degrees, for `x` in `[-180, 180]` and small `t`.
pub(super) fn sincosde(x: f64, t: f64) -> (f64, f64) {
    let q = if x.is_finite() {
        (x / 90.0).round_ties_even() as i64
    } else {
        0
    };
    let r = ang_round(x - 90.0 * q as f64 + t).to_radians();
    let (s, c) = quadrant(r.sin(), r.cos(), q);
    let c = c + 0.0;
    let s = if s == 0.0 { s.copysign(x) } else { s };
    (s, c)
}

pub(super) fn norm2(x: f64, y: f64) -> (f64, f64) {
    let r = x.hypot(y);
    (x / r, y / r)
}

/// Clenshaw summation of a sine (`sinp`) or cosine series.
pub(super) fn sin_cos_series(sinp: bool, sinx: f64, cosx: f64, c: &[f64]) -> f64 {
    let mut k = c.len();
    let mut n = k - usize::from(sinp);
    let ar = 2.0 * (cosx - sinx) * (cosx + sinx);
    let mut y1 = 0.0;
    let mut y0 = if n & 1 == 1 {
        k -= 1;
        c[k]
    } else {
        0.0
    };
    n /= 2;
    while n > 0 {
        n -= 1;
        k -= 1;
        y1 = ar * y0 - y1 + c[k];
        k -= 1;
        y0 = ar * y1 - y0 + c[k];
    }
    if sinp {
        2.0 * sinx * cosx * y0
    } else {
        cosx * (y0 - y1)
    }
}

pub(super) fn astroid(x: f64, y: f64) -> f64 {
    let p = sq(x);
    let q = sq(y);
    let r = (p + q - 1.0) / 6.0;
    if q == 0.0 && r <= 0.0 {
        return 0.0;
    }
    let s = p * q / 4.0;
    let r2 = sq(r);
    let r3 = r * r2;
    let disc = s * (s + 2.0 * r3);
    let mut u = r;
    if disc >= 0.0 {
        let mut t3 = s + r3;
        t3 += if t3 < 0.0 { -disc.sqrt() } else { disc.sqrt() };
        let t = t3.cbrt();
        u += t + if t != 0.0 { r2 / t } else { 0.0 };
    } else {
        let ang = (-disc).sqrt().atan2(-(s + r3));
        u += 2.0 * r * (ang / 3.0).cos();
    }
    let v = (sq(u) + q).sqrt();
    let uv = if u < 0.0 { q / (v - u) } else { u + v };
    let w = (uv - q) / (2.0 * v);
    uv / ((uv + sq(w)).sqrt() + w)
}

pub(super) fn a1m1f(eps: f64) -> f64 {
    const COEFF: [f64; 5] = [1.0, 4.0, 64.0, 0.0, 256.0];
    let m = (N / 2) as isize;
    let t = polyval(m, &COEFF, 0, sq(eps)) / COEFF[m as usize + 1];
    (t + eps) / (1.0 - eps)
}

pub(super) fn c1f(eps: f64, c: &mut [f64; N + 1]) {
    const COEFF: [f64; 18] = [
        -1.0, 6.0, -16.0, 32.0, -9.0, 64.0, -128.0, 2048.0, 9.0, -16.0, 768.0, 3.0, -5.0, 512.0,
        -7.0, 1280.0, -7.0, 2048.0,
    ];
    fill_even_series(eps, &COEFF, c);
}

pub(super) fn a2m1f(eps: f64) -> f64 {
    const COEFF: [f64; 5] = [-11.0, -28.0, -192.0, 0.0, 256.0];
    let m = (N / 2) as isize;
    let t = polyval(m, &COEFF, 0, sq(eps)) / COEFF[m as usize + 1];
    (t - eps) / (1.0 + eps)
}

pub(super) fn c2f(eps: f64, c: &mut [f64; N + 1]) {
    const COEFF: [f64; 18] = [
        1.0, 2.0, 16.0, 32.0, 35.0, 64.0, 384.0, 2048.0, 15.0, 80.0, 768.0, 7.0, 35.0, 512.0, 63.0,
        1280.0, 77.0, 2048.0,
    ];
    fill_even_series(eps, &COEFF, c);
}

/// The shared loop of `C1f` and `C2f`: coefficient `l` is `eps^l` times a polynomial in
/// `eps^2`.
pub(super) fn fill_even_series(eps: f64, coeff: &[f64], c: &mut [f64; N + 1]) {
    let eps2 = sq(eps);
    let mut d = eps;
    let mut o = 0usize;
    for (l, slot) in c.iter_mut().enumerate().skip(1) {
        let m = ((N - l) / 2) as isize;
        *slot = d * polyval(m, coeff, o, eps2) / coeff[o + m as usize + 1];
        o += m as usize + 2;
        d *= eps;
    }
}
