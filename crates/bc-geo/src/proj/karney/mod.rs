//! Karney's geodesic inverse on the WGS 84 ellipsoid: distance and polygon area.
//!
//! The `*_spheroid` functions promise answers on the ellipsoid, and this is the solver
//! that keeps the promise everywhere. Vincenty's inverse, which this replaced, is
//! accurate where it converges and returns nothing for near-antipodal pairs; a sphere,
//! which the length and area functions used to use, is off by 0.3-0.5% everywhere. This
//! solver converges for every pair of points on the globe, antipodal included, and is
//! accurate to a few nanometres in distance.
//!
//! It is a transcription of Charles Karney's GeographicLib (`geodesic.py` and
//! `polygonarea.py`, version 2.1, MIT/X11 licence, copyright Charles Karney 2011-2022),
//! restricted to what the engine needs: the inverse problem returning the distance
//! `s12` and the area term `S12`, on an oblate ellipsoid (`f > 0`). The algorithm is
//! described in C. F. F. Karney, "Algorithms for geodesics", J. Geodesy 87, 43-55
//! (2013). DuckDB's spatial extension and PROJ link the same library, which is why the
//! differential tests can hold this to their answers at 1e-9 relative rather than to a
//! tolerance chosen to pass.
//!
//! The code follows the reference's structure and variable names on purpose. It is a
//! numerical kernel whose correctness rests on matching a published algorithm line for
//! line; renaming `salp1` to something friendlier would make that comparison harder and
//! buy nothing.

use std::sync::OnceLock;

use crate::proj::geodesy::{WGS84_A, WGS84_F};

mod area;
mod math;

pub use area::ring_area;

use math::{
    a1m1f, a2m1f, ang_diff, ang_round, astroid, c1f, c2f, norm2, polyval, sin_cos_series, sincosd,
    sincosde, sq, tiny, tol2, xthresh, MAXIT1, MAXIT2, N_C3X, N_C4X, TOL0, TOL1, TOLB,
};

/// Series order. Six is GeographicLib's default and is accurate to round-off for WGS 84.
const N: usize = 6;

/// The reference's branch threshold `comg12 > -0.7071` ("omg12 < 3/4 pi"), kept at its
/// four published digits rather than `-FRAC_1_SQRT_2` so this picks the same formula for
/// the area term as GeographicLib does on the same input.
#[allow(clippy::approx_constant)]
const COS_3PI_4_ROUNDED: f64 = -0.7071;

/// One ellipsoid's precomputed constants.
struct Geodesic {
    a: f64,
    f: f64,
    f1: f64,
    e2: f64,
    ep2: f64,
    n: f64,
    b: f64,
    c2: f64,
    etol2: f64,
    a3x: [f64; N],
    c3x: [f64; N_C3X],
    c4x: [f64; N_C4X],
}

impl Geodesic {
    fn new(a: f64, f: f64) -> Self {
        let f1 = 1.0 - f;
        let e2 = f * (2.0 - f);
        let ep2 = e2 / sq(f1);
        let n = f / (2.0 - f);
        let b = a * f1;
        // Oblate only: `e2 > 0`, so the atanh branch of the reference applies.
        let c2 = (sq(a) + sq(b) * e2.sqrt().atanh() / e2.abs().sqrt()) / 2.0;
        let etol2 = 0.1 * tol2() / ((0.001f64.max(f.abs()) * 1f64.min(1.0 - f / 2.0) / 2.0).sqrt());
        let mut g = Geodesic {
            a,
            f,
            f1,
            e2,
            ep2,
            n,
            b,
            c2,
            etol2,
            a3x: [0.0; N],
            c3x: [0.0; N_C3X],
            c4x: [0.0; N_C4X],
        };
        g.a3coeff();
        g.c3coeff();
        g.c4coeff();
        g
    }

    fn a3coeff(&mut self) {
        const COEFF: [f64; 18] = [
            -3.0, 128.0, -2.0, -3.0, 64.0, -1.0, -3.0, -1.0, 16.0, 3.0, -1.0, -2.0, 8.0, 1.0, -1.0,
            2.0, 1.0, 1.0,
        ];
        let (mut o, mut k) = (0usize, 0usize);
        for j in (0..N).rev() {
            let m = (N - j - 1).min(j) as isize;
            self.a3x[k] = polyval(m, &COEFF, o, self.n) / COEFF[o + m as usize + 1];
            k += 1;
            o += m as usize + 2;
        }
    }

    fn c3coeff(&mut self) {
        const COEFF: [f64; 45] = [
            3.0, 128.0, 2.0, 5.0, 128.0, -1.0, 3.0, 3.0, 64.0, -1.0, 0.0, 1.0, 8.0, -1.0, 1.0, 4.0,
            5.0, 256.0, 1.0, 3.0, 128.0, -3.0, -2.0, 3.0, 64.0, 1.0, -3.0, 2.0, 32.0, 7.0, 512.0,
            -10.0, 9.0, 384.0, 5.0, -9.0, 5.0, 192.0, 7.0, 512.0, -14.0, 7.0, 512.0, 21.0, 2560.0,
        ];
        let (mut o, mut k) = (0usize, 0usize);
        for l in 1..N {
            for j in (l..N).rev() {
                let m = (N - j - 1).min(j) as isize;
                self.c3x[k] = polyval(m, &COEFF, o, self.n) / COEFF[o + m as usize + 1];
                k += 1;
                o += m as usize + 2;
            }
        }
    }

    fn c4coeff(&mut self) {
        const COEFF: [f64; 77] = [
            97.0, 15015.0, 1088.0, 156.0, 45045.0, -224.0, -4784.0, 1573.0, 45045.0, -10656.0,
            14144.0, -4576.0, -858.0, 45045.0, 64.0, 624.0, -4576.0, 6864.0, -3003.0, 15015.0,
            100.0, 208.0, 572.0, 3432.0, -12012.0, 30030.0, 45045.0, 1.0, 9009.0, -2944.0, 468.0,
            135135.0, 5792.0, 1040.0, -1287.0, 135135.0, 5952.0, -11648.0, 9152.0, -2574.0,
            135135.0, -64.0, -624.0, 4576.0, -6864.0, 3003.0, 135135.0, 8.0, 10725.0, 1856.0,
            -936.0, 225225.0, -8448.0, 4992.0, -1144.0, 225225.0, -1440.0, 4160.0, -4576.0, 1716.0,
            225225.0, -136.0, 63063.0, 1024.0, -208.0, 105105.0, 3584.0, -3328.0, 1144.0, 315315.0,
            -128.0, 135135.0, -2560.0, 832.0, 405405.0, 128.0, 99099.0,
        ];
        let (mut o, mut k) = (0usize, 0usize);
        for l in 0..N {
            for j in (l..N).rev() {
                let m = (N - j - 1) as isize;
                self.c4x[k] = polyval(m, &COEFF, o, self.n) / COEFF[o + m as usize + 1];
                k += 1;
                o += m as usize + 2;
            }
        }
    }

    fn a3f(&self, eps: f64) -> f64 {
        polyval(N as isize - 1, &self.a3x, 0, eps)
    }

    fn c3f(&self, eps: f64, c: &mut [f64; N]) {
        let mut mult = 1.0;
        let mut o = 0usize;
        for (l, slot) in c.iter_mut().enumerate().skip(1) {
            let m = (N - l - 1) as isize;
            mult *= eps;
            *slot = mult * polyval(m, &self.c3x, o, eps);
            o += m as usize + 1;
        }
    }

    fn c4f(&self, eps: f64, c: &mut [f64; N]) {
        let mut mult = 1.0;
        let mut o = 0usize;
        for (l, slot) in c.iter_mut().enumerate() {
            let m = (N - l - 1) as isize;
            *slot = mult * polyval(m, &self.c4x, o, eps);
            o += m as usize + 1;
            mult *= eps;
        }
    }

    /// The reference's `_Lengths`, restricted to distance and reduced length.
    /// Returns `(s12b, m12b, m0)`.
    #[allow(clippy::too_many_arguments)]
    fn lengths(
        &self,
        eps: f64,
        sig12: f64,
        (ssig1, csig1, dn1): (f64, f64, f64),
        (ssig2, csig2, dn2): (f64, f64, f64),
        distance: bool,
        reduced: bool,
        c1a: &mut [f64; N + 1],
        c2a: &mut [f64; N + 1],
    ) -> (f64, f64, f64) {
        let (mut s12b, mut m12b, mut m0) = (f64::NAN, f64::NAN, f64::NAN);
        let (mut a1, mut a2, mut m0x) = (0.0, 0.0, 0.0);
        if distance || reduced {
            a1 = a1m1f(eps);
            c1f(eps, c1a);
            if reduced {
                a2 = a2m1f(eps);
                c2f(eps, c2a);
                m0x = a1 - a2;
                a2 += 1.0;
            }
            a1 += 1.0;
        }
        let mut j12 = f64::NAN;
        if distance {
            let b1 =
                sin_cos_series(true, ssig2, csig2, c1a) - sin_cos_series(true, ssig1, csig1, c1a);
            s12b = a1 * (sig12 + b1);
            if reduced {
                let b2 = sin_cos_series(true, ssig2, csig2, c2a)
                    - sin_cos_series(true, ssig1, csig1, c2a);
                j12 = m0x * sig12 + (a1 * b1 - a2 * b2);
            }
        } else if reduced {
            for l in 1..=N {
                c2a[l] = a1 * c1a[l] - a2 * c2a[l];
            }
            j12 = m0x * sig12
                + (sin_cos_series(true, ssig2, csig2, c2a)
                    - sin_cos_series(true, ssig1, csig1, c2a));
        }
        if reduced {
            m0 = m0x;
            m12b = dn2 * (csig1 * ssig2) - dn1 * (ssig1 * csig2) - csig1 * csig2 * j12;
        }
        (s12b, m12b, m0)
    }

    /// A starting value for Newton's method. Returns
    /// `(sig12, salp1, calp1, salp2, calp2, dnm)`; `sig12 >= 0` means the short-line
    /// solution is already final.
    #[allow(clippy::too_many_arguments)]
    fn inverse_start(
        &self,
        (sbet1, cbet1): (f64, f64),
        (sbet2, cbet2): (f64, f64),
        lam12: f64,
        slam12: f64,
        clam12: f64,
    ) -> (f64, f64, f64, f64, f64, f64) {
        let mut sig12 = -1.0;
        let (mut salp2, mut calp2, mut dnm) = (f64::NAN, f64::NAN, f64::NAN);
        let sbet12 = sbet2 * cbet1 - cbet2 * sbet1;
        let cbet12 = cbet2 * cbet1 + sbet2 * sbet1;
        let sbet12a = sbet2 * cbet1 + cbet2 * sbet1;
        let shortline = cbet12 >= 0.0 && sbet12 < 0.5 && cbet2 * lam12 < 0.5;
        let (mut somg12, mut comg12);
        if shortline {
            let mut sbetm2 = sq(sbet1 + sbet2);
            sbetm2 /= sbetm2 + sq(cbet1 + cbet2);
            dnm = (1.0 + self.ep2 * sbetm2).sqrt();
            let omg12 = lam12 / (self.f1 * dnm);
            somg12 = omg12.sin();
            comg12 = omg12.cos();
        } else {
            somg12 = slam12;
            comg12 = clam12;
        }
        let mut salp1 = cbet2 * somg12;
        let mut calp1 = if comg12 >= 0.0 {
            sbet12 + cbet2 * sbet1 * sq(somg12) / (1.0 + comg12)
        } else {
            sbet12a - cbet2 * sbet1 * sq(somg12) / (1.0 - comg12)
        };
        let ssig12 = salp1.hypot(calp1);
        let csig12 = sbet1 * sbet2 + cbet1 * cbet2 * comg12;
        if shortline && ssig12 < self.etol2 {
            salp2 = cbet1 * somg12;
            calp2 = sbet12
                - cbet1
                    * sbet2
                    * if comg12 >= 0.0 {
                        sq(somg12) / (1.0 + comg12)
                    } else {
                        1.0 - comg12
                    };
            (salp2, calp2) = norm2(salp2, calp2);
            sig12 = ssig12.atan2(csig12);
        } else if self.n.abs() >= 0.1
            || csig12 >= 0.0
            || ssig12 >= 6.0 * self.n.abs() * std::f64::consts::PI * sq(cbet1)
        {
            // Nothing to do: the zeroth-order spherical approximation is good enough.
        } else {
            // Near-antipodal: solve the astroid problem. Oblate branch only.
            let lam12x = (-slam12).atan2(-clam12);
            let k2 = sq(sbet1) * self.ep2;
            let eps = k2 / (2.0 * (1.0 + (1.0 + k2).sqrt()) + k2);
            let lamscale = self.f * cbet1 * self.a3f(eps) * std::f64::consts::PI;
            let betscale = lamscale * cbet1;
            let x = lam12x / lamscale;
            let y = sbet12a / betscale;
            if y > -TOL1 && x > -1.0 - xthresh() {
                salp1 = 1f64.min(-x);
                calp1 = -(1.0 - sq(salp1)).sqrt();
            } else {
                let k = astroid(x, y);
                let omg12a = lamscale * (-x * k / (1.0 + k));
                somg12 = omg12a.sin();
                comg12 = -omg12a.cos();
                salp1 = cbet2 * somg12;
                calp1 = sbet12a - cbet2 * sbet1 * sq(somg12) / (1.0 - comg12);
            }
        }
        if salp1 > 0.0 {
            (salp1, calp1) = norm2(salp1, calp1);
        } else {
            salp1 = 1.0;
            calp1 = 0.0;
        }
        (sig12, salp1, calp1, salp2, calp2, dnm)
    }

    /// The reference's `_Lambda12`: the longitude difference reached by leaving point 1
    /// at azimuth `alp1`, plus what the Newton step and the area term need.
    #[allow(clippy::too_many_arguments)]
    fn lambda12(
        &self,
        (sbet1, cbet1, dn1): (f64, f64, f64),
        (sbet2, cbet2, dn2): (f64, f64, f64),
        salp1: f64,
        mut calp1: f64,
        slam120: f64,
        clam120: f64,
        diffp: bool,
        c1a: &mut [f64; N + 1],
        c2a: &mut [f64; N + 1],
        c3a: &mut [f64; N],
    ) -> Lambda {
        if sbet1 == 0.0 && calp1 == 0.0 {
            calp1 = -tiny();
        }
        let salp0 = salp1 * cbet1;
        let calp0 = calp1.hypot(salp1 * sbet1);
        let somg1 = salp0 * sbet1;
        let comg1 = calp1 * cbet1;
        let (ssig1, csig1) = norm2(sbet1, calp1 * cbet1);
        let salp2 = if cbet2 != cbet1 { salp0 / cbet2 } else { salp1 };
        let calp2 = if cbet2 != cbet1 || sbet2.abs() != -sbet1 {
            (sq(calp1 * cbet1)
                + if cbet1 < -sbet1 {
                    (cbet2 - cbet1) * (cbet1 + cbet2)
                } else {
                    (sbet1 - sbet2) * (sbet1 + sbet2)
                })
            .sqrt()
                / cbet2
        } else {
            calp1.abs()
        };
        let somg2 = salp0 * sbet2;
        let comg2 = calp2 * cbet2;
        let (ssig2, csig2) = norm2(sbet2, calp2 * cbet2);
        let sig12 =
            ((0f64.max(csig1 * ssig2 - ssig1 * csig2)) + 0.0).atan2(csig1 * csig2 + ssig1 * ssig2);
        let somg12 = 0f64.max(comg1 * somg2 - somg1 * comg2) + 0.0;
        let comg12 = comg1 * comg2 + somg1 * somg2;
        let eta = (somg12 * clam120 - comg12 * slam120).atan2(comg12 * clam120 + somg12 * slam120);
        let k2 = sq(calp0) * self.ep2;
        let eps = k2 / (2.0 * (1.0 + (1.0 + k2).sqrt()) + k2);
        self.c3f(eps, c3a);
        let b312 =
            sin_cos_series(true, ssig2, csig2, c3a) - sin_cos_series(true, ssig1, csig1, c3a);
        let domg12 = -self.f * self.a3f(eps) * salp0 * (sig12 + b312);
        let lam12 = eta + domg12;
        let dlam12 = if diffp {
            if calp2 == 0.0 {
                -2.0 * self.f1 * dn1 / sbet1
            } else {
                let (_, m12b, _) = self.lengths(
                    eps,
                    sig12,
                    (ssig1, csig1, dn1),
                    (ssig2, csig2, dn2),
                    false,
                    true,
                    c1a,
                    c2a,
                );
                m12b * self.f1 / (calp2 * cbet2)
            }
        } else {
            f64::NAN
        };
        Lambda {
            lam12,
            salp2,
            calp2,
            sig12,
            ssig1,
            csig1,
            ssig2,
            csig2,
            eps,
            domg12,
            dlam12,
        }
    }

    /// Solve the inverse problem between `(lat1, lon1)` and `(lat2, lon2)` in degrees.
    /// Returns the distance `s12` in metres and the area term `S12` in square metres
    /// (the area between the geodesic and the equator, which `polygon_area` sums).
    #[allow(clippy::too_many_lines)]
    fn inverse(&self, lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> (f64, f64) {
        let (lon12, lon12s) = ang_diff(lon1, lon2);
        let mut lonsign = 1f64.copysign(lon12);
        let lon12 = lonsign * lon12;
        let lon12s = lonsign * lon12s;
        let lam12 = lon12.to_radians();
        let (slam12, clam12) = sincosde(lon12, lon12s);
        let lon12s = (180.0 - lon12) - lon12s;

        let mut lat1 = ang_round(lat_fix(lat1));
        let mut lat2 = ang_round(lat_fix(lat2));
        let swapp = if lat1.abs() < lat2.abs() || lat2.is_nan() {
            -1.0
        } else {
            1.0
        };
        if swapp < 0.0 {
            lonsign *= -1.0;
            std::mem::swap(&mut lat1, &mut lat2);
        }
        let latsign = 1f64.copysign(-lat1);
        lat1 *= latsign;
        lat2 *= latsign;

        let (mut sbet1, mut cbet1) = sincosd(lat1);
        sbet1 *= self.f1;
        (sbet1, cbet1) = norm2(sbet1, cbet1);
        cbet1 = cbet1.max(tiny());
        let (mut sbet2, mut cbet2) = sincosd(lat2);
        sbet2 *= self.f1;
        (sbet2, cbet2) = norm2(sbet2, cbet2);
        cbet2 = cbet2.max(tiny());
        if cbet1 < -sbet1 {
            if cbet2 == cbet1 {
                sbet2 = sbet1.copysign(sbet2);
            }
        } else if sbet2.abs() == -sbet1 {
            cbet2 = cbet1;
        }
        let dn1 = (1.0 + self.ep2 * sq(sbet1)).sqrt();
        let dn2 = (1.0 + self.ep2 * sq(sbet2)).sqrt();

        let mut c1a = [0.0; N + 1];
        let mut c2a = [0.0; N + 1];
        let mut c3a = [0.0; N];

        let mut s12x = f64::NAN;
        let (mut salp1, mut calp1, mut salp2, mut calp2) = (0.0, 0.0, 0.0, 0.0);
        let mut meridian = lat1 == -90.0 || slam12 == 0.0;
        if meridian {
            calp1 = clam12;
            salp1 = slam12;
            calp2 = 1.0;
            salp2 = 0.0;
            let (ssig1, csig1) = (sbet1, calp1 * cbet1);
            let (ssig2, csig2) = (sbet2, calp2 * cbet2);
            let sig12 = ((0f64.max(csig1 * ssig2 - ssig1 * csig2)) + 0.0)
                .atan2(csig1 * csig2 + ssig1 * ssig2);
            let (s, m12x, _) = self.lengths(
                self.n,
                sig12,
                (ssig1, csig1, dn1),
                (ssig2, csig2, dn2),
                true,
                true,
                &mut c1a,
                &mut c2a,
            );
            s12x = s;
            // `m12 < 0` means the meridian is not the shortest path (the points are too
            // close to antipodal), so fall through to the general solution.
            if sig12 < tol2() || m12x >= 0.0 {
                if sig12 < 3.0 * tiny() || (sig12 < TOL0 && (s12x < 0.0 || m12x < 0.0)) {
                    s12x = 0.0;
                }
                s12x *= self.b;
            } else {
                meridian = false;
            }
        }

        let (mut somg12, mut comg12, mut omg12) = (2.0, 0.0, 0.0);
        if !meridian && sbet1 == 0.0 && (self.f <= 0.0 || lon12s >= self.f * 180.0) {
            // Along the equator.
            calp1 = 0.0;
            calp2 = 0.0;
            salp1 = 1.0;
            salp2 = 1.0;
            s12x = self.a * lam12;
            omg12 = lam12 / self.f1;
        } else if !meridian {
            let (sig12, s1, c1, s2, c2, dnm) =
                self.inverse_start((sbet1, cbet1), (sbet2, cbet2), lam12, slam12, clam12);
            salp1 = s1;
            calp1 = c1;
            if sig12 >= 0.0 {
                salp2 = s2;
                calp2 = c2;
                s12x = sig12 * self.b * dnm;
                omg12 = lam12 / (self.f1 * dnm);
            } else {
                let mut numit = 0u32;
                let (mut tripn, mut tripb) = (false, false);
                let (mut salp1a, mut calp1a) = (tiny(), 1.0);
                let (mut salp1b, mut calp1b) = (tiny(), -1.0);
                let mut lam;
                loop {
                    lam = self.lambda12(
                        (sbet1, cbet1, dn1),
                        (sbet2, cbet2, dn2),
                        salp1,
                        calp1,
                        slam12,
                        clam12,
                        numit < MAXIT1,
                        &mut c1a,
                        &mut c2a,
                        &mut c3a,
                    );
                    let v = lam.lam12;
                    if tripb
                        || v.abs() < if tripn { 8.0 } else { 1.0 } * TOL0
                        || v.is_nan()
                        || numit == MAXIT2
                    {
                        break;
                    }
                    if v > 0.0 && (numit > MAXIT1 || calp1 / salp1 > calp1b / salp1b) {
                        salp1b = salp1;
                        calp1b = calp1;
                    } else if v < 0.0 && (numit > MAXIT1 || calp1 / salp1 < calp1a / salp1a) {
                        salp1a = salp1;
                        calp1a = calp1;
                    }
                    numit += 1;
                    if numit < MAXIT1 && lam.dlam12 > 0.0 {
                        let dalp1 = -v / lam.dlam12;
                        if dalp1.abs() < std::f64::consts::PI {
                            let (sdalp1, cdalp1) = dalp1.sin_cos();
                            let nsalp1 = salp1 * cdalp1 + calp1 * sdalp1;
                            if nsalp1 > 0.0 {
                                calp1 = calp1 * cdalp1 - salp1 * sdalp1;
                                salp1 = nsalp1;
                                (salp1, calp1) = norm2(salp1, calp1);
                                tripn = v.abs() <= 16.0 * TOL0;
                                continue;
                            }
                        }
                    }
                    salp1 = (salp1a + salp1b) / 2.0;
                    calp1 = (calp1a + calp1b) / 2.0;
                    (salp1, calp1) = norm2(salp1, calp1);
                    tripn = false;
                    tripb = (salp1a - salp1).abs() + (calp1a - calp1) < TOLB
                        || (salp1 - salp1b).abs() + (calp1 - calp1b) < TOLB;
                }
                salp2 = lam.salp2;
                calp2 = lam.calp2;
                let (s, _, _) = self.lengths(
                    lam.eps,
                    lam.sig12,
                    (lam.ssig1, lam.csig1, dn1),
                    (lam.ssig2, lam.csig2, dn2),
                    true,
                    false,
                    &mut c1a,
                    &mut c2a,
                );
                s12x = s * self.b;
                let (sdomg12, cdomg12) = lam.domg12.sin_cos();
                somg12 = slam12 * cdomg12 - clam12 * sdomg12;
                comg12 = clam12 * cdomg12 + slam12 * sdomg12;
            }
        }
        let s12 = 0.0 + s12x;

        // The area term.
        let salp0 = salp1 * cbet1;
        let calp0 = calp1.hypot(salp1 * sbet1);
        let mut area = if calp0 != 0.0 && salp0 != 0.0 {
            let (ssig1, csig1) = norm2(sbet1, calp1 * cbet1);
            let (ssig2, csig2) = norm2(sbet2, calp2 * cbet2);
            let k2 = sq(calp0) * self.ep2;
            let eps = k2 / (2.0 * (1.0 + (1.0 + k2).sqrt()) + k2);
            let a4 = sq(self.a) * calp0 * salp0 * self.e2;
            let mut c4a = [0.0; N];
            self.c4f(eps, &mut c4a);
            let b41 = sin_cos_series(false, ssig1, csig1, &c4a);
            let b42 = sin_cos_series(false, ssig2, csig2, &c4a);
            a4 * (b42 - b41)
        } else {
            0.0
        };
        if !meridian && somg12 == 2.0 {
            somg12 = omg12.sin();
            comg12 = omg12.cos();
        }
        let alp12 = if !meridian && comg12 > COS_3PI_4_ROUNDED && sbet2 - sbet1 < 1.75 {
            let domg12 = 1.0 + comg12;
            let dbet1 = 1.0 + cbet1;
            let dbet2 = 1.0 + cbet2;
            2.0 * (somg12 * (sbet1 * dbet2 + sbet2 * dbet1))
                .atan2(domg12 * (sbet1 * sbet2 + dbet1 * dbet2))
        } else {
            let mut salp12 = salp2 * calp1 - calp2 * salp1;
            let mut calp12 = calp2 * calp1 + salp2 * salp1;
            if salp12 == 0.0 && calp12 < 0.0 {
                salp12 = tiny() * calp1;
                calp12 = -1.0;
            }
            salp12.atan2(calp12)
        };
        area += self.c2 * alp12;
        area *= swapp * lonsign * latsign;
        (s12, area + 0.0)
    }
}

fn lat_fix(x: f64) -> f64 {
    if x.abs() > 90.0 {
        f64::NAN
    } else {
        x
    }
}

/// What one evaluation of `lambda12` hands back to the Newton loop.
struct Lambda {
    lam12: f64,
    salp2: f64,
    calp2: f64,
    sig12: f64,
    ssig1: f64,
    csig1: f64,
    ssig2: f64,
    csig2: f64,
    eps: f64,
    domg12: f64,
    dlam12: f64,
}

fn wgs84() -> &'static Geodesic {
    static G: OnceLock<Geodesic> = OnceLock::new();
    G.get_or_init(|| Geodesic::new(WGS84_A, WGS84_F))
}

/// The ellipsoidal (WGS 84) distance in metres between two lon/lat positions.
///
/// Defined for every pair of valid positions, antipodal ones included. The caller is
/// responsible for the positions being on the globe; `proj::geodesy` checks that.
#[must_use]
pub fn distance(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
    wgs84().inverse(lat1, lon1, lat2, lon2).0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(got: f64, want: f64) -> f64 {
        ((got - want) / want).abs()
    }

    #[test]
    fn matches_the_published_geographiclib_example() {
        // GeographicLib's own documented example (Wellington to Salamanca):
        // Geodesic.WGS84.Inverse(-41.32, 174.81, 40.96, -5.50)['s12'] == 19959679.26735382
        let s = distance(174.81, -41.32, -5.50, 40.96);
        assert!(rel(s, 19_959_679.267_353_82) < 1e-12, "{s}");
    }

    #[test]
    fn antipodal_and_near_antipodal_pairs_have_a_distance() {
        // The pairs Vincenty's iteration does not converge for.
        let equator = distance(0.0, 0.0, 180.0, 0.0);
        // Half the meridian ellipse: the shortest path between equatorial antipodes
        // runs over a pole, 20003931.4586 m on WGS 84.
        assert!(rel(equator, 20_003_931.458_6) < 1e-9, "{equator}");
        let near = distance(0.0, 0.0, 179.9999, 0.0);
        assert!(near.is_finite() && near > 19_990_000.0, "{near}");
        let poles = distance(0.0, 90.0, 0.0, -90.0);
        assert!(rel(poles, 20_003_931.458_6) < 1e-9, "{poles}");
    }

    #[test]
    fn short_and_zero_distances_are_exact() {
        assert_eq!(distance(1.0, 2.0, 1.0, 2.0), 0.0);
        // One degree of longitude on the equator is a * pi / 180 exactly.
        let d = distance(0.0, 0.0, 1.0, 0.0);
        assert!(
            rel(d, WGS84_A * std::f64::consts::PI / 180.0) < 1e-14,
            "{d}"
        );
    }
}
