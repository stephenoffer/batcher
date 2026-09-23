//! Polygon area on the ellipsoid: GeographicLib's `PolygonArea`, reduced to a ring.
//!
//! Each edge contributes the area between its geodesic and the equator (the `S12` term
//! of the inverse problem); the sum, reduced modulo the ellipsoid's total area and
//! corrected for how many times the ring crosses the prime meridian, is the enclosed
//! area. Split from `mod.rs` only for size.

use super::math::{ang_diff, ang_normalize, remainder, sum};
use super::wgs84;

/// A double-double running sum, so a polygon's area does not lose its low digits to the
/// large per-edge area terms that cancel around the ring.
#[derive(Default, Clone, Copy)]
struct Accumulator {
    s: f64,
    t: f64,
}

impl Accumulator {
    fn add(&mut self, y: f64) {
        let (y, u) = sum(y, self.t);
        let (s, t) = sum(y, self.s);
        self.s = s;
        self.t = t;
        if self.s == 0.0 {
            self.s = u;
        } else {
            self.t += u;
        }
    }

    fn remainder(&mut self, y: f64) {
        self.s = remainder(self.s, y);
        self.add(0.0);
    }
}

/// How many times the edge from `lon1` to `lon2` crosses the prime meridian, signed.
fn transit(lon1: f64, lon2: f64) -> i64 {
    let (lon12, _) = ang_diff(lon1, lon2);
    let lon1 = ang_normalize(lon1);
    let lon2 = ang_normalize(lon2);
    if lon12 > 0.0 && ((lon1 < 0.0 && lon2 >= 0.0) || (lon1 > 0.0 && lon2 == 0.0)) {
        1
    } else if lon12 < 0.0 && lon2 < 0.0 && lon1 >= 0.0 {
        -1
    } else {
        0
    }
}

/// The ellipsoidal (WGS 84) area of a lon/lat ring in square metres, unsigned.
///
/// The ring is closed implicitly. Edges are geodesics, not straight lines in lon/lat —
/// the same definition PostGIS `geography`, DuckDB `ST_Area_Spheroid` and GeographicLib
/// use — and a ring that crosses the antimeridian is measured the short way across it.
/// Winding does not change the answer.
#[must_use]
pub fn ring_area(ring: &[(f64, f64)]) -> f64 {
    let g = wgs84();
    if ring.len() < 2 {
        return 0.0;
    }
    let mut acc = Accumulator::default();
    let mut crossings = 0i64;
    let n = ring.len();
    for i in 0..n {
        let (lon1, lat1) = ring[i];
        let (lon2, lat2) = ring[(i + 1) % n];
        let (_, s12_area) = g.inverse(lat1, lon1, lat2, lon2);
        acc.add(s12_area);
        crossings += transit(lon1, lon2);
    }
    let area0 = 4.0 * std::f64::consts::PI * g.c2;
    // The reference's `_areareduceA` with `reverse = false, sign = true`: reduce to the
    // smaller of the two regions the ring bounds, signed by traversal direction.
    acc.remainder(area0);
    if crossings & 1 == 1 {
        let adj = if acc.s < 0.0 { 1.0 } else { -1.0 } * area0 / 2.0;
        acc.add(adj);
    }
    acc.s = -acc.s;
    acc.t = -acc.t;
    if acc.s > area0 / 2.0 {
        acc.add(-area0);
    } else if acc.s <= -area0 / 2.0 {
        acc.add(area0);
    }
    (acc.s + 0.0).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(got: f64, want: f64) -> f64 {
        ((got - want) / want).abs()
    }

    #[test]
    fn area_is_independent_of_winding_and_of_the_antimeridian() {
        let square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)];
        let mut rev = square;
        rev.reverse();
        let a = ring_area(&square);
        // GeographicLib: a 1x1 degree cell at the equator is 12308778361.469 m^2.
        assert!(rel(a, 12_308_778_361.469) < 1e-9, "{a}");
        assert!(rel(ring_area(&rev), a) < 1e-12);
        // The same cell shifted across the antimeridian has the same area.
        let across = [
            (179.5, 0.0),
            (-179.5, 0.0),
            (-179.5, 1.0),
            (179.5, 1.0),
            (179.5, 0.0),
        ];
        assert!(rel(ring_area(&across), a) < 1e-9, "{}", ring_area(&across));
    }
}
