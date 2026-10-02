//! The nearest-point geodesic distance between two geometries, in metres.
//!
//! `st_distance_sphere`, `st_distance_spheroid` and `st_dwithin_sphere` all answer "how
//! far apart are these two shapes on the ground", and for anything with an edge that is
//! a question about the *edges*, not just the vertices. Two coastlines a kilometre
//! apart whose vertices are spaced ten kilometres along each are a kilometre apart, not
//! five.
//!
//! # What is guaranteed
//!
//! An edge is the shorter great-circle arc between its two positions, which is the edge
//! model PostGIS `geography`, BigQuery and S2 use (`bc_geo::proj::arc`). The distance is
//! the minimum over:
//!
//! * every pair of positions (the whole answer for two points, exactly as before);
//! * every position of one shape against every edge of the other, at the arc's nearest
//!   point.
//!
//! For two great-circle arcs that do not cross, the minimum distance between them is
//! attained at an endpoint of one of them, so those two sets cover every candidate. A
//! pair of arcs that *do* cross, and a pair of shapes that intersect in longitude and
//! latitude (which is how a point inside a polygon is recognised), are zero apart.
//!
//! On the sphere the arc's nearest point is closed-form and the answer is exact for the
//! model. On the ellipsoid the nearest point of an arc is not the spherical one, so the
//! spherical candidate is refined by a golden-section search along the arc on the
//! ellipsoidal distance. The ellipsoid's radius of curvature stays within 0.6% of the
//! mean radius the sphere uses, which bounds how far the two metrics can disagree and
//! lets every arc whose spherical distance cannot beat the current best by that margin
//! skip the refinement. The answer is never larger than the vertex-to-vertex minimum,
//! which is what these functions computed before arcs were considered.

use bc_geo::algo::predicate;
use bc_geo::proj::{arc, geodesy};
use bc_geo::{Coord, Geom};

use crate::ExprError;

/// A lower bound on the ratio of the ellipsoidal distance to the spherical one between
/// the same two positions.
///
/// The meridional radius of curvature at the equator, `b^2 / a`, is 0.99441 of the
/// 6,371,008.8 m mean radius and is the smallest radius of curvature on WGS 84. Mapping
/// the ellipsoidal geodesic onto the sphere at the same longitudes and latitudes
/// therefore stretches it by at most `1 / 0.99441`, and the spherical distance is no
/// longer than that mapped path, so the ellipsoidal distance between two positions a
/// spherical distance `d` apart is never less than `0.99441 * d`. Rounded down for margin.
const ELLIPSOID_SHORTFALL: f64 = 0.994;

/// Golden-section steps over one arc. Each shrinks the bracket by 0.618, so 60 steps
/// bring a 20,000 km arc down to well under a millimetre.
const REFINE_STEPS: usize = 60;

/// The smallest geodesic distance between two geometries.
///
/// `None` for an empty operand or for a position off the globe (NaN, a longitude of
/// 200), which the caller surfaces as a null row like every other row-local failure in
/// this family.
pub(super) fn nearest_geodesic(
    a: &Geom,
    b: &Geom,
    spheroid: bool,
) -> Result<Option<f64>, ExprError> {
    let (ca, cb) = (a.coords(), b.coords());
    if ca.is_empty() || cb.is_empty() {
        return Ok(None);
    }
    // Intersecting shapes are zero apart, and no scan of the boundaries can see that a
    // point lies inside a polygon.
    if predicate::intersects(a, b) {
        return Ok(Some(0.0));
    }
    let metric = |p: Coord, q: Coord| {
        if spheroid {
            geodesy::ellipsoidal_distance(p.x, p.y, q.x, q.y)
        } else {
            geodesy::haversine(p.x, p.y, q.x, q.y)
        }
    };
    let mut best = f64::INFINITY;
    for p in &ca {
        for q in &cb {
            match metric(*p, *q) {
                Ok(v) => best = best.min(v),
                Err(_) => return Ok(None),
            }
        }
    }
    let (ea, eb) = (edges(a), edges(b));
    if ea.is_empty() && eb.is_empty() {
        return Ok(best.is_finite().then_some(best));
    }
    for &(s, t) in &ea {
        for &(u, v) in &eb {
            if arc::arcs_cross(s, t, u, v) {
                return Ok(Some(0.0));
            }
        }
    }
    for (points, arcs) in [(&ca, &eb), (&cb, &ea)] {
        for &p in points {
            for &(s, t) in arcs {
                best = best.min(to_arc(p, s, t, spheroid, best));
            }
        }
    }
    Ok(best.is_finite().then_some(best))
}

/// Every edge of `g`'s chains and rings, skipping repeated positions.
fn edges(g: &Geom) -> Vec<(Coord, Coord)> {
    let mut out = Vec::new();
    for line in g.geometry.lines() {
        for w in line.windows(2) {
            if w[0] != w[1] {
                out.push((w[0], w[1]));
            }
        }
    }
    out
}

/// The distance from `p` to the arc `s`→`t`, or infinity when it cannot beat `best`.
///
/// Endpoint distances are already in `best`, so only an interior nearest point can
/// improve on it.
fn to_arc(p: Coord, s: Coord, t: Coord, spheroid: bool, best: f64) -> f64 {
    let (c, frac) = arc::closest_on_arc(p, s, t);
    if frac <= 0.0 || frac >= 1.0 {
        return f64::INFINITY;
    }
    let Ok(near) = geodesy::haversine(p.x, p.y, c.x, c.y) else {
        return f64::INFINITY;
    };
    if !spheroid {
        return near;
    }
    if near * ELLIPSOID_SHORTFALL >= best {
        return f64::INFINITY;
    }
    refine(p, s, t)
}

/// Golden-section search for the ellipsoidal distance from `p` to the arc `s`→`t`.
///
/// Along an arc shorter than half the globe the distance to a fixed point falls to one
/// minimum and rises again, so the search converges to it. The endpoints are compared
/// by the caller, which keeps the result an upper bound even where that shape fails.
fn refine(p: Coord, s: Coord, t: Coord) -> f64 {
    let ratio = (5f64.sqrt() - 1.0) / 2.0;
    let at = |x: f64| {
        let q = arc::point_on_arc(s, t, x);
        geodesy::ellipsoidal_distance(p.x, p.y, q.x, q.y).unwrap_or(f64::INFINITY)
    };
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    let mut x1 = hi - ratio * (hi - lo);
    let mut x2 = lo + ratio * (hi - lo);
    let (mut f1, mut f2) = (at(x1), at(x2));
    for _ in 0..REFINE_STEPS {
        if f1 <= f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - ratio * (hi - lo);
            f1 = at(x1);
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + ratio * (hi - lo);
            f2 = at(x2);
        }
    }
    f1.min(f2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bc_geo::codec::wkt::read_wkt;

    fn d(a: &str, b: &str, spheroid: bool) -> f64 {
        nearest_geodesic(&read_wkt(a).unwrap(), &read_wkt(b).unwrap(), spheroid)
            .unwrap()
            .unwrap()
    }

    #[test]
    fn a_point_beside_a_long_edge_measures_to_the_edge_not_its_vertices() {
        // One degree north of the middle of a ten-degree equatorial edge. The nearest
        // vertex is sqrt(26) degrees away; the edge is one degree of meridian away.
        let line = "LINESTRING(0 0, 10 0)";
        let sphere = d("POINT(5 1)", line, false);
        let expected = geodesy::EARTH_RADIUS_M * 1f64.to_radians();
        assert!((sphere - expected).abs() < 1e-6, "{sphere} vs {expected}");
        let spheroid = d("POINT(5 1)", line, true);
        let meridian = geodesy::ellipsoidal_distance(5.0, 0.0, 5.0, 1.0).unwrap();
        assert!(
            (spheroid - meridian).abs() < 1e-3,
            "{spheroid} vs {meridian}"
        );
        // The vertex-to-vertex answer this used to give was over five times larger.
        let vertex = geodesy::ellipsoidal_distance(5.0, 1.0, 0.0, 0.0).unwrap();
        assert!(vertex > 5.0 * spheroid);
    }

    #[test]
    fn a_point_outside_a_many_sided_polygon_measures_to_its_boundary() {
        // A 0.5-degree gap to the east side of a square whose vertices are all far
        // from the point.
        let square = "POLYGON((0 -5, 10 -5, 10 5, 0 5, 0 -5))";
        let got = d("POINT(10.5 0)", square, true);
        let want = geodesy::ellipsoidal_distance(10.5, 0.0, 10.0, 0.0).unwrap();
        assert!((got - want).abs() < 1e-3, "{got} vs {want}");
    }

    #[test]
    fn inside_and_crossing_are_zero() {
        let square = "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))";
        assert_eq!(d("POINT(5 5)", square, true), 0.0);
        assert_eq!(
            d("LINESTRING(-1 5, 11 5)", "LINESTRING(5 -1, 5 11)", false),
            0.0
        );
    }

    #[test]
    fn two_points_are_unchanged() {
        let got = d("POINT(0 0)", "POINT(1 1)", true);
        assert_eq!(
            got,
            geodesy::ellipsoidal_distance(0.0, 0.0, 1.0, 1.0).unwrap()
        );
    }

    #[test]
    fn meridian_edges_meet_where_the_meridians_converge() {
        // Two meridian segments one degree of longitude apart, offset so that no vertex
        // of one faces a vertex of the other. Meridians converge away from the equator,
        // so the nearest pair is an end of the short segment against the interior of the
        // long one: `asin(cos(1 deg) * sin(1 deg))` of arc, a little under one degree.
        let got = d("LINESTRING(0 -3, 0 3)", "LINESTRING(1 -1, 1 1)", false);
        let one = 1f64.to_radians();
        let want = geodesy::EARTH_RADIUS_M * (one.cos() * one.sin()).asin();
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }

    #[test]
    fn the_answer_never_exceeds_the_vertex_minimum() {
        let (a, b) = (
            "LINESTRING(0 0, 3 4, 7 1, 12 8)",
            "POLYGON((20 20, 25 21, 24 30, 20 20))",
        );
        let (ga, gb) = (read_wkt(a).unwrap(), read_wkt(b).unwrap());
        for spheroid in [false, true] {
            let got = d(a, b, spheroid);
            let mut vertex = f64::INFINITY;
            for p in ga.coords() {
                for q in gb.coords() {
                    let v = if spheroid {
                        geodesy::ellipsoidal_distance(p.x, p.y, q.x, q.y)
                    } else {
                        geodesy::haversine(p.x, p.y, q.x, q.y)
                    };
                    vertex = vertex.min(v.unwrap());
                }
            }
            assert!(got <= vertex, "{got} > {vertex}");
        }
    }
}
