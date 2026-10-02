//! Great-circle arcs: the edge model the geodesic distance functions measure against.
//!
//! A chain or a ring in longitude and latitude is a list of positions, and "the distance
//! to it" is only defined once something says what lies *between* two positions. On the
//! globe the natural answer is the shorter great-circle arc, which is what PostGIS
//! `geography`, BigQuery `GEOGRAPHY` and S2 all use, and what this module computes with.
//!
//! Everything here works on unit vectors, where an arc is the span of two vectors and
//! "the closest point of an arc" is a projection onto a plane. That keeps every function
//! closed-form: no iteration, no convergence threshold, and nothing that behaves
//! differently near the antimeridian, because a unit vector has no longitude seam.
//!
//! An arc between two antipodal positions is not determined (every meridian joins the
//! poles), and an arc of length zero is a point. Both are treated as the two endpoints
//! and nothing between them, which is the answer that never under-reports a distance.

use crate::types::Coord;

type V3 = [f64; 3];

/// Below this length a cross product is treated as zero: two positions a few
/// nanometres apart on the ground, or a position on the pole of an arc's plane.
const DEGENERATE: f64 = 1e-15;

fn unit(c: Coord) -> V3 {
    let (lon, lat) = (c.x.to_radians(), c.y.to_radians());
    let (so, co) = lon.sin_cos();
    let (sa, ca) = lat.sin_cos();
    [ca * co, ca * so, sa]
}

fn lonlat(v: V3) -> Coord {
    Coord::new(
        v[1].atan2(v[0]).to_degrees(),
        v[2].atan2(v[0].hypot(v[1])).to_degrees(),
    )
}

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

fn scaled(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// The central angle between two unit vectors, accurate at every separation. `acos` of
/// the dot product loses half its digits for nearly equal vectors, which is exactly the
/// short-distance case.
fn angle(a: V3, b: V3) -> f64 {
    norm(cross(a, b)).atan2(dot(a, b))
}

/// The unit normal of the arc's plane, or `None` for a degenerate arc.
fn arc_normal(a: V3, b: V3) -> Option<V3> {
    let n = cross(a, b);
    let len = norm(n);
    (len > DEGENERATE).then(|| scaled(n, 1.0 / len))
}

/// Whether the unit vector `c`, already on the arc's great circle, lies between `a` and
/// `b` on the shorter arc.
fn within(a: V3, b: V3, n: V3, c: V3) -> bool {
    dot(cross(a, c), n) >= 0.0 && dot(cross(c, b), n) >= 0.0
}

/// The position on the arc `a`→`b` nearest to `p` on the sphere, and its fraction of
/// the way along the arc (0 at `a`, 1 at `b`).
///
/// The nearest point of a great circle to `p` is `p` projected onto the circle's plane.
/// When that projection falls outside the arc, the nearest point of the arc is the
/// nearer endpoint. A degenerate arc, or a `p` on the pole of the arc's plane (where
/// every point of the circle is equally far), answers with an endpoint.
#[must_use]
pub fn closest_on_arc(p: Coord, a: Coord, b: Coord) -> (Coord, f64) {
    let (pv, av, bv) = (unit(p), unit(a), unit(b));
    let nearer = || {
        if dot(pv, av) >= dot(pv, bv) {
            (a, 0.0)
        } else {
            (b, 1.0)
        }
    };
    let Some(n) = arc_normal(av, bv) else {
        return nearer();
    };
    let off = dot(pv, n);
    let c = [pv[0] - n[0] * off, pv[1] - n[1] * off, pv[2] - n[2] * off];
    let len = norm(c);
    if len <= DEGENERATE {
        return nearer();
    }
    let c = scaled(c, 1.0 / len);
    if !within(av, bv, n, c) {
        return nearer();
    }
    let total = angle(av, bv);
    let t = if total > 0.0 {
        angle(av, c) / total
    } else {
        0.0
    };
    (lonlat(c), t.clamp(0.0, 1.0))
}

/// The position a fraction `t` of the way along the shorter arc `a`→`b`.
///
/// Spherical linear interpolation of the two unit vectors, so equal steps in `t` are
/// equal steps of arc length. A degenerate arc returns `a`.
#[must_use]
pub fn point_on_arc(a: Coord, b: Coord, t: f64) -> Coord {
    let (av, bv) = (unit(a), unit(b));
    let omega = angle(av, bv);
    let s = omega.sin();
    if s <= DEGENERATE {
        return a;
    }
    let (wa, wb) = (((1.0 - t) * omega).sin() / s, (t * omega).sin() / s);
    lonlat([
        av[0] * wa + bv[0] * wb,
        av[1] * wa + bv[1] * wb,
        av[2] * wa + bv[2] * wb,
    ])
}

/// Whether the shorter arcs `a`→`b` and `c`→`d` cross or touch.
///
/// Two distinct great circles meet at exactly two antipodal points, `±(n1 x n2)`. The
/// arcs cross when one of those two points lies on both. Testing the candidate points
/// directly, rather than the four side-of-plane signs alone, is what rules out the case
/// where each arc straddles the other's circle on opposite sides of the globe.
///
/// Arcs on the *same* great circle (overlapping collinear edges) are reported as not
/// crossing. Their nearest endpoints are then at distance zero from the other arc,
/// which the endpoint-to-arc scan already finds.
#[must_use]
pub fn arcs_cross(a: Coord, b: Coord, c: Coord, d: Coord) -> bool {
    let (av, bv, cv, dv) = (unit(a), unit(b), unit(c), unit(d));
    let (Some(n1), Some(n2)) = (arc_normal(av, bv), arc_normal(cv, dv)) else {
        return false;
    };
    let x = cross(n1, n2);
    let len = norm(x);
    if len <= DEGENERATE {
        return false;
    }
    let x = scaled(x, 1.0 / len);
    [x, scaled(x, -1.0)]
        .into_iter()
        .any(|p| within(av, bv, n1, p) && within(cv, dv, n2, p))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Coord, b: Coord) -> bool {
        (a.x - b.x).abs() < 1e-9 && (a.y - b.y).abs() < 1e-9
    }

    #[test]
    fn the_equator_is_its_own_arc() {
        // A point one degree north of the middle of an equatorial arc projects straight
        // down onto the equator, halfway along.
        let (c, t) = closest_on_arc(
            Coord::new(5.0, 1.0),
            Coord::new(0.0, 0.0),
            Coord::new(10.0, 0.0),
        );
        assert!(close(c, Coord::new(5.0, 0.0)), "{c:?}");
        assert!((t - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_projection_past_the_end_clamps_to_the_nearer_endpoint() {
        let (c, t) = closest_on_arc(
            Coord::new(20.0, 1.0),
            Coord::new(0.0, 0.0),
            Coord::new(10.0, 0.0),
        );
        assert!(close(c, Coord::new(10.0, 0.0)));
        assert_eq!(t, 1.0);
    }

    #[test]
    fn a_degenerate_arc_is_its_endpoints() {
        let (c, t) = closest_on_arc(
            Coord::new(1.0, 1.0),
            Coord::new(0.0, 0.0),
            Coord::new(0.0, 0.0),
        );
        assert!(close(c, Coord::new(0.0, 0.0)));
        assert_eq!(t, 0.0);
    }

    #[test]
    fn arcs_across_the_antimeridian_take_the_short_way() {
        // 179E to 179W along the equator is a two-degree arc through 180, not a
        // 358-degree one through Greenwich.
        let mid = point_on_arc(Coord::new(179.0, 0.0), Coord::new(-179.0, 0.0), 0.5);
        assert!((mid.x.abs() - 180.0).abs() < 1e-9, "{mid:?}");
        let (c, _) = closest_on_arc(
            Coord::new(180.0, 2.0),
            Coord::new(179.0, 0.0),
            Coord::new(-179.0, 0.0),
        );
        assert!(
            (c.x.abs() - 180.0).abs() < 1e-9 && c.y.abs() < 1e-9,
            "{c:?}"
        );
    }

    #[test]
    fn crossing_is_tested_on_the_arcs_not_the_circles() {
        let (a, b) = (Coord::new(-1.0, 0.0), Coord::new(1.0, 0.0));
        assert!(arcs_cross(
            a,
            b,
            Coord::new(0.0, -1.0),
            Coord::new(0.0, 1.0)
        ));
        // The same meridian, but the arc sits north of the equator: the circles meet,
        // the arcs do not.
        assert!(!arcs_cross(
            a,
            b,
            Coord::new(0.0, 1.0),
            Coord::new(0.0, 2.0)
        ));
        // Each arc straddles the other's circle, on opposite sides of the globe.
        assert!(!arcs_cross(
            a,
            b,
            Coord::new(180.0, -1.0),
            Coord::new(180.0, 1.0)
        ));
    }

    #[test]
    fn interpolation_hits_both_ends() {
        let (a, b) = (Coord::new(10.0, 20.0), Coord::new(-40.0, 60.0));
        assert!(close(point_on_arc(a, b, 0.0), a));
        assert!(close(point_on_arc(a, b, 1.0), b));
    }
}
