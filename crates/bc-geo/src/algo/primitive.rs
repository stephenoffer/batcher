//! The handful of primitives every planar algorithm in the crate is built from.
//!
//! These are separated out because the correctness of `st_intersects`, `st_contains`,
//! `st_distance` and the boolean overlay all reduce to *the same four questions*:
//! which side of a line a point is on, whether two segments cross, how far a point is
//! from a segment, and whether a point is inside a ring. Implementing any of them
//! twice is how two predicates come to disagree about the same pair of geometries.
//!
//! Orientation uses an adaptive filter rather than a bare sign test. A plain
//! cross-product on doubles gets the sign wrong for nearly-collinear inputs, and
//! nearly-collinear is not exotic — it is what a shared polygon edge looks like after
//! a projection. When the floating-point result is not provably correct the fallback
//! recomputes the determinant in a two-product expansion, which is exact for the
//! magnitudes coordinate data actually holds.

use crate::types::Coord;

/// Which way the turn `a → b → c` bends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// Counter-clockwise (left turn).
    CounterClockwise,
    /// Clockwise (right turn).
    Clockwise,
    /// The three points are collinear.
    Collinear,
}

/// Twice the signed area of the triangle `a b c`, positive when counter-clockwise.
pub fn cross(a: Coord, b: Coord, c: Coord) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// The exact orientation of the turn `a → b → c`.
///
/// The error bound follows Shewchuk's adaptive predicate: the floating-point
/// determinant is trustworthy whenever its magnitude exceeds the accumulated rounding
/// error of the four products, and that covers essentially every non-degenerate input.
/// Only inside the bound is the slower exact path taken.
pub fn orientation(a: Coord, b: Coord, c: Coord) -> Orientation {
    let detleft = (b.x - a.x) * (c.y - a.y);
    let detright = (b.y - a.y) * (c.x - a.x);
    let det = detleft - detright;
    let sum = detleft.abs() + detright.abs();
    // 3 * 2^-52 is the standard first-order bound for this two-product difference.
    const EPS: f64 = 6.661338147750939e-16;
    if det.abs() > EPS * sum {
        return if det > 0.0 {
            Orientation::CounterClockwise
        } else {
            Orientation::Clockwise
        };
    }
    match exact_sign(a, b, c) {
        s if s > 0.0 => Orientation::CounterClockwise,
        s if s < 0.0 => Orientation::Clockwise,
        _ => Orientation::Collinear,
    }
}

/// The determinant recomputed with error-free products, for the near-degenerate case.
///
/// Each product is split into a high part and an exact residual (Dekker's two-product
/// via FMA-free splitting), so the difference is evaluated without the cancellation
/// that makes the plain expression unreliable near zero.
fn exact_sign(a: Coord, b: Coord, c: Coord) -> f64 {
    let (ax, ay) = (b.x - a.x, b.y - a.y);
    let (bx, by) = (c.x - a.x, c.y - a.y);
    let (p1, e1) = two_product(ax, by);
    let (p2, e2) = two_product(ay, bx);
    // Sum the four terms from smallest to largest so the residuals are not lost.
    ((e1 - e2) + p1) - p2
}

/// `a * b` as an exact (high, low) pair.
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    let err = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    (p, err)
}

/// Split a double into two 26-bit halves whose sum is exact.
fn split(a: f64) -> (f64, f64) {
    // 2^27 + 1, the Veltkamp splitter for the 53-bit double significand.
    const SPLITTER: f64 = 134_217_729.0;
    let c = SPLITTER * a;
    let hi = c - (c - a);
    (hi, a - hi)
}

/// The Euclidean distance between two positions (planar, ignoring z).
pub fn dist(a: Coord, b: Coord) -> f64 {
    (a.x - b.x).hypot(a.y - b.y)
}

/// The squared Euclidean distance — the comparison form, with no square root.
pub fn dist2(a: Coord, b: Coord) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy
}

/// The point on segment `ab` nearest to `p`, and the parameter `t ∈ [0,1]` at which
/// it sits. A degenerate segment (`a == b`) yields `a` at `t = 0`.
pub fn closest_on_segment(p: Coord, a: Coord, b: Coord) -> (Coord, f64) {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let len2 = dx * dx + dy * dy;
    if len2 == 0.0 {
        return (a, 0.0);
    }
    let t = (((p.x - a.x) * dx + (p.y - a.y) * dy) / len2).clamp(0.0, 1.0);
    (
        Coord {
            x: a.x + t * dx,
            y: a.y + t * dy,
            z: a.z + t * (b.z - a.z),
        },
        t,
    )
}

/// The distance from `p` to segment `ab`.
pub fn point_segment_distance(p: Coord, a: Coord, b: Coord) -> f64 {
    dist(p, closest_on_segment(p, a, b).0)
}

/// True when `p` lies on segment `ab` (collinear and within the bounding box).
pub fn on_segment(p: Coord, a: Coord, b: Coord) -> bool {
    orientation(a, b, p) == Orientation::Collinear
        && p.x >= a.x.min(b.x)
        && p.x <= a.x.max(b.x)
        && p.y >= a.y.min(b.y)
        && p.y <= a.y.max(b.y)
}

/// True when segments `p1p2` and `q1q2` share at least one point.
///
/// The four-orientation test plus the four collinear-overlap cases. Touching at an
/// endpoint counts, because OGC `intersects` is closed: two polygons sharing an edge
/// do intersect, and a version of this that treated touching as disjoint would make
/// every adjacent-parcel query return nothing.
pub fn segments_intersect(p1: Coord, p2: Coord, q1: Coord, q2: Coord) -> bool {
    let o1 = orientation(p1, p2, q1);
    let o2 = orientation(p1, p2, q2);
    let o3 = orientation(q1, q2, p1);
    let o4 = orientation(q1, q2, p2);
    if o1 != o2 && o3 != o4 {
        return true;
    }
    (o1 == Orientation::Collinear && on_segment(q1, p1, p2))
        || (o2 == Orientation::Collinear && on_segment(q2, p1, p2))
        || (o3 == Orientation::Collinear && on_segment(p1, q1, q2))
        || (o4 == Orientation::Collinear && on_segment(p2, q1, q2))
}

/// The intersection point of two segments, when they cross at exactly one point.
///
/// `None` for parallel, collinear, or non-intersecting segments — the overlay code
/// handles collinear overlap separately because it produces a segment, not a point.
pub fn segment_intersection(p1: Coord, p2: Coord, q1: Coord, q2: Coord) -> Option<Coord> {
    let r = (p2.x - p1.x, p2.y - p1.y);
    let s = (q2.x - q1.x, q2.y - q1.y);
    let denom = r.0 * s.1 - r.1 * s.0;
    if denom == 0.0 {
        return None;
    }
    let qp = (q1.x - p1.x, q1.y - p1.y);
    let t = (qp.0 * s.1 - qp.1 * s.0) / denom;
    let u = (qp.0 * r.1 - qp.1 * r.0) / denom;
    if !(0.0..=1.0).contains(&t) || !(0.0..=1.0).contains(&u) {
        return None;
    }
    Some(Coord {
        x: p1.x + t * r.0,
        y: p1.y + t * r.1,
        z: p1.z + t * (p2.z - p1.z),
    })
}

/// The smallest distance between two segments, 0 when they intersect.
pub fn segment_segment_distance(p1: Coord, p2: Coord, q1: Coord, q2: Coord) -> f64 {
    if segments_intersect(p1, p2, q1, q2) {
        return 0.0;
    }
    point_segment_distance(p1, q1, q2)
        .min(point_segment_distance(p2, q1, q2))
        .min(point_segment_distance(q1, p1, p2))
        .min(point_segment_distance(q2, p1, p2))
}

/// Where a point sits relative to a ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointRing {
    /// Strictly inside.
    Inside,
    /// Exactly on the ring.
    Boundary,
    /// Strictly outside.
    Outside,
}

/// Locate `p` against the closed ring `ring`.
///
/// Boundary is tested first and separately from the crossing count. A ray-cast alone
/// answers inside-or-outside and is *arbitrary* on the boundary — which is precisely
/// the case that distinguishes `contains` from `covers`, so collapsing it would make
/// the two predicates indistinguishable.
pub fn point_in_ring(p: Coord, ring: &[Coord]) -> PointRing {
    if ring.len() < 3 {
        return PointRing::Outside;
    }
    for w in ring.windows(2) {
        if on_segment(p, w[0], w[1]) {
            return PointRing::Boundary;
        }
    }
    // The ring may be given unclosed; the implicit closing segment still counts.
    let (first, last) = (ring[0], ring[ring.len() - 1]);
    if (first.x != last.x || first.y != last.y) && on_segment(p, last, first) {
        return PointRing::Boundary;
    }
    // Crossing-number ray cast along +x. The half-open `[yi, yj)` comparison counts a
    // vertex exactly once, which is what keeps a ray through a vertex from double-
    // counting and reporting a point inside as outside.
    let mut inside = false;
    let n = ring.len();
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (ring[i], ring[j]);
        if (a.y > p.y) != (b.y > p.y) {
            let x_at = (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x;
            if p.x < x_at {
                inside = !inside;
            }
        }
        j = i;
    }
    if inside {
        PointRing::Inside
    } else {
        PointRing::Outside
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(x: f64, y: f64) -> Coord {
        Coord::new(x, y)
    }

    #[test]
    fn orientation_is_right_on_the_easy_cases() {
        assert_eq!(
            orientation(c(0.0, 0.0), c(1.0, 0.0), c(0.0, 1.0)),
            Orientation::CounterClockwise
        );
        assert_eq!(
            orientation(c(0.0, 0.0), c(1.0, 0.0), c(0.0, -1.0)),
            Orientation::Clockwise
        );
        assert_eq!(
            orientation(c(0.0, 0.0), c(1.0, 1.0), c(2.0, 2.0)),
            Orientation::Collinear
        );
    }

    #[test]
    fn orientation_survives_the_near_degenerate_case_a_plain_sign_test_fails() {
        // Three points that are collinear in exact arithmetic but whose naive
        // cross product is a denormal of the wrong sign on many inputs.
        let a = c(0.5, 0.5);
        let b = c(12.0, 12.0);
        let p = c(24.0, 24.0);
        assert_eq!(orientation(a, b, p), Orientation::Collinear);
        // A point one ULP off the line must not be reported collinear.
        let off = c(24.0, 24.0 + f64::EPSILON * 32.0);
        assert_ne!(orientation(a, b, off), Orientation::Collinear);
    }

    #[test]
    fn touching_segments_intersect() {
        assert!(segments_intersect(
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(1.0, 0.0),
            c(2.0, 0.0)
        ));
        assert!(segments_intersect(
            c(0.0, 0.0),
            c(2.0, 2.0),
            c(0.0, 2.0),
            c(2.0, 0.0)
        ));
        assert!(!segments_intersect(
            c(0.0, 0.0),
            c(1.0, 0.0),
            c(0.0, 1.0),
            c(1.0, 1.0)
        ));
    }

    #[test]
    fn segment_intersection_finds_the_crossing_point() {
        let p = segment_intersection(c(0.0, 0.0), c(2.0, 2.0), c(0.0, 2.0), c(2.0, 0.0)).unwrap();
        assert!((p.x - 1.0).abs() < 1e-12 && (p.y - 1.0).abs() < 1e-12);
        // Collinear overlap has no single point.
        assert!(segment_intersection(c(0.0, 0.0), c(2.0, 0.0), c(1.0, 0.0), c(3.0, 0.0)).is_none());
    }

    #[test]
    fn point_in_ring_distinguishes_boundary_from_inside() {
        let ring = vec![
            c(0.0, 0.0),
            c(4.0, 0.0),
            c(4.0, 4.0),
            c(0.0, 4.0),
            c(0.0, 0.0),
        ];
        assert_eq!(point_in_ring(c(2.0, 2.0), &ring), PointRing::Inside);
        assert_eq!(point_in_ring(c(0.0, 2.0), &ring), PointRing::Boundary);
        assert_eq!(point_in_ring(c(4.0, 4.0), &ring), PointRing::Boundary);
        assert_eq!(point_in_ring(c(5.0, 2.0), &ring), PointRing::Outside);
    }

    #[test]
    fn a_ray_through_a_vertex_is_counted_once() {
        // A diamond whose left and right vertices sit exactly on the ray y = 0.
        let ring = vec![
            c(0.0, -1.0),
            c(1.0, 0.0),
            c(0.0, 1.0),
            c(-1.0, 0.0),
            c(0.0, -1.0),
        ];
        assert_eq!(point_in_ring(c(0.0, 0.0), &ring), PointRing::Inside);
        assert_eq!(point_in_ring(c(-2.0, 0.0), &ring), PointRing::Outside);
        assert_eq!(point_in_ring(c(2.0, 0.0), &ring), PointRing::Outside);
    }

    #[test]
    fn distances_agree_with_hand_computed_values() {
        assert_eq!(
            point_segment_distance(c(0.0, 3.0), c(0.0, 0.0), c(4.0, 0.0)),
            3.0
        );
        // Past the end of the segment: measured to the endpoint, not the infinite line.
        assert_eq!(
            point_segment_distance(c(-3.0, 0.0), c(0.0, 0.0), c(4.0, 0.0)),
            3.0
        );
        assert_eq!(
            segment_segment_distance(c(0.0, 0.0), c(1.0, 0.0), c(0.0, 2.0), c(1.0, 2.0)),
            2.0
        );
    }
}
