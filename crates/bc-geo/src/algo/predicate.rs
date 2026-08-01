//! The OGC spatial predicates.
//!
//! Every one of these is defined in the standard as a pattern over the DE-9IM matrix
//! of two geometries' interiors, boundaries and exteriors. This module computes the
//! answers directly from that definition rather than materializing the matrix, because
//! the pieces the definitions actually need — "do the interiors meet", "is every point
//! of B in A", "what dimension is the intersection" — are each cheaper to decide than
//! the full matrix, and the named predicates are what queries are written in.
//!
//! Two properties hold across the whole set and are what the tests pin:
//!
//! * **Closure.** Touching counts as intersecting. Two parcels sharing a fence line do
//!   intersect, and a predicate that said otherwise would make adjacency queries empty.
//! * **Duality.** `within(a, b) == contains(b, a)` and `covered_by(a, b) == covers(b, a)`,
//!   by construction rather than by two implementations that must be kept in agreement.
//!
//! The areal predicates are exact for polygonal input: `relate::segment_midpoints`
//! splits every segment at each crossing with the other geometry, so each piece is
//! wholly inside or wholly outside and one point decides it. No sampling tolerance is
//! involved, and no overlay is needed.

use crate::algo::primitive::{on_segment, segments_intersect, PointRing};
use crate::algo::relate::{
    interior_point, linear_parts, point_in_geometry, point_in_polygon, probe_points,
};
use crate::types::{Coord, Geometry};
use crate::Geom;

/// True when the two geometries share at least one point.
pub fn intersects(a: &Geom, b: &Geom) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    // The bounding-box reject is not an optimization detail — it is what makes a
    // spatial join affordable, and it is exact in the negative direction.
    match (a.bbox(), b.bbox()) {
        (Some(x), Some(y)) if !x.intersects(&y) => return false,
        _ => {}
    }
    let (ga, gb) = (&a.geometry, &b.geometry);
    // Any vertex of one lying on or in the other settles it.
    for c in ga.points().iter().chain(vertices(ga).iter()) {
        if point_in_geometry(*c, gb) != PointRing::Outside {
            return true;
        }
    }
    for c in gb.points().iter().chain(vertices(gb).iter()) {
        if point_in_geometry(*c, ga) != PointRing::Outside {
            return true;
        }
    }
    // Boundaries that cross without sharing a vertex.
    for la in ga.lines() {
        for lb in gb.lines() {
            for s in la.windows(2) {
                for t in lb.windows(2) {
                    if segments_intersect(s[0], s[1], t[0], t[1]) {
                        return true;
                    }
                }
            }
        }
    }
    // One wholly inside the other, sharing no boundary point at all.
    if let Some(p) = interior_point(ga) {
        if point_in_geometry(p, gb) != PointRing::Outside {
            return true;
        }
    }
    if let Some(p) = interior_point(gb) {
        if point_in_geometry(p, ga) != PointRing::Outside {
            return true;
        }
    }
    false
}

/// Every vertex of every chain of a geometry.
fn vertices(g: &Geometry) -> Vec<Coord> {
    let mut out = Vec::new();
    g.collect_coords(&mut out);
    out
}

/// True when the two geometries share no point at all.
pub fn disjoint(a: &Geom, b: &Geom) -> bool {
    !intersects(a, b)
}

/// True when every point of `b` lies in `a` (boundary included).
///
/// The difference from `contains` is exactly the boundary: a polygon covers a point on
/// its edge but does not contain it, and a polygon covers the polygon it shares an edge
/// with only if it also swallows its interior.
pub fn covers(a: &Geom, b: &Geom) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    match (a.bbox(), b.bbox()) {
        (Some(x), Some(y)) if !x.contains(&y) => return false,
        _ => {}
    }
    for p in probe_points(&b.geometry, &a.geometry) {
        if point_in_geometry(p, &a.geometry) == PointRing::Outside {
            return false;
        }
    }
    // A hole of `a` strictly inside `b` is a region of `b` that `a` does not cover, and
    // no probe point of `b` can see it: the hole is interior to `b`, so `b`'s own
    // segments never cross it. This is the one case sampling `b` alone gets wrong.
    for poly in a.geometry.polygons() {
        for hole in &poly.interiors {
            let ring_geom = Geometry::Polygon(crate::types::Polygon {
                exterior: hole.clone(),
                interiors: Vec::new(),
            });
            if let Some(p) = interior_point(&ring_geom) {
                if point_in_geometry(p, &b.geometry) == PointRing::Inside {
                    return false;
                }
            }
        }
    }
    true
}

/// True when `b` lies in `a` and touches at least `a`'s interior.
pub fn contains(a: &Geom, b: &Geom) -> bool {
    covers(a, b) && interiors_intersect(a, b)
}

/// True when `a` lies in `b` and touches at least `b`'s interior.
pub fn within(a: &Geom, b: &Geom) -> bool {
    contains(b, a)
}

/// True when every point of `a` lies in `b` (boundary included).
pub fn covered_by(a: &Geom, b: &Geom) -> bool {
    covers(b, a)
}

/// True when `p` is in the *interior* of `g`: strictly inside an areal part, or on a
/// chain but not at one of its endpoints, or equal to one of its positions.
///
/// A geometry's interior depends on its own dimension, not on the plane's — a line has
/// no area but it does have an interior, and every predicate that distinguishes
/// `touches` from `crosses` turns on that.
pub fn in_interior(p: Coord, g: &Geometry) -> bool {
    for poly in g.polygons() {
        if point_in_polygon(p, poly) == PointRing::Inside {
            return true;
        }
    }
    for l in linear_parts(g) {
        if l.len() < 2 {
            continue;
        }
        let endpoints = [l[0], l[l.len() - 1]];
        let is_endpoint = endpoints.iter().any(|e| e.x == p.x && e.y == p.y);
        if is_endpoint {
            continue;
        }
        if l.windows(2).any(|w| on_segment(p, w[0], w[1])) {
            return true;
        }
    }
    // A closed chain has no boundary, so every one of its points is interior — which is
    // why a ring touching another geometry at its own start vertex still "crosses" it.
    for l in linear_parts(g) {
        if l.len() >= 2 && crate::types::is_closed(l) && l.windows(2).any(|w| on_segment(p, w[0], w[1]))
        {
            return true;
        }
    }
    g.points().iter().any(|q| q.x == p.x && q.y == p.y)
}

/// Positions where the two geometries' linework meets: crossing points plus any vertex
/// of one lying on the other. The interior-versus-boundary tests need these because a
/// crossing point is by construction not a vertex of either input.
fn meeting_points(a: &Geometry, b: &Geometry) -> Vec<Coord> {
    let mut out = Vec::new();
    for la in a.lines() {
        for lb in b.lines() {
            for s in la.windows(2) {
                for t in lb.windows(2) {
                    if let Some(p) = crate::algo::primitive::segment_intersection(s[0], s[1], t[0], t[1])
                    {
                        out.push(p);
                    }
                }
            }
        }
    }
    for c in vertices(a) {
        if point_in_geometry(c, b) != PointRing::Outside {
            out.push(c);
        }
    }
    for c in vertices(b) {
        if point_in_geometry(c, a) != PointRing::Outside {
            out.push(c);
        }
    }
    out.extend(a.points().iter().copied());
    out.extend(b.points().iter().copied());
    out
}

/// True when the two geometries' interiors share at least one point.
///
/// Decided from a finite witness set, in two passes, because one pass is not enough and
/// the reason is subtle. Every probe point of a geometry lies *on* it, and an areal
/// geometry's boundary is not part of its interior — so two overlapping squares have no
/// witness that is in both interiors, even though their overlap is a whole square. Two
/// squares sharing an edge have exactly the same witnesses, and their interiors are
/// disjoint. The direct test cannot tell those apart.
///
/// The second pass is what separates them: when **both** geometries are areal, a point
/// on one's boundary that lies in the other's interior proves the interiors meet. Every
/// neighbourhood of a boundary point contains interior points of its own geometry, and
/// an areal interior is open in the plane, so it contains a whole neighbourhood of that
/// point and therefore some of them. A shared edge fails this — its points are on both
/// boundaries, never inside either — while a genuine overlap passes it.
///
/// Both operands must be areal for that argument to hold. A line's interior is a curve,
/// not an open set, so a polygon vertex landing exactly on a line proves nothing about
/// the polygon's interior; the direct pass already handles the linear cases, because a
/// line's probe points lie in its own interior rather than merely on its boundary.
pub fn interiors_intersect(a: &Geom, b: &Geom) -> bool {
    if !intersects(a, b) {
        return false;
    }
    let (ga, gb) = (&a.geometry, &b.geometry);
    let probe_a = probe_points(ga, gb);
    let probe_b = probe_points(gb, ga);
    let mut candidates = meeting_points(ga, gb);
    candidates.extend(probe_a.iter().copied());
    candidates.extend(probe_b.iter().copied());
    if let Some(p) = interior_point(ga) {
        candidates.push(p);
    }
    if let Some(p) = interior_point(gb) {
        candidates.push(p);
    }
    if candidates
        .iter()
        .any(|p| in_interior(*p, ga) && in_interior(*p, gb))
    {
        return true;
    }
    if dimension(ga) == 2 && dimension(gb) == 2 {
        return probe_a.iter().any(|p| in_interior(*p, gb))
            || probe_b.iter().any(|p| in_interior(*p, ga));
    }
    false
}

/// The topological dimension of the two interiors' intersection, or `None` when they
/// do not meet.
///
/// Determined by the operand dimensions rather than by constructing the intersection:
/// two areal interiors that meet at all meet in an area, an areal and a linear one meet
/// in a line, and two linear ones meet in a line only where a pair of their segments is
/// collinear and overlapping. That last case is the only one needing a scan, and it is
/// what separates `crosses` from `overlaps` for line pairs.
pub fn interior_intersection_dim(a: &Geom, b: &Geom) -> Option<i64> {
    if !interiors_intersect(a, b) {
        return None;
    }
    let da = dimension(&a.geometry);
    let db = dimension(&b.geometry);
    Some(match (da, db) {
        (2, 2) => 2,
        (2, 1) | (1, 2) => 1,
        (1, 1) if collinear_overlap(&a.geometry, &b.geometry) => 1,
        _ => 0,
    })
}

/// The highest topological dimension present in a geometry.
pub fn dimension(g: &Geometry) -> i64 {
    if !g.polygons().is_empty() {
        return 2;
    }
    if !linear_parts(g).is_empty() {
        return 1;
    }
    0
}

/// True when some segment of `a` runs along some segment of `b` for a positive length.
fn collinear_overlap(a: &Geometry, b: &Geometry) -> bool {
    for la in a.lines() {
        for lb in b.lines() {
            for s in la.windows(2) {
                for t in lb.windows(2) {
                    // Collinear and overlapping means each segment has an endpoint on
                    // the other, or one contains the other, with a shared extent.
                    let touch = [
                        on_segment(s[0], t[0], t[1]),
                        on_segment(s[1], t[0], t[1]),
                        on_segment(t[0], s[0], s[1]),
                        on_segment(t[1], s[0], s[1]),
                    ];
                    if touch.iter().filter(|x| **x).count() >= 2 {
                        // Two shared points on a line means a shared segment, unless
                        // both are the same degenerate position.
                        let pts: Vec<Coord> = [s[0], s[1], t[0], t[1]]
                            .into_iter()
                            .filter(|p| on_segment(*p, s[0], s[1]) && on_segment(*p, t[0], t[1]))
                            .collect();
                        if pts
                            .iter()
                            .any(|p| pts.iter().any(|q| p.x != q.x || p.y != q.y))
                        {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

/// True when the geometries meet but their interiors do not.
pub fn touches(a: &Geom, b: &Geom) -> bool {
    intersects(a, b) && !interiors_intersect(a, b)
}

/// True when the interiors meet in something of lower dimension than the operands, and
/// neither geometry covers the other.
pub fn crosses(a: &Geom, b: &Geom) -> bool {
    let Some(d) = interior_intersection_dim(a, b) else {
        return false;
    };
    let max_dim = dimension(&a.geometry).max(dimension(&b.geometry));
    d < max_dim && !covers(a, b) && !covers(b, a)
}

/// True when the geometries have the same dimension, their interiors meet in something
/// of that same dimension, and neither covers the other.
pub fn overlaps(a: &Geom, b: &Geom) -> bool {
    let da = dimension(&a.geometry);
    let db = dimension(&b.geometry);
    if da != db {
        return false;
    }
    let Some(d) = interior_intersection_dim(a, b) else {
        return false;
    };
    d == da && !covers(a, b) && !covers(b, a)
}

/// True when the two geometries occupy exactly the same set of points.
///
/// This is *topological* equality, not structural: the same square written clockwise
/// and counter-clockwise, or with an extra collinear vertex, is equal. `Geom`'s derived
/// `PartialEq` is the structural comparison, and the two are deliberately different
/// operations because `ST_Equals` and "is the same WKB" answer different questions.
pub fn geom_equals(a: &Geom, b: &Geom) -> bool {
    if a.is_empty() && b.is_empty() {
        return true;
    }
    covers(a, b) && covers(b, a)
}

/// True when the geometries are within `radius` of each other in planar units.
///
/// A negative radius is a caller error, not an empty result: it means the query is
/// asking a question with no answer, and silently returning false would hide it.
pub fn dwithin(a: &Geom, b: &Geom, radius: f64) -> Option<bool> {
    if radius < 0.0 || radius.is_nan() {
        return None;
    }
    if a.is_empty() || b.is_empty() {
        return Some(false);
    }
    // Reject on boxes first: expanding one box by the radius is a sound over-estimate,
    // so a pair rejected here can never be within the radius.
    if let (Some(x), Some(y)) = (a.bbox(), b.bbox()) {
        if x.distance(&y) > radius {
            return Some(false);
        }
    }
    Some(crate::algo::measure::distance(a, b).is_some_and(|d| d <= radius))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::wkt::read_wkt;

    fn g(t: &str) -> Geom {
        read_wkt(t).expect(t)
    }

    const SQUARE: &str = "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))";
    const RIGHT: &str = "POLYGON((4 0, 8 0, 8 4, 4 4, 4 0))";
    const OVERLAP: &str = "POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))";

    #[test]
    fn adjacent_polygons_intersect_and_touch_but_do_not_overlap() {
        let (a, b) = (g(SQUARE), g(RIGHT));
        assert!(intersects(&a, &b));
        assert!(touches(&a, &b));
        assert!(!overlaps(&a, &b));
        assert!(!disjoint(&a, &b));
    }

    #[test]
    fn overlapping_polygons_overlap_and_do_not_touch() {
        let (a, b) = (g(SQUARE), g(OVERLAP));
        assert!(overlaps(&a, &b));
        assert!(!touches(&a, &b));
        assert!(interiors_intersect(&a, &b));
    }

    #[test]
    fn containment_distinguishes_contains_from_covers_on_the_boundary() {
        let poly = g(SQUARE);
        let edge = g("POINT(0 2)");
        let inside = g("POINT(2 2)");
        assert!(covers(&poly, &edge));
        assert!(!contains(&poly, &edge));
        assert!(covers(&poly, &inside));
        assert!(contains(&poly, &inside));
        assert!(within(&inside, &poly));
        assert!(covered_by(&edge, &poly));
    }

    #[test]
    fn a_line_crossing_a_polygon_crosses_it() {
        let poly = g(SQUARE);
        let through = g("LINESTRING(-1 2, 5 2)");
        let inside = g("LINESTRING(1 1, 3 3)");
        assert!(crosses(&poly, &through));
        assert!(!crosses(&poly, &inside));
        assert!(contains(&poly, &inside));
    }

    #[test]
    fn crossing_lines_cross_and_overlapping_lines_overlap() {
        let a = g("LINESTRING(0 0, 4 4)");
        let x = g("LINESTRING(0 4, 4 0)");
        let along = g("LINESTRING(2 2, 6 6)");
        assert!(crosses(&a, &x));
        assert!(!overlaps(&a, &x));
        assert!(overlaps(&a, &along));
        assert!(!crosses(&a, &along));
    }

    #[test]
    fn a_hole_is_not_covered_even_though_the_shell_is() {
        let donut = g("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 7 3, 7 7, 3 7, 3 3))");
        let filled = g("POLYGON((1 1, 9 1, 9 9, 1 9, 1 1))");
        assert!(!covers(&donut, &filled), "the hole is a gap in the cover");
        assert!(!contains(&donut, &g("POINT(5 5)")));
        assert!(contains(&donut, &g("POINT(1 5)")));
    }

    #[test]
    fn equality_is_topological_not_structural() {
        let cw = g("POLYGON((0 0, 0 4, 4 4, 4 0, 0 0))");
        let ccw = g(SQUARE);
        let extra_vertex = g("POLYGON((0 0, 2 0, 4 0, 4 4, 0 4, 0 0))");
        assert!(geom_equals(&cw, &ccw));
        assert!(geom_equals(&ccw, &extra_vertex));
        assert_ne!(cw, ccw, "structurally different");
        assert!(!geom_equals(&ccw, &g(OVERLAP)));
    }

    #[test]
    fn dwithin_rejects_a_negative_radius_rather_than_answering_false() {
        let (a, b) = (g("POINT(0 0)"), g("POINT(3 4)"));
        assert_eq!(dwithin(&a, &b, 5.0), Some(true));
        assert_eq!(dwithin(&a, &b, 4.999), Some(false));
        assert_eq!(dwithin(&a, &b, -1.0), None);
    }

    #[test]
    fn duality_holds_by_construction() {
        let cases = [(SQUARE, "POINT(2 2)"), (SQUARE, OVERLAP), (SQUARE, RIGHT)];
        for (x, y) in cases {
            let (a, b) = (g(x), g(y));
            assert_eq!(within(&a, &b), contains(&b, &a));
            assert_eq!(covered_by(&a, &b), covers(&b, &a));
            assert_eq!(disjoint(&a, &b), !intersects(&a, &b));
        }
    }

    #[test]
    fn disjoint_geometries_agree_across_every_predicate() {
        let (a, b) = (g(SQUARE), g("POLYGON((20 20, 24 20, 24 24, 20 24, 20 20))"));
        assert!(disjoint(&a, &b));
        for p in [intersects, touches, crosses, overlaps, contains, within, covers] {
            assert!(!p(&a, &b), "a disjoint pair satisfies no positive predicate");
        }
    }

    #[test]
    fn empty_geometries_satisfy_nothing() {
        let e = g("POLYGON EMPTY");
        let a = g(SQUARE);
        assert!(!intersects(&a, &e));
        assert!(!covers(&a, &e));
        assert!(!contains(&a, &e));
        assert!(geom_equals(&e, &g("LINESTRING EMPTY")));
    }
}
