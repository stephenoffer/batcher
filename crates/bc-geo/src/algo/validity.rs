//! OGC validity — and, more usefully, *why* a geometry is invalid.
//!
//! Real geometry columns are full of invalid polygons: self-intersecting rings from a
//! digitizing error, holes that poke outside their shell, rings with two vertices.
//! Every areal predicate silently produces nonsense on those, so the ability to find
//! them is not a nicety — it is the difference between a spatial join that is wrong and
//! one that is wrong *and undetected*.
//!
//! `validity_reason` therefore returns a sentence naming the failure and, where it is a
//! single location, the position it happens at. A boolean alone tells you a row is bad;
//! a reason tells you which vertex to fix.

use crate::algo::primitive::{on_segment, segments_cross_properly, segments_intersect, PointRing};
use crate::algo::relate::point_in_polygon;
use crate::types::{is_closed, Coord, Geometry, LineString, Polygon};
use crate::Geom;

/// True when the chain's first and last positions coincide.
#[must_use]
pub fn line_is_closed(g: &Geometry) -> bool {
    match g {
        Geometry::LineString(l) => l.len() >= 2 && is_closed(l),
        Geometry::MultiLineString(ls) => {
            !ls.is_empty() && ls.iter().all(|l| l.len() >= 2 && is_closed(l))
        }
        // A polygon's rings are closed by definition, so PostGIS reports true.
        Geometry::Polygon(_) | Geometry::MultiPolygon(_) => true,
        _ => false,
    }
}

/// True when the chain is closed and does not cross itself — an OGC linear ring.
#[must_use]
pub fn is_ring(g: &Geometry) -> bool {
    match g {
        Geometry::LineString(l) => l.len() >= 4 && is_closed(l) && !self_intersects(l),
        _ => false,
    }
}

/// True when the geometry has no anomalous self-intersection.
///
/// For a chain that means it does not cross or touch itself except at a closing
/// endpoint; for a point set it means no duplicates. Areal geometries are simple by
/// definition once they are valid, which is why `ST_IsSimple` on a polygon is not a
/// second validity check.
pub fn is_simple(g: &Geometry) -> bool {
    match g {
        Geometry::LineString(l) => !self_intersects(l),
        Geometry::MultiLineString(ls) => {
            ls.iter().all(|l| !self_intersects(l))
                && (0..ls.len())
                    .all(|i| ((i + 1)..ls.len()).all(|j| !chains_meet_improperly(&ls[i], &ls[j])))
        }
        Geometry::MultiPoint(ps) => {
            let pts: Vec<Coord> = ps.iter().flatten().copied().collect();
            !pts.iter()
                .enumerate()
                .any(|(i, a)| pts.iter().skip(i + 1).any(|b| a.x == b.x && a.y == b.y))
        }
        Geometry::GeometryCollection(gs) => gs.iter().all(is_simple),
        _ => true,
    }
}

/// True when two *different* chains of a MULTILINESTRING meet anywhere other than at a
/// position that is a boundary point of both.
///
/// OGC simplicity is a property of the whole collection, not of its members one at a
/// time: two perfectly simple chains that cross each other make the multi-line
/// non-simple. Checking members in isolation reports such a collection as simple, which
/// is the same answer it gives for two disjoint chains.
///
/// A closed chain has no boundary at all, so *any* contact with another member is
/// improper for it.
fn chains_meet_improperly(a: &LineString, b: &LineString) -> bool {
    if a.len() < 2 || b.len() < 2 {
        return false;
    }
    let ends_a: &[Coord] = if is_closed(a) {
        &[]
    } else {
        &[a[0], a[a.len() - 1]]
    };
    let ends_b: &[Coord] = if is_closed(b) {
        &[]
    } else {
        &[b[0], b[b.len() - 1]]
    };
    let is_end = |p: Coord, ends: &[Coord]| ends.iter().any(|e| e.x == p.x && e.y == p.y);
    for s in a.windows(2) {
        for t in b.windows(2) {
            if !segments_intersect(s[0], s[1], t[0], t[1]) {
                continue;
            }
            // A transversal crossing is interior to both segments, so it can never be a
            // chain endpoint. Decided from orientations, since the crossing point itself
            // is not exactly representable.
            if segments_cross_properly(s[0], s[1], t[0], t[1]) {
                return true;
            }
            // Otherwise the contact is at one or more of the four vertices. Every shared
            // position must be a boundary point of both chains; a collinear overlap
            // contributes two such positions, at least one of which is interior.
            for p in [s[0], s[1], t[0], t[1]] {
                if on_segment(p, s[0], s[1])
                    && on_segment(p, t[0], t[1])
                    && !(is_end(p, ends_a) && is_end(p, ends_b))
                {
                    return true;
                }
            }
        }
    }
    false
}

/// The chain with consecutive duplicate positions removed.
///
/// A repeated vertex carries no geometry: it contributes a zero-length segment, and the
/// two real segments on either side of it are still *adjacent*. Leaving it in makes them
/// look like a non-adjacent pair that shares a point, which is the signature of a genuine
/// self-intersection — so `LINESTRING(0 0, 1 1, 1 1, 2 2)` was reported non-simple while
/// GEOS, PostGIS and DuckDB all call it simple.
fn without_repeated_positions(l: &LineString) -> Vec<Coord> {
    let mut out: Vec<Coord> = Vec::with_capacity(l.len());
    for c in l {
        if out.last().is_none_or(|p: &Coord| p.x != c.x || p.y != c.y) {
            out.push(*c);
        }
    }
    out
}

/// True when the chain meets itself anywhere other than at a shared vertex it is
/// entitled to share.
///
/// Two exclusions are legitimate and both are narrower than they look:
///
/// * **Adjacent segments** share their common vertex, which is proper — but only that
///   one point. When the chain doubles back along the segment it just drew
///   (`LINESTRING(0 0, 2 2, 1 1)`) the two adjacent segments *overlap*, which is a real
///   self-intersection. Skipping every adjacent pair outright, as this used to, reported
///   every such retrace as simple; the overlap is now tested for directly.
/// * **A closed chain's first and last segments** share the closing position, and that
///   is what closure means.
///
/// The old `len < 4` early exit was the same mistake in a cheaper disguise: a three-point
/// chain is exactly the shortest one that can double back, so the one shape the guard
/// excluded is the one it most needed to see.
fn self_intersects(l: &LineString) -> bool {
    let pts = without_repeated_positions(l);
    let n = pts.len().saturating_sub(1); // segment count
    if n < 2 {
        return false;
    }
    let closed = pts[0].x == pts[n].x && pts[0].y == pts[n].y;
    // Non-adjacent pairs: any contact at all is improper.
    for i in 0..n {
        for j in (i + 2)..n {
            if closed && i == 0 && j == n - 1 {
                continue;
            }
            if segments_intersect(pts[i], pts[i + 1], pts[j], pts[j + 1]) {
                return true;
            }
        }
    }
    // Adjacent pairs: improper only when they share more than the common vertex, which
    // (after de-duplication) happens exactly when the far endpoint of the second segment
    // lies back on the first.
    let doubles_back = |a: usize, b: usize, c: usize| on_segment(pts[c], pts[a], pts[b]);
    for i in 0..n - 1 {
        if doubles_back(i, i + 1, i + 2) {
            return true;
        }
    }
    if closed && n >= 2 && doubles_back(n - 1, n, 1) {
        return true;
    }
    false
}

/// `None` when the geometry is valid, or the reason it is not.
///
/// The checks run shallowest-first so the reported reason is the most actionable one:
/// there is no point telling someone their hole is outside the shell when the shell is
/// not a closed ring.
pub fn validity_reason(g: &Geom) -> Option<String> {
    if g.coords().iter().any(|c| c.is_nan()) {
        return Some("geometry contains a non-finite coordinate".to_string());
    }
    reason_of(&g.geometry)
}

fn reason_of(g: &Geometry) -> Option<String> {
    match g {
        Geometry::Polygon(p) => polygon_reason(p),
        Geometry::MultiPolygon(ps) => {
            for (i, p) in ps.iter().enumerate() {
                if let Some(r) = polygon_reason(p) {
                    return Some(format!("polygon {}: {r}", i + 1));
                }
            }
            // Two members of a multipolygon may touch but not overlap.
            for i in 0..ps.len() {
                for j in (i + 1)..ps.len() {
                    if polygons_overlap(&ps[i], &ps[j]) {
                        return Some(format!(
                            "polygons {} and {} overlap; a MULTIPOLYGON's members may touch but not share interior area",
                            i + 1,
                            j + 1
                        ));
                    }
                }
            }
            None
        }
        Geometry::LineString(l) => {
            if !l.is_empty() && l.len() < 2 {
                Some("line has a single position".to_string())
            } else {
                None
            }
        }
        Geometry::MultiLineString(ls) => ls
            .iter()
            .position(|l| !l.is_empty() && l.len() < 2)
            .map(|i| format!("line {} has a single position", i + 1)),
        Geometry::GeometryCollection(gs) => gs.iter().find_map(reason_of),
        _ => None,
    }
}

fn ring_reason(ring: &LineString, what: &str) -> Option<String> {
    if ring.is_empty() {
        return None;
    }
    if ring.len() < 4 {
        return Some(format!(
            "{what} has {} positions; a ring needs at least 4 (3 distinct plus the closing repeat)",
            ring.len()
        ));
    }
    if !is_closed(ring) {
        return Some(format!(
            "{what} is not closed: it starts at ({}, {}) and ends at ({}, {})",
            ring[0].x,
            ring[0].y,
            ring[ring.len() - 1].x,
            ring[ring.len() - 1].y
        ));
    }
    if self_intersects(ring) {
        let at = first_self_intersection(ring);
        return Some(match at {
            Some(c) => format!("{what} self-intersects near ({}, {})", c.x, c.y),
            None => format!("{what} self-intersects"),
        });
    }
    None
}

fn first_self_intersection(l: &LineString) -> Option<Coord> {
    let closed = is_closed(l);
    let n = l.len() - 1;
    for i in 0..n {
        for j in (i + 2)..n {
            if closed && i == 0 && j == n - 1 {
                continue;
            }
            if let Some(p) =
                crate::algo::primitive::segment_intersection(l[i], l[i + 1], l[j], l[j + 1])
            {
                return Some(p);
            }
            if segments_intersect(l[i], l[i + 1], l[j], l[j + 1]) {
                return Some(l[j]);
            }
        }
    }
    None
}

fn polygon_reason(p: &Polygon) -> Option<String> {
    if p.exterior.is_empty() {
        // An empty polygon is valid; it just has nothing in it.
        return if p.interiors.iter().any(|r| !r.is_empty()) {
            Some("polygon has interior rings but no exterior ring".to_string())
        } else {
            None
        };
    }
    if let Some(r) = ring_reason(&p.exterior, "exterior ring") {
        return Some(r);
    }
    for (i, hole) in p.interiors.iter().enumerate() {
        if let Some(r) = ring_reason(hole, &format!("interior ring {}", i + 1)) {
            return Some(r);
        }
        let shell = Polygon {
            exterior: p.exterior.clone(),
            interiors: Vec::new(),
        };
        if let Some(c) = hole
            .iter()
            .find(|c| point_in_polygon(**c, &shell) == PointRing::Outside)
        {
            return Some(format!(
                "interior ring {} lies outside the exterior ring at ({}, {})",
                i + 1,
                c.x,
                c.y
            ));
        }
        for (j, other) in p.interiors.iter().enumerate().skip(i + 1) {
            let hole_poly = Polygon {
                exterior: hole.clone(),
                interiors: Vec::new(),
            };
            let other_poly = Polygon {
                exterior: other.clone(),
                interiors: Vec::new(),
            };
            if polygons_overlap(&hole_poly, &other_poly) {
                return Some(format!("interior rings {} and {} overlap", i + 1, j + 1));
            }
        }
    }
    None
}

/// True when two simple polygons share interior area.
///
/// Deliberately narrower than `predicate::overlaps`: it only has to distinguish
/// "shares area" from "touches", and answering it from the vertex containment plus a
/// crossing test avoids the mutual recursion a call into the predicate layer would
/// create (the predicates use validity to decide whether to trust their inputs).
fn polygons_overlap(a: &Polygon, b: &Polygon) -> bool {
    let inside = |p: Coord, poly: &Polygon| point_in_polygon(p, poly) == PointRing::Inside;
    if a.exterior.iter().any(|c| inside(*c, b)) || b.exterior.iter().any(|c| inside(*c, a)) {
        return true;
    }
    // Crossing boundaries that share no vertex: a crossing that is not merely a touch
    // means a strip of one lies inside the other.
    for s in a.exterior.windows(2) {
        for t in b.exterior.windows(2) {
            if segments_intersect(s[0], s[1], t[0], t[1])
                && !on_segment(s[0], t[0], t[1])
                && !on_segment(s[1], t[0], t[1])
                && !on_segment(t[0], s[0], s[1])
                && !on_segment(t[1], s[0], s[1])
            {
                return true;
            }
        }
    }
    false
}

/// True when the geometry satisfies OGC validity.
#[must_use]
pub fn is_valid(g: &Geom) -> bool {
    validity_reason(g).is_none()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::wkt::read_wkt;

    fn g(t: &str) -> Geom {
        read_wkt(t).expect(t)
    }

    #[test]
    fn a_bowtie_is_invalid_and_says_where() {
        let r = validity_reason(&g("POLYGON((0 0, 4 4, 4 0, 0 4, 0 0))")).unwrap();
        assert!(r.contains("self-intersects"), "{r}");
        assert!(r.contains('('), "the reason names a position: {r}");
    }

    #[test]
    fn an_unclosed_ring_names_both_ends() {
        let r = validity_reason(&g("POLYGON((0 0, 4 0, 4 4, 0 4))")).unwrap();
        assert!(r.contains("not closed"), "{r}");
    }

    #[test]
    fn a_hole_outside_the_shell_is_caught() {
        let r = validity_reason(&g(
            "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0), (10 10, 12 10, 12 12, 10 10))",
        ))
        .unwrap();
        assert!(r.contains("outside the exterior"), "{r}");
    }

    #[test]
    fn ordinary_geometries_are_valid() {
        for t in [
            "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))",
            "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 4 2, 4 4, 2 4, 2 2))",
            "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 0)), ((5 5, 6 5, 6 6, 5 5)))",
            "LINESTRING(0 0, 1 1, 2 0)",
            "POINT(1 2)",
            "GEOMETRYCOLLECTION EMPTY",
        ] {
            assert!(is_valid(&g(t)), "{t}: {:?}", validity_reason(&g(t)));
        }
    }

    #[test]
    fn overlapping_multipolygon_members_are_invalid_but_touching_ones_are_not() {
        assert!(!is_valid(&g(
            "MULTIPOLYGON(((0 0, 4 0, 4 4, 0 4, 0 0)), ((2 2, 6 2, 6 6, 2 6, 2 2)))"
        )));
        assert!(is_valid(&g(
            "MULTIPOLYGON(((0 0, 4 0, 4 4, 0 4, 0 0)), ((4 0, 8 0, 8 4, 4 4, 4 0)))"
        )));
    }

    #[test]
    fn simplicity_and_ringness_are_separate_questions() {
        let closed = g("LINESTRING(0 0, 4 0, 4 4, 0 0)");
        assert!(line_is_closed(&closed.geometry));
        assert!(is_ring(&closed.geometry));
        assert!(is_simple(&closed.geometry));
        let bowtie = g("LINESTRING(0 0, 4 4, 4 0, 0 4, 0 0)");
        assert!(line_is_closed(&bowtie.geometry));
        assert!(!is_ring(&bowtie.geometry), "closed but self-crossing");
        assert!(!is_simple(&bowtie.geometry));
        let open = g("LINESTRING(0 0, 1 1)");
        assert!(!line_is_closed(&open.geometry));
        assert!(is_simple(&open.geometry));
    }

    #[test]
    fn duplicate_points_make_a_multipoint_non_simple() {
        assert!(!is_simple(&g("MULTIPOINT((1 1), (1 1))").geometry));
        assert!(is_simple(&g("MULTIPOINT((1 1), (2 2))").geometry));
    }

    #[test]
    fn a_nan_coordinate_is_reported_before_anything_else() {
        let mut geom = g("POINT(1 2)");
        geom.geometry = Geometry::Point(Some(Coord::new(f64::NAN, 2.0)));
        assert!(validity_reason(&geom).unwrap().contains("non-finite"));
    }

    #[test]
    fn a_multilinestring_whose_members_cross_is_not_simple() {
        // Each chain is simple on its own; the collection is not, because they cross.
        let g = read_wkt(
            "MULTILINESTRING((-31.666 31.623, -23.108 -38.653), (-41.143 -19.156, 10.818 -22.207))",
        )
        .expect("wkt");
        assert!(!is_simple(&g.geometry));
    }

    #[test]
    fn multilinestring_members_may_meet_at_shared_endpoints() {
        let g = read_wkt("MULTILINESTRING((0 0, 5 5), (5 5, 10 0))").expect("wkt");
        assert!(is_simple(&g.geometry));
    }

    #[test]
    fn disjoint_multilinestring_members_stay_simple() {
        let g = read_wkt("MULTILINESTRING((0 0, 1 1), (10 10, 11 11))").expect("wkt");
        assert!(is_simple(&g.geometry));
    }

    /// Every expectation below is GEOS's, cross-checked against DuckDB's spatial
    /// extension. The two shapes at the top are the ones the previous implementation got
    /// wrong, in opposite directions, and neither was reachable by the tests above.
    #[test]
    fn simplicity_matches_geos_on_retraces_and_repeated_vertices() {
        // A chain that doubles back along the segment it just drew overlaps itself. The
        // adjacent-pair skip hid it, and the `len < 4` exit hid it again: three points is
        // the *shortest* chain that can retrace, so the guard excluded exactly the case
        // it needed to see. Reported simple; it is not.
        assert!(!is_simple(&g("LINESTRING(0 0, 2 2, 1 1)").geometry));
        assert!(!is_simple(&g("LINESTRING(0 0, 1 1, 0 0)").geometry));

        // A repeated vertex is not a self-intersection: it contributes a zero-length
        // segment, and the two real segments either side of it stay adjacent. Reported
        // non-simple; it is simple.
        assert!(is_simple(&g("LINESTRING(0 0, 1 1, 1 1, 2 2)").geometry));
        assert!(is_simple(&g("LINESTRING(0 0, 0 0)").geometry));

        // Unchanged verdicts, kept here so a fix in one direction cannot pay for itself
        // by breaking the other.
        assert!(is_simple(&g("LINESTRING(0 0, 1 1, 2 2)").geometry));
        assert!(!is_simple(&g("LINESTRING(0 0, 2 0, 1 1, 1 -1)").geometry));
        assert!(is_simple(
            &g("LINESTRING(0 0, 1 0, 1 1, 0 1, 0 0)").geometry
        ));
        // Collinear continuation is not a retrace: the far point is beyond the shared
        // vertex, not back inside the first segment.
        assert!(is_simple(&g("LINESTRING(0 0, 1 1, 3 3)").geometry));
    }

    #[test]
    fn a_repeated_vertex_does_not_stop_a_ring_being_one() {
        // The same de-duplication has to reach `is_ring`, which shares `self_intersects`.
        assert!(is_ring(&g("LINESTRING(0 0, 4 0, 4 4, 4 4, 0 0)").geometry));
    }
}
