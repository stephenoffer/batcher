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

use crate::algo::primitive::{on_segment, segments_intersect, PointRing};
use crate::algo::relate::point_in_polygon;
use crate::types::{is_closed, Coord, Geometry, LineString, Polygon};
use crate::Geom;

/// True when the chain's first and last positions coincide.
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
        Geometry::MultiLineString(ls) => ls.iter().all(|l| !self_intersects(l)),
        Geometry::MultiPoint(ps) => {
            let pts: Vec<Coord> = ps.iter().flatten().copied().collect();
            !pts.iter().enumerate().any(|(i, a)| {
                pts.iter()
                    .skip(i + 1)
                    .any(|b| a.x == b.x && a.y == b.y)
            })
        }
        Geometry::GeometryCollection(gs) => gs.iter().all(is_simple),
        _ => true,
    }
}

/// True when two non-adjacent segments of the chain meet.
///
/// Adjacent segments always share their common vertex, and a closed chain's first and
/// last segments share the closing one; neither is an anomaly, so both are excluded.
/// Anything else — including a chain that merely touches itself without crossing — is.
fn self_intersects(l: &LineString) -> bool {
    if l.len() < 4 {
        return false;
    }
    let closed = is_closed(l);
    let n = l.len() - 1;
    for i in 0..n {
        for j in (i + 1)..n {
            if j == i + 1 {
                continue;
            }
            if closed && i == 0 && j == n - 1 {
                continue;
            }
            if segments_intersect(l[i], l[i + 1], l[j], l[j + 1]) {
                return true;
            }
        }
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
        if let Some(c) = hole.iter().find(|c| point_in_polygon(**c, &shell) == PointRing::Outside)
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
                return Some(format!(
                    "interior rings {} and {} overlap",
                    i + 1,
                    j + 1
                ));
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
}
