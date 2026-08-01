//! Locating a point against a geometry, and noding a line against one.
//!
//! Both are the machinery the predicates are written in terms of, and both exist to
//! answer the same question at two different granularities: *is this piece of geometry
//! inside, on the boundary of, or outside that one?*
//!
//! Noding is the part worth explaining. Asking whether a whole segment lies inside a
//! polygon by testing its endpoints is wrong — a segment can enter and leave through a
//! notch with both endpoints inside. Splitting the segment at every crossing with the
//! polygon's boundary fixes that: between two consecutive crossings the segment is
//! entirely inside or entirely outside, so its midpoint decides for the whole piece.
//! That reduction from "a segment" to "a point" is what lets every areal predicate be
//! written as a finite set of point-in-polygon tests.

use crate::algo::primitive::{point_in_ring, segment_intersection, PointRing};
use crate::types::{Coord, Geometry, Polygon};

/// Locate `p` against one polygon: inside the exterior and outside every hole.
///
/// A position on a hole's boundary is on the polygon's boundary, not outside it —
/// which is what makes `covers` accept a point sitting exactly on the edge of a hole
/// while `contains` rejects it.
pub fn point_in_polygon(p: Coord, poly: &Polygon) -> PointRing {
    match point_in_ring(p, &poly.exterior) {
        PointRing::Outside => PointRing::Outside,
        PointRing::Boundary => PointRing::Boundary,
        PointRing::Inside => {
            for hole in &poly.interiors {
                match point_in_ring(p, hole) {
                    PointRing::Inside => return PointRing::Outside,
                    PointRing::Boundary => return PointRing::Boundary,
                    PointRing::Outside => {}
                }
            }
            PointRing::Inside
        }
    }
}

/// Locate `p` against a whole geometry, taking the most-inside answer over its parts.
///
/// "Most inside" is the right fold because a geometry is the union of its parts: a
/// point inside any member is inside the union, and one on a member's boundary is on
/// the union's boundary only if no other member swallows it.
pub fn point_in_geometry(p: Coord, g: &Geometry) -> PointRing {
    let mut best = PointRing::Outside;
    for poly in g.polygons() {
        match point_in_polygon(p, poly) {
            PointRing::Inside => return PointRing::Inside,
            PointRing::Boundary => best = PointRing::Boundary,
            PointRing::Outside => {}
        }
    }
    if best == PointRing::Boundary {
        return best;
    }
    // Non-areal parts have no interior in the plane: every point of a line or a point
    // set is on that geometry's boundary, never inside it.
    for l in linear_parts(g) {
        if l.len() == 1 {
            if l[0].x == p.x && l[0].y == p.y {
                return PointRing::Boundary;
            }
            continue;
        }
        for w in l.windows(2) {
            if crate::algo::primitive::on_segment(p, w[0], w[1]) {
                return PointRing::Boundary;
            }
        }
    }
    for q in g.points() {
        if q.x == p.x && q.y == p.y {
            return PointRing::Boundary;
        }
    }
    PointRing::Outside
}

/// The chains of a geometry that are *not* polygon rings.
///
/// `Geometry::lines` deliberately includes rings, because the linear predicates test
/// against a polygon's boundary. Locating a point wants the opposite: a ring is
/// already accounted for by the areal test, and counting it again would report a point
/// strictly inside a polygon as being on its boundary.
pub fn linear_parts(g: &Geometry) -> Vec<&Vec<Coord>> {
    let mut out = Vec::new();
    fn walk<'a>(g: &'a Geometry, out: &mut Vec<&'a Vec<Coord>>) {
        match g {
            Geometry::LineString(l) => out.push(l),
            Geometry::MultiLineString(ls) => out.extend(ls.iter()),
            Geometry::GeometryCollection(gs) => gs.iter().for_each(|c| walk(c, out)),
            _ => {}
        }
    }
    walk(g, &mut out);
    out
}

/// Split segment `a → b` at every point where it crosses `other`'s linework, and
/// return the midpoints of the resulting pieces.
///
/// Each midpoint is strictly interior to a piece that does not cross the boundary, so
/// one point-in-polygon test per midpoint decides the whole piece. A segment that
/// crosses nothing yields its own midpoint, so the caller never special-cases it.
pub fn segment_midpoints(a: Coord, b: Coord, other: &Geometry) -> Vec<Coord> {
    let mut ts: Vec<f64> = vec![0.0, 1.0];
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let len2 = dx * dx + dy * dy;
    if len2 == 0.0 {
        return vec![a];
    }
    for line in other.lines() {
        for w in line.windows(2) {
            if let Some(p) = segment_intersection(a, b, w[0], w[1]) {
                ts.push((((p.x - a.x) * dx + (p.y - a.y) * dy) / len2).clamp(0.0, 1.0));
            }
            // A collinear overlap produces no single crossing point, but its
            // endpoints still split the segment, so project any endpoint that lies
            // on it. Without this a segment running along a polygon edge and then
            // leaving it is sampled as one piece and answered by whichever half the
            // midpoint happens to land in.
            for endpoint in [w[0], w[1]] {
                if crate::algo::primitive::on_segment(endpoint, a, b) {
                    ts.push(
                        (((endpoint.x - a.x) * dx + (endpoint.y - a.y) * dy) / len2)
                            .clamp(0.0, 1.0),
                    );
                }
            }
        }
    }
    ts.sort_by(|x, y| x.partial_cmp(y).expect("parameters are finite"));
    let mut out = Vec::with_capacity(ts.len());
    for w in ts.windows(2) {
        if w[1] - w[0] <= f64::EPSILON {
            continue;
        }
        let t = (w[0] + w[1]) / 2.0;
        out.push(Coord {
            x: a.x + t * dx,
            y: a.y + t * dy,
            z: a.z + t * (b.z - a.z),
        });
    }
    if out.is_empty() {
        out.push(Coord {
            x: a.x + 0.5 * dx,
            y: a.y + 0.5 * dy,
            z: (a.z + b.z) / 2.0,
        });
    }
    out
}

/// Every position that decides whether `g` lies inside another geometry: its vertices
/// plus a midpoint of each noded piece of each of its segments.
///
/// This is the sample set the areal predicates run point-in-polygon over. It is finite
/// and exact for polygonal inputs — the noding guarantees no piece straddles a
/// boundary — which is why the predicates built on it do not need an overlay.
pub fn probe_points(g: &Geometry, against: &Geometry) -> Vec<Coord> {
    let mut out = Vec::new();
    g.collect_coords(&mut out);
    for line in g.lines() {
        for w in line.windows(2) {
            out.extend(segment_midpoints(w[0], w[1], against));
        }
    }
    out
}

/// A position strictly inside the geometry, when one exists.
///
/// Used to separate `covers` from `contains`: two polygons sharing only an edge cover
/// nothing of each other's interior, and the only way to see that is to find a point
/// that is in one's interior and ask where it is in the other. For an areal geometry
/// this is the centroid when the centroid is inside, and otherwise a point found by
/// scanning a horizontal line across the shape — which is what makes it correct for
/// the crescent-shaped polygons whose centroid falls in the notch.
pub fn interior_point(g: &Geometry) -> Option<Coord> {
    let polys = g.polygons();
    if !polys.is_empty() {
        if let Some(c) = crate::algo::measure::centroid(g) {
            if point_in_geometry(c, g) == PointRing::Inside {
                return Some(c);
            }
        }
        // Scan: take the y midway between two consecutive distinct vertex ordinates
        // and walk x across the ring's span looking for an interior sample.
        for poly in &polys {
            if let Some(p) = scan_interior(poly) {
                return Some(p);
            }
        }
        return None;
    }
    // A line's "interior" for these purposes is any point on it that is not an
    // endpoint; a point set's is the point itself.
    for l in linear_parts(g) {
        if l.len() >= 2 {
            let mid = Coord {
                x: (l[0].x + l[1].x) / 2.0,
                y: (l[0].y + l[1].y) / 2.0,
                z: (l[0].z + l[1].z) / 2.0,
            };
            return Some(mid);
        }
    }
    g.points().first().copied()
}

fn scan_interior(poly: &Polygon) -> Option<Coord> {
    let mut ys: Vec<f64> = poly.exterior.iter().map(|c| c.y).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ys.dedup();
    for pair in ys.windows(2) {
        let y = (pair[0] + pair[1]) / 2.0;
        let mut xs: Vec<f64> = Vec::new();
        for ring in std::iter::once(&poly.exterior).chain(poly.interiors.iter()) {
            for w in ring.windows(2) {
                if (w[0].y > y) != (w[1].y > y) {
                    xs.push((w[1].x - w[0].x) * (y - w[0].y) / (w[1].y - w[0].y) + w[0].x);
                }
            }
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for pair in xs.chunks_exact(2) {
            if pair[1] > pair[0] {
                let c = Coord::new((pair[0] + pair[1]) / 2.0, y);
                if point_in_polygon(c, poly) == PointRing::Inside {
                    return Some(c);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::wkt::read_wkt;

    fn geom(t: &str) -> Geometry {
        read_wkt(t).expect(t).geometry
    }

    #[test]
    fn a_hole_makes_the_inside_outside() {
        let p = geom("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (4 4, 6 4, 6 6, 4 6, 4 4))");
        let poly = p.polygons()[0];
        assert_eq!(point_in_polygon(Coord::new(1.0, 1.0), poly), PointRing::Inside);
        assert_eq!(point_in_polygon(Coord::new(5.0, 5.0), poly), PointRing::Outside);
        assert_eq!(point_in_polygon(Coord::new(4.0, 5.0), poly), PointRing::Boundary);
    }

    #[test]
    fn a_line_has_no_interior_in_the_plane() {
        let l = geom("LINESTRING(0 0, 10 0)");
        assert_eq!(point_in_geometry(Coord::new(5.0, 0.0), &l), PointRing::Boundary);
        assert_eq!(point_in_geometry(Coord::new(5.0, 1.0), &l), PointRing::Outside);
    }

    #[test]
    fn noding_splits_a_segment_that_enters_and_leaves() {
        // A U-shaped polygon; the segment crosses in, out through the notch, and back.
        let u = geom("POLYGON((0 0, 10 0, 10 10, 7 10, 7 3, 3 3, 3 10, 0 10, 0 0))");
        let mids = segment_midpoints(Coord::new(-1.0, 5.0), Coord::new(11.0, 5.0), &u);
        let inside = mids
            .iter()
            .filter(|m| point_in_geometry(**m, &u) == PointRing::Inside)
            .count();
        let outside = mids.len() - inside;
        assert_eq!(inside, 2, "two arms of the U");
        assert!(outside >= 3, "outside, the notch, and outside again");
    }

    #[test]
    fn interior_point_lands_inside_even_for_a_crescent() {
        // A C-shape whose centroid falls in the gap.
        let c = geom("POLYGON((0 0, 10 0, 10 2, 2 2, 2 8, 10 8, 10 10, 0 10, 0 0))");
        let p = interior_point(&c).expect("a non-degenerate polygon has an interior");
        assert_eq!(point_in_geometry(p, &c), PointRing::Inside);
    }

    #[test]
    fn linear_parts_excludes_polygon_rings() {
        let g = geom("GEOMETRYCOLLECTION(POLYGON((0 0, 1 0, 1 1, 0 0)), LINESTRING(5 5, 6 6))");
        assert_eq!(linear_parts(&g).len(), 1);
        assert_eq!(g.lines().len(), 2);
    }
}
