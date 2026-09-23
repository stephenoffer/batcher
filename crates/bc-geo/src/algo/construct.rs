//! Geometries derived from other geometries — hulls, envelopes, simplification.
//!
//! `buffer` has its own module, because doing it correctly needs a polygon union.
//!
//! These are the constructors a pipeline reaches for between a raw geometry column and
//! a join or a map: shrink the vertex count before a shuffle, grow a point into a
//! catchment area, reduce a shape to the box an index can hold. Each is a pure function
//! of one geometry, which is what lets them all run as ordinary scalar expressions.

use crate::algo::primitive::{cross, dist};
use crate::error::{GeoError, GeoResult};
use crate::types::{close_ring, is_ccw, Bbox, Coord, Geometry, LineString, Polygon};
use crate::Geom;

/// The axis-aligned bounding box as a geometry.
///
/// A degenerate box is not returned as a degenerate polygon: a point's envelope is a
/// point and a horizontal line's is a line, matching PostGIS, because a zero-area
/// "polygon" breaks every areal predicate downstream.
#[must_use]
pub fn envelope(g: &Geom) -> Geometry {
    let Some(b) = g.bbox() else {
        return Geometry::Polygon(Polygon::default());
    };
    if b.xmin == b.xmax && b.ymin == b.ymax {
        return Geometry::Point(Some(Coord::new(b.xmin, b.ymin)));
    }
    if b.xmin == b.xmax || b.ymin == b.ymax {
        return Geometry::LineString(vec![Coord::new(b.xmin, b.ymin), Coord::new(b.xmax, b.ymax)]);
    }
    Geometry::Polygon(Polygon {
        exterior: b.to_ring(),
        interiors: Vec::new(),
    })
}

/// A rectangle from explicit bounds. Errors when the bounds are inverted, which is
/// almost always a swapped-argument bug rather than an intentional empty box.
pub fn make_envelope(xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> GeoResult<Geometry> {
    if xmin.is_nan()
        || ymin.is_nan()
        || xmax.is_nan()
        || ymax.is_nan()
        || xmin > xmax
        || ymin > ymax
    {
        return Err(GeoError::invalid(format!(
            "envelope bounds are inverted: xmin {xmin} > xmax {xmax} or ymin {ymin} > ymax {ymax}"
        )));
    }
    Ok(Geometry::Polygon(Polygon {
        exterior: Bbox {
            xmin,
            ymin,
            xmax,
            ymax,
        }
        .to_ring(),
        interiors: Vec::new(),
    }))
}

/// The boundary of a geometry: a polygon's rings, a chain's endpoints, nothing for a
/// point set or a closed chain.
///
/// "Nothing" is not one spelling. OGC represents the empty boundary in the type the
/// operand's boundary *would* have had: a chain's boundary is a point set, so a closed
/// chain reports `MULTIPOINT EMPTY`; a point set has no lower dimension to fall to, so
/// a `Point`/`MultiPoint` reports `GEOMETRYCOLLECTION EMPTY`. Both are empty and both
/// compare equal by area, which is exactly why returning the wrong one is invisible to
/// every value assertion and shows up only in the column's type.
pub fn boundary(g: &Geometry) -> Geometry {
    match g {
        Geometry::Polygon(p) => {
            let mut rings: Vec<LineString> = Vec::new();
            if !p.exterior.is_empty() {
                rings.push(p.exterior.clone());
            }
            rings.extend(p.interiors.iter().cloned());
            match rings.len() {
                0 => Geometry::MultiLineString(Vec::new()),
                1 => Geometry::LineString(rings.remove(0)),
                _ => Geometry::MultiLineString(rings),
            }
        }
        Geometry::MultiPolygon(ps) => Geometry::MultiLineString(
            ps.iter()
                .flat_map(|p| std::iter::once(&p.exterior).chain(p.interiors.iter()))
                .filter(|r| !r.is_empty())
                .cloned()
                .collect(),
        ),
        Geometry::LineString(l) => {
            if l.len() < 2 || crate::types::is_closed(l) {
                // A closed chain has no boundary, which is the topological fact that
                // makes a ring "closed" mean something.
                Geometry::MultiPoint(Vec::new())
            } else {
                Geometry::MultiPoint(vec![Some(l[0]), Some(l[l.len() - 1])])
            }
        }
        Geometry::MultiLineString(ls) => {
            Geometry::MultiPoint(mod2_boundary(ls).into_iter().map(Some).collect())
        }
        // The empty set at dimension -1: a point set's boundary has no type of its own
        // to be empty in, so OGC (and GEOS, and PostGIS) spell it as an empty collection
        // rather than as an empty MULTIPOINT.
        Geometry::Point(_) | Geometry::MultiPoint(_) => Geometry::GeometryCollection(Vec::new()),
        Geometry::GeometryCollection(gs) => {
            Geometry::GeometryCollection(gs.iter().map(boundary).collect())
        }
    }
}

/// The OGC "mod 2" boundary of a set of chains: every endpoint that ends an *odd*
/// number of member chains.
///
/// Two chains meeting end to end at a point are one path through it, so that point is
/// interior, not boundary: `MULTILINESTRING((0 0, 1 1), (1 1, 2 0))` has the boundary
/// `MULTIPOINT((0 0), (2 0))`. Listing every member's endpoints, as this used to, put
/// `(1 1)` in the boundary twice. A closed member contributes its start twice and so
/// nothing. The points come back sorted by x then y, which is the order GEOS (and so
/// PostGIS and DuckDB) emits them in.
fn mod2_boundary(ls: &[LineString]) -> Vec<Coord> {
    let mut ends: Vec<Coord> = ls
        .iter()
        .filter(|l| l.len() >= 2)
        .flat_map(|l| [l[0], l[l.len() - 1]])
        .collect();
    ends.sort_by(|a, b| {
        a.x.total_cmp(&b.x)
            .then(a.y.total_cmp(&b.y))
            .then(a.z.total_cmp(&b.z))
    });
    let mut out = Vec::new();
    let mut i = 0;
    while i < ends.len() {
        let mut j = i + 1;
        while j < ends.len() && ends[j].x == ends[i].x && ends[j].y == ends[i].y {
            j += 1;
        }
        if (j - i) % 2 == 1 {
            out.push(ends[i]);
        }
        i = j;
    }
    out
}

/// The convex hull, by monotone chain.
///
/// Degenerate inputs degrade rather than error: fewer than three distinct positions
/// yield a point or a line, because the hull of two points *is* a line and returning a
/// zero-area polygon would be a lie an areal predicate would then act on.
#[must_use]
pub fn convex_hull(g: &Geom) -> Geometry {
    let mut pts = g.coords();
    pts.retain(|c| !c.is_nan());
    pts.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
    });
    pts.dedup_by(|a, b| a.x == b.x && a.y == b.y);
    match pts.len() {
        0 => return Geometry::Polygon(Polygon::default()),
        1 => return Geometry::Point(Some(pts[0])),
        2 => return Geometry::LineString(pts),
        _ => {}
    }
    let mut hull: Vec<Coord> = Vec::with_capacity(pts.len() * 2);
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &Coord>> = if pass == 0 {
            Box::new(pts.iter())
        } else {
            Box::new(pts.iter().rev())
        };
        for p in iter {
            while hull.len() >= start + 2
                && cross(hull[hull.len() - 2], hull[hull.len() - 1], *p) <= 0.0
            {
                hull.pop();
            }
            hull.push(*p);
        }
        hull.pop();
    }
    if hull.len() < 3 {
        return Geometry::LineString(pts);
    }
    close_ring(&mut hull);
    Geometry::Polygon(Polygon {
        exterior: hull,
        interiors: Vec::new(),
    })
}

/// Douglas-Peucker simplification with tolerance `eps`.
///
/// Rings are simplified with their closing vertex pinned, and a ring that would fall
/// below three distinct vertices is left at its previous state rather than collapsed —
/// dropping it would turn a polygon column into a mix of polygons and nothing, which
/// every downstream areal predicate then answers differently.
pub fn simplify(g: &Geometry, eps: f64) -> GeoResult<Geometry> {
    if eps.is_nan() || eps < 0.0 {
        return Err(GeoError::invalid(format!(
            "simplify tolerance must be >= 0, got {eps}"
        )));
    }
    Ok(match g {
        Geometry::LineString(l) => Geometry::LineString(dp(l, eps)),
        Geometry::MultiLineString(ls) => {
            Geometry::MultiLineString(ls.iter().map(|l| dp(l, eps)).collect())
        }
        Geometry::Polygon(p) => Geometry::Polygon(simplify_polygon(p, eps)),
        Geometry::MultiPolygon(ps) => {
            Geometry::MultiPolygon(ps.iter().map(|p| simplify_polygon(p, eps)).collect())
        }
        Geometry::GeometryCollection(gs) => Geometry::GeometryCollection(
            gs.iter()
                .map(|c| simplify(c, eps))
                .collect::<GeoResult<_>>()?,
        ),
        other => other.clone(),
    })
}

fn simplify_polygon(p: &Polygon, eps: f64) -> Polygon {
    Polygon {
        exterior: simplify_ring(&p.exterior, eps),
        interiors: p.interiors.iter().map(|r| simplify_ring(r, eps)).collect(),
    }
}

fn simplify_ring(ring: &LineString, eps: f64) -> LineString {
    if ring.len() <= 4 {
        return ring.clone();
    }
    let mut out = dp(ring, eps);
    if out.len() < 4 {
        return ring.clone();
    }
    close_ring(&mut out);
    out
}

fn dp(line: &[Coord], eps: f64) -> LineString {
    if line.len() <= 2 || eps == 0.0 {
        return line.to_vec();
    }
    let mut keep = vec![false; line.len()];
    keep[0] = true;
    keep[line.len() - 1] = true;
    dp_recurse(line, 0, line.len() - 1, eps, &mut keep);
    line.iter()
        .zip(keep)
        .filter_map(|(c, k)| k.then_some(*c))
        .collect()
}

fn dp_recurse(line: &[Coord], first: usize, last: usize, eps: f64, keep: &mut [bool]) {
    if last <= first + 1 {
        return;
    }
    let (mut best, mut best_i) = (0.0, first);
    for (i, p) in line.iter().enumerate().take(last).skip(first + 1) {
        let d = crate::algo::primitive::point_segment_distance(*p, line[first], line[last]);
        if d > best {
            best = d;
            best_i = i;
        }
    }
    if best > eps {
        keep[best_i] = true;
        dp_recurse(line, first, best_i, eps, keep);
        dp_recurse(line, best_i, last, eps, keep);
    }
}

/// Drop consecutive duplicate positions, optionally merging any pair closer than
/// `tolerance`. Rings stay closed.
///
/// Both endpoints of a chain survive, and a chain never thins below two positions —
/// GEOS's rule, and so PostGIS's and DuckDB's. `LINESTRING(0 0, 0 0)` stays as it is
/// rather than becoming a one-position "line", which no encoding treats as a line and
/// every length and validity function then answered differently.
#[must_use]
pub fn remove_repeated_points(g: &Geometry, tolerance: f64) -> Geometry {
    fn thin(l: &LineString, tol: f64) -> LineString {
        let (Some(&first), Some(&last)) = (l.first(), l.last()) else {
            return l.clone();
        };
        if l.len() < 2 {
            return l.clone();
        }
        let mut out: Vec<Coord> = Vec::with_capacity(l.len());
        out.push(first);
        for c in &l[1..l.len() - 1] {
            if out.last().is_some_and(|prev| dist(*prev, *c) > tol) {
                out.push(*c);
            }
        }
        // The last position is kept even when it is within tolerance of the one before;
        // that one gives way instead, so the chain still ends where it ended.
        if out.len() >= 2 && out.last().is_some_and(|prev| dist(*prev, last) <= tol) {
            out.pop();
        }
        out.push(last);
        out
    }
    fn thin_ring(l: &LineString, tol: f64) -> LineString {
        let mut out = thin(l, tol);
        // Thinning can unclose a ring by dropping its final repeat; three distinct
        // vertices plus the closer is the minimum a ring can be.
        if out.len() >= 3 {
            close_ring(&mut out);
            out
        } else {
            l.clone()
        }
    }
    match g {
        Geometry::LineString(l) => Geometry::LineString(thin(l, tolerance)),
        Geometry::MultiLineString(ls) => {
            Geometry::MultiLineString(ls.iter().map(|l| thin(l, tolerance)).collect())
        }
        Geometry::Polygon(p) => Geometry::Polygon(Polygon {
            exterior: thin_ring(&p.exterior, tolerance),
            interiors: p
                .interiors
                .iter()
                .map(|r| thin_ring(r, tolerance))
                .collect(),
        }),
        Geometry::MultiPolygon(ps) => Geometry::MultiPolygon(
            ps.iter()
                .map(|p| Polygon {
                    exterior: thin_ring(&p.exterior, tolerance),
                    interiors: p
                        .interiors
                        .iter()
                        .map(|r| thin_ring(r, tolerance))
                        .collect(),
                })
                .collect(),
        ),
        Geometry::GeometryCollection(gs) => Geometry::GeometryCollection(
            gs.iter()
                .map(|c| remove_repeated_points(c, tolerance))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Force every ring of every polygon to the given winding.
///
/// Shapefiles want clockwise exteriors, GeoJSON wants counter-clockwise ones, and a
/// mixed column is what makes a renderer punch holes in the wrong places.
#[must_use]
pub fn force_winding(g: &Geometry, exterior_ccw: bool) -> Geometry {
    fn fix(ring: &LineString, want_ccw: bool) -> LineString {
        if ring.len() < 4 || is_ccw(ring) == want_ccw {
            ring.clone()
        } else {
            ring.iter().rev().copied().collect()
        }
    }
    fn poly(p: &Polygon, exterior_ccw: bool) -> Polygon {
        Polygon {
            exterior: fix(&p.exterior, exterior_ccw),
            // Holes always wind opposite the shell; that opposition is the encoding of
            // "this ring subtracts".
            interiors: p.interiors.iter().map(|r| fix(r, !exterior_ccw)).collect(),
        }
    }
    match g {
        Geometry::Polygon(p) => Geometry::Polygon(poly(p, exterior_ccw)),
        Geometry::MultiPolygon(ps) => {
            Geometry::MultiPolygon(ps.iter().map(|p| poly(p, exterior_ccw)).collect())
        }
        Geometry::GeometryCollection(gs) => Geometry::GeometryCollection(
            gs.iter().map(|c| force_winding(c, exterior_ccw)).collect(),
        ),
        other => other.clone(),
    }
}

/// Reverse the vertex order of every chain and ring.
pub fn reverse(g: &Geometry) -> Geometry {
    fn rev(l: &LineString) -> LineString {
        l.iter().rev().copied().collect()
    }
    match g {
        Geometry::LineString(l) => Geometry::LineString(rev(l)),
        Geometry::MultiLineString(ls) => Geometry::MultiLineString(ls.iter().map(rev).collect()),
        Geometry::Polygon(p) => Geometry::Polygon(Polygon {
            exterior: rev(&p.exterior),
            interiors: p.interiors.iter().map(rev).collect(),
        }),
        Geometry::MultiPolygon(ps) => Geometry::MultiPolygon(
            ps.iter()
                .map(|p| Polygon {
                    exterior: rev(&p.exterior),
                    interiors: p.interiors.iter().map(rev).collect(),
                })
                .collect(),
        ),
        Geometry::GeometryCollection(gs) => {
            Geometry::GeometryCollection(gs.iter().map(reverse).collect())
        }
        other => other.clone(),
    }
}

/// Swap x and y in every position — the fix for a lat/lon column loaded as lon/lat.
#[must_use]
pub fn flip_coordinates(g: &Geometry) -> Geometry {
    g.map_coords(&mut |c| Coord {
        x: c.y,
        y: c.x,
        z: c.z,
    })
}

/// Combine two geometries into one collection without computing an overlay.
///
/// This is `ST_Collect`, not `ST_Union`: it concatenates. Two adjacent polygons
/// collected stay two polygons that happen to touch, which is the cheap and lossless
/// operation, and is what you want before a single `ST_Envelope` or `ST_ConvexHull`.
#[must_use]
pub fn collect(a: &Geometry, b: &Geometry) -> Geometry {
    fn parts(g: &Geometry) -> Vec<Geometry> {
        match g {
            Geometry::GeometryCollection(gs) => gs.clone(),
            other if other.is_empty() => Vec::new(),
            other => vec![other.clone()],
        }
    }
    let mut all = parts(a);
    all.extend(parts(b));
    // A homogeneous collection is spelled as the matching multi-geometry, which is what
    // PostGIS returns and what keeps `ST_GeometryType` informative.
    if all.iter().all(|g| matches!(g, Geometry::Point(_))) && !all.is_empty() {
        return Geometry::MultiPoint(
            all.iter()
                .map(|g| match g {
                    Geometry::Point(p) => *p,
                    _ => unreachable!("checked above"),
                })
                .collect(),
        );
    }
    if all.iter().all(|g| matches!(g, Geometry::LineString(_))) && !all.is_empty() {
        return Geometry::MultiLineString(
            all.iter()
                .map(|g| match g {
                    Geometry::LineString(l) => l.clone(),
                    _ => unreachable!("checked above"),
                })
                .collect(),
        );
    }
    if all.iter().all(|g| matches!(g, Geometry::Polygon(_))) && !all.is_empty() {
        return Geometry::MultiPolygon(
            all.iter()
                .map(|g| match g {
                    Geometry::Polygon(p) => p.clone(),
                    _ => unreachable!("checked above"),
                })
                .collect(),
        );
    }
    Geometry::GeometryCollection(all)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::measure::area;
    use crate::codec::wkt::{read_wkt, write_wkt};

    fn g(t: &str) -> Geom {
        read_wkt(t).expect(t)
    }

    fn wkt(geom: Geometry) -> String {
        write_wkt(&Geom::new(geom))
    }

    #[test]
    fn envelope_degrades_rather_than_returning_a_zero_area_polygon() {
        assert_eq!(wkt(envelope(&g("POINT(1 2)"))), "POINT(1 2)");
        assert_eq!(
            wkt(envelope(&g("LINESTRING(0 5, 4 5)"))),
            "LINESTRING(0 5, 4 5)"
        );
        assert_eq!(
            wkt(envelope(&g("POLYGON((0 0, 4 0, 4 3, 0 0))"))),
            "POLYGON((0 0, 4 0, 4 3, 0 3, 0 0))"
        );
    }

    #[test]
    fn make_envelope_rejects_inverted_bounds() {
        assert!(make_envelope(0.0, 0.0, 1.0, 1.0).is_ok());
        assert!(make_envelope(1.0, 0.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn convex_hull_of_a_square_with_an_interior_point_is_the_square() {
        let h = convex_hull(&g("MULTIPOINT((0 0), (4 0), (4 4), (0 4), (2 2))"));
        assert_eq!(area(&h), 16.0);
        assert_eq!(h.num_points(), 5);
    }

    #[test]
    fn convex_hull_degrades_for_fewer_than_three_points() {
        assert_eq!(wkt(convex_hull(&g("POINT(1 1)"))), "POINT(1 1)");
        assert_eq!(
            wkt(convex_hull(&g("MULTIPOINT((0 0), (2 2))"))),
            "LINESTRING(0 0, 2 2)"
        );
        // Three collinear points have no area, so the hull is still a line.
        assert_eq!(
            wkt(convex_hull(&g("MULTIPOINT((0 0), (1 1), (2 2))"))),
            "LINESTRING(0 0, 1 1, 2 2)"
        );
    }

    #[test]
    fn simplify_drops_collinear_vertices_and_keeps_the_ends() {
        let s = simplify(&g("LINESTRING(0 0, 1 0.001, 2 0, 3 0)").geometry, 0.01).unwrap();
        assert_eq!(wkt(s), "LINESTRING(0 0, 3 0)");
        let unchanged = simplify(&g("LINESTRING(0 0, 1 5, 2 0)").geometry, 0.01).unwrap();
        assert_eq!(unchanged.num_points(), 3);
        assert!(simplify(&g("LINESTRING(0 0, 1 1)").geometry, -1.0).is_err());
    }

    #[test]
    fn simplify_never_collapses_a_ring_below_a_triangle() {
        let tiny = "POLYGON((0 0, 0.001 0, 0.001 0.001, 0 0.001, 0 0))";
        let s = simplify(&g(tiny).geometry, 10.0).unwrap();
        assert!(s.num_points() >= 4, "got {}", wkt(s));
    }

    #[test]
    fn boundary_of_a_closed_line_is_empty() {
        assert_eq!(
            wkt(boundary(&g("LINESTRING(0 0, 1 0, 1 1, 0 0)").geometry)),
            "MULTIPOINT EMPTY"
        );
        assert_eq!(
            wkt(boundary(&g("LINESTRING(0 0, 1 1)").geometry)),
            "MULTIPOINT((0 0), (1 1))"
        );
        assert_eq!(
            wkt(boundary(&g("POLYGON((0 0, 1 0, 1 1, 0 0))").geometry)),
            "LINESTRING(0 0, 1 0, 1 1, 0 0)"
        );
    }

    #[test]
    fn the_empty_boundary_keeps_the_type_ogc_gives_it() {
        // Three different empty geometries, and which one you get is the whole content
        // of the answer: every value-level assertion passes on any of them. GEOS and
        // PostGIS agree on each of these.
        for w in ["POINT(1 2)", "MULTIPOINT((0 0), (1 1))"] {
            assert_eq!(
                wkt(boundary(&g(w).geometry)),
                "GEOMETRYCOLLECTION EMPTY",
                "a point set falls to no lower dimension, so its empty boundary is a \
                 collection, not an empty MULTIPOINT ({w})"
            );
        }
        assert_eq!(
            wkt(boundary(&g("LINESTRING(0 0, 1 0, 1 1, 0 0)").geometry)),
            "MULTIPOINT EMPTY",
            "a chain's boundary is a point set even when it is empty"
        );
        assert_eq!(
            wkt(boundary(&g("POLYGON EMPTY").geometry)),
            "MULTILINESTRING EMPTY",
            "an areal boundary is a line set even when it is empty"
        );
    }

    #[test]
    fn winding_is_forced_consistently_including_holes() {
        let p = g("POLYGON((0 0, 0 4, 4 4, 4 0, 0 0), (1 1, 2 1, 2 2, 1 2, 1 1))");
        let ccw = force_winding(&p.geometry, true);
        let poly = ccw.polygons()[0];
        assert!(is_ccw(&poly.exterior));
        assert!(
            !is_ccw(&poly.interiors[0]),
            "a hole winds against its shell"
        );
        assert_eq!(
            area(&ccw),
            area(&p.geometry),
            "winding is not an area change"
        );
    }

    #[test]
    fn flip_swaps_the_axes() {
        assert_eq!(
            wkt(flip_coordinates(&g("POINT(1 2)").geometry)),
            "POINT(2 1)"
        );
    }

    #[test]
    fn collect_produces_the_narrowest_container() {
        assert_eq!(
            wkt(collect(
                &g("POINT(0 0)").geometry,
                &g("POINT(1 1)").geometry
            )),
            "MULTIPOINT((0 0), (1 1))"
        );
        assert_eq!(
            wkt(collect(
                &g("POINT(0 0)").geometry,
                &g("LINESTRING(0 0, 1 1)").geometry
            )),
            "GEOMETRYCOLLECTION(POINT(0 0), LINESTRING(0 0, 1 1))"
        );
    }

    #[test]
    fn repeated_points_are_dropped_without_unclosing_a_ring() {
        let out =
            remove_repeated_points(&g("POLYGON((0 0, 0 0, 4 0, 4 4, 0 4, 0 0))").geometry, 0.0);
        assert_eq!(wkt(out), "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))");
    }

    #[test]
    fn a_multiline_boundary_follows_the_mod_2_rule() {
        // DuckDB: MULTIPOINT (0 0, 2 0), and MULTIPOINT (0 0, 1 1, 1 5, 2 0).
        let b = |t: &str| write_wkt(&Geom::new(boundary(&read_wkt(t).unwrap().geometry)));
        assert_eq!(
            b("MULTILINESTRING((0 0, 1 1), (1 1, 2 0))"),
            "MULTIPOINT((0 0), (2 0))"
        );
        assert_eq!(
            b("MULTILINESTRING((0 0, 1 1), (1 1, 2 0), (1 1, 1 5))"),
            "MULTIPOINT((0 0), (1 1), (1 5), (2 0))"
        );
        assert_eq!(
            b("MULTILINESTRING((0 0, 1 1), (1 1, 0 0))"),
            "MULTIPOINT EMPTY"
        );
    }

    #[test]
    fn thinning_keeps_both_endpoints_and_never_leaves_one_position() {
        let t = |w: &str, tol: f64| {
            write_wkt(&Geom::new(remove_repeated_points(
                &read_wkt(w).unwrap().geometry,
                tol,
            )))
        };
        // Expected values are DuckDB's ST_RemoveRepeatedPoints.
        assert_eq!(t("LINESTRING(0 0, 0 0)", 0.0), "LINESTRING(0 0, 0 0)");
        assert_eq!(t("LINESTRING(0 0, 0 0, 0 0)", 0.0), "LINESTRING(0 0, 0 0)");
        assert_eq!(
            t("LINESTRING(0 0, 5 5, 5.1 5)", 1.0),
            "LINESTRING(0 0, 5.1 5)"
        );
        assert_eq!(
            t("LINESTRING(0 0, 0.1 0, 0.2 0)", 1.0),
            "LINESTRING(0 0, 0.2 0)"
        );
        assert_eq!(
            t("LINESTRING(0 0, 1 1, 1 1, 2 2)", 0.0),
            "LINESTRING(0 0, 1 1, 2 2)"
        );
    }
}
