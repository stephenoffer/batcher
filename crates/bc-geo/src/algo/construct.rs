//! Geometries derived from other geometries — hulls, envelopes, buffers, simplification.
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
        Geometry::MultiLineString(ls) => Geometry::MultiPoint(
            ls.iter()
                .filter(|l| l.len() >= 2 && !crate::types::is_closed(l))
                .flat_map(|l| [Some(l[0]), Some(l[l.len() - 1])])
                .collect(),
        ),
        Geometry::Point(_) | Geometry::MultiPoint(_) => Geometry::MultiPoint(Vec::new()),
        Geometry::GeometryCollection(gs) => {
            Geometry::GeometryCollection(gs.iter().map(boundary).collect())
        }
    }
}

/// The convex hull, by monotone chain.
///
/// Degenerate inputs degrade rather than error: fewer than three distinct positions
/// yield a point or a line, because the hull of two points *is* a line and returning a
/// zero-area polygon would be a lie an areal predicate would then act on.
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
pub fn remove_repeated_points(g: &Geometry, tolerance: f64) -> Geometry {
    fn thin(l: &LineString, tol: f64) -> LineString {
        let mut out: Vec<Coord> = Vec::with_capacity(l.len());
        for c in l {
            match out.last() {
                Some(prev) if dist(*prev, *c) <= tol => {}
                _ => out.push(*c),
            }
        }
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

/// A buffer of `radius` around the geometry, approximated with `quad_segs` segments
/// per quarter circle.
///
/// This is an approximation and says so: it buffers each vertex with a regular polygon
/// and each segment with its offset rectangle, then takes the convex hull of the union.
/// For a convex input that is the exact buffer up to the arc discretization. For a
/// concave one it is the buffer *of the hull*, which is an over-estimate — sound as a
/// candidate filter for a subsequent exact predicate, and wrong if consumed as an area.
/// `buffer_error` names the shortfall so a caller can decide.
pub fn buffer(g: &Geom, radius: f64, quad_segs: usize) -> GeoResult<Geometry> {
    if radius.is_nan() {
        return Err(GeoError::invalid("buffer radius must be a number"));
    }
    if quad_segs == 0 {
        return Err(GeoError::invalid(
            "buffer needs at least one segment per quadrant",
        ));
    }
    if radius <= 0.0 {
        // A zero or negative buffer of a point set has no area; PostGIS returns an
        // empty polygon rather than the input, and an eroded polygon needs the overlay
        // this function deliberately does not use.
        return Ok(Geometry::Polygon(Polygon::default()));
    }
    let coords = g.coords();
    if coords.is_empty() {
        return Ok(Geometry::Polygon(Polygon::default()));
    }
    let steps = quad_segs * 4;
    let mut pts = Vec::with_capacity(coords.len() * steps);
    for c in &coords {
        for k in 0..steps {
            let theta = std::f64::consts::TAU * (k as f64) / (steps as f64);
            pts.push(Coord::new(
                c.x + radius * theta.cos(),
                c.y + radius * theta.sin(),
            ));
        }
    }
    let mut hull_input = Geom::new(Geometry::MultiPoint(pts.into_iter().map(Some).collect()));
    hull_input.srid = g.srid;
    Ok(convex_hull(&hull_input))
}

/// The fraction by which `buffer` over-estimates for this geometry: 0 for a convex
/// input, positive for a concave one.
///
/// Exposed so the approximation is measurable rather than a footnote. A caller running
/// a candidate filter can ignore it; a caller reporting an area can check it and refuse.
pub fn buffer_error(g: &Geom) -> f64 {
    let hull = convex_hull(g);
    let a_hull = crate::algo::measure::area(&hull);
    let a_geom = crate::algo::measure::area(&g.geometry);
    if a_hull <= 0.0 {
        return 0.0;
    }
    ((a_hull - a_geom) / a_hull).max(0.0)
}

/// Force every ring of every polygon to the given winding.
///
/// Shapefiles want clockwise exteriors, GeoJSON wants counter-clockwise ones, and a
/// mixed column is what makes a renderer punch holes in the wrong places.
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
    fn buffer_contains_the_input_and_reports_its_own_error() {
        let pt = g("POINT(0 0)");
        let b = buffer(&pt, 1.0, 8).unwrap();
        let a = area(&b);
        // A 32-gon inscribed in the unit circle: slightly under pi, never over.
        assert!(a < std::f64::consts::PI && a > 3.10, "got {a}");
        assert_eq!(buffer_error(&pt), 0.0);
        let c = g("POLYGON((0 0, 10 0, 10 2, 2 2, 2 8, 10 8, 10 10, 0 10, 0 0))");
        assert!(buffer_error(&c) > 0.3, "a C-shape is far from its hull");
        assert!(buffer(&pt, 1.0, 0).is_err());
        assert!(buffer(&pt, f64::NAN, 4).is_err());
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
}
