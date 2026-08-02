//! Planar measurements — area, length, distance, azimuth.
//!
//! Everything here is Cartesian: it treats coordinates as points on a flat plane and
//! reports answers in whatever unit those coordinates are stated in. For a projected
//! CRS that is metres and the answer is a real measurement. For EPSG:4326 it is
//! *degrees*, and a "distance" of 0.01 is not 1.1 km — it is 1.1 km north-south and
//! anywhere from 1.1 km to nothing east-west depending on latitude.
//!
//! That is not a defect to be papered over: PostGIS `ST_Distance` on a `geometry` does
//! exactly this, and every spatial index and join is built on the planar metric because
//! it is the one that is cheap and monotone. The geodesic answers live in `proj::geodesy`
//! and are named for it (`st_distance_sphere`, `st_area_geodesic`), so a caller who
//! needs metres on a globe asks for metres on a globe.

use crate::algo::primitive::{dist, point_segment_distance, segment_segment_distance, PointRing};
use crate::algo::relate::point_in_polygon;
use crate::types::{measurement, ring_area, Coord, Geometry, LineString, Polygon};
use crate::Geom;

/// The planar area of a geometry: a polygon's exterior minus its holes, summed over
/// members. Points and lines have zero area.
pub fn area(g: &Geometry) -> f64 {
    measurement(g.polygons().iter().copied().map(polygon_area).sum())
}

/// The area of one polygon: exterior minus interiors, never negative.
pub fn polygon_area(p: &Polygon) -> f64 {
    let holes: f64 = p.interiors.iter().map(|r| ring_area(r)).sum();
    measurement((ring_area(&p.exterior) - holes).max(0.0))
}

/// The total length of every line chain. Polygon boundaries are *excluded* — that is
/// `perimeter`. PostGIS `ST_Length` returns 0 for a polygon, and matching it keeps a
/// mixed-geometry column from silently summing two different quantities.
pub fn length(g: &Geometry) -> f64 {
    measurement(match g {
        Geometry::LineString(l) => line_length(l),
        Geometry::MultiLineString(ls) => ls.iter().map(line_length).sum(),
        Geometry::GeometryCollection(gs) => gs.iter().map(length).sum(),
        _ => 0.0,
    })
}

/// The total boundary length of every polygon, holes included. Zero for a non-areal
/// geometry, matching PostGIS `ST_Perimeter`.
pub fn perimeter(g: &Geometry) -> f64 {
    measurement(
        g.polygons()
            .iter()
            .map(|p| line_length(&p.exterior) + p.interiors.iter().map(line_length).sum::<f64>())
            .sum(),
    )
}

/// The length of one chain.
pub fn line_length(l: &LineString) -> f64 {
    measurement(l.windows(2).map(|w| dist(w[0], w[1])).sum())
}

/// The smallest planar distance between two geometries; 0 when they intersect.
///
/// `None` when either geometry is empty, which is what PostGIS returns and is the
/// honest answer: there is no pair of points to measure between.
pub fn distance(a: &Geom, b: &Geom) -> Option<f64> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    // Containment has distance 0 and no shared boundary point, so the segment-to-segment
    // sweep below would miss it and report the distance to the ring instead.
    if crate::algo::predicate::intersects(a, b) {
        return Some(0.0);
    }
    Some(min_distance(&a.geometry, &b.geometry))
}

/// The minimum distance between the component parts of two geometries, assuming they
/// do not intersect (the caller has already checked).
fn min_distance(a: &Geometry, b: &Geometry) -> f64 {
    let mut best = f64::INFINITY;
    let (pa, pb) = (a.points(), b.points());
    let (la, lb) = (a.lines(), b.lines());
    for x in &pa {
        for y in &pb {
            best = best.min(dist(*x, *y));
        }
        for l in &lb {
            best = best.min(point_line_distance(*x, l));
        }
    }
    for y in &pb {
        for l in &la {
            best = best.min(point_line_distance(*y, l));
        }
    }
    for l1 in &la {
        for l2 in &lb {
            best = best.min(line_line_distance(l1, l2));
        }
    }
    best
}

/// The distance from a position to a chain. Infinite for an empty chain, so it never
/// wins a `min` it should not.
pub fn point_line_distance(p: Coord, l: &LineString) -> f64 {
    if l.is_empty() {
        return f64::INFINITY;
    }
    if l.len() == 1 {
        return dist(p, l[0]);
    }
    l.windows(2)
        .map(|w| point_segment_distance(p, w[0], w[1]))
        .fold(f64::INFINITY, f64::min)
}

/// The smallest distance between two chains.
fn line_line_distance(a: &LineString, b: &LineString) -> f64 {
    if a.len() < 2 || b.len() < 2 {
        return a
            .iter()
            .map(|p| point_line_distance(*p, b))
            .chain(b.iter().map(|p| point_line_distance(*p, a)))
            .fold(f64::INFINITY, f64::min);
    }
    let mut best = f64::INFINITY;
    for s in a.windows(2) {
        for t in b.windows(2) {
            best = best.min(segment_segment_distance(s[0], s[1], t[0], t[1]));
            if best == 0.0 {
                return 0.0;
            }
        }
    }
    best
}

/// The largest distance between any pair of vertices of the two geometries
/// (PostGIS `ST_MaxDistance`).
pub fn max_distance(a: &Geom, b: &Geom) -> Option<f64> {
    let (ca, cb) = (a.coords(), b.coords());
    if ca.is_empty() || cb.is_empty() {
        return None;
    }
    let mut best: f64 = 0.0;
    for x in &ca {
        for y in &cb {
            best = best.max(dist(*x, *y));
        }
    }
    Some(best)
}

/// The discrete Hausdorff distance: the largest distance from any vertex of one
/// geometry to the *nearest point* of the other, symmetrized.
///
/// "Discrete" because it samples only the vertices of each input, which is what
/// PostGIS `ST_HausdorffDistance` also does. On densified inputs it converges to the
/// continuous value; on sparse ones it under-reports, and that is the documented
/// trade rather than a hidden one.
pub fn hausdorff_distance(a: &Geom, b: &Geom) -> Option<f64> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    Some(directed_hausdorff(a, b).max(directed_hausdorff(b, a)))
}

fn directed_hausdorff(from: &Geom, to: &Geom) -> f64 {
    from.coords()
        .iter()
        .map(|c| nearest_distance(*c, &to.geometry))
        .fold(0.0, f64::max)
}

/// The distance from a position to the nearest point of a geometry (0 when inside a
/// polygon).
pub fn nearest_distance(p: Coord, g: &Geometry) -> f64 {
    for poly in g.polygons() {
        if point_in_polygon(p, poly) != PointRing::Outside {
            return 0.0;
        }
    }
    let mut best = f64::INFINITY;
    for q in g.points() {
        best = best.min(dist(p, q));
    }
    for l in g.lines() {
        best = best.min(point_line_distance(p, l));
    }
    best
}

/// The azimuth from `a` to `b`: the angle in radians clockwise from north, in
/// `[0, 2π)`. `None` when the two positions coincide, which has no direction.
///
/// This is the planar azimuth PostGIS `ST_Azimuth` computes on a `geometry`: it uses
/// the coordinate axes, so on lon/lat it is only correct near the equator. The
/// geodesic bearing is `proj::geodesy::bearing`.
pub fn azimuth(a: Coord, b: Coord) -> Option<f64> {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    if dx == 0.0 && dy == 0.0 {
        return None;
    }
    let mut theta = dx.atan2(dy);
    if theta < 0.0 {
        theta += std::f64::consts::TAU;
    }
    Some(theta)
}

/// The centroid of a geometry, by the highest dimension present.
///
/// Areal geometries use the area-weighted polygon centroid, linear ones the
/// length-weighted midpoint, and point sets the arithmetic mean — the OGC rule, and
/// the reason a `GEOMETRYCOLLECTION(POLYGON, POINT)` centroid ignores the point.
/// A zero-measure areal or linear geometry (a degenerate polygon, a zero-length line)
/// falls back to the vertex mean rather than dividing by zero.
pub fn centroid(g: &Geometry) -> Option<Coord> {
    let polys = g.polygons();
    if !polys.is_empty() {
        if let Some(c) = area_centroid(&polys) {
            return Some(c);
        }
    }
    let lines = g.lines();
    if !lines.is_empty() {
        if let Some(c) = line_centroid(&lines) {
            return Some(c);
        }
    }
    let pts = g.points();
    if !pts.is_empty() {
        let n = pts.len() as f64;
        return Some(Coord::new(
            pts.iter().map(|c| c.x).sum::<f64>() / n,
            pts.iter().map(|c| c.y).sum::<f64>() / n,
        ));
    }
    // Nothing but degenerate geometry: fall back to the mean of every vertex so a
    // sliver polygon still reports a position rather than nothing.
    let all: Vec<Coord> = {
        let mut v = Vec::new();
        g.collect_coords(&mut v);
        v
    };
    if all.is_empty() {
        return None;
    }
    let n = all.len() as f64;
    Some(Coord::new(
        all.iter().map(|c| c.x).sum::<f64>() / n,
        all.iter().map(|c| c.y).sum::<f64>() / n,
    ))
}

fn ring_centroid_moment(ring: &[Coord]) -> (f64, f64, f64) {
    let mut a2 = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    for w in ring.windows(2) {
        let f = w[0].x * w[1].y - w[1].x * w[0].y;
        a2 += f;
        cx += (w[0].x + w[1].x) * f;
        cy += (w[0].y + w[1].y) * f;
    }
    (a2, cx, cy)
}

fn area_centroid(polys: &[&Polygon]) -> Option<Coord> {
    let mut a2 = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    for p in polys {
        for ring in std::iter::once(&p.exterior).chain(p.interiors.iter()) {
            let mut closed = ring.to_vec();
            crate::types::close_ring(&mut closed);
            let (ra, rx, ry) = ring_centroid_moment(&closed);
            // A hole subtracts, which the shoelace sign already encodes when the hole
            // winds opposite the shell. Force the sign so a same-winding hole still
            // subtracts rather than doubling the shell.
            let sign = if std::ptr::eq(ring, &p.exterior) {
                1.0
            } else {
                -1.0
            };
            let flip = if (ra > 0.0) == (sign > 0.0) {
                1.0
            } else {
                -1.0
            };
            a2 += flip * ra;
            cx += flip * rx;
            cy += flip * ry;
        }
    }
    if a2.abs() < f64::MIN_POSITIVE {
        return None;
    }
    Some(Coord::new(cx / (3.0 * a2), cy / (3.0 * a2)))
}

fn line_centroid(lines: &[&LineString]) -> Option<Coord> {
    let mut total = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    for l in lines {
        for w in l.windows(2) {
            let d = dist(w[0], w[1]);
            total += d;
            cx += d * (w[0].x + w[1].x) / 2.0;
            cy += d * (w[0].y + w[1].y) / 2.0;
        }
    }
    (total > 0.0).then(|| Coord::new(cx / total, cy / total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::wkt::read_wkt;

    fn g(t: &str) -> Geom {
        read_wkt(t).expect(t)
    }

    #[test]
    fn area_subtracts_holes() {
        assert_eq!(
            area(&g("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").geometry),
            16.0
        );
        assert_eq!(
            area(&g("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 2 1, 2 2, 1 2, 1 1))").geometry),
            15.0
        );
        assert_eq!(area(&g("LINESTRING(0 0, 1 1)").geometry), 0.0);
    }

    #[test]
    fn an_empty_measurement_is_positive_zero() {
        // Rust's `Sum` for f64 has `-0.0` as its identity, so every one of these was
        // returning `-0.0` for a geometry with nothing to measure — a value that
        // compares equal to zero, prints as `-0.0`, and reached users in an area column.
        for t in ["POINT(2 2)", "LINESTRING(0 0, 1 1)", "POINT EMPTY"] {
            for v in [
                area(&g(t).geometry),
                length(&g(t).geometry),
                perimeter(&g(t).geometry),
            ] {
                assert!(!v.is_sign_negative(), "{t}: {v} is a negative zero");
            }
        }
        assert!(
            !crate::proj::geodesy::geodesic_area_m2(&g("POINT(2 2)").geometry).is_sign_negative()
        );
    }

    #[test]
    fn length_and_perimeter_measure_different_things() {
        let poly = g("POLYGON((0 0, 3 0, 3 4, 0 0))");
        assert_eq!(length(&poly.geometry), 0.0);
        assert_eq!(perimeter(&poly.geometry), 12.0);
        assert_eq!(length(&g("LINESTRING(0 0, 3 0, 3 4)").geometry), 7.0);
        assert_eq!(perimeter(&g("LINESTRING(0 0, 3 0)").geometry), 0.0);
    }

    #[test]
    fn distance_is_zero_for_intersecting_and_for_contained() {
        let poly = g("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))");
        assert_eq!(distance(&poly, &g("POINT(2 2)")), Some(0.0));
        assert_eq!(distance(&poly, &g("POINT(6 2)")), Some(2.0));
        assert_eq!(distance(&g("POINT(0 0)"), &g("POINT(3 4)")), Some(5.0));
        assert_eq!(distance(&g("POINT EMPTY"), &g("POINT(0 0)")), None);
    }

    #[test]
    fn line_to_line_distance_uses_the_closest_segments() {
        assert_eq!(
            distance(&g("LINESTRING(0 0, 10 0)"), &g("LINESTRING(3 5, 7 5)")),
            Some(5.0)
        );
    }

    #[test]
    fn centroid_prefers_the_highest_dimension() {
        let c = centroid(&g("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").geometry).unwrap();
        assert!((c.x - 2.0).abs() < 1e-12 && (c.y - 2.0).abs() < 1e-12);
        let c = centroid(
            &g("GEOMETRYCOLLECTION(POLYGON((0 0, 2 0, 2 2, 0 2, 0 0)), POINT(100 100))").geometry,
        )
        .unwrap();
        assert!(
            (c.x - 1.0).abs() < 1e-12,
            "the point must not move an areal centroid"
        );
        let c = centroid(&g("MULTIPOINT((0 0), (4 0))").geometry).unwrap();
        assert_eq!((c.x, c.y), (2.0, 0.0));
    }

    #[test]
    fn a_hole_pulls_the_centroid_away_from_it() {
        let solid = centroid(&g("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))").geometry).unwrap();
        let holed = centroid(
            &g("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (6 6, 9 6, 9 9, 6 9, 6 6))").geometry,
        )
        .unwrap();
        assert!(holed.x < solid.x && holed.y < solid.y);
    }

    #[test]
    fn azimuth_measures_clockwise_from_north() {
        let north = azimuth(Coord::new(0.0, 0.0), Coord::new(0.0, 1.0)).unwrap();
        let east = azimuth(Coord::new(0.0, 0.0), Coord::new(1.0, 0.0)).unwrap();
        assert!(north.abs() < 1e-12);
        assert!((east - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert_eq!(azimuth(Coord::new(1.0, 1.0), Coord::new(1.0, 1.0)), None);
    }

    #[test]
    fn hausdorff_is_symmetric_and_bounded_by_max_distance() {
        let a = g("LINESTRING(0 0, 10 0)");
        let b = g("LINESTRING(0 1, 10 1)");
        assert_eq!(hausdorff_distance(&a, &b), Some(1.0));
        assert_eq!(hausdorff_distance(&b, &a), hausdorff_distance(&a, &b));
        assert!(max_distance(&a, &b).unwrap() >= hausdorff_distance(&a, &b).unwrap());
    }
}
