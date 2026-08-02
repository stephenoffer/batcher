//! Linear referencing — positions along a chain, expressed as a fraction of its length.
//!
//! This is the vocabulary route data is described in: "the incident is 0.32 of the way
//! along segment 4471", "give me the stretch between mile 3 and mile 7". Everything
//! here is parameterized on the `[0, 1]` fraction rather than on an absolute distance,
//! which is what makes a query portable across chains of different lengths and what
//! PostGIS's `ST_LineInterpolatePoint` / `ST_LineLocatePoint` pair standardizes on.
//!
//! Every function here operates on a single chain. A multi-chain input has no single
//! parameterization — there is no defined order in which its members are traversed —
//! so it is refused rather than silently answered against whichever member happens to
//! be first.

use crate::algo::primitive::{closest_on_segment, dist};
use crate::error::{GeoError, GeoResult};
use crate::types::{Coord, Geometry, LineString};

/// The single chain of a geometry, or an error naming what was passed instead.
fn require_line<'a>(g: &'a Geometry, op: &'static str) -> GeoResult<&'a LineString> {
    match g {
        Geometry::LineString(l) if l.len() >= 2 => Ok(l),
        Geometry::LineString(_) => Err(GeoError::invalid(format!(
            "{op} needs a line with at least two positions"
        ))),
        other => Err(GeoError::Unsupported {
            op,
            geom_type: crate::Geom::new(other.clone()).geom_type().name(),
        }),
    }
}

/// The cumulative length at each vertex, and the total.
fn measures(l: &LineString) -> (Vec<f64>, f64) {
    let mut acc = Vec::with_capacity(l.len());
    let mut total = 0.0;
    acc.push(0.0);
    for w in l.windows(2) {
        total += dist(w[0], w[1]);
        acc.push(total);
    }
    (acc, total)
}

/// The position `fraction` of the way along the chain.
///
/// A zero-length chain answers with its own start rather than erroring: every position
/// on it is the same position, so the fraction is meaningless but the answer is not.
pub fn interpolate_point(g: &Geometry, fraction: f64) -> GeoResult<Geometry> {
    let l = require_line(g, "st_line_interpolate_point")?;
    if !(0.0..=1.0).contains(&fraction) {
        return Err(GeoError::invalid(format!(
            "line fraction must be in [0, 1], got {fraction}"
        )));
    }
    let (acc, total) = measures(l);
    if total == 0.0 {
        return Ok(Geometry::Point(Some(l[0])));
    }
    let target = fraction * total;
    Ok(Geometry::Point(Some(at_measure(l, &acc, target))))
}

/// The position at cumulative distance `target` along the chain.
fn at_measure(l: &LineString, acc: &[f64], target: f64) -> Coord {
    for i in 0..l.len() - 1 {
        if target <= acc[i + 1] || i == l.len() - 2 {
            let seg = acc[i + 1] - acc[i];
            let t = if seg == 0.0 {
                0.0
            } else {
                ((target - acc[i]) / seg).clamp(0.0, 1.0)
            };
            return Coord {
                x: l[i].x + t * (l[i + 1].x - l[i].x),
                y: l[i].y + t * (l[i + 1].y - l[i].y),
                z: l[i].z + t * (l[i + 1].z - l[i].z),
            };
        }
    }
    l[l.len() - 1]
}

/// The fraction along the chain at which it passes closest to `p`.
pub fn locate_point(g: &Geometry, p: Coord) -> GeoResult<f64> {
    let l = require_line(g, "st_line_locate_point")?;
    let (acc, total) = measures(l);
    if total == 0.0 {
        return Ok(0.0);
    }
    let mut best = (f64::INFINITY, 0.0);
    for i in 0..l.len() - 1 {
        let (q, t) = closest_on_segment(p, l[i], l[i + 1]);
        let d = dist(p, q);
        if d < best.0 {
            best = (d, acc[i] + t * (acc[i + 1] - acc[i]));
        }
    }
    Ok((best.1 / total).clamp(0.0, 1.0))
}

/// The stretch of chain between two fractions.
///
/// The endpoints are inserted as real vertices, so the result starts and ends exactly
/// where asked rather than at the nearest original vertex — which matters when the
/// substring is then measured or joined.
pub fn substring(g: &Geometry, from: f64, to: f64) -> GeoResult<Geometry> {
    let l = require_line(g, "st_line_substring")?;
    if !(0.0..=1.0).contains(&from) || !(0.0..=1.0).contains(&to) {
        return Err(GeoError::invalid(format!(
            "line fractions must be in [0, 1], got {from} and {to}"
        )));
    }
    let (lo, hi) = if from <= to { (from, to) } else { (to, from) };
    let (acc, total) = measures(l);
    if total == 0.0 {
        return Ok(Geometry::Point(Some(l[0])));
    }
    let (start, end) = (lo * total, hi * total);
    if start == end {
        return Ok(Geometry::Point(Some(at_measure(l, &acc, start))));
    }
    let mut out = vec![at_measure(l, &acc, start)];
    for (i, m) in acc.iter().enumerate() {
        if *m > start && *m < end {
            out.push(l[i]);
        }
    }
    out.push(at_measure(l, &acc, end));
    Ok(Geometry::LineString(out))
}

/// Insert vertices so no segment is longer than `max_len`.
///
/// The prerequisite for any projection or geodesic measurement of a long segment: a
/// straight line in one CRS is not straight in another, so a two-vertex transatlantic
/// segment reprojects to a visibly wrong path. Densifying first fixes it, and the
/// segments are split evenly rather than at `max_len` steps with a short remainder,
/// which keeps the added vertices regular.
pub fn segmentize(g: &Geometry, max_len: f64) -> GeoResult<Geometry> {
    if max_len.is_nan() || max_len <= 0.0 {
        return Err(GeoError::invalid(format!(
            "segment length must be positive, got {max_len}"
        )));
    }
    fn densify(l: &LineString, max_len: f64) -> LineString {
        if l.len() < 2 {
            return l.clone();
        }
        let mut out = Vec::with_capacity(l.len());
        for w in l.windows(2) {
            out.push(w[0]);
            let d = dist(w[0], w[1]);
            if d > max_len {
                let n = (d / max_len).ceil() as usize;
                for k in 1..n {
                    let t = k as f64 / n as f64;
                    out.push(Coord {
                        x: w[0].x + t * (w[1].x - w[0].x),
                        y: w[0].y + t * (w[1].y - w[0].y),
                        z: w[0].z + t * (w[1].z - w[0].z),
                    });
                }
            }
        }
        out.push(l[l.len() - 1]);
        out
    }
    fn walk(g: &Geometry, max_len: f64) -> Geometry {
        match g {
            Geometry::LineString(l) => Geometry::LineString(densify(l, max_len)),
            Geometry::MultiLineString(ls) => {
                Geometry::MultiLineString(ls.iter().map(|l| densify(l, max_len)).collect())
            }
            Geometry::Polygon(p) => Geometry::Polygon(crate::types::Polygon {
                exterior: densify(&p.exterior, max_len),
                interiors: p.interiors.iter().map(|r| densify(r, max_len)).collect(),
            }),
            Geometry::MultiPolygon(ps) => Geometry::MultiPolygon(
                ps.iter()
                    .map(|p| crate::types::Polygon {
                        exterior: densify(&p.exterior, max_len),
                        interiors: p.interiors.iter().map(|r| densify(r, max_len)).collect(),
                    })
                    .collect(),
            ),
            Geometry::GeometryCollection(gs) => {
                Geometry::GeometryCollection(gs.iter().map(|c| walk(c, max_len)).collect())
            }
            other => other.clone(),
        }
    }
    Ok(walk(g, max_len))
}

/// The point of `g` closest to `other`'s nearest position.
///
/// Asymmetric on purpose, like PostGIS `ST_ClosestPoint`: it returns a point *on the
/// first geometry*, which is what a snap-to-road or snap-to-boundary step needs.
pub fn closest_point(g: &Geometry, other: &Geometry) -> Option<Geometry> {
    let targets: Vec<Coord> = {
        let mut v = Vec::new();
        other.collect_coords(&mut v);
        v
    };
    if targets.is_empty() {
        return None;
    }
    let mut best: Option<(f64, Coord)> = None;
    let mut consider = |c: Coord, to: Coord| {
        let d = dist(c, to);
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, c));
        }
    };
    for t in &targets {
        for l in g.lines() {
            if l.len() == 1 {
                consider(l[0], *t);
            }
            for w in l.windows(2) {
                consider(closest_on_segment(*t, w[0], w[1]).0, *t);
            }
        }
        for p in g.points() {
            consider(p, *t);
        }
    }
    best.map(|(_, c)| Geometry::Point(Some(c)))
}

/// The two-point line joining the closest positions of two geometries.
pub fn shortest_line(a: &Geometry, b: &Geometry) -> Option<Geometry> {
    let pa = match closest_point(a, b)? {
        Geometry::Point(Some(c)) => c,
        _ => return None,
    };
    let pb = match closest_point(b, &Geometry::Point(Some(pa)))? {
        Geometry::Point(Some(c)) => c,
        _ => return None,
    };
    Some(Geometry::LineString(vec![pa, pb]))
}

/// A position guaranteed to lie on the geometry — inside it when it has an interior.
///
/// Unlike a centroid, which can fall outside a crescent or in a hole, this is always
/// *on* the geometry. It is what you label a shape with, and what you use as a
/// representative point for a spatial join that must not miss.
pub fn point_on_surface(g: &Geometry) -> Option<Geometry> {
    crate::algo::relate::interior_point(g).map(|c| Geometry::Point(Some(c)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::wkt::{read_wkt, write_wkt};
    use crate::Geom;

    fn geom(t: &str) -> Geometry {
        read_wkt(t).expect(t).geometry
    }

    fn wkt(g: Geometry) -> String {
        write_wkt(&Geom::new(g))
    }

    const L: &str = "LINESTRING(0 0, 10 0, 10 10)";

    #[test]
    fn interpolate_and_locate_are_inverses() {
        let l = geom(L);
        for f in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let p = interpolate_point(&l, f).unwrap();
            let c = Geom::new(p).coords()[0];
            let back = locate_point(&l, c).unwrap();
            assert!((back - f).abs() < 1e-9, "{f} -> {back}");
        }
    }

    #[test]
    fn interpolate_lands_where_hand_computation_says() {
        assert_eq!(
            wkt(interpolate_point(&geom(L), 0.5).unwrap()),
            "POINT(10 0)"
        );
        assert_eq!(wkt(interpolate_point(&geom(L), 0.0).unwrap()), "POINT(0 0)");
        assert_eq!(
            wkt(interpolate_point(&geom(L), 1.0).unwrap()),
            "POINT(10 10)"
        );
        assert!(interpolate_point(&geom(L), 1.5).is_err());
    }

    #[test]
    fn a_multi_chain_is_refused_rather_than_answered_arbitrarily() {
        let err =
            interpolate_point(&geom("MULTILINESTRING((0 0, 1 0), (5 5, 6 5))"), 0.5).unwrap_err();
        assert!(matches!(err, GeoError::Unsupported { .. }), "{err:?}");
    }

    #[test]
    fn substring_inserts_exact_endpoints() {
        let s = substring(&geom(L), 0.25, 0.75).unwrap();
        assert_eq!(wkt(s), "LINESTRING(5 0, 10 0, 10 5)");
        // Reversed fractions are normalized rather than producing an empty result.
        assert_eq!(
            wkt(substring(&geom(L), 0.75, 0.25).unwrap()),
            "LINESTRING(5 0, 10 0, 10 5)"
        );
    }

    #[test]
    fn segmentize_bounds_every_segment() {
        let s = segmentize(&geom("LINESTRING(0 0, 10 0)"), 3.0).unwrap();
        let coords = Geom::new(s).coords();
        for w in coords.windows(2) {
            assert!(dist(w[0], w[1]) <= 3.0 + 1e-9);
        }
        assert!(segmentize(&geom(L), 0.0).is_err());
    }

    #[test]
    fn segmentize_leaves_short_segments_alone() {
        let l = geom("LINESTRING(0 0, 1 0)");
        assert_eq!(segmentize(&l, 5.0).unwrap(), l);
    }

    #[test]
    fn closest_point_returns_a_point_on_the_first_geometry() {
        let road = geom("LINESTRING(0 0, 10 0)");
        let p = closest_point(&road, &geom("POINT(4 7)")).unwrap();
        assert_eq!(wkt(p), "POINT(4 0)");
        assert_eq!(
            wkt(shortest_line(&road, &geom("POINT(4 7)")).unwrap()),
            "LINESTRING(4 0, 4 7)"
        );
    }

    #[test]
    fn point_on_surface_is_inside_a_crescent_where_the_centroid_is_not() {
        let c = geom("POLYGON((0 0, 10 0, 10 2, 2 2, 2 8, 10 8, 10 10, 0 10, 0 0))");
        let p = point_on_surface(&c).unwrap();
        let coord = Geom::new(p).coords()[0];
        assert_eq!(
            crate::algo::relate::point_in_geometry(coord, &c),
            crate::algo::primitive::PointRing::Inside
        );
    }
}
