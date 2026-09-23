//! `ST_Buffer`: every position within a distance of a geometry, as a polygon.
//!
//! A positive buffer is a union. A point grows into a disc; a chain into a disc at every
//! vertex plus a rectangle along every segment (together, a capsule per segment); a
//! polygon into itself plus the same band around each of its rings. `overlay::union`
//! merges the pieces, so parts of a multi-geometry join exactly where their buffers
//! overlap and stay separate polygons where they do not.
//!
//! A negative buffer erodes a polygon: `overlay::difference` removes that same band from
//! it. The result keeps sharp convex corners and rounds the reflex ones, which is the
//! erosion of a polygon by a disc; a band wider than the polygon leaves nothing.
//!
//! The only approximation is the one every implementation makes: a circle is a regular
//! polygon with `4 * quad_segs` sides, with vertices at the same angles GEOS uses, so a
//! point's buffer is vertex-for-vertex GEOS's. Where GEOS draws a round join as an arc
//! starting exactly at a segment's offset point, this draws the vertex disc, so the areas
//! of a chain's or polygon's buffer differ from GEOS's by a fraction of the area
//! between a circle and its inscribed polygon — under 0.3% of `pi * r^2` per vertex at
//! the default eight segments per quadrant.
//!
//! This replaced a convex hull of the vertex discs, which was exact for a convex input and
//! silently wrong for everything else: `MULTIPOINT((0 0), (10 0))` buffered by 1 had an
//! area of 23.1 instead of 6.2, and a concave polygon's buffer filled in its notch.

use crate::algo::overlay;
use crate::error::{GeoError, GeoResult};
use crate::types::{Coord, Geometry, LineString, Polygon};
use crate::Geom;

/// A buffer of `radius` around the geometry, with `quad_segs` segments per quarter
/// circle.
///
/// * `radius > 0` — the union of the pieces described in the module documentation.
/// * `radius == 0` — the areal part of the input, unchanged; a point or chain has no
///   area, so it buffers to `POLYGON EMPTY` (GEOS, PostGIS and DuckDB agree).
/// * `radius < 0` — each polygon eroded by `|radius|`; points and chains, and polygons
///   narrower than twice the radius, leave `POLYGON EMPTY`.
///
/// A NaN radius or zero `quad_segs` is a caller error. An input the overlay cannot trace
/// robustly is `Domain` — a null for that row — rather than a polygon with a piece
/// silently missing.
pub fn buffer(g: &Geom, radius: f64, quad_segs: usize) -> GeoResult<Geometry> {
    if !radius.is_finite() {
        return Err(GeoError::invalid(format!(
            "buffer radius must be a finite number, got {radius}"
        )));
    }
    if quad_segs == 0 {
        return Err(GeoError::invalid(
            "buffer needs at least one segment per quadrant",
        ));
    }
    let mut points = Vec::new();
    let mut lines = Vec::new();
    let mut polygons = Vec::new();
    collect_parts(&g.geometry, &mut points, &mut lines, &mut polygons);
    if radius == 0.0 {
        return Ok(from_polygons(polygons.into_iter().cloned().collect()));
    }
    let steps = quad_segs * 4;
    let failed = || GeoError::domain("buffer could not be traced robustly for this geometry");
    if radius < 0.0 {
        let r = -radius;
        let mut out = Vec::new();
        for p in polygons {
            let band = ring_band(p, r, steps);
            out.extend(overlay::difference(p, &band).ok_or_else(failed)?);
        }
        return Ok(from_polygons(out));
    }
    let mut pieces: Vec<Polygon> = Vec::new();
    for c in points {
        pieces.push(disc(c, radius, steps));
    }
    for l in lines {
        chain_band(l, radius, steps, &mut pieces);
    }
    for p in polygons {
        if p.exterior.is_empty() {
            continue;
        }
        pieces.push(p.clone());
        pieces.extend(ring_band(p, radius, steps));
    }
    if pieces.is_empty() {
        return Ok(Geometry::Polygon(Polygon::default()));
    }
    Ok(from_polygons(overlay::union(&pieces).ok_or_else(failed)?))
}

fn collect_parts<'a>(
    g: &'a Geometry,
    points: &mut Vec<Coord>,
    lines: &mut Vec<&'a LineString>,
    polygons: &mut Vec<&'a Polygon>,
) {
    match g {
        Geometry::Point(p) => points.extend(p.iter().copied().filter(|c| !c.is_nan())),
        Geometry::MultiPoint(ps) => points.extend(ps.iter().flatten().filter(|c| !c.is_nan())),
        Geometry::LineString(l) => lines.push(l),
        Geometry::MultiLineString(ls) => lines.extend(ls.iter()),
        Geometry::Polygon(p) => polygons.push(p),
        Geometry::MultiPolygon(ps) => polygons.extend(ps.iter()),
        Geometry::GeometryCollection(gs) => {
            for c in gs {
                collect_parts(c, points, lines, polygons);
            }
        }
    }
}

fn from_polygons(mut ps: Vec<Polygon>) -> Geometry {
    ps.retain(|p| !p.exterior.is_empty());
    match ps.len() {
        0 => Geometry::Polygon(Polygon::default()),
        1 => Geometry::Polygon(ps.remove(0)),
        _ => Geometry::MultiPolygon(ps),
    }
}

/// A regular `steps`-gon of circumradius `r` around `c`, counter-clockwise, starting at
/// angle 0 as GEOS's does.
fn disc(c: Coord, r: f64, steps: usize) -> Polygon {
    let mut ring: LineString = (0..steps)
        .map(|k| {
            let theta = std::f64::consts::TAU * (k as f64) / (steps as f64);
            Coord::new(c.x + r * theta.cos(), c.y + r * theta.sin())
        })
        .collect();
    ring.push(ring[0]);
    Polygon {
        exterior: ring,
        interiors: Vec::new(),
    }
}

/// The rectangle of half-width `r` along segment `a -> b`, or `None` when it is a point.
fn segment_rect(a: Coord, b: Coord, r: f64) -> Option<Polygon> {
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    let len = dx.hypot(dy);
    if len == 0.0 || !len.is_finite() {
        return None;
    }
    let (nx, ny) = (-dy / len * r, dx / len * r);
    Some(Polygon {
        exterior: vec![
            Coord::new(a.x - nx, a.y - ny),
            Coord::new(b.x - nx, b.y - ny),
            Coord::new(b.x + nx, b.y + ny),
            Coord::new(a.x + nx, a.y + ny),
            Coord::new(a.x - nx, a.y - ny),
        ],
        interiors: Vec::new(),
    })
}

/// The discs and rectangles whose union is every position within `r` of a chain.
fn chain_band(l: &[Coord], r: f64, steps: usize, out: &mut Vec<Polygon>) {
    let pts: Vec<Coord> = l.iter().copied().filter(|c| !c.is_nan()).collect();
    for (i, c) in pts.iter().enumerate() {
        // A vertex repeated by the next one would contribute the same disc twice.
        if i == 0 || pts[i - 1].x != c.x || pts[i - 1].y != c.y {
            out.push(disc(*c, r, steps));
        }
    }
    out.extend(pts.windows(2).filter_map(|w| segment_rect(w[0], w[1], r)));
}

/// The band of width `r` either side of every ring of a polygon.
fn ring_band(p: &Polygon, r: f64, steps: usize) -> Vec<Polygon> {
    let mut out = Vec::new();
    for ring in std::iter::once(&p.exterior).chain(p.interiors.iter()) {
        // The closing vertex repeats the first; its disc is already there.
        let open = if crate::types::is_closed(ring) && ring.len() > 1 {
            &ring[..ring.len() - 1]
        } else {
            &ring[..]
        };
        chain_band(open, r, steps, &mut out);
        if let (Some(&a), Some(&b)) = (open.last(), open.first()) {
            out.extend(segment_rect(a, b, r));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::measure::area;
    use crate::codec::wkt::{read_wkt, write_wkt};

    fn buf(t: &str, r: f64, q: usize) -> Geometry {
        buffer(&read_wkt(t).unwrap(), r, q).unwrap()
    }

    fn rel(got: f64, want: f64) -> f64 {
        ((got - want) / want).abs()
    }

    #[test]
    fn a_point_buffer_is_the_geos_polygon() {
        // DuckDB ST_Area(ST_Buffer(POINT(0 0), 1, 2)) = 2.82842712474619, and with 8
        // segments per quadrant a 32-gon.
        assert!(rel(area(&buf("POINT(0 0)", 1.0, 2)), 2.828_427_124_746_19) < 1e-12);
        let n = 32.0_f64;
        let want = 0.5 * n * (std::f64::consts::TAU / n).sin();
        assert!(rel(area(&buf("POINT(0 0)", 1.0, 8)), want) < 1e-12);
    }

    #[test]
    fn separate_parts_buffer_separately_rather_than_into_one_hull() {
        // The hull-based buffer said 23.1 for this; two discs are 2 * 3.1214.
        let g = buf("MULTIPOINT((0 0), (10 0))", 1.0, 8);
        assert!(
            matches!(g, Geometry::MultiPolygon(ref ps) if ps.len() == 2),
            "{g:?}"
        );
        assert!(
            rel(area(&g), 2.0 * 3.121_445_152_258_052) < 1e-9,
            "{}",
            area(&g)
        );
        // Overlapping discs merge into one polygon whose area is less than two discs.
        let m = buf("MULTIPOINT((0 0), (1 0))", 1.0, 8);
        assert!(matches!(m, Geometry::Polygon(_)), "{m:?}");
        assert!(area(&m) < 2.0 * 3.1215 && area(&m) > 3.1215);
    }

    #[test]
    fn a_concave_polygon_keeps_its_notch() {
        // DuckDB (GEOS) gives 80.93 for this; the hull-based buffer gave 120.8.
        let g = buf("POLYGON((0 0, 10 0, 10 10, 5 1, 0 10, 0 0))", 0.5, 8);
        assert!(rel(area(&g), 80.93) < 0.005, "{}", area(&g));
    }

    #[test]
    fn a_line_buffer_is_a_capsule() {
        // A 10-long segment, radius 1: a 20 x 1 rectangle plus two half 32-gons.
        let g = buf("LINESTRING(0 0, 10 0)", 1.0, 8);
        let n = 32.0_f64;
        let disc = 0.5 * n * (std::f64::consts::TAU / n).sin();
        assert!(rel(area(&g), 20.0 + disc) < 1e-9, "{}", area(&g));
        // An L-shaped chain: its buffer is not its hull's.
        let l = buf("LINESTRING(0 0, 10 0, 10 10)", 1.0, 8);
        assert!(area(&l) < 45.0, "{}", area(&l));
    }

    #[test]
    fn negative_buffers_erode_and_zero_returns_the_input() {
        let sq = "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))";
        let e = buf(sq, -1.0, 8);
        assert_eq!(
            write_wkt(&Geom::new(e)),
            "POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))"
        );
        assert!(matches!(buf(sq, -3.0, 8), Geometry::Polygon(ref p) if p.exterior.is_empty()));
        assert_eq!(buf(sq, 0.0, 8), read_wkt(sq).unwrap().geometry);
        assert!(
            matches!(buf("POINT(1 1)", 0.0, 8), Geometry::Polygon(ref p) if p.exterior.is_empty())
        );
        assert!(
            matches!(buf("POINT(1 1)", -1.0, 8), Geometry::Polygon(ref p) if p.exterior.is_empty())
        );
        // An L-shaped polygon eroded: the reflex corner rounds, the convex ones stay sharp.
        let l = buf("POLYGON((0 0, 6 0, 6 2, 2 2, 2 6, 0 6, 0 0))", -0.5, 8);
        // Exact erosion: the L of arm width 1 (areas 5x1 + 1x4 = 9) plus the reflex
        // corner's quarter disc of radius 0.5 minus its square.
        let want = 9.0 + 0.25 - std::f64::consts::PI * 0.25 / 4.0;
        assert!(rel(area(&l), want) < 0.01, "{} vs {want}", area(&l));
    }

    #[test]
    fn a_nan_radius_or_no_segments_is_a_caller_error() {
        let pt = read_wkt("POINT(0 0)").unwrap();
        assert!(buffer(&pt, 1.0, 0).is_err());
        let e = buffer(&pt, f64::NAN, 4).unwrap_err();
        assert!(!e.is_row_local());
    }
}
