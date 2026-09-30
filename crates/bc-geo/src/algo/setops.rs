//! Union, intersection and difference of two areal geometries: the public overlay.
//!
//! `overlay` already computes a union of polygons and a polygon minus a set of polygons,
//! by noding, edge classification and ring tracing; `buffer` has been built on it from
//! the start. This module exposes the same machinery as the three set operations a
//! clipping or erase workflow needs, for two polygons or multipolygons:
//!
//! * `A ∪ B` is one union of every member polygon of both.
//! * `A − B` is each member of `A` minus every member of `B`.
//! * `A ∩ B` is `A − (A − B)`, member by member. It needs no third keep rule, and it
//!   inherits the difference's handling of shared and touching edges, which `buffer`'s
//!   negative radius exercises on every call.
//!
//! # What is refused
//!
//! The operands must be areal: a `POLYGON`, a `MULTIPOLYGON`, or a collection holding
//! only those. A point or a chain has no area to combine, and GEOS's mixed-dimension
//! results (a polygon with a dangling line) are a different operation. An operand that
//! is not valid by `validity::validity_reason` is refused too, because the overlay's edge
//! classification assumes each ring bounds its interior once; a bowtie would give an
//! answer with no meaning. Both refusals are `Domain` errors, which the expression layer
//! turns into a null for that row, as it does for a trace the overlay cannot close.
//!
//! An empty result is `POLYGON EMPTY`, the typed empty geometry GEOS writes for a
//! polygonal overlay, not a null: two disjoint parcels have an intersection, and it is
//! empty.

use crate::algo::{buffer::from_polygons, overlay, validity};
use crate::error::{GeoError, GeoResult};
use crate::types::{Geometry, Polygon};
use crate::Geom;

/// Which set operation to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOp {
    /// Every point in either operand.
    Union,
    /// Every point in both operands.
    Intersection,
    /// Every point in the first operand and not the second.
    Difference,
}

/// `a op b`, as a `Polygon` (possibly `POLYGON EMPTY`) or a `MultiPolygon`.
///
/// # Errors
///
/// `Domain` when an operand is not areal, is not valid, or when the overlay cannot trace
/// its result robustly.
pub fn set_op(op: SetOp, a: &Geom, b: &Geom) -> GeoResult<Geometry> {
    let (pa, pb) = (areal(a)?, areal(b)?);
    let failed = || GeoError::domain("the overlay could not be traced robustly");
    let out: Vec<Polygon> = match op {
        SetOp::Union => {
            let all: Vec<Polygon> = pa.into_iter().chain(pb).collect();
            if all.is_empty() {
                Vec::new()
            } else {
                overlay::union(&all).ok_or_else(failed)?
            }
        }
        SetOp::Difference => {
            let mut out = Vec::new();
            for p in &pa {
                if pb.is_empty() {
                    out.push(p.clone());
                } else {
                    out.extend(overlay::difference(p, &pb).ok_or_else(failed)?);
                }
            }
            out
        }
        SetOp::Intersection => {
            let mut out = Vec::new();
            if !pb.is_empty() {
                for p in &pa {
                    let outside = overlay::difference(p, &pb).ok_or_else(failed)?;
                    if outside.is_empty() {
                        out.push(p.clone());
                    } else {
                        out.extend(overlay::difference(p, &outside).ok_or_else(failed)?);
                    }
                }
            }
            out
        }
    };
    Ok(from_polygons(out))
}

/// The non-empty member polygons of an areal operand, or `Domain` for anything else.
fn areal(g: &Geom) -> GeoResult<Vec<Polygon>> {
    if let Some(reason) = validity::validity_reason(g) {
        return Err(GeoError::domain(format!(
            "an overlay operand must be valid: {reason}"
        )));
    }
    let mut out = Vec::new();
    collect(&g.geometry, &mut out)?;
    Ok(out)
}

fn collect(g: &Geometry, out: &mut Vec<Polygon>) -> GeoResult<()> {
    match g {
        Geometry::Polygon(p) => {
            if !p.exterior.is_empty() {
                out.push(p.clone());
            }
        }
        Geometry::MultiPolygon(ps) => {
            out.extend(ps.iter().filter(|p| !p.exterior.is_empty()).cloned())
        }
        Geometry::GeometryCollection(gs) => {
            for c in gs {
                collect(c, out)?;
            }
        }
        Geometry::Point(None) => {}
        Geometry::MultiPoint(ps) if ps.is_empty() => {}
        Geometry::LineString(l) if l.is_empty() => {}
        Geometry::MultiLineString(ls) if ls.is_empty() => {}
        _ => {
            return Err(GeoError::domain(
                "an overlay operand must be a polygon or multipolygon",
            ))
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::wkt::read_wkt;
    use crate::types::ring_area;

    fn area(g: &Geometry) -> f64 {
        g.polygons()
            .iter()
            .map(|p| {
                ring_area(&p.exterior).abs()
                    - p.interiors.iter().map(|h| ring_area(h).abs()).sum::<f64>()
            })
            .sum()
    }

    fn run(op: SetOp, a: &str, b: &str) -> Geometry {
        set_op(op, &read_wkt(a).unwrap(), &read_wkt(b).unwrap()).unwrap()
    }

    const A: &str = "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))";
    const B: &str = "POLYGON((2 2, 6 2, 6 6, 2 6, 2 2))";

    #[test]
    fn two_overlapping_squares() {
        assert!((area(&run(SetOp::Intersection, A, B)) - 4.0).abs() < 1e-9);
        assert!((area(&run(SetOp::Union, A, B)) - 28.0).abs() < 1e-9);
        assert!((area(&run(SetOp::Difference, A, B)) - 12.0).abs() < 1e-9);
        assert!((area(&run(SetOp::Difference, B, A)) - 12.0).abs() < 1e-9);
    }

    #[test]
    fn inclusion_exclusion_holds_for_a_holed_polygon() {
        let holed = "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (3 3, 7 3, 7 7, 3 7, 3 3))";
        let bar = "POLYGON((-1 4, 11 4, 11 6, -1 6, -1 4))";
        let (a, b) = (84.0, 24.0);
        let i = area(&run(SetOp::Intersection, holed, bar));
        let u = area(&run(SetOp::Union, holed, bar));
        let d = area(&run(SetOp::Difference, holed, bar));
        assert!((i - 12.0).abs() < 1e-9, "{i}");
        assert!((u - (a + b - i)).abs() < 1e-9, "{u}");
        assert!((d - (a - i)).abs() < 1e-9, "{d}");
    }

    #[test]
    fn disjoint_operands_intersect_in_the_empty_polygon() {
        let far = "POLYGON((10 10, 11 10, 11 11, 10 11, 10 10))";
        let got = run(SetOp::Intersection, A, far);
        assert_eq!(got, Geometry::Polygon(Polygon::default()));
        assert!((area(&run(SetOp::Difference, A, far)) - 16.0).abs() < 1e-9);
        assert!(
            matches!(run(SetOp::Union, A, far), Geometry::MultiPolygon(ref ps) if ps.len() == 2)
        );
    }

    #[test]
    fn a_contained_operand_is_the_intersection() {
        let inner = "POLYGON((1 1, 2 1, 2 2, 1 2, 1 1))";
        assert!((area(&run(SetOp::Intersection, A, inner)) - 1.0).abs() < 1e-9);
        assert!((area(&run(SetOp::Intersection, inner, A)) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn identical_operands() {
        assert!((area(&run(SetOp::Intersection, A, A)) - 16.0).abs() < 1e-9);
        assert!((area(&run(SetOp::Union, A, A)) - 16.0).abs() < 1e-9);
        assert_eq!(
            run(SetOp::Difference, A, A),
            Geometry::Polygon(Polygon::default())
        );
    }

    #[test]
    fn a_line_or_an_invalid_polygon_is_refused() {
        let line = read_wkt("LINESTRING(0 0, 1 1)").unwrap();
        let bowtie = read_wkt("POLYGON((0 0, 4 4, 4 0, 0 4, 0 0))").unwrap();
        let a = read_wkt(A).unwrap();
        for bad in [&line, &bowtie] {
            let e = set_op(SetOp::Union, &a, bad).unwrap_err();
            assert!(e.is_row_local(), "{e}");
        }
    }
}
