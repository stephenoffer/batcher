//! The scalar-returning geospatial functions: accessors, measures, predicates, codecs.
//!
//! Grouped here because they share the one thing that matters at the array level — the
//! output is a plain typed column, not WKB — and split from each other only by which
//! builder they append to. The output type is a property of the function alone, which is
//! what `output_type` states and what lets the planner know a column's type before a row
//! is read.

use arrow::array::{ArrayRef, BinaryBuilder, BooleanBuilder, Float64Builder, Int64Builder, StringBuilder};
use std::sync::Arc;

use bc_geo::algo::{linear, measure, predicate, validity};
use bc_geo::proj::geodesy;
use bc_geo::Geom;

use crate::{ExprError, GeoFunc};

use super::{caller_error, f64_at, geom_at, row_result, ScalarOut};

/// Which typed column a function produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Out {
    Float,
    Int,
    Bool,
    Text,
    /// Raw WKB bytes. Distinct from a *geometry* output: `st_as_binary` exists precisely
    /// to control the encoding, so it cannot go through the geometry builder, which
    /// always writes EWKB to keep an SRID alive.
    Bytes,
}

fn output_type(func: GeoFunc) -> Out {
    use GeoFunc::*;
    match func {
        StX | StY | StZ | StXmin | StYmin | StXmax | StYmax | StArea | StLength
        | StPerimeter | StDistance | StMaxDistance | StHausdorffDistance | StAzimuth
        | StDistanceSphere | StDistanceSpheroid | StAreaSpheroid | StLengthSpheroid
        | StPerimeterSpheroid | StLineLocatePoint => Out::Float,

        StDimension | StSrid | StNumPoints | StNumGeometries | StNumInteriorRings
        | StCoordDim => Out::Int,

        StIsEmpty | StIsValid | StIsClosed | StIsRing | StIsSimple | StIsCollection
        | StHasZ | StIntersects | StDisjoint | StContains | StWithin | StCovers
        | StCoveredBy | StTouches | StCrosses | StOverlaps | StEquals | StDwithin
        | StDwithinSphere | StIntersectsExtent | StContainsExtent => Out::Bool,

        StAsText | StAsEwkt | StAsHexWkb | StAsGeojson | StGeometryType
        | StIsValidReason => Out::Text,

        StAsBinary | StAsEwkb => Out::Bytes,

        // Everything else is routed to `build` or `grid` before reaching here.
        other => unreachable!("{other:?} is not a scalar-returning geo function"),
    }
}

/// Evaluate a scalar-returning function over `rows` rows of `cols`.
pub(super) fn eval(
    func: GeoFunc,
    cols: &[ArrayRef],
    rows: usize,
) -> Result<ArrayRef, ExprError> {
    if output_type(func) == Out::Bytes {
        return eval_bytes(func, cols, rows);
    }
    let mut out = match output_type(func) {
        Out::Float => ScalarOut::Float(Float64Builder::with_capacity(rows)),
        Out::Int => ScalarOut::Int(Int64Builder::with_capacity(rows)),
        Out::Bool => ScalarOut::Bool(BooleanBuilder::with_capacity(rows)),
        Out::Text => ScalarOut::Text(StringBuilder::with_capacity(rows, rows * 24)),
        Out::Bytes => unreachable!("handled above"),
    };
    for i in 0..rows {
        let a = geom_at(&cols[0], i, func)?;
        match output_type(func) {
            Out::Float => out.push_f64(float_of(func, a.as_ref(), cols, i)?),
            Out::Int => out.push_i64(int_of(func, a.as_ref())),
            Out::Bool => out.push_bool(bool_of(func, a.as_ref(), cols, i)?),
            Out::Text => {
                let s = text_of(func, a.as_ref());
                out.push_str(s.as_deref());
            }
            Out::Bytes => unreachable!("handled above"),
        }
    }
    Ok(out.finish())
}

/// `st_as_binary` / `st_as_ewkb`, which produce raw bytes rather than a typed scalar.
fn eval_bytes(func: GeoFunc, cols: &[ArrayRef], rows: usize) -> Result<ArrayRef, ExprError> {
    let mut b = BinaryBuilder::with_capacity(rows, rows * 48);
    for i in 0..rows {
        match geom_at(&cols[0], i, func)? {
            Some(g) => {
                let bytes = if func == GeoFunc::StAsEwkb {
                    bc_geo::codec::wkb::write_ewkb(&g)
                } else {
                    bc_geo::codec::wkb::write_wkb(&g)
                };
                b.append_value(bytes);
            }
            None => b.append_null(),
        }
    }
    Ok(Arc::new(b.finish()))
}

/// The Float64-valued functions.
fn float_of(
    func: GeoFunc,
    g: Option<&Geom>,
    cols: &[ArrayRef],
    i: usize,
) -> Result<Option<f64>, ExprError> {
    use GeoFunc::*;
    let Some(g) = g else { return Ok(None) };
    // The single-position accessors are null unless the geometry really is one point;
    // `ST_X` of a polygon is not its first vertex, it is undefined.
    let single = || match &g.geometry {
        bc_geo::Geometry::Point(p) => *p,
        _ => None,
    };
    Ok(match func {
        StX => single().map(|c| c.x),
        StY => single().map(|c| c.y),
        StZ => g.has_z.then(single).flatten().map(|c| c.z),
        StXmin => g.bbox().map(|b| b.xmin),
        StYmin => g.bbox().map(|b| b.ymin),
        StXmax => g.bbox().map(|b| b.xmax),
        StYmax => g.bbox().map(|b| b.ymax),
        StArea => Some(measure::area(&g.geometry)),
        StLength => Some(measure::length(&g.geometry)),
        StPerimeter => Some(measure::perimeter(&g.geometry)),
        StAreaSpheroid => Some(geodesy::geodesic_area_m2(&g.geometry)),
        StLengthSpheroid => row_result(geodesy::geodesic_length_m(&g.geometry), func)?,
        StPerimeterSpheroid => row_result(geodesy::geodesic_perimeter_m(&g.geometry), func)?,
        StLineLocatePoint => {
            let Some(p) = geom_at(&cols[1], i, func)? else {
                return Ok(None);
            };
            let Some(c) = p.geometry.points().first().copied() else {
                return Ok(None);
            };
            match linear::locate_point(&g.geometry, c) {
                Ok(v) => Some(v),
                Err(e) if e.is_row_local() => None,
                Err(e) => return Err(caller_error(func, e)),
            }
        }
        StDistance | StMaxDistance | StHausdorffDistance | StAzimuth | StDistanceSphere
        | StDistanceSpheroid => {
            let Some(b) = geom_at(&cols[1], i, func)? else {
                return Ok(None);
            };
            match func {
                StDistance => measure::distance(g, &b),
                StMaxDistance => measure::max_distance(g, &b),
                StHausdorffDistance => measure::hausdorff_distance(g, &b),
                StAzimuth => {
                    let (Some(p), Some(q)) = (
                        g.geometry.points().first().copied(),
                        b.geometry.points().first().copied(),
                    ) else {
                        return Ok(None);
                    };
                    measure::azimuth(p, q)
                }
                StDistanceSphere => nearest_geodesic(g, &b, false)?,
                _ => nearest_geodesic(g, &b, true)?,
            }
        }
        other => unreachable!("{other:?} is not a float-valued geo function"),
    })
}

/// The smallest geodesic distance between any pair of positions of two geometries.
///
/// Vertex-to-vertex, not the true geodesic distance between the shapes: on the sphere
/// the nearest point of a segment is not the nearest point of its chord, and computing
/// it exactly needs an iterative geodesic solver per segment pair. For point-to-point
/// work — which is the overwhelming majority of "how far apart are these" queries —
/// the two coincide exactly. For polygon-to-polygon it over-reports by at most the
/// segment length, so it is an upper bound and safe to filter with. Densify with
/// `st_segmentize` first when the segments are long and the answer must be tight.
fn nearest_geodesic(a: &Geom, b: &Geom, spheroid: bool) -> Result<Option<f64>, ExprError> {
    let (ca, cb) = (a.coords(), b.coords());
    if ca.is_empty() || cb.is_empty() {
        return Ok(None);
    }
    // Intersecting shapes are zero apart, and the vertex scan cannot see that.
    if predicate::intersects(a, b) {
        return Ok(Some(0.0));
    }
    let mut best = f64::INFINITY;
    for p in &ca {
        for q in &cb {
            let d = if spheroid {
                match geodesy::vincenty(p.x, p.y, q.x, q.y) {
                    Ok(v) => v,
                    // Non-convergence is near-antipodal, which is never the minimum of a
                    // set that also holds convergent pairs; skip rather than fail.
                    Err(_) => continue,
                }
            } else {
                match geodesy::haversine(p.x, p.y, q.x, q.y) {
                    Ok(v) => v,
                    Err(_) => return Ok(None),
                }
            };
            best = best.min(d);
        }
    }
    Ok(best.is_finite().then_some(best))
}

/// The Int64-valued functions.
fn int_of(func: GeoFunc, g: Option<&Geom>) -> Option<i64> {
    use GeoFunc::*;
    let g = g?;
    Some(match func {
        StDimension => g.geom_type().dimension(),
        StSrid => g.srid as i64,
        StNumPoints => g.num_points() as i64,
        StNumGeometries => g.geometry.num_geometries() as i64,
        StNumInteriorRings => g
            .geometry
            .polygons()
            .first()
            .map(|p| p.interiors.len() as i64)
            .unwrap_or(0),
        StCoordDim => {
            if g.has_z {
                3
            } else {
                2
            }
        }
        other => unreachable!("{other:?} is not an integer-valued geo function"),
    })
}

/// The Boolean-valued functions.
fn bool_of(
    func: GeoFunc,
    g: Option<&Geom>,
    cols: &[ArrayRef],
    i: usize,
) -> Result<Option<bool>, ExprError> {
    use GeoFunc::*;
    let Some(a) = g else { return Ok(None) };
    // The single-geometry predicates first; the rest need a second operand.
    match func {
        StIsEmpty => return Ok(Some(a.is_empty())),
        StIsValid => return Ok(Some(validity::is_valid(a))),
        StIsClosed => return Ok(Some(validity::line_is_closed(&a.geometry))),
        StIsRing => return Ok(Some(validity::is_ring(&a.geometry))),
        StIsSimple => return Ok(Some(validity::is_simple(&a.geometry))),
        StIsCollection => {
            return Ok(Some(matches!(
                a.geometry,
                bc_geo::Geometry::GeometryCollection(_)
                    | bc_geo::Geometry::MultiPoint(_)
                    | bc_geo::Geometry::MultiLineString(_)
                    | bc_geo::Geometry::MultiPolygon(_)
            )))
        }
        StHasZ => return Ok(Some(a.has_z)),
        _ => {}
    }
    let Some(b) = geom_at(&cols[1], i, func)? else {
        return Ok(None);
    };
    Ok(Some(match func {
        StIntersects => predicate::intersects(a, &b),
        StDisjoint => predicate::disjoint(a, &b),
        StContains => predicate::contains(a, &b),
        StWithin => predicate::within(a, &b),
        StCovers => predicate::covers(a, &b),
        StCoveredBy => predicate::covered_by(a, &b),
        StTouches => predicate::touches(a, &b),
        StCrosses => predicate::crosses(a, &b),
        StOverlaps => predicate::overlaps(a, &b),
        StEquals => predicate::geom_equals(a, &b),
        StIntersectsExtent => match (a.bbox(), b.bbox()) {
            (Some(x), Some(y)) => x.intersects(&y),
            _ => false,
        },
        StContainsExtent => match (a.bbox(), b.bbox()) {
            (Some(x), Some(y)) => x.contains(&y),
            _ => false,
        },
        StDwithin => {
            let Some(r) = f64_at(&cols[2], i, func)? else {
                return Ok(None);
            };
            match predicate::dwithin(a, &b, r) {
                Some(v) => v,
                None => {
                    return Err(caller_error(
                        func,
                        bc_geo::GeoError::invalid(format!(
                            "radius must be a non-negative number, got {r}"
                        )),
                    ))
                }
            }
        }
        StDwithinSphere => {
            let Some(r) = f64_at(&cols[2], i, func)? else {
                return Ok(None);
            };
            if r < 0.0 || r.is_nan() {
                return Err(caller_error(
                    func,
                    bc_geo::GeoError::invalid(format!(
                        "radius in metres must be a non-negative number, got {r}"
                    )),
                ));
            }
            match nearest_geodesic(a, &b, false)? {
                Some(d) => d <= r,
                None => return Ok(None),
            }
        }
        other => unreachable!("{other:?} is not a boolean-valued geo function"),
    }))
}

/// The Utf8-valued functions.
fn text_of(func: GeoFunc, g: Option<&Geom>) -> Option<String> {
    use GeoFunc::*;
    let g = g?;
    match func {
        StAsText => Some(bc_geo::codec::wkt::write_wkt(g)),
        StAsEwkt => Some(bc_geo::codec::wkt::write_ewkt(g)),
        StAsHexWkb => Some(bc_geo::codec::wkb::write_hex_wkb(g)),
        StAsGeojson => Some(bc_geo::codec::geojson::write_geojson(g)),
        StGeometryType => Some(g.geom_type().name().to_string()),
        // Null means valid, which is what makes `WHERE reason IS NOT NULL` the query
        // that finds every broken row.
        StIsValidReason => validity::validity_reason(g),
        other => unreachable!("{other:?} is not a text-valued geo function"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_scalar_function_has_exactly_one_output_type() {
        for f in super::super::tests::ALL {
            if super::super::build::handles(f) || super::super::grid::handles(f) {
                continue;
            }
            // `output_type` panics on anything it does not classify, so reaching here
            // for every unrouted function is the assertion.
            let t = output_type(f);
            assert!(
                matches!(t, Out::Float | Out::Int | Out::Bool | Out::Text | Out::Bytes),
                "{f:?}"
            );
        }
    }
}
