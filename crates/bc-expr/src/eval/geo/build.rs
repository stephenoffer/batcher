//! The geometry-returning functions: constructors, transforms, derived shapes.
//!
//! One dispatcher for everything whose output column is WKB, which is exactly the set
//! `GeoFunc::returns_geometry` names — the two are the same predicate, so a new
//! geometry-returning function cannot be added to one and forgotten in the other.
//!
//! Every function here preserves the input's SRID. That is not incidental: a centroid
//! of a geometry in EPSG:3857 is in EPSG:3857, and dropping the label would make a later
//! `st_transform` silently reproject from the wrong system. `st_transform` is the one
//! function that changes it, because changing it is what it is for.

use arrow::array::ArrayRef;

use bc_geo::algo::{affine, construct, linear};
use bc_geo::{Geom, Geometry};

use crate::{ExprError, GeoFunc};

use super::{caller_error, f64_at, geom_at, i64_at, row_result, str_at, GeomOut};

/// True when this dispatcher owns `func`.
pub(super) fn handles(func: GeoFunc) -> bool {
    func.returns_geometry()
}

/// Evaluate a geometry-returning function over `rows` rows of `cols`.
pub(super) fn eval(func: GeoFunc, cols: &[ArrayRef], rows: usize) -> Result<ArrayRef, ExprError> {
    let mut out = GeomOut::with_capacity(rows);
    for i in 0..rows {
        out.push(one(func, cols, i)?.as_ref());
    }
    Ok(out.finish())
}

/// A helper for the many single-geometry-plus-parameters shapes: decode the geometry,
/// short-circuit to null when it is absent, and rebuild the result with the same SRID.
macro_rules! geom_arg {
    ($func:expr, $cols:expr, $i:expr) => {
        match geom_at(&$cols[0], $i, $func)? {
            Some(g) => g,
            None => return Ok(None),
        }
    };
}

/// A helper for the numeric parameters, which null the row when absent.
macro_rules! num_arg {
    ($func:expr, $cols:expr, $idx:expr, $i:expr) => {
        match f64_at(&$cols[$idx], $i, $func)? {
            Some(v) => v,
            None => return Ok(None),
        }
    };
}

/// Evaluate one row.
fn one(func: GeoFunc, cols: &[ArrayRef], i: usize) -> Result<Option<Geom>, ExprError> {
    use GeoFunc::*;
    // The constructors that build from plain numbers or text, before any geometry decode.
    match func {
        StPoint => {
            let (Some(x), Some(y)) = (f64_at(&cols[0], i, func)?, f64_at(&cols[1], i, func)?)
            else {
                return Ok(None);
            };
            return Ok(Some(Geom::new(Geometry::Point(Some(bc_geo::Coord::new(
                x, y,
            ))))));
        }
        StPointZ => {
            let (Some(x), Some(y), Some(z)) = (
                f64_at(&cols[0], i, func)?,
                f64_at(&cols[1], i, func)?,
                f64_at(&cols[2], i, func)?,
            ) else {
                return Ok(None);
            };
            let mut g = Geom::new(Geometry::Point(Some(bc_geo::Coord::new_z(x, y, z))));
            g.has_z = true;
            return Ok(Some(g));
        }
        StMakeEnvelope => {
            let mut b = [0.0f64; 4];
            for (k, slot) in b.iter_mut().enumerate() {
                match f64_at(&cols[k], i, func)? {
                    Some(v) => *slot = v,
                    None => return Ok(None),
                }
            }
            let g = construct::make_envelope(b[0], b[1], b[2], b[3])
                .map_err(|e| caller_error(func, e))?;
            return Ok(Some(Geom::new(g)));
        }
        StGeomFromText | StGeomFromWkb | StGeomFromGeojson => {
            // All three re-validate an existing column as geometry. `geom_at` already
            // accepts every encoding, so they differ only in the error they would give
            // — and they give none, because a bad row nulls.
            return geom_at(&cols[0], i, func);
        }
        StGeomFromGeohash => {
            let Some(h) = str_at(&cols[0], i, func)? else {
                return Ok(None);
            };
            let Some(b) = row_result(bc_geo::grid::geohash::decode_bbox(h), func)? else {
                return Ok(None);
            };
            let g = construct::make_envelope(b.xmin, b.ymin, b.xmax, b.ymax)
                .map_err(|e| caller_error(func, e))?;
            return Ok(Some(Geom::new(g).with_srid(4326)));
        }
        _ => {}
    }

    let g = geom_arg!(func, cols, i);
    let srid = g.srid;
    let has_z = g.has_z;
    let rebuilt = |geometry: Geometry| {
        Some(Geom {
            srid,
            has_z,
            geometry,
        })
    };

    Ok(match func {
        StMakeLine => {
            let Some(b) = geom_at(&cols[1], i, func)? else {
                return Ok(None);
            };
            let (Some(p), Some(q)) = (g.coords().first().copied(), b.coords().first().copied())
            else {
                return Ok(None);
            };
            rebuilt(Geometry::LineString(vec![p, q]))
        }
        StMakePolygon => {
            let mut ring = g.coords();
            if ring.len() < 3 {
                return Ok(None);
            }
            bc_geo::types::close_ring(&mut ring);
            rebuilt(Geometry::Polygon(bc_geo::Polygon {
                exterior: ring,
                interiors: Vec::new(),
            }))
        }
        StSetSrid => {
            let Some(s) = i64_at(&cols[1], i, func)? else {
                return Ok(None);
            };
            Some(g.with_srid(s as i32))
        }
        StGeometryN => {
            let Some(n) = i64_at(&cols[1], i, func)? else {
                return Ok(None);
            };
            g.geometry.geometry_n(n.max(0) as usize).and_then(rebuilt)
        }
        StPointN => {
            let Some(n) = i64_at(&cols[1], i, func)? else {
                return Ok(None);
            };
            nth_position(&g.geometry, n).and_then(|c| rebuilt(Geometry::Point(Some(c))))
        }
        StStartPoint => chain(&g.geometry)
            .and_then(|l| l.first().copied())
            .and_then(|c| rebuilt(Geometry::Point(Some(c)))),
        StEndPoint => chain(&g.geometry)
            .and_then(|l| l.last().copied())
            .and_then(|c| rebuilt(Geometry::Point(Some(c)))),
        StExteriorRing => g
            .geometry
            .polygons()
            .first()
            .map(|p| p.exterior.clone())
            .filter(|r| !r.is_empty())
            .and_then(|r| rebuilt(Geometry::LineString(r))),
        StInteriorRingN => {
            let Some(n) = i64_at(&cols[1], i, func)? else {
                return Ok(None);
            };
            g.geometry
                .polygons()
                .first()
                .and_then(|p| p.interiors.get((n.max(1) - 1) as usize))
                .cloned()
                .and_then(|r| rebuilt(Geometry::LineString(r)))
        }
        StCentroid => bc_geo::algo::measure::centroid(&g.geometry)
            .and_then(|c| rebuilt(Geometry::Point(Some(c)))),
        StEnvelope => rebuilt(construct::envelope(&g)),
        StBoundary => rebuilt(construct::boundary(&g.geometry)),
        StConvexHull => rebuilt(construct::convex_hull(&g)),
        StPointOnSurface => linear::point_on_surface(&g.geometry).and_then(rebuilt),
        StBuffer => {
            let r = num_arg!(func, cols, 1, i);
            let Some(segs) = i64_at(&cols[2], i, func)? else {
                return Ok(None);
            };
            if segs <= 0 {
                return Err(caller_error(
                    func,
                    bc_geo::GeoError::invalid(format!("quad_segs must be positive, got {segs}")),
                ));
            }
            rebuilt(construct::buffer(&g, r, segs as usize).map_err(|e| caller_error(func, e))?)
        }
        StSimplify => {
            let eps = num_arg!(func, cols, 1, i);
            rebuilt(construct::simplify(&g.geometry, eps).map_err(|e| caller_error(func, e))?)
        }
        StReverse => rebuilt(construct::reverse(&g.geometry)),
        StForce2d => Some(Geom {
            srid,
            has_z: false,
            geometry: affine::force_2d(&g.geometry),
        }),
        StForce3d => {
            let z = num_arg!(func, cols, 1, i);
            Some(Geom {
                srid,
                has_z: true,
                geometry: affine::force_3d(&g.geometry, z),
            })
        }
        StForcePolygonCcw => rebuilt(construct::force_winding(&g.geometry, true)),
        StForcePolygonCw => rebuilt(construct::force_winding(&g.geometry, false)),
        StFlipCoordinates => rebuilt(construct::flip_coordinates(&g.geometry)),
        StTranslate => {
            let (dx, dy) = (num_arg!(func, cols, 1, i), num_arg!(func, cols, 2, i));
            rebuilt(affine::translate(&g.geometry, dx, dy, 0.0))
        }
        StScale => {
            let (sx, sy) = (num_arg!(func, cols, 1, i), num_arg!(func, cols, 2, i));
            rebuilt(affine::scale(&g.geometry, sx, sy, 1.0))
        }
        StRotate => {
            let r = num_arg!(func, cols, 1, i);
            rebuilt(affine::rotate(&g.geometry, r, 0.0, 0.0))
        }
        StAffine => {
            let mut p = [0.0f64; 6];
            for (k, slot) in p.iter_mut().enumerate() {
                match f64_at(&cols[k + 1], i, func)? {
                    Some(v) => *slot = v,
                    None => return Ok(None),
                }
            }
            rebuilt(affine::affine(
                &g.geometry,
                p[0],
                p[1],
                p[2],
                p[3],
                p[4],
                p[5],
            ))
        }
        StSnapToGrid => {
            let size = num_arg!(func, cols, 1, i);
            rebuilt(
                affine::snap_to_grid(&g.geometry, size, size, 0.0, 0.0)
                    .map_err(|e| caller_error(func, e))?,
            )
        }
        StSegmentize => {
            let max_len = num_arg!(func, cols, 1, i);
            rebuilt(linear::segmentize(&g.geometry, max_len).map_err(|e| caller_error(func, e))?)
        }
        StExpand => {
            let (dx, dy) = (num_arg!(func, cols, 1, i), num_arg!(func, cols, 2, i));
            rebuilt(affine::expand(&g, dx, dy).map_err(|e| caller_error(func, e))?)
        }
        StCollect => {
            let Some(b) = geom_at(&cols[1], i, func)? else {
                return Ok(None);
            };
            rebuilt(construct::collect(&g.geometry, &b.geometry))
        }
        StRemoveRepeatedPoints => {
            let tol = num_arg!(func, cols, 1, i);
            rebuilt(construct::remove_repeated_points(&g.geometry, tol))
        }
        StLineInterpolatePoint => {
            let f = num_arg!(func, cols, 1, i);
            match linear::interpolate_point(&g.geometry, f) {
                Ok(p) => rebuilt(p),
                Err(e) if e.is_row_local() => None,
                Err(e) => return Err(caller_error(func, e)),
            }
        }
        StLineSubstring => {
            let (a, b) = (num_arg!(func, cols, 1, i), num_arg!(func, cols, 2, i));
            match linear::substring(&g.geometry, a, b) {
                Ok(p) => rebuilt(p),
                Err(e) if e.is_row_local() => None,
                Err(e) => return Err(caller_error(func, e)),
            }
        }
        StClosestPoint => {
            let Some(b) = geom_at(&cols[1], i, func)? else {
                return Ok(None);
            };
            linear::closest_point(&g.geometry, &b.geometry).and_then(rebuilt)
        }
        StShortestLine => {
            let Some(b) = geom_at(&cols[1], i, func)? else {
                return Ok(None);
            };
            linear::shortest_line(&g.geometry, &b.geometry).and_then(rebuilt)
        }
        StProject => {
            let (dist, azimuth) = (num_arg!(func, cols, 1, i), num_arg!(func, cols, 2, i));
            let Some(p) = g.geometry.points().first().copied() else {
                return Ok(None);
            };
            let Some(c) = row_result(
                bc_geo::proj::geodesy::destination(p.x, p.y, azimuth, dist),
                func,
            )?
            else {
                return Ok(None);
            };
            rebuilt(Geometry::Point(Some(c)))
        }
        StTransform => {
            let (Some(from), Some(to)) = (i64_at(&cols[1], i, func)?, i64_at(&cols[2], i, func)?)
            else {
                return Ok(None);
            };
            let geometry = bc_geo::proj::crs::transform(&g.geometry, from as i32, to as i32)
                .map_err(|e| caller_error(func, e))?;
            Some(Geom {
                srid: to as i32,
                has_z,
                geometry,
            })
        }
        other => {
            return Err(ExprError::InvalidArgument {
                func: super::fn_name(other),
                reason: "is not a geometry-returning function".to_string(),
            })
        }
    })
}

/// The single chain of a geometry, for `st_start_point` / `st_end_point`.
///
/// `None` for anything but a lone `LineString`: a multi-chain has no defined first
/// position, and answering with whichever member the encoder happened to write first is
/// how a route's start silently becomes a different segment after a re-export.
fn chain(g: &Geometry) -> Option<&Vec<bc_geo::Coord>> {
    match g {
        Geometry::LineString(l) if !l.is_empty() => Some(l),
        _ => None,
    }
}

/// The 1-based `n`-th position of a chain, counting from the end when `n` is negative
/// (PostGIS `ST_PointN` accepts both).
fn nth_position(g: &Geometry, n: i64) -> Option<bc_geo::Coord> {
    let l = chain(g)?;
    let idx = if n > 0 {
        (n - 1) as usize
    } else if n < 0 {
        let from_end = (-n) as usize;
        l.len().checked_sub(from_end)?
    } else {
        return None;
    };
    l.get(idx).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nth_position_counts_from_either_end() {
        let g = Geometry::LineString(vec![
            bc_geo::Coord::new(0.0, 0.0),
            bc_geo::Coord::new(1.0, 1.0),
            bc_geo::Coord::new(2.0, 2.0),
        ]);
        assert_eq!(nth_position(&g, 1).unwrap().x, 0.0);
        assert_eq!(nth_position(&g, 3).unwrap().x, 2.0);
        assert_eq!(nth_position(&g, -1).unwrap().x, 2.0);
        assert_eq!(nth_position(&g, -3).unwrap().x, 0.0);
        assert_eq!(nth_position(&g, 0), None);
        assert_eq!(nth_position(&g, 4), None);
        assert_eq!(nth_position(&g, -4), None);
    }

    #[test]
    fn a_multi_chain_has_no_start_point() {
        let multi = Geometry::MultiLineString(vec![vec![
            bc_geo::Coord::new(0.0, 0.0),
            bc_geo::Coord::new(1.0, 1.0),
        ]]);
        assert!(chain(&multi).is_none());
        assert!(chain(&Geometry::LineString(Vec::new())).is_none());
    }
}
