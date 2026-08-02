//! The grid and reference-system functions, which take plain numbers rather than
//! geometry.
//!
//! Separated from the rest because their inputs are ordinary `lon`/`lat`/`zoom` columns.
//! That is how a table actually stores positions before anyone builds a geometry from
//! them, and it is where these functions earn their keep: `st_s2_cell(lon, lat, 12)`
//! turns two float columns into one integer group key without materializing a geometry
//! at all, which is the difference between a spatial rollup that shuffles 8 bytes a row
//! and one that shuffles a WKB blob.
//!
//! `st_geohash` is the exception: it takes a geometry, because PostGIS spells it that
//! way and because a polygon's hash is a meaningful thing to ask for. It reduces to the
//! centroid, which is what a cell id of an extended shape can mean.

use arrow::array::{ArrayRef, Float64Builder, Int64Builder, StringBuilder};

use bc_geo::grid::{geohash, hexbin, s2, tile};
use bc_geo::proj::crs;

use crate::{ExprError, GeoFunc};

use super::{f64_at, geom_at, i64_at, row_result, str_at, ScalarOut};

/// True when this dispatcher owns `func`.
pub(super) fn handles(func: GeoFunc) -> bool {
    use GeoFunc::*;
    matches!(
        func,
        StGeohash
            | GeohashEncode
            | GeohashDecodeLon
            | GeohashDecodeLat
            | StTileX
            | StTileY
            | StQuadkey
            | StS2Cell
            | StS2CellParent
            | StHexBin
            | StHexCenterX
            | StHexCenterY
            | StUtmZone
            | StUtmEpsg
    )
}

/// Evaluate a grid function over `rows` rows of `cols`.
pub(super) fn eval(func: GeoFunc, cols: &[ArrayRef], rows: usize) -> Result<ArrayRef, ExprError> {
    use GeoFunc::*;
    let mut out = match func {
        StGeohash | GeohashEncode | StQuadkey => {
            ScalarOut::Text(StringBuilder::with_capacity(rows, rows * 12))
        }
        GeohashDecodeLon | GeohashDecodeLat | StHexCenterX | StHexCenterY => {
            ScalarOut::Float(Float64Builder::with_capacity(rows))
        }
        _ => ScalarOut::Int(Int64Builder::with_capacity(rows)),
    };
    for i in 0..rows {
        match func {
            StGeohash | GeohashEncode | StQuadkey => {
                let s = text_row(func, cols, i)?;
                out.push_str(s.as_deref());
            }
            GeohashDecodeLon | GeohashDecodeLat | StHexCenterX | StHexCenterY => {
                out.push_f64(float_row(func, cols, i)?)
            }
            _ => out.push_i64(int_row(func, cols, i)?),
        }
    }
    Ok(out.finish())
}

/// Read a `(lon, lat)` pair from the first two columns.
fn lonlat(func: GeoFunc, cols: &[ArrayRef], i: usize) -> Result<Option<(f64, f64)>, ExprError> {
    let (Some(lon), Some(lat)) = (f64_at(&cols[0], i, func)?, f64_at(&cols[1], i, func)?) else {
        return Ok(None);
    };
    Ok(Some((lon, lat)))
}

fn text_row(func: GeoFunc, cols: &[ArrayRef], i: usize) -> Result<Option<String>, ExprError> {
    use GeoFunc::*;
    Ok(match func {
        StGeohash => {
            let Some(g) = geom_at(&cols[0], i, func)? else {
                return Ok(None);
            };
            let Some(p) = i64_at(&cols[1], i, func)? else {
                return Ok(None);
            };
            let Some(c) = bc_geo::algo::measure::centroid(&g.geometry) else {
                return Ok(None);
            };
            row_result(geohash::encode(c.x, c.y, p.max(0) as usize), func)?
        }
        GeohashEncode => {
            let Some((lon, lat)) = lonlat(func, cols, i)? else {
                return Ok(None);
            };
            let Some(p) = i64_at(&cols[2], i, func)? else {
                return Ok(None);
            };
            row_result(geohash::encode(lon, lat, p.max(0) as usize), func)?
        }
        StQuadkey => {
            let Some((lon, lat)) = lonlat(func, cols, i)? else {
                return Ok(None);
            };
            let Some(z) = i64_at(&cols[2], i, func)? else {
                return Ok(None);
            };
            let Some(t) = row_result(tile::tile_of(lon, lat, clamp_zoom(z)), func)? else {
                return Ok(None);
            };
            row_result(tile::quadkey(t), func)?
        }
        other => unreachable!("{other:?} is not a text-valued grid function"),
    })
}

fn float_row(func: GeoFunc, cols: &[ArrayRef], i: usize) -> Result<Option<f64>, ExprError> {
    use GeoFunc::*;
    Ok(match func {
        GeohashDecodeLon | GeohashDecodeLat => {
            let Some(h) = str_at(&cols[0], i, func)? else {
                return Ok(None);
            };
            let Some(c) = row_result(geohash::decode(h), func)? else {
                return Ok(None);
            };
            Some(if func == GeohashDecodeLon { c.x } else { c.y })
        }
        StHexCenterX | StHexCenterY => {
            let (Some(key), Some(size)) = (i64_at(&cols[0], i, func)?, f64_at(&cols[1], i, func)?)
            else {
                return Ok(None);
            };
            let Some(c) = row_result(hexbin::hex_center(hexbin::hex_from_key(key), size), func)?
            else {
                return Ok(None);
            };
            Some(if func == StHexCenterX { c.x } else { c.y })
        }
        other => unreachable!("{other:?} is not a float-valued grid function"),
    })
}

fn int_row(func: GeoFunc, cols: &[ArrayRef], i: usize) -> Result<Option<i64>, ExprError> {
    use GeoFunc::*;
    Ok(match func {
        StTileX | StTileY => {
            let Some((lon, lat)) = lonlat(func, cols, i)? else {
                return Ok(None);
            };
            let Some(z) = i64_at(&cols[2], i, func)? else {
                return Ok(None);
            };
            let Some(t) = row_result(tile::tile_of(lon, lat, clamp_zoom(z)), func)? else {
                return Ok(None);
            };
            Some(if func == StTileX { t.x } else { t.y })
        }
        StS2Cell => {
            let Some((lon, lat)) = lonlat(func, cols, i)? else {
                return Ok(None);
            };
            let Some(level) = i64_at(&cols[2], i, func)? else {
                return Ok(None);
            };
            let Some(id) = row_result(s2::cell_id(lon, lat, clamp_level(level)), func)? else {
                return Ok(None);
            };
            // S2 ids fill 64 bits, and Arrow's integer column is signed. The
            // reinterpretation is lossless and order-preserving *within* a face, which
            // is what the Hilbert locality actually depends on; a cast that saturated
            // would collapse every face-4-and-up cell onto one value.
            Some(id as i64)
        }
        StS2CellParent => {
            let (Some(cell), Some(level)) =
                (i64_at(&cols[0], i, func)?, i64_at(&cols[1], i, func)?)
            else {
                return Ok(None);
            };
            Some(match s2::parent(cell as u64, clamp_level(level)) {
                Some(p) => p as i64,
                None => return Ok(None),
            })
        }
        StHexBin => {
            let (Some(x), Some(y), Some(size)) = (
                f64_at(&cols[0], i, func)?,
                f64_at(&cols[1], i, func)?,
                f64_at(&cols[2], i, func)?,
            ) else {
                return Ok(None);
            };
            let Some(h) = row_result(hexbin::hex_of(x, y, size), func)? else {
                return Ok(None);
            };
            row_result(hexbin::hex_key(h), func)?
        }
        StUtmZone => {
            let Some(lon) = f64_at(&cols[0], i, func)? else {
                return Ok(None);
            };
            row_result(crs::utm_zone(lon), func)?.map(|z| z as i64)
        }
        StUtmEpsg => {
            let Some((lon, lat)) = lonlat(func, cols, i)? else {
                return Ok(None);
            };
            row_result(crs::utm_epsg(lon, lat), func)?.map(|e| e as i64)
        }
        other => unreachable!("{other:?} is not an integer-valued grid function"),
    })
}

/// A zoom level as `u32`, saturating at the module's own maximum.
///
/// The clamp is not a silent truncation: `tile_of` rejects anything past its maximum
/// with a message, so an out-of-range value still reaches the user as an error. This
/// exists only so a negative `i64` does not wrap to four billion on the cast.
fn clamp_zoom(z: i64) -> u32 {
    z.clamp(0, i64::from(tile::MAX_ZOOM) + 1) as u32
}

/// An S2 level as `u32`, with the same reasoning as `clamp_zoom`.
fn clamp_level(level: i64) -> u32 {
    level.clamp(0, i64::from(s2::MAX_LEVEL) + 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_of_range_levels_saturate_into_the_error_range_rather_than_wrapping() {
        assert_eq!(clamp_zoom(-5), 0);
        assert_eq!(clamp_zoom(1_000), tile::MAX_ZOOM + 1);
        assert!(tile::tile_of(0.0, 0.0, clamp_zoom(1_000)).is_err());
        assert_eq!(clamp_level(-1), 0);
        assert!(s2::cell_id(0.0, 0.0, clamp_level(99)).is_err());
    }

    #[test]
    fn every_grid_function_is_claimed_by_this_dispatcher_and_returns_a_scalar() {
        for f in super::super::tests::ALL {
            if handles(f) {
                assert!(
                    !f.returns_geometry(),
                    "{f:?} is a grid function returning a geometry"
                );
            }
        }
    }
}
