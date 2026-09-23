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
    use GeoFunc::{
        GeohashDecodeLat, GeohashDecodeLon, GeohashEncode, StGeohash, StHexBin, StHexCenterX,
        StHexCenterY, StQuadkey, StS2Cell, StS2CellParent, StTileX, StTileY, StUtmEpsg, StUtmZone,
    };
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
    use GeoFunc::{
        GeohashDecodeLat, GeohashDecodeLon, GeohashEncode, StGeohash, StHexCenterX, StHexCenterY,
        StQuadkey,
    };
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
                out.push_f64(float_row(func, cols, i)?);
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
    use GeoFunc::{GeohashEncode, StGeohash, StQuadkey};
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
            row_result(geohash::encode(c.x, c.y, geohash_precision(func, p)?), func)?
        }
        GeohashEncode => {
            let Some((lon, lat)) = lonlat(func, cols, i)? else {
                return Ok(None);
            };
            let Some(p) = i64_at(&cols[2], i, func)? else {
                return Ok(None);
            };
            row_result(geohash::encode(lon, lat, geohash_precision(func, p)?), func)?
        }
        StQuadkey => {
            let Some((lon, lat)) = lonlat(func, cols, i)? else {
                return Ok(None);
            };
            let Some(z) = i64_at(&cols[2], i, func)? else {
                return Ok(None);
            };
            let Some(t) = row_result(tile::tile_of(lon, lat, zoom(func, z)?), func)? else {
                return Ok(None);
            };
            row_result(tile::quadkey(t), func)?
        }
        other => unreachable!("{other:?} is not a text-valued grid function"),
    })
}

fn float_row(func: GeoFunc, cols: &[ArrayRef], i: usize) -> Result<Option<f64>, ExprError> {
    use GeoFunc::{GeohashDecodeLat, GeohashDecodeLon, StHexCenterX, StHexCenterY};
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
    use GeoFunc::{StHexBin, StS2Cell, StS2CellParent, StTileX, StTileY, StUtmEpsg, StUtmZone};
    Ok(match func {
        StTileX | StTileY => {
            let Some((lon, lat)) = lonlat(func, cols, i)? else {
                return Ok(None);
            };
            let Some(z) = i64_at(&cols[2], i, func)? else {
                return Ok(None);
            };
            let Some(t) = row_result(tile::tile_of(lon, lat, zoom(func, z)?), func)? else {
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
            let Some(id) = row_result(s2::cell_id(lon, lat, s2_level(func, level)?), func)? else {
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
            Some(match s2::parent(cell as u64, s2_level(func, level)?) {
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
            row_result(crs::utm_zone(lon), func)?.map(i64::from)
        }
        StUtmEpsg => {
            let Some((lon, lat)) = lonlat(func, cols, i)? else {
                return Ok(None);
            };
            row_result(crs::utm_epsg(lon, lat), func)?.map(i64::from)
        }
        other => unreachable!("{other:?} is not an integer-valued grid function"),
    })
}

/// A grid parameter checked against its range, naming the value the caller passed.
///
/// These are *parameters*, not row data: a precision of -1 fails on every row, so it is a
/// query error rather than a null. The check lives here, before the `i64` narrows, because
/// narrowing first is exactly how `-1` used to arrive downstream as `0` — geohash then
/// reported "got 0" for a value nobody typed, and a quadkey at zoom `-1` silently became
/// the zoom-0 empty string.
fn ranged(func: GeoFunc, what: &str, v: i64, lo: i64, hi: i64) -> Result<i64, ExprError> {
    if (lo..=hi).contains(&v) {
        Ok(v)
    } else {
        Err(ExprError::InvalidArgument {
            func: super::fn_name(func),
            reason: format!("{what} must be {lo}..={hi}, got {v}"),
        })
    }
}

fn geohash_precision(func: GeoFunc, p: i64) -> Result<usize, ExprError> {
    let max = geohash::MAX_PRECISION as i64;
    Ok(ranged(func, "geohash precision", p, 1, max)? as usize)
}

fn zoom(func: GeoFunc, z: i64) -> Result<u32, ExprError> {
    Ok(ranged(func, "tile zoom", z, 0, i64::from(tile::MAX_ZOOM))? as u32)
}

fn s2_level(func: GeoFunc, level: i64) -> Result<u32, ExprError> {
    Ok(ranged(func, "S2 level", level, 0, i64::from(s2::MAX_LEVEL))? as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_out_of_range_parameter_is_an_error_naming_the_value_passed() {
        let msg = |r: Result<u32, ExprError>| r.unwrap_err().to_string();
        assert!(msg(zoom(GeoFunc::StQuadkey, -1)).contains("got -1"));
        assert!(msg(zoom(GeoFunc::StQuadkey, 31)).contains("got 31"));
        assert!(msg(s2_level(GeoFunc::StS2Cell, -1)).contains("got -1"));
        let gh = geohash_precision(GeoFunc::GeohashEncode, -1)
            .unwrap_err()
            .to_string();
        assert!(gh.contains("got -1") && !gh.contains("got 0"), "{gh}");
        assert!(geohash_precision(GeoFunc::GeohashEncode, 0).is_err());
        assert_eq!(geohash_precision(GeoFunc::GeohashEncode, 12).unwrap(), 12);
        assert_eq!(zoom(GeoFunc::StTileX, 0).unwrap(), 0);
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
