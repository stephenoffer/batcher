//! Slippy-map tiles and Bing quadkeys — the grid every map tile server is indexed by.
//!
//! A tile is `(z, x, y)`: at zoom `z` the Web Mercator square is cut into `2^z` columns
//! and rows, `x` increasing east and `y` increasing *south*. That southward `y` is the
//! single most common source of off-by-a-hemisphere bugs in tile code, so it is stated
//! here rather than left to be rediscovered: `y = 0` is the top of the map, near 85°N.
//!
//! The quadkey is the same tile written as a base-4 string, one digit per zoom level,
//! and it has the geohash's prefix property: a tile's quadkey extends its parent's. So
//! the same trick applies — a spatial rollup across zoom levels is a `substr` on a
//! string column, and a tile-range scan is a prefix predicate.
//!
//! Mercator's latitude limit (±85.0511°) is a property of the projection, not a choice:
//! the pole maps to infinity, so the square has to be cut somewhere and every tile
//! scheme cuts it at the latitude that makes the map square.

use std::f64::consts::PI;

use crate::error::{GeoError, GeoResult};
use crate::types::{Bbox, Coord};

/// The latitude where the Web Mercator square is truncated, in degrees.
pub const MERCATOR_MAX_LAT: f64 = 85.051_128_779_806_59;

/// The largest zoom this module accepts. At zoom 30 a tile is a few centimetres across
/// and `2^z` still fits in the `i64` the engine carries integers in.
pub const MAX_ZOOM: u32 = 30;

fn check_zoom(z: u32) -> GeoResult<()> {
    if z > MAX_ZOOM {
        return Err(GeoError::invalid(format!(
            "tile zoom must be 0..={MAX_ZOOM}, got {z}"
        )));
    }
    Ok(())
}

/// A slippy-map tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    /// Zoom level.
    pub z: u32,
    /// Column, increasing east.
    pub x: i64,
    /// Row, increasing **south**.
    pub y: i64,
}

/// The tile containing a lon/lat position at the given zoom.
///
/// Latitudes beyond the Mercator limit are clamped rather than refused: a GPS fix at
/// 87°N is a real observation, and the tile that covers the top of the map is the
/// honest answer for it.
pub fn tile_of(lon: f64, lat: f64, z: u32) -> GeoResult<Tile> {
    check_zoom(z)?;
    if !(-180.0..=180.0).contains(&lon) || !(-90.0..=90.0).contains(&lat) {
        return Err(GeoError::invalid(format!(
            "tile lookup needs lon in [-180, 180] and lat in [-90, 90], got ({lon}, {lat})"
        )));
    }
    let n = 2f64.powi(z as i32);
    let lat = lat.clamp(-MERCATOR_MAX_LAT, MERCATOR_MAX_LAT);
    let x = ((lon + 180.0) / 360.0 * n).floor() as i64;
    let lat_rad = lat.to_radians();
    let y = ((1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / PI) / 2.0 * n).floor() as i64;
    let max = (n as i64) - 1;
    Ok(Tile {
        z,
        x: x.clamp(0, max),
        y: y.clamp(0, max),
    })
}

/// The lon/lat bounds of a tile.
pub fn tile_bbox(t: Tile) -> GeoResult<Bbox> {
    check_zoom(t.z)?;
    let n = 2f64.powi(t.z as i32);
    if t.x < 0 || t.y < 0 || t.x >= n as i64 || t.y >= n as i64 {
        return Err(GeoError::invalid(format!(
            "tile ({}, {}) is outside the {}x{} grid at zoom {}",
            t.x, t.y, n as i64, n as i64, t.z
        )));
    }
    let lon = |x: f64| x / n * 360.0 - 180.0;
    let lat = |y: f64| {
        let m = PI * (1.0 - 2.0 * y / n);
        m.sinh().atan().to_degrees()
    };
    Ok(Bbox {
        xmin: lon(t.x as f64),
        // y increases south, so the tile's *lower* y bound is its higher row index.
        ymin: lat((t.y + 1) as f64),
        xmax: lon((t.x + 1) as f64),
        ymax: lat(t.y as f64),
    })
}

/// The Bing quadkey of a tile: one base-4 digit per zoom level.
///
/// Zoom 0 has one tile and the empty quadkey, which is correct and is why the return
/// type is a possibly-empty string rather than an error.
pub fn quadkey(t: Tile) -> GeoResult<String> {
    tile_bbox(t)?;
    let mut out = String::with_capacity(t.z as usize);
    for i in (1..=t.z).rev() {
        let mask = 1i64 << (i - 1);
        let mut digit = 0u8;
        if t.x & mask != 0 {
            digit += 1;
        }
        if t.y & mask != 0 {
            digit += 2;
        }
        out.push((b'0' + digit) as char);
    }
    Ok(out)
}

/// The tile a quadkey names.
pub fn from_quadkey(key: &str) -> GeoResult<Tile> {
    if key.len() as u32 > MAX_ZOOM {
        return Err(GeoError::invalid(format!(
            "quadkey of length {} exceeds zoom {MAX_ZOOM}",
            key.len()
        )));
    }
    let z = key.len() as u32;
    let (mut x, mut y) = (0i64, 0i64);
    for (i, c) in key.bytes().enumerate() {
        let mask = 1i64 << (z as usize - i - 1);
        match c {
            b'0' => {}
            b'1' => x |= mask,
            b'2' => y |= mask,
            b'3' => {
                x |= mask;
                y |= mask;
            }
            other => {
                return Err(GeoError::parse(
                    "quadkey",
                    format!("{:?} is not a quadkey digit (0-3)", other as char),
                ))
            }
        }
    }
    Ok(Tile { z, x, y })
}

/// Project lon/lat to Web Mercator metres (EPSG:3857).
///
/// The projection every tile scheme is defined in, and the one to bin in when cells
/// must be equal-*area-ish* rather than equal-degree. It is conformal, not equal-area:
/// a cell at 60°N covers a quarter of the ground a cell at the equator does, which is
/// why a density comparison across latitudes needs an equal-area projection instead.
pub fn to_web_mercator(lon: f64, lat: f64) -> GeoResult<Coord> {
    if !(-180.0..=180.0).contains(&lon) || !(-90.0..=90.0).contains(&lat) {
        return Err(GeoError::invalid(format!(
            "web mercator needs lon in [-180, 180] and lat in [-90, 90], got ({lon}, {lat})"
        )));
    }
    let lat = lat.clamp(-MERCATOR_MAX_LAT, MERCATOR_MAX_LAT);
    let x = lon.to_radians() * crate::proj::geodesy::EARTH_RADIUS_M;
    let y = ((PI / 4.0 + lat.to_radians() / 2.0).tan()).ln() * crate::proj::geodesy::EARTH_RADIUS_M;
    Ok(Coord::new(x, y))
}

/// Invert `to_web_mercator`.
pub fn from_web_mercator(x: f64, y: f64) -> Coord {
    let lon = (x / crate::proj::geodesy::EARTH_RADIUS_M).to_degrees();
    let lat =
        (2.0 * (y / crate::proj::geodesy::EARTH_RADIUS_M).exp().atan() - PI / 2.0).to_degrees();
    Coord::new(lon, lat)
}

/// The tiles at zoom `z` that a bounding box touches, capped at `limit`.
///
/// The cap is not defensive clutter: a whole-world box at zoom 20 names a trillion
/// tiles, and a function that tried would hang rather than fail. Exceeding it is an
/// error naming the count, so the caller can pick a coarser zoom.
pub fn tiles_covering(b: &Bbox, z: u32, limit: usize) -> GeoResult<Vec<Tile>> {
    check_zoom(z)?;
    let lo = tile_of(b.xmin.max(-180.0), b.ymax.min(90.0), z)?;
    let hi = tile_of(b.xmax.min(180.0), b.ymin.max(-90.0), z)?;
    let (x0, x1) = (lo.x.min(hi.x), lo.x.max(hi.x));
    let (y0, y1) = (lo.y.min(hi.y), lo.y.max(hi.y));
    let count = ((x1 - x0 + 1) as usize).saturating_mul((y1 - y0 + 1) as usize);
    if count > limit {
        return Err(GeoError::invalid(format!(
            "box covers {count} tiles at zoom {z}, over the limit of {limit}; use a coarser zoom"
        )));
    }
    let mut out = Vec::with_capacity(count);
    for x in x0..=x1 {
        for y in y0..=y1 {
            out.push(Tile { z, x, y });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_tiles_match_the_slippy_map_convention() {
        // The tile containing San Francisco at zoom 12, from the OSM reference.
        assert_eq!(
            tile_of(-122.4194, 37.7749, 12).unwrap(),
            Tile {
                z: 12,
                x: 655,
                y: 1583
            }
        );
        // Zoom 0 is one tile covering the world.
        assert_eq!(tile_of(0.0, 0.0, 0).unwrap(), Tile { z: 0, x: 0, y: 0 });
    }

    #[test]
    fn y_increases_southward() {
        let north = tile_of(0.0, 60.0, 4).unwrap();
        let south = tile_of(0.0, -60.0, 4).unwrap();
        assert!(north.y < south.y, "y must grow toward the south pole");
    }

    #[test]
    fn a_tile_contains_the_position_that_produced_it() {
        for (lon, lat) in [
            (-122.4194, 37.7749),
            (0.0, 0.0),
            (151.2093, -33.8688),
            (-0.1278, 51.5074),
        ] {
            for z in 0..=18 {
                let t = tile_of(lon, lat, z).unwrap();
                let b = tile_bbox(t).unwrap();
                assert!(
                    b.contains_coord(crate::types::Coord::new(lon, lat)),
                    "zoom {z} tile {t:?} does not contain ({lon}, {lat})"
                );
            }
        }
    }

    #[test]
    fn quadkeys_round_trip_and_nest() {
        let t = tile_of(-122.4194, 37.7749, 12).unwrap();
        let k = quadkey(t).unwrap();
        assert_eq!(k.len(), 12);
        assert_eq!(from_quadkey(&k).unwrap(), t);
        // Every zoom's quadkey is a prefix of the next.
        for z in 1..12u32 {
            let parent = quadkey(tile_of(-122.4194, 37.7749, z).unwrap()).unwrap();
            assert!(k.starts_with(&parent), "{parent} must prefix {k}");
        }
        assert_eq!(quadkey(Tile { z: 0, x: 0, y: 0 }).unwrap(), "");
    }

    #[test]
    fn quadkey_digits_are_validated() {
        assert!(from_quadkey("0123").is_ok());
        assert!(from_quadkey("0124").is_err());
        assert!(from_quadkey("abc").is_err());
    }

    #[test]
    fn web_mercator_round_trips_within_the_projection_limit() {
        for (lon, lat) in [
            (0.0, 0.0),
            (-122.4194, 37.7749),
            (151.0, -33.0),
            (179.0, 84.0),
        ] {
            let m = to_web_mercator(lon, lat).unwrap();
            let back = from_web_mercator(m.x, m.y);
            assert!((back.x - lon).abs() < 1e-9, "{lon} -> {}", back.x);
            assert!((back.y - lat).abs() < 1e-9, "{lat} -> {}", back.y);
        }
        // The origin is the origin.
        let o = to_web_mercator(0.0, 0.0).unwrap();
        assert!(o.x.abs() < 1e-9 && o.y.abs() < 1e-9);
    }

    #[test]
    fn mercator_clamps_the_poles_rather_than_returning_infinity() {
        let p = to_web_mercator(0.0, 90.0).unwrap();
        assert!(p.y.is_finite());
        assert!(tile_of(0.0, 90.0, 5).unwrap().y == 0);
    }

    #[test]
    fn covering_a_box_is_bounded_and_says_so_when_it_is_not() {
        let b = Bbox {
            xmin: -122.5,
            ymin: 37.7,
            xmax: -122.4,
            ymax: 37.8,
        };
        let tiles = tiles_covering(&b, 12, 1000).unwrap();
        assert!(!tiles.is_empty());
        for t in &tiles {
            assert!(tile_bbox(*t).unwrap().intersects(&b));
        }
        let world = Bbox {
            xmin: -180.0,
            ymin: -85.0,
            xmax: 180.0,
            ymax: 85.0,
        };
        let err = tiles_covering(&world, 20, 10_000).unwrap_err();
        assert!(format!("{err}").contains("coarser zoom"), "{err}");
    }

    #[test]
    fn out_of_range_tiles_and_zooms_are_refused() {
        assert!(tile_of(0.0, 0.0, 31).is_err());
        assert!(tile_bbox(Tile { z: 2, x: 4, y: 0 }).is_err());
        assert!(tile_bbox(Tile { z: 2, x: -1, y: 0 }).is_err());
        assert!(to_web_mercator(200.0, 0.0).is_err());
    }
}
