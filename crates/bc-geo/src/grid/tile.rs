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
use crate::types::Bbox;

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
        return Err(GeoError::domain(format!(
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

/// The tile a quadkey names: the inverse of [`quadkey`], kept as its round-trip oracle.
#[cfg(test)]
fn from_quadkey(key: &str) -> GeoResult<Tile> {
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
    fn a_pole_maps_to_the_edge_tile_rather_than_off_the_grid() {
        assert!(tile_of(0.0, 90.0, 5).unwrap().y == 0);
    }

    #[test]
    fn out_of_range_tiles_and_zooms_are_refused() {
        assert!(tile_of(0.0, 0.0, 31).is_err());
        assert!(tile_bbox(Tile { z: 2, x: 4, y: 0 }).is_err());
        assert!(tile_bbox(Tile { z: 2, x: -1, y: 0 }).is_err());
    }
}
