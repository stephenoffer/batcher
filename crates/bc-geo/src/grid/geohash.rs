//! Geohash — a lon/lat position as a short base-32 string.
//!
//! The property that makes it worth having in a data engine is not compactness, it is
//! that the string is a *prefix code over space*: two positions in the same cell share
//! a prefix, so a spatial "near me" becomes a `LIKE 'u09tv%'`, a spatial group-by
//! becomes an ordinary hash group-by on a string column, and a spatial sort becomes a
//! lexicographic one that keeps nearby rows nearby. All three run on machinery Batcher
//! already has, at full speed, with no spatial index.
//!
//! The failure mode is equally worth stating: prefix proximity is one-directional.
//! Sharing a prefix means being close, but being close does *not* mean sharing a
//! prefix — two positions either side of a cell boundary can differ in the first
//! character, so a proximity query must cover the adjacent cells too or it silently
//! misses everything across the seam.

use crate::error::{GeoError, GeoResult};
use crate::types::{Bbox, Coord};

/// The geohash alphabet: base 32 with `a`, `i`, `l` and `o` removed so a hash cannot be
/// misread by a human or confused with a digit.
const ALPHABET: &[u8; 32] = b"0123456789bcdefghjkmnpqrstuvwxyz";

/// The longest hash this encoder produces. Twelve characters is 60 bits, which resolves
/// to under 4 cm; beyond that the extra characters encode float noise, not position.
pub const MAX_PRECISION: usize = 12;

fn decode_char(c: u8) -> GeoResult<u32> {
    ALPHABET
        .iter()
        .position(|a| *a == c.to_ascii_lowercase())
        .map(|i| i as u32)
        .ok_or_else(|| {
            GeoError::parse(
                "geohash",
                format!("{:?} is not a geohash character", c as char),
            )
        })
}

fn check_precision(precision: usize) -> GeoResult<()> {
    if precision == 0 || precision > MAX_PRECISION {
        return Err(GeoError::invalid(format!(
            "geohash precision must be 1..={MAX_PRECISION}, got {precision}"
        )));
    }
    Ok(())
}

fn check_lonlat(lon: f64, lat: f64) -> GeoResult<()> {
    if !(-180.0..=180.0).contains(&lon) || !(-90.0..=90.0).contains(&lat) {
        return Err(GeoError::domain(format!(
            "geohash needs lon in [-180, 180] and lat in [-90, 90], got ({lon}, {lat})"
        )));
    }
    Ok(())
}

/// Encode a position at the given precision.
///
/// Bits alternate longitude-first, which is the convention every geohash
/// implementation shares and the reason a cell is wider than it is tall at odd
/// precisions.
pub fn encode(lon: f64, lat: f64, precision: usize) -> GeoResult<String> {
    check_precision(precision)?;
    check_lonlat(lon, lat)?;
    let mut lon_range = (-180.0f64, 180.0f64);
    let mut lat_range = (-90.0f64, 90.0f64);
    let mut out = String::with_capacity(precision);
    let mut bit = 0;
    let mut acc = 0u32;
    let mut even = true;
    while out.len() < precision {
        let (range, value) = if even {
            (&mut lon_range, lon)
        } else {
            (&mut lat_range, lat)
        };
        let mid = f64::midpoint(range.0, range.1);
        if value >= mid {
            acc = (acc << 1) | 1;
            range.0 = mid;
        } else {
            acc <<= 1;
            range.1 = mid;
        }
        even = !even;
        bit += 1;
        if bit == 5 {
            out.push(ALPHABET[acc as usize] as char);
            bit = 0;
            acc = 0;
        }
    }
    Ok(out)
}

/// The cell a hash names, as a bounding box.
pub fn decode_bbox(hash: &str) -> GeoResult<Bbox> {
    if hash.is_empty() {
        return Err(GeoError::parse("geohash", "hash is empty"));
    }
    let mut lon_range = (-180.0f64, 180.0f64);
    let mut lat_range = (-90.0f64, 90.0f64);
    let mut even = true;
    for c in hash.bytes() {
        let v = decode_char(c)?;
        for shift in (0..5).rev() {
            let bit = (v >> shift) & 1;
            let range = if even { &mut lon_range } else { &mut lat_range };
            let mid = f64::midpoint(range.0, range.1);
            if bit == 1 {
                range.0 = mid;
            } else {
                range.1 = mid;
            }
            even = !even;
        }
    }
    Ok(Bbox {
        xmin: lon_range.0,
        ymin: lat_range.0,
        xmax: lon_range.1,
        ymax: lat_range.1,
    })
}

/// The centre of the cell a hash names.
pub fn decode(hash: &str) -> GeoResult<Coord> {
    let b = decode_bbox(hash)?;
    Ok(Coord::new(
        f64::midpoint(b.xmin, b.xmax),
        f64::midpoint(b.ymin, b.ymax),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_hashes_match_the_reference_implementation() {
        // The two examples Wikipedia's geohash article publishes, which every
        // implementation is checked against, plus the origin.
        assert_eq!(encode(-5.6, 42.6, 5).unwrap(), "ezs42");
        assert_eq!(encode(10.40744, 57.64911, 11).unwrap(), "u4pruydqqvj");
        assert_eq!(encode(0.0, 0.0, 5).unwrap(), "s0000");
    }

    #[test]
    fn decoding_lands_back_in_the_cell() {
        for (lon, lat) in [
            (-122.4194, 37.7749),
            (0.0, 0.0),
            (179.9, -89.9),
            (-180.0, 90.0),
        ] {
            for p in 1..=MAX_PRECISION {
                let h = encode(lon, lat, p).unwrap();
                let b = decode_bbox(&h).unwrap();
                assert!(
                    b.contains_coord(Coord::new(lon, lat)),
                    "{h} at precision {p} does not contain ({lon}, {lat})"
                );
                let c = decode(&h).unwrap();
                assert_eq!(
                    encode(c.x, c.y, p).unwrap(),
                    h,
                    "centre re-encodes to itself"
                );
            }
        }
    }

    #[test]
    fn prefixes_nest() {
        let long = encode(-122.4194, 37.7749, 9).unwrap();
        for p in 1..9 {
            let short = encode(-122.4194, 37.7749, p).unwrap();
            assert!(long.starts_with(&short), "{short} must prefix {long}");
        }
    }

    #[test]
    fn bad_input_is_refused() {
        assert!(encode(181.0, 0.0, 5).is_err());
        assert!(encode(0.0, 91.0, 5).is_err());
        assert!(encode(0.0, 0.0, 0).is_err());
        assert!(decode_bbox("").is_err());
        assert!(
            decode_bbox("aio").is_err(),
            "a, i and o are not in the alphabet"
        );
    }

    #[test]
    fn case_is_ignored_on_decode() {
        assert_eq!(
            decode_bbox("9Q8YYK").unwrap(),
            decode_bbox("9q8yyk").unwrap()
        );
    }
}
