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
//! character. `neighbors` exists precisely so a proximity query can cover the eight
//! adjacent cells instead of silently missing everything across the seam.

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
        return Err(GeoError::invalid(format!(
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
        let mid = (range.0 + range.1) / 2.0;
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
            let mid = (range.0 + range.1) / 2.0;
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
    Ok(Coord::new((b.xmin + b.xmax) / 2.0, (b.ymin + b.ymax) / 2.0))
}

/// The eight cells around `hash`, in the order N, NE, E, SE, S, SW, W, NW.
///
/// Computed by stepping the cell centre by one cell width rather than by the classic
/// base-32 border tables: the arithmetic is the same length, is obviously correct, and
/// does not silently produce a wrong neighbour at the ±180° seam — it wraps, which is
/// the right answer, because the cell east of the date line really is on the other side.
///
/// Poleward of the top and bottom rows there is no neighbour, and those directions are
/// omitted rather than clamped onto the same row.
pub fn neighbors(hash: &str) -> GeoResult<Vec<String>> {
    let b = decode_bbox(hash)?;
    let precision = hash.len();
    let (w, h) = (b.xmax - b.xmin, b.ymax - b.ymin);
    let (cx, cy) = ((b.xmin + b.xmax) / 2.0, (b.ymin + b.ymax) / 2.0);
    let mut out = Vec::with_capacity(8);
    for (dx, dy) in [
        (0.0, 1.0),
        (1.0, 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
        (0.0, -1.0),
        (-1.0, -1.0),
        (-1.0, 0.0),
        (-1.0, 1.0),
    ] {
        let lat = cy + dy * h;
        if !(-90.0..=90.0).contains(&lat) {
            continue;
        }
        let mut lon = cx + dx * w;
        // Wrap across the antimeridian rather than dropping the neighbour.
        if lon > 180.0 {
            lon -= 360.0;
        } else if lon < -180.0 {
            lon += 360.0;
        }
        out.push(encode(lon, lat, precision)?);
    }
    Ok(out)
}

/// The shortest hash that covers the whole box, or `None` when the box straddles a
/// top-level cell boundary and no single hash contains it.
///
/// This is the operation that turns a bounding-box filter into a prefix filter: the
/// covering hash is the `LIKE` prefix that provably contains every row in the box.
pub fn covering_prefix(b: &Bbox) -> GeoResult<Option<String>> {
    let lo = encode(b.xmin.max(-180.0), b.ymin.max(-90.0), MAX_PRECISION)?;
    let hi = encode(b.xmax.min(180.0), b.ymax.min(90.0), MAX_PRECISION)?;
    let n = lo
        .bytes()
        .zip(hi.bytes())
        .take_while(|(a, b)| a == b)
        .count();
    Ok((n > 0).then(|| lo[..n].to_string()))
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
    fn neighbors_surround_the_cell_and_wrap_at_the_antimeridian() {
        let n = neighbors("9q8yyk").unwrap();
        assert_eq!(n.len(), 8);
        assert!(!n.contains(&"9q8yyk".to_string()));
        // Every neighbour's cell touches the original's box.
        let b = decode_bbox("9q8yyk").unwrap();
        for h in &n {
            let nb = decode_bbox(h).unwrap();
            assert!(b.expand(1e-9, 1e-9).intersects(&nb), "{h} is not adjacent");
        }
        // At the date line the eastern neighbours exist rather than being dropped.
        let seam = encode(179.999, 0.0, 5).unwrap();
        assert_eq!(neighbors(&seam).unwrap().len(), 8);
    }

    #[test]
    fn polar_cells_have_fewer_neighbors_rather_than_wrong_ones() {
        let top = encode(0.0, 90.0, 3).unwrap();
        let n = neighbors(&top).unwrap();
        assert!(n.len() < 8, "no cell exists north of the top row");
        assert!(!n.is_empty());
    }

    #[test]
    fn covering_prefix_contains_the_box() {
        let b = Bbox {
            xmin: -122.42,
            ymin: 37.77,
            xmax: -122.41,
            ymax: 37.78,
        };
        let p = covering_prefix(&b)
            .unwrap()
            .expect("a small box has a covering cell");
        let cell = decode_bbox(&p).unwrap();
        assert!(cell.contains(&b), "{p} must cover the box");
        // A box spanning the globe shares no prefix.
        let whole = Bbox {
            xmin: -180.0,
            ymin: -90.0,
            xmax: 180.0,
            ymax: 90.0,
        };
        assert_eq!(covering_prefix(&whole).unwrap(), None);
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
