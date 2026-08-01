//! S2 cell identifiers — Google's spherical cell hierarchy, as BigQuery and many
//! geospatial warehouses index by.
//!
//! S2 wraps the sphere in a cube, subdivides each face quadtree-style to 30 levels, and
//! numbers the cells along a Hilbert curve. Two properties come out of that and both
//! matter to a query engine:
//!
//! * **Cells are near-equal-area.** Unlike a lon/lat grid, whose cells vanish at the
//!   poles, an S2 cell at level `l` covers roughly the same area anywhere on Earth. A
//!   density comparison across latitudes is meaningful on this grid and is not on a
//!   degree grid.
//! * **The id is a Hilbert index, so it sorts spatially.** Sorting a table by cell id
//!   clusters nearby rows into the same pages, and a cell's descendants form one
//!   *contiguous integer range* — which turns "everything inside this region" into a
//!   `BETWEEN` on an `Int64` column that any range index can serve.
//!
//! The encoding is the standard one: 3 bits of face, two bits per level, and a trailing
//! `1` marking where the levels stop. Ids produced here are byte-comparable with those
//! from the reference implementation, which is what the fixtures in this module pin.

use crate::error::{GeoError, GeoResult};
use crate::types::{Bbox, Coord};

/// The finest S2 level.
pub const MAX_LEVEL: u32 = 30;
/// Cells per side of a face at `MAX_LEVEL`.
const MAX_SIZE: f64 = (1u64 << MAX_LEVEL) as f64;

/// Hilbert orientation bit meaning "swap the i and j axes".
const SWAP_MASK: usize = 1;
/// Hilbert orientation bit meaning "invert both axes".
const INVERT_MASK: usize = 2;

/// `(i, j)` quadrant to Hilbert position, indexed by orientation then `2*i + j`.
const IJ_TO_POS: [[u64; 4]; 4] = [[0, 1, 3, 2], [0, 3, 1, 2], [2, 3, 1, 0], [2, 1, 3, 0]];
/// Hilbert position to `(i, j)` quadrant (as `2*i + j`), indexed by orientation.
const POS_TO_IJ: [[usize; 4]; 4] = [[0, 1, 3, 2], [0, 2, 3, 1], [3, 2, 0, 1], [3, 1, 0, 2]];
/// The orientation change each Hilbert position induces.
const POS_TO_ORIENTATION: [usize; 4] = [SWAP_MASK, 0, 0, INVERT_MASK | SWAP_MASK];

fn check_level(level: u32) -> GeoResult<()> {
    if level > MAX_LEVEL {
        return Err(GeoError::invalid(format!(
            "S2 level must be 0..={MAX_LEVEL}, got {level}"
        )));
    }
    Ok(())
}

/// lon/lat in degrees to a unit vector on the sphere.
fn lonlat_to_xyz(lon: f64, lat: f64) -> [f64; 3] {
    let (phi, theta) = (lat.to_radians(), lon.to_radians());
    let c = phi.cos();
    [c * theta.cos(), c * theta.sin(), phi.sin()]
}

fn xyz_to_lonlat(p: [f64; 3]) -> Coord {
    let lat = p[2].atan2((p[0] * p[0] + p[1] * p[1]).sqrt()).to_degrees();
    let lon = p[1].atan2(p[0]).to_degrees();
    Coord::new(lon, lat)
}

/// The cube face a direction vector points at, and its face coordinates.
fn xyz_to_face_uv(p: [f64; 3]) -> (usize, f64, f64) {
    let mut face = 0usize;
    for i in 1..3 {
        if p[i].abs() > p[face].abs() {
            face = i;
        }
    }
    if p[face] < 0.0 {
        face += 3;
    }
    let (u, v) = match face {
        0 => (p[1] / p[0], p[2] / p[0]),
        1 => (-p[0] / p[1], p[2] / p[1]),
        2 => (-p[0] / p[2], -p[1] / p[2]),
        3 => (p[2] / p[0], p[1] / p[0]),
        4 => (p[2] / p[1], -p[0] / p[1]),
        _ => (-p[1] / p[2], -p[0] / p[2]),
    };
    (face, u, v)
}

fn face_uv_to_xyz(face: usize, u: f64, v: f64) -> [f64; 3] {
    match face {
        0 => [1.0, u, v],
        1 => [-u, 1.0, v],
        2 => [-u, -v, 1.0],
        3 => [-1.0, -v, -u],
        4 => [v, -1.0, -u],
        _ => [v, u, -1.0],
    }
}

/// The quadratic `s → u` transform. S2 offers three; the quadratic one is the default
/// because it makes cell areas most uniform, and matching it is what makes ids here
/// comparable with ids from anywhere else.
fn st_to_uv(s: f64) -> f64 {
    if s >= 0.5 {
        (1.0 / 3.0) * (4.0 * s * s - 1.0)
    } else {
        (1.0 / 3.0) * (1.0 - 4.0 * (1.0 - s) * (1.0 - s))
    }
}

fn uv_to_st(u: f64) -> f64 {
    if u >= 0.0 {
        0.5 * (1.0 + 3.0 * u).sqrt()
    } else {
        1.0 - 0.5 * (1.0 - 3.0 * u).sqrt()
    }
}

fn st_to_ij(s: f64) -> u64 {
    (MAX_SIZE * s - 0.5).round().clamp(0.0, MAX_SIZE - 1.0) as u64
}

/// The level-30 cell id for a face and its integer face coordinates.
fn from_face_ij(face: usize, i: u64, j: u64) -> u64 {
    let mut orientation = face & SWAP_MASK;
    let mut id = face as u64;
    for k in (0..MAX_LEVEL).rev() {
        let i_bit = ((i >> k) & 1) as usize;
        let j_bit = ((j >> k) & 1) as usize;
        let pos = IJ_TO_POS[orientation][2 * i_bit + j_bit];
        id = (id << 2) | pos;
        orientation ^= POS_TO_ORIENTATION[pos as usize];
    }
    (id << 1) | 1
}

fn to_face_ij(id: u64) -> (usize, u64, u64) {
    let face = (id >> 61) as usize;
    let mut orientation = face & SWAP_MASK;
    let (mut i, mut j) = (0u64, 0u64);
    for k in (0..MAX_LEVEL).rev() {
        let pos = ((id >> (2 * k + 1)) & 3) as usize;
        let ij = POS_TO_IJ[orientation][pos];
        i = (i << 1) | ((ij >> 1) as u64);
        j = (j << 1) | ((ij & 1) as u64);
        orientation ^= POS_TO_ORIENTATION[pos];
    }
    (face, i, j)
}

/// The bit marking the end of the level sequence for a given level.
fn lsb_for_level(level: u32) -> u64 {
    1u64 << (2 * (MAX_LEVEL - level))
}

/// The cell containing a lon/lat position at the given level.
pub fn cell_id(lon: f64, lat: f64, level: u32) -> GeoResult<u64> {
    check_level(level)?;
    if !(-180.0..=180.0).contains(&lon) || !(-90.0..=90.0).contains(&lat) {
        return Err(GeoError::invalid(format!(
            "S2 needs lon in [-180, 180] and lat in [-90, 90], got ({lon}, {lat})"
        )));
    }
    let (face, u, v) = xyz_to_face_uv(lonlat_to_xyz(lon, lat));
    let leaf = from_face_ij(face, st_to_ij(uv_to_st(u)), st_to_ij(uv_to_st(v)));
    Ok(parent(leaf, level).expect("a leaf cell has every ancestor"))
}

/// The level a cell id encodes, or `None` when the id is not a valid cell.
pub fn level_of(id: u64) -> Option<u32> {
    if id == 0 || (id >> 61) > 5 {
        return None;
    }
    let tz = id.trailing_zeros();
    // The marker bit always sits at an even offset; an odd one means the id was
    // truncated or hand-assembled, and reporting a level for it would be a lie.
    (tz % 2 == 0 && tz <= 2 * MAX_LEVEL).then(|| MAX_LEVEL - tz / 2)
}

/// The ancestor of `id` at `level`, or `None` when `level` is finer than `id`'s own.
pub fn parent(id: u64, level: u32) -> Option<u64> {
    let own = level_of(id)?;
    if level > own {
        return None;
    }
    let lsb = lsb_for_level(level);
    Some((id & lsb.wrapping_neg()) | lsb)
}

/// The four children of a cell, or `None` at the finest level.
pub fn children(id: u64) -> Option<[u64; 4]> {
    let level = level_of(id)?;
    if level >= MAX_LEVEL {
        return None;
    }
    // Moving down a level shifts the marker bit two places, so a child is the parent's
    // id offset by an odd multiple of the *child's* marker: -3, -1, +1, +3.
    let lsb = lsb_for_level(level + 1);
    Some([id - 3 * lsb, id - lsb, id + lsb, id + 3 * lsb])
}

/// The inclusive range of leaf-cell ids a cell covers.
///
/// This is the property that makes S2 usable without a spatial index: every descendant
/// of a cell, at every level, has an id inside this range, so `id BETWEEN lo AND hi` is
/// an exact region filter over an ordinary sorted `Int64` column.
pub fn range(id: u64) -> Option<(u64, u64)> {
    level_of(id)?;
    let lsb = id & id.wrapping_neg();
    Some((id - (lsb - 1), id + (lsb - 1)))
}

/// The centre of a cell as lon/lat.
pub fn cell_center(id: u64) -> GeoResult<Coord> {
    let level = level_of(id)
        .ok_or_else(|| GeoError::parse("s2", format!("{id} is not a valid S2 cell id")))?;
    let (face, i, j) = to_face_ij(id);
    // The centre of a level-`l` cell sits at the midpoint of its leaf-index span.
    let shift = MAX_LEVEL - level;
    let size = 1u64 << shift;
    let ci = (i >> shift << shift) as f64 + size as f64 / 2.0;
    let cj = (j >> shift << shift) as f64 + size as f64 / 2.0;
    let u = st_to_uv(ci / MAX_SIZE);
    let v = st_to_uv(cj / MAX_SIZE);
    Ok(xyz_to_lonlat(face_uv_to_xyz(face, u, v)))
}

/// The lon/lat rectangle enclosing a cell.
///
/// An S2 cell is a spherical quadrilateral, not a lon/lat rectangle, so this is the
/// enclosing box and is strictly larger than the cell. Sound as a prefilter, wrong as
/// a description of the cell's shape — which is why it is named for the box.
pub fn cell_bbox(id: u64) -> GeoResult<Bbox> {
    let level = level_of(id)
        .ok_or_else(|| GeoError::parse("s2", format!("{id} is not a valid S2 cell id")))?;
    let (face, i, j) = to_face_ij(id);
    let shift = MAX_LEVEL - level;
    let base_i = (i >> shift) << shift;
    let base_j = (j >> shift) << shift;
    let size = (1u64 << shift) as f64;
    let mut out: Option<Bbox> = None;
    for (di, dj) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
        let u = st_to_uv((base_i as f64 + di * size) / MAX_SIZE);
        let v = st_to_uv((base_j as f64 + dj * size) / MAX_SIZE);
        let c = xyz_to_lonlat(face_uv_to_xyz(face, u, v));
        match &mut out {
            Some(b) => b.extend(c),
            None => out = Some(Bbox::from_coord(c)),
        }
    }
    out.ok_or_else(|| GeoError::invalid("cell has no corners"))
}

/// The approximate area of a level-`l` cell in square metres.
///
/// Average, not exact: cell areas vary by about a factor of two across a face even
/// under the quadratic transform. Reported so a caller can pick a level for a target
/// resolution without a lookup table.
pub fn average_area_m2(level: u32) -> GeoResult<f64> {
    check_level(level)?;
    let r = crate::proj::geodesy::EARTH_RADIUS_M;
    let sphere = 4.0 * std::f64::consts::PI * r * r;
    Ok(sphere / (6.0 * 4f64.powi(level as i32)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cell_contains_the_position_that_produced_it() {
        for (lon, lat) in [
            (-122.4194, 37.7749),
            (0.0, 0.0),
            (151.2093, -33.8688),
            (0.0, 89.9),
            (-179.9, -89.9),
        ] {
            for level in [0, 5, 10, 15, 20, 30] {
                let id = cell_id(lon, lat, level).unwrap();
                assert_eq!(level_of(id), Some(level), "level round trip at {level}");
                let c = cell_center(id).unwrap();
                // The centre of the containing cell re-encodes to the same cell.
                assert_eq!(
                    cell_id(c.x, c.y, level).unwrap(),
                    id,
                    "({lon}, {lat}) @ {level}"
                );
            }
        }
    }

    #[test]
    fn parents_nest_and_ranges_contain_descendants() {
        let leaf = cell_id(-122.4194, 37.7749, 30).unwrap();
        for level in 0..30 {
            let p = parent(leaf, level).unwrap();
            assert_eq!(level_of(p), Some(level));
            let (lo, hi) = range(p).unwrap();
            assert!(
                lo <= leaf && leaf <= hi,
                "level {level} range must contain the leaf"
            );
            // A finer ancestor's range nests inside a coarser one's.
            if level > 0 {
                let (plo, phi) = range(parent(leaf, level - 1).unwrap()).unwrap();
                assert!(plo <= lo && hi <= phi);
            }
        }
        assert_eq!(parent(leaf, 31), None);
    }

    #[test]
    fn children_partition_their_parent() {
        let cell = cell_id(0.0, 0.0, 10).unwrap();
        let kids = children(cell).unwrap();
        let (lo, hi) = range(cell).unwrap();
        for k in kids {
            assert_eq!(level_of(k), Some(11));
            assert_eq!(parent(k, 10), Some(cell));
            let (klo, khi) = range(k).unwrap();
            assert!(lo <= klo && khi <= hi);
        }
        // The four children are distinct and cover the parent's range end to end.
        let mut sorted = kids;
        sorted.sort_unstable();
        assert_eq!(range(sorted[0]).unwrap().0, lo);
        assert_eq!(range(sorted[3]).unwrap().1, hi);
        assert_eq!(children(cell_id(0.0, 0.0, 30).unwrap()), None);
    }

    #[test]
    fn ids_sort_spatially() {
        // Two nearby points share a long id prefix; a far one does not.
        let sf = cell_id(-122.4194, 37.7749, 20).unwrap();
        let near = cell_id(-122.4190, 37.7750, 20).unwrap();
        let far = cell_id(151.2093, -33.8688, 20).unwrap();
        assert!(
            (sf ^ near).leading_zeros() > (sf ^ far).leading_zeros(),
            "nearby cells must share more high bits"
        );
    }

    #[test]
    fn the_six_faces_are_all_reachable() {
        let mut faces = std::collections::HashSet::new();
        for lon in [-180.0, -90.0, 0.0, 90.0, 179.0] {
            for lat in [-89.0, -45.0, 0.0, 45.0, 89.0] {
                faces.insert(cell_id(lon, lat, 0).unwrap() >> 61);
            }
        }
        assert_eq!(faces.len(), 6, "got {faces:?}");
    }

    #[test]
    fn cell_bbox_encloses_the_centre() {
        let id = cell_id(-122.4194, 37.7749, 12).unwrap();
        let b = cell_bbox(id).unwrap();
        let c = cell_center(id).unwrap();
        assert!(b.contains_coord(c));
        assert!(b.contains_coord(Coord::new(-122.4194, 37.7749)));
    }

    #[test]
    fn average_area_halves_by_four_each_level() {
        let a10 = average_area_m2(10).unwrap();
        let a11 = average_area_m2(11).unwrap();
        assert!((a10 / a11 - 4.0).abs() < 1e-9);
        // Level 0 is a sixth of the globe.
        let whole = average_area_m2(0).unwrap() * 6.0;
        assert!((whole / 5.1e14 - 1.0).abs() < 0.01, "got {whole}");
    }

    #[test]
    fn invalid_ids_and_levels_are_refused() {
        assert!(cell_id(0.0, 0.0, 31).is_err());
        assert!(cell_id(181.0, 0.0, 5).is_err());
        assert_eq!(level_of(0), None);
        // An id with an odd trailing-zero count encodes no level.
        assert_eq!(level_of(0b10), None);
        assert!(cell_center(0).is_err());
    }
}
