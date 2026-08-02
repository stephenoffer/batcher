//! Hexagonal binning on a projected plane.
//!
//! Square bins have a defect that matters for density maps: a cell's neighbours are not
//! all the same distance away, so a diagonal neighbour is 41% further than an edge one
//! and any smoothing or "adjacent cell" logic is biased along the axes. A hexagon's six
//! neighbours are equidistant, which is why hex binning is the standard for spatial
//! density and why aggregation over hexes produces maps without the grid artefacts a
//! square grid leaves.
//!
//! This grid is planar and honest about it. It bins whatever coordinates it is given,
//! so the caller projects first — `grid::tile::to_web_mercator` for a map, an
//! equal-area projection for a density comparison across latitudes. It is deliberately
//! **not** an H3 implementation and does not produce H3 indexes: H3's cells live on an
//! icosahedron and are not a planar hex grid, so calling this H3 would be wrong in a
//! way that only surfaced when someone joined against a real H3 column.
//!
//! Cells are addressed by axial coordinates `(q, r)`, which pack into a single `Int64`
//! for use as a group key.

use crate::error::{GeoError, GeoResult};
use crate::types::{Coord, Geometry, Polygon};

/// A hexagon's axial address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hex {
    /// The column axis.
    pub q: i64,
    /// The diagonal axis.
    pub r: i64,
}

fn check_size(size: f64) -> GeoResult<()> {
    if !size.is_finite() || size <= 0.0 {
        return Err(GeoError::invalid(format!(
            "hex size must be a positive finite number, got {size}"
        )));
    }
    Ok(())
}

/// The flat-top hexagon containing `(x, y)`, where `size` is the distance from a
/// hexagon's centre to any of its corners.
pub fn hex_of(x: f64, y: f64, size: f64) -> GeoResult<Hex> {
    check_size(size)?;
    if !x.is_finite() || !y.is_finite() {
        return Err(GeoError::invalid("hex binning needs finite coordinates"));
    }
    const SQRT3: f64 = 1.732_050_807_568_877_2;
    let q = (2.0 / 3.0 * x) / size;
    let r = (-1.0 / 3.0 * x + SQRT3 / 3.0 * y) / size;
    Ok(axial_round(q, r))
}

/// Round fractional axial coordinates to the nearest hexagon.
///
/// Done in cube coordinates, where the three axes sum to zero: rounding each
/// independently can break that invariant, so the axis with the largest rounding error
/// is recomputed from the other two. Rounding in axial coordinates directly puts points
/// in the wrong cell near every corner.
fn axial_round(q: f64, r: f64) -> Hex {
    let (x, z) = (q, r);
    let y = -x - z;
    let (mut rx, ry, mut rz) = (x.round(), y.round(), z.round());
    let (dx, dy, dz) = ((rx - x).abs(), (ry - y).abs(), (rz - z).abs());
    // Recompute whichever of the two stored axes carried the largest rounding error.
    // When `y` did, nothing needs fixing: it is derived, not stored, so restoring the
    // sum-to-zero invariant costs no change to `q` or `r`.
    if dx > dy && dx > dz {
        rx = -ry - rz;
    } else if dz > dy {
        rz = -rx - ry;
    }
    Hex {
        q: rx as i64,
        r: rz as i64,
    }
}

/// The centre of a hexagon.
pub fn hex_center(h: Hex, size: f64) -> GeoResult<Coord> {
    check_size(size)?;
    const SQRT3: f64 = 1.732_050_807_568_877_2;
    let x = size * 1.5 * h.q as f64;
    let y = size * (SQRT3 / 2.0 * h.q as f64 + SQRT3 * h.r as f64);
    Ok(Coord::new(x, y))
}

/// The hexagon as a closed six-sided ring.
pub fn hex_polygon(h: Hex, size: f64) -> GeoResult<Geometry> {
    let c = hex_center(h, size)?;
    let mut ring = Vec::with_capacity(7);
    for k in 0..6 {
        let angle = std::f64::consts::PI / 3.0 * k as f64;
        ring.push(Coord::new(
            c.x + size * angle.cos(),
            c.y + size * angle.sin(),
        ));
    }
    ring.push(ring[0]);
    Ok(Geometry::Polygon(Polygon {
        exterior: ring,
        interiors: Vec::new(),
    }))
}

/// The six hexagons sharing an edge with `h`.
pub fn hex_neighbors(h: Hex) -> [Hex; 6] {
    [
        Hex { q: h.q + 1, r: h.r },
        Hex {
            q: h.q + 1,
            r: h.r - 1,
        },
        Hex { q: h.q, r: h.r - 1 },
        Hex { q: h.q - 1, r: h.r },
        Hex {
            q: h.q - 1,
            r: h.r + 1,
        },
        Hex { q: h.q, r: h.r + 1 },
    ]
}

/// The number of steps between two hexagons on the grid.
pub fn hex_distance(a: Hex, b: Hex) -> i64 {
    let (dq, dr) = (a.q - b.q, a.r - b.r);
    ((dq.abs() + dr.abs()) + (dq + dr).abs()) / 2
}

/// The half-width of the packable axial range. Each coordinate is biased by this and
/// stored in 31 bits, which keeps the packed key inside a *signed* 63-bit value — the
/// reason the range is ±2^30 rather than the ±2^31 a naive 32-bit split would suggest.
const PACK_LIMIT: i64 = 1 << 30;
/// Bits each coordinate occupies in a packed key.
const PACK_BITS: u32 = 31;

/// Pack an axial address into one `Int64` for use as a group key.
///
/// Reversible rather than hashed: two different hexagons can never collide onto the
/// same key, which a hash of the pair could not promise and which matters because the
/// key is what a `GROUP BY` counts by.
pub fn hex_key(h: Hex) -> GeoResult<i64> {
    if h.q.abs() >= PACK_LIMIT || h.r.abs() >= PACK_LIMIT {
        return Err(GeoError::invalid(format!(
            "hex ({}, {}) is outside the packable range of +/-{PACK_LIMIT}; the cell \
             size is too small for these coordinates",
            h.q, h.r
        )));
    }
    Ok(((h.q + PACK_LIMIT) << PACK_BITS) | (h.r + PACK_LIMIT))
}

/// Recover an axial address from its packed key.
pub fn hex_from_key(key: i64) -> Hex {
    Hex {
        q: (key >> PACK_BITS) - PACK_LIMIT,
        r: (key & ((1 << PACK_BITS) - 1)) - PACK_LIMIT,
    }
}

/// The area of one hexagon in the coordinate system's squared units.
pub fn hex_area(size: f64) -> GeoResult<f64> {
    check_size(size)?;
    Ok(3.0 * 1.732_050_807_568_877_2 / 2.0 * size * size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::primitive::PointRing;
    use crate::algo::relate::point_in_polygon;

    #[test]
    fn a_point_is_binned_to_the_hexagon_whose_centre_is_nearest() {
        // The defining property of the grid, and the one that is exact: a point on a
        // shared edge is equidistant from two centres, so "inside the polygon" is a
        // floating-point coin flip there while "nearest centre" is not.
        let size = 10.0;
        for i in -40..40i64 {
            for j in -40..40i64 {
                let (x, y) = (i as f64 * 3.7, j as f64 * 2.9);
                let p = Coord::new(x, y);
                let h = hex_of(x, y, size).unwrap();
                let own = crate::algo::primitive::dist(p, hex_center(h, size).unwrap());
                for n in hex_neighbors(h) {
                    let d = crate::algo::primitive::dist(p, hex_center(n, size).unwrap());
                    assert!(
                        own <= d + 1e-9,
                        "({x}, {y}) binned to {h:?} at {own} but {n:?} is nearer at {d}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_point_well_inside_a_cell_is_inside_its_polygon() {
        let size = 10.0;
        for q in -4..4 {
            for r in -4..4 {
                let h = Hex { q, r };
                let c = hex_center(h, size).unwrap();
                // Half a cell in from the centre stays clear of every boundary.
                for (dx, dy) in [(0.0, 0.0), (3.0, 0.0), (-3.0, 2.0), (0.0, -3.5)] {
                    let p = Coord::new(c.x + dx, c.y + dy);
                    assert_eq!(hex_of(p.x, p.y, size).unwrap(), h);
                    let poly = hex_polygon(h, size).unwrap();
                    assert_eq!(
                        point_in_polygon(p, poly.polygons()[0]),
                        PointRing::Inside,
                        "({}, {}) should be inside {h:?}",
                        p.x,
                        p.y
                    );
                }
            }
        }
    }

    #[test]
    fn centres_map_back_to_themselves() {
        for q in -5..5 {
            for r in -5..5 {
                let h = Hex { q, r };
                let c = hex_center(h, 7.0).unwrap();
                assert_eq!(hex_of(c.x, c.y, 7.0).unwrap(), h);
            }
        }
    }

    #[test]
    fn neighbours_are_all_one_step_away_and_equidistant() {
        let h = Hex { q: 3, r: -2 };
        let c = hex_center(h, 5.0).unwrap();
        let mut dists = Vec::new();
        for n in hex_neighbors(h) {
            assert_eq!(hex_distance(h, n), 1);
            let nc = hex_center(n, 5.0).unwrap();
            dists.push(crate::algo::primitive::dist(c, nc));
        }
        let first = dists[0];
        for d in &dists {
            assert!((d - first).abs() < 1e-9, "neighbours must be equidistant");
        }
    }

    #[test]
    fn distance_is_a_metric_on_the_grid() {
        let a = Hex { q: 0, r: 0 };
        let b = Hex { q: 3, r: -1 };
        let c = Hex { q: -2, r: 4 };
        assert_eq!(hex_distance(a, a), 0);
        assert_eq!(hex_distance(a, b), hex_distance(b, a));
        assert!(hex_distance(a, c) <= hex_distance(a, b) + hex_distance(b, c));
    }

    #[test]
    fn keys_round_trip_and_never_collide() {
        let mut seen = std::collections::HashSet::new();
        for q in -50..50 {
            for r in -50..50 {
                let h = Hex { q, r };
                let k = hex_key(h).unwrap();
                assert!(seen.insert(k), "key collision at {h:?}");
                assert_eq!(hex_from_key(k), h);
            }
        }
    }

    #[test]
    fn hex_area_matches_the_polygon_it_describes() {
        let size = 9.0;
        let poly = hex_polygon(Hex { q: 0, r: 0 }, size).unwrap();
        let measured = crate::algo::measure::area(&poly);
        assert!((measured - hex_area(size).unwrap()).abs() < 1e-9);
    }

    #[test]
    fn degenerate_input_is_refused() {
        assert!(hex_of(0.0, 0.0, 0.0).is_err());
        assert!(hex_of(0.0, 0.0, -1.0).is_err());
        assert!(hex_of(f64::NAN, 0.0, 1.0).is_err());
        assert!(hex_key(Hex { q: 1 << 40, r: 0 }).is_err());
    }
}
