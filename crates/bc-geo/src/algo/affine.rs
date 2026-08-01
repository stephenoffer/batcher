//! Affine transforms and grid snapping.
//!
//! All of these are `Geometry::map_coords` with a different closure, which is the
//! point: an affine transform never changes a geometry's structure, only where its
//! positions are, so topology (ring nesting, closure, vertex count) is preserved for
//! free and no operation here can produce an invalid geometry from a valid one.
//!
//! The one exception is `snap_to_grid`, which can collapse distinct positions onto the
//! same cell. It says so in its own documentation and leaves the de-duplication to
//! `construct::remove_repeated_points`, because whether a collapsed ring should be
//! dropped or kept is the caller's decision, not the transform's.

use crate::error::{GeoError, GeoResult};
use crate::types::{Coord, Geometry};

/// Shift every position by `(dx, dy, dz)`.
pub fn translate(g: &Geometry, dx: f64, dy: f64, dz: f64) -> Geometry {
    g.map_coords(&mut |c| Coord {
        x: c.x + dx,
        y: c.y + dy,
        z: c.z + dz,
    })
}

/// Scale every position about the origin.
pub fn scale(g: &Geometry, sx: f64, sy: f64, sz: f64) -> Geometry {
    g.map_coords(&mut |c| Coord {
        x: c.x * sx,
        y: c.y * sy,
        z: c.z * sz,
    })
}

/// Rotate counter-clockwise by `radians` about `(ox, oy)`.
pub fn rotate(g: &Geometry, radians: f64, ox: f64, oy: f64) -> Geometry {
    let (s, c) = radians.sin_cos();
    g.map_coords(&mut |p| {
        let (x, y) = (p.x - ox, p.y - oy);
        Coord {
            x: ox + x * c - y * s,
            y: oy + x * s + y * c,
            z: p.z,
        }
    })
}

/// The general 2D affine map `x' = a·x + b·y + xoff`, `y' = d·x + e·y + yoff`.
///
/// Named the way PostGIS `ST_Affine` names its arguments so a transform matrix can be
/// carried across from an existing pipeline without re-deriving it.
pub fn affine(g: &Geometry, a: f64, b: f64, d: f64, e: f64, xoff: f64, yoff: f64) -> Geometry {
    g.map_coords(&mut |c| Coord {
        x: a * c.x + b * c.y + xoff,
        y: d * c.x + e * c.y + yoff,
        z: c.z,
    })
}

/// Round every position onto a grid of the given cell size, anchored at `(ox, oy)`.
///
/// The canonical use is shrinking a geometry column before a shuffle: snapping to a
/// grid coarser than the data's real precision makes vertices repeat, which makes the
/// column compress and makes equality joins on geometry actually hit.
///
/// Snapping can move two distinct vertices onto one position. That is the intended
/// effect and it can leave a ring with repeated points; run
/// `construct::remove_repeated_points` after if the result feeds an areal predicate.
pub fn snap_to_grid(
    g: &Geometry,
    size_x: f64,
    size_y: f64,
    ox: f64,
    oy: f64,
) -> GeoResult<Geometry> {
    if size_x.is_nan() || size_y.is_nan() || size_x <= 0.0 || size_y <= 0.0 {
        return Err(GeoError::invalid(format!(
            "grid cell size must be positive, got ({size_x}, {size_y})"
        )));
    }
    Ok(g.map_coords(&mut |c| Coord {
        x: ((c.x - ox) / size_x).round() * size_x + ox,
        y: ((c.y - oy) / size_y).round() * size_y + oy,
        z: c.z,
    }))
}

/// Drop the z ordinate, turning a 3D geometry into a 2D one.
///
/// Structural, not numeric: `Geom::has_z` is what decides whether z is written, so the
/// caller clears that flag alongside calling this. Kept separate because the coordinate
/// rewrite and the flag live on different types.
pub fn force_2d(g: &Geometry) -> Geometry {
    g.map_coords(&mut |c| Coord {
        x: c.x,
        y: c.y,
        z: 0.0,
    })
}

/// Set a constant z on every position, making a 2D geometry 3D.
pub fn force_3d(g: &Geometry, z: f64) -> Geometry {
    g.map_coords(&mut |c| Coord { x: c.x, y: c.y, z })
}

/// Grow a geometry's bounding box by `dx`/`dy` and return it as a rectangle.
///
/// The cheap "give me everything near this" region: one box, no arcs, exact to compute
/// and exact to index — which is what makes it the right prefilter for a subsequent
/// `dwithin`, where a true buffer would be both slower and still approximate.
pub fn expand(g: &crate::Geom, dx: f64, dy: f64) -> GeoResult<Geometry> {
    if dx.is_nan() || dy.is_nan() {
        return Err(GeoError::invalid("expand distances must be numbers"));
    }
    let Some(b) = g.bbox() else {
        return Ok(Geometry::Polygon(crate::types::Polygon::default()));
    };
    let e = b.expand(dx, dy);
    crate::algo::construct::make_envelope(e.xmin, e.ymin, e.xmax, e.ymax)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::wkt::{read_wkt, write_wkt};
    use crate::Geom;

    fn geom(t: &str) -> Geometry {
        read_wkt(t).expect(t).geometry
    }

    fn wkt(g: Geometry) -> String {
        write_wkt(&Geom::new(g))
    }

    #[test]
    fn translate_and_scale_compose_as_expected() {
        assert_eq!(
            wkt(translate(&geom("POINT(1 2)"), 3.0, 4.0, 0.0)),
            "POINT(4 6)"
        );
        assert_eq!(wkt(scale(&geom("POINT(2 3)"), 2.0, 3.0, 1.0)), "POINT(4 9)");
    }

    #[test]
    fn rotation_is_counter_clockwise_about_the_given_origin() {
        let r = rotate(&geom("POINT(1 0)"), std::f64::consts::FRAC_PI_2, 0.0, 0.0);
        let c = Geom::new(r).coords()[0];
        assert!(c.x.abs() < 1e-12 && (c.y - 1.0).abs() < 1e-12, "{c:?}");
        // About (1,0) the point does not move.
        let r = rotate(&geom("POINT(1 0)"), 1.0, 1.0, 0.0);
        let c = Geom::new(r).coords()[0];
        assert!((c.x - 1.0).abs() < 1e-12 && c.y.abs() < 1e-12);
    }

    #[test]
    fn affine_reproduces_the_simple_transforms() {
        // Identity plus offset is translate.
        assert_eq!(
            wkt(affine(&geom("POINT(1 2)"), 1.0, 0.0, 0.0, 1.0, 5.0, 6.0)),
            "POINT(6 8)"
        );
    }

    #[test]
    fn snap_to_grid_collapses_and_rejects_a_zero_cell() {
        assert_eq!(
            wkt(snap_to_grid(&geom("POINT(1.234 5.678)"), 0.1, 0.1, 0.0, 0.0).unwrap()),
            "POINT(1.2000000000000002 5.7)"
        );
        assert!(snap_to_grid(&geom("POINT(0 0)"), 0.0, 1.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn transforms_preserve_structure_and_vertex_count() {
        let p = geom("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 2 1, 2 2, 1 1))");
        for out in [
            translate(&p, 1.0, 1.0, 0.0),
            scale(&p, 2.0, 2.0, 1.0),
            rotate(&p, 0.3, 0.0, 0.0),
            force_2d(&p),
        ] {
            assert_eq!(out.num_points(), p.num_points());
            assert_eq!(out.polygons()[0].interiors.len(), 1);
        }
    }

    #[test]
    fn expand_grows_the_box_on_every_side() {
        let e = expand(&read_wkt("POINT(5 5)").unwrap(), 1.0, 2.0).unwrap();
        assert_eq!(wkt(e), "POLYGON((4 3, 6 3, 6 7, 4 7, 4 3))");
    }
}
