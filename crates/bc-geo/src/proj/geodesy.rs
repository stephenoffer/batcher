//! Distances and areas on the Earth, in metres.
//!
//! The planar functions in `algo::measure` answer in coordinate units, which for
//! EPSG:4326 means degrees — a number that is not a distance and that changes meaning
//! with latitude. This module is where a question about the *ground* gets a ground
//! answer, and every function here takes longitude and latitude in degrees and returns
//! metres.
//!
//! Two models, and the choice between them is a real trade rather than a preference:
//!
//! * **Spherical (haversine).** One sphere of mean radius. Accurate to about 0.5%,
//!   which is a few kilometres on a transcontinental leg and a few centimetres across a
//!   city. Cheap: a handful of trigonometric calls, no iteration, no failure mode.
//!   `st_distance_sphere` and `st_dwithin_sphere` use it, and say so in their names.
//! * **Ellipsoidal (Karney on WGS 84).** Every `*_spheroid` function: distance, length,
//!   perimeter and area. Accurate to nanometres in distance and to round-off in area,
//!   defined for every pair of points including antipodal ones, and the same algorithm
//!   (GeographicLib) that PostGIS `geography`, PROJ and DuckDB spatial use — so the
//!   answers agree with theirs to about nine significant figures. See `karney`.
//!
//! Use the sphere for filtering and ranking, the ellipsoid when the number is the
//! deliverable.

use crate::error::{GeoError, GeoResult};
use crate::types::Coord;

/// Mean Earth radius in metres (IUGG), the sphere the haversine functions use.
pub const EARTH_RADIUS_M: f64 = 6_371_008.8;

/// WGS 84 semi-major axis in metres.
pub const WGS84_A: f64 = 6_378_137.0;
/// WGS 84 flattening.
pub const WGS84_F: f64 = 1.0 / 298.257_223_563;
/// WGS 84 semi-minor axis in metres.
pub const WGS84_B: f64 = WGS84_A * (1.0 - WGS84_F);

fn check_lonlat(lon: f64, lat: f64) -> GeoResult<()> {
    if !(-180.0..=180.0).contains(&lon) || !(-90.0..=90.0).contains(&lat) {
        return Err(GeoError::domain(format!(
            "geodesic functions need lon in [-180, 180] and lat in [-90, 90], got ({lon}, {lat})"
        )));
    }
    Ok(())
}

/// Great-circle distance in metres between two lon/lat positions.
///
/// Haversine rather than the spherical law of cosines: the latter loses all its
/// precision for short distances, where `cos(d/R)` is within rounding of 1, and short
/// distances are the overwhelming majority of what a proximity query asks about.
pub fn haversine(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> GeoResult<f64> {
    check_lonlat(lon1, lat1)?;
    check_lonlat(lon2, lat2)?;
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let dp = p2 - p1;
    let dl = (lon2 - lon1).to_radians();
    let a = (dp / 2.0).sin().powi(2) + p1.cos() * p2.cos() * (dl / 2.0).sin().powi(2);
    Ok(2.0 * EARTH_RADIUS_M * a.sqrt().clamp(0.0, 1.0).asin())
}

/// Initial bearing in degrees clockwise from north, in `[0, 360)`.
///
/// "Initial" is not a hedge: a great circle's bearing changes along its length, so the
/// bearing at the destination is generally not this value plus 180. A route that holds
/// one bearing is a rhumb line, which is `rhumb_bearing`.
pub fn bearing(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> GeoResult<f64> {
    check_lonlat(lon1, lat1)?;
    check_lonlat(lon2, lat2)?;
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let dl = (lon2 - lon1).to_radians();
    let y = dl.sin() * p2.cos();
    let x = p1.cos() * p2.sin() - p1.sin() * p2.cos() * dl.cos();
    Ok((y.atan2(x).to_degrees() + 360.0) % 360.0)
}

/// The position reached by travelling `distance_m` from a position along `bearing_deg`.
///
/// The inverse of `haversine` + `bearing`, and the primitive behind "everything within
/// 5 km": expanding a point into a bounding box needs the four cardinal destinations,
/// not a degree offset guessed from a latitude.
pub fn destination(lon: f64, lat: f64, bearing_deg: f64, distance_m: f64) -> GeoResult<Coord> {
    check_lonlat(lon, lat)?;
    if !distance_m.is_finite() || distance_m < 0.0 {
        return Err(GeoError::invalid(format!(
            "distance must be a non-negative number of metres, got {distance_m}"
        )));
    }
    let d = distance_m / EARTH_RADIUS_M;
    let brg = bearing_deg.to_radians();
    let p1 = lat.to_radians();
    let l1 = lon.to_radians();
    let p2 = (p1.sin() * d.cos() + p1.cos() * d.sin() * brg.cos()).asin();
    let l2 = l1 + (brg.sin() * d.sin() * p1.cos()).atan2(d.cos() - p1.sin() * p2.sin());
    // Normalize longitude into [-180, 180] so a route crossing the date line produces a
    // coordinate the rest of the stack accepts.
    let lon2 = (l2.to_degrees() + 540.0) % 360.0 - 180.0;
    Ok(Coord::new(lon2, p2.to_degrees()))
}

/// Ellipsoidal distance in metres on WGS 84, by Karney's geodesic inverse.
///
/// Converges for every pair of positions, antipodal ones included — which is the case
/// Vincenty's formula, used here before, reported as an error and the engine then
/// surfaced as a null distance between two perfectly valid points.
pub fn ellipsoidal_distance(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> GeoResult<f64> {
    check_lonlat(lon1, lat1)?;
    check_lonlat(lon2, lat2)?;
    Ok(crate::proj::karney::distance(lon1, lat1, lon2, lat2))
}

/// Rhumb-line (constant-bearing) distance in metres.
///
/// Longer than the great circle, and the one a vessel holding a compass heading
/// actually travels. Reported separately because a route length and a straight-line
/// distance are different questions and conflating them under one name is how a
/// logistics estimate quietly runs short.
pub fn rhumb_distance(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> GeoResult<f64> {
    check_lonlat(lon1, lat1)?;
    check_lonlat(lon2, lat2)?;
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let dp = p2 - p1;
    let mut dl = (lon2 - lon1).to_radians();
    // The stretched latitude difference; the limit as dp → 0 is cos(lat).
    let dpsi = ((p2 / 2.0 + std::f64::consts::FRAC_PI_4).tan()
        / (p1 / 2.0 + std::f64::consts::FRAC_PI_4).tan())
    .ln();
    let q = if dpsi.abs() > 1e-12 {
        dp / dpsi
    } else {
        p1.cos()
    };
    // Always take the shorter way round the globe.
    if dl.abs() > std::f64::consts::PI {
        dl = if dl > 0.0 {
            dl - std::f64::consts::TAU
        } else {
            dl + std::f64::consts::TAU
        };
    }
    Ok((dp * dp + q * q * dl * dl).sqrt() * EARTH_RADIUS_M)
}

/// Constant bearing of the rhumb line, in degrees clockwise from north.
pub fn rhumb_bearing(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> GeoResult<f64> {
    check_lonlat(lon1, lat1)?;
    check_lonlat(lon2, lat2)?;
    let (p1, p2) = (lat1.to_radians(), lat2.to_radians());
    let mut dl = (lon2 - lon1).to_radians();
    if dl.abs() > std::f64::consts::PI {
        dl = if dl > 0.0 {
            dl - std::f64::consts::TAU
        } else {
            dl + std::f64::consts::TAU
        };
    }
    let dpsi = ((p2 / 2.0 + std::f64::consts::FRAC_PI_4).tan()
        / (p1 / 2.0 + std::f64::consts::FRAC_PI_4).tan())
    .ln();
    Ok((dl.atan2(dpsi).to_degrees() + 360.0) % 360.0)
}

/// The ellipsoidal (WGS 84) area of a lon/lat ring, in square metres.
///
/// Edges are geodesics and the ring is closed implicitly. Correct for a ring of any
/// size — including one spanning a hemisphere, where projecting to a plane first and
/// taking the shoelace area is wrong by an unbounded factor — and for one crossing the
/// antimeridian, which the spherical-excess sum this replaced measured the long way
/// round the globe (a 20x20-degree box straddling 180 came out at 8.4e13 m² instead of
/// 4.9e12). Sign is dropped: the caller asked for an area.
pub fn ring_area_m2(ring: &[Coord]) -> GeoResult<f64> {
    for c in ring {
        check_lonlat(c.x, c.y)?;
    }
    let pts: Vec<(f64, f64)> = ring.iter().map(|c| (c.x, c.y)).collect();
    Ok(crate::proj::karney::ring_area(&pts))
}

/// The ellipsoidal area of a whole geometry in square metres, holes subtracted.
///
/// `Domain` (a null, at the expression layer) when a coordinate is off the globe.
pub fn geodesic_area_m2(g: &crate::types::Geometry) -> GeoResult<f64> {
    let mut total = 0.0;
    for p in g.polygons() {
        if p.exterior.is_empty() {
            continue;
        }
        let shell = ring_area_m2(&p.exterior)?;
        let mut holes = 0.0;
        for r in &p.interiors {
            holes += ring_area_m2(r)?;
        }
        total += (shell - holes).max(0.0);
    }
    Ok(crate::types::measurement(total))
}

/// The summed ellipsoidal length of consecutive positions of one chain.
fn chain_length_m(l: &[Coord]) -> GeoResult<f64> {
    let mut total = 0.0;
    for w in l.windows(2) {
        total += ellipsoidal_distance(w[0].x, w[0].y, w[1].x, w[1].y)?;
    }
    Ok(total)
}

/// The ellipsoidal length of every chain of a geometry in metres.
///
/// Each segment is the geodesic between its endpoints, measured on WGS 84 — the
/// definition `ST_Length_Spheroid` has in DuckDB and `ST_Length(geography)` has in
/// PostGIS. It is not the length of the straight line in lon/lat that a map draws.
pub fn geodesic_length_m(g: &crate::types::Geometry) -> GeoResult<f64> {
    let mut total = 0.0;
    for l in crate::algo::relate::linear_parts(g) {
        total += chain_length_m(l)?;
    }
    Ok(crate::types::measurement(total))
}

/// The ellipsoidal perimeter of every polygon of a geometry in metres, holes included.
pub fn geodesic_perimeter_m(g: &crate::types::Geometry) -> GeoResult<f64> {
    let mut total = 0.0;
    for p in g.polygons() {
        for ring in std::iter::once(&p.exterior).chain(p.interiors.iter()) {
            total += chain_length_m(ring)?;
        }
    }
    Ok(crate::types::measurement(total))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assert `got` is within `pct` percent of `want`.
    fn close(got: f64, want: f64, pct: f64) {
        let err = (got - want).abs() / want * 100.0;
        assert!(err < pct, "{got} vs {want}: off by {err:.4}%");
    }

    #[test]
    fn known_distances_match_published_values() {
        // London to New York: 5570 km great circle.
        close(
            haversine(-0.1278, 51.5074, -74.0060, 40.7128).unwrap(),
            5_570_000.0,
            0.5,
        );
        // One degree of latitude at the equator: about 111.2 km.
        close(haversine(0.0, 0.0, 0.0, 1.0).unwrap(), 111_195.0, 0.1);
        // The same two points are zero apart.
        assert_eq!(haversine(1.0, 2.0, 1.0, 2.0).unwrap(), 0.0);
    }

    #[test]
    fn the_ellipsoid_is_close_to_the_sphere_and_more_precise() {
        let (a, b, c, d) = (-0.1278, 51.5074, -74.0060, 40.7128);
        let h = haversine(a, b, c, d).unwrap();
        let v = ellipsoidal_distance(a, b, c, d).unwrap();
        close(v, h, 0.6);
        // The published WGS 84 value for this pair is 5 585 234 m.
        close(v, 5_585_234.0, 0.05);
    }

    #[test]
    fn antipodal_points_have_an_ellipsoidal_distance() {
        // Vincenty's non-convergent case, which used to surface as a null.
        let near = ellipsoidal_distance(0.0, 0.0, 179.9999, 0.0).unwrap();
        assert!((near - 20_003_931.457_702_395).abs() < 1e-6, "{near}");
        let exact = ellipsoidal_distance(0.0, 0.0, 180.0, 0.0).unwrap();
        assert!((exact - 20_003_931.458_625_447).abs() < 1e-6, "{exact}");
    }

    #[test]
    fn an_off_globe_coordinate_is_a_row_local_domain_error() {
        for r in [
            ellipsoidal_distance(f64::NAN, 0.0, 1.0, 1.0),
            ellipsoidal_distance(200.0, 0.0, 1.0, 1.0),
            geodesic_length_m(&crate::types::Geometry::LineString(vec![
                Coord::new(0.0, 0.0),
                Coord::new(0.0, 95.0),
            ])),
        ] {
            let e = r.unwrap_err();
            assert!(e.is_row_local(), "{e:?}");
        }
    }

    #[test]
    fn an_antimeridian_box_has_the_area_of_the_same_box_elsewhere() {
        let bx = |x0: f64, x1: f64| {
            ring_area_m2(&[
                Coord::new(x0, -10.0),
                Coord::new(x1, -10.0),
                Coord::new(x1, 10.0),
                Coord::new(x0, 10.0),
                Coord::new(x0, -10.0),
            ])
            .unwrap()
        };
        // GeographicLib PolygonArea on these four vertices: 4948480469169.516 m^2.
        let across = bx(170.0, -170.0);
        assert!(
            (across / 4_948_480_469_169.516 - 1.0).abs() < 1e-9,
            "{across}"
        );
        assert!((across / bx(-10.0, 10.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn length_and_perimeter_are_ellipsoidal() {
        // One degree of longitude on the equator is exactly a * pi / 180 on WGS 84; the
        // mean-radius sphere this used to sum over says 111 195 m, 0.1% short.
        let l = geodesic_length_m(&crate::types::Geometry::LineString(vec![
            Coord::new(0.0, 0.0),
            Coord::new(1.0, 0.0),
        ]))
        .unwrap();
        assert!(
            (l - WGS84_A * std::f64::consts::PI / 180.0).abs() < 1e-6,
            "{l}"
        );
    }

    #[test]
    fn destination_inverts_distance_and_bearing() {
        let (lon, lat) = (-122.4194, 37.7749);
        for brg in [0.0, 45.0, 90.0, 180.0, 271.0] {
            for d in [10.0, 1_000.0, 100_000.0] {
                let p = destination(lon, lat, brg, d).unwrap();
                close(haversine(lon, lat, p.x, p.y).unwrap(), d, 0.01);
                let back = bearing(lon, lat, p.x, p.y).unwrap();
                assert!(
                    (back - brg).abs() < 1e-6 || (back - brg).abs() > 359.999,
                    "{back} vs {brg}"
                );
            }
        }
    }

    #[test]
    fn bearing_points_the_right_way() {
        close(bearing(0.0, 0.0, 0.0, 1.0).unwrap() + 1.0, 1.0, 1e-6); // due north = 0
        close(bearing(0.0, 0.0, 1.0, 0.0).unwrap(), 90.0, 1e-6); // due east
        close(bearing(0.0, 0.0, 0.0, -1.0).unwrap(), 180.0, 1e-6); // due south
    }

    #[test]
    fn a_rhumb_line_is_never_shorter_than_the_great_circle() {
        for (a, b, c, d) in [
            (-0.1278, 51.5074, -74.0060, 40.7128),
            (0.0, 0.0, 90.0, 0.0),
            (-122.0, 37.0, 139.0, 35.0),
        ] {
            let gc = haversine(a, b, c, d).unwrap();
            let rl = rhumb_distance(a, b, c, d).unwrap();
            assert!(rl >= gc * 0.9999, "rhumb {rl} < great circle {gc}");
        }
        // Along the equator the two coincide.
        close(
            rhumb_distance(0.0, 0.0, 10.0, 0.0).unwrap(),
            haversine(0.0, 0.0, 10.0, 0.0).unwrap(),
            0.001,
        );
    }

    #[test]
    fn geodesic_area_matches_a_known_country_scale_polygon() {
        // A one-degree cell at the equator: about 111.2 km on a side.
        let ring = vec![
            Coord::new(0.0, 0.0),
            Coord::new(1.0, 0.0),
            Coord::new(1.0, 1.0),
            Coord::new(0.0, 1.0),
            Coord::new(0.0, 0.0),
        ];
        // GeographicLib: 12308778361.469452 m^2.
        close(ring_area_m2(&ring).unwrap(), 12_308_778_361.469_452, 1e-9);
        // Winding does not change the area.
        let mut rev = ring.clone();
        rev.reverse();
        close(
            ring_area_m2(&rev).unwrap(),
            ring_area_m2(&ring).unwrap(),
            1e-9,
        );
    }

    #[test]
    fn geodesic_area_shrinks_toward_the_pole_as_a_planar_one_does_not() {
        let cell = |lat: f64| {
            ring_area_m2(&[
                Coord::new(0.0, lat),
                Coord::new(1.0, lat),
                Coord::new(1.0, lat + 1.0),
                Coord::new(0.0, lat + 1.0),
                Coord::new(0.0, lat),
            ])
            .unwrap()
        };
        assert!(
            cell(60.0) < cell(0.0) * 0.6,
            "a high-latitude cell covers less ground"
        );
    }

    #[test]
    fn out_of_range_coordinates_are_refused() {
        assert!(haversine(200.0, 0.0, 0.0, 0.0).is_err());
        assert!(bearing(0.0, 91.0, 0.0, 0.0).is_err());
        assert!(destination(0.0, 0.0, 0.0, -5.0).is_err());
    }
}
