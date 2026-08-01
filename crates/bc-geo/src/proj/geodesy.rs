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
//! * **Ellipsoidal (Vincenty on WGS 84).** Accurate to under a millimetre. Iterative,
//!   roughly an order of magnitude slower, and famously non-convergent for
//!   near-antipodal pairs — which this implementation reports rather than returning the
//!   last iterate as though it were an answer.
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
        return Err(GeoError::invalid(format!(
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

/// Ellipsoidal distance in metres on WGS 84, by Vincenty's inverse formula.
///
/// Returns an error rather than a number for the near-antipodal pairs where the
/// iteration does not converge. That case is real (opposite sides of the Earth, to
/// within a fraction of a degree) and the alternative — returning the last iterate — is
/// a plausible-looking answer that can be wrong by hundreds of kilometres.
pub fn vincenty(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> GeoResult<f64> {
    check_lonlat(lon1, lat1)?;
    check_lonlat(lon2, lat2)?;
    if lon1 == lon2 && lat1 == lat2 {
        return Ok(0.0);
    }
    let l = (lon2 - lon1).to_radians();
    let u1 = ((1.0 - WGS84_F) * lat1.to_radians().tan()).atan();
    let u2 = ((1.0 - WGS84_F) * lat2.to_radians().tan()).atan();
    let (sin_u1, cos_u1) = u1.sin_cos();
    let (sin_u2, cos_u2) = u2.sin_cos();

    let mut lambda = l;
    let mut sin_sigma;
    let mut cos_sigma;
    let mut sigma;
    let mut cos_sq_alpha;
    let mut cos2_sigma_m;
    for _ in 0..200 {
        let (sin_l, cos_l) = lambda.sin_cos();
        sin_sigma =
            ((cos_u2 * sin_l).powi(2) + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_l).powi(2)).sqrt();
        if sin_sigma == 0.0 {
            return Ok(0.0); // coincident points
        }
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_l;
        sigma = sin_sigma.atan2(cos_sigma);
        let sin_alpha = cos_u1 * cos_u2 * sin_l / sin_sigma;
        cos_sq_alpha = 1.0 - sin_alpha * sin_alpha;
        cos2_sigma_m = if cos_sq_alpha == 0.0 {
            0.0 // equatorial line
        } else {
            cos_sigma - 2.0 * sin_u1 * sin_u2 / cos_sq_alpha
        };
        let c = WGS84_F / 16.0 * cos_sq_alpha * (4.0 + WGS84_F * (4.0 - 3.0 * cos_sq_alpha));
        let lambda_prev = lambda;
        lambda = l
            + (1.0 - c)
                * WGS84_F
                * sin_alpha
                * (sigma
                    + c * sin_sigma
                        * (cos2_sigma_m
                            + c * cos_sigma * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)));
        if (lambda - lambda_prev).abs() < 1e-12 {
            let u_sq = cos_sq_alpha * (WGS84_A * WGS84_A - WGS84_B * WGS84_B) / (WGS84_B * WGS84_B);
            let a =
                1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
            let b = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));
            let delta_sigma = b
                * sin_sigma
                * (cos2_sigma_m
                    + b / 4.0
                        * (cos_sigma * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)
                            - b / 6.0
                                * cos2_sigma_m
                                * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                                * (-3.0 + 4.0 * cos2_sigma_m * cos2_sigma_m)));
            return Ok(WGS84_B * a * (sigma - delta_sigma));
        }
    }
    Err(GeoError::invalid(format!(
        "Vincenty did not converge for ({lon1}, {lat1}) to ({lon2}, {lat2}); the points \
         are nearly antipodal — use the spherical distance for this pair"
    )))
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

/// The geodesic area of a lon/lat ring, in square metres.
///
/// Computed from the spherical excess, so it is correct for a ring of any size —
/// including one spanning a hemisphere, where projecting to a plane first and taking
/// the shoelace area is wrong by an unbounded factor. Sign is dropped: the caller asked
/// for an area.
pub fn ring_area_m2(ring: &[Coord]) -> f64 {
    if ring.len() < 3 {
        return 0.0;
    }
    let mut total = 0.0;
    for w in ring.windows(2) {
        let l1 = w[0].x.to_radians();
        let l2 = w[1].x.to_radians();
        let p1 = w[0].y.to_radians();
        let p2 = w[1].y.to_radians();
        total += (l2 - l1) * (2.0 + p1.sin() + p2.sin());
    }
    // Close the ring if the caller did not.
    let (first, last) = (ring[0], ring[ring.len() - 1]);
    if first.x != last.x || first.y != last.y {
        let l1 = last.x.to_radians();
        let l2 = first.x.to_radians();
        total += (l2 - l1) * (2.0 + last.y.to_radians().sin() + first.y.to_radians().sin());
    }
    (total * EARTH_RADIUS_M * EARTH_RADIUS_M / 2.0).abs()
}

/// The geodesic area of a whole geometry in square metres, holes subtracted.
pub fn geodesic_area_m2(g: &crate::types::Geometry) -> f64 {
    crate::types::measurement(
        g.polygons()
            .iter()
            .map(|p| {
                let shell = ring_area_m2(&p.exterior);
                let holes: f64 = p.interiors.iter().map(|r| ring_area_m2(r)).sum();
                (shell - holes).max(0.0)
            })
            .sum(),
    )
}

/// The geodesic length of every chain of a geometry in metres.
pub fn geodesic_length_m(g: &crate::types::Geometry) -> GeoResult<f64> {
    let mut total = 0.0;
    for l in crate::algo::relate::linear_parts(g) {
        for w in l.windows(2) {
            total += haversine(w[0].x, w[0].y, w[1].x, w[1].y)?;
        }
    }
    Ok(crate::types::measurement(total))
}

/// The geodesic perimeter of every polygon of a geometry in metres.
pub fn geodesic_perimeter_m(g: &crate::types::Geometry) -> GeoResult<f64> {
    let mut total = 0.0;
    for p in g.polygons() {
        for ring in std::iter::once(&p.exterior).chain(p.interiors.iter()) {
            for w in ring.windows(2) {
                total += haversine(w[0].x, w[0].y, w[1].x, w[1].y)?;
            }
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
    fn vincenty_is_close_to_haversine_and_more_precise() {
        let (a, b, c, d) = (-0.1278, 51.5074, -74.0060, 40.7128);
        let h = haversine(a, b, c, d).unwrap();
        let v = vincenty(a, b, c, d).unwrap();
        close(v, h, 0.6);
        // The published WGS 84 value for this pair is 5 585 234 m.
        close(v, 5_585_234.0, 0.05);
    }

    #[test]
    fn vincenty_reports_non_convergence_rather_than_guessing() {
        // Very nearly antipodal: the classic non-convergent case.
        let r = vincenty(0.0, 0.0, 179.9999, 0.0);
        assert!(r.is_err(), "expected a non-convergence error, got {r:?}");
        assert!(format!("{}", r.unwrap_err()).contains("antipodal"));
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
        close(ring_area_m2(&ring), 1.2308e10, 0.5);
        // Winding does not change the area.
        let mut rev = ring.clone();
        rev.reverse();
        close(ring_area_m2(&rev), ring_area_m2(&ring), 1e-9);
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
