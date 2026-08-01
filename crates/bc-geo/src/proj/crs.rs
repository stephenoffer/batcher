//! Coordinate reference system transforms, for a deliberately small set of systems.
//!
//! A general CRS engine is a database (PROJ ships one) and a network of grid-shift
//! files. Vendoring that is out of scope for a query engine, so this module supports
//! the four systems that cover the overwhelming majority of analytics work and
//! **refuses everything else by EPSG code** rather than silently returning the input.
//! An unsupported transform that quietly did nothing would produce coordinates that
//! look plausible and are wrong by hundreds of kilometres.
//!
//! | EPSG | System | Use it for |
//! |---|---|---|
//! | 4326 | WGS 84 lon/lat degrees | storage, interchange, the geodesic functions |
//! | 3857 | Web Mercator metres | map tiles, and anything drawn on a slippy map |
//! | 326xx / 327xx | UTM zone metres | local distance and area, to a metre, within a zone |
//! | 6933 | Cylindrical equal area metres | density comparisons across latitudes |
//!
//! The transforms are datum-free: every system here is on WGS 84, so a conversion is a
//! projection change and nothing is lost. Reprojecting between datums (NAD 27, OSGB 36)
//! needs a grid shift this module does not have, and their codes are rejected.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

use crate::error::{GeoError, GeoResult};
use crate::proj::geodesy::{WGS84_A, WGS84_F};
use crate::types::{Coord, Geometry};

/// WGS 84 geographic coordinates in degrees.
pub const EPSG_WGS84: i32 = 4326;
/// Web Mercator, in metres.
pub const EPSG_WEB_MERCATOR: i32 = 3857;
/// WGS 84 / NSIDC EASE-Grid 2.0 Global — a cylindrical equal-area projection in metres.
pub const EPSG_EQUAL_AREA: i32 = 6933;

/// The first eccentricity squared of the WGS 84 ellipsoid.
const E2: f64 = WGS84_F * (2.0 - WGS84_F);

/// The UTM zone number for a longitude, 1..=60.
pub fn utm_zone(lon: f64) -> GeoResult<u32> {
    if !(-180.0..=180.0).contains(&lon) {
        return Err(GeoError::invalid(format!(
            "UTM zone needs lon in [-180, 180], got {lon}"
        )));
    }
    Ok((((lon + 180.0) / 6.0).floor() as u32).min(59) + 1)
}

/// The EPSG code of the UTM zone covering a position.
///
/// Northern-hemisphere zones are 326xx and southern ones 327xx, which is the convention
/// every EPSG-aware tool uses. A dataset spanning the equator therefore has no single
/// UTM code, and that is a property of UTM rather than a limitation here.
pub fn utm_epsg(lon: f64, lat: f64) -> GeoResult<i32> {
    if !(-90.0..=90.0).contains(&lat) {
        return Err(GeoError::invalid(format!(
            "UTM zone needs lat in [-90, 90], got {lat}"
        )));
    }
    let zone = utm_zone(lon)? as i32;
    Ok(if lat >= 0.0 { 32600 + zone } else { 32700 + zone })
}

/// Split a UTM EPSG code into its zone and hemisphere.
fn parse_utm(epsg: i32) -> Option<(u32, bool)> {
    let (base, north) = if (32601..=32660).contains(&epsg) {
        (32600, true)
    } else if (32701..=32760).contains(&epsg) {
        (32700, false)
    } else {
        return None;
    };
    Some(((epsg - base) as u32, north))
}

/// True when this module can transform to and from `epsg`.
pub fn is_supported(epsg: i32) -> bool {
    epsg == EPSG_WGS84
        || epsg == EPSG_WEB_MERCATOR
        || epsg == EPSG_EQUAL_AREA
        || parse_utm(epsg).is_some()
}

/// A human-readable name for a supported EPSG code.
pub fn crs_name(epsg: i32) -> Option<String> {
    if epsg == EPSG_WGS84 {
        return Some("WGS 84 (lon/lat degrees)".to_string());
    }
    if epsg == EPSG_WEB_MERCATOR {
        return Some("WGS 84 / Pseudo-Mercator (metres)".to_string());
    }
    if epsg == EPSG_EQUAL_AREA {
        return Some("WGS 84 / NSIDC EASE-Grid 2.0 Global (equal-area metres)".to_string());
    }
    parse_utm(epsg).map(|(zone, north)| {
        format!(
            "WGS 84 / UTM zone {zone}{} (metres)",
            if north { "N" } else { "S" }
        )
    })
}

fn unsupported(epsg: i32) -> GeoError {
    GeoError::invalid(format!(
        "EPSG:{epsg} is not a supported CRS. Supported: 4326 (WGS 84 lon/lat), \
         3857 (Web Mercator), 6933 (equal area), 32601-32660 and 32701-32760 (UTM). \
         Reproject with a full PROJ-backed tool before loading, or state the data's \
         CRS with st_set_srid if it is already in one of these."
    ))
}

/// Project WGS 84 lon/lat to a UTM zone's easting and northing in metres.
fn to_utm(lon: f64, lat: f64, zone: u32, north: bool) -> Coord {
    let k0 = 0.9996;
    let lon0 = ((zone as f64 - 1.0) * 6.0 - 180.0 + 3.0).to_radians();
    let (phi, lam) = (lat.to_radians(), lon.to_radians());
    let n = WGS84_A / (1.0 - E2 * phi.sin().powi(2)).sqrt();
    let t = phi.tan().powi(2);
    let c = E2 / (1.0 - E2) * phi.cos().powi(2);
    let a = phi.cos() * (lam - lon0);
    let e4 = E2 * E2;
    let e6 = e4 * E2;
    let m = WGS84_A
        * ((1.0 - E2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * phi
            - (3.0 * E2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * (2.0 * phi).sin()
            + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * (4.0 * phi).sin()
            - (35.0 * e6 / 3072.0) * (6.0 * phi).sin());
    let easting = k0
        * n
        * (a + (1.0 - t + c) * a.powi(3) / 6.0
            + (5.0 - 18.0 * t + t * t + 72.0 * c - 58.0 * E2 / (1.0 - E2)) * a.powi(5) / 120.0)
        + 500_000.0;
    let mut northing = k0
        * (m + n
            * phi.tan()
            * (a * a / 2.0
                + (5.0 - t + 9.0 * c + 4.0 * c * c) * a.powi(4) / 24.0
                + (61.0 - 58.0 * t + t * t + 600.0 * c - 330.0 * E2 / (1.0 - E2)) * a.powi(6)
                    / 720.0));
    if !north {
        northing += 10_000_000.0;
    }
    Coord::new(easting, northing)
}

/// Invert `to_utm`.
fn from_utm(easting: f64, northing: f64, zone: u32, north: bool) -> Coord {
    let k0 = 0.9996;
    let lon0 = ((zone as f64 - 1.0) * 6.0 - 180.0 + 3.0).to_radians();
    let x = easting - 500_000.0;
    let y = if north {
        northing
    } else {
        northing - 10_000_000.0
    };
    let e1 = (1.0 - (1.0 - E2).sqrt()) / (1.0 + (1.0 - E2).sqrt());
    let e4 = E2 * E2;
    let e6 = e4 * E2;
    let m = y / k0;
    let mu = m / (WGS84_A * (1.0 - E2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0));
    let phi1 = mu
        + (3.0 * e1 / 2.0 - 27.0 * e1.powi(3) / 32.0) * (2.0 * mu).sin()
        + (21.0 * e1 * e1 / 16.0 - 55.0 * e1.powi(4) / 32.0) * (4.0 * mu).sin()
        + (151.0 * e1.powi(3) / 96.0) * (6.0 * mu).sin()
        + (1097.0 * e1.powi(4) / 512.0) * (8.0 * mu).sin();
    let c1 = E2 / (1.0 - E2) * phi1.cos().powi(2);
    let t1 = phi1.tan().powi(2);
    let n1 = WGS84_A / (1.0 - E2 * phi1.sin().powi(2)).sqrt();
    let r1 = WGS84_A * (1.0 - E2) / (1.0 - E2 * phi1.sin().powi(2)).powf(1.5);
    let d = x / (n1 * k0);
    let phi = phi1
        - (n1 * phi1.tan() / r1)
            * (d * d / 2.0
                - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1 * c1 - 9.0 * E2 / (1.0 - E2)) * d.powi(4)
                    / 24.0
                + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1 * t1
                    - 252.0 * E2 / (1.0 - E2)
                    - 3.0 * c1 * c1)
                    * d.powi(6)
                    / 720.0);
    let lam = lon0
        + (d - (1.0 + 2.0 * t1 + c1) * d.powi(3) / 6.0
            + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1 * c1 + 8.0 * E2 / (1.0 - E2) + 24.0 * t1 * t1)
                * d.powi(5)
                / 120.0)
            / phi1.cos();
    Coord::new(lam.to_degrees(), phi.to_degrees())
}

/// The standard parallel of EPSG:6933, in radians.
const EASE_STD_PARALLEL: f64 = std::f64::consts::FRAC_PI_6;

/// Project lon/lat to the EPSG:6933 cylindrical equal-area plane.
fn to_equal_area(lon: f64, lat: f64) -> Coord {
    let phi0 = EASE_STD_PARALLEL;
    let k0 = phi0.cos() / (1.0 - E2 * phi0.sin().powi(2)).sqrt();
    let q = authalic_q(lat.to_radians());
    Coord::new(
        WGS84_A * k0 * lon.to_radians(),
        WGS84_A * q / (2.0 * k0),
    )
}

/// Invert `to_equal_area`.
fn from_equal_area(x: f64, y: f64) -> Coord {
    let phi0 = EASE_STD_PARALLEL;
    let k0 = phi0.cos() / (1.0 - E2 * phi0.sin().powi(2)).sqrt();
    let lon = (x / (WGS84_A * k0)).to_degrees();
    let q = 2.0 * k0 * y / WGS84_A;
    // Invert the authalic latitude by Newton iteration; it converges in a handful of
    // steps for every latitude and has no singularity at the pole.
    let mut phi = (q / 2.0).asin().clamp(-FRAC_PI_2, FRAC_PI_2);
    for _ in 0..12 {
        let s = phi.sin();
        let denom = 1.0 - E2 * s * s;
        let f = authalic_q(phi) - q;
        let dfd = (1.0 - E2) * (2.0 * phi.cos() / (denom * denom));
        if dfd.abs() < 1e-15 {
            break;
        }
        let step = f / dfd;
        phi -= step;
        if step.abs() < 1e-14 {
            break;
        }
    }
    Coord::new(lon, phi.clamp(-FRAC_PI_2, FRAC_PI_2).to_degrees())
}

/// The authalic (equal-area) parameter `q` for a geodetic latitude.
fn authalic_q(phi: f64) -> f64 {
    let s = phi.sin();
    let e = E2.sqrt();
    if e == 0.0 {
        return 2.0 * s;
    }
    (1.0 - E2) * (s / (1.0 - E2 * s * s) - (1.0 / (2.0 * e)) * ((1.0 - e * s) / (1.0 + e * s)).ln())
}

/// Convert a position from WGS 84 lon/lat to `epsg`.
fn from_wgs84(c: Coord, epsg: i32) -> GeoResult<Coord> {
    if epsg == EPSG_WGS84 {
        return Ok(c);
    }
    if epsg == EPSG_WEB_MERCATOR {
        let lat = c.y.clamp(
            -crate::grid::tile::MERCATOR_MAX_LAT,
            crate::grid::tile::MERCATOR_MAX_LAT,
        );
        // EPSG:3857 is a *spherical* Mercator on the WGS 84 semi-major axis: it treats
        // the ellipsoid as a sphere of radius `a`, which is what makes it "pseudo".
        return Ok(Coord::new(
            WGS84_A * c.x.to_radians(),
            WGS84_A * ((FRAC_PI_4 + lat.to_radians() / 2.0).tan()).ln(),
        ));
    }
    if epsg == EPSG_EQUAL_AREA {
        return Ok(to_equal_area(c.x, c.y));
    }
    if let Some((zone, north)) = parse_utm(epsg) {
        return Ok(to_utm(c.x, c.y, zone, north));
    }
    Err(unsupported(epsg))
}

/// Convert a position from `epsg` to WGS 84 lon/lat.
fn to_wgs84(c: Coord, epsg: i32) -> GeoResult<Coord> {
    if epsg == EPSG_WGS84 {
        return Ok(c);
    }
    if epsg == EPSG_WEB_MERCATOR {
        return Ok(Coord::new(
            (c.x / WGS84_A).to_degrees(),
            (2.0 * (c.y / WGS84_A).exp().atan() - FRAC_PI_2).to_degrees(),
        ));
    }
    if epsg == EPSG_EQUAL_AREA {
        return Ok(from_equal_area(c.x, c.y));
    }
    if let Some((zone, north)) = parse_utm(epsg) {
        return Ok(from_utm(c.x, c.y, zone, north));
    }
    Err(unsupported(epsg))
}

/// Transform one position between two supported CRSs.
pub fn transform_coord(c: Coord, from: i32, to: i32) -> GeoResult<Coord> {
    if from == to {
        return Ok(c);
    }
    if !is_supported(from) {
        return Err(unsupported(from));
    }
    if !is_supported(to) {
        return Err(unsupported(to));
    }
    // Every supported system is on the WGS 84 datum, so lon/lat is the hub and a
    // transform is at most two projections. Adding a datum shift later means changing
    // this one function, not every pair.
    from_wgs84(to_wgs84(c, from)?, to)
}

/// Transform a whole geometry between two supported CRSs.
///
/// Structure is preserved exactly — a projection moves positions, it does not add or
/// drop them. Long segments are *not* densified: a straight line in one CRS is curved
/// in another, so run `algo::linear::segmentize` first when a segment spans degrees.
pub fn transform(g: &Geometry, from: i32, to: i32) -> GeoResult<Geometry> {
    if from == to {
        return Ok(g.clone());
    }
    if !is_supported(from) {
        return Err(unsupported(from));
    }
    if !is_supported(to) {
        return Err(unsupported(to));
    }
    let mut err: Option<GeoError> = None;
    let out = g.map_coords(&mut |c| match transform_coord(c, from, to) {
        Ok(p) => p,
        Err(e) => {
            err.get_or_insert(e);
            c
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

/// A scale factor: how many metres one unit of `epsg` is, at the given latitude.
///
/// For a projected CRS in metres this is 1 (or the Mercator stretch, which grows without
/// bound toward the poles); for degrees it is the local metres-per-degree. Exposed so a
/// tolerance stated in metres can be converted into the units a planar predicate needs.
pub fn metres_per_unit(epsg: i32, lat: f64) -> GeoResult<f64> {
    if epsg == EPSG_WGS84 {
        // One degree of latitude, which is the smaller of the two and therefore the
        // conservative choice for a tolerance.
        return crate::proj::geodesy::haversine(0.0, lat, 0.0, lat + 1.0);
    }
    if epsg == EPSG_WEB_MERCATOR {
        // Mercator exaggerates by 1/cos(lat); one projected metre is cos(lat) real ones.
        return Ok(lat.to_radians().cos().max(1e-9));
    }
    if epsg == EPSG_EQUAL_AREA || parse_utm(epsg).is_some() {
        return Ok(1.0);
    }
    Err(unsupported(epsg))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(got: f64, want: f64, tol: f64) {
        assert!((got - want).abs() < tol, "{got} vs {want} (tol {tol})");
    }

    #[test]
    fn every_supported_crs_round_trips() {
        let cases = [(-122.4194, 37.7749), (0.0, 0.0), (13.4050, 52.5200), (151.2093, -33.8688)];
        for (lon, lat) in cases {
            let utm = utm_epsg(lon, lat).unwrap();
            for epsg in [EPSG_WEB_MERCATOR, EPSG_EQUAL_AREA, utm] {
                let p = transform_coord(Coord::new(lon, lat), EPSG_WGS84, epsg).unwrap();
                let back = transform_coord(p, epsg, EPSG_WGS84).unwrap();
                close(back.x, lon, 1e-6);
                close(back.y, lat, 1e-6);
            }
        }
    }

    #[test]
    fn utm_zones_and_codes_match_the_convention() {
        assert_eq!(utm_zone(-122.4194).unwrap(), 10);
        assert_eq!(utm_zone(0.0).unwrap(), 31);
        assert_eq!(utm_zone(-180.0).unwrap(), 1);
        assert_eq!(utm_zone(180.0).unwrap(), 60);
        assert_eq!(utm_epsg(-122.4194, 37.7749).unwrap(), 32610);
        assert_eq!(utm_epsg(151.2093, -33.8688).unwrap(), 32756);
    }

    #[test]
    fn utm_easting_is_near_the_false_origin_at_a_zone_centre() {
        // Zone 10N's central meridian is -123.
        let p = transform_coord(Coord::new(-123.0, 37.0), EPSG_WGS84, 32610).unwrap();
        close(p.x, 500_000.0, 0.001);
        assert!(p.y > 4_000_000.0 && p.y < 4_200_000.0, "{}", p.y);
    }

    #[test]
    fn utm_distances_agree_with_the_geodesic_ones_it_is_meant_to_replace() {
        // This is the property UTM exists for, and the one worth pinning: within a
        // zone its planar metric matches the ellipsoid to better than a tenth of a
        // percent, so measuring in projected metres is measuring on the ground.
        let cases = [
            ((8.5417, 47.3777), (8.6417, 47.4777), 32632),
            ((-122.4194, 37.7749), (-122.3194, 37.8749), 32610),
            ((151.2093, -33.8688), (151.3093, -33.7688), 32756),
        ];
        for ((lon1, lat1), (lon2, lat2), epsg) in cases {
            let a = transform_coord(Coord::new(lon1, lat1), EPSG_WGS84, epsg).unwrap();
            let b = transform_coord(Coord::new(lon2, lat2), EPSG_WGS84, epsg).unwrap();
            let planar = ((b.x - a.x).powi(2) + (b.y - a.y).powi(2)).sqrt();
            let geodesic = crate::proj::geodesy::vincenty(lon1, lat1, lon2, lat2).unwrap();
            let err = (planar - geodesic).abs() / geodesic;
            assert!(err < 1e-3, "EPSG:{epsg}: {planar} vs {geodesic} ({err:.2e})");
        }
    }

    #[test]
    fn utm_northing_encodes_the_hemisphere_with_the_false_origin() {
        // Southern-hemisphere zones add 10 000 km so northings stay positive; that
        // offset is the only difference between 326xx and 327xx.
        let north = transform_coord(Coord::new(151.2093, 33.8688), EPSG_WGS84, 32656).unwrap();
        let south = transform_coord(Coord::new(151.2093, -33.8688), EPSG_WGS84, 32756).unwrap();
        close(north.y + south.y, 10_000_000.0, 1.0);
        close(north.x, south.x, 1.0);
    }

    #[test]
    fn web_mercator_matches_the_tile_module() {
        let a = transform_coord(Coord::new(-122.4194, 37.7749), EPSG_WGS84, EPSG_WEB_MERCATOR)
            .unwrap();
        // The tile module uses the mean-radius sphere and this uses the WGS 84
        // semi-major axis; they agree to within that radius ratio, which is 0.13%.
        let b = crate::grid::tile::to_web_mercator(-122.4194, 37.7749).unwrap();
        assert!((a.x / b.x - 1.0).abs() < 0.002, "{} vs {}", a.x, b.x);
    }

    #[test]
    fn equal_area_preserves_area_ratios_where_mercator_does_not() {
        // Two one-degree cells, one at the equator and one at 60N.
        let cell = |lat: f64, epsg: i32| {
            let a = transform_coord(Coord::new(0.0, lat), EPSG_WGS84, epsg).unwrap();
            let b = transform_coord(Coord::new(1.0, lat + 1.0), EPSG_WGS84, epsg).unwrap();
            ((b.x - a.x) * (b.y - a.y)).abs()
        };
        let eq_ratio = cell(0.0, EPSG_EQUAL_AREA) / cell(60.0, EPSG_EQUAL_AREA);
        let true_ratio = {
            let g = |lat: f64| {
                crate::proj::geodesy::ring_area_m2(&[
                    Coord::new(0.0, lat),
                    Coord::new(1.0, lat),
                    Coord::new(1.0, lat + 1.0),
                    Coord::new(0.0, lat + 1.0),
                    Coord::new(0.0, lat),
                ])
            };
            g(0.0) / g(60.0)
        };
        assert!(
            (eq_ratio / true_ratio - 1.0).abs() < 0.02,
            "equal area ratio {eq_ratio} should track the true ratio {true_ratio}"
        );
        let merc_ratio = cell(0.0, EPSG_WEB_MERCATOR) / cell(60.0, EPSG_WEB_MERCATOR);
        assert!(merc_ratio < 0.5, "Mercator inflates the high-latitude cell");
    }

    #[test]
    fn an_unsupported_crs_is_refused_and_the_message_says_what_to_do() {
        let e = transform_coord(Coord::new(0.0, 0.0), EPSG_WGS84, 27700).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("27700") && msg.contains("st_set_srid"), "{msg}");
        assert!(!is_supported(27700));
        assert!(is_supported(32610) && is_supported(4326));
    }

    #[test]
    fn transforming_a_geometry_preserves_its_structure() {
        let g = crate::codec::wkt::read_wkt(
            "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0), (0.2 0.2, 0.4 0.2, 0.4 0.4, 0.2 0.2))",
        )
        .unwrap();
        let out = transform(&g.geometry, EPSG_WGS84, EPSG_WEB_MERCATOR).unwrap();
        assert_eq!(out.num_points(), g.geometry.num_points());
        assert_eq!(out.polygons()[0].interiors.len(), 1);
        assert!(transform(&g.geometry, 4326, 9999).is_err());
    }

    #[test]
    fn metres_per_unit_reports_the_tolerance_conversion() {
        close(metres_per_unit(EPSG_WGS84, 0.0).unwrap(), 111_195.0, 200.0);
        assert_eq!(metres_per_unit(32610, 37.0).unwrap(), 1.0);
        assert!(metres_per_unit(EPSG_WEB_MERCATOR, 60.0).unwrap() < 0.51);
    }
}
