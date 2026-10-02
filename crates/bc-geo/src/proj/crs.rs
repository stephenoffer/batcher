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
//! projection change and needs no datum shift. That is not the same as losing nothing.
//! Web Mercator is undefined at the poles, so a latitude beyond ±85.0511° is clamped to that
//! bound on the way in and does not come back. UTM is meant for its own zone. Its series
//! round-trips to under a millimetre out to 60° of longitude from the central meridian, but
//! the projection's scale error grows with that distance, the series diverges past about 70°
//! near the equator, and at 90° on the equator the projection is singular.
//! Every conversion is also subject to `f64` rounding, which a round trip shows as a change
//! in the last few digits. Reprojecting between datums (NAD 27, OSGB 36) needs a grid shift
//! this module does not have, and their codes are rejected.

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
        return Err(GeoError::domain(format!(
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
        return Err(GeoError::domain(format!(
            "UTM zone needs lat in [-90, 90], got {lat}"
        )));
    }
    let zone = utm_zone(lon)? as i32;
    Ok(if lat >= 0.0 {
        32600 + zone
    } else {
        32700 + zone
    })
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
#[must_use]
pub fn is_supported(epsg: i32) -> bool {
    epsg == EPSG_WGS84
        || epsg == EPSG_WEB_MERCATOR
        || epsg == EPSG_EQUAL_AREA
        || parse_utm(epsg).is_some()
}

fn unsupported(epsg: i32) -> GeoError {
    GeoError::invalid(format!(
        "EPSG:{epsg} is not a supported CRS. Supported: 4326 (WGS 84 lon/lat), \
         3857 (Web Mercator), 6933 (equal area), 32601-32660 and 32701-32760 (UTM). \
         Reproject with a full PROJ-backed tool before loading, or state the data's \
         CRS with st_set_srid if it is already in one of these."
    ))
}

/// UTM's scale on the central meridian.
const UTM_K0: f64 = 0.9996;
/// UTM's false easting, and the false northing of a southern-hemisphere zone, in metres.
const UTM_FALSE_EASTING: f64 = 500_000.0;
const UTM_FALSE_NORTHING_SOUTH: f64 = 10_000_000.0;

/// The constants of Krüger's transverse-Mercator series on WGS 84, to sixth order in the
/// third flattening `n`, as tabulated in Karney (2011), "Transverse Mercator with an
/// accuracy of a few nanometers".
struct Kruger {
    /// The rectifying radius `A`: the meridian arc length is `A` times the rectifying
    /// latitude.
    rect_radius: f64,
    /// Forward coefficients `α₁..α₆`, from the conformal sphere to the projection.
    alpha: [f64; 6],
    /// Inverse coefficients `β₁..β₆`.
    beta: [f64; 6],
}

impl Kruger {
    fn wgs84() -> Self {
        let n = WGS84_F / (2.0 - WGS84_F);
        let n2 = n * n;
        let n3 = n2 * n;
        let n4 = n3 * n;
        let n5 = n4 * n;
        let n6 = n5 * n;
        Self {
            rect_radius: WGS84_A / (1.0 + n) * (1.0 + n2 / 4.0 + n4 / 64.0 + n6 / 256.0),
            alpha: [
                n / 2.0 - 2.0 * n2 / 3.0 + 5.0 * n3 / 16.0 + 41.0 * n4 / 180.0 - 127.0 * n5 / 288.0
                    + 7891.0 * n6 / 37800.0,
                13.0 * n2 / 48.0 - 3.0 * n3 / 5.0 + 557.0 * n4 / 1440.0 + 281.0 * n5 / 630.0
                    - 1_983_433.0 * n6 / 1_935_360.0,
                61.0 * n3 / 240.0 - 103.0 * n4 / 140.0
                    + 15061.0 * n5 / 26880.0
                    + 167_603.0 * n6 / 181_440.0,
                49561.0 * n4 / 161_280.0 - 179.0 * n5 / 168.0 + 6_601_661.0 * n6 / 7_257_600.0,
                34729.0 * n5 / 80640.0 - 3_418_889.0 * n6 / 1_995_840.0,
                212_378_941.0 * n6 / 319_334_400.0,
            ],
            beta: [
                n / 2.0 - 2.0 * n2 / 3.0 + 37.0 * n3 / 96.0 - n4 / 360.0 - 81.0 * n5 / 512.0
                    + 96199.0 * n6 / 604_800.0,
                n2 / 48.0 + n3 / 15.0 - 437.0 * n4 / 1440.0 + 46.0 * n5 / 105.0
                    - 1_118_711.0 * n6 / 3_870_720.0,
                17.0 * n3 / 480.0 - 37.0 * n4 / 840.0 - 209.0 * n5 / 4480.0 + 5569.0 * n6 / 90720.0,
                4397.0 * n4 / 161_280.0 - 11.0 * n5 / 504.0 - 830_251.0 * n6 / 7_257_600.0,
                4583.0 * n5 / 161_280.0 - 108_847.0 * n6 / 3_991_680.0,
                20_648_693.0 * n6 / 638_668_800.0,
            ],
        }
    }
}

/// `τ' = tan χ`, the conformal latitude's tangent, from the geodetic `τ = tan φ`.
///
/// Karney's eq. 7, written with `hypot` so it holds at the poles, where `τ` is huge.
fn conformal_tan(tau: f64) -> f64 {
    let e = E2.sqrt();
    let tau1 = tau.hypot(1.0);
    let sig = (e * (e * tau / tau1).atanh()).sinh();
    tau * sig.hypot(1.0) - sig * tau1
}

/// Invert [`conformal_tan`] by Newton's method, as Karney (2011) does.
fn geodetic_tan(taup: f64) -> f64 {
    let mut tau = taup / (1.0 - E2);
    for _ in 0..10 {
        let tp = conformal_tan(tau);
        let tau1 = tau.hypot(1.0);
        let step =
            (taup - tp) * (1.0 + (1.0 - E2) * tau * tau) / ((1.0 - E2) * tau1 * tp.hypot(1.0));
        tau += step;
        if step.abs() <= 1e-15 * tau.abs().max(1.0) {
            break;
        }
    }
    tau
}

/// The central meridian of a UTM zone, in radians.
fn utm_central_meridian(zone: u32) -> f64 {
    ((f64::from(zone) - 1.0) * 6.0 - 180.0 + 3.0).to_radians()
}

/// Project WGS 84 lon/lat to a UTM zone's easting and northing in metres.
///
/// Krüger's series to sixth order in `n`. It replaced Snyder's expansion in powers of the
/// longitude offset, which is accurate only near the central meridian: 10° from it, in zone
/// 10 at 37°N, Snyder's round trip was off by about 1.5 m, at 30° by about 5 km, and further
/// out it returned latitudes above 90°. Krüger's series agreed with PROJ to a few
/// nanometres at every point measured, out to 40° from the central meridian and 84° of
/// latitude, and `test_diff_geospatial_geodesy.py` holds it to PROJ there. It is a series
/// too, and diverges past about 70° from the central meridian near the equator.
fn to_utm(lon: f64, lat: f64, zone: u32, north: bool) -> Coord {
    let k = Kruger::wgs84();
    let lam = lon.to_radians() - utm_central_meridian(zone);
    let taup = conformal_tan(lat.to_radians().tan());
    let (sin_lam, cos_lam) = lam.sin_cos();
    let xip = taup.atan2(cos_lam);
    let etap = (sin_lam / taup.hypot(cos_lam)).asinh();
    let (mut xi, mut eta) = (xip, etap);
    for (j, a) in k.alpha.iter().enumerate() {
        let m = 2.0 * (j as f64 + 1.0);
        xi += a * (m * xip).sin() * (m * etap).cosh();
        eta += a * (m * xip).cos() * (m * etap).sinh();
    }
    let scale = UTM_K0 * k.rect_radius;
    let false_northing = if north { 0.0 } else { UTM_FALSE_NORTHING_SOUTH };
    Coord::new(UTM_FALSE_EASTING + scale * eta, false_northing + scale * xi)
}

/// Invert `to_utm`.
fn from_utm(easting: f64, northing: f64, zone: u32, north: bool) -> Coord {
    let k = Kruger::wgs84();
    let scale = UTM_K0 * k.rect_radius;
    let false_northing = if north { 0.0 } else { UTM_FALSE_NORTHING_SOUTH };
    let xi = (northing - false_northing) / scale;
    let eta = (easting - UTM_FALSE_EASTING) / scale;
    let (mut xip, mut etap) = (xi, eta);
    for (j, b) in k.beta.iter().enumerate() {
        let m = 2.0 * (j as f64 + 1.0);
        xip -= b * (m * xi).sin() * (m * eta).cosh();
        etap -= b * (m * xi).cos() * (m * eta).sinh();
    }
    let (sinh_etap, cos_xip) = (etap.sinh(), xip.cos());
    let taup = xip.sin() / sinh_etap.hypot(cos_xip);
    let lat = geodetic_tan(taup).atan();
    let lam = sinh_etap.atan2(cos_xip) + utm_central_meridian(zone);
    Coord::new(lam.to_degrees(), lat.to_degrees())
}

/// The standard parallel of EPSG:6933, in radians.
const EASE_STD_PARALLEL: f64 = std::f64::consts::FRAC_PI_6;

/// Project lon/lat to the EPSG:6933 cylindrical equal-area plane.
fn to_equal_area(lon: f64, lat: f64) -> Coord {
    let phi0 = EASE_STD_PARALLEL;
    let k0 = phi0.cos() / (1.0 - E2 * phi0.sin().powi(2)).sqrt();
    let q = authalic_q(lat.to_radians());
    Coord::new(WGS84_A * k0 * lon.to_radians(), WGS84_A * q / (2.0 * k0))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn close(got: f64, want: f64, tol: f64) {
        assert!((got - want).abs() < tol, "{got} vs {want} (tol {tol})");
    }

    #[test]
    fn every_supported_crs_round_trips() {
        let cases = [
            (-122.4194, 37.7749),
            (0.0, 0.0),
            (13.4050, 52.5200),
            (151.2093, -33.8688),
        ];
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

    /// UTM against PROJ, including far outside the zone, where Snyder's series failed.
    ///
    /// The references are PROJ's output through DuckDB's spatial extension
    /// (`ST_Transform(..., always_xy := true)`).
    #[test]
    fn utm_matches_proj_inside_and_far_outside_its_zone() {
        let cases = [
            (
                (-113.0, 37.0),
                32610,
                (1_391_004.039_948_434_3, 4_141_940.724_648_419),
            ),
            (
                (-93.0, 60.0),
                32610,
                (2_132_525.464_391_378, 7_034_625.010_991_832),
            ),
            (
                (151.2093, -33.8688),
                32756,
                (334_368.633_648_097, 6_250_948.345_385_009),
            ),
            (
                (170.0, -80.0),
                32756,
                (825_022.689_869_743_8, 1_070_556.131_080_560_4),
            ),
        ];
        for ((lon, lat), epsg, (want_x, want_y)) in cases {
            let p = transform_coord(Coord::new(lon, lat), EPSG_WGS84, epsg).unwrap();
            close(p.x, want_x, 1e-6);
            close(p.y, want_y, 1e-6);
        }
    }

    /// Far from the central meridian the inverse still returns the input position.
    ///
    /// Out to 60° of longitude, where the series is still good to micrometres. Beyond that
    /// it diverges: measured near the equator, 1.5 mm at 70° and 19 m at 80°.
    #[test]
    fn utm_round_trips_out_to_sixty_degrees_from_its_central_meridian() {
        for dlon in [0.0, 3.0, 10.0, 30.0, 45.0, 60.0] {
            for lat in [-84.0, -37.0, 0.5, 37.0, 60.0, 84.0] {
                let lon = -123.0 + dlon;
                let epsg = if lat >= 0.0 { 32610 } else { 32710 };
                let p = transform_coord(Coord::new(lon, lat), EPSG_WGS84, epsg).unwrap();
                let back = transform_coord(p, epsg, EPSG_WGS84).unwrap();
                close(back.x, lon, 1e-9);
                close(back.y, lat, 1e-9);
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
            let geodesic =
                crate::proj::geodesy::ellipsoidal_distance(lon1, lat1, lon2, lat2).unwrap();
            let err = (planar - geodesic).abs() / geodesic;
            assert!(
                err < 1e-3,
                "EPSG:{epsg}: {planar} vs {geodesic} ({err:.2e})"
            );
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
                .unwrap()
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
        assert!(
            msg.contains("27700") && msg.contains("st_set_srid"),
            "{msg}"
        );
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
}
