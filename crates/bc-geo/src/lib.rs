//! `bc-geo` — the geometry data plane: codecs, planar algorithms, grids, projections.
//!
//! Batcher stores a geometry column as **WKB in an Arrow `Binary` column**, not as a
//! bespoke type. That single decision is what lets geospatial work exist here at all
//! without breaking the Arrow-is-the-only-columnar-contract invariant: every operator,
//! every spill path, every shuffle and the whole FFI boundary already handle `Binary`,
//! so a geometry column moves through the engine with no new machinery. It is also the
//! same representation GeoParquet, PostGIS, GeoPackage and DuckDB spatial use, so a
//! column round-trips through any of them without conversion.
//!
//! This crate is a **near-leaf**: it depends on nothing in the workspace and pulls in no
//! third-party geometry stack. It sits below `bc-expr`, which calls it per row to
//! evaluate the `Expr::Geo` variant. Nothing here knows about Arrow, and that is on
//! purpose — the array-level plumbing belongs in `bc-expr`, so this crate stays a
//! testable library of pure geometry functions.
//!
//! # What lives where
//!
//! | Module | Answers |
//! |---|---|
//! | `types` | the geometry model itself, plus bounding boxes |
//! | `codec` | WKB / WKT / GeoJSON, in and out |
//! | `algo` | predicates, measurements, constructions, validity — all planar |
//! | `grid` | geohash, map tiles, S2 cells, hex bins — position to group key |
//! | `proj` | metres on the globe, and moving between reference systems |
//!
//! # Planar versus geodesic, stated once
//!
//! Everything in `algo` is Cartesian and answers in the coordinate system's own units.
//! On EPSG:4326 those units are degrees, and a degree is not a distance. That is not an
//! oversight — it is what PostGIS's `geometry` type does, and it is what makes spatial
//! joins affordable, because the planar metric is the one a bounding box can bound.
//! When the answer has to be metres on the Earth, the function is in `proj::geodesy`
//! and is named for it. The two are never silently interchanged.

pub mod algo;
pub mod codec;
pub mod error;
pub mod grid;
pub mod proj;
pub mod types;

pub use error::{GeoError, GeoResult};
pub use types::{Bbox, Coord, Geom, GeomType, Geometry, LineString, Polygon};

/// Parse a geometry from WKB, the storage encoding.
pub fn from_wkb(bytes: &[u8]) -> GeoResult<Geom> {
    codec::wkb::read_wkb(bytes)
}

/// Serialize a geometry to WKB, the storage encoding.
pub fn to_wkb(g: &Geom) -> Vec<u8> {
    codec::wkb::write_wkb(g)
}

/// Parse a geometry from any of the three text encodings, chosen by inspecting the
/// input.
///
/// A single entry point exists because the distinction is invisible at the call site: a
/// user pasting a geometry into a filter has a string, and whether it came out of a
/// PostGIS client (`SRID=4326;POINT(...)`), a GDAL dump (`POINT (...)`), a hex `bytea`
/// (`0101000000...`) or a web API (`{"type":"Point",...}`) is not something they should
/// have to declare. Detection is by first non-space character and is unambiguous: JSON
/// starts with `{`, hex WKB is all hex digits, everything else is WKT.
pub fn from_text(text: &str) -> GeoResult<Geom> {
    let t = text.trim_start();
    if t.starts_with('{') {
        return codec::geojson::read_geojson(text);
    }
    let body = t
        .strip_prefix("0x")
        .or_else(|| t.strip_prefix("0X"))
        .unwrap_or(t);
    if !body.is_empty() && body.len() % 2 == 0 && body.bytes().all(|c| c.is_ascii_hexdigit()) {
        return codec::wkb::read_hex_wkb(t);
    }
    codec::wkt::read_wkt(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_text_detects_each_encoding() {
        let point = Geometry::Point(Some(Coord::new(1.0, 2.0)));
        assert_eq!(from_text("POINT(1 2)").unwrap().geometry, point);
        assert_eq!(
            from_text(r#"{"type":"Point","coordinates":[1,2]}"#)
                .unwrap()
                .geometry,
            point
        );
        let hex = codec::wkb::write_hex_wkb(&Geom::new(point.clone()));
        assert_eq!(from_text(&hex).unwrap().geometry, point);
        assert_eq!(from_text(&format!("0x{hex}")).unwrap().geometry, point);
        assert_eq!(from_text("  SRID=4326;POINT(1 2)").unwrap().srid, 4326);
    }

    #[test]
    fn a_geometry_survives_every_encoding_round_trip() {
        let cases = [
            "POINT(1 2)",
            "LINESTRING(0 0, 1 1, 2 4)",
            "POLYGON((0 0, 4 0, 4 4, 0 4, 0 0), (1 1, 2 1, 2 2, 1 1))",
            "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 0)), ((5 5, 6 5, 6 6, 5 5)))",
            "GEOMETRYCOLLECTION(POINT(1 2), POLYGON((0 0, 1 0, 1 1, 0 0)))",
        ];
        for t in cases {
            let g = codec::wkt::read_wkt(t).unwrap();
            assert_eq!(
                from_wkb(&to_wkb(&g)).unwrap().geometry,
                g.geometry,
                "WKB: {t}"
            );
            assert_eq!(
                codec::wkt::read_wkt(&codec::wkt::write_wkt(&g))
                    .unwrap()
                    .geometry,
                g.geometry,
                "WKT: {t}"
            );
            assert_eq!(
                codec::geojson::read_geojson(&codec::geojson::write_geojson(&g))
                    .unwrap()
                    .geometry,
                g.geometry,
                "GeoJSON: {t}"
            );
        }
    }

    #[test]
    fn measurements_are_invariant_under_the_encoding_used() {
        let g = codec::wkt::read_wkt("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))").unwrap();
        let via_wkb = from_wkb(&to_wkb(&g)).unwrap();
        let via_json = codec::geojson::read_geojson(&codec::geojson::write_geojson(&g)).unwrap();
        assert_eq!(algo::measure::area(&g.geometry), 16.0);
        assert_eq!(algo::measure::area(&via_wkb.geometry), 16.0);
        assert_eq!(algo::measure::area(&via_json.geometry), 16.0);
    }
}
