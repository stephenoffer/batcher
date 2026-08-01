//! GeoJSON — the interchange encoding, and the one a web map speaks.
//!
//! RFC 7946 fixes two things this codec honours and other geometry encodings leave
//! open. Coordinates are longitude-then-latitude in WGS 84, so a GeoJSON geometry has
//! an implied SRID of 4326 whether or not anyone says so; and a `Feature` is a
//! geometry plus properties, which is a *row*, not a geometry — so the reader unwraps
//! one to its geometry and leaves the properties to the IO layer that owns rows.
//!
//! The writer emits the geometry object only. A `FeatureCollection` is assembled by
//! the GeoJSON sink out of a whole batch, because that is where the other columns are.

use std::fmt::Write as _;

use serde_json::Value;

use crate::error::{GeoError, GeoResult};
use crate::types::{Coord, Geometry, LineString, Polygon};
use crate::Geom;

/// The SRID every RFC 7946 document is stated in.
pub const GEOJSON_SRID: i32 = 4326;

fn bad(detail: impl Into<String>) -> GeoError {
    GeoError::parse("GeoJSON", detail)
}

fn coord_of(v: &Value) -> GeoResult<Coord> {
    let a = v.as_array().ok_or_else(|| bad("position must be an array"))?;
    if a.len() < 2 {
        return Err(bad("position needs at least two ordinates"));
    }
    let n = |i: usize| -> GeoResult<f64> {
        a[i].as_f64()
            .ok_or_else(|| bad("ordinate must be a number"))
    };
    Ok(Coord {
        x: n(0)?,
        y: n(1)?,
        z: if a.len() > 2 { n(2)? } else { 0.0 },
    })
}

fn line_of(v: &Value) -> GeoResult<LineString> {
    v.as_array()
        .ok_or_else(|| bad("coordinate list must be an array"))?
        .iter()
        .map(coord_of)
        .collect()
}

fn polygon_of(v: &Value) -> GeoResult<Polygon> {
    let rings: Vec<LineString> = v
        .as_array()
        .ok_or_else(|| bad("polygon coordinates must be an array of rings"))?
        .iter()
        .map(line_of)
        .collect::<GeoResult<_>>()?;
    let mut it = rings.into_iter();
    Ok(Polygon {
        exterior: it.next().unwrap_or_default(),
        interiors: it.collect(),
    })
}

/// True when any position in the value carries a third ordinate.
fn any_z(v: &Value) -> bool {
    match v {
        Value::Array(a) => {
            // A bare position: three numbers means 3D.
            if a.len() >= 3 && a.iter().all(Value::is_number) {
                true
            } else {
                a.iter().any(any_z)
            }
        }
        _ => false,
    }
}

/// Parse a GeoJSON geometry, `Feature`, or single-member `FeatureCollection`.
///
/// A multi-feature collection is rejected rather than silently collapsed: it holds
/// many rows, and folding it into one `GEOMETRYCOLLECTION` would lose the properties
/// that distinguish them. Reading a whole collection is the GeoJSON *source*'s job.
pub fn read_geojson(text: &str) -> GeoResult<Geom> {
    let v: Value = serde_json::from_str(text).map_err(|e| bad(e.to_string()))?;
    read_geojson_value(&v)
}

/// Parse an already-decoded GeoJSON value.
pub fn read_geojson_value(v: &Value) -> GeoResult<Geom> {
    let obj = v.as_object().ok_or_else(|| bad("expected a JSON object"))?;
    let ty = obj
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("object has no \"type\""))?;
    if ty == "Feature" {
        let inner = obj
            .get("geometry")
            .ok_or_else(|| bad("Feature has no \"geometry\""))?;
        if inner.is_null() {
            return Ok(Geom::new(Geometry::GeometryCollection(Vec::new()))
                .with_srid(GEOJSON_SRID));
        }
        return read_geojson_value(inner);
    }
    if ty == "FeatureCollection" {
        let feats = obj
            .get("features")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("FeatureCollection has no \"features\" array"))?;
        return match feats.len() {
            1 => read_geojson_value(&feats[0]),
            n => Err(bad(format!(
                "FeatureCollection holds {n} features, which is {} — read it with the \
                 GeoJSON source, which produces one row per feature",
                if n == 0 { "none" } else { "many rows, not one geometry" }
            ))),
        };
    }
    if ty == "GeometryCollection" {
        let members = obj
            .get("geometries")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("GeometryCollection has no \"geometries\" array"))?;
        let mut has_z = false;
        let mut out = Vec::with_capacity(members.len());
        for m in members {
            let g = read_geojson_value(m)?;
            has_z |= g.has_z;
            out.push(g.geometry);
        }
        return Ok(Geom {
            srid: GEOJSON_SRID,
            has_z,
            geometry: Geometry::GeometryCollection(out),
        });
    }

    let coords = obj
        .get("coordinates")
        .ok_or_else(|| bad(format!("{ty} has no \"coordinates\"")))?;
    let has_z = any_z(coords);
    let geometry = match ty {
        "Point" => Geometry::Point(match coords.as_array() {
            Some(a) if a.is_empty() => None,
            _ => Some(coord_of(coords)?),
        }),
        "LineString" => Geometry::LineString(line_of(coords)?),
        "Polygon" => Geometry::Polygon(polygon_of(coords)?),
        "MultiPoint" => Geometry::MultiPoint(
            coords
                .as_array()
                .ok_or_else(|| bad("MultiPoint coordinates must be an array"))?
                .iter()
                .map(|c| coord_of(c).map(Some))
                .collect::<GeoResult<_>>()?,
        ),
        "MultiLineString" => Geometry::MultiLineString(
            coords
                .as_array()
                .ok_or_else(|| bad("MultiLineString coordinates must be an array"))?
                .iter()
                .map(line_of)
                .collect::<GeoResult<_>>()?,
        ),
        "MultiPolygon" => Geometry::MultiPolygon(
            coords
                .as_array()
                .ok_or_else(|| bad("MultiPolygon coordinates must be an array"))?
                .iter()
                .map(polygon_of)
                .collect::<GeoResult<_>>()?,
        ),
        other => return Err(bad(format!("unknown geometry type {other:?}"))),
    };
    Ok(Geom {
        srid: GEOJSON_SRID,
        has_z,
        geometry,
    })
}

fn num(out: &mut String, v: f64) {
    if v == v.trunc() && v.abs() < 1e15 {
        let _ = write!(out, "{}", v as i64);
    } else {
        let _ = write!(out, "{v}");
    }
}

fn pos(out: &mut String, c: Coord, has_z: bool) {
    out.push('[');
    num(out, c.x);
    out.push(',');
    num(out, c.y);
    if has_z {
        out.push(',');
        num(out, c.z);
    }
    out.push(']');
}

fn pos_list(out: &mut String, cs: &[Coord], has_z: bool) {
    out.push('[');
    for (i, c) in cs.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        pos(out, *c, has_z);
    }
    out.push(']');
}

fn rings(out: &mut String, p: &Polygon, has_z: bool) {
    out.push('[');
    if !p.exterior.is_empty() {
        pos_list(out, &p.exterior, has_z);
        for r in &p.interiors {
            out.push(',');
            pos_list(out, r, has_z);
        }
    }
    out.push(']');
}

fn write_into(out: &mut String, g: &Geometry, has_z: bool) {
    let name = Geom::new(g.clone()).geom_type().name();
    // GeoJSON type names are CamelCase, not the OGC uppercase.
    let camel = match name {
        "POINT" => "Point",
        "LINESTRING" => "LineString",
        "POLYGON" => "Polygon",
        "MULTIPOINT" => "MultiPoint",
        "MULTILINESTRING" => "MultiLineString",
        "MULTIPOLYGON" => "MultiPolygon",
        _ => "GeometryCollection",
    };
    let _ = write!(out, "{{\"type\":\"{camel}\",");
    match g {
        Geometry::GeometryCollection(gs) => {
            out.push_str("\"geometries\":[");
            for (i, child) in gs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_into(out, child, has_z);
            }
            out.push(']');
        }
        _ => {
            out.push_str("\"coordinates\":");
            match g {
                Geometry::Point(p) => match p {
                    // RFC 7946 has no empty-point encoding; `[]` is the convention
                    // GDAL writes and reads back.
                    None => out.push_str("[]"),
                    Some(c) => pos(out, *c, has_z),
                },
                Geometry::LineString(l) => pos_list(out, l, has_z),
                Geometry::Polygon(p) => rings(out, p, has_z),
                Geometry::MultiPoint(ps) => {
                    out.push('[');
                    for (i, p) in ps.iter().flatten().enumerate() {
                        if i > 0 {
                            out.push(',');
                        }
                        pos(out, *p, has_z);
                    }
                    out.push(']');
                }
                Geometry::MultiLineString(ls) => {
                    out.push('[');
                    for (i, l) in ls.iter().enumerate() {
                        if i > 0 {
                            out.push(',');
                        }
                        pos_list(out, l, has_z);
                    }
                    out.push(']');
                }
                Geometry::MultiPolygon(ps) => {
                    out.push('[');
                    for (i, p) in ps.iter().enumerate() {
                        if i > 0 {
                            out.push(',');
                        }
                        rings(out, p, has_z);
                    }
                    out.push(']');
                }
                Geometry::GeometryCollection(_) => unreachable!("handled above"),
            }
        }
    }
    out.push('}');
}

/// Render an RFC 7946 geometry object.
pub fn write_geojson(g: &Geom) -> String {
    let mut out = String::with_capacity(24 + 24 * g.num_points());
    write_into(&mut out, &g.geometry, g.has_z);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(text: &str) -> String {
        write_geojson(&read_geojson(text).expect(text))
    }

    #[test]
    fn every_type_round_trips() {
        for t in [
            r#"{"type":"Point","coordinates":[1,2]}"#,
            r#"{"type":"LineString","coordinates":[[0,0],[1,1]]}"#,
            r#"{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]]}"#,
            r#"{"type":"MultiPoint","coordinates":[[1,2],[3,4]]}"#,
            r#"{"type":"MultiLineString","coordinates":[[[0,0],[1,1]]]}"#,
            r#"{"type":"MultiPolygon","coordinates":[[[[0,0],[1,0],[1,1],[0,0]]]]}"#,
            r#"{"type":"GeometryCollection","geometries":[{"type":"Point","coordinates":[1,2]}]}"#,
        ] {
            assert_eq!(rt(t), t);
        }
    }

    #[test]
    fn geojson_implies_wgs84() {
        assert_eq!(
            read_geojson(r#"{"type":"Point","coordinates":[1,2]}"#).unwrap().srid,
            4326
        );
    }

    #[test]
    fn a_feature_unwraps_to_its_geometry() {
        let g = read_geojson(
            r#"{"type":"Feature","properties":{"n":1},"geometry":{"type":"Point","coordinates":[5,6]}}"#,
        )
        .unwrap();
        assert_eq!(g.geometry, Geometry::Point(Some(Coord::new(5.0, 6.0))));
    }

    #[test]
    fn a_multi_feature_collection_is_refused_with_a_pointer_to_the_source() {
        let err = read_geojson(
            r#"{"type":"FeatureCollection","features":[
                 {"type":"Feature","geometry":{"type":"Point","coordinates":[0,0]}},
                 {"type":"Feature","geometry":{"type":"Point","coordinates":[1,1]}}]}"#,
        )
        .unwrap_err();
        assert!(format!("{err}").contains("GeoJSON source"), "{err}");
    }

    #[test]
    fn three_dimensional_positions_are_detected() {
        let g = read_geojson(r#"{"type":"LineString","coordinates":[[0,0,1],[1,1,2]]}"#).unwrap();
        assert!(g.has_z);
        assert_eq!(write_geojson(&g), r#"{"type":"LineString","coordinates":[[0,0,1],[1,1,2]]}"#);
    }

    #[test]
    fn malformed_documents_are_rejected() {
        assert!(read_geojson("not json").is_err());
        assert!(read_geojson(r#"{"type":"Point"}"#).is_err());
        assert!(read_geojson(r#"{"type":"Point","coordinates":[1]}"#).is_err());
        assert!(read_geojson(r#"{"type":"Sphere","coordinates":[1,2]}"#).is_err());
    }
}
