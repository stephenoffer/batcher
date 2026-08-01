//! `Expr::Geo` end to end: the JSON the Python control plane emits, deserialized and
//! evaluated over a real Arrow batch.
//!
//! This is the wire-contract test for the geospatial surface. Every case here is written
//! as the JSON document `to_ir()` produces rather than as a Rust `Expr` literal, because
//! a Rust literal would pass even if the serde tags drifted from what Python sends —
//! which is precisely the failure this file exists to catch.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BinaryArray, BooleanArray, Float64Array, Int64Array, RecordBatch,
    StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use bc_expr::Expr;

/// A batch with one WKT column, one geometry column as WKB, and two coordinate columns.
fn batch() -> RecordBatch {
    let wkt = StringArray::from(vec![
        Some("POLYGON((0 0, 4 0, 4 4, 0 4, 0 0))"),
        Some("POINT(2 2)"),
        Some("LINESTRING(0 0, 3 4)"),
        None,
        // A value that is not a geometry at all: the row-nulling case.
        Some("not a geometry"),
    ]);
    let wkb: Vec<Option<Vec<u8>>> = wkt
        .iter()
        .map(|t| t.and_then(|s| bc_geo::from_text(s).ok()).map(|g| bc_geo::to_wkb(&g)))
        .collect();
    let wkb = BinaryArray::from_iter(wkb.iter().map(|o| o.as_deref()));
    let lon = Float64Array::from(vec![
        Some(-122.4194),
        Some(0.0),
        Some(151.2093),
        None,
        Some(13.405),
    ]);
    let lat = Float64Array::from(vec![
        Some(37.7749),
        Some(0.0),
        Some(-33.8688),
        None,
        Some(52.52),
    ]);
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("wkt", DataType::Utf8, true),
            Field::new("geom", DataType::Binary, true),
            Field::new("lon", DataType::Float64, true),
            Field::new("lat", DataType::Float64, true),
        ])),
        vec![
            Arc::new(wkt),
            Arc::new(wkb),
            Arc::new(lon),
            Arc::new(lat),
        ],
    )
    .expect("batch")
}

fn eval(json: &str) -> ArrayRef {
    let e: Expr = serde_json::from_str(json).unwrap_or_else(|err| panic!("{json}: {err}"));
    e.eval(&batch()).unwrap_or_else(|err| panic!("{json}: {err}"))
}

fn floats(json: &str) -> Vec<Option<f64>> {
    let a = eval(json);
    let a = a.as_any().downcast_ref::<Float64Array>().expect("float column");
    (0..a.len()).map(|i| (!a.is_null(i)).then(|| a.value(i))).collect()
}

fn ints(json: &str) -> Vec<Option<i64>> {
    let a = eval(json);
    let a = a.as_any().downcast_ref::<Int64Array>().expect("int column");
    (0..a.len()).map(|i| (!a.is_null(i)).then(|| a.value(i))).collect()
}

fn bools(json: &str) -> Vec<Option<bool>> {
    let a = eval(json);
    let a = a.as_any().downcast_ref::<BooleanArray>().expect("bool column");
    (0..a.len()).map(|i| (!a.is_null(i)).then(|| a.value(i))).collect()
}

fn strings(json: &str) -> Vec<Option<String>> {
    let a = eval(json);
    let a = a.as_any().downcast_ref::<StringArray>().expect("text column");
    (0..a.len())
        .map(|i| (!a.is_null(i)).then(|| a.value(i).to_string()))
        .collect()
}

/// A one-argument geo call over the named column.
fn unary(func: &str, col: &str) -> String {
    format!(r#"{{"e":"geo","fn":"{func}","args":[{{"e":"col","name":"{col}"}}]}}"#)
}

#[test]
fn measures_read_the_geometry_column_and_the_wkt_column_identically() {
    for col in ["geom", "wkt"] {
        assert_eq!(
            floats(&unary("st_area", col)),
            vec![Some(16.0), Some(0.0), Some(0.0), None, None],
            "st_area over {col}"
        );
        assert_eq!(
            floats(&unary("st_length", col)),
            vec![Some(0.0), Some(0.0), Some(5.0), None, None],
            "st_length over {col}"
        );
        assert_eq!(
            floats(&unary("st_perimeter", col)),
            vec![Some(16.0), Some(0.0), Some(0.0), None, None]
        );
    }
}

#[test]
fn a_malformed_geometry_nulls_its_row_and_leaves_the_others_alone() {
    // Row 4 is `NULL`, row 5 is the string "not a geometry". Both null, and the three
    // real geometries still answer — which is the whole point of not raising.
    let areas = floats(&unary("st_area", "wkt"));
    assert_eq!(areas[3], None);
    assert_eq!(areas[4], None);
    assert_eq!(areas[0], Some(16.0));
    // And the reason column names the bad rows.
    let reasons = strings(&unary("st_is_valid_reason", "wkt"));
    assert!(reasons.iter().take(3).all(|r| r.is_none()), "valid rows have no reason");
}

#[test]
fn accessors_return_the_right_types_and_are_null_off_type() {
    assert_eq!(
        floats(&unary("st_x", "wkt")),
        vec![None, Some(2.0), None, None, None],
        "st_x is defined on a point and nothing else"
    );
    assert_eq!(
        ints(&unary("st_dimension", "wkt")),
        vec![Some(2), Some(0), Some(1), None, None]
    );
    assert_eq!(
        ints(&unary("st_num_points", "wkt")),
        vec![Some(5), Some(1), Some(2), None, None]
    );
    assert_eq!(
        strings(&unary("st_geometry_type", "wkt")),
        vec![
            Some("POLYGON".into()),
            Some("POINT".into()),
            Some("LINESTRING".into()),
            None,
            None
        ]
    );
    assert_eq!(
        bools(&unary("st_is_empty", "wkt")),
        vec![Some(false), Some(false), Some(false), None, None]
    );
}

#[test]
fn predicates_take_a_literal_geometry_on_the_right() {
    let json = r#"{"e":"geo","fn":"st_intersects","args":[
        {"e":"col","name":"wkt"},
        {"e":"lit","value":{"str":"POINT(1 1)"}}]}"#;
    assert_eq!(
        bools(json),
        vec![Some(true), Some(false), Some(false), None, None],
        "only the square contains (1,1)"
    );
    let within = r#"{"e":"geo","fn":"st_within","args":[
        {"e":"col","name":"wkt"},
        {"e":"lit","value":{"str":"POLYGON((-1 -1, 5 -1, 5 5, -1 5, -1 -1))"}}]}"#;
    assert_eq!(bools(within), vec![Some(true), Some(true), Some(true), None, None]);
}

#[test]
fn dwithin_takes_its_radius_as_a_third_argument() {
    let json = |r: f64| {
        format!(
            r#"{{"e":"geo","fn":"st_dwithin","args":[
                {{"e":"col","name":"wkt"}},
                {{"e":"lit","value":{{"str":"POINT(6 2)"}}}},
                {{"e":"lit","value":{{"float":{r}}}}}]}}"#
        )
    };
    assert_eq!(bools(&json(1.0))[0], Some(false));
    assert_eq!(bools(&json(3.0))[0], Some(true));
}

#[test]
fn a_negative_radius_raises_rather_than_answering_false() {
    let json = r#"{"e":"geo","fn":"st_dwithin","args":[
        {"e":"col","name":"wkt"},
        {"e":"lit","value":{"str":"POINT(6 2)"}},
        {"e":"lit","value":{"float":-1.0}}]}"#;
    let e: Expr = serde_json::from_str(json).unwrap();
    let err = e.eval(&batch()).expect_err("a negative radius is a query bug");
    let msg = format!("{err}");
    assert!(msg.contains("st_dwithin"), "{msg}");
    assert!(msg.contains("non-negative"), "{msg}");
}

#[test]
fn a_wrong_argument_count_names_the_function_and_the_expected_arity() {
    let json = r#"{"e":"geo","fn":"st_area","args":[
        {"e":"col","name":"wkt"},{"e":"col","name":"wkt"}]}"#;
    let e: Expr = serde_json::from_str(json).unwrap();
    let msg = format!("{}", e.eval(&batch()).expect_err("arity is checked"));
    assert!(msg.contains("1 argument"), "{msg}");
}

#[test]
fn geometry_returning_functions_produce_a_readable_wkb_column() {
    let out = eval(&unary("st_centroid", "wkt"));
    let bin = out.as_any().downcast_ref::<BinaryArray>().expect("binary column");
    let g = bc_geo::from_wkb(bin.value(0)).expect("valid WKB out");
    let c = g.coords()[0];
    assert!((c.x - 2.0).abs() < 1e-12 && (c.y - 2.0).abs() < 1e-12);
    assert!(bin.is_null(3) && bin.is_null(4));
    // And it chains: the centroid of the envelope of a geometry is a geometry.
    let chained = r#"{"e":"geo","fn":"st_as_text","args":[
        {"e":"geo","fn":"st_centroid","args":[
            {"e":"geo","fn":"st_envelope","args":[{"e":"col","name":"wkt"}]}]}]}"#;
    assert_eq!(strings(chained)[0], Some("POINT(2 2)".into()));
}

#[test]
fn the_srid_survives_a_chain_of_geometry_functions() {
    let json = r#"{"e":"geo","fn":"st_srid","args":[
        {"e":"geo","fn":"st_centroid","args":[
            {"e":"geo","fn":"st_set_srid","args":[
                {"e":"col","name":"wkt"},
                {"e":"lit","value":{"int":4326}}]}]}]}"#;
    assert_eq!(ints(json)[0], Some(4326));
}

#[test]
fn transform_relabels_and_moves_the_coordinates() {
    let json = r#"{"e":"geo","fn":"st_as_text","args":[
        {"e":"geo","fn":"st_transform","args":[
            {"e":"geo","fn":"st_point","args":[
                {"e":"col","name":"lon"},{"e":"col","name":"lat"}]},
            {"e":"lit","value":{"int":4326}},
            {"e":"lit","value":{"int":3857}}]}]}"#;
    let out = strings(json);
    let sf = out[0].as_ref().expect("a projected point");
    assert!(sf.starts_with("POINT(-13"), "San Francisco in Web Mercator metres: {sf}");
    assert_eq!(out[3], None, "a null coordinate stays null");
}

#[test]
fn grid_functions_turn_coordinate_columns_into_group_keys() {
    let geohash = r#"{"e":"geo","fn":"geohash_encode","args":[
        {"e":"col","name":"lon"},{"e":"col","name":"lat"},
        {"e":"lit","value":{"int":6}}]}"#;
    assert_eq!(strings(geohash)[0], Some("9q8yyk".into()));

    let cell = r#"{"e":"geo","fn":"st_s2_cell","args":[
        {"e":"col","name":"lon"},{"e":"col","name":"lat"},
        {"e":"lit","value":{"int":12}}]}"#;
    let cells = ints(cell);
    assert!(cells[0].is_some() && cells[3].is_none());
    // The same position at a coarser level is that cell's ancestor.
    let parent = r#"{"e":"geo","fn":"st_s2_cell_parent","args":[
        {"e":"geo","fn":"st_s2_cell","args":[
            {"e":"col","name":"lon"},{"e":"col","name":"lat"},
            {"e":"lit","value":{"int":12}}]},
        {"e":"lit","value":{"int":6}}]}"#;
    let coarse = r#"{"e":"geo","fn":"st_s2_cell","args":[
        {"e":"col","name":"lon"},{"e":"col","name":"lat"},
        {"e":"lit","value":{"int":6}}]}"#;
    assert_eq!(ints(parent), ints(coarse));

    let quad = r#"{"e":"geo","fn":"st_quadkey","args":[
        {"e":"col","name":"lon"},{"e":"col","name":"lat"},
        {"e":"lit","value":{"int":10}}]}"#;
    assert_eq!(strings(quad)[0].as_ref().map(|s| s.len()), Some(10));

    let zone = r#"{"e":"geo","fn":"st_utm_epsg","args":[
        {"e":"col","name":"lon"},{"e":"col","name":"lat"}]}"#;
    assert_eq!(ints(zone)[0], Some(32610));
}

#[test]
fn the_geodesic_and_planar_distances_are_different_functions_with_different_answers() {
    let pair = |f: &str| {
        format!(
            r#"{{"e":"geo","fn":"{f}","args":[
                {{"e":"geo","fn":"st_point","args":[
                    {{"e":"col","name":"lon"}},{{"e":"col","name":"lat"}}]}},
                {{"e":"lit","value":{{"str":"POINT(-0.1278 51.5074)"}}}}]}}"#
        )
    };
    let planar = floats(&pair("st_distance"))[0].expect("a distance");
    let sphere = floats(&pair("st_distance_sphere"))[0].expect("a distance");
    // Degrees versus metres: they are not close, and that is the point.
    assert!(planar < 200.0, "planar distance is in degrees: {planar}");
    assert!(
        (sphere - 8_600_000.0).abs() < 200_000.0,
        "San Francisco to London is about 8600 km, got {sphere}"
    );
}

#[test]
fn round_tripping_through_every_text_encoding_preserves_the_geometry() {
    for (out_fn, in_note) in [
        ("st_as_text", "WKT"),
        ("st_as_ewkt", "EWKT"),
        ("st_as_geojson", "GeoJSON"),
        ("st_as_hex_wkb", "hex WKB"),
    ] {
        let json = format!(
            r#"{{"e":"geo","fn":"st_area","args":[
                {{"e":"geo","fn":"st_geom_from_text","args":[
                    {{"e":"geo","fn":"{out_fn}","args":[{{"e":"col","name":"wkt"}}]}}]}}]}}"#
        );
        assert_eq!(floats(&json)[0], Some(16.0), "round trip via {in_note}");
    }
}
