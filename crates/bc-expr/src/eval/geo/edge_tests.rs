//! Row-local failure versus caller error, and the edge semantics fixed alongside it.
//!
//! Each test builds the expression the way the Python control plane does and evaluates
//! it over a real batch, so it checks the whole array-level path: that one bad row nulls
//! only itself, that a bad *parameter* still raises and names the value the caller
//! passed, and that the accessors whose edge cases were wrong now answer as PostGIS and
//! DuckDB do.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BinaryArray, Float64Array, Int64Array, RecordBatch};
use arrow::datatypes::{Field, Schema};

use crate::{Expr, GeoFunc, Literal};

use super::eval_geo;

fn batch(cols: Vec<(&str, ArrayRef)>) -> RecordBatch {
    let fields: Vec<Field> = cols
        .iter()
        .map(|(n, a)| Field::new(*n, a.data_type().clone(), true))
        .collect();
    let arrays: Vec<ArrayRef> = cols.into_iter().map(|(_, a)| a).collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

fn col(name: &str) -> Expr {
    Expr::Col { name: name.into() }
}

fn int(v: i64) -> Expr {
    Expr::Lit {
        value: Literal::Int(v),
    }
}

fn float(v: f64) -> Expr {
    Expr::Lit {
        value: Literal::Float(v),
    }
}

fn text(v: &str) -> Expr {
    Expr::Lit {
        value: Literal::Str(v.into()),
    }
}

/// Five positions, two of them off the globe: NaN longitude and longitude 200.
fn positions() -> RecordBatch {
    batch(vec![
        (
            "lon",
            Arc::new(Float64Array::from(vec![
                Some(13.4),
                Some(f64::NAN),
                Some(200.0),
                None,
                Some(-122.4194),
            ])) as ArrayRef,
        ),
        (
            "lat",
            Arc::new(Float64Array::from(vec![
                Some(52.5),
                Some(0.0),
                Some(0.0),
                Some(1.0),
                Some(37.7749),
            ])) as ArrayRef,
        ),
    ])
}

fn one_row() -> RecordBatch {
    batch(vec![(
        "k",
        Arc::new(Float64Array::from(vec![0.0])) as ArrayRef,
    )])
}

fn nulls(a: &ArrayRef) -> Vec<bool> {
    (0..a.len()).map(|i| a.is_null(i)).collect()
}

#[test]
fn one_off_globe_position_nulls_its_row_instead_of_failing_the_column() {
    let b = positions();
    let lonlat = || vec![col("lon"), col("lat")];
    let with = |mut a: Vec<Expr>, extra: Expr| {
        a.push(extra);
        a
    };
    let point = Expr::Geo {
        func: GeoFunc::StPoint,
        args: lonlat(),
    };
    let cases: Vec<(GeoFunc, Vec<Expr>)> = vec![
        (GeoFunc::GeohashEncode, with(lonlat(), int(9))),
        (GeoFunc::StGeohash, vec![point, int(9)]),
        (GeoFunc::StQuadkey, with(lonlat(), int(10))),
        (GeoFunc::StTileX, with(lonlat(), int(10))),
        (GeoFunc::StTileY, with(lonlat(), int(10))),
        (GeoFunc::StS2Cell, with(lonlat(), int(10))),
        (GeoFunc::StUtmEpsg, lonlat()),
        (GeoFunc::StUtmZone, vec![col("lon")]),
    ];
    for (func, args) in cases {
        let out = eval_geo(func, &args, &b).unwrap_or_else(|e| panic!("{func:?}: {e}"));
        let n = nulls(&out);
        assert_eq!(
            n,
            vec![false, true, true, true, false],
            "{func:?}: only the NaN, lon=200 and null rows should be null"
        );
    }
}

#[test]
fn a_constant_parameter_out_of_range_raises_naming_the_value_passed() {
    let b = positions();
    for (func, bad) in [
        (GeoFunc::GeohashEncode, -1),
        (GeoFunc::GeohashEncode, 13),
        (GeoFunc::StQuadkey, -1),
        (GeoFunc::StTileX, 31),
        (GeoFunc::StS2Cell, -1),
    ] {
        let err = eval_geo(func, &[col("lon"), col("lat"), int(bad)], &b).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(&format!("got {bad}")), "{func:?}: {msg}");
    }
    let err = eval_geo(
        GeoFunc::StS2CellParent,
        &[int(1 << 40), int(-1)],
        &one_row(),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("got -1"), "{err}");
}

fn wkt(func: GeoFunc, args: Vec<Expr>) -> Option<String> {
    let out = eval_geo(func, &args, &one_row()).unwrap();
    let bytes = out.as_any().downcast_ref::<BinaryArray>().unwrap();
    (!bytes.is_null(0))
        .then(|| bc_geo::codec::wkt::write_wkt(&bc_geo::from_wkb(bytes.value(0)).unwrap()))
}

fn float_of(func: GeoFunc, args: Vec<Expr>) -> Option<f64> {
    let out = eval_geo(func, &args, &one_row()).unwrap();
    let a = out.as_any().downcast_ref::<Float64Array>().unwrap();
    (!a.is_null(0)).then(|| a.value(0))
}

#[test]
fn interpolating_along_an_empty_line_is_null() {
    assert_eq!(
        wkt(
            GeoFunc::StLineInterpolatePoint,
            vec![text("LINESTRING EMPTY"), float(0.5)]
        ),
        None
    );
}

#[test]
fn ring_accessors_are_null_outside_their_domain() {
    let poly = "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0), (2 2, 2 4, 4 4, 4 2, 2 2))";
    let ring = |n: i64| wkt(GeoFunc::StInteriorRingN, vec![text(poly), int(n)]);
    assert_eq!(ring(0), None);
    assert_eq!(ring(-3), None);
    assert_eq!(ring(2), None);
    assert_eq!(
        ring(1).as_deref(),
        Some("LINESTRING(2 2, 2 4, 4 4, 4 2, 2 2)")
    );
    let multi = "MULTIPOLYGON(((0 0, 1 0, 1 1, 0 1, 0 0)), ((5 5, 6 5, 6 6, 5 6, 5 5)))";
    assert_eq!(wkt(GeoFunc::StExteriorRing, vec![text(multi)]), None);
}

#[test]
fn z_is_kept_where_it_means_something_and_dropped_where_it_does_not() {
    assert_eq!(
        wkt(GeoFunc::StCentroid, vec![text("POINT Z (1 2 3)")]).as_deref(),
        Some("POINT Z(1 2 3)")
    );
    assert_eq!(
        wkt(
            GeoFunc::StCentroid,
            vec![text("LINESTRING Z (0 0 0, 1 1 1)")]
        )
        .as_deref(),
        Some("POINT Z(0.5 0.5 0.5)")
    );
    // A bounding box is 2D, as in PostGIS and DuckDB.
    assert_eq!(
        wkt(GeoFunc::StEnvelope, vec![text("POINT Z (1 2 3)")]).as_deref(),
        Some("POINT(1 2)")
    );
    // Force3D adds z to a 2D geometry and leaves an existing z alone.
    assert_eq!(
        wkt(
            GeoFunc::StForce3d,
            vec![text("POINT Z (0 0 1)"), float(5.0)]
        )
        .as_deref(),
        Some("POINT Z(0 0 1)")
    );
    assert_eq!(
        wkt(GeoFunc::StForce3d, vec![text("POINT(0 0)"), float(5.0)]).as_deref(),
        Some("POINT Z(0 0 5)")
    );
}

#[test]
fn the_spheroid_family_answers_antipodes_and_the_antimeridian() {
    let d = float_of(
        GeoFunc::StDistanceSpheroid,
        vec![text("POINT(0 0)"), text("POINT(180 0)")],
    )
    .expect("an antipodal pair has a distance");
    assert!((d - 20_003_931.458_625_447).abs() < 1e-6, "{d}");
    let a = float_of(
        GeoFunc::StAreaSpheroid,
        vec![text(
            "POLYGON((170 -10, -170 -10, -170 10, 170 10, 170 -10))",
        )],
    )
    .unwrap();
    // DuckDB ST_Area_Spheroid on the same ring: 4948480469169.516.
    assert!((a / 4_948_480_469_169.516 - 1.0).abs() < 1e-9, "{a}");
    // An off-globe coordinate is null for this row, not an error for the column.
    assert_eq!(
        float_of(
            GeoFunc::StLengthSpheroid,
            vec![text("LINESTRING(0 0, 200 0)")]
        ),
        None
    );
}

#[test]
fn buffers_are_unions_of_their_parts() {
    let out = eval_geo(
        GeoFunc::StBuffer,
        &[text("MULTIPOINT((0 0), (10 0))"), float(1.0), int(8)],
        &one_row(),
    )
    .unwrap();
    let bytes = out.as_any().downcast_ref::<BinaryArray>().unwrap();
    let g = bc_geo::from_wkb(bytes.value(0)).unwrap();
    let area = bc_geo::algo::measure::area(&g.geometry);
    // Two 32-gon discs, not the hull of both (which was 23.1).
    assert!((area - 2.0 * 3.121_445_152_258_052).abs() < 1e-9, "{area}");
}

#[test]
fn an_empty_simple_geometry_has_no_members() {
    let out = eval_geo(GeoFunc::StNumGeometries, &[text("POINT EMPTY")], &one_row()).unwrap();
    assert_eq!(
        out.as_any().downcast_ref::<Int64Array>().unwrap().value(0),
        0
    );
    let out = eval_geo(
        GeoFunc::StIsSimple,
        &[text("POLYGON((0 0, 2 2, 2 0, 0 2, 0 0))")],
        &one_row(),
    )
    .unwrap();
    assert!(!out
        .as_any()
        .downcast_ref::<arrow::array::BooleanArray>()
        .unwrap()
        .value(0));
}
