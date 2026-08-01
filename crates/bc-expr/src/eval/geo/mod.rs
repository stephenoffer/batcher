//! Evaluation of the `Expr::Geo` variant — the array-level half of geospatial support.
//!
//! `bc-geo` knows geometry and knows nothing about Arrow; this module is the seam. It
//! evaluates each argument to a column, walks the rows, decodes WKB, calls into
//! `bc_geo`, and builds a typed output array. Everything Arrow-shaped lives here and
//! everything geometry-shaped lives there, which is what keeps the geometry algorithms
//! unit-testable without a `RecordBatch`.
//!
//! # Null semantics, stated once
//!
//! A geometry argument yields null when the row is null **or when the bytes are not a
//! geometry**. That is a deliberate choice and it differs from PostGIS, which raises.
//! In a columnar engine one corrupt row in a hundred million would otherwise abort a
//! scan that is 99.999999% fine, and the corruption is usually in data nobody controls.
//! Nulling keeps it findable — `WHERE st_is_valid_reason(g) IS NOT NULL` names every bad
//! row and why — instead of turning a data-quality question into an outage.
//!
//! A *caller* error is different and does raise: a negative buffer radius, a grid
//! precision out of range, an unsupported EPSG code. Those are properties of the query,
//! not of a row, so nulling them would hide a bug on every row at once. `bc_geo`'s
//! `GeoError::is_row_local` draws exactly that line.
//!
//! # Why a geometry argument accepts text
//!
//! A geometry column is `Binary` holding WKB, but `st_area('POLYGON((...))')` and
//! `st_area(wkt_column)` both work: a `Utf8` argument is parsed with `bc_geo::from_text`,
//! which detects WKT, EWKT, GeoJSON and hex WKB by content. Requiring an explicit
//! `st_geom_from_text` around every literal would be a wart with no upside, and the
//! detection is unambiguous.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryBuilder, BooleanBuilder, Float64Builder, Int64Builder,
    LargeBinaryArray, LargeStringArray, RecordBatch, StringArray, StringBuilder,
};
use arrow::datatypes::DataType;

use bc_geo::{GeoError, Geom};

use crate::{Expr, ExprError, GeoFunc};

mod build;
mod grid;
mod scalar;

/// Evaluate a geospatial function over a batch.
pub fn eval_geo(func: GeoFunc, args: &[Expr], batch: &RecordBatch) -> Result<ArrayRef, ExprError> {
    if args.len() != func.arity() {
        return Err(ExprError::InvalidArgument {
            func: format!("{func:?}"),
            reason: format!("expects {} argument(s), got {}", func.arity(), args.len()),
        });
    }
    let cols: Vec<ArrayRef> = args
        .iter()
        .map(|a| a.eval(batch))
        .collect::<Result<_, _>>()?;
    let rows = batch.num_rows();
    if grid::handles(func) {
        return grid::eval(func, &cols, rows);
    }
    if build::handles(func) {
        return build::eval(func, &cols, rows);
    }
    scalar::eval(func, &cols, rows)
}

/// The name a user typed, for error messages. `GeoFunc`'s `Debug` is the Rust variant
/// (`StAsText`); users wrote `st_as_text`, and an error naming the wrong one sends them
/// searching for a function that does not exist.
pub(crate) fn fn_name(func: GeoFunc) -> String {
    let debug = format!("{func:?}");
    let mut out = String::with_capacity(debug.len() + 6);
    for (i, c) in debug.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            out.push('_');
        }
        out.extend(c.to_lowercase());
    }
    out
}

/// A caller error, raised rather than nulled.
pub(crate) fn caller_error(func: GeoFunc, e: GeoError) -> ExprError {
    ExprError::InvalidArgument {
        func: fn_name(func),
        reason: e.to_string(),
    }
}

/// Decode row `i` of a geometry column.
///
/// `Ok(None)` covers both "the row is null" and "the bytes are not a geometry"; see the
/// module documentation for why those share an outcome.
pub(crate) fn geom_at(col: &ArrayRef, i: usize, func: GeoFunc) -> Result<Option<Geom>, ExprError> {
    if col.is_null(i) {
        return Ok(None);
    }
    let parsed = match col.data_type() {
        DataType::Binary => {
            let a = col
                .as_any()
                .downcast_ref::<BinaryArray>()
                .expect("checked data type");
            bc_geo::from_wkb(a.value(i))
        }
        DataType::LargeBinary => {
            let a = col
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .expect("checked data type");
            bc_geo::from_wkb(a.value(i))
        }
        DataType::Utf8 => {
            let a = col
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("checked data type");
            bc_geo::from_text(a.value(i))
        }
        DataType::LargeUtf8 => {
            let a = col
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("checked data type");
            bc_geo::from_text(a.value(i))
        }
        other => {
            return Err(ExprError::ExpectedType {
                func: fn_name(func),
                want: "a geometry column (Binary WKB, or Utf8 WKT/GeoJSON/hex)",
                got: other.to_string(),
            })
        }
    };
    match parsed {
        Ok(g) => Ok(Some(g)),
        Err(e) if e.is_row_local() => Ok(None),
        Err(e) => Err(caller_error(func, e)),
    }
}

/// Read row `i` of a numeric column as `f64`.
pub(crate) fn f64_at(col: &ArrayRef, i: usize, func: GeoFunc) -> Result<Option<f64>, ExprError> {
    if col.is_null(i) {
        return Ok(None);
    }
    use arrow::array::{Float64Array, Int64Array};
    match col.data_type() {
        DataType::Float64 => Ok(Some(
            col.as_any()
                .downcast_ref::<Float64Array>()
                .expect("checked data type")
                .value(i),
        )),
        DataType::Int64 => Ok(Some(
            col.as_any()
                .downcast_ref::<Int64Array>()
                .expect("checked data type")
                .value(i) as f64,
        )),
        other => Err(ExprError::ExpectedType {
            func: fn_name(func),
            want: "a numeric argument",
            got: other.to_string(),
        }),
    }
}

/// Read row `i` of an integer column as `i64`.
pub(crate) fn i64_at(col: &ArrayRef, i: usize, func: GeoFunc) -> Result<Option<i64>, ExprError> {
    if col.is_null(i) {
        return Ok(None);
    }
    use arrow::array::{Float64Array, Int64Array};
    match col.data_type() {
        DataType::Int64 => Ok(Some(
            col.as_any()
                .downcast_ref::<Int64Array>()
                .expect("checked data type")
                .value(i),
        )),
        // A float literal reaching an integer slot is normal: `st_s2_cell(x, y, 10)`
        // parses `10` as an integer, but `10.0` from a computed column is equally
        // meaningful. Only a value that is not integral is a mistake.
        DataType::Float64 => {
            let v = col
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("checked data type")
                .value(i);
            if v.fract() != 0.0 || !v.is_finite() {
                return Err(ExprError::InvalidArgument {
                    func: fn_name(func),
                    reason: format!("expected a whole number, got {v}"),
                });
            }
            Ok(Some(v as i64))
        }
        other => Err(ExprError::ExpectedType {
            func: fn_name(func),
            want: "an integer argument",
            got: other.to_string(),
        }),
    }
}

/// Read row `i` of a text column.
pub(crate) fn str_at(col: &ArrayRef, i: usize, func: GeoFunc) -> Result<Option<&str>, ExprError> {
    if col.is_null(i) {
        return Ok(None);
    }
    match col.data_type() {
        DataType::Utf8 => Ok(Some(
            col.as_any()
                .downcast_ref::<StringArray>()
                .expect("checked data type")
                .value(i),
        )),
        DataType::LargeUtf8 => Ok(Some(
            col.as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("checked data type")
                .value(i),
        )),
        other => Err(ExprError::ExpectedType {
            func: fn_name(func),
            want: "a text argument",
            got: other.to_string(),
        }),
    }
}

/// Accumulates a geometry-valued output column as WKB.
pub(crate) struct GeomOut(BinaryBuilder);

impl GeomOut {
    pub(crate) fn with_capacity(rows: usize) -> Self {
        // 48 bytes is roughly a 2D point with its header; the builder grows from there.
        GeomOut(BinaryBuilder::with_capacity(rows, rows * 48))
    }

    /// Append a geometry, or a null when it is absent.
    pub(crate) fn push(&mut self, g: Option<&Geom>) {
        match g {
            // EWKB rather than plain WKB so an SRID set upstream survives into the next
            // operator. A geometry that forgot its CRS midway through a pipeline is the
            // single most common way a reprojection silently becomes a no-op.
            Some(g) => self.0.append_value(bc_geo::codec::wkb::write_ewkb(g)),
            None => self.0.append_null(),
        }
    }

    pub(crate) fn finish(mut self) -> ArrayRef {
        Arc::new(self.0.finish())
    }
}

/// A row-local `bc_geo` result: `Invalid` raises, everything else nulls.
pub(crate) fn row_result<T>(r: Result<T, GeoError>, func: GeoFunc) -> Result<Option<T>, ExprError> {
    match r {
        Ok(v) => Ok(Some(v)),
        Err(e) if e.is_row_local() => Ok(None),
        Err(e) => Err(caller_error(func, e)),
    }
}

/// Builders for the four non-geometry output types, so the dispatchers below can hold
/// one value instead of branching on the output type at every append.
pub(crate) enum ScalarOut {
    /// A Float64 column.
    Float(Float64Builder),
    /// An Int64 column.
    Int(Int64Builder),
    /// A Boolean column.
    Bool(BooleanBuilder),
    /// A Utf8 column.
    Text(StringBuilder),
}

impl ScalarOut {
    pub(crate) fn push_f64(&mut self, v: Option<f64>) {
        match self {
            ScalarOut::Float(b) => b.append_option(v),
            _ => unreachable!("push_f64 on a non-float output"),
        }
    }

    pub(crate) fn push_i64(&mut self, v: Option<i64>) {
        match self {
            ScalarOut::Int(b) => b.append_option(v),
            _ => unreachable!("push_i64 on a non-integer output"),
        }
    }

    pub(crate) fn push_bool(&mut self, v: Option<bool>) {
        match self {
            ScalarOut::Bool(b) => b.append_option(v),
            _ => unreachable!("push_bool on a non-boolean output"),
        }
    }

    pub(crate) fn push_str(&mut self, v: Option<&str>) {
        match self {
            ScalarOut::Text(b) => b.append_option(v),
            _ => unreachable!("push_str on a non-text output"),
        }
    }

    pub(crate) fn finish(self) -> ArrayRef {
        match self {
            ScalarOut::Float(mut b) => Arc::new(b.finish()),
            ScalarOut::Int(mut b) => Arc::new(b.finish()),
            ScalarOut::Bool(mut b) => Arc::new(b.finish()),
            ScalarOut::Text(mut b) => Arc::new(b.finish()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_names_render_as_the_user_spelled_them() {
        assert_eq!(fn_name(GeoFunc::StAsText), "st_as_text");
        assert_eq!(fn_name(GeoFunc::StDwithin), "st_dwithin");
        assert_eq!(fn_name(GeoFunc::GeohashEncode), "geohash_encode");
        assert_eq!(fn_name(GeoFunc::StS2CellParent), "st_s2_cell_parent");
    }

    #[test]
    fn every_function_is_routed_by_exactly_one_dispatcher() {
        // Serde gives no enumeration of a fieldless enum, so the list is spelled out
        // here; a new variant that nobody routes shows up as a compile error in the
        // `scalar` fallback's exhaustive match rather than as a silent wrong answer.
        for f in ALL {
            let n = usize::from(grid::handles(f)) + usize::from(build::handles(f));
            assert!(n <= 1, "{f:?} claimed by two dispatchers");
            assert!(f.arity() >= 1, "{f:?} has no arguments");
        }
    }

    #[test]
    fn geometry_returning_functions_agree_with_the_build_dispatcher() {
        for f in ALL {
            if f.returns_geometry() {
                assert!(
                    build::handles(f) || grid::handles(f),
                    "{f:?} returns a geometry but is not built by a geometry dispatcher"
                );
            }
        }
    }

    /// Every `GeoFunc`, for the routing tests above.
    pub(super) const ALL: [GeoFunc; 113] = [
        GeoFunc::StPoint,
        GeoFunc::StPointZ,
        GeoFunc::StMakeLine,
        GeoFunc::StMakePolygon,
        GeoFunc::StMakeEnvelope,
        GeoFunc::StGeomFromText,
        GeoFunc::StGeomFromWkb,
        GeoFunc::StGeomFromGeojson,
        GeoFunc::StGeomFromGeohash,
        GeoFunc::StAsText,
        GeoFunc::StAsEwkt,
        GeoFunc::StAsBinary,
        GeoFunc::StAsEwkb,
        GeoFunc::StAsHexWkb,
        GeoFunc::StAsGeojson,
        GeoFunc::StX,
        GeoFunc::StY,
        GeoFunc::StZ,
        GeoFunc::StXmin,
        GeoFunc::StYmin,
        GeoFunc::StXmax,
        GeoFunc::StYmax,
        GeoFunc::StGeometryType,
        GeoFunc::StDimension,
        GeoFunc::StSrid,
        GeoFunc::StSetSrid,
        GeoFunc::StNumPoints,
        GeoFunc::StNumGeometries,
        GeoFunc::StNumInteriorRings,
        GeoFunc::StGeometryN,
        GeoFunc::StPointN,
        GeoFunc::StStartPoint,
        GeoFunc::StEndPoint,
        GeoFunc::StExteriorRing,
        GeoFunc::StInteriorRingN,
        GeoFunc::StIsEmpty,
        GeoFunc::StIsValid,
        GeoFunc::StIsValidReason,
        GeoFunc::StIsClosed,
        GeoFunc::StIsRing,
        GeoFunc::StIsSimple,
        GeoFunc::StIsCollection,
        GeoFunc::StHasZ,
        GeoFunc::StCoordDim,
        GeoFunc::StArea,
        GeoFunc::StLength,
        GeoFunc::StPerimeter,
        GeoFunc::StDistance,
        GeoFunc::StMaxDistance,
        GeoFunc::StHausdorffDistance,
        GeoFunc::StAzimuth,
        GeoFunc::StDistanceSphere,
        GeoFunc::StDistanceSpheroid,
        GeoFunc::StAreaSpheroid,
        GeoFunc::StLengthSpheroid,
        GeoFunc::StPerimeterSpheroid,
        GeoFunc::StIntersects,
        GeoFunc::StDisjoint,
        GeoFunc::StContains,
        GeoFunc::StWithin,
        GeoFunc::StCovers,
        GeoFunc::StCoveredBy,
        GeoFunc::StTouches,
        GeoFunc::StCrosses,
        GeoFunc::StOverlaps,
        GeoFunc::StEquals,
        GeoFunc::StDwithin,
        GeoFunc::StDwithinSphere,
        GeoFunc::StIntersectsExtent,
        GeoFunc::StContainsExtent,
        GeoFunc::StCentroid,
        GeoFunc::StEnvelope,
        GeoFunc::StBoundary,
        GeoFunc::StConvexHull,
        GeoFunc::StPointOnSurface,
        GeoFunc::StBuffer,
        GeoFunc::StSimplify,
        GeoFunc::StReverse,
        GeoFunc::StForce2d,
        GeoFunc::StForce3d,
        GeoFunc::StForcePolygonCcw,
        GeoFunc::StForcePolygonCw,
        GeoFunc::StFlipCoordinates,
        GeoFunc::StTranslate,
        GeoFunc::StScale,
        GeoFunc::StRotate,
        GeoFunc::StAffine,
        GeoFunc::StSnapToGrid,
        GeoFunc::StSegmentize,
        GeoFunc::StExpand,
        GeoFunc::StCollect,
        GeoFunc::StRemoveRepeatedPoints,
        GeoFunc::StLineInterpolatePoint,
        GeoFunc::StLineLocatePoint,
        GeoFunc::StLineSubstring,
        GeoFunc::StClosestPoint,
        GeoFunc::StShortestLine,
        GeoFunc::StProject,
        GeoFunc::StTransform,
        GeoFunc::StGeohash,
        GeoFunc::GeohashEncode,
        GeoFunc::GeohashDecodeLon,
        GeoFunc::GeohashDecodeLat,
        GeoFunc::StTileX,
        GeoFunc::StTileY,
        GeoFunc::StQuadkey,
        GeoFunc::StS2Cell,
        GeoFunc::StS2CellParent,
        GeoFunc::StHexBin,
        GeoFunc::StHexCenterX,
        GeoFunc::StHexCenterY,
        GeoFunc::StUtmZone,
        GeoFunc::StUtmEpsg,
    ];
}
