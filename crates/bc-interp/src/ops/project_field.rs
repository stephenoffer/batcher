//! Output-field construction for [`super::project_batch_jit`].
//!
//! Picking each projected column's Arrow `Field` is its own concern: a bare column
//! carries its source field through (preserving extension metadata), while a native
//! image-decode column is tagged as a fixed-shape tensor. Kept out of `ops/mod.rs` so
//! that file stays within its size budget.

use std::collections::HashMap;

use arrow::array::ArrayRef;
use arrow::datatypes::{DataType, Field};
use arrow::record_batch::RecordBatch;
use bc_ir::ProjectionItem;

/// The output `Field` for one projected column.
///
/// A bare-column passthrough carries the *source field* through (renamed), preserving
/// its metadata — notably the Arrow extension type (e.g. `FixedShapeTensor` for
/// embeddings / already-decoded media); rebuilding from `array.data_type()` would drop
/// it, downgrading a tensor column to its plain storage type. A native image decode to a
/// fixed-shape `(H, W, 3)` RGB8 tensor is tagged with the canonical
/// `arrow.fixed_shape_tensor` extension metadata. Every other computed expression gets a
/// fresh field from its array's type.
pub(super) fn output_field(item: &ProjectionItem, array: &ArrayRef, batch: &RecordBatch) -> Field {
    match &item.expr {
        bc_expr::Expr::Col { name } => match batch.schema().index_of(name) {
            Ok(idx) => {
                let src = batch
                    .schema()
                    .field(idx)
                    .clone()
                    .with_name(item.alias.clone());
                // The **evaluated array** is the authority on the physical type; the source
                // field is the authority on everything else. A bare column is normally a
                // zero-copy `Arc` clone of the source, so the two agree and this is the source
                // field unchanged, which is what preserves the extension metadata above.
                //
                // They disagree exactly when `Expr::eval` changed the encoding, and today that
                // means one thing: a `Dictionary` column is decoded at the `Col` leaf
                // (`bc_expr::eval::dispatch::decode_dict`), so the field would claim
                // `Dictionary(Int32, Utf8)` for a `Utf8` array and `RecordBatch::try_new`
                // rejects the whole batch — "column types must match schema types". Taking the
                // type from the array while keeping the name, nullability and metadata
                // satisfies both concerns at once.
                //
                // This is the operator `bc_py::normalize.rs`'s NOTE predicts would break if the
                // FFI boundary stopped decoding dictionaries, and it was the only one:
                // `tests/dictionary_operators.rs` runs every operator over a dictionary against
                // the decoded oracle, and Project was the single failure (Distinct and HashJoin
                // failed only through the Project inside them).
                if src.data_type() == array.data_type() {
                    src
                } else {
                    src.with_data_type(array.data_type().clone())
                }
            }
            Err(_) => Field::new(&item.alias, array.data_type().clone(), true),
        },
        bc_expr::Expr::Image {
            func: bc_expr::ImageFunc::ToTensor | bc_expr::ImageFunc::CenterCrop,
            width: Some(w),
            height: Some(h),
            ..
        } => tensor_field(&item.alias, array.data_type().clone(), [*h, *w, 3]),
        bc_expr::Expr::Image {
            func: bc_expr::ImageFunc::ToGrayscale,
            width: Some(w),
            height: Some(h),
            ..
        } => tensor_field(&item.alias, array.data_type().clone(), [*h, *w, 1]),
        bc_expr::Expr::Image {
            func: bc_expr::ImageFunc::ToTensorF32,
            width: Some(w),
            height: Some(h),
            channels_first,
            ..
        } => {
            // The float tensor's shape follows the requested layout: CHW vs HWC.
            let shape = if *channels_first {
                [3, *h, *w]
            } else {
                [*h, *w, 3]
            };
            tensor_field(&item.alias, array.data_type().clone(), shape)
        }
        _ => Field::new(&item.alias, array.data_type().clone(), true),
    }
}

/// A field tagged with the canonical `arrow.fixed_shape_tensor` extension metadata for
/// the given per-row `shape` — the *same* metadata `pa.fixed_shape_tensor` writes, so the
/// flat `FixedSizeList` storage reconstructs to a shaped tensor on the pyarrow side of the
/// FFI, with no per-batch Python re-type pass (which otherwise forces the whole decode
/// through the slow opaque-UDF path — the physical-AI ingest bottleneck).
fn tensor_field(alias: &str, dtype: DataType, shape: [i64; 3]) -> Field {
    let [a, b, c] = shape;
    let metadata = HashMap::from([
        (
            "ARROW:extension:name".to_string(),
            "arrow.fixed_shape_tensor".to_string(),
        ),
        (
            "ARROW:extension:metadata".to_string(),
            format!("{{\"shape\":[{a},{b},{c}]}}"),
        ),
    ]);
    Field::new(alias, dtype, true).with_metadata(metadata)
}
