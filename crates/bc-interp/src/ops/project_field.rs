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
            Ok(idx) => batch
                .schema()
                .field(idx)
                .clone()
                .with_name(item.alias.clone()),
            Err(_) => Field::new(&item.alias, array.data_type().clone(), true),
        },
        bc_expr::Expr::Image {
            func: bc_expr::ImageFunc::ToTensor,
            width: Some(w),
            height: Some(h),
            ..
        } => tensor_field(&item.alias, array.data_type().clone(), *h, *w),
        _ => Field::new(&item.alias, array.data_type().clone(), true),
    }
}

/// A field tagged with the canonical `arrow.fixed_shape_tensor` extension metadata
/// (shape `(H, W, 3)`, RGB8) — the *same* metadata `pa.fixed_shape_tensor` writes, so the
/// storage `FixedSizeList<u8, H*W*3>` reconstructs to a shaped tensor on the pyarrow side
/// of the FFI, with no per-batch Python re-type pass (which otherwise forces the whole
/// decode through the slow opaque-UDF path — the physical-AI ingest bottleneck).
fn tensor_field(alias: &str, dtype: DataType, h: i64, w: i64) -> Field {
    let metadata = HashMap::from([
        (
            "ARROW:extension:name".to_string(),
            "arrow.fixed_shape_tensor".to_string(),
        ),
        (
            "ARROW:extension:metadata".to_string(),
            format!("{{\"shape\":[{h},{w},3]}}"),
        ),
    ]);
    Field::new(alias, dtype, true).with_metadata(metadata)
}
