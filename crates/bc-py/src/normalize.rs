//! Boundary type normalization: the input/output type adaptations the FFI applies
//! so the engine's kernels stay on a small, well-tested set of column types.
//!
//! On the way **in**, narrow numerics widen to Int64/Float64 and dictionary-encoded
//! columns decode to their value type, so no operator special-cases narrow or
//! dictionary inputs. On the way **out** (only when the control plane opts in via
//! `shrink_output_dtypes`), a pass-through of a narrow *source* column is cast back
//! to its source width where lossless. All of this is value-preserving.

use std::sync::Arc;

use arrow::array::{Array, RecordBatch};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Field, Schema};
use arrow_pyarrow::PyArrowType;
use bc_ir::{AggregateItem, ProjectionItem};
use pyo3::PyResult;

use crate::to_pyerr;

/// The cast dtype names the engine accepts on `Expr::Cast` (the live wire
/// vocabulary). The Python `plan.types.CAST_DTYPES` set is parity-tested against
/// this so the two cannot drift.
#[pyo3::pyfunction]
pub(crate) fn supported_cast_dtypes() -> Vec<String> {
    bc_arrow::CAST_DTYPE_NAMES
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Deserialize a group-key projection list from the control plane's JSON.
pub(crate) fn parse_group_keys(json: &str) -> PyResult<Vec<ProjectionItem>> {
    serde_json::from_str(json).map_err(to_pyerr)
}

/// Deserialize an aggregate-item list from the control plane's JSON.
pub(crate) fn parse_aggregates(json: &str) -> PyResult<Vec<AggregateItem>> {
    serde_json::from_str(json).map_err(to_pyerr)
}

/// The widened type the engine's Int64/Float64 kernels operate on, or `None` to
/// leave a column as-is. Real-world data is full of narrow numerics (Int32 ids,
/// Float32 features, unsigned counts); normalizing them once at the boundary lets
/// every operator stay on the two well-tested numeric paths.
pub(crate) fn widen_to(dt: &DataType) -> Option<DataType> {
    use DataType::*;
    match dt {
        Int8 | Int16 | Int32 | UInt8 | UInt16 | UInt32 | UInt64 => Some(Int64),
        Float16 | Float32 => Some(Float64),
        _ => None,
    }
}

/// The type a column is normalized to at the boundary, or `None` to leave it as-is.
///
/// Narrow numerics widen to Int64/Float64 (see [`widen_to`]); a `Dictionary` column
/// is **decoded** to its value type (then widened if that value is a narrow numeric),
/// so every operator sees plain primitive/string columns and never has to special-
/// case dictionary encoding — the same rationale as numeric widening.
fn normalize_to(dt: &DataType) -> Option<DataType> {
    match dt {
        DataType::Dictionary(_, value) => {
            Some(widen_to(value).unwrap_or_else(|| value.as_ref().clone()))
        }
        other => widen_to(other),
    }
}

/// Upcast narrow numeric columns of one batch to Int64/Float64 and decode any
/// dictionary-encoded columns to their value type. Non-numeric, already-wide,
/// non-dictionary columns are passed through untouched (a cheap `Arc` clone).
pub(crate) fn normalize_batch(batch: &RecordBatch) -> RecordBatch {
    let schema = batch.schema();
    let mut changed = false;
    let mut fields: Vec<Field> = Vec::with_capacity(schema.fields().len());
    let mut columns = Vec::with_capacity(batch.num_columns());
    for (i, field) in schema.fields().iter().enumerate() {
        let col = batch.column(i);
        match normalize_to(col.data_type()) {
            Some(target) => match cast(col, &target) {
                Ok(arr) => {
                    changed = true;
                    fields.push(Field::new(field.name(), target, field.is_nullable()));
                    columns.push(arr);
                }
                Err(_) => {
                    fields.push(field.as_ref().clone());
                    columns.push(col.clone());
                }
            },
            None => {
                fields.push(field.as_ref().clone());
                columns.push(col.clone());
            }
        }
    }
    if !changed {
        return batch.clone();
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap_or_else(|_| batch.clone())
}

/// Unwrap a Python list of pyarrow batches into normalized Arrow record batches.
pub(crate) fn unwrap_batches(batches: Vec<PyArrowType<RecordBatch>>) -> Vec<RecordBatch> {
    batches.into_iter().map(|b| normalize_batch(&b.0)).collect()
}

/// Map each narrow-numeric *source* column name to its original (pre-widening)
/// `DataType`, the target an output pass-through column can be re-narrowed back to.
///
/// Only columns whose type the boundary actually widens are recorded; a name that
/// appears in two sources with different narrow types is dropped (ambiguous), so a
/// re-narrow is never applied to the wrong width.
pub(crate) fn original_narrow_types(
    sources: &[Vec<RecordBatch>],
) -> std::collections::HashMap<String, DataType> {
    use std::collections::{HashMap, HashSet};
    let mut seen: HashMap<String, DataType> = HashMap::new();
    let mut ambiguous: HashSet<String> = HashSet::new();
    for relation in sources {
        if let Some(batch) = relation.first() {
            for field in batch.schema().fields() {
                if widen_to(field.data_type()).is_none() {
                    continue; // not a widened narrow numeric — nothing to restore
                }
                match seen.get(field.name()) {
                    Some(t) if t != field.data_type() => {
                        ambiguous.insert(field.name().clone());
                    }
                    None => {
                        seen.insert(field.name().clone(), field.data_type().clone());
                    }
                    _ => {}
                }
            }
        }
    }
    for name in ambiguous {
        seen.remove(&name);
    }
    seen
}

/// Re-narrow output columns back to their source numeric width where lossless.
///
/// For each output column that shares a name with a recorded narrow source column
/// and currently carries that column's widened type, a checked cast to the source
/// width is attempted; it is kept only if it introduces no new nulls (i.e. every
/// value was representable), so the result is always value-identical. A
/// pass-through Int32 id column thus leaves as Int32 instead of Int64.
pub(crate) fn narrow_output(
    batches: Vec<RecordBatch>,
    targets: &std::collections::HashMap<String, DataType>,
) -> Vec<RecordBatch> {
    use arrow::compute::{cast_with_options, CastOptions};
    if targets.is_empty() {
        return batches;
    }
    let opts = CastOptions {
        safe: true, // out-of-range → null, which we detect and reject
        ..Default::default()
    };
    batches
        .into_iter()
        .map(|batch| {
            let mut changed = false;
            let mut fields: Vec<Field> = Vec::with_capacity(batch.num_columns());
            let mut columns = Vec::with_capacity(batch.num_columns());
            for (i, field) in batch.schema().fields().iter().enumerate() {
                let col = batch.column(i);
                let target = targets.get(field.name());
                let widened_match = target
                    .map(|t| widen_to(t).as_ref() == Some(col.data_type()))
                    .unwrap_or(false);
                if let (Some(t), true) = (target, widened_match) {
                    match cast_with_options(col, t, &opts) {
                        Ok(arr) if arr.null_count() == col.null_count() => {
                            changed = true;
                            fields.push(Field::new(field.name(), t.clone(), field.is_nullable()));
                            columns.push(arr);
                            continue;
                        }
                        _ => {}
                    }
                }
                fields.push(field.as_ref().clone());
                columns.push(col.clone());
            }
            if !changed {
                return batch;
            }
            RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap_or(batch)
        })
        .collect()
}
