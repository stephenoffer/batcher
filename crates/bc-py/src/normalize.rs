//! Boundary type normalization: the input/output type adaptations the FFI applies
//! so the engine's kernels stay on a small, well-tested set of column types.
//!
//! On the way **in**, narrow numerics widen to Int64/Float64 and dictionary-encoded
//! columns decode to their value type, so no operator special-cases narrow or
//! dictionary inputs. On the way **out** (only when the control plane opts in via
//! `shrink_output_dtypes`), a pass-through of a narrow *source* column is cast back
//! to its source width where lossless. All of this is value-preserving.

use std::sync::Arc;

use arrow::array::{make_array, Array, ArrayRef, RecordBatch, RunArray, UInt64Array};
use arrow::compute::{cast, take};
use arrow::datatypes::{
    DataType, Field, Fields, Int16Type, Int32Type, Int64Type, RunEndIndexType, Schema,
};
use arrow_pyarrow::PyArrowType;
use bc_ir::{AggregateItem, ProjectionItem};
use pyo3::exceptions::PyRuntimeError;
use pyo3::PyResult;

use crate::to_pyerr;

/// The **fixed** cast dtype names the engine accepts on `Expr::Cast` (the live wire
/// vocabulary). The Python `plan.types.CAST_DTYPES` set is parity-tested against
/// this so the two cannot drift.
#[pyo3::pyfunction]
pub(crate) fn supported_cast_dtypes() -> Vec<String> {
    bc_arrow::CAST_DTYPE_NAMES
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Resolve one cast dtype *name* to the Arrow type the engine would build from it, or
/// `None` when the engine rejects the name.
///
/// The parametrized half of the vocabulary (`decimal(12,4)`, `timestamp(us, UTC)`,
/// `time64(ns)`) is a grammar rather than a set, so it cannot be parity-tested by
/// comparing name lists the way `supported_cast_dtypes` allows. This resolves a single
/// spelling instead, so the Python mirror can be checked to produce the *same type* the
/// engine does — which is the property that actually matters, and the one a name list
/// cannot see.
#[pyo3::pyfunction]
pub(crate) fn resolve_cast_dtype(name: &str) -> Option<PyArrowType<arrow::datatypes::DataType>> {
    bc_arrow::dtype_from_name(name).map(PyArrowType)
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
///
/// NOTE (2026-07-10): `assign_groups` *does* now have a dictionary fast path (grouping on
/// codes is ~7x faster than the decoded string), so preserving a string dictionary here
/// would win group-by/distinct/window. Measured from the other side on 2026-08-15, the
/// decode is what a dictionary input costs today: `GROUP BY id1, id2` over 10M rows runs in
/// **38.8 ms on plain `Utf8` and 151.5 ms on the same two columns dictionary-encoded** —
/// an encoding that exists to make this cheaper makes it 3.9x dearer, and the whole
/// difference is this cast. It is **not** yet safe to preserve: the logical plan's
/// schema treats a column by its Arrow type, so a preserved `Dictionary` propagates through
/// intermediate schemas, and an operator that decodes it (e.g. `distinct`'s rep column) then
/// produces a `Utf8` batch that fails the plan's `Dictionary` schema check. Enabling this is
/// RFC `rfc-streaming-executor.md` Proposal 3: separate the plan's *logical* type (value
/// type) from the morsel's *physical* encoding (dictionary), so the encoding is an internal
/// optimization the schema does not see. Until then, decode at the boundary.
///
/// Run-end encoding is handled *before* this, by [`decode_run_ends`] at the column level,
/// so a `RunEndEncoded` never reaches here. That split is deliberate rather than tidy:
/// decoding is a materialization (it expands runs into rows), and this function's contract
/// is that everything it names is reachable by a single `cast`. A run-end column nested
/// inside a struct or list is therefore left encoded, which is honest -- the boundary does
/// not rebuild a nested array's children -- and `plan.types.widen` mirrors the same split.
///
/// Normalization **recurses into nested types**: a `struct<a: int32>` normalizes to
/// `struct<a: int64>`, a `list<float32>` to `list<float64>`, and a `Dictionary` whose value
/// type is itself nested is decoded and then normalized. Without this, a narrow numeric buried
/// in a struct/list keeps its narrow width, and later arithmetic on `struct.field("a")` wraps
/// (an `int32` `2_000_000_000 + 2_000_000_000` silently becomes `-294967296`) where the same
/// value as a top-level column widens to `int64` and gives `4_000_000_000`. The Python type
/// inference (`plan/types/lattice.py::widen`) mirrors this recursion so `Dataset.schema` and
/// the engine agree on the nested widths.
fn normalize_to(dt: &DataType) -> Option<DataType> {
    use DataType::*;
    match dt {
        Dictionary(_, value) => Some(normalize_to(value).unwrap_or_else(|| value.as_ref().clone())),
        // The engine's string kernels (compare, `contains`, `upper`, join keys) accept only
        // `Utf8`; a `LargeUtf8` column would crash them ("expected a Utf8 argument, got
        // LargeUtf8") where the identical `Utf8` column succeeds. Normalize it to `Utf8` here so
        // every operator sees one string type — the same rationale as numeric widening. Values
        // are identical; the only difference is 32- vs 64-bit offset buffers, and a single morsel
        // (≤16,384 rows) cannot exceed the 32-bit offset range. (If a batch ever did, `cast`
        // errors and `normalize_batch` falls back to passing the column through unchanged.)
        LargeUtf8 => Some(Utf8),
        // The **view** layouts, for the same reason and with more urgency. `Utf8View` /
        // `BinaryView` are what a Parquet reader with view types enabled, DuckDB, Polars
        // and every Velox-backed producer hand over, and the engine's kernels reject them
        // outright ("Invalid comparison operation: Utf8View > Utf8"), so a column that is
        // *only* a different physical spelling of `Utf8` failed the whole query. A view
        // array is a 16-byte view struct per row plus one or more data buffers; casting to
        // `Utf8` re-lays those out behind a 32-bit offset buffer, which a single morsel
        // (<=16,384 rows) cannot overflow.
        Utf8View => Some(Utf8),
        BinaryView => Some(Binary),
        // The list *view* layouts carry an offset **and** a size per row instead of a
        // monotonic offset buffer, so they are a distinct physical layout the list kernels
        // do not read. Normalize to the plain `List` they are equivalent to, recursing into
        // the child so a `list_view<float32>` widens like a `list<float32>` does.
        ListView(field) | LargeListView(field) => Some(List(Arc::new(
            normalize_field(field).unwrap_or_else(|| field.as_ref().clone()),
        ))),
        Struct(fields) => normalize_fields(fields).map(Struct),
        List(field) => normalize_field(field).map(|f| List(Arc::new(f))),
        LargeList(field) => normalize_field(field).map(|f| LargeList(Arc::new(f))),
        FixedSizeList(field, n) => normalize_field(field).map(|f| FixedSizeList(Arc::new(f), *n)),
        Map(field, sorted) => normalize_field(field).map(|f| Map(Arc::new(f), *sorted)),
        other => widen_to(other),
    }
}

/// Expand a run-end-encoded column into its logical value array, or `None` if `arr` is not
/// one.
///
/// Run-end encoding is a *compression* of the value type in exactly the sense a dictionary
/// is: `RunArray` stores one value and one end index per **run**, and its `len()` is the
/// logical row count the runs expand to. Every operator would otherwise need a kernel for
/// it -- `Aggregate`, `Sort`, `Join` and the group-key identity in `bc_runtime::keys` all
/// reject the type outright -- so it decodes at the boundary for the same reason a
/// dictionary does: no operator should have to know how the column it reads was encoded.
///
/// `arrow::compute::cast` does not implement this in arrow 56, so the decode is done
/// directly: map each logical row to the physical slot holding its run's value, then
/// `take`. That is one gather over the column, and it is the same work the first operator
/// to touch the column would have had to do anyway.
///
/// This runs at the **column** level only. A run-end column nested inside a struct or list
/// is left encoded, because decoding it means rebuilding the containing array rather than
/// casting it. See the note on [`normalize_to`].
fn decode_run_ends(arr: &ArrayRef) -> Option<ArrayRef> {
    let DataType::RunEndEncoded(run_ends, _) = arr.data_type() else {
        return None;
    };
    // `get_physical_indices` is generic over the run-end index type, and the Arrow spec
    // admits exactly these three. An unrecognized one falls through to `None`, leaving the
    // column encoded rather than guessing at its layout.
    let physical = match run_ends.data_type() {
        DataType::Int16 => run_physical_indices::<Int16Type>(arr),
        DataType::Int32 => run_physical_indices::<Int32Type>(arr),
        DataType::Int64 => run_physical_indices::<Int64Type>(arr),
        _ => None,
    }?;
    let values = match run_ends.data_type() {
        DataType::Int16 => arr.as_any().downcast_ref::<RunArray<Int16Type>>()?.values(),
        DataType::Int32 => arr.as_any().downcast_ref::<RunArray<Int32Type>>()?.values(),
        DataType::Int64 => arr.as_any().downcast_ref::<RunArray<Int64Type>>()?.values(),
        _ => return None,
    };
    take(values.as_ref(), &physical, None).ok()
}

/// The physical slot each logical row of a `RunArray<R>` reads its value from.
///
/// Split out only because the three run-end index widths are three distinct types and the
/// lookup is identical for all of them. `get_physical_indices` accounts for the array's
/// slice offset, so a sliced run-end column decodes to its own rows and not its parent's.
fn run_physical_indices<R: RunEndIndexType>(arr: &ArrayRef) -> Option<UInt64Array> {
    let run = arr.as_any().downcast_ref::<RunArray<R>>()?;
    let logical: Vec<u32> = (0..u32::try_from(run.len()).ok()?).collect();
    let physical = run.get_physical_indices(&logical).ok()?;
    Some(UInt64Array::from_iter_values(
        physical.into_iter().map(|i| i as u64),
    ))
}

/// Whether `dt` is a type whose normalization is a *pure numeric widening* — either a
/// narrow numeric itself, or a container whose element type is one — and so can be
/// losslessly restored on the way out.
///
/// This deliberately mirrors [`normalize_to`]'s container recursion but stops short of
/// its `Dictionary` and `LargeUtf8` cases: those are normalized for operator
/// compatibility, not width, and re-encoding a dictionary on output is explicitly not
/// safe yet (see the NOTE on [`normalize_to`]).
///
/// The container arms are the point. [`normalize_to`] widens a `FixedSizeList<Float32>`
/// tensor column's *child* to `Float64`, but the restore side used to ask the flat
/// [`widen_to`], which answers `None` for any container — so an embedding or image
/// tensor column was widened on the way in and never narrowed back. That doubled the
/// bytes of every tensor result and, because `pa.fixed_shape_tensor` encodes its value
/// type, silently dropped the canonical extension type on the round trip: a column
/// written as `fixed_shape_tensor<float>` came back as a bare
/// `fixed_size_list<double>`.
fn restorable_narrow(dt: &DataType) -> bool {
    use DataType::*;
    match dt {
        Struct(fields) => fields.iter().any(|f| restorable_narrow(f.data_type())),
        List(field) | LargeList(field) | FixedSizeList(field, _) | Map(field, _) => {
            restorable_narrow(field.data_type())
        }
        other => widen_to(other).is_some(),
    }
}

/// Total nulls in `arr` *including* one level of children.
///
/// The restore cast runs in `safe` mode, where an unrepresentable value becomes a null
/// rather than an error. For a top-level primitive that shows up in the array's own
/// `null_count`, but for a container the loss lands in the **child**, leaving the
/// top-level count untouched — so a lossy `FixedSizeList<Float64> -> FixedSizeList<Float32>`
/// narrowing would otherwise pass the "introduced no new nulls" guard unnoticed.
fn nulls_including_children(arr: &dyn Array) -> usize {
    let own = arr.null_count();
    let data = arr.to_data();
    own + data
        .child_data()
        .iter()
        .map(|c| c.null_count())
        .sum::<usize>()
}

/// Whether `field` carries an Arrow extension type (`ARROW:extension:name`).
///
/// An extension column is opaque at the boundary: its storage type and element widths
/// are part of the type's contract, not incidental. The canonical
/// `arrow.fixed_shape_tensor` (an embedding / decoded-image column) encodes its
/// `value_type` — so widening its `f32` child to `f64` does not merely double the bytes
/// of the AI hot path, it *changes the extension type* from `tensor<float>` to
/// `tensor<double>`. Carry-through operators treat a tensor as an opaque unit (no
/// element-wise arithmetic), so there is nothing for the widening to make compatible.
/// Leaving it untouched also keeps the column genuinely zero-copy across the FFI.
pub(crate) fn is_extension_field(field: &Field) -> bool {
    field.metadata().contains_key("ARROW:extension:name")
}

/// The normalized form of `field` (recursing its data type), or `None` if unchanged.
///
/// Field metadata is carried across any width change so a nested tensor keeps its
/// extension type. Extension columns themselves are left entirely alone (see
/// [`is_extension_field`]).
fn normalize_field(field: &Field) -> Option<Field> {
    if is_extension_field(field) {
        return None;
    }
    normalize_to(field.data_type()).map(|t| {
        Field::new(field.name(), t, field.is_nullable()).with_metadata(field.metadata().clone())
    })
}

/// Recurse [`normalize_to`] over a struct's fields, returning the rebuilt `Fields` only if at
/// least one child changed (so an all-wide struct passes through with no reallocation).
fn normalize_fields(fields: &Fields) -> Option<Fields> {
    let mut changed = false;
    let out: Vec<Arc<Field>> = fields
        .iter()
        .map(|f| match normalize_field(f) {
            Some(nf) => {
                changed = true;
                Arc::new(nf)
            }
            None => f.clone(),
        })
        .collect();
    changed.then(|| out.into())
}

/// Total null count across an array **and every nested child**, so a data-loss check sees a null
/// introduced deep inside a struct/list (e.g. a `UInt64` above `i64::MAX` cast to `Int64`), not
/// only at the top level where `Array::null_count` looks.
///
/// A `Dictionary` is measured by its **logical** (row-level) null count and *not* recursed into:
/// its value buffer holds one entry per *distinct* value, so a single null value is shared by
/// every row that references it. Decoding replicates that null across all those rows, so a raw
/// physical sum of `keys.null_count() + values.null_count()` under-counts the input relative to
/// the decoded output and would flag a value-preserving decode as data loss. `logical_null_count`
/// counts null *rows* (null key or key pointing at a null value), which is exactly the granularity
/// the decoded column reports — so a genuine overflow buried in the dictionary's values (a
/// `Dictionary<_, UInt64>` above `i64::MAX`) still trips the guard, while a null-valued string
/// dictionary passes through unchanged.
fn deep_null_count(arr: &ArrayRef) -> usize {
    if matches!(arr.data_type(), DataType::Dictionary(_, _)) {
        return arr.logical_null_count();
    }
    let data = arr.to_data();
    let mut total = arr.null_count();
    for child in data.child_data() {
        total += deep_null_count(&make_array(child.clone()));
    }
    total
}

/// Push a column through unchanged, carrying whatever type it actually has.
///
/// The two fall-through arms of [`normalize_batch`] must agree with the column they push:
/// after [`decode_run_ends`] the array's type is the decoded value type, not the schema
/// field's `RunEndEncoded`, and pushing the original field there would build a
/// `RecordBatch` whose schema contradicts its columns.
fn push_as_is(fields: &mut Vec<Field>, columns: &mut Vec<ArrayRef>, field: &Field, col: &ArrayRef) {
    fields.push(
        Field::new(field.name(), col.data_type().clone(), field.is_nullable())
            .with_metadata(field.metadata().clone()),
    );
    columns.push(col.clone());
}

/// Upcast narrow numeric columns of one batch to Int64/Float64 and decode any
/// dictionary-encoded columns to their value type. Non-numeric, already-wide,
/// non-dictionary columns are passed through untouched (a cheap `Arc` clone).
///
/// Widening is meant to be **value-preserving**, and every recorded narrow→wide cast is
/// (Int8/16/32, UInt8/16/32 all fit in Int64; Float16/32 in Float64) — *except*
/// `UInt64 → Int64`, whose upper half (values above `i64::MAX`) has no Int64
/// representation. Arrow's safe cast turns such a value into a **null**, silently
/// replacing real data with a missing value. Rather than hand back a corrupted column,
/// this refuses the batch with a clear error naming the column: a `UInt64` above
/// `i64::MAX` is unsupported at the boundary, not silently lost.
pub(crate) fn normalize_batch(batch: &RecordBatch) -> PyResult<RecordBatch> {
    let schema = batch.schema();
    let mut changed = false;
    let mut fields: Vec<Field> = Vec::with_capacity(schema.fields().len());
    let mut columns = Vec::with_capacity(batch.num_columns());
    for (i, field) in schema.fields().iter().enumerate() {
        let raw = batch.column(i);
        // Run-end encoding is expanded first, so what follows sees the value type. An
        // extension column is opaque at the boundary and skips both steps.
        let decoded;
        let col = if is_extension_field(field) {
            raw
        } else {
            match decode_run_ends(raw) {
                Some(arr) => {
                    changed = true;
                    decoded = arr;
                    &decoded
                }
                None => raw,
            }
        };
        // An extension column (e.g. a fixed-shape-tensor embedding) is opaque at the
        // boundary — never rewrite its storage type. See `is_extension_field`.
        let normalized = if is_extension_field(field) {
            None
        } else {
            normalize_to(col.data_type())
        };
        match normalized {
            Some(target) => match cast(col, &target) {
                Ok(arr) => {
                    // A lossless widening never introduces a null. The one cast that can
                    // (UInt64 → Int64 overflow, at any nesting depth) would corrupt data
                    // silently — refuse it. Count nulls deeply so an overflow buried in a
                    // struct/list child is caught, not only a top-level one.
                    if deep_null_count(&arr) > deep_null_count(col) {
                        return Err(PyRuntimeError::new_err(format!(
                            "column {:?}: {} value exceeds the Int64 range the engine \
                             normalizes to (a value above i64::MAX cannot be represented \
                             without data loss); unsupported at the FFI boundary",
                            field.name(),
                            col.data_type(),
                        )));
                    }
                    changed = true;
                    // Preserve field metadata (the tensor extension type) across the
                    // input-side widening, same as the output-side restore.
                    fields.push(
                        Field::new(field.name(), target, field.is_nullable())
                            .with_metadata(field.metadata().clone()),
                    );
                    columns.push(arr);
                }
                Err(_) => {
                    push_as_is(&mut fields, &mut columns, field, col);
                }
            },
            None => {
                push_as_is(&mut fields, &mut columns, field, col);
            }
        }
    }
    if !changed {
        return Ok(batch.clone());
    }
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .unwrap_or_else(|_| batch.clone()))
}

/// Unwrap a Python list of pyarrow batches into normalized Arrow record batches.
pub(crate) fn unwrap_batches(batches: Vec<PyArrowType<RecordBatch>>) -> PyResult<Vec<RecordBatch>> {
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
                if !restorable_narrow(field.data_type()) {
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
                // `normalize_to`, not `widen_to`: the recorded target may be a container
                // (a tensor column) whose *child* was the thing widened.
                let widened_match = target
                    .map(|t| normalize_to(t).as_ref() == Some(col.data_type()))
                    .unwrap_or(false);
                if let (Some(t), true) = (target, widened_match) {
                    match cast_with_options(col, t, &opts) {
                        Ok(arr)
                            if nulls_including_children(&arr)
                                == nulls_including_children(col.as_ref()) =>
                        {
                            changed = true;
                            // Carry the field metadata so a restored tensor column keeps
                            // its `arrow.fixed_shape_tensor` extension type (mirrors
                            // `normalize_field`).
                            fields.push(
                                Field::new(field.name(), t.clone(), field.is_nullable())
                                    .with_metadata(field.metadata().clone()),
                            );
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        DictionaryArray, Float32Array, Int32Array, Int64Array, LargeStringArray, ListArray,
        StringArray, StructArray, UInt64Array,
    };
    use arrow::buffer::OffsetBuffer;
    use arrow::datatypes::Int8Type;

    fn batch_of(name: &str, arr: arrow::array::ArrayRef) -> RecordBatch {
        let field = Field::new(name, arr.data_type().clone(), true);
        RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![arr]).unwrap()
    }

    #[test]
    fn large_utf8_normalizes_to_utf8_value_preserving() {
        let arr = Arc::new(LargeStringArray::from(vec![Some("a"), None, Some("bb")]));
        let out = normalize_batch(&batch_of("s", arr)).unwrap();
        assert_eq!(out.schema().field(0).data_type(), &DataType::Utf8);
        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("column normalized to Utf8");
        assert_eq!(col.value(0), "a");
        assert!(col.is_null(1));
        assert_eq!(col.value(2), "bb");
    }

    #[test]
    fn dictionary_of_large_utf8_decodes_to_utf8() {
        // A dictionary whose value type is LargeUtf8 must decode to Utf8, not LargeUtf8,
        // so downstream string kernels (which accept only Utf8) never see the large form.
        let values = Arc::new(LargeStringArray::from(vec!["x", "y"]));
        let keys = Int32Array::from(vec![0, 1, 0])
            .iter()
            .map(|k| k.map(|v| v as i8))
            .collect::<arrow::array::PrimitiveArray<Int8Type>>();
        let dict = DictionaryArray::new(keys, values);
        let out = normalize_batch(&batch_of("d", Arc::new(dict))).unwrap();
        assert_eq!(out.schema().field(0).data_type(), &DataType::Utf8);
        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("dictionary decoded to Utf8");
        assert_eq!(col.value(0), "x");
        assert_eq!(col.value(1), "y");
        assert_eq!(col.value(2), "x");
    }

    #[test]
    fn dictionary_with_null_values_decodes_not_rejected() {
        // A dictionary whose VALUES array contains a null, referenced by several rows, must
        // decode to a plain column with a null in each of those rows — not be rejected by the
        // UInt64-overflow data-loss guard. Physically the dict has one null value; decoding
        // replicates it to two null rows, so a raw physical null-count comparison false-flags
        // this value-preserving decode as data loss (the bug: `deep_null_count` double-counting).
        let values = Arc::new(StringArray::from(vec![Some("a"), None]));
        let keys = Int32Array::from(vec![0, 1, 0, 1])
            .iter()
            .map(|k| k.map(|v| v as i8))
            .collect::<arrow::array::PrimitiveArray<Int8Type>>();
        let dict = DictionaryArray::new(keys, values);
        let out = normalize_batch(&batch_of("d", Arc::new(dict)))
            .expect("dictionary with a null value must decode, not error");
        assert_eq!(out.schema().field(0).data_type(), &DataType::Utf8);
        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("decoded to Utf8");
        assert_eq!(col.value(0), "a");
        assert!(col.is_null(1));
        assert_eq!(col.value(2), "a");
        assert!(col.is_null(3));
    }

    #[test]
    fn dictionary_of_uint64_overflow_is_still_rejected() {
        // The guard must still fire when a dictionary's VALUES genuinely overflow Int64: the
        // logical-null path must not blind it to real data loss inside a dictionary.
        let values = Arc::new(UInt64Array::from(vec![1u64, u64::MAX]));
        let keys = Int32Array::from(vec![0, 1])
            .iter()
            .map(|k| k.map(|v| v as i8))
            .collect::<arrow::array::PrimitiveArray<Int8Type>>();
        let dict = DictionaryArray::new(keys, values);
        assert!(normalize_batch(&batch_of("d", Arc::new(dict))).is_err());
    }

    #[test]
    fn narrow_int_still_widens_to_int64() {
        let arr = Arc::new(Int32Array::from(vec![1, -2, 3]));
        let out = normalize_batch(&batch_of("i", arr)).unwrap();
        assert_eq!(out.schema().field(0).data_type(), &DataType::Int64);
    }

    #[test]
    fn uint64_above_i64_max_is_rejected_not_corrupted() {
        let arr = Arc::new(UInt64Array::from(vec![1u64, u64::MAX]));
        assert!(normalize_batch(&batch_of("u", arr)).is_err());
    }

    #[test]
    fn struct_of_narrow_int_widens_child_to_int64() {
        // A narrow int inside a struct must widen, or `struct.field("a") + struct.field("a")`
        // on `2_000_000_000` wraps (int32 overflow) instead of giving `4_000_000_000`.
        let a = Arc::new(Int32Array::from(vec![2_000_000_000, -2])) as arrow::array::ArrayRef;
        let field = Arc::new(Field::new("a", DataType::Int32, true));
        let s = StructArray::from(vec![(field, a)]);
        let out = normalize_batch(&batch_of("s", Arc::new(s))).unwrap();
        assert_eq!(
            out.schema().field(0).data_type(),
            &DataType::Struct(vec![Field::new("a", DataType::Int64, true)].into())
        );
    }

    #[test]
    fn list_of_narrow_float_widens_child_to_float64() {
        let values = Arc::new(Float32Array::from(vec![1.0f32, 2.0, 3.0]));
        let offsets = OffsetBuffer::new(vec![0, 2, 3].into());
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let list = ListArray::new(field, offsets, values, None);
        let out = normalize_batch(&batch_of("l", Arc::new(list))).unwrap();
        assert_eq!(
            out.schema().field(0).data_type(),
            &DataType::List(Arc::new(Field::new("item", DataType::Float64, true)))
        );
    }

    #[test]
    fn nested_uint64_overflow_is_rejected_not_corrupted() {
        // The deep null-count check must catch a UInt64 above i64::MAX buried in a struct,
        // exactly as it does for a top-level column.
        let a = Arc::new(UInt64Array::from(vec![1u64, u64::MAX])) as arrow::array::ArrayRef;
        let field = Arc::new(Field::new("a", DataType::UInt64, true));
        let s = StructArray::from(vec![(field, a)]);
        assert!(normalize_batch(&batch_of("s", Arc::new(s))).is_err());
    }

    #[test]
    fn struct_of_wide_types_passes_through_unchanged() {
        // An all-wide struct must not be reallocated (the fast path).
        let a = Arc::new(Int64Array::from(vec![1i64, 2])) as arrow::array::ArrayRef;
        let field = Arc::new(Field::new("a", DataType::Int64, true));
        let s = StructArray::from(vec![(field, a)]);
        assert!(normalize_to(&DataType::Struct(
            vec![Field::new("a", DataType::Int64, true)].into()
        ))
        .is_none());
        let out = normalize_batch(&batch_of("s", Arc::new(s))).unwrap();
        assert_eq!(
            out.schema().field(0).data_type(),
            &DataType::Struct(vec![Field::new("a", DataType::Int64, true)].into())
        );
    }
}
