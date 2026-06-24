//! Bounded out-of-core `histogram(value)` — the `Map<value, count>` member of the
//! value-list aggregate family (`super`), split out so the parent module stays
//! within the file-size budget. Shares the native-value sorted-run machinery
//! (`flatten_native_value` / `native_value_sort_keys` / `empty_key_columns`) with
//! the distinct and mode paths.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch, UInt32Array};
use arrow::compute::{concat, take};
use arrow::datatypes::{DataType, Field};
use arrow::row::{OwnedRow, RowConverter, SortField};
use bc_ir::ProjectionItem;

use super::{empty_key_columns, flatten_native_value, native_value_sort_keys};
use crate::error::InterpError;
use crate::ops::external_sort_to_final_store;

/// Exact per-group `histogram(value)` with bounded memory: a `Map<value, count>` per
/// group (keys the distinct non-null values ascending, values their counts). Returns
/// the group key columns and the map column. Bit-for-bit the in-memory
/// `finalize_histogram`: nulls excluded, an empty/all-null group → a NULL map. Each
/// distinct value's contiguous run in the sorted `(group, value)` stream is one map
/// entry, so no group's value list is ever fully resident.
pub(crate) fn bounded_group_histogram(
    parts: &[RecordBatch],
    group_keys: &[ProjectionItem],
    value_expr: &bc_expr::Expr,
    dir: &std::path::Path,
) -> Result<(Vec<ArrayRef>, ArrayRef), InterpError> {
    use arrow::array::new_empty_array;

    let n_keys = group_keys.len();
    let value_type = match parts.first() {
        Some(p) => value_expr.eval(p)?.data_type().clone(),
        None => DataType::Null,
    };

    let (flat, schema) = flatten_native_value(parts, group_keys, value_expr)?;
    let Some(schema) = schema else {
        let empty = histogram_map(
            new_empty_array(&value_type),
            Vec::new(),
            vec![0],
            Vec::new(),
        )?;
        return Ok((Vec::new(), empty));
    };
    let sort_keys = native_value_sort_keys(n_keys);
    let Some(mut store) = external_sort_to_final_store(flat, &sort_keys, dir)? else {
        let empty = histogram_map(
            new_empty_array(&value_type),
            Vec::new(),
            vec![0],
            Vec::new(),
        )?;
        return Ok((empty_key_columns(&schema, n_keys), empty));
    };

    let key_conv = RowConverter::new(
        (0..n_keys)
            .map(|i| SortField::new(schema.field(i).data_type().clone()))
            .collect(),
    )?;
    let val_conv = RowConverter::new(vec![SortField::new(
        schema.field(n_keys).data_type().clone(),
    )])?;

    // Each distinct value's run → one (key, count) entry; `map_offsets`/`valid`
    // delimit each group's entries (an all-null group has no entries → NULL map).
    let mut key_rows: Vec<OwnedRow> = Vec::new();
    let mut counts: Vec<i64> = Vec::new();
    let mut map_offsets: Vec<i32> = vec![0];
    let mut valid: Vec<bool> = Vec::new();
    let mut key_cols: Vec<Vec<ArrayRef>> = vec![Vec::new(); n_keys];
    let mut prev_group: Option<OwnedRow> = None;
    let mut started = false;
    let mut cur_key: Option<OwnedRow> = None;
    let mut cur_count = 0i64;
    let mut group_has_value = false;

    if let Some(reader) = store.open_reader(0).map_err(InterpError::from)? {
        for batch in reader {
            let batch = batch?;
            let vcol = batch.column(n_keys);
            let vrows = val_conv.convert_columns(std::slice::from_ref(vcol))?;
            let grows = if n_keys > 0 {
                Some(key_conv.convert_columns(&batch.columns()[..n_keys])?)
            } else {
                None
            };
            let mut firsts: Vec<u32> = Vec::new();
            for i in 0..batch.num_rows() {
                let group = grows.as_ref().map(|g| g.row(i).owned());
                if !started || (n_keys > 0 && prev_group != group) {
                    if started {
                        if let Some(k) = cur_key.take() {
                            key_rows.push(k);
                            counts.push(cur_count);
                        }
                        map_offsets.push(key_rows.len() as i32);
                        valid.push(group_has_value);
                    }
                    started = true;
                    prev_group = group;
                    cur_key = None;
                    cur_count = 0;
                    group_has_value = false;
                    if n_keys > 0 {
                        firsts.push(i as u32);
                    }
                }
                if vcol.is_valid(i) {
                    let vr = vrows.row(i).owned();
                    group_has_value = true;
                    if cur_key.as_ref() == Some(&vr) {
                        cur_count += 1;
                    } else {
                        if let Some(k) = cur_key.take() {
                            key_rows.push(k);
                            counts.push(cur_count);
                        }
                        cur_key = Some(vr);
                        cur_count = 1;
                    }
                }
            }
            if !firsts.is_empty() && n_keys > 0 {
                let idx = UInt32Array::from(firsts);
                for (c, slot) in key_cols.iter_mut().enumerate() {
                    slot.push(take(batch.column(c), &idx, None)?);
                }
            }
        }
    }
    if started {
        if let Some(k) = cur_key.take() {
            key_rows.push(k);
            counts.push(cur_count);
        }
        map_offsets.push(key_rows.len() as i32);
        valid.push(group_has_value);
    }

    let group_columns: Vec<ArrayRef> = (0..n_keys)
        .map(|c| -> Result<ArrayRef, InterpError> {
            if key_cols[c].is_empty() {
                Ok(new_empty_array(schema.field(c).data_type()))
            } else {
                let refs: Vec<&dyn Array> = key_cols[c].iter().map(|a| a.as_ref()).collect();
                Ok(concat(&refs)?)
            }
        })
        .collect::<Result<_, _>>()?;
    let keys = if key_rows.is_empty() {
        new_empty_array(&value_type)
    } else {
        val_conv
            .convert_rows(key_rows.iter().map(|r| r.row()))?
            .into_iter()
            .next()
            .unwrap_or_else(|| new_empty_array(&value_type))
    };
    Ok((
        group_columns,
        histogram_map(keys, counts, map_offsets, valid)?,
    ))
}

/// Assemble a `Map<key, count>` array from per-group entry slices — the exact output
/// shape of the in-memory `finalize_histogram` (keys non-null, counts `Int64`, an
/// empty group marked NULL via `valid`).
fn histogram_map(
    keys: ArrayRef,
    counts: Vec<i64>,
    map_offsets: Vec<i32>,
    valid: Vec<bool>,
) -> Result<ArrayRef, InterpError> {
    use arrow::array::{Int64Array, MapArray, StructArray};
    use arrow::buffer::{NullBuffer, OffsetBuffer};
    use arrow::datatypes::Fields;

    let vals: ArrayRef = Arc::new(Int64Array::from(counts));
    let key_field = Arc::new(Field::new("key", keys.data_type().clone(), false));
    let val_field = Arc::new(Field::new("value", DataType::Int64, true));
    let struct_fields = Fields::from(vec![key_field, val_field]);
    let entries = StructArray::new(struct_fields.clone(), vec![keys, vals], None);
    let entries_field = Arc::new(Field::new(
        "entries",
        DataType::Struct(struct_fields),
        false,
    ));
    let nulls = (!valid.is_empty()).then(|| NullBuffer::from(valid));
    let map = MapArray::try_new(
        entries_field,
        OffsetBuffer::new(map_offsets.into()),
        entries,
        nulls,
        false,
    )?;
    Ok(Arc::new(map))
}
