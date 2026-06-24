//! Bounded out-of-core exact value-list aggregates for a single grouped aggregate.
//!
//! `median(x)` / `quantile(x, q)` and `n_unique(x)` (COUNT DISTINCT) keep **every**
//! value per group as their partial state; on a hot key that per-group list can
//! exceed memory (the unbounded case the in-memory aggregate has). This computes the
//! exact result with bounded memory by sorting the `(group_keys.., value)` rows out
//! of core (the spilling external sort) and streaming over the sorted run, so no
//! group's values are ever fully resident:
//!
//! * **median / quantile** — value cast to `f64`, two passes (count rows + null
//!   values per group, then pick the interpolated position). Bit-for-bit the
//!   in-memory `finalize_quantile`.
//! * **n_unique / mode / histogram** — value kept in its **native** type (so they are
//!   correct for strings etc., where an `f64` cast would be wrong), one pass over the
//!   sorted run (equal values adjacent): `n_unique` counts distinct non-null values
//!   per group (bit-for-bit `finalize_count_distinct`, nulls excluded, `Int64`);
//!   `mode` tracks the longest equal-value run, ties → smallest (bit-for-bit
//!   `finalize_mode`, native type, empty group → null); `histogram` (in the
//!   [`histogram`] submodule) emits a `Map<value, count>` per group (bit-for-bit
//!   `finalize_histogram`, empty group → NULL map).
//!
//! All stay differential-equal to DuckDB. Every shape here produces an output
//! *smaller* than the per-group value list (a count, one picked value, or a
//! distinct-keyed map), which is what lets a streaming finalizer bound memory.
//! `listagg`/`array_agg` is deliberately **not** here: its output *is* the whole
//! value list, so no streaming finalizer can shrink it below what the grace
//! partition path (`combine_finalize_spilling`) already bounds — it stays on that
//! path. Such shapes return `None` so the caller uses the in-memory grace path.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, Float64Array, Float64Builder, Int64Array, RecordBatch, UInt32Array,
};
use arrow::compute::{cast, concat, take};
use arrow::datatypes::{DataType, Field, Float64Type, Schema};
use arrow::row::{OwnedRow, RowConverter, SortField};
use bc_ir::{ProjectionItem, SortKey};

use crate::error::InterpError;
use crate::ops::external_sort_to_final_store;

mod histogram;

/// Finalized group-key columns paired with the finalized aggregate columns.
type GroupedColumns = (Vec<ArrayRef>, Vec<ArrayRef>);

/// If `aggregates` is a lone `median`/`quantile`, compute it out-of-core with bounded
/// memory and return `(group columns, [quantile column])`; otherwise `None` so the
/// caller uses the in-memory grace path (unchanged for every other aggregate shape).
/// `median`/`quantile` keep every value per group, so a hot key can exceed memory —
/// this is the spilling path's bounded answer for them.
pub(crate) fn try_bounded_quantile_spill(
    parts: &[RecordBatch],
    group_keys: &[ProjectionItem],
    aggregates: &[bc_ir::AggregateItem],
    spill_dir: &std::path::Path,
) -> Result<Option<GroupedColumns>, InterpError> {
    let Some((value_expr, q)) = single_quantile(aggregates) else {
        return Ok(None);
    };
    let dir = spill_dir.join("agg-quantile");
    let (gc, qc) = bounded_group_quantile(parts, group_keys, value_expr, q, &dir)?;
    Ok(Some((gc, vec![qc])))
}

/// If `aggregates` is a lone `n_unique` (COUNT DISTINCT), compute it out-of-core
/// with bounded memory and return `(group columns, [count column])`; otherwise
/// `None` so the caller uses the in-memory grace path. The exact distinct value set
/// per group is the unbounded in-memory state a hot key can blow — this is its
/// bounded answer.
pub(crate) fn try_bounded_distinct_spill(
    parts: &[RecordBatch],
    group_keys: &[ProjectionItem],
    aggregates: &[bc_ir::AggregateItem],
    spill_dir: &std::path::Path,
) -> Result<Option<GroupedColumns>, InterpError> {
    let Some(value_expr) = single_distinct(aggregates) else {
        return Ok(None);
    };
    let dir = spill_dir.join("agg-distinct");
    let (gc, cc) = bounded_group_distinct(parts, group_keys, value_expr, &dir)?;
    Ok(Some((gc, vec![cc])))
}

/// The value expr of a lone `n_unique` (COUNT DISTINCT) aggregate, else `None`.
fn single_distinct(aggregates: &[bc_ir::AggregateItem]) -> Option<&bc_expr::Expr> {
    if aggregates.len() != 1 {
        return None;
    }
    let a = &aggregates[0];
    match a.func {
        bc_ir::AggFunc::CountDistinct => a.input.as_ref(),
        _ => None,
    }
}

/// If `aggregates` is a lone `mode`, compute it out-of-core with bounded memory and
/// return `(group columns, [mode column])`; otherwise `None` so the caller uses the
/// in-memory grace path. The exact per-group value list is the unbounded in-memory
/// state a hot key can blow — this is its bounded answer.
pub(crate) fn try_bounded_mode_spill(
    parts: &[RecordBatch],
    group_keys: &[ProjectionItem],
    aggregates: &[bc_ir::AggregateItem],
    spill_dir: &std::path::Path,
) -> Result<Option<GroupedColumns>, InterpError> {
    let Some(value_expr) = single_mode(aggregates) else {
        return Ok(None);
    };
    let dir = spill_dir.join("agg-mode");
    let (gc, mc) = bounded_group_mode(parts, group_keys, value_expr, &dir)?;
    Ok(Some((gc, vec![mc])))
}

/// The value expr of a lone `mode` aggregate, else `None`.
fn single_mode(aggregates: &[bc_ir::AggregateItem]) -> Option<&bc_expr::Expr> {
    if aggregates.len() != 1 {
        return None;
    }
    let a = &aggregates[0];
    match a.func {
        bc_ir::AggFunc::Mode => a.input.as_ref(),
        _ => None,
    }
}

/// If `aggregates` is a lone `histogram`, compute it out-of-core with bounded memory
/// and return `(group columns, [map column])`; otherwise `None` so the caller uses
/// the in-memory grace path. The exact per-group value list is the unbounded
/// in-memory state a hot key can blow — this is its bounded answer.
pub(crate) fn try_bounded_histogram_spill(
    parts: &[RecordBatch],
    group_keys: &[ProjectionItem],
    aggregates: &[bc_ir::AggregateItem],
    spill_dir: &std::path::Path,
) -> Result<Option<GroupedColumns>, InterpError> {
    let Some(value_expr) = single_histogram(aggregates) else {
        return Ok(None);
    };
    let dir = spill_dir.join("agg-histogram");
    let (gc, mc) = histogram::bounded_group_histogram(parts, group_keys, value_expr, &dir)?;
    Ok(Some((gc, vec![mc])))
}

/// The value expr of a lone `histogram` aggregate, else `None`.
fn single_histogram(aggregates: &[bc_ir::AggregateItem]) -> Option<&bc_expr::Expr> {
    if aggregates.len() != 1 {
        return None;
    }
    let a = &aggregates[0];
    match a.func {
        bc_ir::AggFunc::Histogram => a.input.as_ref(),
        _ => None,
    }
}

/// The `(value expr, q)` of a lone `median`/`quantile` aggregate, else `None`.
fn single_quantile(aggregates: &[bc_ir::AggregateItem]) -> Option<(&bc_expr::Expr, f64)> {
    if aggregates.len() != 1 {
        return None;
    }
    let a = &aggregates[0];
    let q = match a.func {
        bc_ir::AggFunc::Median => 0.5,
        bc_ir::AggFunc::Quantile => a.param?, // the quantile in [0,1]
        _ => return None,
    };
    a.input.as_ref().map(|e| (e, q))
}

/// Exact per-group `quantile(value, q)` (median is `q = 0.5`) with bounded memory.
/// Returns the group key columns and the `Float64` quantile column, aligned row-wise.
/// Only valid for a single median/quantile aggregate; the caller falls back to the
/// in-memory grace path for any other (mixed/multiple) aggregate.
pub(crate) fn bounded_group_quantile(
    parts: &[RecordBatch],
    group_keys: &[ProjectionItem],
    value_expr: &bc_expr::Expr,
    q: f64,
    dir: &std::path::Path,
) -> Result<(Vec<ArrayRef>, ArrayRef), InterpError> {
    let n_keys = group_keys.len();

    // 1. Flatten each input batch to `(g0..gN, v:f64)` rows — value cast to f64, nulls
    //    kept so an all-null group still yields a null (matching the in-memory path).
    let mut flat: Vec<RecordBatch> = Vec::with_capacity(parts.len());
    let mut schema: Option<Arc<Schema>> = None;
    for part in parts {
        if part.num_rows() == 0 {
            continue;
        }
        let mut fields: Vec<Field> = Vec::with_capacity(n_keys + 1);
        let mut cols: Vec<ArrayRef> = Vec::with_capacity(n_keys + 1);
        for (i, gk) in group_keys.iter().enumerate() {
            let a = gk.expr.eval(part)?;
            fields.push(Field::new(format!("g{i}"), a.data_type().clone(), true));
            cols.push(a);
        }
        let v = value_expr.eval(part)?;
        cols.push(cast(&v, &DataType::Float64)?);
        fields.push(Field::new("v", DataType::Float64, true));
        let s = Arc::new(Schema::new(fields));
        schema.get_or_insert_with(|| s.clone());
        flat.push(RecordBatch::try_new(s, cols)?);
    }
    let Some(schema) = schema else {
        // Every input batch empty — only reachable defensively (an empty aggregate
        // does not spill). Zero output groups.
        return Ok((Vec::new(), Arc::new(Float64Array::from(Vec::<f64>::new()))));
    };

    // 2. Sort `(group asc, value asc nulls-first)` out of core. Value nulls sort first
    //    within a group, so they are that group's leading rows (counted, then skipped).
    let mut sort_keys: Vec<SortKey> = (0..n_keys)
        .map(|i| SortKey {
            expr: bc_expr::Expr::Col {
                name: format!("g{i}"),
            },
            descending: false,
            nulls_first: false,
        })
        .collect();
    sort_keys.push(SortKey {
        expr: bc_expr::Expr::Col { name: "v".into() },
        descending: false,
        nulls_first: true,
    });

    let Some(mut store) = external_sort_to_final_store(flat, &sort_keys, dir)? else {
        return Ok((
            empty_key_columns(&schema, n_keys),
            Arc::new(Float64Array::from(Vec::<f64>::new())),
        ));
    };

    // Group-key row converter for boundary detection across the sorted stream.
    let key_conv = RowConverter::new(
        (0..n_keys)
            .map(|i| SortField::new(schema.field(i).data_type().clone()))
            .collect(),
    )?;

    // 3a. Pass 1 — per-group total and null-value counts (groups are contiguous).
    let mut counts: Vec<usize> = Vec::new();
    let mut null_counts: Vec<usize> = Vec::new();
    let mut prev: Option<OwnedRow> = None;
    if let Some(reader) = store.open_reader(0).map_err(InterpError::from)? {
        for batch in reader {
            let batch = batch?;
            let vcol = batch.column(n_keys);
            if n_keys == 0 {
                // Global aggregate (no GROUP BY): every row is the single group.
                if counts.is_empty() {
                    counts.push(0);
                    null_counts.push(0);
                }
                counts[0] += batch.num_rows();
                null_counts[0] += vcol.null_count();
            } else {
                let grows = key_conv.convert_columns(&batch.columns()[..n_keys])?;
                for i in 0..batch.num_rows() {
                    let row = grows.row(i).owned();
                    if prev.as_ref().is_none_or(|p| *p != row) {
                        counts.push(0);
                        null_counts.push(0);
                        prev = Some(row);
                    }
                    *counts.last_mut().unwrap() += 1;
                    if !vcol.is_valid(i) {
                        *null_counts.last_mut().unwrap() += 1;
                    }
                }
            }
        }
    }

    // 3b. Pass 2 — pick the interpolated quantile per group + capture its key row.
    let qc = q.clamp(0.0, 1.0);
    let mut out = Float64Builder::with_capacity(counts.len());
    let mut key_cols: Vec<Vec<ArrayRef>> = vec![Vec::new(); n_keys];
    let mut g = 0usize; // current group index
    let mut within = 0usize; // running row index inside the current group
    let mut n = 0usize; // group total rows
    let mut nn = 0usize; // group non-null values
    let mut pos = 0.0;
    let mut lo = 0usize;
    let mut lo_t = 0usize;
    let mut hi_t = 0usize;
    let mut v_lo = 0.0;
    let mut v_hi = 0.0;
    if let Some(reader) = store.open_reader(0).map_err(InterpError::from)? {
        for batch in reader {
            let batch = batch?;
            let vcol = batch.column(n_keys).as_primitive::<Float64Type>();
            let mut firsts: Vec<u32> = Vec::new();
            for i in 0..batch.num_rows() {
                if within == 0 {
                    n = counts[g];
                    nn = n - null_counts[g];
                    if nn > 0 {
                        pos = qc * (nn - 1) as f64;
                        lo = pos.floor() as usize;
                        lo_t = null_counts[g] + lo;
                        hi_t = null_counts[g] + pos.ceil() as usize;
                    }
                    firsts.push(i as u32);
                }
                if nn > 0 {
                    // Rows `[null_count, n)` are the sorted non-null values; lo_t/hi_t
                    // index the interpolation neighbours within that suffix.
                    if within == lo_t {
                        v_lo = vcol.value(i);
                    }
                    if within == hi_t {
                        v_hi = vcol.value(i);
                    }
                }
                within += 1;
                if within == n {
                    if nn == 0 {
                        out.append_null();
                    } else {
                        out.append_value(v_lo + (v_hi - v_lo) * (pos - lo as f64));
                    }
                    g += 1;
                    within = 0;
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

    // 4. Assemble the group key columns (concat the per-batch group-first takes).
    let group_columns: Vec<ArrayRef> = (0..n_keys)
        .map(|c| -> Result<ArrayRef, InterpError> {
            if key_cols[c].is_empty() {
                Ok(arrow::array::new_empty_array(schema.field(c).data_type()))
            } else {
                let refs: Vec<&dyn Array> = key_cols[c].iter().map(|a| a.as_ref()).collect();
                Ok(concat(&refs)?)
            }
        })
        .collect::<Result<_, _>>()?;
    Ok((group_columns, Arc::new(out.finish())))
}

/// Flatten `parts` to `(g0..gN, v)` rows with the value kept in its **native** type
/// (so distinctness/equality is exact for any type — an `f64` cast would collide
/// strings etc.). Returns the flattened batches and their shared schema, or a `None`
/// schema when every input batch is empty. Shared by the bounded distinct and mode
/// paths, which need the same native-value run.
pub(super) fn flatten_native_value(
    parts: &[RecordBatch],
    group_keys: &[ProjectionItem],
    value_expr: &bc_expr::Expr,
) -> Result<(Vec<RecordBatch>, Option<Arc<Schema>>), InterpError> {
    let n_keys = group_keys.len();
    let mut flat: Vec<RecordBatch> = Vec::with_capacity(parts.len());
    let mut schema: Option<Arc<Schema>> = None;
    for part in parts {
        if part.num_rows() == 0 {
            continue;
        }
        let mut fields: Vec<Field> = Vec::with_capacity(n_keys + 1);
        let mut cols: Vec<ArrayRef> = Vec::with_capacity(n_keys + 1);
        for (i, gk) in group_keys.iter().enumerate() {
            let a = gk.expr.eval(part)?;
            fields.push(Field::new(format!("g{i}"), a.data_type().clone(), true));
            cols.push(a);
        }
        let v = value_expr.eval(part)?;
        fields.push(Field::new("v", v.data_type().clone(), true));
        cols.push(v);
        let s = Arc::new(Schema::new(fields));
        schema.get_or_insert_with(|| s.clone());
        flat.push(RecordBatch::try_new(s, cols)?);
    }
    Ok((flat, schema))
}

/// Sort keys for the native-value run: `(group asc, value asc nulls-first)`. Value
/// nulls sort first within a group, so they are skippable leading rows; equal values
/// become adjacent so a single streaming pass suffices.
pub(super) fn native_value_sort_keys(n_keys: usize) -> Vec<SortKey> {
    let mut sort_keys: Vec<SortKey> = (0..n_keys)
        .map(|i| SortKey {
            expr: bc_expr::Expr::Col {
                name: format!("g{i}"),
            },
            descending: false,
            nulls_first: false,
        })
        .collect();
    sort_keys.push(SortKey {
        expr: bc_expr::Expr::Col { name: "v".into() },
        descending: false,
        nulls_first: true,
    });
    sort_keys
}

/// Exact per-group `n_unique(value)` (COUNT DISTINCT) with bounded memory. Returns
/// the group key columns and the `Int64` distinct-count column, aligned row-wise.
/// Only valid for a single `n_unique` aggregate; the caller falls back to the
/// in-memory grace path for any other shape. Nulls are excluded (SQL semantics),
/// matching the in-memory `finalize_count_distinct`.
pub(crate) fn bounded_group_distinct(
    parts: &[RecordBatch],
    group_keys: &[ProjectionItem],
    value_expr: &bc_expr::Expr,
    dir: &std::path::Path,
) -> Result<(Vec<ArrayRef>, ArrayRef), InterpError> {
    let n_keys = group_keys.len();

    let (flat, schema) = flatten_native_value(parts, group_keys, value_expr)?;
    let Some(schema) = schema else {
        return Ok((Vec::new(), Arc::new(Int64Array::from(Vec::<i64>::new()))));
    };
    let sort_keys = native_value_sort_keys(n_keys);
    let Some(mut store) = external_sort_to_final_store(flat, &sort_keys, dir)? else {
        return Ok((
            empty_key_columns(&schema, n_keys),
            Arc::new(Int64Array::from(Vec::<i64>::new())),
        ));
    };

    let key_conv = RowConverter::new(
        (0..n_keys)
            .map(|i| SortField::new(schema.field(i).data_type().clone()))
            .collect(),
    )?;
    let val_conv = RowConverter::new(vec![SortField::new(
        schema.field(n_keys).data_type().clone(),
    )])?;

    // 3. Single pass: per contiguous group, count distinct non-null values (a value
    //    change among non-null rows). Group keys captured at each group's first row.
    let mut counts: Vec<i64> = Vec::new();
    let mut key_cols: Vec<Vec<ArrayRef>> = vec![Vec::new(); n_keys];
    let mut prev_group: Option<OwnedRow> = None;
    let mut prev_val: Option<OwnedRow> = None; // last non-null value in current group
    let mut cur = 0i64;
    let mut started = false;
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
                let new_group = !started || (n_keys > 0 && prev_group != group);
                if new_group {
                    if started {
                        counts.push(cur);
                    }
                    started = true;
                    cur = 0;
                    prev_val = None;
                    prev_group = group;
                    if n_keys > 0 {
                        firsts.push(i as u32);
                    }
                }
                if vcol.is_valid(i) {
                    let vr = vrows.row(i).owned();
                    if prev_val.as_ref() != Some(&vr) {
                        cur += 1;
                        prev_val = Some(vr);
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
        counts.push(cur);
    }

    let group_columns: Vec<ArrayRef> = (0..n_keys)
        .map(|c| -> Result<ArrayRef, InterpError> {
            if key_cols[c].is_empty() {
                Ok(arrow::array::new_empty_array(schema.field(c).data_type()))
            } else {
                let refs: Vec<&dyn Array> = key_cols[c].iter().map(|a| a.as_ref()).collect();
                Ok(concat(&refs)?)
            }
        })
        .collect::<Result<_, _>>()?;
    Ok((group_columns, Arc::new(Int64Array::from(counts))))
}

/// Exact per-group `mode(value)` with bounded memory. Returns the group key columns
/// and the mode column (the value's **native** type, null for an empty/all-null
/// group), aligned row-wise. Bit-for-bit the in-memory `finalize_mode`: the most
/// frequent non-null value per group, ties broken by the **smallest** value; nulls
/// excluded. Sorted `(group, value asc)`, so each value's run is contiguous and the
/// longest (first-seen, i.e. smallest, on a tie) run is the mode — one streaming pass.
pub(crate) fn bounded_group_mode(
    parts: &[RecordBatch],
    group_keys: &[ProjectionItem],
    value_expr: &bc_expr::Expr,
    dir: &std::path::Path,
) -> Result<(Vec<ArrayRef>, ArrayRef), InterpError> {
    use arrow::array::{new_empty_array, new_null_array};

    let n_keys = group_keys.len();
    // The output element type, knowable even from a zero-row input (eval gives a
    // typed empty array) so the empty result still carries the right type.
    let value_type = match parts.first() {
        Some(p) => value_expr.eval(p)?.data_type().clone(),
        None => DataType::Null,
    };

    let (flat, schema) = flatten_native_value(parts, group_keys, value_expr)?;
    let Some(schema) = schema else {
        return Ok((Vec::new(), new_empty_array(&value_type)));
    };
    let sort_keys = native_value_sort_keys(n_keys);
    let Some(mut store) = external_sort_to_final_store(flat, &sort_keys, dir)? else {
        return Ok((
            empty_key_columns(&schema, n_keys),
            new_empty_array(&value_type),
        ));
    };

    let key_conv = RowConverter::new(
        (0..n_keys)
            .map(|i| SortField::new(schema.field(i).data_type().clone()))
            .collect(),
    )?;
    let val_conv = RowConverter::new(vec![SortField::new(
        schema.field(n_keys).data_type().clone(),
    )])?;
    // A row encoding NULL of the value type — the winner placeholder for an
    // empty/all-null group; it never equals a (non-null) value row.
    let null_row = {
        let null_arr = new_null_array(schema.field(n_keys).data_type(), 1);
        val_conv.convert_columns(&[null_arr])?.row(0).owned()
    };

    // Per group: longest run of equal non-null values (ties → smallest). `cur_*` is
    // the run in progress, `best_*` the winner so far; both carry across batch
    // boundaries within a group. `winners` collects one value row per group.
    let mut winners: Vec<OwnedRow> = Vec::new();
    let mut key_cols: Vec<Vec<ArrayRef>> = vec![Vec::new(); n_keys];
    let mut prev_group: Option<OwnedRow> = None;
    let mut started = false;
    let mut cur_val: Option<OwnedRow> = None;
    let mut cur_len = 0usize;
    let mut best_val: Option<OwnedRow> = None;
    let mut best_len = 0usize;

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
                        // Close the previous group: fold its last run into `best`
                        // (strict `>` keeps the smaller value on a frequency tie),
                        // then emit its winner (null when it had no non-null value).
                        if cur_len > best_len {
                            best_val = cur_val.take();
                        }
                        winners.push(best_val.take().unwrap_or_else(|| null_row.clone()));
                    }
                    started = true;
                    prev_group = group;
                    cur_val = None;
                    cur_len = 0;
                    best_val = None;
                    best_len = 0;
                    if n_keys > 0 {
                        firsts.push(i as u32);
                    }
                }
                if vcol.is_valid(i) {
                    let vr = vrows.row(i).owned();
                    if cur_val.as_ref() == Some(&vr) {
                        cur_len += 1;
                    } else {
                        if cur_len > best_len {
                            best_len = cur_len;
                            best_val = cur_val.take();
                        }
                        cur_val = Some(vr);
                        cur_len = 1;
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
        if cur_len > best_len {
            best_val = cur_val.take();
        }
        winners.push(best_val.take().unwrap_or_else(|| null_row.clone()));
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
    // Decode the winner rows back into a native-type column (null rows → nulls).
    let mode_col = val_conv
        .convert_rows(winners.iter().map(|r| r.row()))?
        .into_iter()
        .next()
        .unwrap_or_else(|| new_empty_array(&value_type));
    Ok((group_columns, mode_col))
}

/// Empty arrays of the group-key types (for an all-empty input).
pub(super) fn empty_key_columns(schema: &Schema, n_keys: usize) -> Vec<ArrayRef> {
    (0..n_keys)
        .map(|c| arrow::array::new_empty_array(schema.field(c).data_type()))
        .collect()
}
