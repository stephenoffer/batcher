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

use bc_runtime::agg::spill::SpillCodec;

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
    codec: SpillCodec,
) -> Result<Option<GroupedColumns>, InterpError> {
    let Some((value_expr, q)) = single_quantile(aggregates) else {
        return Ok(None);
    };
    let dir = spill_dir.join("agg-quantile");
    let (gc, qc) = bounded_group_quantile(parts, group_keys, value_expr, q, &dir, codec)?;
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
    codec: SpillCodec,
) -> Result<Option<GroupedColumns>, InterpError> {
    let Some(value_expr) = single_distinct(aggregates) else {
        return Ok(None);
    };
    let dir = spill_dir.join("agg-distinct");
    let (gc, cc) = bounded_group_distinct(parts, group_keys, value_expr, &dir, codec)?;
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
    codec: SpillCodec,
) -> Result<Option<GroupedColumns>, InterpError> {
    let Some(value_expr) = single_mode(aggregates) else {
        return Ok(None);
    };
    let dir = spill_dir.join("agg-mode");
    let (gc, mc) = bounded_group_mode(parts, group_keys, value_expr, &dir, codec)?;
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
    codec: SpillCodec,
) -> Result<Option<GroupedColumns>, InterpError> {
    let Some(value_expr) = single_histogram(aggregates) else {
        return Ok(None);
    };
    let dir = spill_dir.join("agg-histogram");
    let (gc, mc) = histogram::bounded_group_histogram(parts, group_keys, value_expr, &dir, codec)?;
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
    codec: SpillCodec,
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
            let a = canon_float_key(&gk.expr.eval(part)?);
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

    let Some((mut store, _)) = external_sort_to_final_store(
        flat,
        &sort_keys,
        dir,
        bc_arrow::RuntimeTuning::default().sort_merge_fanin,
        crate::ops::external_sort::DEFAULT_RUN_TARGET_BYTES,
        codec,
        None,
    )?
    else {
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
                    // Compare borrowed, own only on a change. `prev` has to outlive `grows`
                    // (a group run spans batches, and `grows` is per batch), so it is an
                    // `OwnedRow` — but owning it per *row* allocated once per row to answer a
                    // question that changes once per *group*. On a spilled quantile with few
                    // groups that was an allocation for almost every row of the relation.
                    let row = grows.row(i);
                    if prev.as_ref().is_none_or(|p| p.row() != row) {
                        counts.push(0);
                        null_counts.push(0);
                        prev = Some(row.owned());
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
            let mut firsts: Vec<u32> = Vec::with_capacity(batch.num_rows());
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

/// Canonicalize a float group-key column so `-0.0`/`0.0` fold to one value and every
/// NaN bit-pattern maps to one canonical quiet NaN — the same identity the in-memory
/// `assign_groups` groups by. A non-float column is returned unchanged.
///
/// Without this the bounded out-of-core paths sort/boundary-detect on the raw
/// `RowConverter` encoding, which maps `-0.0` and `0.0` to different bytes — so a
/// GROUP BY on a float key holding both would return **two groups where the in-memory
/// grace path (and DuckDB) return one**: a silent spill-only wrong answer.
///
/// The identity itself is [`bc_arrow::canon_float_array`], the workspace's single
/// definition, so this path cannot drift from the in-memory one it must match. This
/// wrapper survives only to name *why* the spill path canonicalizes.
pub(super) fn canon_float_key(arr: &ArrayRef) -> ArrayRef {
    bc_arrow::canon_float_array(arr)
}

/// Flatten `parts` to `(g0..gN, v)` rows with the value kept in its **native** type
/// (so distinctness/equality is exact for any type — an `f64` cast would collide
/// strings etc.). Returns the flattened batches and their shared schema, or a `None`
/// schema when every input batch is empty. Shared by the bounded distinct and mode
/// paths, which need the same native-value run.
/// `canon_value` folds a `Float64` *value* column to canonical float identity
/// (`-0.0`==`0.0`, one NaN) as well. `n_unique`, `mode`, and `histogram` all need this:
/// each in-memory finalizer canonicalizes the value's float identity (`n_unique` via
/// `assign_groups`; `mode`/`histogram` via `canonicalize_float_keys` in
/// `bc_runtime::agg::median`), so the spilled path must fold identically or the same
/// column yields a different mode/histogram/distinct-count under memory pressure.
pub(super) fn flatten_native_value(
    parts: &[RecordBatch],
    group_keys: &[ProjectionItem],
    value_expr: &bc_expr::Expr,
    canon_value: bool,
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
            let a = canon_float_key(&gk.expr.eval(part)?);
            fields.push(Field::new(format!("g{i}"), a.data_type().clone(), true));
            cols.push(a);
        }
        let mut v = value_expr.eval(part)?;
        if canon_value {
            v = canon_float_key(&v);
        }
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
    codec: SpillCodec,
) -> Result<(Vec<ArrayRef>, ArrayRef), InterpError> {
    let n_keys = group_keys.len();

    // `canon_value = true`: fold a float value's `-0.0`/`0.0`/NaN so distinctness matches
    // the in-memory `assign_groups`-based dedup (which canonicalizes floats).
    let (flat, schema) = flatten_native_value(parts, group_keys, value_expr, true)?;
    let Some(schema) = schema else {
        return Ok((Vec::new(), Arc::new(Int64Array::from(Vec::<i64>::new()))));
    };
    let sort_keys = native_value_sort_keys(n_keys);
    let Some((mut store, _)) = external_sort_to_final_store(
        flat,
        &sort_keys,
        dir,
        bc_arrow::RuntimeTuning::default().sort_merge_fanin,
        crate::ops::external_sort::DEFAULT_RUN_TARGET_BYTES,
        codec,
        None,
    )?
    else {
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
            let mut firsts: Vec<u32> = Vec::with_capacity(batch.num_rows());
            // Both `prev_group` and `prev_val` must outlive this batch's `grows`/`vrows`, so
            // they are owned — but owning them per *row* allocated twice per row to answer
            // questions that change once per group and once per distinct value. Compare
            // borrowed and own only on a change.
            //
            // `vcol` is an `Arc<dyn Array>`, so its validity check was a virtual call per row
            // as well; the null buffer is resolved once here.
            let vnulls = vcol.nulls();
            for i in 0..batch.num_rows() {
                let group = grows.as_ref().map(|g| g.row(i));
                let new_group =
                    !started || (n_keys > 0 && prev_group.as_ref().map(|p| p.row()) != group);
                if new_group {
                    if started {
                        counts.push(cur);
                    }
                    started = true;
                    cur = 0;
                    prev_val = None;
                    prev_group = group.map(|r| r.owned());
                    if n_keys > 0 {
                        firsts.push(i as u32);
                    }
                }
                if vnulls.is_none_or(|n| n.is_valid(i)) {
                    let vr = vrows.row(i);
                    if prev_val.as_ref().map(|p| p.row()) != Some(vr) {
                        cur += 1;
                        prev_val = Some(vr.owned());
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
    codec: SpillCodec,
) -> Result<(Vec<ArrayRef>, ArrayRef), InterpError> {
    use arrow::array::{new_empty_array, new_null_array};

    let n_keys = group_keys.len();
    // The output element type, knowable even from a zero-row input (eval gives a
    // typed empty array) so the empty result still carries the right type.
    let value_type = match parts.first() {
        Some(p) => value_expr.eval(p)?.data_type().clone(),
        None => DataType::Null,
    };

    // `canon_value = true`: the in-memory `finalize_mode` canonicalizes float leaves
    // (`bc_runtime::agg::median::finalize_mode`), so the spill path must fold `-0.0`/`0.0`
    // and every NaN identically — else `mode` over `[-0.0, -0.0, 0.0]` differs under spill.
    let (flat, schema) = flatten_native_value(parts, group_keys, value_expr, true)?;
    let Some(schema) = schema else {
        return Ok((Vec::new(), new_empty_array(&value_type)));
    };
    let sort_keys = native_value_sort_keys(n_keys);
    let Some((mut store, _)) = external_sort_to_final_store(
        flat,
        &sort_keys,
        dir,
        bc_arrow::RuntimeTuning::default().sort_merge_fanin,
        crate::ops::external_sort::DEFAULT_RUN_TARGET_BYTES,
        codec,
        None,
    )?
    else {
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
            let mut firsts: Vec<u32> = Vec::with_capacity(batch.num_rows());
            // Same as the two passes above: compare the borrowed row, own only on a change,
            // and resolve the `dyn Array` null buffer once instead of per row.
            let vnulls = vcol.nulls();
            for i in 0..batch.num_rows() {
                let group = grows.as_ref().map(|g| g.row(i));
                if !started || (n_keys > 0 && prev_group.as_ref().map(|p| p.row()) != group) {
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
                    prev_group = group.map(|r| r.owned());
                    cur_val = None;
                    cur_len = 0;
                    best_val = None;
                    best_len = 0;
                    if n_keys > 0 {
                        firsts.push(i as u32);
                    }
                }
                if vnulls.is_none_or(|n| n.is_valid(i)) {
                    let vr = vrows.row(i);
                    if cur_val.as_ref().map(|c| c.row()) == Some(vr) {
                        cur_len += 1;
                    } else {
                        if cur_len > best_len {
                            best_len = cur_len;
                            best_val = cur_val.take();
                        }
                        cur_val = Some(vr.owned());
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

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array};
    use arrow::datatypes::{Field, Schema};
    use bc_runtime::agg::spill::SpillCodec;

    /// Build a `(k: Float64, v: Int64)` batch for the float-group-key tests.
    fn fbatch(ks: &[f64], vs: &[i64]) -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("k", DataType::Float64, true),
            Field::new("v", DataType::Int64, true),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Float64Array::from(ks.to_vec())),
                Arc::new(Int64Array::from(vs.to_vec())),
            ],
        )
        .unwrap()
    }

    fn gk() -> Vec<ProjectionItem> {
        vec![ProjectionItem {
            expr: bc_expr::Expr::Col { name: "k".into() },
            alias: "k".into(),
        }]
    }

    fn vexpr() -> bc_expr::Expr {
        bc_expr::Expr::Col { name: "v".into() }
    }

    /// A GROUP BY on a `Float64` key holding both `-0.0` and `0.0` is ONE group under
    /// SQL/DuckDB semantics (and the in-memory `assign_groups`, which canonicalizes
    /// float keys). The bounded out-of-core paths sort/boundary-detect on the raw
    /// (`RowConverter`, total-order) key, which encodes `-0.0` and `0.0` differently —
    /// so without canonicalization they split one group into two: a silent spill-only
    /// wrong answer.
    #[test]
    fn neg_zero_and_zero_are_one_group_quantile() {
        let parts = vec![fbatch(&[-0.0, 0.0, -0.0, 0.0], &[1, 3, 5, 7])];
        let dir = std::env::temp_dir().join(format!("bc_negzero_q_{}", std::process::id()));
        let (gc, _qc) =
            bounded_group_quantile(&parts, &gk(), &vexpr(), 0.5, &dir, SpillCodec::None).unwrap();
        assert_eq!(gc[0].len(), 1, "-0.0 and 0.0 must be one group");
    }

    #[test]
    fn neg_zero_and_zero_are_one_group_distinct() {
        let parts = vec![fbatch(&[-0.0, 0.0, -0.0, 0.0], &[1, 3, 5, 7])];
        let dir = std::env::temp_dir().join(format!("bc_negzero_d_{}", std::process::id()));
        let (gc, cc) =
            bounded_group_distinct(&parts, &gk(), &vexpr(), &dir, SpillCodec::None).unwrap();
        assert_eq!(gc[0].len(), 1, "-0.0 and 0.0 must be one group");
        // 4 distinct values {1,3,5,7} all in the single group.
        let cc = cc.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(cc.value(0), 4);
    }

    /// `n_unique` on a `Float64` **value** column must fold `-0.0`/`0.0` to one distinct
    /// value — the in-memory `distinct_pairs_to_list` routes a non-int value through
    /// `assign_groups`, which canonicalizes floats. The spill path deduped the raw
    /// `RowConverter` encoding (`-0.0` != `0.0`), over-counting under memory pressure.
    #[test]
    fn distinct_value_folds_signed_zero() {
        // One group (k=1.0), float values {-0.0, 0.0, 5.0} → 2 distinct ({0.0, 5.0}).
        let schema = Schema::new(vec![
            Field::new("k", DataType::Float64, true),
            Field::new("v", DataType::Float64, true),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Float64Array::from(vec![1.0, 1.0, 1.0])),
                Arc::new(Float64Array::from(vec![-0.0, 0.0, 5.0])),
            ],
        )
        .unwrap();
        let dir = std::env::temp_dir().join(format!("bc_distinct_negz_{}", std::process::id()));
        let (_gc, cc) =
            bounded_group_distinct(&[batch], &gk(), &vexpr(), &dir, SpillCodec::None).unwrap();
        let cc = cc.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(cc.value(0), 2, "-0.0 and 0.0 are one distinct value");
    }

    /// All NaN bit-patterns are one group (matching `canon_f64`/DuckDB). A single NaN
    /// column is degenerate but this pins the contract.
    #[test]
    fn nan_keys_are_one_group_quantile() {
        let nan = f64::NAN;
        let parts = vec![fbatch(&[nan, nan, nan], &[1, 2, 3])];
        let dir = std::env::temp_dir().join(format!("bc_nan_q_{}", std::process::id()));
        let (gc, _qc) =
            bounded_group_quantile(&parts, &gk(), &vexpr(), 0.5, &dir, SpillCodec::None).unwrap();
        assert_eq!(gc[0].len(), 1, "all NaN keys must be one group");
    }

    /// `mode` over a `Float64` **value** column must fold `-0.0`/`0.0` to one value — the
    /// in-memory `finalize_mode` canonicalizes float leaves. The spill path once passed
    /// `canon_value = false`, so `mode([-0.0, -0.0, 0.0])` returned `-0.0` (a spurious
    /// 2-vs-1 split) under memory pressure while the in-memory answer is `0.0` (count 3):
    /// a silent spill-only wrong answer that violates single-node==spilled.
    #[test]
    fn mode_value_folds_signed_zero() {
        let schema = Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("v", DataType::Float64, true),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int64Array::from(vec![1, 1, 1])),
                Arc::new(Float64Array::from(vec![-0.0, -0.0, 0.0])),
            ],
        )
        .unwrap();
        let gk = vec![ProjectionItem {
            expr: bc_expr::Expr::Col { name: "k".into() },
            alias: "k".into(),
        }];
        let dir = std::env::temp_dir().join(format!("bc_mode_negz_{}", std::process::id()));
        let (_gc, mc) =
            bounded_group_mode(&[batch], &gk, &vexpr(), &dir, SpillCodec::None).unwrap();
        let mc = mc.as_any().downcast_ref::<Float64Array>().unwrap();
        // Canonical representative of the single folded value is `+0.0`.
        assert_eq!(
            mc.value(0).to_bits(),
            0.0_f64.to_bits(),
            "mode folds signed zero"
        );
    }
}
