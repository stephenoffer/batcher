//! Bounded out-of-core aggregation for a *mix* of value-list and constant-state
//! aggregates in one `GROUP BY`.
//!
//! A lone value-list aggregate (`median`/`n_unique`/`mode`) gets a bounded
//! external-sort path ([`super::quantile_spill`]); but `median(x), sum(y)` — a
//! value-list aggregate mixed with constant-state ones — falls to the in-memory
//! grace path, whose per-group value list can still blow memory on a hot key. This
//! bounds the mixed case **compositionally**, reusing the proven primitives:
//!
//! * each value-list aggregate is computed by its own bounded external sort (sorted
//!   by *its* value column, so the per-group list is never resident);
//! * the constant-state aggregates (sum/min/max/count/mean/…) are computed by the
//!   grace path, whose state is one accumulator per group (already memory-frugal);
//! * the results — which share the same group set — are **merge-aligned by group
//!   key** (each sorted by its group columns, then zipped) and reassembled into the
//!   original aggregate order.
//!
//! The result is identical to the in-memory grace path (the seq==spill oracle and a
//! randomized fuzz test pin this); only peak memory differs. `array_agg` (output *is*
//! the list) and `histogram` (its bounded path is private to `quantile_spill`) keep
//! the grace path — this returns `None` for them so the caller falls back.

use std::path::Path;

use arrow::array::{Array, ArrayRef, RecordBatch, UInt32Array};
use arrow::compute::{lexsort_to_indices, take, SortColumn, SortOptions};
use bc_ir::{AggFunc, AggregateItem, ProjectionItem};
use bc_runtime::agg;
use bc_runtime::agg::spill::{combine_finalize_spilling, DiskSpillStore};

use super::quantile_spill::{bounded_group_distinct, bounded_group_mode, bounded_group_quantile};
use super::{agg_funcs, eval_partial};
use crate::error::InterpError;

/// Finalized group-key columns paired with the finalized aggregate columns.
type GroupedColumns = (Vec<ArrayRef>, Vec<ArrayRef>);

/// One path's result: the original aggregate indices it covers, its group columns,
/// and its value columns (parallel to the indices).
type PathResult = (Vec<usize>, Vec<ArrayRef>, Vec<ArrayRef>);

/// A value-list aggregate this module can bound via an external sort (excludes
/// `Histogram`, whose bounded path is private to `quantile_spill`, and `ListAgg`,
/// whose output *is* the value list).
fn bounded_value_list(func: AggFunc) -> bool {
    matches!(
        func,
        AggFunc::Median | AggFunc::Quantile | AggFunc::CountDistinct | AggFunc::Mode
    )
}

/// Bound a mixed grouped aggregate out of core, or `None` to fall back to grace.
///
/// Returns `None` (caller uses the in-memory grace path) unless there are ≥2
/// aggregates including ≥1 *bounded* value-list aggregate and **every** value-list
/// aggregate present is boundable (a `ListAgg`/`Histogram` forces the fallback).
pub(crate) fn try_bounded_mixed_spill(
    parts: &[RecordBatch],
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    spill_dir: &Path,
    budget_bytes: usize,
) -> Result<Option<GroupedColumns>, InterpError> {
    if aggregates.len() < 2 {
        return Ok(None); // a lone aggregate is handled by the single-aggregate paths
    }
    let mut has_bounded_vl = false;
    for a in aggregates {
        if bounded_value_list(a.func) {
            has_bounded_vl = true;
        } else if matches!(a.func, AggFunc::ListAgg | AggFunc::Histogram) {
            return Ok(None); // a value-list aggregate we can't bound → grace
        }
    }
    if !has_bounded_vl {
        return Ok(None); // all constant-state → grace already bounds it (per-group accumulators)
    }

    // Each value-list aggregate via its own bounded external sort.
    let mut results: Vec<PathResult> = Vec::new();
    for (i, a) in aggregates.iter().enumerate() {
        if !bounded_value_list(a.func) {
            continue;
        }
        let Some(value_expr) = a.input.as_ref() else {
            return Ok(None); // defensive: a value-list aggregate without an input
        };
        let dir = spill_dir.join(format!("mixed-vl-{i}"));
        let (gc, col) = match a.func {
            AggFunc::Median => bounded_group_quantile(parts, group_keys, value_expr, 0.5, &dir)?,
            AggFunc::Quantile => {
                let q = a.param.unwrap_or(0.5);
                bounded_group_quantile(parts, group_keys, value_expr, q, &dir)?
            }
            AggFunc::CountDistinct => bounded_group_distinct(parts, group_keys, value_expr, &dir)?,
            AggFunc::Mode => bounded_group_mode(parts, group_keys, value_expr, &dir)?,
            _ => unreachable!("bounded_value_list gate"),
        };
        results.push((vec![i], gc, vec![col]));
    }

    // The constant-state aggregates via the grace path (one accumulator per group).
    let cs_idx: Vec<usize> = aggregates
        .iter()
        .enumerate()
        .filter(|(_, a)| !bounded_value_list(a.func))
        .map(|(i, _)| i)
        .collect();
    if !cs_idx.is_empty() {
        let cs_aggs: Vec<AggregateItem> = cs_idx.iter().map(|&i| aggregates[i].clone()).collect();
        let funcs = agg_funcs(&cs_aggs);
        let partials: Vec<agg::Partial> = parts
            .iter()
            .map(|b| eval_partial(b, group_keys, &cs_aggs))
            .collect::<Result<_, _>>()?;
        let p = cs_partitions(&partials, budget_bytes);
        let mut store = DiskSpillStore::new(spill_dir.join("mixed-cs"), p)?;
        let res = combine_finalize_spilling(partials, &funcs, &mut store)?;
        results.push((cs_idx, res.group_columns, res.agg_columns));
    }

    Ok(Some(align_and_assemble(results, aggregates.len())?))
}

/// Grace fan-out for the constant-state sub-aggregate: enough partitions that each
/// holds ~one budget of per-group accumulator state (mirrors `par::grace_partitions`).
fn cs_partitions(partials: &[agg::Partial], budget_bytes: usize) -> usize {
    let total: usize = partials
        .iter()
        .map(|p| {
            let g: usize = p
                .group_columns
                .iter()
                .map(|c| c.get_array_memory_size())
                .sum();
            let s: usize = p
                .states
                .iter()
                .flat_map(|x| x.iter())
                .map(|c| c.get_array_memory_size())
                .sum();
            g + s
        })
        .sum();
    total.div_ceil(budget_bytes.max(1)).max(2)
}

/// Merge-align every path's result by group key and reassemble the aggregate columns
/// into the original order.
///
/// Each path produced the *same* set of groups (same `GROUP BY` over the same input)
/// but in its own row order. Sorting each path's columns by *its* group keys with the
/// same options yields one shared row order — each group is unique within a result,
/// so the sort is a total order over identical key sets and row `i` is the same group
/// in every path. Group columns are taken from the first path; each value column is
/// slotted back into its aggregate's original position.
fn align_and_assemble(
    results: Vec<PathResult>,
    n_aggs: usize,
) -> Result<GroupedColumns, InterpError> {
    let mut sorted: Vec<PathResult> = Vec::with_capacity(results.len());
    for (idx, gc, vc) in results {
        // With GROUP BY keys, sort each path by them into one shared row order. A
        // global aggregate (no GROUP BY) has a single group and no key to sort on —
        // its one row is already aligned across paths — so use the identity
        // permutation (sorting zero columns is an error). Length comes from the value
        // columns, which every path carries.
        let perm = if gc.is_empty() {
            let n = vc.first().map_or(0, |c| c.len());
            UInt32Array::from_iter_values(0..n as u32)
        } else {
            sort_perm(&gc)?
        };
        let gc_s = take_all(&gc, &perm)?;
        let vc_s = take_all(&vc, &perm)?;
        sorted.push((idx, gc_s, vc_s));
    }

    // Row count from the value columns — present in every path (grouped or global),
    // unlike the group columns which are empty for a global aggregate.
    let nrows = sorted
        .first()
        .and_then(|(_, _, vc)| vc.first())
        .map_or(0, |c| c.len());
    // Defensive: the paths must agree on the group set (they group identically). A
    // disagreement would misalign columns, so fail loudly rather than emit a wrong row.
    for (_, _, vc) in &sorted {
        let r = vc.first().map_or(0, |c| c.len());
        if r != nrows {
            return Err(InterpError::MixedAggregateGroupMismatch {
                expected: nrows,
                found: r,
            });
        }
    }

    let group_cols = sorted
        .first()
        .map(|(_, gc, _)| gc.clone())
        .unwrap_or_default();
    let mut agg_cols: Vec<Option<ArrayRef>> = vec![None; n_aggs];
    for (idx, _, vc) in &sorted {
        for (j, &orig) in idx.iter().enumerate() {
            agg_cols[orig] = Some(vc[j].clone());
        }
    }
    // Every aggregate is covered (value-list ones individually, the rest by the grace
    // path), so no slot is left empty.
    let agg_cols: Vec<ArrayRef> = agg_cols
        .into_iter()
        .map(|c| c.expect("every aggregate column is produced by one path"))
        .collect();
    Ok((group_cols, agg_cols))
}

/// A stable lexicographic sort permutation over `group_cols` (ascending, nulls first).
fn sort_perm(group_cols: &[ArrayRef]) -> Result<UInt32Array, InterpError> {
    let cols: Vec<SortColumn> = group_cols
        .iter()
        .map(|c| SortColumn {
            values: c.clone(),
            options: Some(SortOptions {
                descending: false,
                nulls_first: true,
            }),
        })
        .collect();
    Ok(lexsort_to_indices(&cols, None)?)
}

/// Apply a row permutation to every column.
fn take_all(cols: &[ArrayRef], perm: &UInt32Array) -> Result<Vec<ArrayRef>, InterpError> {
    cols.iter()
        .map(|c| take(c, perm, None).map_err(InterpError::from))
        .collect()
}
