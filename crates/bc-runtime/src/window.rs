//! Window functions — partition, order, and append one column per function.
//!
//! Rows are partitioned by the partition keys (an empty key list means a single
//! partition over all rows). Within each partition rows are ordered by the order
//! keys. Supported function families:
//!
//! * **Ranking** (`row_number`/`rank`/`dense_rank`) — assigned from the order.
//! * **Aggregates** (`sum`/`avg`/`min`/`max`/`count`) — *whole-partition* when
//!   there is no ORDER BY (one value broadcast to every row), or a *running*
//!   (cumulative) aggregate over the ordered partition when an ORDER BY is
//!   present, with `RANGE` peer semantics (tied rows share the end-of-peer-group
//!   value) — matching SQL's default frame.
//! * **Value** (`first_value`/`last_value`/`lag`/`lead`) — select another row's
//!   value by position within the ordered partition (type-generic via `take`).
//!
//! Each output column is produced in ORIGINAL row order (results are scattered
//! back to the row positions they came from). Partitioning reuses arrow's row
//! format (`RowConverter`) like `agg.rs`, and intra-partition ordering reuses
//! `lexsort_to_indices`; the typed accumulation mirrors `agg.rs`.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, Float64Array, Int64Array, StringArray, UInt32Array};
use arrow::compute::{lexsort_to_indices, take, SortColumn, SortOptions};
use arrow::datatypes::{DataType, Float64Type, Int64Type};
use arrow::row::{RowConverter, Rows, SortField};

use crate::error::RuntimeError;

/// Above this row count, independent per-partition work (sorting each partition)
/// is spread across cores. Below it, the single-threaded path avoids pool overhead
/// so small windows stay sub-millisecond.
const PARALLEL_ROW_THRESHOLD: usize = 1 << 15;

/// A window function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFn {
    RowNumber,
    Rank,
    DenseRank,
    /// `(rank - 1) / (rows - 1)`; `0` for a single-row partition. → Float64.
    PercentRank,
    /// Fraction of partition rows at or before the current row's peer group. → Float64.
    CumeDist,
    /// Distribute the ordered partition into `offset` buckets numbered `1..=offset`,
    /// as evenly as possible (earlier buckets absorb the remainder). → Int64.
    Ntile,
    Sum,
    Avg,
    Min,
    Max,
    Count,
    /// Value of the first row of the partition (in order).
    FirstValue,
    /// Value of the last row of the partition (in order).
    LastValue,
    /// Value `offset` rows before the current row (null if out of range).
    Lag,
    /// Value `offset` rows after the current row (null if out of range).
    Lead,
    /// Value of the `offset`-th row (1-based) of the partition in order; null if the
    /// partition has fewer than `offset` rows (SQL `nth_value`).
    NthValue,
    /// Nearest non-null value at or before the current row of the ordered partition.
    ForwardFill,
    /// Nearest non-null value at or after the current row of the ordered partition.
    BackwardFill,
}

impl WindowFn {
    pub(crate) fn name(self) -> &'static str {
        match self {
            WindowFn::RowNumber => "row_number",
            WindowFn::Rank => "rank",
            WindowFn::DenseRank => "dense_rank",
            WindowFn::PercentRank => "percent_rank",
            WindowFn::CumeDist => "cume_dist",
            WindowFn::Ntile => "ntile",
            WindowFn::Sum => "sum",
            WindowFn::Avg => "avg",
            WindowFn::Min => "min",
            WindowFn::Max => "max",
            WindowFn::Count => "count",
            WindowFn::FirstValue => "first_value",
            WindowFn::LastValue => "last_value",
            WindowFn::Lag => "lag",
            WindowFn::Lead => "lead",
            WindowFn::NthValue => "nth_value",
            WindowFn::ForwardFill => "forward_fill",
            WindowFn::BackwardFill => "backward_fill",
        }
    }

    /// Reducing aggregate functions (one value per partition). Without an ORDER BY or
    /// explicit frame these are whole-partition aggregates — computable as a plain
    /// group-by broadcast (the fast path in [`window_with`]).
    fn is_aggregate(self) -> bool {
        matches!(
            self,
            WindowFn::Sum | WindowFn::Avg | WindowFn::Min | WindowFn::Max | WindowFn::Count
        )
    }

    /// Positional "value" functions select a row's value by offset rather than
    /// reducing the partition; they preserve the input column's type. The fills select
    /// by *nullness* rather than by offset, but share the same take-based shape.
    fn is_value(self) -> bool {
        matches!(
            self,
            WindowFn::FirstValue
                | WindowFn::LastValue
                | WindowFn::Lag
                | WindowFn::Lead
                | WindowFn::NthValue
                | WindowFn::ForwardFill
                | WindowFn::BackwardFill
        )
    }

    /// The null-carrying functions (`forward_fill`/`backward_fill`).
    pub(crate) fn is_fill(self) -> bool {
        matches!(self, WindowFn::ForwardFill | WindowFn::BackwardFill)
    }
}

/// One window function to compute: a function and its (optional) pre-evaluated
/// input. Ranking functions ignore `values`; aggregates and value functions
/// require it. `offset` is the lag/lead distance (ignored by other functions).
pub struct WindowCall {
    pub func: WindowFn,
    pub values: Option<ArrayRef>,
    pub offset: i64,
    /// Explicit `ROWS` frame; `None` is the default `RANGE …` frame. Honored only
    /// by the aggregate functions.
    pub frame: Option<crate::window_frame::Frame>,
}

/// Compute every window function over the partitioned/ordered input, returning
/// one output column per call, each in ORIGINAL row order.
///
/// `partition_keys` (possibly empty → one partition), `order_keys` as
/// `(array, options)` pairs, `funcs` the calls, and `num_rows` the input length.
pub fn window(
    partition_keys: &[ArrayRef],
    order_keys: &[(ArrayRef, SortOptions)],
    funcs: &[WindowCall],
    num_rows: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    window_with(
        partition_keys,
        order_keys,
        funcs,
        num_rows,
        PARALLEL_ROW_THRESHOLD,
    )
}

/// [`window`] with the parallel-row threshold supplied by the caller.
///
/// `parallel_row_threshold` is a performance-only knob: above it the independent
/// per-partition sorts run across cores, below it single-threaded. The output
/// columns are identical regardless of the choice.
pub fn window_with(
    partition_keys: &[ArrayRef],
    order_keys: &[(ArrayRef, SortOptions)],
    funcs: &[WindowCall],
    num_rows: usize,
    parallel_row_threshold: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    // Parallelize across CORES by hash-partitioning rows on the PARTITION BY keys:
    // equal keys co-partition, so every window partition lands wholly inside one
    // bucket and each bucket is an independent window over a disjoint row set. Run the
    // serial kernel per bucket on rayon, then scatter every function's column back to
    // original row order. This spreads the dominant per-partition cost — the ordering
    // sort behind ranking / running-aggregate / value / framed functions — across all
    // cores (the serial kernel left it single-threaded).
    //
    // Parallelized for two shapes: (1) an ORDER BY window (each partition pays a sort
    // that dwarfs the partition + scatter-back plumbing), and (2) a *frameless
    // whole-partition aggregate* (`sum() OVER (PARTITION BY k)`, empty order). The
    // frameless case is a group-by + broadcast, and on a high-cardinality key its single
    // pass over the whole input hits a cache-cold million-entry hash table — bucketing by
    // key across cores makes each bucket's table cache-resident and the assign/broadcast
    // parallel (measured ~340 ms → tens of ms at 6 M rows / 1.5 M groups). Skip below the
    // row threshold and when there is no PARTITION BY (a single global partition can't be
    // split; the serial group-id fast path handles it). A frameless window that mixes a
    // non-aggregate (a value function with no order) also stays serial.
    let nthreads = rayon::current_num_threads();
    let frameless_agg = order_keys.is_empty()
        && funcs
            .iter()
            .all(|c| c.frame.is_none() && c.func.is_aggregate());
    let worth_parallel = !partition_keys.is_empty()
        && (!order_keys.is_empty() || frameless_agg)
        && num_rows >= parallel_row_threshold
        && nthreads > 1;
    if !worth_parallel {
        return window_serial(partition_keys, order_keys, funcs, num_rows);
    }
    crate::window_parallel::window_parallel(partition_keys, order_keys, funcs, num_rows, nthreads)
}

pub(crate) fn window_serial(
    partition_keys: &[ArrayRef],
    order_keys: &[(ArrayRef, SortOptions)],
    funcs: &[WindowCall],
    num_rows: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    // Fast path: PARTITION BY with no ORDER BY and only plain aggregates (no frame, no
    // ranking / value functions) is exactly "group-by the partition keys, aggregate,
    // broadcast back to each row". Assign dense group ids once via the shared native
    // fast paths (`agg::assign_groups`) and reduce + broadcast in linear passes — far
    // cheaper than materializing per-partition index lists and gathering by scattered
    // index, and it parallelizes the grouping. The result is identical (the order within
    // a partition never affects a whole-partition aggregate).
    if order_keys.is_empty()
        && !partition_keys.is_empty()
        && funcs
            .iter()
            .all(|c| c.frame.is_none() && c.func.is_aggregate())
    {
        let (group_ids, num_groups, _) = crate::agg::assign_groups(partition_keys, num_rows)?;
        return funcs
            .iter()
            .map(|call| {
                crate::window_partition_agg::broadcast_partition_aggregate(
                    call.func,
                    &group_ids,
                    num_groups,
                    call.values.as_ref(),
                )
            })
            .collect();
    }

    // Build the per-partition ordered row-index lists (`ordered`) and — only for the
    // no-ORDER-BY aggregate broadcast below — the unordered partitions.
    //
    // With order keys, a *single* lexsort by `(partition_keys ++ order_keys)` yields the
    // rows grouped by partition (partition keys lead the sort) and ordered within each by
    // the order keys, so splitting the sorted indices into contiguous partition runs gives
    // exactly the same `ordered` structure that grouping-then-sorting-each-partition does —
    // but from one big sort instead of a million tiny per-partition sorts (a
    // near-unique PARTITION BY like `l_orderkey` produces ~1.5 M 4-row partitions, whose
    // per-partition sort + index-list overhead dominated). Peer/tie semantics are unchanged
    // (ties on the order keys are still peers via `order_rows`), so every ranking / running
    // / value function computes an identical per-row result. Without order keys there is
    // nothing to sort, so just group.
    let partitions: Vec<Vec<usize>>;
    let ordered: Vec<Vec<usize>>;
    if order_keys.is_empty() {
        partitions = assign_partitions(partition_keys, num_rows)?;
        ordered = partitions.clone();
    } else {
        ordered = ordered_partitions_by_global_sort(partition_keys, order_keys, num_rows)?;
        partitions = Vec::new(); // unused when order keys are present (see the match below)
    }

    // Encode the order keys once into arrow's row format. Peer/tie checks then cost
    // one byte comparison by row index instead of re-encoding per comparison.
    let order_rows = if order_keys.is_empty() {
        None
    } else {
        Some(encode_order_keys(order_keys)?)
    };

    let mut out = Vec::with_capacity(funcs.len());
    for call in funcs {
        out.push(match call.func {
            WindowFn::RowNumber => row_number(&ordered, num_rows),
            WindowFn::Rank => rank(&ordered, order_rows.as_ref(), num_rows, false)?,
            WindowFn::DenseRank => rank(&ordered, order_rows.as_ref(), num_rows, true)?,
            WindowFn::PercentRank => percent_rank(&ordered, order_rows.as_ref(), num_rows)?,
            WindowFn::CumeDist => cume_dist(&ordered, order_rows.as_ref(), num_rows)?,
            WindowFn::Ntile => ntile(&ordered, call.offset, num_rows),
            // first_value/last_value/lag/lead select a row's value by position.
            f if f.is_value() => value_window(
                f,
                &ordered,
                require(call.values.as_ref(), f)?,
                call.offset,
                num_rows,
            )?,
            // An explicit ROWS frame aggregates the physical rows in [start, end]
            // of the ordered partition (overrides the default running/whole frame).
            f if call.frame.is_some() => crate::window_frame::framed_aggregate(
                f,
                &ordered,
                require(call.values.as_ref(), f)?,
                call.frame.expect("frame present"),
                order_rows.as_ref(),
                num_rows,
            )?,
            // With an ORDER BY, an aggregate window is a *running* (cumulative)
            // aggregate over the ordered partition; without one it's whole-partition.
            f if order_keys.is_empty() => crate::window_partition_agg::partition_aggregate(
                f,
                &partitions,
                call.values.as_ref(),
                num_rows,
            )?,
            f => running_aggregate(
                f,
                &ordered,
                order_rows.as_ref().expect("order keys present"),
                call.values.as_ref(),
                num_rows,
            )?,
        });
    }
    Ok(out)
}

/// Positional value functions (`first_value`/`last_value`/`lag`/`lead`). Each
/// output row selects another row's value by position within its ordered
/// partition, so the result is type-generic: we build a per-row source-index map
/// (with nulls for out-of-range) and `take` from the input column.
fn value_window(
    func: WindowFn,
    ordered: &[Vec<usize>],
    values: &ArrayRef,
    offset: i64,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    if func.is_fill() {
        return crate::window_fill::fill_window(func, ordered, values, num_rows);
    }
    let off = offset.max(0) as usize;
    let mut src: Vec<Option<u32>> = vec![None; num_rows];
    for part in ordered {
        let len = part.len();
        for (pos, &row) in part.iter().enumerate() {
            let take_pos = match func {
                WindowFn::FirstValue => Some(0),
                WindowFn::LastValue => Some(len - 1),
                WindowFn::Lag => pos.checked_sub(off),
                WindowFn::Lead => (pos + off < len).then_some(pos + off),
                // nth_value: the `off`-th row (1-based), same for every row of the
                // partition; null if the partition is shorter than `off`.
                WindowFn::NthValue => (off >= 1 && off <= len).then_some(off - 1),
                _ => unreachable!("value_window on non-value/non-fill function"),
            };
            src[row] = take_pos.map(|p| part[p] as u32);
        }
    }
    Ok(take(values.as_ref(), &UInt32Array::from(src), None)?)
}

/// Group row indices by partition key. Empty keys → one partition of all rows.
fn assign_partitions(
    partition_keys: &[ArrayRef],
    num_rows: usize,
) -> Result<Vec<Vec<usize>>, RuntimeError> {
    if partition_keys.is_empty() {
        return Ok(vec![(0..num_rows).collect()]);
    }
    let fields: Vec<SortField> = partition_keys
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    let rows = converter.convert_columns(partition_keys)?;

    // Key partitions by the *borrowed* row bytes (`Row: Hash + Eq`) — owning a key
    // per row would be a million allocations on a million-row window. First-seen
    // order is irrelevant: results scatter back to original positions by index.
    let mut index: hashbrown::HashMap<arrow::row::Row<'_>, usize> = hashbrown::HashMap::new();
    let mut partitions: Vec<Vec<usize>> = Vec::new();
    for i in 0..num_rows {
        let next = partitions.len();
        let pid = *index.entry(rows.row(i)).or_insert(next);
        if pid == next {
            partitions.push(Vec::new());
        }
        partitions[pid].push(i);
    }
    Ok(partitions)
}

/// Per-partition, order-sorted row-index lists built from ONE global lexsort.
///
/// Sorts all rows by `(partition_keys ascending, then order_keys with their options)`.
/// Partition keys leading the sort make each partition's rows contiguous in the sorted
/// order, and the order keys sort within; splitting the sorted indices at partition-key
/// boundaries therefore yields each partition's rows in order-key order — identical to
/// grouping first then sorting each partition, but with a single sort instead of one per
/// (often tiny) partition. Callers use this only when `order_keys` is non-empty.
fn ordered_partitions_by_global_sort(
    partition_keys: &[ArrayRef],
    order_keys: &[(ArrayRef, SortOptions)],
    num_rows: usize,
) -> Result<Vec<Vec<usize>>, RuntimeError> {
    // Sort columns: partition keys first (ascending — grouping is order-agnostic), then
    // the order keys with their own ASC/DESC + nulls placement, and finally the original
    // row index as the last ascending key. The index tie-break makes the sort a *total*
    // order, so rows tied on the partition + order keys land in original-row order — which
    // (a) is a stable, deterministic choice for `row_number`'s otherwise-unspecified order
    // among peers, and (b) is identical whether this runs over the whole input or over one
    // hash bucket of it, so the parallel per-bucket path matches the serial kernel exactly.
    let mut sort_columns: Vec<SortColumn> = partition_keys
        .iter()
        .map(|a| SortColumn {
            values: a.clone(),
            options: None,
        })
        .collect();
    for (arr, opts) in order_keys {
        sort_columns.push(SortColumn {
            values: arr.clone(),
            options: Some(*opts),
        });
    }
    let row_index: ArrayRef = Arc::new(Int64Array::from_iter_values(0..num_rows as i64));
    sort_columns.push(SortColumn {
        values: row_index,
        options: None,
    });
    let sorted = lexsort_to_indices(&sort_columns, None)?;
    let sorted = sorted.values();

    // No PARTITION BY: every row is one partition, already globally ordered by the sort.
    if partition_keys.is_empty() {
        return Ok(vec![sorted.iter().map(|&r| r as usize).collect()]);
    }

    // Encode the partition keys once so a run boundary is one byte-row comparison by index
    // (nulls compare equal here, so a null partition key forms one partition — SQL).
    let pfields: Vec<SortField> = partition_keys
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect();
    let pconv = RowConverter::new(pfields)?;
    let prows = pconv.convert_columns(partition_keys)?;

    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::with_capacity(0);
    for (pos, &ri) in sorted.iter().enumerate() {
        let ri = ri as usize;
        if pos > 0 {
            let prev = sorted[pos - 1] as usize;
            if prows.row(ri) != prows.row(prev) {
                out.push(std::mem::take(&mut cur));
            }
        }
        cur.push(ri);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    Ok(out)
}

/// `row_number`: 1..n in order, unique per row. Scattered to original positions.
fn row_number(ordered: &[Vec<usize>], num_rows: usize) -> ArrayRef {
    let mut out = vec![0i64; num_rows];
    for part in ordered {
        for (rank0, &row) in part.iter().enumerate() {
            out[row] = rank0 as i64 + 1;
        }
    }
    Arc::new(Int64Array::from(out))
}

/// `rank` (gaps, ties share min) or `dense_rank` (no gaps, ties share). Ties are
/// rows that compare equal on every order key.
fn rank(
    ordered: &[Vec<usize>],
    order_rows: Option<&Rows>,
    num_rows: usize,
    dense: bool,
) -> Result<ArrayRef, RuntimeError> {
    let Some(rows) = order_rows else {
        return Err(RuntimeError::WindowRequiresOrder {
            func: if dense { "dense_rank" } else { "rank" }.to_string(),
        });
    };
    let mut out = vec![0i64; num_rows];
    for part in ordered {
        let mut current = 0i64; // last assigned rank
        for (pos, &row) in part.iter().enumerate() {
            let tie = pos > 0 && rows_equal(rows, part[pos - 1], row);
            if pos == 0 {
                current = 1;
            } else if !tie {
                current = if dense { current + 1 } else { pos as i64 + 1 };
            }
            out[row] = current;
        }
    }
    Ok(Arc::new(Int64Array::from(out)))
}

/// `percent_rank`: `(rank - 1) / (rows - 1)` over the ordered partition, where
/// `rank` is the gaps-after-ties RANK; a single-row partition is `0`. Requires
/// order keys (the rank is otherwise undefined). → Float64.
fn percent_rank(
    ordered: &[Vec<usize>],
    order_rows: Option<&Rows>,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    let Some(rows) = order_rows else {
        return Err(RuntimeError::WindowRequiresOrder {
            func: "percent_rank".to_string(),
        });
    };
    let mut out = vec![0f64; num_rows];
    for part in ordered {
        let n = part.len();
        let mut current = 0i64; // last assigned RANK (1-based, gaps after ties)
        for (pos, &row) in part.iter().enumerate() {
            let tie = pos > 0 && rows_equal(rows, part[pos - 1], row);
            if pos == 0 {
                current = 1;
            } else if !tie {
                current = pos as i64 + 1;
            }
            out[row] = if n > 1 {
                (current - 1) as f64 / (n - 1) as f64
            } else {
                0.0
            };
        }
    }
    Ok(Arc::new(Float64Array::from(out)))
}

/// `cume_dist`: the fraction of partition rows at or before the current row's peer
/// group — `(rows through end of peer group) / partition rows`. Tied rows share the
/// value at the end of their peer group. Requires order keys. → Float64.
fn cume_dist(
    ordered: &[Vec<usize>],
    order_rows: Option<&Rows>,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    let Some(rows) = order_rows else {
        return Err(RuntimeError::WindowRequiresOrder {
            func: "cume_dist".to_string(),
        });
    };
    let mut out = vec![0f64; num_rows];
    for part in ordered {
        let n = part.len();
        let mut group_start = 0usize;
        for pos in 0..n {
            if peer_boundary(part, rows, pos) {
                let cd = (pos + 1) as f64 / n as f64;
                for j in group_start..=pos {
                    out[part[j]] = cd;
                }
                group_start = pos + 1;
            }
        }
    }
    Ok(Arc::new(Float64Array::from(out)))
}

/// `ntile(buckets)`: distribute the ordered partition into `buckets` groups numbered
/// `1..=buckets`, as evenly as possible — the first `n % buckets` buckets take one
/// extra row. With fewer rows than buckets each row is its own bucket. → Int64.
fn ntile(ordered: &[Vec<usize>], buckets: i64, num_rows: usize) -> ArrayRef {
    let b = buckets.max(1) as usize;
    let mut out = vec![0i64; num_rows];
    for part in ordered {
        let n = part.len();
        let base = n / b; // minimum rows per bucket
        let rem = n % b; // earlier buckets absorb the remainder
        let mut idx = 0usize;
        for bucket in 0..b {
            let size = base + usize::from(bucket < rem);
            for _ in 0..size {
                out[part[idx]] = bucket as i64 + 1;
                idx += 1;
            }
        }
    }
    Arc::new(Int64Array::from(out))
}

/// Encode the order-key columns once into arrow's row format, so peer/tie checks
/// are an O(1) byte comparison by row index rather than a per-comparison re-encode.
/// Sort direction is irrelevant to *equality*, so default `SortField`s are used
/// (the row encoding is injective: equal encodings iff equal values, nulls
/// included).
fn encode_order_keys(order_keys: &[(ArrayRef, SortOptions)]) -> Result<Rows, RuntimeError> {
    let cols: Vec<ArrayRef> = order_keys.iter().map(|(a, _)| a.clone()).collect();
    let fields: Vec<SortField> = cols
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    Ok(converter.convert_columns(&cols)?)
}

/// Whether two original row indices compare equal on every order key (peers),
/// using the pre-encoded order rows.
fn rows_equal(rows: &Rows, a: usize, b: usize) -> bool {
    rows.row(a) == rows.row(b)
}

/// Running aggregate over an ordered partition (the default `RANGE UNBOUNDED
/// PRECEDING TO CURRENT ROW` frame): each row sees the accumulation of all rows
/// up to and including it in sort order. Tied rows (equal on every order key)
/// share the value at the end of their peer group, matching SQL `RANGE`.
fn running_aggregate(
    func: WindowFn,
    ordered: &[Vec<usize>],
    order_rows: &Rows,
    values: Option<&ArrayRef>,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    let values = require(values, func)?;
    if func == WindowFn::Count {
        let mut out = vec![0i64; num_rows];
        for part in ordered {
            let (mut acc, mut gs) = (0i64, 0usize);
            for pos in 0..part.len() {
                if values.is_valid(part[pos]) {
                    acc += 1;
                }
                if peer_boundary(part, order_rows, pos) {
                    for j in gs..=pos {
                        out[part[j]] = acc;
                    }
                    gs = pos + 1;
                }
            }
        }
        return Ok(Arc::new(Int64Array::from(out)));
    }
    match values.data_type() {
        DataType::Int64 => running_numeric_i64(func, ordered, order_rows, values, num_rows),
        DataType::Float64 => running_numeric_f64(func, ordered, order_rows, values, num_rows),
        DataType::Utf8 if matches!(func, WindowFn::Min | WindowFn::Max) => {
            running_str_minmax(func, ordered, order_rows, values, num_rows)
        }
        other => Err(RuntimeError::UnsupportedWindow {
            func: func.name().to_string(),
            dtype: other.to_string(),
        }),
    }
}

/// True if `pos` is the last row of its peer group (next row differs on the
/// order keys, or it's the partition's last row).
fn peer_boundary(part: &[usize], order_rows: &Rows, pos: usize) -> bool {
    pos + 1 == part.len() || !rows_equal(order_rows, part[pos], part[pos + 1])
}

fn running_numeric_i64(
    func: WindowFn,
    ordered: &[Vec<usize>],
    order_rows: &Rows,
    values: &ArrayRef,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    let arr = values.as_primitive::<Int64Type>();
    if func == WindowFn::Avg {
        let mut out: Vec<Option<f64>> = vec![None; num_rows];
        for part in ordered {
            let (mut sum, mut cnt, mut gs) = (0f64, 0i64, 0usize);
            for pos in 0..part.len() {
                if arr.is_valid(part[pos]) {
                    sum += arr.value(part[pos]) as f64;
                    cnt += 1;
                }
                if peer_boundary(part, order_rows, pos) {
                    let v = (cnt > 0).then(|| sum / cnt as f64);
                    for j in gs..=pos {
                        out[part[j]] = v;
                    }
                    gs = pos + 1;
                }
            }
        }
        return Ok(Arc::new(Float64Array::from(out)));
    }
    let mut out: Vec<Option<i64>> = vec![None; num_rows];
    for part in ordered {
        let (mut acc, mut gs): (Option<i64>, usize) = (None, 0);
        for pos in 0..part.len() {
            let row = part[pos];
            if arr.is_valid(row) {
                let v = arr.value(row);
                acc = Some(match (func, acc) {
                    (_, None) => v,
                    (WindowFn::Sum, Some(a)) => a + v,
                    (WindowFn::Min, Some(a)) => a.min(v),
                    (WindowFn::Max, Some(a)) => a.max(v),
                    (_, Some(a)) => a,
                });
            }
            if peer_boundary(part, order_rows, pos) {
                for j in gs..=pos {
                    out[part[j]] = acc;
                }
                gs = pos + 1;
            }
        }
    }
    Ok(Arc::new(Int64Array::from(out)))
}

fn running_numeric_f64(
    func: WindowFn,
    ordered: &[Vec<usize>],
    order_rows: &Rows,
    values: &ArrayRef,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    let arr = values.as_primitive::<Float64Type>();
    let is_avg = func == WindowFn::Avg;
    let mut out: Vec<Option<f64>> = vec![None; num_rows];
    for part in ordered {
        let (mut acc, mut cnt, mut gs): (Option<f64>, i64, usize) = (None, 0, 0);
        for pos in 0..part.len() {
            let row = part[pos];
            if arr.is_valid(row) {
                let v = arr.value(row);
                cnt += 1;
                acc = Some(match (func, acc) {
                    (_, None) => v,
                    (WindowFn::Sum | WindowFn::Avg, Some(a)) => a + v,
                    (WindowFn::Min, Some(a)) => a.min(v),
                    (WindowFn::Max, Some(a)) => a.max(v),
                    (_, Some(a)) => a,
                });
            }
            if peer_boundary(part, order_rows, pos) {
                let v = acc.map(|a| if is_avg { a / cnt as f64 } else { a });
                for j in gs..=pos {
                    out[part[j]] = v;
                }
                gs = pos + 1;
            }
        }
    }
    Ok(Arc::new(Float64Array::from(out)))
}

fn running_str_minmax(
    func: WindowFn,
    ordered: &[Vec<usize>],
    order_rows: &Rows,
    values: &ArrayRef,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    let arr = values.as_any().downcast_ref::<StringArray>().expect("utf8");
    let mut out: Vec<Option<String>> = vec![None; num_rows];
    for part in ordered {
        let (mut acc, mut gs): (Option<String>, usize) = (None, 0);
        for pos in 0..part.len() {
            let row = part[pos];
            if arr.is_valid(row) {
                let v = arr.value(row);
                let replace = match &acc {
                    None => true,
                    Some(a) => {
                        (func == WindowFn::Min && v < a.as_str())
                            || (func == WindowFn::Max && v > a.as_str())
                    }
                };
                if replace {
                    acc = Some(v.to_string());
                }
            }
            if peer_boundary(part, order_rows, pos) {
                for j in gs..=pos {
                    out[part[j]] = acc.clone();
                }
                gs = pos + 1;
            }
        }
    }
    Ok(Arc::new(StringArray::from(out)))
}

/// Whole-partition aggregate: compute one value per partition and broadcast it to
/// every row of that partition (same value regardless of order — v1 semantics).
pub(crate) fn require(
    values: Option<&ArrayRef>,
    func: WindowFn,
) -> Result<&ArrayRef, RuntimeError> {
    values.ok_or_else(|| RuntimeError::MissingWindowInput {
        func: func.name().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array, StringArray};

    fn i64s(v: &[i64]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }
    fn strs(v: &[&str]) -> ArrayRef {
        Arc::new(StringArray::from(v.to_vec()))
    }
    fn asc(arr: ArrayRef) -> (ArrayRef, SortOptions) {
        (
            arr,
            SortOptions {
                descending: false,
                nulls_first: false,
            },
        )
    }

    fn ints(a: &ArrayRef) -> Vec<i64> {
        let x = a.as_any().downcast_ref::<Int64Array>().unwrap();
        (0..x.len()).map(|i| x.value(i)).collect()
    }
    fn floats(a: &ArrayRef) -> Vec<f64> {
        let x = a.as_any().downcast_ref::<Float64Array>().unwrap();
        (0..x.len()).map(|i| x.value(i)).collect()
    }

    /// The bucket-parallel path (`window_with` with a low row threshold) must produce
    /// exactly what the serial kernel does, for ordered windows across many partitions
    /// — ranking, running aggregate, and a positional (lag) function. Partitioning by
    /// key and scattering back must not change any per-row result.
    #[test]
    fn parallel_matches_serial_ordered() {
        let n = 600usize;
        // ~40 partitions (key = i % 40), each ordered by a shuffled value.
        let part = i64s(&(0..n as i64).map(|i| i % 40).collect::<Vec<_>>());
        let ord = i64s(
            &(0..n as i64)
                .map(|i| (i * 7 + 13) % 100)
                .collect::<Vec<_>>(),
        );
        let vals = i64s(&(0..n as i64).map(|i| i * 2 - 5).collect::<Vec<_>>());
        let order = [asc(ord)];
        let cases: Vec<WindowCall> = vec![
            WindowCall {
                func: WindowFn::Rank,
                values: None,
                offset: 1,
                frame: None,
            },
            WindowCall {
                func: WindowFn::RowNumber,
                values: None,
                offset: 1,
                frame: None,
            },
            WindowCall {
                func: WindowFn::Sum,
                values: Some(vals.clone()),
                offset: 1,
                frame: None,
            },
            WindowCall {
                func: WindowFn::Lag,
                values: Some(vals.clone()),
                offset: 1,
                frame: None,
            },
        ];
        // Null-aware read (Lag yields nulls at each partition's first row; the raw
        // value in a null slot is unspecified, so compare validity + value together).
        let opt_ints = |a: &ArrayRef| -> Vec<Option<i64>> {
            let x = a.as_any().downcast_ref::<Int64Array>().unwrap();
            (0..x.len())
                .map(|i| x.is_valid(i).then(|| x.value(i)))
                .collect()
        };
        for call in cases {
            let funcs = [call];
            // threshold 1 forces the parallel path; usize::MAX keeps the serial oracle.
            let par = window_with(&[part.clone()], &order, &funcs, n, 1).unwrap();
            let ser = window_serial(&[part.clone()], &order, &funcs, n).unwrap();
            assert_eq!(
                opt_ints(&par[0]),
                opt_ints(&ser[0]),
                "parallel != serial for {:?}",
                funcs[0].func
            );
        }
    }

    /// The bucket-parallel path must also match the serial fast path for a *frameless*
    /// whole-partition aggregate (no ORDER BY) — the case now routed through
    /// `window_parallel` so a high-cardinality `sum() OVER (PARTITION BY k)` bucket the
    /// key across cores. Every row of a partition must still get that partition's total.
    #[test]
    fn parallel_matches_serial_frameless_aggregate() {
        let n = 800usize;
        let part = i64s(&(0..n as i64).map(|i| i % 50).collect::<Vec<_>>());
        let vals = i64s(&(0..n as i64).map(|i| i * 3 - 7).collect::<Vec<_>>());
        for func in [WindowFn::Sum, WindowFn::Min, WindowFn::Max, WindowFn::Count] {
            let funcs = [WindowCall {
                func,
                values: Some(vals.clone()),
                offset: 1,
                frame: None,
            }];
            // threshold 1 forces the parallel bucket path; usize::MAX keeps the oracle.
            let par = window_with(&[part.clone()], &[], &funcs, n, 1).unwrap();
            let ser = window_serial(&[part.clone()], &[], &funcs, n).unwrap();
            assert_eq!(
                ints(&par[0]),
                ints(&ser[0]),
                "parallel != serial for {func:?}"
            );
        }
    }

    #[test]
    fn row_number_single_partition() {
        // order by val asc → ranks follow sorted positions, scattered back.
        let order = i64s(&[30, 10, 20]);
        let funcs = [WindowCall {
            func: WindowFn::RowNumber,
            values: None,
            offset: 1,
            frame: None,
        }];
        let out = window(&[], &[asc(order)], &funcs, 3).unwrap();
        // sorted order: idx1(10)=1, idx2(20)=2, idx0(30)=3
        assert_eq!(ints(&out[0]), vec![3, 1, 2]);
    }

    #[test]
    fn rank_and_dense_rank_with_ties() {
        // values: [10, 10, 20, 30] in order → ranks 1,1,3,4; dense 1,1,2,3.
        let order = i64s(&[10, 10, 20, 30]);
        let funcs = [
            WindowCall {
                func: WindowFn::Rank,
                values: None,
                offset: 1,
                frame: None,
            },
            WindowCall {
                func: WindowFn::DenseRank,
                values: None,
                offset: 1,
                frame: None,
            },
            WindowCall {
                func: WindowFn::RowNumber,
                values: None,
                offset: 1,
                frame: None,
            },
        ];
        let out = window(&[], &[asc(order)], &funcs, 4).unwrap();
        assert_eq!(ints(&out[0]), vec![1, 1, 3, 4]); // rank (gaps)
        assert_eq!(ints(&out[1]), vec![1, 1, 2, 3]); // dense_rank (no gaps)
                                                     // row_number is 1..n; the two tied rows get 1 and 2 in some order.
        let rn = ints(&out[2]);
        assert_eq!(rn[2], 3);
        assert_eq!(rn[3], 4);
        assert_eq!(
            {
                let mut s = vec![rn[0], rn[1]];
                s.sort();
                s
            },
            vec![1, 2]
        );
    }

    #[test]
    fn percent_rank_and_cume_dist_with_ties() {
        // values: [10, 10, 20, 30] in order.
        // RANK: 1,1,3,4 → percent_rank (n=4): 0, 0, 2/3, 1.
        // cume_dist: peer {10,10} ends at pos 1 → 2/4; 20 → 3/4; 30 → 4/4.
        let order = i64s(&[10, 10, 20, 30]);
        let funcs = [
            WindowCall {
                func: WindowFn::PercentRank,
                values: None,
                offset: 1,
                frame: None,
            },
            WindowCall {
                func: WindowFn::CumeDist,
                values: None,
                offset: 1,
                frame: None,
            },
        ];
        let out = window(&[], &[asc(order)], &funcs, 4).unwrap();
        assert_eq!(floats(&out[0]), vec![0.0, 0.0, 2.0 / 3.0, 1.0]);
        assert_eq!(floats(&out[1]), vec![0.5, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn percent_rank_single_row_is_zero() {
        let order = i64s(&[42]);
        let funcs = [WindowCall {
            func: WindowFn::PercentRank,
            values: None,
            offset: 1,
            frame: None,
        }];
        let out = window(&[], &[asc(order)], &funcs, 1).unwrap();
        assert_eq!(floats(&out[0]), vec![0.0]);
    }

    #[test]
    fn ntile_distributes_remainder_to_early_buckets() {
        // 5 rows, 2 buckets → sizes 3,2 → buckets 1,1,1,2,2 in order.
        let order = i64s(&[10, 20, 30, 40, 50]);
        let funcs = [WindowCall {
            func: WindowFn::Ntile,
            values: None,
            offset: 2,
            frame: None,
        }];
        let out = window(&[], &[asc(order)], &funcs, 5).unwrap();
        assert_eq!(ints(&out[0]), vec![1, 1, 1, 2, 2]);
    }

    #[test]
    fn ntile_more_buckets_than_rows() {
        // 2 rows, 4 buckets → each row its own bucket 1,2; buckets 3,4 empty.
        let order = i64s(&[10, 20]);
        let funcs = [WindowCall {
            func: WindowFn::Ntile,
            values: None,
            offset: 4,
            frame: None,
        }];
        let out = window(&[], &[asc(order)], &funcs, 2).unwrap();
        assert_eq!(ints(&out[0]), vec![1, 2]);
    }

    #[test]
    fn percent_rank_without_order_is_error() {
        let funcs = [WindowCall {
            func: WindowFn::PercentRank,
            values: None,
            offset: 1,
            frame: None,
        }];
        assert!(window(&[], &[], &funcs, 3).is_err());
    }

    #[test]
    fn rank_multiple_partitions() {
        // partition key p: [a,a,b,b], order val: [10,20,5,5]
        // partition a: 10,20 → rank 1,2 ; partition b: 5,5 → rank 1,1.
        let part = strs(&["a", "a", "b", "b"]);
        let order = i64s(&[10, 20, 5, 5]);
        let funcs = [
            WindowCall {
                func: WindowFn::Rank,
                values: None,
                offset: 1,
                frame: None,
            },
            WindowCall {
                func: WindowFn::DenseRank,
                values: None,
                offset: 1,
                frame: None,
            },
        ];
        let out = window(&[part], &[asc(order)], &funcs, 4).unwrap();
        assert_eq!(ints(&out[0]), vec![1, 2, 1, 1]);
        assert_eq!(ints(&out[1]), vec![1, 2, 1, 1]);
    }

    #[test]
    fn sum_over_partition_broadcasts() {
        // partition p: [a,b,a,b,a], vals: [1,2,3,4,5]
        // a: 1+3+5=9 ; b: 2+4=6 — same value for every row in the partition.
        let part = strs(&["a", "b", "a", "b", "a"]);
        let vals = i64s(&[1, 2, 3, 4, 5]);
        let funcs = [WindowCall {
            func: WindowFn::Sum,
            values: Some(vals),
            offset: 1,
            frame: None,
        }];
        let out = window(&[part], &[], &funcs, 5).unwrap();
        assert_eq!(ints(&out[0]), vec![9, 6, 9, 6, 9]);
    }

    #[test]
    fn aggregates_over_whole_input_no_partition() {
        let vals = i64s(&[1, 2, 3, 4]);
        let funcs = [
            WindowCall {
                func: WindowFn::Sum,
                values: Some(vals.clone()),
                offset: 1,
                frame: None,
            },
            WindowCall {
                func: WindowFn::Min,
                values: Some(vals.clone()),
                offset: 1,
                frame: None,
            },
            WindowCall {
                func: WindowFn::Max,
                values: Some(vals.clone()),
                offset: 1,
                frame: None,
            },
            WindowCall {
                func: WindowFn::Count,
                values: Some(vals.clone()),
                offset: 1,
                frame: None,
            },
            WindowCall {
                func: WindowFn::Avg,
                values: Some(vals),
                offset: 1,
                frame: None,
            },
        ];
        let out = window(&[], &[], &funcs, 4).unwrap();
        assert_eq!(ints(&out[0]), vec![10, 10, 10, 10]); // sum
        assert_eq!(ints(&out[1]), vec![1, 1, 1, 1]); // min
        assert_eq!(ints(&out[2]), vec![4, 4, 4, 4]); // max
        assert_eq!(ints(&out[3]), vec![4, 4, 4, 4]); // count
        assert_eq!(floats(&out[4]), vec![2.5, 2.5, 2.5, 2.5]); // avg
    }

    #[test]
    fn min_max_over_strings() {
        let part = strs(&["g", "g", "h"]);
        let vals = strs(&["banana", "apple", "cherry"]);
        let funcs = [
            WindowCall {
                func: WindowFn::Min,
                values: Some(vals.clone()),
                offset: 1,
                frame: None,
            },
            WindowCall {
                func: WindowFn::Max,
                values: Some(vals),
                offset: 1,
                frame: None,
            },
        ];
        let out = window(&[part], &[], &funcs, 3).unwrap();
        let mins = out[0].as_any().downcast_ref::<StringArray>().unwrap();
        let maxs = out[1].as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(mins.value(0), "apple");
        assert_eq!(mins.value(1), "apple");
        assert_eq!(mins.value(2), "cherry");
        assert_eq!(maxs.value(0), "banana");
        assert_eq!(maxs.value(2), "cherry");
    }

    #[test]
    fn rank_without_order_is_error() {
        let funcs = [WindowCall {
            func: WindowFn::Rank,
            values: None,
            offset: 1,
            frame: None,
        }];
        assert!(window(&[], &[], &funcs, 3).is_err());
    }

    #[test]
    fn avg_float_partition() {
        let part = strs(&["a", "a", "b"]);
        let vals: ArrayRef = Arc::new(Float64Array::from(vec![1.0, 2.0, 10.0]));
        let funcs = [WindowCall {
            func: WindowFn::Avg,
            values: Some(vals),
            offset: 1,
            frame: None,
        }];
        let out = window(&[part], &[], &funcs, 3).unwrap();
        assert_eq!(floats(&out[0]), vec![1.5, 1.5, 10.0]);
    }

    // --- forward_fill / backward_fill ------------------------------------------------

    fn opt_i64s(v: &[Option<i64>]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }
    fn opt_ints(a: &ArrayRef) -> Vec<Option<i64>> {
        let x = a.as_any().downcast_ref::<Int64Array>().unwrap();
        (0..x.len())
            .map(|i| (!x.is_null(i)).then(|| x.value(i)))
            .collect()
    }
    fn fill(func: WindowFn, part: &[ArrayRef], ord: ArrayRef, vals: ArrayRef) -> Vec<Option<i64>> {
        let n = vals.len();
        let funcs = [WindowCall {
            func,
            values: Some(vals),
            offset: 1,
            frame: None,
        }];
        let out = window(part, &[asc(ord)], &funcs, n).unwrap();
        opt_ints(&out[0])
    }

    #[test]
    fn forward_fill_carries_the_last_non_null() {
        let ord = i64s(&[0, 1, 2, 3, 4]);
        let vals = opt_i64s(&[Some(1), None, None, Some(4), None]);
        let got = fill(WindowFn::ForwardFill, &[], ord, vals);
        assert_eq!(got, vec![Some(1), Some(1), Some(1), Some(4), Some(4)]);
    }

    #[test]
    fn backward_fill_carries_the_next_non_null() {
        let ord = i64s(&[0, 1, 2, 3, 4]);
        let vals = opt_i64s(&[None, Some(2), None, None, Some(5)]);
        let got = fill(WindowFn::BackwardFill, &[], ord, vals);
        assert_eq!(got, vec![Some(2), Some(2), Some(5), Some(5), Some(5)]);
    }

    #[test]
    fn leading_nulls_have_nothing_to_carry_and_stay_null() {
        let ord = i64s(&[0, 1, 2]);
        let vals = opt_i64s(&[None, None, Some(3)]);
        assert_eq!(
            fill(WindowFn::ForwardFill, &[], ord.clone(), vals.clone()),
            vec![None, None, Some(3)]
        );
        // ... and symmetrically, trailing nulls under a backward fill.
        let vals = opt_i64s(&[Some(1), None, None]);
        assert_eq!(
            fill(WindowFn::BackwardFill, &[], ord, vals),
            vec![Some(1), None, None]
        );
    }

    #[test]
    fn an_all_null_partition_stays_all_null() {
        let ord = i64s(&[0, 1]);
        let vals = opt_i64s(&[None, None]);
        assert_eq!(
            fill(WindowFn::ForwardFill, &[], ord, vals),
            vec![None, None]
        );
    }

    #[test]
    fn a_fill_never_crosses_a_partition_boundary() {
        // Two series interleaved in arrival order. `b`'s leading null must NOT pick up
        // `a`'s value — that is the bug a naive whole-relation scan would ship.
        let part = strs(&["a", "b", "a", "b"]);
        let ord = i64s(&[0, 0, 1, 1]);
        let vals = opt_i64s(&[Some(10), None, None, Some(20)]);
        let got = fill(WindowFn::ForwardFill, &[part], ord, vals);
        assert_eq!(got, vec![Some(10), None, Some(10), Some(20)]);
    }

    #[test]
    fn the_fill_follows_the_order_key_not_arrival_order() {
        // Rows arrive out of order; the fill must respect ORDER BY t.
        let ord = i64s(&[2, 0, 1]);
        let vals = opt_i64s(&[None, Some(7), None]);
        let got = fill(WindowFn::ForwardFill, &[], ord, vals);
        // t=0 -> 7, t=1 -> 7 (row 2), t=2 -> 7 (row 0).
        assert_eq!(got, vec![Some(7), Some(7), Some(7)]);
    }

    #[test]
    fn a_fill_is_the_identity_where_the_column_is_non_null() {
        let ord = i64s(&[0, 1, 2]);
        let vals = opt_i64s(&[Some(1), Some(2), Some(3)]);
        assert_eq!(
            fill(WindowFn::ForwardFill, &[], ord.clone(), vals.clone()),
            vec![Some(1), Some(2), Some(3)]
        );
        assert_eq!(
            fill(WindowFn::BackwardFill, &[], ord, vals),
            vec![Some(1), Some(2), Some(3)]
        );
    }

    #[test]
    fn fill_is_type_generic() {
        let ord = i64s(&[0, 1, 2]);
        let vals: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None, Some("c")]));
        let funcs = [WindowCall {
            func: WindowFn::ForwardFill,
            values: Some(vals),
            offset: 1,
            frame: None,
        }];
        let out = window(&[], &[asc(ord)], &funcs, 3).unwrap();
        let s = out[0].as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(
            (0..3).map(|i| s.value(i)).collect::<Vec<_>>(),
            vec!["a", "a", "c"]
        );
    }

    /// The parallel bucket path hash-partitions on the PARTITION BY keys, so a fill —
    /// which carries state along a partition — must be identical to the serial kernel.
    #[test]
    fn parallel_matches_serial_for_fills() {
        let n = 600usize;
        let part = i64s(&(0..n as i64).map(|i| i % 40).collect::<Vec<_>>());
        let ord = i64s(
            &(0..n as i64)
                .map(|i| (i * 7 + 13) % 100)
                .collect::<Vec<_>>(),
        );
        // Every third row is null, so each partition has real gaps to carry across.
        let vals = opt_i64s(
            &(0..n as i64)
                .map(|i| (i % 3 != 0).then_some(i * 2 - 5))
                .collect::<Vec<_>>(),
        );
        for func in [WindowFn::ForwardFill, WindowFn::BackwardFill] {
            let funcs = [WindowCall {
                func,
                values: Some(vals.clone()),
                offset: 1,
                frame: None,
            }];
            let order = [asc(ord.clone())];
            let serial = window_with(&[part.clone()], &order, &funcs, n, usize::MAX).unwrap();
            let parallel = window_with(&[part.clone()], &order, &funcs, n, 1).unwrap();
            assert_eq!(opt_ints(&serial[0]), opt_ints(&parallel[0]), "{func:?}");
        }
    }
}
