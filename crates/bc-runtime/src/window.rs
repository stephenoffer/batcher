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

use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, Float64Array, Int64Array, StringArray, UInt32Array,
};
use arrow::compute::{take, SortOptions};
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
    // Canonicalize float PARTITION BY keys once, here, before any grouping path sees them, so
    // `PARTITION BY f` folds `-0.0`/`0.0` into one partition and all NaNs into one — the same
    // key identity GROUP BY uses (`crate::keys`). The RowConverter-based partition groupers
    // (`assign_partitions`, `ordered_partitions_by_global_sort`, the parallel hash bucketer) do
    // NOT canonicalize on their own, so `distinct(subset)` — which lowers to
    // `row_number() OVER (PARTITION BY subset ...)` — returned two rows for `[-0.0, 0.0]` where
    // DuckDB returns one. Window function *outputs* don't include the key, so folding the key's
    // identity never changes an emitted value. Non-float keys are returned unchanged.
    let canon = crate::keys::canonicalize_float_keys(partition_keys);
    let partition_keys: &[ArrayRef] = canon.as_deref().unwrap_or(partition_keys);

    // Canonicalize float ORDER BY keys the same way, and for the same reason: the
    // RowConverter-based order paths (`ordered_partitions_by_global_sort`, `encode_order_keys`
    // → `rows_equal` → `peer_boundary`) rank raw bits, so without this `-0.0` and `0.0` are not
    // *peers* — `RANK`/`DENSE_RANK` split them and every `RANGE`/`GROUPS` bound moves — and a
    // *negative* NaN ranks below -inf instead of last, all disagreeing with the `GROUP BY`,
    // `=`, and `MIN`/`MAX` the same column feeds. The order key is used only to order/peer/frame,
    // never emitted (value functions reference their own argument), so folding its identity
    // changes no output. Non-float keys are returned unchanged.
    let canon_order = crate::keys::canonicalize_float_order_keys(order_keys);
    let order_keys: &[(ArrayRef, SortOptions)] = canon_order.as_deref().unwrap_or(order_keys);

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
    //
    // Only the functions that actually consult *peers* or a *frame* read this: the rank
    // family (`rank`/`percent_rank`/`cume_dist`), the framed value/aggregate paths, and the
    // running aggregate. `ROW_NUMBER` and `NTILE` are pure positions within `ordered`, and a
    // frameless value function (`LAG`/`LEAD`/`FIRST_VALUE`/…) selects by position too — none
    // of them look at ties. Encoding unconditionally therefore built a full `RowConverter`
    // image of every order key for queries that never touched it; at 6M rows that is a ~54 MB
    // allocation and a full encode pass of pure waste on, e.g., a `LAG` window.
    let needs_peers = funcs.iter().any(|c| {
        !matches!(c.func, WindowFn::RowNumber | WindowFn::Ntile)
            && !(c.func.is_value() && c.frame.is_none())
    });
    let order_rows = if order_keys.is_empty() || !needs_peers {
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
            // Value functions with no explicit frame select a row's value by
            // position within the *whole* ordered partition
            // (first_value/last_value/lag/lead/nth_value/fills).
            f if f.is_value() && call.frame.is_none() => value_window(
                f,
                &ordered,
                require(call.values.as_ref(), f)?,
                call.offset,
                num_rows,
            )?,
            // first_value/last_value/nth_value over an explicit frame — the frame's
            // first / last / nth row. SQL's default value-function frame is
            // `RANGE UNBOUNDED PRECEDING TO CURRENT ROW`, which makes last_value /
            // nth_value *running* (the current peer's value / null-until-nth-peer)
            // rather than the whole-partition value the frameless path computes.
            f if f.is_value() => crate::window_frame::framed_value(
                f,
                &ordered,
                require(call.values.as_ref(), f)?,
                call.offset,
                call.frame.expect("frame present"),
                order_rows.as_ref(),
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
    let mut src: Vec<Option<u32>> = vec![None; num_rows];
    for part in ordered {
        let len = part.len();
        for (pos, &row) in part.iter().enumerate() {
            let pos = pos as i64;
            let take_pos: Option<usize> = match func {
                WindowFn::FirstValue => Some(0),
                WindowFn::LastValue => Some(len - 1),
                // A negative `lag`/`lead` offset flips direction (`lag(v, -n)` == `lead(v, n)`,
                // matching DuckDB), so index by the SIGNED target and range-check it — a plain
                // `offset.max(0)` would collapse every negative offset to the current row.
                // `checked_*` guards an absurd offset (e.g. i64::MIN) that would overflow.
                WindowFn::Lag => pos
                    .checked_sub(offset)
                    .filter(|&t| (0..len as i64).contains(&t))
                    .map(|t| t as usize),
                WindowFn::Lead => pos
                    .checked_add(offset)
                    .filter(|&t| (0..len as i64).contains(&t))
                    .map(|t| t as usize),
                // nth_value: the `offset`-th row (1-based), same for every row of the
                // partition; null if the partition is shorter than `offset`.
                WindowFn::NthValue => {
                    (offset >= 1 && offset <= len as i64).then_some((offset - 1) as usize)
                }
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
    // Fast path: exactly one non-null primitive-numeric partition key and one non-null
    // primitive-numeric order key — the dominant window shape (`PARTITION BY id ORDER BY
    // value`). Pack each into an order-preserving `u64` and sort `(part, ord, idx)` tuples
    // directly: NO RowConverter encode at all, and every comparison is a register-resident
    // tuple compare instead of a random-access row-byte compare. Multi-column, nullable, or
    // non-numeric keys fall through to the general row-encoded path below.
    if let Some(out) = try_ordered_partitions_packed(partition_keys, order_keys, num_rows) {
        return Ok(out);
    }

    // Sort columns: partition keys first (ascending — grouping is order-agnostic), then
    // the order keys with their own ASC/DESC + nulls placement, and finally the original
    // row index as the last ascending key. The index tie-break makes the sort a *total*
    // order, so rows tied on the partition + order keys land in original-row order — which
    // (a) is a stable, deterministic choice for `row_number`'s otherwise-unspecified order
    // among peers, and (b) is identical whether this runs over the whole input or over one
    // hash bucket of it, so the parallel per-bucket path matches the serial kernel exactly.
    // Encode all sort keys into arrow's row format once (a serial O(n) pass) and then sort the
    // row indices **in parallel** by comparing those encoded bytes. arrow's `lexsort_to_indices`
    // is single-threaded, and on a large window (a 60M-row `RANK() OVER (PARTITION BY flag ORDER
    // BY price)`, whose leading key is a string) that serial O(n log n) sort was the whole
    // operator — ~118s at ~1% CPU. The Row encoding is designed so a byte-lexicographic compare
    // equals the multi-column ordering (partition keys ascending, order keys with their own
    // ASC/DESC + nulls placement, then the original row index ascending as a unique tie-break),
    // so `par_sort_unstable_by` over it yields the identical total order — just across all cores.
    use rayon::slice::ParallelSliceMut;
    let mut enc_fields: Vec<SortField> = partition_keys
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect();
    let mut enc_arrays: Vec<ArrayRef> = partition_keys.to_vec();
    for (arr, opts) in order_keys {
        enc_fields.push(SortField::new_with_options(arr.data_type().clone(), *opts));
        enc_arrays.push(arr.clone());
    }
    // Encode only the (partition ++ order) keys — NOT a row-index tie-break column. The
    // original-row-order tie-break is applied directly in the comparator below (`a.cmp(&b)`
    // on the inline indices), so encoding it into the row format would be pure overhead: it
    // widened every row by ~9 bytes (slowing both the encode and every full-row compare) to
    // recompute an order we already carry for free.
    let converter = RowConverter::new(enc_fields)?;
    let rows = converter.convert_columns(&enc_arrays)?;
    // Sort `(prefix, row_index)` PAIRS, not bare indices. The prefix is the order-preserving
    // leading 8 bytes of each encoded row (arrow's row format is byte-lexicographic, so
    // `u64::from_be_bytes(first 8 bytes)` orders identically to the row's leading bytes). By
    // carrying the key *inline* with its index, the comparator reads both prefixes straight
    // from the two elements being compared (in-register, no indirection) — where sorting a
    // bare index array instead chased `prefixes[a]`/`rows.row(a)` to random positions in a
    // 10 MB buffer on every comparison, and those cache misses were the whole window cost.
    // The full-row compare (the wider, random-access read) is kept only as the tie-break on
    // equal prefixes — rare when the leading key is discriminating — so the total order and
    // every downstream rank/peer result stay byte-for-byte identical. A final `a.cmp(&b)` on
    // the inline original-row indices makes the order total (rows equal on partition+order
    // keys fall back to input order), so the unstable sort is deterministic.
    let mut keyed: Vec<(u64, u32)> = (0..num_rows)
        .map(|i| (row_prefix(rows.row(i).data()), i as u32))
        .collect();
    keyed.par_sort_unstable_by(|&(pa, a), &(pb, b)| {
        pa.cmp(&pb)
            .then_with(|| rows.row(a as usize).cmp(&rows.row(b as usize)))
            .then_with(|| a.cmp(&b))
    });
    let sorted: Vec<u32> = keyed.iter().map(|&(_, i)| i).collect();
    let sorted = &sorted[..];

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

    // Runs are collected from slices of known length, so each partition's `Vec` is allocated
    // once at its exact size rather than regrown from the capacity-0 `Vec` that `mem::take`
    // leaves behind — see the matching note in `try_ordered_partitions_packed`.
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut start = 0usize;
    for pos in 1..=sorted.len() {
        let boundary = pos == sorted.len()
            || prows.row(sorted[pos] as usize) != prows.row(sorted[start] as usize);
        if boundary {
            out.push(sorted[start..pos].iter().map(|&r| r as usize).collect());
            start = pos;
        }
    }
    Ok(out)
}

/// Order-preserving `u64` for an `i64`: flip the sign bit so an unsigned compare reproduces
/// the signed order (`i64::MIN` → 0, `i64::MAX` → `u64::MAX`).
#[inline]
fn i64_ordered(x: i64) -> u64 {
    (x as u64) ^ (1u64 << 63)
}

/// Order-preserving `u64` for an `f64` (IEEE total order): non-negative floats set the sign
/// bit, negatives invert every bit, so an unsigned compare reproduces `<` on the floats (and
/// sorts `NaN` last, matching arrow's row encoding). Float keys are canonicalized upstream
/// (`-0.0`→`0.0`, every NaN folded), so equal floats map to equal `u64`s here too.
#[inline]
fn f64_ordered(x: f64) -> u64 {
    let b = x.to_bits();
    if b & (1u64 << 63) == 0 {
        b | (1u64 << 63)
    } else {
        !b
    }
}

/// Pack a non-null `Int64`/`Float64` column into order-preserving `u64`s; `None` for any
/// other type or a column with nulls (the general row-encoded path handles those). `value(i)`
/// is used (not the raw buffer) so a sliced array's offset is honored.
fn pack_ordered_u64(a: &ArrayRef) -> Option<Vec<u64>> {
    if a.null_count() != 0 {
        return None;
    }
    match a.data_type() {
        DataType::Int64 => {
            let v = a.as_primitive::<Int64Type>();
            Some((0..v.len()).map(|i| i64_ordered(v.value(i))).collect())
        }
        DataType::Float64 => {
            let v = a.as_primitive::<Float64Type>();
            Some((0..v.len()).map(|i| f64_ordered(v.value(i))).collect())
        }
        _ => None,
    }
}

/// Packed fast path for [`ordered_partitions_by_global_sort`]: at most one non-null numeric
/// partition key plus one non-null numeric order key. Returns the same per-partition ordered
/// index lists the general path does — partition keys group each partition's rows contiguously,
/// the order key orders within, and the trailing original index makes the total order match the
/// general path's `(partition, order, index)` tie-break exactly. `None` when not eligible.
///
/// Zero partition keys (`… OVER (ORDER BY x)`) is a single global partition: a constant
/// partition component leaves the split below emitting one run, exactly as the general path's
/// no-PARTITION-BY branch does. Covering it here keeps an unpartitioned window off the
/// RowConverter path, which was ~7x DuckDB on a 6M-row `rank() OVER (ORDER BY …)`.
fn try_ordered_partitions_packed(
    partition_keys: &[ArrayRef],
    order_keys: &[(ArrayRef, SortOptions)],
    num_rows: usize,
) -> Option<Vec<Vec<usize>>> {
    use rayon::slice::ParallelSliceMut;
    if partition_keys.len() > 1 || order_keys.len() != 1 {
        return None;
    }
    // `None` = no PARTITION BY; every row shares the constant partition component below (kept
    // as an `Option` rather than a materialized constant column to avoid an 8 B/row allocation).
    let part: Option<Vec<u64>> = match partition_keys.first() {
        Some(k) => Some(pack_ordered_u64(k)?),
        None => None,
    };
    let (ord_arr, opts) = &order_keys[0];
    let mut ord = pack_ordered_u64(ord_arr)?;
    // DESC: invert the order-preserving key so an ascending unsigned sort yields descending.
    // (`nulls_first` is irrelevant — this path requires non-null keys.)
    if opts.descending {
        for x in &mut ord {
            *x = !*x;
        }
    }
    // Sort (partition, order, original-index) tuples. The derived tuple `Ord` sorts by
    // partition, then order, then index; the index makes it a total order, so the unstable
    // sort is deterministic. `par_sort` is kept even though this runs per hash bucket inside
    // `window_parallel`: buckets finish unevenly, so rayon work-stealing puts the freed cores
    // onto the still-running buckets' sorts — measurably faster than a serial per-bucket sort.
    let mut keyed: Vec<(u64, u64, u32)> = (0..num_rows)
        .map(|i| (part.as_ref().map_or(0, |p| p[i]), ord[i], i as u32))
        .collect();
    keyed.par_sort_unstable();
    // Split the sorted tuples into contiguous same-partition runs (partition key changed).
    //
    // Each run is collected from a slice of known length, so its `Vec` is allocated once at the
    // exact size. The previous form pushed into a `cur` reset by `mem::take` — which hands back
    // a *capacity-0* `Vec` — so every partition regrew 1→2→4, ~3 allocations each. A near-unique
    // PARTITION BY is the normal case here (`PARTITION BY l_orderkey` is ~1.5M partitions of ~4
    // rows at SF1), which made that a few million malloc/free pairs inside the timed region.
    // Identical output: the same runs, in the same order, holding the same indices.
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut start = 0usize;
    for pos in 1..=keyed.len() {
        if pos == keyed.len() || keyed[pos].0 != keyed[start].0 {
            out.push(
                keyed[start..pos]
                    .iter()
                    .map(|&(_, _, i)| i as usize)
                    .collect(),
            );
            start = pos;
        }
    }
    Some(out)
}

/// The order-preserving leading 8 bytes of an encoded row, as a `u64`.
///
/// arrow's row format is byte-lexicographic, so the big-endian `u64` of a row's first 8 bytes
/// orders identically to the row's leading bytes; carrying this inline with the row index
/// (see the sort in [`ordered_partitions_by_global_sort`]) resolves almost every comparison
/// in-register, without dereferencing the full (wider, randomly accessed) Rows buffer. Rows
/// shorter than 8 bytes are zero-padded — correct, since a shorter row is lexicographically
/// smaller and the full-row tie-break handles the pad-equal case.
fn row_prefix(d: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let n = d.len().min(8);
    buf[..n].copy_from_slice(&d[..n]);
    u64::from_be_bytes(buf)
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
        // Boolean running MIN (AND) / MAX (OR), `false < true` — matches the aggregate
        // MIN/MAX (B23), the whole-partition path, and DuckDB.
        DataType::Boolean if matches!(func, WindowFn::Min | WindowFn::Max) => {
            running_bool_minmax(func, ordered, order_rows, values, num_rows)
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
            // Accumulate the running sum in i128 (like DuckDB's HUGEINT), not f64: a
            // running `AVG(i64)` over values past 2^53 (e.g. `[2^53+1, 1]`) loses its low
            // bit in an f64 accumulator (avg came back `…496.0` instead of `…497.0`). The
            // exact i128 sum divided once at the peer boundary matches the interpreter and
            // DuckDB; i128 can't overflow for any realistic i64 column (~2^64 rows).
            let (mut sum, mut cnt, mut gs) = (0i128, 0i64, 0usize);
            for pos in 0..part.len() {
                if arr.is_valid(part[pos]) {
                    sum += arr.value(part[pos]) as i128;
                    cnt += 1;
                }
                if peer_boundary(part, order_rows, pos) {
                    let v = (cnt > 0).then(|| sum as f64 / cnt as f64);
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
                    // checked_add: an i64 running-SUM overflow errors instead of wrapping.
                    (WindowFn::Sum, Some(a)) => {
                        a.checked_add(v).ok_or(RuntimeError::SumOverflow)?
                    }
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
                // Total-order min/max so NaN is greatest (matches aggregate MIN/MAX/DuckDB).
                acc = Some(match (func, acc) {
                    (_, None) => v,
                    (WindowFn::Sum | WindowFn::Avg, Some(a)) => a + v,
                    (WindowFn::Min, Some(a)) => {
                        if crate::keys::float_total_cmp(v, a).is_lt() {
                            v
                        } else {
                            a
                        }
                    }
                    (WindowFn::Max, Some(a)) => {
                        if crate::keys::float_total_cmp(v, a).is_gt() {
                            v
                        } else {
                            a
                        }
                    }
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

/// Running boolean MIN (AND) / MAX (OR) over the ordered partition, `false < true`,
/// with the same peer-tie sharing as the numeric running aggregates.
fn running_bool_minmax(
    func: WindowFn,
    ordered: &[Vec<usize>],
    order_rows: &Rows,
    values: &ArrayRef,
    num_rows: usize,
) -> Result<ArrayRef, RuntimeError> {
    let arr = values.as_boolean();
    let is_min = func == WindowFn::Min;
    let mut out: Vec<Option<bool>> = vec![None; num_rows];
    for part in ordered {
        let (mut acc, mut gs): (Option<bool>, usize) = (None, 0);
        for pos in 0..part.len() {
            let row = part[pos];
            if arr.is_valid(row) {
                let v = arr.value(row);
                acc = Some(match acc {
                    None => v,
                    Some(a) if is_min => a && v,
                    Some(a) => a || v,
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
    Ok(Arc::new(BooleanArray::from(out)))
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

    /// Independent brute-force `rank` oracle (shares no sort code with production) over data
    /// whose i64 partition keys collide in the 8-byte sort prefix — they differ only in their
    /// lowest byte, which the big-endian prefix does not capture — and whose order keys are
    /// heavily tied. Both are exactly the cases the prefix-accelerated global sort in
    /// [`ordered_partitions_by_global_sort`] must resolve through its full-row + inline-index
    /// tie-break; a prefix that silently mis-grouped or mis-ordered these would diverge here.
    #[test]
    fn rank_matches_bruteforce_on_prefix_collisions() {
        let n = 500usize;
        // (i % 6) picks a high-byte group; (i % 2) perturbs only the low byte, so rows in the
        // same group share the leading prefix bytes but are distinct partitions.
        let part_vals: Vec<i64> = (0..n as i64).map(|i| (i % 6) * 256 + (i % 2)).collect();
        let ord_vals: Vec<i64> = (0..n as i64).map(|i| (i * 17) % 7).collect(); // many ties
        let part = i64s(&part_vals);
        let order = [asc(i64s(&ord_vals))];
        let call = [WindowCall {
            func: WindowFn::Rank,
            values: None,
            offset: 0,
            frame: None,
        }];
        // Both the serial kernel (huge threshold) and the bucket-parallel path (threshold 1).
        for threshold in [1usize, 1 << 20] {
            let got =
                window_with(std::slice::from_ref(&part), &order, &call, n, threshold).unwrap();
            let got = ints(&got[0]);
            for r in 0..n {
                // rank = 1 + number of same-partition rows with a strictly smaller order key.
                let expected = 1
                    + (0..n)
                        .filter(|&s| part_vals[s] == part_vals[r] && ord_vals[s] < ord_vals[r])
                        .count() as i64;
                assert_eq!(got[r], expected, "row {r} threshold {threshold}");
            }
        }
    }

    /// Independent brute-force oracle for the exact `win-rank` shape: an i64 PARTITION BY and
    /// a **f64 ORDER BY DESC** with negative values and ties. This is the case the packed
    /// numeric fast path ([`try_ordered_partitions_packed`]) handles via `f64_ordered` + the
    /// descending-key inversion — the riskiest packing — so a wrong float total order or a
    /// botched DESC would diverge from the hand-computed rank here.
    #[test]
    fn rank_matches_bruteforce_float_desc() {
        let n = 400usize;
        let part_vals: Vec<i64> = (0..n as i64).map(|i| i % 20).collect();
        let ord_vals: Vec<f64> = (0..n).map(|i| ((i as f64 * 13.0) % 11.0) - 5.0).collect();
        let part = i64s(&part_vals);
        let ord: ArrayRef = Arc::new(Float64Array::from(ord_vals.clone()));
        let order = [(
            ord,
            SortOptions {
                descending: true,
                nulls_first: false,
            },
        )];
        let call = [WindowCall {
            func: WindowFn::Rank,
            values: None,
            offset: 0,
            frame: None,
        }];
        for threshold in [1usize, 1 << 20] {
            let out =
                window_with(std::slice::from_ref(&part), &order, &call, n, threshold).unwrap();
            let got = ints(&out[0]);
            for r in 0..n {
                // DESC rank = 1 + number of same-partition rows with a strictly GREATER key.
                let expected = 1
                    + (0..n)
                        .filter(|&s| part_vals[s] == part_vals[r] && ord_vals[s] > ord_vals[r])
                        .count() as i64;
                assert_eq!(got[r], expected, "row {r} threshold {threshold}");
            }
        }
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
            let par = window_with(std::slice::from_ref(&part), &order, &funcs, n, 1).unwrap();
            let ser = window_serial(std::slice::from_ref(&part), &order, &funcs, n).unwrap();
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
            let par = window_with(std::slice::from_ref(&part), &[], &funcs, n, 1).unwrap();
            let ser = window_serial(std::slice::from_ref(&part), &[], &funcs, n).unwrap();
            assert_eq!(
                ints(&par[0]),
                ints(&ser[0]),
                "parallel != serial for {func:?}"
            );
        }
    }

    /// A negative `lag`/`lead` offset flips direction (`lag(v, -1)` == `lead(v, 1)`,
    /// `lead(v, -1)` == `lag(v, 1)`), matching DuckDB — a `.max(0)` clamp returned the
    /// current row for every negative offset instead.
    #[test]
    fn negative_offset_lag_lead_flip_direction() {
        let order = i64s(&[1, 2, 3, 4]);
        let vals = i64s(&[10, 20, 30, 40]);
        let opt_ints = |a: &ArrayRef| -> Vec<Option<i64>> {
            let x = a.as_any().downcast_ref::<Int64Array>().unwrap();
            (0..x.len())
                .map(|i| x.is_valid(i).then(|| x.value(i)))
                .collect()
        };
        // lag(v, -1) == lead(v, 1): [20, 30, 40, NULL]
        let f = [WindowCall {
            func: WindowFn::Lag,
            values: Some(vals.clone()),
            offset: -1,
            frame: None,
        }];
        let out = window(&[], &[asc(order.clone())], &f, 4).unwrap();
        assert_eq!(opt_ints(&out[0]), vec![Some(20), Some(30), Some(40), None]);
        // lead(v, -1) == lag(v, 1): [NULL, 10, 20, 30]
        let f = [WindowCall {
            func: WindowFn::Lead,
            values: Some(vals),
            offset: -1,
            frame: None,
        }];
        let out = window(&[], &[asc(order)], &f, 4).unwrap();
        assert_eq!(opt_ints(&out[0]), vec![None, Some(10), Some(20), Some(30)]);
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
    fn rank_treats_signed_zero_and_nan_on_engine_float_identity() {
        // ORDER BY a float key must rank on the engine's float identity, not raw bits:
        // `-0.0` and `0.0` are one value (peers → same RANK), and a *negative* NaN (what
        // `0.0/0.0` yields on x86) ranks greatest (last), not below -inf. Order key
        // [-0.0, 0.0, 1.0, -NaN]: sorted identity order is {-0.0,0.0} peers, then 1.0,
        // then NaN → RANK 1,1,3,4 and DENSE_RANK 1,1,2,3. With the raw RowConverter the
        // two zeros split (1,2,3,4) and the -NaN sorted first — disagreeing with GROUP BY.
        let neg_nan = f64::from_bits(0xfff8_0000_0000_0000);
        let order: ArrayRef = Arc::new(Float64Array::from(vec![-0.0, 0.0, 1.0, neg_nan]));
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
        let out = window(&[], &[asc(order)], &funcs, 4).unwrap();
        assert_eq!(
            ints(&out[0]),
            vec![1, 1, 3, 4],
            "RANK: signed zeros are peers"
        );
        assert_eq!(
            ints(&out[1]),
            vec![1, 1, 2, 3],
            "DENSE_RANK: one -NaN group last"
        );
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
    fn min_max_over_booleans() {
        use arrow::array::BooleanArray;
        // partition a: [true, false, null] → min=false (AND), max=true (OR).
        // partition b: [true] → min=max=true.
        let part = strs(&["a", "a", "a", "b"]);
        let vals: ArrayRef = Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(false),
            None,
            Some(true),
        ]));
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
        // Whole-partition (no ORDER BY).
        let out = window(std::slice::from_ref(&part), &[], &funcs, 4).unwrap();
        let mn = out[0].as_any().downcast_ref::<BooleanArray>().unwrap();
        let mx = out[1].as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(!mn.value(0)); // a: AND
        assert!(mx.value(0)); // a: OR
        assert!(mn.value(3)); // b
        assert!(mx.value(3));

        // Running (with an ORDER BY) — cumulative AND/OR.
        let ord = i64s(&[1, 2, 3, 1]);
        let out = window(std::slice::from_ref(&part), &[asc(ord)], &funcs, 4).unwrap();
        let mn = out[0].as_any().downcast_ref::<BooleanArray>().unwrap();
        let mx = out[1].as_any().downcast_ref::<BooleanArray>().unwrap();
        // a ordered [true, false, null]: min running = [true, false, false]; max = [true,true,true].
        assert!(mn.value(0));
        assert!(!mn.value(1));
        assert!(mx.value(1));
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
            let serial =
                window_with(std::slice::from_ref(&part), &order, &funcs, n, usize::MAX).unwrap();
            let parallel = window_with(std::slice::from_ref(&part), &order, &funcs, n, 1).unwrap();
            assert_eq!(opt_ints(&serial[0]), opt_ints(&parallel[0]), "{func:?}");
        }
    }

    /// `window()` must route a value function that carries an explicit frame to the
    /// frame-aware path (SQL's default value frame). `last_value` over
    /// `RANGE UNBOUNDED PRECEDING TO CURRENT ROW` with a tied order key is the current
    /// peer group's value, not the whole-partition last (the frameless path).
    #[test]
    fn last_value_with_range_frame_is_running() {
        use crate::window_frame::{Frame, FrameBound, FrameUnit};
        // Order key [10,10,20,20,30]: peer groups {0,1},{2,3},{4}.
        let ord = i64s(&[10, 10, 20, 20, 30]);
        let vals = i64s(&[1, 2, 3, 4, 5]);
        let range_running = Frame {
            unit: FrameUnit::Range,
            start: FrameBound::UnboundedPreceding,
            end: FrameBound::CurrentRow,
        };
        let framed = [WindowCall {
            func: WindowFn::LastValue,
            values: Some(vals.clone()),
            offset: 1,
            frame: Some(range_running),
        }];
        let out = window(&[], &[asc(ord.clone())], &framed, 5).unwrap();
        assert_eq!(ints(&out[0]), vec![2, 2, 4, 4, 5]);

        // Frameless last_value stays whole-partition (the DataFrame default).
        let frameless = [WindowCall {
            func: WindowFn::LastValue,
            values: Some(vals),
            offset: 1,
            frame: None,
        }];
        let out2 = window(&[], &[asc(ord)], &frameless, 5).unwrap();
        assert_eq!(ints(&out2[0]), vec![5, 5, 5, 5, 5]);
    }
}
