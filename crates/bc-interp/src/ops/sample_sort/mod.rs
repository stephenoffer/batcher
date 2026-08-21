//! Single-node parallel full sort by **sample-sort**.
//!
//! Range-partition the rows by the leading key (sampled quantile boundaries) and sort each
//! range in parallel. The ranges are globally ordered relative to each other, so the sorted
//! relation is simply the ranges in key order — no final merge, and no concat: the executor
//! consumes a `Vec<RecordBatch>` already.
//!
//! **The payload is gathered exactly once.** Routing produces per-range *row indices*, not
//! range batches; each range sorts a cheap gather of just its key columns, then composes
//! that permutation with its row indices and gathers every column once. Materializing the
//! ranges up front (and again to sort them, and a third time to concatenate) copied every
//! column three times — on a 5 M-row, 6-column sort that was two thirds of the work.
//!
//! **A temporal leading key routes as `i64`, and used not to.** `DataType::is_integer()` is
//! false for `Date32`, `Timestamp` and their neighbours, so the routing below fell to its
//! `_ => return Ok(None)` arm and a date sort ran **serially** — on 96 cores. It is not a
//! corner case: `ORDER BY <date>` and `ORDER BY <timestamp>` are the commonest sorts an
//! analytic query writes, and the gap they left was 16x. Measured on 6 M rows,
//! `ORDER BY <date>` with 2,500 distinct values: **617 ms serial against 38 ms** for the same
//! sort on an `Int64` column holding *more* distinct values, which is the same shape taking
//! this path. See [`is_integer_ordered_temporal`] for which types are admitted and why
//! `Interval` is not.
//!
//! This is the single-node form of the distributed range sort (`dist/flight_sort.py`), so
//! one implementation serves both: the boundaries and the routing come from the same
//! `bc_runtime::shuffle` range partitioners.

use arrow::array::{
    Array, ArrayRef, GenericStringArray, OffsetSizeTrait, RecordBatch, UInt32Array,
};
use arrow::compute::take;
use arrow::datatypes::DataType;
use std::sync::Arc;

use bc_ir::SortKey;
use bc_runtime::byte_key::is_byte_key;
use rayon::prelude::*;

mod lowcard;

use crate::error::InterpError;

/// Temporal types whose physical value is a signed integer *whose numeric order is the type's
/// order*, so the leading-key routing may cast them to `i64` and compare them as integers.
///
/// Days since the epoch, milliseconds, microseconds and nanoseconds all order as their integer
/// value does, in every unit and with or without a time zone — a zone shifts the wall-clock
/// rendering, never the instant the value encodes, and the routing orders instants.
///
/// Two types are deliberately absent, for different reasons, and both are pinned by
/// `every_admitted_temporal_type_casts_to_i64`.
///
/// `Interval` is not one integer at all: `MonthDayNano` is three fields in 128 bits, and no
/// `i64` orders it — a month is not a fixed number of days, so two intervals are not even
/// totally ordered by duration.
///
/// `Time32` *is* one integer, and arrow simply will not cast it: `can_cast_types(Time32, Int64)`
/// is false in arrow-rs, and arrow C++ refuses `date32 -> int64` as well, so this is a gap the
/// two implementations already answer differently rather than a hypothetical. Admitting it would
/// turn `ORDER BY <a time column>` from a working serial sort into a **raise**. The routing
/// below declines on a failed cast for the same reason, so this list being wrong costs a missed
/// optimization and never an error.
fn is_integer_ordered_temporal(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Date32
            | DataType::Date64
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
    )
}

/// Rows below which the single-node sample-sort stays serial — the sampling + range
/// partition overhead only pays off on a large full sort.
const PARALLEL_SORT_MIN_ROWS: usize = 1 << 17;

/// Rows sampled to estimate the quantile boundaries. Enough to balance 64 ranges well
/// while staying a negligible fraction of a large sort.
const SAMPLE_TARGET: usize = 8192;

/// One routed range: the rows it owns, and whether its key was *proved* constant.
///
/// The flag is what [`split_constant_ranges`] learned and used to throw away. It proves the
/// range's sort permutation is the identity — every row ties on the single sort key, ties
/// resolve to input order, and `bucket_indices` built the range in ascending input order — so
/// the per-range worker can skip gathering the key column and sorting it. That is not a
/// micro-optimization on this shape: it is the whole per-range cost. `ORDER BY <a 7-value
/// string>` splits into `parts` constant pieces that between them cover **every row of the
/// column**, so gathering their keys is a full `take` of a 6 M-row `Utf8` array — offsets,
/// data buffer and all — performed only for `sort_indices_of` to hand back `0..n` unchanged.
///
/// Set only on the single-key path (`keys.len() == 1`). With two keys a range constant in the
/// leading key still has to sort by the second, so the identity does not follow.
struct Range {
    idx: Vec<u32>,
    constant: bool,
}

/// Parallel single-node full sort by sample-sort.
///
/// Returns `None` (caller uses the serial `sort_batch`) unless it applies: a full sort (no
/// `LIMIT` — top-N is already cheap), a large input, and a **float, integer, temporal, or
/// string** leading key (the boundaries route it exactly — floats by `f64`, integers and
/// temporals by `i64`, strings lexicographically by bytes). Multi-key sorts are supported: rows bucket by the
/// leading key (equal leading keys never span a boundary), then each range sorts by the
/// full key list, so a plain concatenation in leading-key order is the globally sorted
/// multi-key relation.
pub(crate) fn parallel_sort_batch(
    batch: &RecordBatch,
    keys: &[SortKey],
    limit: Option<usize>,
) -> Result<Option<Vec<RecordBatch>>, InterpError> {
    let Some(k0) = keys.first() else {
        return Ok(None);
    };
    if limit.is_some() || batch.num_rows() < PARALLEL_SORT_MIN_ROWS {
        return Ok(None);
    }
    // Evaluate every sort key once over the whole batch: the per-range sorts reuse these
    // arrays, so a computed `ORDER BY` expression is evaluated once, not once per range.
    //
    // Canonicalize here rather than leaving it to `sort_indices_of`, so the *routing* and the
    // per-range *sort* rank a float the same way. `canon_float_array` rewrites `-0.0` to `0.0`
    // and canonicalizes NaN; without it the row encoding below orders `-0.0` strictly below
    // `0.0` (arrow's row format is a total order) while the per-range sort, seeing them
    // canonicalized, calls them equal and falls back to input order. Rows would then be routed
    // to different ranges than the order they belong in, and the concatenation would put a
    // later `-0.0` row ahead of an earlier `0.0` one — a wrong answer that appears only with a
    // signed zero or a NaN in a key that is skewed enough to take the composite path. The call
    // is free on the common case: it returns the same `Arc` when there is nothing to rewrite,
    // and `sort_indices_of` re-applying it is then idempotent.
    let key_arrays: Vec<ArrayRef> = keys
        .iter()
        .map(|k| Ok(super::normalize_sort_key(k.expr.eval(batch)?)))
        .collect::<Result<_, InterpError>>()?;
    let key = &key_arrays[0];
    // 64 ranges saturate the gather: measured 32/64/96/128/192 ranges on 96 cores at
    // 157/123/124/120/116 ms — past 64 the sort is memory-bandwidth bound, not
    // parallelism bound, so more ranges only add sampling and concat overhead.
    let parts = rayon::current_num_threads().clamp(2, 64);

    // A SINGLE LOW-CARDINALITY STRING KEY routes by **rank**, not by boundary search.
    // Quantile boundaries drawn from seven distinct values cannot separate `parts` ranges, so
    // on that shape the routing below binary-searches ~64 duplicate boundaries per row and
    // `split_constant_ranges` then reads every row of every oversized bucket to prove it
    // constant. Ranking asks the question once per distinct value, and its buckets are
    // constant *by construction* — one bucket is one key value — so the proof pass is not
    // merely cheaper, it is unnecessary. `lowcard` declines above its distinct-count cap,
    // which is exactly where the sample-sort is the right algorithm again.
    if keys.len() == 1 {
        if let Some((part_of, ranks)) = lowcard::rank_part_of(key, k0.descending, k0.nulls_first) {
            let buckets = bc_runtime::shuffle::bucket_indices(&part_of, ranks);
            let fair = (batch.num_rows() / parts).max(1);
            // Every bucket is one key value, so it is already in its final order and may be
            // cut anywhere — which makes the pieces plain **slices** of the buckets rather
            // than the owned index vectors the sample-sort's `Range` carries. On this shape
            // the index list is the whole relation, so copying it to split it is a full extra
            // pass over 4 bytes a row for nothing.
            let pieces: Vec<&[u32]> = buckets
                .iter()
                .filter(|b| !b.is_empty())
                .flat_map(|b| b.chunks(b.len().div_ceil(b.len().div_ceil(fair)).max(1)))
                .collect();
            // Every output column is the key on the shape this arm exists for
            // (`SELECT k FROM t ORDER BY k`), and a bucket holds exactly one key value — so
            // the piece's answer is **one row repeated**, not a gather of its rows.
            //
            // That is worth distinguishing because a repeated index is not merely a smaller
            // gather, it is a different memory access pattern: every read hits the same cache
            // line and only the writes remain. Measured on `l_shipmode` (6 M rows, seven
            // distinct values, ~8 bytes each) a single-threaded arrow `take` costs **668.8 ms
            // scattered against 38.5 ms with a constant index** — 17x, and this arm was
            // spending essentially all of its time in the scattered one.
            //
            // `Arc`-identity again (see `gather_ranges`): the key here is the array
            // `normalize_sort_key` returned, and only pointer identity proves it was not
            // rewritten and so still carries the column's stored values.
            let constant_out = batch.columns().iter().all(|c| Arc::ptr_eq(c, key));
            let sorted: Vec<RecordBatch> = pieces
                .par_iter()
                .map(|idx| {
                    if constant_out {
                        if let Some(&rep) = idx.first() {
                            let repeated = UInt32Array::from(vec![rep; idx.len()]);
                            let columns = batch
                                .columns()
                                .iter()
                                .map(|c| take(c.as_ref(), &repeated, None))
                                .collect::<Result<Vec<_>, _>>()?;
                            return Ok(RecordBatch::try_new(batch.schema(), columns)?);
                        }
                    }
                    bc_runtime::shuffle::gather_rows(batch, idx).map_err(InterpError::from)
                })
                .collect::<Result<_, InterpError>>()?;
            return Ok(Some(sorted));
        }
    }

    // Route each row to a range, as *indices only*. Gathering the payload into range
    // batches here (and again to sort each one, and a third time to concatenate) copies
    // every column three times; composing the range's indices with its sort permutation
    // gathers exactly once.
    let part_of = match key.data_type() {
        DataType::Float64 | DataType::Float32 => {
            let key_f64 = arrow::compute::cast(key, &DataType::Float64)?;
            let keyv = key_f64
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .expect("cast to Float64");
            let Some(b) = sample_boundaries_f64(keyv, parts) else {
                return Ok(None);
            };
            bc_runtime::shuffle::range_part_of_f64(
                &key_f64,
                &b,
                parts,
                k0.nulls_first,
                k0.descending,
            )?
        }
        // Every byte-lexicographic key — text and binary alike — routes through one sampler
        // and one partitioner. Binary used to fall past this match to its `_` arm, so a
        // `Binary` `ORDER BY` ran serially however many cores the machine had.
        dt if is_byte_key(dt) => {
            let Some(b) = sample_boundaries_bytes(key, parts) else {
                return Ok(None);
            };
            bc_runtime::shuffle::range_part_of_bytes(key, &b, parts, k0.nulls_first, k0.descending)?
        }
        dt if dt.is_integer() || is_integer_ordered_temporal(dt) => {
            // Routing compares the leading key as i64. That is order-preserving for every
            // integer width except a `UInt64` value above `i64::MAX`, which has no i64 at all
            // — routing it as one would place it by a wrapped or nulled value instead of by
            // its true, largest unsigned magnitude, silently reordering a descending or
            // nulls-first sort. `UInt64` is therefore checked against the widening cast, which
            // turns exactly those values into nulls, and declined: the caller's serial sort
            // compares `u64` by unsigned order and is correct. (A `UInt64` column that fits in
            // i64 keeps the parallel fast path.) Every other width widens exactly.
            if matches!(dt, DataType::UInt64) {
                let Ok(widened) = arrow::compute::cast(key, &DataType::Int64) else {
                    return Ok(None);
                };
                if widened.null_count() > key.null_count() {
                    return Ok(None);
                }
            }
            // Sampled and routed in the key's **native width** — see
            // `bc_runtime::shuffle::int_buckets` for why the `Int64` materialization this used
            // to do was 41% of a two-key sort.
            let Some(b) = sample_boundaries_int(key, parts) else {
                return Ok(None);
            };
            bc_runtime::shuffle::range_part_of_i64(key, &b, parts, k0.nulls_first, k0.descending)?
        }
        _ => return Ok(None),
    };
    let mut buckets = bc_runtime::shuffle::bucket_indices(&part_of, parts);
    let mut reverse = k0.descending;

    // A LOW-CARDINALITY LEADING KEY cannot separate `parts` ranges: `ORDER BY flag, price`
    // with three distinct flags piles every row into ~3 buckets, and each then pays a SERIAL
    // multi-key lexsort of its share — measured ~11x DuckDB on a 6M-row two-key sort, worse
    // than not parallelizing at all. When the routing comes out that skewed, re-route by the
    // FULL COMPOSITE key: its encoded byte order *is* the multi-key order (each key's
    // ASC/DESC and nulls placement are baked into the encoding), so the ranges stay globally
    // ordered, every core gets an even share, and no final reverse is needed (the encoding
    // already carries the leading key's direction).
    //
    // A **single** low-cardinality key has the same problem and cannot use that fallback: with
    // one key the composite encoding ties for every row in a range, so it would route them all
    // back together. It has a cheaper answer instead. A range whose key is *constant* is
    // already in its final order — every row ties, and ties resolve to input order, which is
    // exactly the order `bucket_indices` handed it — so it can be cut at any point and the
    // pieces, concatenated in order, reproduce it exactly. [`split_constant_ranges`] does that,
    // turning "seven ranges on ninety-six cores" into `parts` even ones.
    //
    // Note what this does and does not fix. The per-range *sort* was never the cost on this
    // shape: `already_ordered` settles a constant range in one pass, so the ranges were cheap
    // and merely too few. The **payload gather** is the cost, and it is per-range — so
    // splitting is what puts it on every core.
    //
    // The two branches trigger at different thresholds, deliberately. The composite re-route
    // pays for a whole row encoding of every key, so it wants real skew (3x) before it is
    // worth it. Splitting costs one short-circuiting scan of a range that is *already* too big
    // for one core, so it pays as soon as a range exceeds a fair share — and it has to, because
    // the failure here is not skew at all: twenty-five evenly-sized ranges on ninety-six cores
    // are perfectly balanced and still leave seventy-one cores idle. A 3x skew test cannot see
    // that, and the shape is exactly `ORDER BY <a 25-value column>`.
    let fair_share = (batch.num_rows() / parts).max(1);
    let max_bucket = buckets.iter().map(Vec::len).max().unwrap_or(0);
    let split_single_key = keys.len() == 1 && max_bucket > fair_share;
    // The composite encoding, kept when the re-route uses it: a range then sorts by comparing
    // rows of *this* encoding instead of building a second one over its own key gather.
    let mut composite_rows: Option<arrow::row::Rows> = None;
    if keys.len() > 1 && max_bucket > fair_share.saturating_mul(3) {
        if let Some((cp, rows)) = composite_part_of(&key_arrays, keys, parts)? {
            buckets = bc_runtime::shuffle::bucket_indices(&cp, parts);
            reverse = false;
            composite_rows = Some(rows);
        }
    }
    let ranges: Vec<Range> = if split_single_key {
        split_constant_ranges(buckets, key, fair_share, reverse)
    } else {
        buckets
            .into_iter()
            .map(|idx| Range {
                idx,
                constant: false,
            })
            .collect()
    };

    let mut sorted = gather_ranges(batch, &ranges, &key_arrays, keys, composite_rows.as_ref())?;

    // Ranges are globally ordered relative to each other, so the sorted relation is simply
    // the ranges in key order.
    if reverse {
        sorted.reverse();
    }
    Ok(Some(sorted))
}

/// Sort and gather every range in parallel, producing the sorted relation as pieces.
///
/// Each range sorts independently: gather only its *key* columns (one or two narrow arrays),
/// sort those, then map the range-local permutation back through the range's row indices and
/// gather the payload once. A range known constant skips both — its permutation is the
/// identity (see [`Range`]) — and goes straight to the one payload gather every range pays.
///
/// `composite_rows`, when present, is the routing encoding: the range's rows are already
/// ordered by it, so they sort by comparing rows of *that* encoding rather than building a
/// second one over their own key gather.
fn gather_ranges(
    batch: &RecordBatch,
    ranges: &[Range],
    key_arrays: &[ArrayRef],
    keys: &[SortKey],
    composite_rows: Option<&arrow::row::Rows>,
) -> Result<Vec<RecordBatch>, InterpError> {
    // Which output column *is* one of the sort keys, so the sorted form can be produced from
    // the key gather this range already paid for instead of gathering the column a second time.
    //
    // The general arm below gathers each range's key columns to sort them, and then gathers the
    // payload. When a key column is also an output column — `ORDER BY k` selecting `k` — that
    // column is gathered **twice out of the whole relation**, and both gathers are the random
    // access a sort's permutation implies. It is invisible on a wide payload (one extra column
    // among many) and it is the dominant cost on a narrow one, which is exactly where this
    // operator loses: measured on a 6 M-row `ORDER BY` of a single `int64`, Batcher pays a
    // fixed ~16 ms against DuckDB's ~7 ms while *beating* DuckDB per additional payload column
    // (5.4 ms/col against 9.6 ms/col). Re-gathering the key is a large part of that fixed cost.
    //
    // Matched by `Arc` identity rather than by name or expression, and that is a correctness
    // requirement rather than a shortcut. `key_arrays` holds keys put through
    // `normalize_sort_key`, which canonicalizes `-0.0` to `0.0` and quiets `NaN` so routing and
    // per-range sorting rank a float the same way. Those are *ordering* equivalences, not value
    // equivalences: emitting a normalized key as the output column would turn a stored `-0.0`
    // into `0.0` and a signalling `NaN` into a quiet one. `canon_float_array` returns the same
    // `Arc` when it rewrites nothing, so pointer identity says precisely "this array was not
    // rewritten", which is the only case where reuse is value-preserving.
    let reuse: Vec<Option<usize>> = batch
        .columns()
        .iter()
        .map(|col| key_arrays.iter().position(|k| Arc::ptr_eq(k, col)))
        .collect();
    // **Every** column, not merely some. Reuse pays by removing the whole-relation gather, and
    // it only removes it when nothing is left behind to gather: a range then produces its output
    // entirely from key arrays it has already materialized, and the `global` index composition
    // is not needed either. Applied to a *partial* match it merely swaps one column's random
    // gather for a local one while the rest still pay the full pass, and that measured a **9.9%
    // regression** on a four-column sort keyed on two of them (`op-sort-multikey-wide`, 127.4 ms
    // -> 140.0 ms interleaved) against a 5.6-7.9% gain on the shapes where the payload is
    // exactly the keys. So the gate is all-or-nothing.
    let reuse_all = reuse.iter().all(Option::is_some);
    ranges
        .par_iter()
        .map(|range| -> Result<RecordBatch, InterpError> {
            let idx = &range.idx;
            if range.constant {
                return Ok(bc_runtime::shuffle::gather_rows(batch, idx)?);
            }
            // Ties on the whole encoded key are rows equal on every sort key, and they break
            // on the input row number — the same input order `stable_lexsort_indices` resolves
            // them to, which is why `sort_unstable_by` is safe here (a unique final key makes
            // the comparator a total order).
            if let Some(rows) = composite_rows {
                let mut order = idx.clone();
                order.sort_unstable_by(|&a, &b| {
                    rows.row(a as usize)
                        .cmp(&rows.row(b as usize))
                        .then(a.cmp(&b))
                });
                return Ok(bc_runtime::shuffle::gather_rows(batch, &order)?);
            }
            let take_idx = UInt32Array::from(idx.clone());
            let range_keys: Vec<ArrayRef> = key_arrays
                .iter()
                .map(|a| take(a.as_ref(), &take_idx, None))
                .collect::<Result<_, _>>()?;
            let local = super::sort_indices_of(&range_keys, keys)?;
            if !reuse_all {
                let global: Vec<u32> = local.values().iter().map(|&l| idx[l as usize]).collect();
                return Ok(bc_runtime::shuffle::gather_rows(batch, &global)?);
            }
            // Every column permutes the range's own key gather — a few tens of thousands of
            // rows, cache-resident — so the whole-relation gather does not happen at all.
            let columns: Vec<ArrayRef> = reuse
                .iter()
                .map(|m| {
                    take(
                        range_keys[m.expect("all-or-nothing gate")].as_ref(),
                        &local,
                        None,
                    )
                })
                .collect::<Result<_, _>>()?;
            Ok(RecordBatch::try_new(batch.schema(), columns)?)
        })
        .collect()
}

/// Whether every row of `idx` carries the same string key, so the range is already in its
/// final order and may be cut anywhere — and, on the single-key path, sorted by the identity
/// permutation rather than sorted at all (see [`Range`]).
///
/// Restricted to the string key types on purpose. This exists for the shape the goal of the
/// whole module keeps running into — `ORDER BY <a column with seven values>` — and a
/// low-cardinality *fixed-width* key does not have the problem in the same way: its ranges
/// radix-sort in `O(n)` and its values are compared without touching a second buffer. Widening
/// this to every type means a `make_comparator` dynamic dispatch per row to answer a question
/// only the string case is asking.
///
/// A range of nulls counts as constant: routing groups nulls together, and they are all equal
/// to each other for ordering, so the same cut-anywhere argument applies.
///
/// **The scan is parallel, and that is not a micro-optimization.** The `false` answer costs
/// two comparisons (`all` short-circuits), but the `true` answer — the one this is asked for —
/// has to read every row of the range, and the ranges it is asked about are by definition
/// oversized. Serially that is a fresh single-threaded pass over most of the column, added to
/// a sort whose every other phase is already spread across the cores: routing, bucketing and
/// the per-range work all run under rayon. Measured serially it cost more than the imbalance
/// it was there to fix, which made the whole optimization a regression on the exact shape it
/// targets.
fn constant_range(key: &ArrayRef, idx: &[u32]) -> bool {
    fn all_equal<O: OffsetSizeTrait + Sync>(a: &GenericStringArray<O>, idx: &[u32]) -> bool {
        let Some(&first) = idx.first() else {
            return true;
        };
        if a.is_null(first as usize) {
            return idx.par_iter().all(|&i| a.is_null(i as usize));
        }
        let head = a.value(first as usize);
        idx.par_iter()
            .all(|&i| !a.is_null(i as usize) && a.value(i as usize) == head)
    }
    match key.data_type() {
        DataType::Utf8 => key
            .as_any()
            .downcast_ref::<GenericStringArray<i32>>()
            .is_some_and(|a| all_equal(a, idx)),
        DataType::LargeUtf8 => key
            .as_any()
            .downcast_ref::<GenericStringArray<i64>>()
            .is_some_and(|a| all_equal(a, idx)),
        _ => false,
    }
}

/// Cut every constant-key range longer than `target` into contiguous pieces, so a key with
/// fewer distinct values than there are cores still spreads across them.
///
/// The pieces stay in input order and stay adjacent, so concatenating them reproduces the
/// range they came from — that is the whole correctness argument, and it holds only because
/// the range is constant (see [`constant_range`]) and because `bucket_indices` builds each
/// range in ascending input order.
///
/// `reverse` is whether the caller is about to reverse the whole range list for a descending
/// sort. A range's pieces must come out of that in input order, because ties resolve to input
/// order in **both** directions — descending inverts the key comparison, never the tie-break —
/// so they are emitted backwards here precisely so the caller's reverse puts them back.
fn split_constant_ranges(
    buckets: Vec<Vec<u32>>,
    key: &ArrayRef,
    target: usize,
    reverse: bool,
) -> Vec<Range> {
    let mut out: Vec<Range> = Vec::with_capacity(buckets.len());
    for bucket in buckets {
        // A bucket small enough to be one core's share is left whole — but it may still be
        // constant, and saying so is free here (the scan short-circuits on the second row of
        // a non-constant one) while saving the worker a key gather it would otherwise pay.
        // Only ranges *proved* constant carry the flag; an unchecked one is merely sorted the
        // ordinary way, so a wrong `false` costs speed and never correctness.
        if bucket.len() <= target {
            let constant = constant_range(key, &bucket);
            out.push(Range {
                idx: bucket,
                constant,
            });
            continue;
        }
        if !constant_range(key, &bucket) {
            out.push(Range {
                idx: bucket,
                constant: false,
            });
            continue;
        }
        let pieces = bucket.len().div_ceil(target);
        let size = bucket.len().div_ceil(pieces);
        let mut chunks: Vec<Range> = bucket
            .chunks(size)
            .map(|c| Range {
                idx: c.to_vec(),
                constant: true,
            })
            .collect();
        if reverse {
            chunks.reverse();
        }
        out.append(&mut chunks);
    }
    out
}

/// Range-route every row by the **full composite sort key**, for when the leading key alone is
/// too low-cardinality to separate `parts` balanced ranges.
///
/// All sort keys are encoded together through arrow's row format with each key's own
/// `SortOptions`, so a byte-lexicographic compare of the encoding reproduces the multi-key
/// ordering exactly — descending keys and nulls placement included. Boundaries are sampled from
/// that encoding and each row is routed by how many boundaries it sorts after, so buckets come
/// out in ascending composite order (the final sorted order — the caller must NOT reverse).
/// rows equal on the whole key always land together. `None` when there is too little data to
/// sample, leaving the caller on its leading-key routing.
/// Returns the per-row partition **and the encoding it routed by**, because the per-range
/// sort wants exactly that encoding and would otherwise build a second one. A `RowConverter`
/// pass over every row of every key is the dominant cost of a multi-key sort — the encoding
/// is the sort — so paying for it twice roughly doubles the work. `Rows::row(i)` compares
/// byte-lexicographically and its order *is* the multi-key order (each key's direction and
/// nulls placement are baked in, which is the same property the routing relies on), so a
/// range sorts by comparing rows of this encoding with no re-encode and no key gather.
fn composite_part_of(
    key_arrays: &[ArrayRef],
    keys: &[SortKey],
    parts: usize,
) -> Result<Option<(Vec<u32>, arrow::row::Rows)>, InterpError> {
    use arrow::compute::SortOptions;
    use arrow::row::{RowConverter, SortField};

    let fields: Vec<SortField> = key_arrays
        .iter()
        .zip(keys)
        .map(|(a, k)| {
            SortField::new_with_options(
                a.data_type().clone(),
                SortOptions {
                    descending: k.descending,
                    nulls_first: k.nulls_first,
                },
            )
        })
        .collect();
    let converter = RowConverter::new(fields)?;
    let rows = converter.convert_columns(key_arrays)?;
    let n = rows.num_rows();

    let target = SAMPLE_TARGET.min(n).max(parts);
    let stride = (n / target).max(1);
    let mut sample: Vec<&[u8]> = (0..n).step_by(stride).map(|i| rows.row(i).data()).collect();
    if sample.len() < parts {
        return Ok(None);
    }
    sample.sort_unstable();
    let m = sample.len();
    let bounds: Vec<Vec<u8>> = (1..parts)
        .map(|j| sample[(j * m / parts).min(m - 1)].to_vec())
        .collect();

    let part_of: Vec<u32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let r = rows.row(i).data();
            bounds.partition_point(|b| b.as_slice() <= r) as u32
        })
        .collect();
    Ok(Some((part_of, rows)))
}

/// Sample `parts-1` ascending f64 quantile boundaries from a float key column. Returns
/// `None` if fewer than `parts` finite values exist (nothing meaningful to split).
fn sample_boundaries_f64(key: &arrow::array::Float64Array, parts: usize) -> Option<Vec<f64>> {
    let n = key.len();
    let target = SAMPLE_TARGET.min(n).max(parts);
    let stride = (n / target).max(1);
    let mut sample: Vec<f64> = (0..n)
        .step_by(stride)
        .filter(|&i| key.is_valid(i))
        .map(|i| key.value(i))
        .filter(|v| !v.is_nan())
        .collect();
    if sample.len() < parts {
        return None;
    }
    sample.sort_unstable_by(|a, b| a.total_cmp(b));
    let m = sample.len();
    Some(
        (1..parts)
            .map(|j| sample[(j * m / parts).min(m - 1)])
            .collect(),
    )
}

/// Sample `parts-1` ascending i64 quantile boundaries from an integer key column (the
/// exact-integer analog of [`sample_boundaries_f64`]). `None` if too few non-null values.
/// [`sample_boundaries_i64`] over any fixed-width integer or integer-ordered temporal key,
/// read in its **native width** so the column is never materialized as `Int64` to be sampled.
///
/// Widening one sampled value costs a register move; widening the column costs a pass over it,
/// and the sample only ever reads `SAMPLE_TARGET` of the rows.
fn sample_boundaries_int(key: &ArrayRef, parts: usize) -> Option<Vec<i64>> {
    macro_rules! sample {
        ($arr:ty) => {{
            let a = key.as_any().downcast_ref::<$arr>()?;
            let n = a.len();
            let target = SAMPLE_TARGET.min(n).max(parts);
            let stride = (n / target).max(1);
            let mut s: Vec<i64> = (0..n)
                .step_by(stride)
                .filter(|&i| a.is_valid(i))
                .map(|i| a.value(i) as i64)
                .collect();
            if s.len() < parts {
                return None;
            }
            s.sort_unstable();
            let m = s.len();
            return Some((1..parts).map(|j| s[(j * m / parts).min(m - 1)]).collect());
        }};
    }
    use arrow::datatypes::TimeUnit;
    match key.data_type() {
        DataType::Int8 => sample!(arrow::array::Int8Array),
        DataType::Int16 => sample!(arrow::array::Int16Array),
        DataType::Int32 => sample!(arrow::array::Int32Array),
        DataType::Int64 => sample!(arrow::array::Int64Array),
        DataType::UInt8 => sample!(arrow::array::UInt8Array),
        DataType::UInt16 => sample!(arrow::array::UInt16Array),
        DataType::UInt32 => sample!(arrow::array::UInt32Array),
        DataType::Date32 => sample!(arrow::array::Date32Array),
        DataType::Date64 => sample!(arrow::array::Date64Array),
        DataType::Time32(TimeUnit::Second) => sample!(arrow::array::Time32SecondArray),
        DataType::Time32(TimeUnit::Millisecond) => sample!(arrow::array::Time32MillisecondArray),
        DataType::Time64(TimeUnit::Microsecond) => sample!(arrow::array::Time64MicrosecondArray),
        DataType::Time64(TimeUnit::Nanosecond) => sample!(arrow::array::Time64NanosecondArray),
        DataType::Timestamp(TimeUnit::Second, _) => sample!(arrow::array::TimestampSecondArray),
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            sample!(arrow::array::TimestampMillisecondArray)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            sample!(arrow::array::TimestampMicrosecondArray)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            sample!(arrow::array::TimestampNanosecondArray)
        }
        DataType::Duration(TimeUnit::Second) => sample!(arrow::array::DurationSecondArray),
        DataType::Duration(TimeUnit::Millisecond) => {
            sample!(arrow::array::DurationMillisecondArray)
        }
        DataType::Duration(TimeUnit::Microsecond) => {
            sample!(arrow::array::DurationMicrosecondArray)
        }
        DataType::Duration(TimeUnit::Nanosecond) => sample!(arrow::array::DurationNanosecondArray),
        // `UInt64` (and anything new) still goes through the widening cast, which the caller
        // has already proved lossless for this column.
        _ => {
            let widened = arrow::compute::cast(key, &DataType::Int64).ok()?;
            let a = widened
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()?;
            sample_boundaries_i64(a, parts)
        }
    }
}

fn sample_boundaries_i64(key: &arrow::array::Int64Array, parts: usize) -> Option<Vec<i64>> {
    let n = key.len();
    let target = SAMPLE_TARGET.min(n).max(parts);
    let stride = (n / target).max(1);
    let mut sample: Vec<i64> = (0..n)
        .step_by(stride)
        .filter(|&i| key.is_valid(i))
        .map(|i| key.value(i))
        .collect();
    if sample.len() < parts {
        return None;
    }
    sample.sort_unstable();
    let m = sample.len();
    Some(
        (1..parts)
            .map(|j| sample[(j * m / parts).min(m - 1)])
            .collect(),
    )
}

/// Sample `parts-1` ascending quantile boundaries of a **byte-lexicographic** key —
/// `Utf8`, `LargeUtf8`, `Binary`, `LargeBinary` or `FixedSizeBinary` — compared by its bytes,
/// which is exactly how arrow orders all five.
///
/// Returns `None` when the key is not a byte key, when there are too few non-null values to
/// split, or when the sample is so skewed that the boundaries collapse to a single distinct
/// value — in that case every row would route to one bucket and the sample-sort would be pure
/// overhead, so the caller falls back to the serial sort.
fn sample_boundaries_bytes(key: &ArrayRef, parts: usize) -> Option<Vec<Vec<u8>>> {
    // Imported here rather than at module scope: `ByteKeys` names `len`/`is_null`, which every
    // concrete arrow array also has through `Array`, and bringing both into scope makes those
    // calls ambiguous everywhere else in this file.
    use bc_runtime::byte_key::{ByteKeyColumn, ByteKeys};

    let keys = ByteKeyColumn::new(key)?;
    let n = keys.len();
    let target = SAMPLE_TARGET.min(n).max(parts);
    let stride = (n / target).max(1);
    let mut sample: Vec<&[u8]> = (0..n)
        .step_by(stride)
        .filter(|&i| !keys.is_null(i))
        .map(|i| keys.key(i))
        .collect();
    if sample.len() < parts {
        return None;
    }
    sample.sort_unstable();
    let m = sample.len();
    let bounds: Vec<Vec<u8>> = (1..parts)
        .map(|j| sample[(j * m / parts).min(m - 1)].to_vec())
        .collect();
    if bounds.first() == bounds.last() {
        return None;
    }
    Some(bounds)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};
    use arrow::compute::concat_batches;
    use arrow::datatypes::{Field, Schema};
    use bc_expr::Expr;

    use super::super::sort_batch;
    use super::*;

    /// The sample-sort returns the ranges in key order; the sorted relation is their
    /// concatenation, which is what the serial oracle produces as one batch.
    fn concat_ranges(schema: &std::sync::Arc<Schema>, ranges: Vec<RecordBatch>) -> RecordBatch {
        concat_batches(schema, ranges.iter()).unwrap()
    }

    fn str_batch(vals: Vec<Option<&str>>, payload: Vec<i64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8, true),
            Field::new("p", DataType::Int64, false),
        ]));
        let s: ArrayRef = Arc::new(StringArray::from(vals));
        let p: ArrayRef = Arc::new(Int64Array::from(payload));
        RecordBatch::try_new(schema, vec![s, p]).unwrap()
    }

    fn key(descending: bool, nulls_first: bool) -> Vec<SortKey> {
        vec![SortKey {
            expr: Expr::Col { name: "s".into() },
            descending,
            nulls_first,
        }]
    }

    /// The sample-sort must produce exactly what the serial `sort_batch` oracle produces.
    fn assert_matches_serial(batch: &RecordBatch, keys: &[SortKey]) {
        let want = sort_batch(batch, keys, None).unwrap();
        let ranges = parallel_sort_batch(batch, keys, None)
            .unwrap()
            .expect("sample-sort should engage");
        assert_eq!(want, concat_ranges(&batch.schema(), ranges));
    }

    /// A LOW-CARDINALITY leading key (three distinct flags over 200 K rows) cannot separate the
    /// sample-sort's ranges by itself, so the router falls back to composite-key routing
    /// ([`composite_part_of`]). That fallback must still produce exactly the serial oracle's
    /// relation — for an ascending *and* a descending leading key, since the composite encoding
    /// carries each key's direction itself instead of the leading-key path's final reverse.
    #[test]
    fn low_cardinality_leading_key_matches_serial() {
        let n = 200_000usize;
        let mut s: u64 = 7;
        let (mut flags, mut price) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            flags.push(Some(["A", "N", "R"][(s >> 33) as usize % 3]));
            price.push(((s >> 20) % 100_000) as i64);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8, true),
            Field::new("p", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(flags)) as ArrayRef,
                Arc::new(Int64Array::from(price)) as ArrayRef,
            ],
        )
        .unwrap();
        for lead_desc in [false, true] {
            let keys = vec![
                SortKey {
                    expr: Expr::Col { name: "s".into() },
                    descending: lead_desc,
                    nulls_first: false,
                },
                SortKey {
                    expr: Expr::Col { name: "p".into() },
                    descending: true,
                    nulls_first: false,
                },
            ];
            assert_matches_serial(&batch, &keys);
        }
    }

    fn big_str_batch(n: usize, nulls: bool) -> RecordBatch {
        let mut s: u64 = 99;
        let mut vals = Vec::with_capacity(n);
        let mut pay = Vec::with_capacity(n);
        for i in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = (s >> 33) % 5000;
            if nulls && i % 97 == 0 {
                vals.push(None);
            } else {
                vals.push(Some(format!("str_{v:05}")));
            }
            pay.push(i as i64);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8, true),
            Field::new("p", DataType::Int64, false),
        ]));
        let sa: ArrayRef = Arc::new(StringArray::from(vals));
        let pa: ArrayRef = Arc::new(Int64Array::from(pay));
        RecordBatch::try_new(schema, vec![sa, pa]).unwrap()
    }

    #[test]
    fn string_sample_sort_matches_serial_ascending() {
        let b = big_str_batch(1 << 18, false);
        assert_matches_serial(&b, &key(false, false));
    }

    #[test]
    fn string_sample_sort_matches_serial_descending() {
        let b = big_str_batch(1 << 18, false);
        assert_matches_serial(&b, &key(true, false));
    }

    #[test]
    fn string_sample_sort_matches_serial_with_nulls() {
        let b = big_str_batch(1 << 18, true);
        assert_matches_serial(&b, &key(false, false));
        assert_matches_serial(&b, &key(false, true));
        assert_matches_serial(&b, &key(true, true));
    }

    /// Sorting a range by the *routing* encoding must equal the serial oracle across every
    /// combination of direction and nulls placement.
    ///
    /// The reuse rests on the encoding carrying each key's `descending`/`nulls_first` itself,
    /// which is the same property the routing depends on — so the way it could be wrong is a
    /// mismatch between the `SortField` options `composite_part_of` encodes with and the
    /// `SortOptions` the per-range sort would have used. Nulls in both keys and all four
    /// direction pairings is what makes such a mismatch show up as a reordered relation
    /// rather than a coincidentally-equal one.
    #[test]
    fn composite_encoding_reuse_matches_serial_under_every_direction() {
        let n = 200_000usize;
        let mut s: u64 = 11;
        let (mut flags, mut price) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for i in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            flags.push(if i % 7 == 0 {
                None
            } else {
                Some(["A", "N", "R"][(s >> 33) as usize % 3])
            });
            price.push(if i % 11 == 0 {
                None
            } else {
                Some(((s >> 20) % 1000) as i64)
            });
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8, true),
            Field::new("p", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(flags)) as ArrayRef,
                Arc::new(Int64Array::from(price)) as ArrayRef,
            ],
        )
        .unwrap();
        for lead_desc in [false, true] {
            for second_desc in [false, true] {
                for nulls_first in [false, true] {
                    let keys = vec![
                        SortKey {
                            expr: Expr::Col { name: "s".into() },
                            descending: lead_desc,
                            nulls_first,
                        },
                        SortKey {
                            expr: Expr::Col { name: "p".into() },
                            descending: second_desc,
                            nulls_first,
                        },
                    ];
                    assert_matches_serial(&batch, &keys);
                }
            }
        }
    }

    /// The constant-range shortcut ([`Range::constant`]) must produce the serial oracle's
    /// relation, not merely a correctly *ordered* one.
    ///
    /// This is the shape that takes it: a single low-cardinality string key, whose oversized
    /// buckets `split_constant_ranges` proves constant and cuts, and whose pieces then skip
    /// the key gather and the per-range sort entirely. Every row of the column ends up in
    /// such a piece, so if the identity permutation were the wrong answer the result would be
    /// wrong for the whole relation rather than at an edge. Payloads are distinct and
    /// ascending, so the oracle comparison sees tie order — an order-independent check would
    /// pass on a shuffled result, which is exactly the bug this shortcut could introduce.
    #[test]
    fn constant_ranges_take_identity_and_match_serial() {
        let n = 1 << 19;
        let modes = ["AIR", "FOB", "MAIL", "RAIL", "REG AIR", "SHIP", "TRUCK"];
        let vals: Vec<Option<&str>> = (0..n).map(|i| Some(modes[i % modes.len()])).collect();
        let b = str_batch(vals, (0..n as i64).collect());
        assert_matches_serial(&b, &key(false, false));
        assert_matches_serial(&b, &key(true, false));
    }

    /// The same shortcut with nulls in the key: a null range is constant too (all nulls are
    /// equal for ordering), so it takes the identity path and must still land where
    /// `nulls_first` says, in input order, under both directions.
    #[test]
    fn constant_null_range_takes_identity_and_matches_serial() {
        let n = 1 << 19;
        // Seven distinct values, as above: fewer and `sample_boundaries_str` cannot cut
        // `parts` boundaries at all, so the sample-sort declines and the test would prove
        // nothing about the shortcut.
        let modes = ["AIR", "FOB", "MAIL", "RAIL", "REG AIR", "SHIP", "TRUCK"];
        // The null period must be coprime with the boundary sampler's stride (a power of two
        // here). At `i % 4` every sampled index is a multiple of 4 and therefore null, the
        // sample comes back empty, and the sample-sort declines — the test would then be
        // asserting nothing while looking like it passed.
        let vals: Vec<Option<&str>> = (0..n)
            .map(|i| {
                if i % 5 == 0 {
                    None
                } else {
                    Some(modes[i % modes.len()])
                }
            })
            .collect();
        let b = str_batch(vals, (0..n as i64).collect());
        for k in [
            key(false, false),
            key(false, true),
            key(true, false),
            key(true, true),
        ] {
            assert_matches_serial(&b, &k);
        }
    }

    /// A range proved constant is exactly the one whose sort permutation is the identity.
    /// Pinning that directly, rather than only through the end-to-end oracle, is what keeps
    /// the shortcut honest if `split_constant_ranges` is ever widened to a key type where the
    /// argument does not hold.
    #[test]
    fn split_marks_only_proved_constant_ranges() {
        let key_arr: ArrayRef = Arc::new(StringArray::from(vec![
            Some("a"),
            Some("a"),
            Some("a"),
            Some("a"),
            Some("b"),
            Some("c"),
        ]));
        // One oversized constant bucket (cut into pieces, all marked), and one mixed bucket
        // (left whole, unmarked).
        let out = split_constant_ranges(vec![vec![0, 1, 2, 3], vec![4, 5]], &key_arr, 2, false);
        let constant: Vec<bool> = out.iter().map(|r| r.constant).collect();
        let idx: Vec<Vec<u32>> = out.iter().map(|r| r.idx.clone()).collect();
        assert_eq!(idx, vec![vec![0, 1], vec![2, 3], vec![4, 5]]);
        assert_eq!(constant, vec![true, true, false]);
    }

    #[test]
    fn string_sample_sort_is_stable_on_ties() {
        // One distinct-ish key repeated: equal keys must keep input order (payload
        // ascending), exactly as the stable serial sort does.
        let n = 1 << 18;
        let vals: Vec<Option<&str>> = (0..n)
            .map(|i| Some(if i % 2 == 0 { "aaa" } else { "bbb" }))
            .collect();
        let b = str_batch(vals, (0..n as i64).collect());
        // Two distinct values only: boundaries collapse is possible, so just assert the
        // result equals the serial oracle whichever path is taken.
        let want = sort_batch(&b, &key(false, false), None).unwrap();
        let got = match parallel_sort_batch(&b, &key(false, false), None).unwrap() {
            Some(ranges) => concat_ranges(&b.schema(), ranges),
            None => sort_batch(&b, &key(false, false), None).unwrap(),
        };
        assert_eq!(want, got);
    }

    #[test]
    fn uint64_above_i64_max_matches_serial() {
        use arrow::array::UInt64Array;
        // A large UInt64 key column whose values straddle i64::MAX. The serial oracle sorts
        // by *unsigned* order; the sample-sort must produce the identical relation. If the
        // range routing casts the key to i64 (lossy for u64 > i64::MAX), those large values
        // misroute and the relation diverges.
        let n = 1usize << 18;
        let mut vals: Vec<u64> = Vec::with_capacity(n);
        let mut s: u64 = 12345;
        for _ in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Spread across the whole u64 range, so many values exceed i64::MAX.
            vals.push(s);
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::UInt64, false),
            Field::new("p", DataType::Int64, false),
        ]));
        let ka: ArrayRef = Arc::new(UInt64Array::from(vals));
        let pa: ArrayRef = Arc::new(Int64Array::from((0..n as i64).collect::<Vec<_>>()));
        let b = RecordBatch::try_new(schema, vec![ka, pa]).unwrap();
        for (descending, nulls_first) in [(false, false), (true, false), (true, true)] {
            let keys = vec![SortKey {
                expr: Expr::Col { name: "k".into() },
                descending,
                nulls_first,
            }];
            let want = sort_batch(&b, &keys, None).unwrap();
            // The parallel path may engage (and must then match) or decline (falling back to
            // the correct serial sort). It must never return a *wrong* relation — which it did
            // before the lossy-cast guard, for descending / nulls-first on u64 > i64::MAX.
            let got = match parallel_sort_batch(&b, &keys, None).unwrap() {
                Some(ranges) => concat_ranges(&b.schema(), ranges),
                None => sort_batch(&b, &keys, None).unwrap(),
            };
            assert_eq!(
                want, got,
                "uint64 sample-sort diverges (descending={descending} nulls_first={nulls_first})"
            );
        }
    }

    #[test]
    fn uint64_within_i64_range_still_parallelizes() {
        use arrow::array::UInt64Array;
        // A large UInt64 key whose values all fit in i64 must keep the parallel fast path
        // (the lossy-cast guard must not over-decline) and match the serial oracle.
        let n = 1usize << 18;
        let mut vals: Vec<u64> = Vec::with_capacity(n);
        let mut s: u64 = 777;
        for _ in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            vals.push((s >> 1) & (i64::MAX as u64)); // strictly < 2^63
        }
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::UInt64, false),
            Field::new("p", DataType::Int64, false),
        ]));
        let ka: ArrayRef = Arc::new(UInt64Array::from(vals));
        let pa: ArrayRef = Arc::new(Int64Array::from((0..n as i64).collect::<Vec<_>>()));
        let b = RecordBatch::try_new(schema, vec![ka, pa]).unwrap();
        let keys = vec![SortKey {
            expr: Expr::Col { name: "k".into() },
            descending: true,
            nulls_first: false,
        }];
        let want = sort_batch(&b, &keys, None).unwrap();
        let ranges = parallel_sort_batch(&b, &keys, None)
            .unwrap()
            .expect("in-range uint64 must keep the parallel path");
        assert_eq!(want, concat_ranges(&b.schema(), ranges));
    }

    /// A temporal leading key must both *engage* the parallel path and match the serial
    /// oracle. Engaging is half the assertion on purpose: this shape used to be correct and
    /// serial, so a test that only checked the relation would have passed on the bug.
    #[test]
    fn temporal_leading_keys_parallelize_and_match_serial() {
        use arrow::array::{Date32Array, Date64Array, TimestampMicrosecondArray};

        let n = 1usize << 18;
        let mut s: u64 = 99;
        let mut spread = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 40) as i64
        };
        let raw: Vec<i64> = (0..n).map(|_| spread()).collect();

        // Nullable, so the routing's null placement is exercised in every direction.
        let nullable = |f: &dyn Fn(usize) -> i64| -> Vec<Option<i64>> {
            (0..n).map(|i| (i % 11 != 0).then(|| f(i))).collect()
        };
        let vals = nullable(&|i| raw[i]);

        let cases: Vec<(DataType, ArrayRef)> = vec![
            (
                DataType::Date32,
                Arc::new(Date32Array::from(
                    vals.iter().map(|v| v.map(|x| x as i32)).collect::<Vec<_>>(),
                )) as ArrayRef,
            ),
            (
                DataType::Date64,
                Arc::new(Date64Array::from(vals.clone())) as ArrayRef,
            ),
            (
                DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
                Arc::new(TimestampMicrosecondArray::from(vals.clone())) as ArrayRef,
            ),
        ];

        for (dt, ka) in cases {
            let schema = Arc::new(Schema::new(vec![
                Field::new("k", dt.clone(), true),
                Field::new("p", DataType::Int64, false),
            ]));
            let pa: ArrayRef = Arc::new(Int64Array::from((0..n as i64).collect::<Vec<_>>()));
            let b = RecordBatch::try_new(schema, vec![ka, pa]).unwrap();
            for (descending, nulls_first) in [(false, false), (true, false), (false, true)] {
                let keys = vec![SortKey {
                    expr: Expr::Col { name: "k".into() },
                    descending,
                    nulls_first,
                }];
                let want = sort_batch(&b, &keys, None).unwrap();
                let ranges = parallel_sort_batch(&b, &keys, None)
                    .unwrap()
                    .unwrap_or_else(|| panic!("{dt:?} must take the parallel path"));
                assert_eq!(
                    want,
                    concat_ranges(&b.schema(), ranges),
                    "{dt:?} sample-sort diverges (descending={descending} nulls_first={nulls_first})"
                );
            }
        }
    }

    /// Every type [`is_integer_ordered_temporal`] admits must actually be *castable* to `i64`
    /// by arrow, or the routing would raise where it means to decline. Arrow C++ refuses
    /// `date32 -> int64` outright, so this is not a theoretical worry about a sibling
    /// implementation — it is one arrow already answers differently in two places.
    #[test]
    fn every_admitted_temporal_type_casts_to_i64() {
        use arrow::compute::can_cast_types;
        use arrow::datatypes::{IntervalUnit, TimeUnit};
        for dt in [
            DataType::Date32,
            DataType::Date64,
            DataType::Time64(TimeUnit::Microsecond),
            DataType::Time64(TimeUnit::Nanosecond),
            DataType::Timestamp(TimeUnit::Second, None),
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            DataType::Duration(TimeUnit::Millisecond),
        ] {
            assert!(
                is_integer_ordered_temporal(&dt),
                "{dt:?} should be admitted"
            );
            assert!(
                can_cast_types(&dt, &DataType::Int64),
                "{dt:?} is admitted but arrow cannot cast it to Int64"
            );
        }
        // Arrow cannot widen `Time32`, so it must stay out of the routing and keep the serial
        // sort — which is correct, only sequential.
        for dt in [
            DataType::Time32(TimeUnit::Second),
            DataType::Time32(TimeUnit::Millisecond),
        ] {
            assert!(
                !can_cast_types(&dt, &DataType::Int64),
                "{dt:?} now casts — admit it"
            );
            assert!(
                !is_integer_ordered_temporal(&dt),
                "{dt:?} must not be admitted"
            );
        }
        // `Interval` is excluded on the stronger ground that no single integer orders it.
        assert!(!is_integer_ordered_temporal(&DataType::Interval(
            IntervalUnit::MonthDayNano
        )));
    }

    /// `Interval` has no single integer that orders it, so it must keep the serial sort
    /// rather than be routed by a cast that would silently compare the wrong thing.
    #[test]
    fn an_interval_key_is_not_treated_as_an_integer() {
        assert!(!is_integer_ordered_temporal(&DataType::Interval(
            arrow::datatypes::IntervalUnit::MonthDayNano
        )));
        assert!(is_integer_ordered_temporal(&DataType::Date32));
        assert!(is_integer_ordered_temporal(&DataType::Timestamp(
            arrow::datatypes::TimeUnit::Nanosecond,
            Some("UTC".into())
        )));
    }

    #[test]
    fn small_input_declines_sample_sort() {
        let b = str_batch(vec![Some("b"), Some("a")], vec![1, 2]);
        assert!(parallel_sort_batch(&b, &key(false, false), None)
            .unwrap()
            .is_none());
    }

    /// A skewed **float** key carrying `-0.0` and NaN must still match the serial oracle.
    ///
    /// This is the case the composite routing gets wrong if it encodes the raw key: arrow's row
    /// format is a total order and puts `-0.0` strictly below `0.0`, while the per-range sort
    /// sees them canonicalized and calls them equal. Route by one ranking and sort by another
    /// and the concatenation surfaces a later row ahead of an earlier one — with the *values*
    /// still looking correct, which is why the payload column is what this checks.
    ///
    /// The key is deliberately low-cardinality so the composite fallback actually fires; on a
    /// well-spread key this path is never taken and the test would prove nothing.
    #[test]
    fn a_skewed_float_key_with_signed_zero_and_nan_matches_serial() {
        let n = 1 << 18;
        for descending in [false, true] {
            for nulls_first in [false, true] {
                let vals: Vec<Option<f64>> = (0..n)
                    .map(|i| match i % 5 {
                        0 => Some(-0.0),
                        1 => Some(0.0),
                        2 => Some(f64::NAN),
                        3 => None,
                        _ => Some(1.0),
                    })
                    .collect();
                let schema = Arc::new(Schema::new(vec![
                    Field::new("s", DataType::Float64, true),
                    Field::new("p", DataType::Int64, false),
                ]));
                let sa: ArrayRef = Arc::new(arrow::array::Float64Array::from(vals));
                let pa: ArrayRef = Arc::new(Int64Array::from((0..n as i64).collect::<Vec<_>>()));
                let b = RecordBatch::try_new(schema, vec![sa, pa]).unwrap();
                let keys = key(descending, nulls_first);

                let want = sort_batch(&b, &keys, None).unwrap();
                let Some(ranges) = parallel_sort_batch(&b, &keys, None).unwrap() else {
                    continue; // declined; the serial path is the oracle and already correct
                };
                assert_eq!(
                    want,
                    concat_ranges(&b.schema(), ranges),
                    "desc={descending} nulls_first={nulls_first}"
                );
            }
        }
    }

    /// A single low-cardinality string key must still produce exactly the serial oracle's
    /// relation once its oversized ranges are cut into pieces — in both directions, and with
    /// nulls, which form a constant range of their own.
    ///
    /// The descending case is the one that can actually break: the caller reverses the range
    /// list, so pieces emitted in input order would come back reversed and rows tied on the key
    /// would leave in the wrong order. That is invisible to an order-independent comparison,
    /// which is why this asserts full batch equality against `sort_batch`.
    ///
    /// **The split only engages above 7 cores**, because the trigger is
    /// `max_bucket > rows/parts` and seven values fill seven buckets — so it needs `parts > 7`.
    /// On a smaller machine this still asserts the right answer, but through the unsplit path.
    /// [`oversized_constant_ranges_are_actually_split`] is the machine-independent proof that
    /// the splitting itself is exercised; without it, a low-core CI box would run this test
    /// green while never touching the code it was written for.
    /// A **low-cardinality binary** key must take the ranked path too, and produce the
    /// serial relation.
    ///
    /// This is the shape the ranked path exists for wearing a different type: a category, an
    /// enum, a fixed-width code someone stored as bytes. While `rank_part_of` matched `Utf8`
    /// alone it fell through to the quantile routing, which is the wrong algorithm for a
    /// handful of distinct values — every row binary-searches duplicate boundaries and the
    /// proof pass then reads every row of every oversized bucket.
    ///
    /// Asserting the ranked path is *taken* is what makes this more than a duplicate of the
    /// string case: without it the test passes just as happily through the fallback.
    #[test]
    fn a_low_cardinality_binary_key_ranks_and_matches_serial() {
        use arrow::array::FixedSizeBinaryArray;

        let n = 1 << 19;
        let codes: Vec<[u8; 4]> = (0..7u8).map(|c| [c, 0xff, c, 0x01]).collect();
        for with_nulls in [false, true] {
            let vals: Vec<Option<Vec<u8>>> = (0..n)
                .map(|i| {
                    (!(with_nulls && i % 11 == 0))
                        .then(|| codes[(i * 7 + i / 5) % codes.len()].to_vec())
                })
                .collect();
            let key_col: ArrayRef = Arc::new(
                FixedSizeBinaryArray::try_from_sparse_iter_with_size(vals.into_iter(), 4)
                    .expect("uniform width"),
            );
            let b = RecordBatch::try_new(
                Arc::new(Schema::new(vec![
                    Field::new("s", key_col.data_type().clone(), true),
                    Field::new("v", DataType::Int64, false),
                ])),
                vec![
                    key_col.clone(),
                    Arc::new(arrow::array::Int64Array::from(
                        (0..n as i64).collect::<Vec<_>>(),
                    )),
                ],
            )
            .unwrap();
            for descending in [false, true] {
                for nulls_first in [false, true] {
                    assert!(
                        lowcard::rank_part_of(&key_col, descending, nulls_first).is_some(),
                        "seven distinct byte values must reach the ranked path"
                    );
                    let keys = key(descending, nulls_first);
                    let want = sort_batch(&b, &keys, None).unwrap();
                    let ranges = parallel_sort_batch(&b, &keys, None)
                        .unwrap()
                        .expect("sample-sort should engage on a 512 K-row binary key");
                    assert_eq!(
                        want,
                        concat_ranges(&b.schema(), ranges),
                        "nulls={with_nulls} desc={descending} nulls_first={nulls_first}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_split_low_cardinality_string_key_matches_serial() {
        let n = 1 << 19;
        let shipmodes = ["AIR", "RAIL", "TRUCK", "MAIL", "SHIP", "FOB", "REG AIR"];
        for with_nulls in [false, true] {
            let vals: Vec<Option<&str>> = (0..n)
                .map(|i| {
                    if with_nulls && i % 11 == 0 {
                        None
                    } else {
                        Some(shipmodes[(i * 7 + i / 5) % shipmodes.len()])
                    }
                })
                .collect();
            let b = str_batch(vals, (0..n as i64).collect());
            for descending in [false, true] {
                for nulls_first in [false, true] {
                    let keys = key(descending, nulls_first);
                    let want = sort_batch(&b, &keys, None).unwrap();
                    let ranges = parallel_sort_batch(&b, &keys, None)
                        .unwrap()
                        .expect("sample-sort should engage on a 512 K-row string key");
                    assert_eq!(
                        want,
                        concat_ranges(&b.schema(), ranges),
                        "nulls={with_nulls} desc={descending} nulls_first={nulls_first}"
                    );
                }
            }
        }
    }

    /// The split must actually happen, or the test above passes on the unsplit path and proves
    /// nothing about the code it exists to cover.
    ///
    /// Seven distinct values over `parts` ranges cannot fill more than seven of them, so the
    /// unsplit routing is guaranteed to be skewed — the assertion is that the balancer reacts.
    #[test]
    fn oversized_constant_ranges_are_actually_split() {
        let idx: Vec<Vec<u32>> = vec![(0..1000u32).collect(), (1000..1010u32).collect()];
        let key: ArrayRef = Arc::new(StringArray::from(
            (0..1010)
                .map(|i| if i < 1000 { "a" } else { "b" })
                .collect::<Vec<_>>(),
        ));

        let split = split_constant_ranges(idx.clone(), &key, 100, false);
        assert_eq!(
            split.len(),
            11,
            "1000 rows at a target of 100, plus the tail"
        );
        assert_eq!(
            split
                .iter()
                .flat_map(|r| r.idx.iter().copied())
                .collect::<Vec<u32>>(),
            (0..1010u32).collect::<Vec<u32>>(),
            "the pieces must concatenate back to the original row order"
        );

        // Reversed, the pieces of one range come out backwards so the caller's reverse of the
        // whole list restores them.
        let split = split_constant_ranges(idx.clone(), &key, 100, true);
        let mut flat: Vec<Range> = split;
        flat.reverse();
        let head: Vec<u32> = flat
            .iter()
            .skip(1)
            .flat_map(|r| r.idx.iter().copied())
            .collect();
        assert_eq!(
            head,
            (0..1000u32).collect::<Vec<u32>>(),
            "after the caller's reverse, the split range is back in input order"
        );

        // A range that is not constant must never be cut, whatever its size.
        let mixed: ArrayRef = Arc::new(StringArray::from(
            (0..1010)
                .map(|i| if i % 2 == 0 { "a" } else { "b" })
                .collect::<Vec<_>>(),
        ));
        assert_eq!(
            split_constant_ranges(idx, &mixed, 100, false).len(),
            2,
            "a range holding two distinct values has no cut point"
        );
    }

    /// A key with ONE distinct value has no quantile boundaries at all, so the sample-sort
    /// used to decline it and the whole relation fell to the serial sort. Rank routing does
    /// not need boundaries: it produces one bucket, which is constant by construction, and is
    /// cut into core-sized slices. The result is still exactly the serial oracle's — the point
    /// of the assertion — and it is now produced in parallel.
    #[test]
    fn single_distinct_key_is_ranked_rather_than_declined() {
        let n = 1 << 18;
        let vals: Vec<Option<&str>> = (0..n).map(|_| Some("same")).collect();
        let b = str_batch(vals, (0..n as i64).collect());
        let ranges = parallel_sort_batch(&b, &key(false, false), None)
            .unwrap()
            .expect("rank routing needs no boundaries");
        assert!(
            ranges.len() > 1,
            "a constant key must still spread over cores"
        );
        assert_eq!(
            sort_batch(&b, &key(false, false), None).unwrap(),
            concat_ranges(&b.schema(), ranges)
        );
    }

    /// The rank-routed path must reproduce the serial oracle under every direction and null
    /// placement, on the shape it exists for: a seven-value key over enough rows to
    /// parallelize. `lowcard_sort`'s own tests check the permutation; this checks the whole
    /// operator, through the caller that splits and gathers it.
    #[test]
    fn rank_routed_low_cardinality_sort_matches_serial() {
        let n = 1 << 18;
        let values = ["AIR", "FOB", "MAIL", "RAIL", "REG AIR", "SHIP", "TRUCK"];
        let vals: Vec<Option<&str>> = (0..n)
            .map(|i| {
                if i % 97 == 0 {
                    None
                } else {
                    Some(values[i % values.len()])
                }
            })
            .collect();
        let b = str_batch(vals, (0..n as i64).collect());
        for descending in [false, true] {
            for nulls_first in [false, true] {
                assert_matches_serial(&b, &key(descending, nulls_first));
            }
        }
    }

    /// A float sort key that is also an output column keeps its **stored** value.
    ///
    /// `gather_ranges` may produce an output column from the key gather it already paid for,
    /// which is only sound when that key array is the batch's own column. `key_arrays` holds
    /// keys put through `normalize_sort_key`, and for a float that rewrites `-0.0` to `0.0` so
    /// routing and the per-range sort agree on where a signed zero ranks. Reusing the rewritten
    /// array would order the rows correctly and hand back the wrong *values* — a defect a
    /// multiset comparison of the sorted output cannot see, because `-0.0 == 0.0`.
    #[test]
    fn a_reused_float_key_column_keeps_its_signed_zero() {
        let n = super::PARALLEL_SORT_MIN_ROWS * 2;
        // Alternating -0.0 / 0.0 with a spread of other values, so the sample-sort engages.
        let vals: Vec<f64> = (0..n)
            .map(|i| match i % 4 {
                0 => -0.0f64,
                1 => 0.0f64,
                2 => -(i as f64),
                _ => i as f64,
            })
            .collect();
        let negative_zeros = vals
            .iter()
            .filter(|v| v.is_sign_negative() && **v == 0.0)
            .count();
        assert!(negative_zeros > 0, "fixture must contain -0.0");

        let schema = Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, false)]));
        let col: ArrayRef = Arc::new(arrow::array::Float64Array::from(vals));
        let batch = RecordBatch::try_new(schema.clone(), vec![col]).unwrap();
        let keys = vec![SortKey {
            expr: Expr::Col { name: "f".into() },
            descending: false,
            nulls_first: false,
        }];

        let ranges = parallel_sort_batch(&batch, &keys, None)
            .unwrap()
            .expect("a float key above the row floor takes the parallel path");
        let got = concat_ranges(&schema, ranges);
        let out = got
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();

        let kept = (0..out.len())
            .filter(|&i| out.value(i).is_sign_negative() && out.value(i) == 0.0)
            .count();
        assert_eq!(
            kept, negative_zeros,
            "the sort must not rewrite -0.0 to 0.0 on its way out"
        );
        // And it is still the answer the sequential oracle gives.
        let want = sort_batch(&batch, &keys, None).unwrap();
        assert_eq!(got, want);
    }
}
