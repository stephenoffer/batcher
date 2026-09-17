//! Merge a result's small batches before it leaves the engine.
//!
//! Handing a `RecordBatch` to Python is not free and does not get cheaper with size: each one
//! is exported through the Arrow C data interface and re-imported by pyarrow, which rebuilds
//! the schema and its fields per batch. Measured on this boundary with a bare scan whose only
//! variable was the batch count, one batch across it costs about **7.7 us** — 3.8 us in, 3.9 us
//! out — so a 6 M-row filter returning 400 morsels spends ~1.5 ms in conversion alone, and a
//! 60 M-row one spends ~15 ms. That cost is serial: it happens on the calling thread with the
//! GIL held, after every worker has finished, which is where it shows up in an Amdahl fit of
//! the executor's parallel scaling.
//!
//! The remedy is to hand the caller fewer, larger batches. Concatenating costs a copy, so it
//! is only worth doing where the copy is cheaper than the conversions it removes: at roughly
//! 10 GB/s per thread, 3.9 us buys about 40 KB, and the copies run across the pool while the
//! conversions cannot run at all. A batch at or above [`MERGE_MAX_BYTES`] is therefore passed
//! through untouched and never copied.
//!
//! **This is not [`crate::ops::morselize`], and the difference is the whole point.** That one
//! sizes morsels *for the executor*: it coalesces a run of small batches and it also **splits**
//! an over-target one, because a scheduler wants uniform work units. Splitting is exactly the
//! wrong move on the way out — it manufactures the conversions this module exists to remove —
//! so this merges and never splits. The slice-aware byte measure is shared rather than
//! restated ([`crate::ops::sliced_batch_bytes`]).
//!
//! This changes no row, no column, no name, no type and no order — only how the same rows are
//! divided into batches on the way out, which is not part of any result contract (`collect`
//! builds one table; `iter_batches` re-chunks to the caller's `batch_size`).

use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use rayon::prelude::*;

use crate::ops::sliced_batch_bytes;

/// Bytes one merged batch is built up to.
const TARGET_BYTES: usize = 1 << 20;

/// The largest batch worth copying into a merge. A batch at or above this is passed through.
///
/// A break-even, measured at both ends rather than derived. A conversion costs ~3.9 us, so on a
/// single thread's ~10 GB/s it buys about 39 KB of copying; the copies run across the pool while
/// the conversions cannot run at all, which moves the line higher. The two measurements that
/// bracket it, 6 M rows, best of 11 or 15: a 32 %-selectivity filter projecting one column,
/// whose morsels are ~36 KB, gains **1.24x**, while a pass-everything filter whose 16,384-row
/// `int64` morsels are exactly 128 KiB — and are near-slices of the input, so the copy buys
/// nothing — *loses* 6 %.
///
/// **128 KiB was tried and reverted.** It admits the ~72 KB two-column filter shape and is
/// worth 1.18x there (8.97/8.72 ms -> 7.71/7.25 ms), but the pass-everything filter measured
/// **4.84-5.02 ms at 64 KiB against 6.81-7.00 ms at 128 KiB** on the same tree, same commit,
/// three interleaved rounds — and it returns the *same 416 batches* either way, so the cost is
/// not the merging it performs. Whatever it is, it is larger than the win, and a threshold that
/// buys 1.18x on one shape and loses 1.4x on another is the wrong threshold.
const MERGE_MAX_BYTES: usize = 64 << 10;

/// What each column past the first adds to [`MERGE_MAX_BYTES`].
///
/// A conversion is paid per *array*, not per batch, so a wide batch buys more copying than a
/// narrow one: measured on the pyarrow C data interface, 408 batches of TPC-H `lineitem`'s 16
/// columns cost 17 ms to export and import, about 2.6 us per array against the 7.7 us a
/// one-column batch costs whole. A flat ceiling therefore never merged a wide result, and a
/// 16-column, 14 %-selectivity filter over sf1 `lineitem` returned all 408 of its ~350 KB
/// batches. Admitting ~24 KiB per extra column merges them, best of 9 over two rounds against
/// the unchanged build: **50-52 ms -> 34-39 ms**, and 8 `int64` columns at the same selectivity
/// **33-34 ms -> 20-21 ms**; one- and two-column results are untouched by construction.
///
/// [`TARGET_BYTES`] deliberately does *not* scale with the columns. Scaling it too merged the
/// same result into 9 batches rather than 118, which measured no better, and it turned a 2 %
/// filter's 21 parallel 1 MiB copies into 2 serial 8 MB ones, costing **16.5 -> 19 ms**: the
/// target sets how finely the copy fans out across the pool, not how many conversions remain.
const MERGE_BYTES_PER_EXTRA_COLUMN: usize = 24 << 10;

/// The largest batch of `columns` columns worth copying into a merge.
fn merge_max_bytes(columns: usize) -> usize {
    MERGE_MAX_BYTES + MERGE_BYTES_PER_EXTRA_COLUMN * columns.saturating_sub(1)
}

/// Total merge bytes below which the concatenation runs on this thread.
///
/// The pool is not free to enter. A `LIMIT 10` over a sharded scan returns one short batch per
/// shard, and handing that to `par_iter` measured **0.54 ms -> 1.10 ms** — a fan-out costing
/// twice the query it was optimizing. Small results are merged inline, where the whole copy is
/// tens of microseconds.
const PARALLEL_MIN_BYTES: usize = 4 << 20;

/// Groups concatenated per round, so the originals of a finished round drop before the next
/// starts. Without it a large result would hold both forms at once; with it the extra live
/// bytes are bounded by this many targets rather than by the whole result.
const GROUPS_PER_ROUND: usize = 64;

/// The batches' rows, redivided so that no batch is needlessly small.
///
/// Order is preserved exactly — a sorted result stays sorted. A group that fails to
/// concatenate (a schema the kernel refuses) is returned as it arrived rather than dropped.
pub fn coalesce_small_batches(batches: Vec<RecordBatch>) -> Vec<RecordBatch> {
    if batches.len() < 2 {
        return batches;
    }
    let (groups, merge_bytes) = group_by_target(batches);
    // Nothing to merge: every batch was already at or above the ceiling, so a concatenation
    // would copy without removing a single conversion.
    if groups.iter().all(|group| group.len() < 2) {
        return groups.into_iter().flatten().collect();
    }
    let mut out: Vec<RecordBatch> = Vec::with_capacity(groups.len());
    if merge_bytes < PARALLEL_MIN_BYTES {
        out.extend(groups.into_iter().flat_map(concat_group));
        return out;
    }
    let mut rounds = groups.into_iter().peekable();
    while rounds.peek().is_some() {
        let round: Vec<Vec<RecordBatch>> = rounds.by_ref().take(GROUPS_PER_ROUND).collect();
        let merged: Vec<Vec<RecordBatch>> = round.into_par_iter().map(concat_group).collect();
        out.extend(merged.into_iter().flatten());
    }
    out
}

/// One group as a single batch, or — when the kernel refuses it — exactly as it arrived.
///
/// A group that will not concatenate is not a reason to lose rows, and it cannot be reported
/// either: this runs after the executor has succeeded. Handing the group back unmerged costs
/// the conversions this module exists to save and returns the same relation, which is the only
/// safe direction for a purely-physical rewrite.
fn concat_group(group: Vec<RecordBatch>) -> Vec<RecordBatch> {
    if group.len() < 2 {
        return group;
    }
    let schema = group[0].schema();
    match concat_batches(&schema, group.iter()) {
        Ok(merged) => vec![merged],
        Err(_) => group,
    }
}

/// Split `batches` into runs of merge candidates, and say how many bytes a merge would copy.
///
/// A batch at or above [`MERGE_MAX_BYTES`] becomes its own group, so it is never copied; the
/// rest accumulate until a run reaches [`TARGET_BYTES`]. Size is the *logical* footprint — the
/// rows this batch actually spans — because that is what a concatenation would copy;
/// `get_array_memory_size` reports the whole underlying buffer for a slice, which would read a
/// 16 K-row window on a 6 M-row column as 48 MB.
fn group_by_target(batches: Vec<RecordBatch>) -> (Vec<Vec<RecordBatch>>, usize) {
    let mut groups: Vec<Vec<RecordBatch>> = Vec::new();
    let mut current: Vec<RecordBatch> = Vec::new();
    let mut current_bytes = 0usize;
    let mut merge_bytes = 0usize;
    for batch in batches {
        let bytes = sliced_batch_bytes(&batch);
        let columns = batch.num_columns().max(1);
        if bytes >= merge_max_bytes(columns) {
            if !current.is_empty() {
                groups.push(std::mem::take(&mut current));
                current_bytes = 0;
            }
            groups.push(vec![batch]);
            continue;
        }
        current_bytes += bytes;
        merge_bytes += bytes;
        current.push(batch);
        if current_bytes >= TARGET_BYTES {
            groups.push(std::mem::take(&mut current));
            current_bytes = 0;
        }
    }
    if !current.is_empty() {
        groups.push(current);
    }
    (groups, merge_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn batch(values: &[i64]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values.to_vec()))]).unwrap()
    }

    fn rows_of(batches: &[RecordBatch]) -> Vec<i64> {
        batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect()
    }

    fn wide_batch(columns: usize, rows: usize) -> RecordBatch {
        let fields: Vec<Field> = (0..columns)
            .map(|c| Field::new(format!("c{c}"), DataType::Int64, false))
            .collect();
        let arrays = (0..columns)
            .map(|c| {
                Arc::new(Int64Array::from_iter_values(
                    (0..rows as i64).map(|r| r * 7 + c as i64),
                )) as _
            })
            .collect();
        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
    }

    #[test]
    fn the_merge_ceiling_grows_with_the_columns_a_conversion_pays_for() {
        // 1,500 rows of 8 `int64` columns is 96 KB: over the one-column ceiling, under the
        // eight-column one, so it merges; the same bytes in one column do not.
        let wide: Vec<RecordBatch> = (0..4).map(|_| wide_batch(8, 1_500)).collect();
        assert!(sliced_batch_bytes(&wide[0]) > MERGE_MAX_BYTES);
        let merged = coalesce_small_batches(wide.clone());
        assert_eq!(
            merged.len(),
            1,
            "four 96 KB eight-column batches should merge"
        );
        assert_eq!(
            merged[0],
            arrow::compute::concat_batches(&wide[0].schema(), &wide).unwrap()
        );

        let narrow: Vec<RecordBatch> = (0..4).map(|_| wide_batch(1, 12_000)).collect();
        assert!(sliced_batch_bytes(&narrow[0]) > merge_max_bytes(1));
        assert_eq!(
            coalesce_small_batches(narrow).len(),
            4,
            "a 96 KB one-column batch is not copied"
        );
    }

    #[test]
    fn merging_preserves_every_row_in_order() {
        let input: Vec<RecordBatch> = (0..50).map(|i| batch(&[i, i + 100, i + 200])).collect();
        let expected = rows_of(&input);
        let out = coalesce_small_batches(input);
        assert_eq!(rows_of(&out), expected);
        assert!(out.len() < 50, "50 three-row batches should merge");
    }

    #[test]
    fn one_batch_and_none_are_returned_untouched() {
        assert!(coalesce_small_batches(vec![]).is_empty());
        let single = coalesce_small_batches(vec![batch(&[1, 2, 3])]);
        assert_eq!(rows_of(&single), vec![1, 2, 3]);
    }

    #[test]
    fn a_batch_already_at_the_target_is_not_copied() {
        // 200,000 i64 rows is 1.6 MB — far over the merge ceiling, so each stays its own batch.
        let wide: Vec<i64> = (0..200_000).collect();
        let input = vec![batch(&wide), batch(&wide)];
        let ptr_before: Vec<*const u8> = input
            .iter()
            .map(|b| b.column(0).to_data().buffers()[0].as_ptr())
            .collect();
        let out = coalesce_small_batches(input);
        assert_eq!(out.len(), 2);
        let ptr_after: Vec<*const u8> = out
            .iter()
            .map(|b| b.column(0).to_data().buffers()[0].as_ptr())
            .collect();
        assert_eq!(
            ptr_before, ptr_after,
            "an over-target batch must not be copied"
        );
    }

    #[test]
    fn a_slice_is_sized_by_its_own_rows_not_its_buffer() {
        // The whole point of `slice_upper_bound`: a 3-row window on a 200,000-row buffer must
        // read as 24 bytes, or a filtered result of slices would never merge.
        let big: Vec<i64> = (0..200_000).collect();
        let sliced = batch(&big).slice(0, 3);
        assert!(
            sliced_batch_bytes(&sliced) < 1024,
            "{}",
            sliced_batch_bytes(&sliced)
        );
    }

    #[test]
    fn a_short_string_column_is_sized_by_its_offsets() {
        // The regression this guards: sizing a string column as `rows * 64` reads 16,384
        // six-character values as 1 MB, so no result carrying a string would ever merge.
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
        let values: Vec<&str> = vec!["abcdef"; 4096];
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(values))]).unwrap();
        assert!(
            sliced_batch_bytes(&batch) < MERGE_MAX_BYTES,
            "4,096 six-character strings must read as a merge candidate, not {} bytes",
            sliced_batch_bytes(&batch)
        );
    }

    #[test]
    fn string_batches_merge_and_keep_their_values() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
        let make = |v: &str| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(StringArray::from(vec![v, v]))],
            )
            .unwrap()
        };
        let input: Vec<RecordBatch> = ["a", "bb", "ccc"].iter().map(|v| make(v)).collect();
        let out = coalesce_small_batches(input);
        assert_eq!(out.len(), 1);
        let col = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(
            col.iter().flatten().collect::<Vec<_>>(),
            vec!["a", "a", "bb", "bb", "ccc", "ccc"]
        );
    }
}
