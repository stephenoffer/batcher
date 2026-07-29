//! Spilling (grace) hash aggregation — bounded-memory `combine` + `finalize`.
//!
//! The in-memory aggregate ([`super::combine`] + [`super::finalize`]) holds every
//! group's state at once: peak memory is the full group cardinality. When that
//! exceeds the operator's memory envelope, this module computes the *same result*
//! with memory bounded to a single hash partition.
//!
//! The mechanism is the mergeable algebra applied locally. Per-morsel partials
//! (the output of [`super::partial`]) are routed to one of `P` partitions by a
//! hash of their group key and written to a [`SpillStore`]. Because a given group
//! key always hashes to the same partition, every partial row for a group lands
//! together — so `combine`+`finalize` run **one partition at a time** equals the
//! global aggregate (`combine` is associative+commutative; partitions are
//! disjoint by key). This is exactly the distributive-equivalence property the
//! distributed path relies on, reused to bound single-node memory.
//!
//! `SpillStore` and its two implementations live in [`store`]; this file is the merge
//! algorithm that drives them.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::{Field, Schema};
use arrow::row::{RowConverter, SortField};

use super::{combine, finalize, AggFunc, GroupAggResult, Partial};
use crate::error::RuntimeError;

pub mod store;

pub use store::{DiskSpillStore, MemSpillStore, SpillCodec, SpillStore};
// The skew metric is tested here beside the merge that produces the imbalance it measures.
#[cfg(test)]
use store::skew_of;

/// Spilling equivalent of `finalize(combine(chunk_partials))`.
///
/// Routes each chunk's partial state to a hash partition in `store`, then merges
/// and finalizes one partition at a time. The result equals
/// [`super::group_aggregate`] over the concatenated input (group order differs —
/// these are unordered relations). `funcs` must match the aggregates used to
/// build the partials; for an all-columns distinct grouping pass `&[]`.
///
/// `budget_bytes` bounds the memory of a single partition's `combine`. The initial
/// partition count is only an *average*-case fit (`total / budget`); under key skew one
/// partition can hold far more than its share and OOM the merge. When a partition's
/// spilled bytes exceed `budget_bytes`, it is **recursively re-partitioned** (a fresh
/// salted hash spreads the colliding keys) and merged sub-partition by sub-partition — so
/// peak memory stays bounded regardless of distribution. `budget_bytes == 0` disables the
/// guard (the historical single-level behavior, for callers that don't supply an envelope).
pub fn combine_finalize_spilling(
    chunk_partials: impl IntoIterator<Item = Partial>,
    funcs: &[AggFunc],
    store: &mut dyn SpillStore,
    budget_bytes: usize,
) -> Result<GroupAggResult, RuntimeError> {
    let partitions = store.num_partitions().max(1);

    // --- spill phase: route every partial's groups to a hash partition ---------
    let mut n_keys = 0usize;
    let mut any = false;
    for partial in chunk_partials {
        any = true;
        n_keys = partial.group_columns.len();
        let packed = pack_partial(&partial)?;
        for (pi, sub) in route(&packed, n_keys, partitions)? {
            store.append(pi, &sub)?;
        }
    }
    if !any {
        return Ok(GroupAggResult {
            group_columns: Vec::new(),
            agg_columns: Vec::new(),
        });
    }

    // --- merge phase: combine + finalize one partition at a time ---------------
    let mut group_parts: Vec<Vec<ArrayRef>> = Vec::with_capacity(partitions);
    let mut agg_parts: Vec<Vec<ArrayRef>> = Vec::with_capacity(partitions);
    for pi in 0..partitions {
        // Ask how big this partition is *before* reading it. The recursive split below
        // used to run inside `merge_partition`, which receives the partition already
        // materialized — so the one partition that provably does not fit was pulled into
        // memory in full before anything decided to split it. That is the exact case the
        // doc comment promises stays bounded, and it was the only case that did not.
        let known = store.partition_bytes(pi) as usize;
        if budget_bytes > 0 && n_keys > 0 && known > budget_bytes {
            if let Some((group_columns, aggs)) =
                split_partition_streaming(store, pi, known, n_keys, funcs, budget_bytes, 0)?
            {
                group_parts.push(group_columns);
                agg_parts.push(aggs);
            }
            continue;
        }
        let batches = store.read(pi)?;
        if batches.is_empty() {
            continue;
        }
        let (group_columns, aggs) =
            merge_partition(batches, n_keys, funcs, budget_bytes, store, 0)?;
        group_parts.push(group_columns);
        agg_parts.push(aggs);
    }

    // Concatenate the per-partition output chunks column by column.
    let group_columns = (0..n_keys)
        .map(|c| concat_cols(group_parts.iter().map(|g| &g[c])))
        .collect::<Result<_, _>>()?;
    let agg_columns = (0..funcs.len())
        .map(|c| concat_cols(agg_parts.iter().map(|a| &a[c])))
        .collect::<Result<_, _>>()?;
    Ok(GroupAggResult {
        group_columns,
        agg_columns,
    })
}

/// One merged partition's output: its group-key columns and its aggregate columns.
///
/// Named because the split helpers below return it wrapped in an `Option` and a `Result`,
/// which is past the point where the raw shape reads.
type MergedColumns = (Vec<ArrayRef>, Vec<ArrayRef>);

/// How deep the recursive re-partition may go before a partition is combined as it stands.
///
/// The backstop for the case no hash can fix: a single group key whose partial state exceeds
/// the budget re-hashes to one sub-partition however it is salted, so without a cap the
/// recursion would not terminate. That case is safe to combine directly — a lone key's
/// constant-size aggregate state is tiny; what is large is the *number* of keys, which
/// splitting does fix.
const MAX_MERGE_DEPTH: u32 = 4;

/// The re-partition salt for recursion `depth`. Nonzero and depth-varying, so keys that
/// collided at one level spread at the next instead of re-colliding identically.
fn split_salt(depth: u32) -> u64 {
    0x9E37_79B9_7F4A_7C15u64.wrapping_mul(depth as u64 + 1) | 1
}

/// Sub-partitions to split `bytes` into so each lands inside `budget`.
///
/// Capped at 256 per level rather than 4,096. The split is recursive, so the cap is not a
/// ceiling on the total: four levels at 256 ways is four billion sub-partitions, far past any
/// real ratio of state to envelope. What the cap *does* bound is the cost of a single level —
/// a disk-backed store creates a file per sub-partition, and a 4,096-way split of a partition
/// that is itself the product of an earlier 4,096-way split writes millions of files holding
/// a handful of rows each. The recursion reaches the same per-partition size through more,
/// cheaper levels.
fn sub_partition_count(bytes: usize, budget: usize) -> usize {
    bytes.div_ceil(budget.max(1)).clamp(2, 256)
}

/// Split a partition known to exceed `budget` **without ever holding it whole**.
///
/// Streams it straight out of the store and into a fresh child store, one batch at a time,
/// then merges the child's sub-partitions. Equal keys still co-locate within the child, so
/// the merged relation is identical to the direct combine — only the peak is different, and
/// it is now one batch plus one sub-partition rather than the whole skewed partition.
///
/// Returns `None` when the partition turned out to be empty after all (a store that does
/// not track per-partition bytes never reaches here, since its `partition_bytes` is `0`).
fn split_partition_streaming(
    store: &mut dyn SpillStore,
    partition: usize,
    bytes: usize,
    n_keys: usize,
    funcs: &[AggFunc],
    budget: usize,
    depth: u32,
) -> Result<Option<MergedColumns>, RuntimeError> {
    let sub_p = sub_partition_count(bytes, budget);
    let salt = split_salt(depth);
    // `child` is created before the drain so the shared `&self` borrow ends first; it owns
    // its own storage, so the `&mut self` drain that follows cannot conflict with it.
    let mut child = store.child(sub_p)?;
    store.drain(partition, &mut |b| {
        for (ci, sub) in route_salted(b, n_keys, sub_p, salt)? {
            child.append(ci, &sub)?;
        }
        Ok(())
    })?;
    merge_child_partitions(&mut child, sub_p, n_keys, funcs, budget, depth + 1)
}

/// Merge every sub-partition of `child`, concatenating their outputs column by column.
///
/// `None` when the child received no rows at all.
fn merge_child_partitions(
    child: &mut Box<dyn SpillStore>,
    sub_p: usize,
    n_keys: usize,
    funcs: &[AggFunc],
    budget: usize,
    depth: u32,
) -> Result<Option<MergedColumns>, RuntimeError> {
    let mut group_parts: Vec<Vec<ArrayRef>> = Vec::with_capacity(sub_p);
    let mut agg_parts: Vec<Vec<ArrayRef>> = Vec::with_capacity(sub_p);
    for pi in 0..sub_p {
        // Ask how big the sub-partition is *before* reading it, exactly as the top level
        // does. Without this the streaming split covered only depth 0: a sub-partition that
        // was itself still over budget — which is precisely what severe skew produces, since
        // one level of re-hashing does not separate a key from itself — was read whole and
        // only then handed to `merge_partition` to be split again. The bound the module
        // promises held for the first level and leaked at every level below it.
        let known = child.partition_bytes(pi) as usize;
        if budget > 0 && n_keys > 0 && depth < MAX_MERGE_DEPTH && known > budget {
            if let Some((g, a)) =
                split_partition_streaming(child.as_mut(), pi, known, n_keys, funcs, budget, depth)?
            {
                group_parts.push(g);
                agg_parts.push(a);
            }
            continue;
        }
        let sub = child.read(pi)?;
        if sub.is_empty() {
            continue;
        }
        let (g, a) = merge_partition(sub, n_keys, funcs, budget, child.as_ref(), depth)?;
        group_parts.push(g);
        agg_parts.push(a);
    }
    if group_parts.is_empty() && agg_parts.is_empty() {
        return Ok(None);
    }
    let group_columns = (0..n_keys)
        .map(|c| concat_cols(group_parts.iter().map(|g| &g[c])))
        .collect::<Result<_, _>>()?;
    let agg_columns = (0..funcs.len())
        .map(|c| concat_cols(agg_parts.iter().map(|a| &a[c])))
        .collect::<Result<_, _>>()?;
    Ok(Some((group_columns, agg_columns)))
}

/// Merge one spilled partition's packed partials into `(group_columns, agg_columns)`,
/// recursively re-partitioning if it is too large to `combine` within `budget`.
///
/// A partition that fits (`bytes <= budget`, or the guard is off, or no keys to split on,
/// or the recursion depth cap is hit) is combined + finalized directly. Otherwise it is
/// re-routed through a fresh child store with a **depth-varying salt** — the mergeable
/// algebra holds because equal keys still co-locate within each level, so merging the
/// sub-partitions and concatenating yields the identical relation, just with peak memory
/// bounded to one sub-partition. The depth cap backstops the pathological case (e.g. one
/// key dominating a partition — where its constant-size state is already tiny, so a direct
/// combine is safe).
fn merge_partition(
    batches: Vec<RecordBatch>,
    n_keys: usize,
    funcs: &[AggFunc],
    budget: usize,
    parent: &dyn SpillStore,
    depth: u32,
) -> Result<(Vec<ArrayRef>, Vec<ArrayRef>), RuntimeError> {
    let bytes: usize = batches.iter().map(|b| b.get_array_memory_size()).sum();
    if budget == 0 || bytes <= budget || n_keys == 0 || depth >= MAX_MERGE_DEPTH {
        let partials: Vec<Partial> = batches
            .iter()
            .map(|b| unpack_partial(b, n_keys, funcs))
            .collect::<Result<_, _>>()?;
        let merged = combine(&partials, funcs)?;
        let aggs = finalize(funcs, &merged)?;
        return Ok((merged.group_columns, aggs));
    }

    // Re-partition this over-large partition into ~`bytes/budget` sub-partitions under a
    // fresh child store, with a nonzero salt (varying by depth) so the colliding keys
    // spread rather than re-collide. The read-back batches are already in packed form.
    let sub_p = sub_partition_count(bytes, budget);
    let salt = split_salt(depth);
    let mut child = parent.child(sub_p)?;
    for b in &batches {
        for (pi, sub) in route_salted(b, n_keys, sub_p, salt)? {
            child.append(pi, &sub)?;
        }
    }
    drop(batches); // release the over-large partition before merging its sub-partitions

    let mut group_parts: Vec<Vec<ArrayRef>> = Vec::with_capacity(sub_p);
    let mut agg_parts: Vec<Vec<ArrayRef>> = Vec::with_capacity(sub_p);
    for pi in 0..sub_p {
        let sub = child.read(pi)?;
        if sub.is_empty() {
            continue;
        }
        let (g, a) = merge_partition(sub, n_keys, funcs, budget, child.as_ref(), depth + 1)?;
        group_parts.push(g);
        agg_parts.push(a);
    }
    let group_columns = (0..n_keys)
        .map(|c| concat_cols(group_parts.iter().map(|g| &g[c])))
        .collect::<Result<_, _>>()?;
    let agg_columns = (0..funcs.len())
        .map(|c| concat_cols(agg_parts.iter().map(|a| &a[c])))
        .collect::<Result<_, _>>()?;
    Ok((group_columns, agg_columns))
}

/// Flatten a [`Partial`] into one batch: group columns first (`g0..`), then each
/// aggregate's state columns (`s{agg}_{col}`). The inverse is [`unpack_partial`].
fn pack_partial(p: &Partial) -> Result<RecordBatch, RuntimeError> {
    let mut fields = Vec::new();
    let mut cols = Vec::new();
    for (i, c) in p.group_columns.iter().enumerate() {
        fields.push(Field::new(format!("g{i}"), c.data_type().clone(), true));
        cols.push(c.clone());
    }
    for (a, state) in p.states.iter().enumerate() {
        for (ci, c) in state.iter().enumerate() {
            fields.push(Field::new(
                format!("s{a}_{ci}"),
                c.data_type().clone(),
                true,
            ));
            cols.push(c.clone());
        }
    }
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), cols)?)
}

/// Rebuild a [`Partial`] from a [`pack_partial`] batch using the key arity and the
/// per-aggregate state arity (which `funcs` determines).
///
/// Validates the batch's column count against the packed format (`n_keys + Σ arity`)
/// before slicing, so a truncated or otherwise malformed spill batch surfaces a
/// typed [`RuntimeError::MalformedPartial`] instead of panicking on an out-of-range
/// slice.
fn unpack_partial(
    b: &RecordBatch,
    n_keys: usize,
    funcs: &[AggFunc],
) -> Result<Partial, RuntimeError> {
    let arities: Vec<usize> = funcs.iter().map(|f| f.state_arity()).collect();
    let expected = n_keys + arities.iter().sum::<usize>();
    if b.num_columns() != expected {
        return Err(RuntimeError::MalformedPartial {
            expected,
            got: b.num_columns(),
        });
    }
    let cols = b.columns();
    let group_columns = cols[..n_keys].to_vec();
    let mut states = Vec::with_capacity(funcs.len());
    let mut idx = n_keys;
    for arity in arities {
        states.push(cols[idx..idx + arity].to_vec());
        idx += arity;
    }
    Ok(Partial {
        group_columns,
        states,
    })
}

/// Partition a packed partial's rows by a stable hash of its group-key columns.
/// A global aggregate (no keys) or a single partition routes everything to 0.
fn route(
    packed: &RecordBatch,
    n_keys: usize,
    partitions: usize,
) -> Result<Vec<(usize, RecordBatch)>, RuntimeError> {
    route_salted(packed, n_keys, partitions, 0)
}

/// [`route`] with a `salt` mixed into the key hash. The initial spill uses `salt == 0`;
/// a recursive re-partition of an over-large partition ([`merge_partition`]) uses a
/// nonzero, depth-varying salt so the keys that collided into one partition spread across
/// the sub-partitions instead of re-colliding under the same hash. Equal keys still route
/// together within a level (the salt is fixed for that level), so the grace algebra holds.
fn route_salted(
    packed: &RecordBatch,
    n_keys: usize,
    partitions: usize,
    salt: u64,
) -> Result<Vec<(usize, RecordBatch)>, RuntimeError> {
    if n_keys == 0 || partitions <= 1 {
        return Ok(vec![(0, packed.clone())]);
    }
    let group_cols = &packed.columns()[..n_keys];
    // Canonicalize float keys BEFORE hashing so `-0.0`/`0.0` (and every NaN bit pattern) route
    // to the SAME partition — arrow's row encoding is not canonical for floats, so without this
    // two partials that stored the same SQL group as `-0.0` vs `0.0` (each `partial` keeps its
    // first-seen value, which can differ per morsel) would land in different partitions and be
    // finalized as two groups, disagreeing with the in-memory `combine` (which canonicalizes
    // when it re-groups). Routing decides only *co-location*; the output group value is still
    // `take`n from the original column below, so the representative the query returns is
    // unchanged. Identity (no realloc) when no key is Float64.
    let canon = crate::keys::canonicalize_float_keys(group_cols);
    let encode_cols: &[ArrayRef] = canon.as_deref().unwrap_or(group_cols);
    let fields: Vec<SortField> = encode_cols
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect();
    let converter = RowConverter::new(fields)?;
    let rows = converter.convert_columns(encode_cols)?;

    // Fixed seeds so the same key routes identically across every chunk.
    let state = ahash::RandomState::with_seeds(0x9E37, 0x79B9, 0x7F4A, 0x7C15);
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); partitions];
    let p = partitions as u64;
    for i in 0..packed.num_rows() {
        let h = state.hash_one(rows.row(i));
        // salt == 0 is the initial spill: `h % p`, byte-for-byte the historical routing.
        // A nonzero (recursive) salt re-mixes through a multiply-shift avalanche so keys
        // that collided into one partition genuinely spread across the sub-partitions
        // (a plain `h ^ salt` before `% p` could leave the bucket unchanged).
        let bucket = if salt == 0 {
            h % p
        } else {
            let mixed = (h ^ salt.rotate_left(31)).wrapping_mul(0xD6E8_FEB8_6659_FD93);
            mixed % p
        };
        buckets[bucket as usize].push(i as u32);
    }

    let mut out = Vec::with_capacity(buckets.len());
    for (pi, idxs) in buckets.into_iter().enumerate() {
        if idxs.is_empty() {
            continue;
        }
        let indices = UInt32Array::from(idxs);
        let cols = packed
            .columns()
            .iter()
            .map(|c| take(c.as_ref(), &indices, None).map_err(RuntimeError::from))
            .collect::<Result<Vec<_>, _>>()?;
        out.push((pi, RecordBatch::try_new(packed.schema(), cols)?));
    }
    Ok(out)
}

fn concat_cols<'a>(it: impl Iterator<Item = &'a ArrayRef>) -> Result<ArrayRef, RuntimeError> {
    let cols: Vec<&dyn Array> = it.map(|a| a.as_ref()).collect();
    Ok(arrow::compute::concat(&cols)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agg::{group_aggregate, partial, AggCall};
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use std::collections::BTreeMap;

    #[test]
    fn skew_of_measures_partition_imbalance() {
        assert_eq!(skew_of(&[]), 1.0); // nothing spilled
        assert_eq!(skew_of(&[100]), 1.0); // one partition — no imbalance
        assert_eq!(skew_of(&[100, 100, 100]), 1.0); // perfectly even
        assert_eq!(skew_of(&[400, 0, 0]), 1.0); // empty partitions ignored → one non-empty
                                                // One partition ~3x its peers: mean = 500/3 ≈ 166.7, max = 300 → ~1.8.
        let s = skew_of(&[300, 100, 100]);
        assert!(
            (1.5..2.0).contains(&s),
            "skew {s} should reflect the imbalance"
        );
    }

    fn strs(v: &[&str]) -> ArrayRef {
        Arc::new(StringArray::from(v.to_vec()))
    }
    fn i64s(v: &[i64]) -> ArrayRef {
        Arc::new(Int64Array::from(v.to_vec()))
    }

    /// A truncated spilled partial (fewer columns than `n_keys + Σ arity`) is
    /// rejected with a typed error rather than panicking on an out-of-range slice.
    #[test]
    fn unpack_rejects_malformed_partial() {
        use arrow::datatypes::{DataType, Field, Schema};
        // n_keys=1 + Sum(1) + CountStar(1) = 3 expected; give a 1-column batch.
        let funcs = [AggFunc::Sum, AggFunc::CountStar];
        let schema = Arc::new(Schema::new(vec![Field::new("g0", DataType::Int64, true)]));
        let bad = RecordBatch::try_new(schema, vec![i64s(&[1])]).unwrap();
        match unpack_partial(&bad, 1, &funcs) {
            Err(RuntimeError::MalformedPartial { expected, got }) => {
                assert_eq!((expected, got), (3, 1))
            }
            _ => panic!("expected Err(MalformedPartial)"),
        }
    }

    const FUNCS: [AggFunc; 6] = [
        AggFunc::Sum,
        AggFunc::CountStar,
        AggFunc::Mean,
        AggFunc::Min,
        AggFunc::Max,
        AggFunc::Median,
    ];

    fn calls(v: &ArrayRef) -> Vec<AggCall> {
        FUNCS
            .iter()
            .map(|&func| {
                AggCall::new(
                    func,
                    match func {
                        AggFunc::CountStar => None,
                        _ => Some(v.clone()),
                    },
                )
            })
            .collect()
    }

    /// Render an aggregation result to a key -> [agg cells] map, order-independent.
    fn to_map(keys: &ArrayRef, aggs: &[ArrayRef]) -> BTreeMap<String, Vec<String>> {
        let keys = keys.as_any().downcast_ref::<StringArray>().unwrap();
        let mut m = BTreeMap::new();
        for i in 0..keys.len() {
            let row: Vec<String> = aggs.iter().map(|a| cell(a, i)).collect();
            m.insert(keys.value(i).to_string(), row);
        }
        m
    }

    fn cell(a: &ArrayRef, i: usize) -> String {
        if let Some(x) = a.as_any().downcast_ref::<Int64Array>() {
            return if x.is_null(i) {
                "∅".into()
            } else {
                x.value(i).to_string()
            };
        }
        if let Some(x) = a.as_any().downcast_ref::<Float64Array>() {
            return if x.is_null(i) {
                "∅".into()
            } else {
                format!("{:.4}", x.value(i))
            };
        }
        "?".into()
    }

    /// Split `(keys, vals)` into `chunks` partials and run the spilling path.
    fn spilled(
        keys: &ArrayRef,
        vals: &ArrayRef,
        chunks: usize,
        store: &mut dyn SpillStore,
    ) -> GroupAggResult {
        let n = keys.len();
        let per = n.div_ceil(chunks);
        let mut partials = Vec::new();
        let mut off = 0;
        while off < n {
            let len = per.min(n - off);
            let k = keys.slice(off, len);
            let v = vals.slice(off, len);
            partials.push(partial(std::slice::from_ref(&k), &calls(&v), len).unwrap());
            off += len;
        }
        combine_finalize_spilling(partials, &FUNCS, store, 0).unwrap()
    }

    #[test]
    fn recursive_spill_under_budget_equals_oracle() {
        // Many distinct keys with a tiny per-partition budget: the merge phase must
        // recursively re-partition over-large partitions (a fresh salted hash) and still
        // reproduce the non-spilling oracle exactly — the skew-safety guarantee.
        let n = 500usize;
        let key_strs: Vec<String> = (0..n).map(|i| format!("k{}", i % 137)).collect();
        let keys: ArrayRef = Arc::new(StringArray::from(
            key_strs.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ));
        let vals = i64s(&(0..n as i64).collect::<Vec<_>>());
        let oracle =
            crate::agg::group_aggregate(std::slice::from_ref(&keys), &calls(&vals), n).unwrap();

        // Start with a single partition (P=1) and a 1-byte budget, so EVERY partition
        // overflows and the guard must recurse to make progress.
        let mut store = MemSpillStore::new(1);
        let per = n.div_ceil(7);
        let mut partials = Vec::new();
        let mut off = 0;
        while off < n {
            let len = per.min(n - off);
            let k = keys.slice(off, len);
            let v = vals.slice(off, len);
            partials.push(partial(std::slice::from_ref(&k), &calls(&v), len).unwrap());
            off += len;
        }
        let got = combine_finalize_spilling(partials, &FUNCS, &mut store, 1).unwrap();
        assert_eq!(
            to_map(&got.group_columns[0], &got.agg_columns),
            to_map(&oracle.group_columns[0], &oracle.agg_columns),
            "recursive-spill result must equal the non-spilling oracle",
        );
    }

    /// A partition known to exceed the budget must be split **without being read whole**.
    ///
    /// The split used to live inside `merge_partition`, which receives its partition already
    /// materialized — so the one partition that provably does not fit was the one pulled
    /// into memory in full before anything decided to split it. This counts the batches the
    /// store hands out: a streaming split asks for them one at a time and never calls the
    /// whole-partition `read`.
    #[test]
    fn an_over_budget_partition_is_split_without_being_materialized() {
        /// Wraps a `MemSpillStore`, recording which read path the merge took.
        struct Counting {
            inner: MemSpillStore,
            whole_reads: std::cell::Cell<usize>,
            drains: std::cell::Cell<usize>,
        }
        impl SpillStore for Counting {
            fn num_partitions(&self) -> usize {
                self.inner.num_partitions()
            }
            fn append(&mut self, p: usize, b: &RecordBatch) -> Result<(), RuntimeError> {
                self.inner.append(p, b)
            }
            fn read(&mut self, p: usize) -> Result<Vec<RecordBatch>, RuntimeError> {
                self.whole_reads.set(self.whole_reads.get() + 1);
                self.inner.read(p)
            }
            fn child(&self, n: usize) -> Result<Box<dyn SpillStore>, RuntimeError> {
                self.inner.child(n)
            }
            fn partition_bytes(&self, p: usize) -> u64 {
                self.inner.partition_bytes(p)
            }
            fn drain(
                &mut self,
                p: usize,
                sink: &mut dyn FnMut(&RecordBatch) -> Result<(), RuntimeError>,
            ) -> Result<(), RuntimeError> {
                self.drains.set(self.drains.get() + 1);
                // Deliberately *not* `self.read` — that is the path under test.
                for b in self.inner.read(p)? {
                    sink(&b)?;
                }
                Ok(())
            }
        }

        let n = 400usize;
        let key_strs: Vec<String> = (0..n).map(|i| format!("k{}", i % 53)).collect();
        let keys: ArrayRef = Arc::new(StringArray::from(
            key_strs.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ));
        let vals = i64s(&(0..n as i64).collect::<Vec<_>>());
        let oracle =
            crate::agg::group_aggregate(std::slice::from_ref(&keys), &calls(&vals), n).unwrap();

        let mut store = Counting {
            inner: MemSpillStore::new(1),
            whole_reads: std::cell::Cell::new(0),
            drains: std::cell::Cell::new(0),
        };
        let partials = vec![partial(std::slice::from_ref(&keys), &calls(&vals), n).unwrap()];
        let got = combine_finalize_spilling(partials, &FUNCS, &mut store, 1).unwrap();

        assert_eq!(
            to_map(&got.group_columns[0], &got.agg_columns),
            to_map(&oracle.group_columns[0], &oracle.agg_columns),
            "splitting without materializing must not change the result",
        );
        assert_eq!(
            store.drains.get(),
            1,
            "the over-budget partition should have been streamed out, not read whole",
        );
        assert_eq!(
            store.whole_reads.get(),
            0,
            "nothing should have asked the top-level store for a whole partition",
        );
    }

    /// The measure-before-read decision must happen at **every** level of the recursion, not
    /// only at the top.
    ///
    /// The streaming split was added where the top-level merge chooses a partition, and the
    /// recursive levels kept the older shape: read the sub-partition whole, then let
    /// `merge_partition` decide it was too big and split it. That is exactly the case severe
    /// skew produces — one level of re-hashing does not separate a key from itself — so the
    /// bound the module promises held for the first level and leaked at every level below it.
    ///
    /// The discriminator is where the batches come from. A store that is streamed reports a
    /// `drain`; one that is materialized reports a whole `read`. Before this fix the whole
    /// run produced exactly **one** drain, at depth 0, however deep the recursion went.
    #[test]
    fn every_recursion_level_splits_without_materializing() {
        /// A counting store whose children keep counting — the reason the earlier version of
        /// this test could not see the leak is that its `child` handed back a plain store.
        struct Counting {
            inner: MemSpillStore,
            whole_reads: std::rc::Rc<std::cell::Cell<usize>>,
            drains: std::rc::Rc<std::cell::Cell<usize>>,
        }
        impl SpillStore for Counting {
            fn num_partitions(&self) -> usize {
                self.inner.num_partitions()
            }
            fn append(&mut self, p: usize, b: &RecordBatch) -> Result<(), RuntimeError> {
                self.inner.append(p, b)
            }
            fn read(&mut self, p: usize) -> Result<Vec<RecordBatch>, RuntimeError> {
                let batches = self.inner.read(p)?;
                // Only a read that actually yields rows is a materialization; scanning past
                // the empty sub-partitions a wide split leaves behind is not.
                if !batches.is_empty() {
                    self.whole_reads.set(self.whole_reads.get() + 1);
                }
                Ok(batches)
            }
            fn child(&self, n: usize) -> Result<Box<dyn SpillStore>, RuntimeError> {
                Ok(Box::new(Counting {
                    inner: MemSpillStore::new(n),
                    whole_reads: self.whole_reads.clone(),
                    drains: self.drains.clone(),
                }))
            }
            fn partition_bytes(&self, p: usize) -> u64 {
                self.inner.partition_bytes(p)
            }
            fn drain(
                &mut self,
                p: usize,
                sink: &mut dyn FnMut(&RecordBatch) -> Result<(), RuntimeError>,
            ) -> Result<(), RuntimeError> {
                self.drains.set(self.drains.get() + 1);
                // Deliberately not `self.read` — that is the path under test.
                for b in self.inner.read(p)? {
                    sink(&b)?;
                }
                Ok(())
            }
        }

        let n = 240usize;
        let key_strs: Vec<String> = (0..n).map(|i| format!("k{}", i % 31)).collect();
        let keys: ArrayRef = Arc::new(StringArray::from(
            key_strs.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ));
        let vals = i64s(&(0..n as i64).collect::<Vec<_>>());
        let oracle =
            crate::agg::group_aggregate(std::slice::from_ref(&keys), &calls(&vals), n).unwrap();

        let whole_reads = std::rc::Rc::new(std::cell::Cell::new(0));
        let drains = std::rc::Rc::new(std::cell::Cell::new(0));
        let mut store = Counting {
            inner: MemSpillStore::new(1),
            whole_reads: whole_reads.clone(),
            drains: drains.clone(),
        };
        // A 1-byte budget makes every partition over-budget at every level, so the recursion
        // runs to its depth cap and each level has to make the same decision.
        let partials = vec![partial(std::slice::from_ref(&keys), &calls(&vals), n).unwrap()];
        let got = combine_finalize_spilling(partials, &FUNCS, &mut store, 1).unwrap();

        assert_eq!(
            to_map(&got.group_columns[0], &got.agg_columns),
            to_map(&oracle.group_columns[0], &oracle.agg_columns),
            "splitting at depth must not change the result",
        );
        assert!(
            drains.get() > 1,
            "only {} drain(s): the recursive levels still read their over-budget \
             sub-partitions whole before deciding to split them",
            drains.get()
        );
    }

    /// A store that cannot report per-partition sizes keeps the old path exactly.
    #[test]
    fn a_store_without_size_tracking_still_merges_correctly() {
        struct Opaque(MemSpillStore);
        impl SpillStore for Opaque {
            fn num_partitions(&self) -> usize {
                self.0.num_partitions()
            }
            fn append(&mut self, p: usize, b: &RecordBatch) -> Result<(), RuntimeError> {
                self.0.append(p, b)
            }
            fn read(&mut self, p: usize) -> Result<Vec<RecordBatch>, RuntimeError> {
                self.0.read(p)
            }
            fn child(&self, n: usize) -> Result<Box<dyn SpillStore>, RuntimeError> {
                self.0.child(n)
            }
            // `partition_bytes` and `drain` deliberately left at their defaults.
        }

        let keys = strs(&["a", "b", "a", "c", "b", "a", "d"]);
        let vals = i64s(&[1, 2, 3, 4, 5, 6, 7]);
        let oracle =
            group_aggregate(std::slice::from_ref(&keys), &calls(&vals), keys.len()).unwrap();
        let mut store = Opaque(MemSpillStore::new(2));
        let partials =
            vec![partial(std::slice::from_ref(&keys), &calls(&vals), keys.len()).unwrap()];
        let got = combine_finalize_spilling(partials, &FUNCS, &mut store, 1).unwrap();
        assert_eq!(
            to_map(&got.group_columns[0], &got.agg_columns),
            to_map(&oracle.group_columns[0], &oracle.agg_columns),
        );
    }

    /// The disk store reports what it wrote per partition, which is what makes the
    /// streaming split reachable at all.
    #[test]
    fn the_disk_store_reports_per_partition_bytes() {
        let dir = std::env::temp_dir();
        let mut store = DiskSpillStore::new(dir, 2).unwrap();
        let keys = strs(&["a", "b", "a"]);
        let vals = i64s(&[1, 2, 3]);
        let p = partial(std::slice::from_ref(&keys), &calls(&vals), 3).unwrap();
        let packed = pack_partial(&p).unwrap();
        store.append(0, &packed).unwrap();
        assert!(store.partition_bytes(0) > 0);
        assert_eq!(store.partition_bytes(1), 0);
        assert_eq!(store.partition_bytes(99), 0); // out of range is not a panic

        // And `drain` streams that partition back without a whole-partition read.
        let mut seen = 0usize;
        store
            .drain(0, &mut |b| {
                seen += b.num_rows();
                Ok(())
            })
            .unwrap();
        assert_eq!(seen, packed.num_rows());
    }

    #[test]
    fn mem_spill_equals_oracle() {
        let keys = strs(&["a", "b", "a", "c", "b", "a", "d", "c", "b", "a"]);
        let vals = i64s(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let oracle =
            group_aggregate(std::slice::from_ref(&keys), &calls(&vals), keys.len()).unwrap();
        let want = to_map(&oracle.group_columns[0], &oracle.agg_columns);

        // Many partitions + many chunks forces routing/merge to do real work.
        let mut store = MemSpillStore::new(4);
        let got = spilled(&keys, &vals, 5, &mut store);
        assert_eq!(want, to_map(&got.group_columns[0], &got.agg_columns));
    }

    #[test]
    fn disk_spill_equals_oracle() {
        let keys = strs(&["a", "b", "a", "c", "b", "a", "d", "c", "b", "a", "e", "a"]);
        let vals = i64s(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

        let oracle =
            group_aggregate(std::slice::from_ref(&keys), &calls(&vals), keys.len()).unwrap();
        let want = to_map(&oracle.group_columns[0], &oracle.agg_columns);

        // Spill is perf-only and result-invariant: every codec (and uncompressed)
        // must reproduce the in-memory oracle exactly. IPC self-describes its
        // compression, so the read path needs no codec.
        for codec in [
            SpillCodec::None,
            SpillCodec::Lz4,
            SpillCodec::Zstd,
            SpillCodec::Auto,
        ] {
            let dir = std::env::temp_dir()
                .join(format!("bc_spill_test_{}_{codec:?}", std::process::id()));
            let mut store = DiskSpillStore::with_codec(dir, 8, codec).unwrap();
            let got = spilled(&keys, &vals, 6, &mut store);
            assert_eq!(
                want,
                to_map(&got.group_columns[0], &got.agg_columns),
                "codec {codec:?} must match the oracle"
            );
        }
    }

    #[test]
    fn compressed_spill_roundtrips_every_codec() {
        // A wider batch (so compression actually engages) appended and read back is
        // logically identical under every codec — the layer that guarantees the
        // spill codec is purely a storage concern.
        let keys = strs(&["alpha", "beta", "alpha", "gamma", "beta", "alpha"]);
        let vals = i64s(&[100, 200, 300, 400, 500, 600]);
        let batch = RecordBatch::try_from_iter(vec![
            ("k", Arc::new(keys.clone()) as ArrayRef),
            ("v", Arc::new(vals.clone()) as ArrayRef),
        ])
        .unwrap();

        for codec in [
            SpillCodec::None,
            SpillCodec::Lz4,
            SpillCodec::Zstd,
            SpillCodec::Auto,
        ] {
            let dir =
                std::env::temp_dir().join(format!("bc_spill_rt_{}_{codec:?}", std::process::id()));
            let mut store = DiskSpillStore::with_codec(dir, 1, codec).unwrap();
            store.append(0, &batch).unwrap();
            store.append(0, &batch).unwrap();
            let back = store.read(0).unwrap();
            let total: usize = back.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total, 2 * batch.num_rows(), "codec {codec:?} row count");
            for b in &back {
                assert_eq!(b.schema(), batch.schema(), "codec {codec:?} schema");
            }
        }
    }

    #[test]
    fn float_key_signed_zero_merges_across_spill_partitions() {
        use arrow::array::Float64Array;
        // Two partials for the SAME SQL group, but one stored its float key as `-0.0` and the
        // other as `0.0` (each `partial` takes the first-seen value, which can differ per
        // morsel). `combine` merges them (it canonicalizes float keys), so the spilling path
        // MUST too — otherwise `-0.0` and `0.0` route to different hash partitions and the
        // group is finalized twice, disagreeing with the in-memory oracle.
        let k1: ArrayRef = Arc::new(Float64Array::from(vec![-0.0f64]));
        let k2: ArrayRef = Arc::new(Float64Array::from(vec![0.0f64]));
        let v1: ArrayRef = Arc::new(Float64Array::from(vec![10.0f64]));
        let v2: ArrayRef = Arc::new(Float64Array::from(vec![5.0f64]));
        let mk = |v: &ArrayRef| vec![AggCall::new(AggFunc::Sum, Some(v.clone()))];
        let p1 = partial(std::slice::from_ref(&k1), &mk(&v1), 1).unwrap();
        let p2 = partial(std::slice::from_ref(&k2), &mk(&v2), 1).unwrap();

        // Many partitions so `-0.0` and `0.0` (which hash differently under a non-canonical
        // float row encoding) land in different partitions if not canonicalized first.
        let mut store = MemSpillStore::new(16);
        let got = combine_finalize_spilling([p1, p2], &[AggFunc::Sum], &mut store, 0).unwrap();
        assert_eq!(
            got.group_columns[0].len(),
            1,
            "-0.0 and 0.0 must be ONE group after spilling, got {} groups",
            got.group_columns[0].len()
        );
        let sum = got.agg_columns[0]
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0);
        assert_eq!(sum, 15.0, "the merged group's sum must be 10 + 5");
    }

    #[test]
    fn single_partition_equals_oracle() {
        // P=1 degenerates to plain combine+finalize — a useful sanity floor.
        let keys = strs(&["x", "y", "x", "y", "z"]);
        let vals = i64s(&[5, 6, 7, 8, 9]);
        let oracle =
            group_aggregate(std::slice::from_ref(&keys), &calls(&vals), keys.len()).unwrap();
        let want = to_map(&oracle.group_columns[0], &oracle.agg_columns);

        let mut store = MemSpillStore::new(1);
        let got = spilled(&keys, &vals, 3, &mut store);
        assert_eq!(want, to_map(&got.group_columns[0], &got.agg_columns));
    }

    #[test]
    fn auto_codec_classifies_by_dominant_type() {
        use arrow::datatypes::{DataType, Field, Schema};
        // All fixed-width numeric → no compression (general codecs barely shrink it).
        let numeric = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Float64, false),
        ]);
        assert_eq!(SpillCodec::classify(&numeric), SpillCodec::None);
        // A string column → still None: compressing string state on fast local disk
        // costs more CPU than the I/O it saves (measured).
        let strings = Schema::new(vec![
            Field::new("k", DataType::Utf8, false),
            Field::new("v", DataType::Int64, false),
        ]);
        assert_eq!(SpillCodec::classify(&strings), SpillCodec::None);
        // A blob/large-binary column → ZSTD (payload dwarfs CPU, best ratio).
        let blobs = Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("blob", DataType::LargeBinary, false),
        ]);
        assert_eq!(SpillCodec::classify(&blobs), SpillCodec::Zstd);
    }

    #[test]
    fn concurrent_disk_stores_under_one_root_are_isolated() {
        // Two stores sharing one spill root must not collide on `part-*.arrow`, and
        // one store's drop must not delete the other's files. Regression for the
        // distributed-reducer clobber bug (many worker processes, one spill dir):
        // interleave appends across both, drop the first, then the second still reads
        // its own data back correctly.
        let keys_a = strs(&["a", "b", "a", "c", "b", "a"]);
        let vals_a = i64s(&[1, 2, 3, 4, 5, 6]);
        let keys_b = strs(&["x", "y", "x", "z", "y", "x"]);
        let vals_b = i64s(&[10, 20, 30, 40, 50, 60]);

        let want_a = to_map(
            &group_aggregate(std::slice::from_ref(&keys_a), &calls(&vals_a), keys_a.len())
                .unwrap()
                .group_columns[0],
            &group_aggregate(std::slice::from_ref(&keys_a), &calls(&vals_a), keys_a.len())
                .unwrap()
                .agg_columns,
        );
        let want_b = to_map(
            &group_aggregate(std::slice::from_ref(&keys_b), &calls(&vals_b), keys_b.len())
                .unwrap()
                .group_columns[0],
            &group_aggregate(std::slice::from_ref(&keys_b), &calls(&vals_b), keys_b.len())
                .unwrap()
                .agg_columns,
        );

        let root = std::env::temp_dir().join(format!("bc_spill_shared_{}", std::process::id()));
        let mut store_a = DiskSpillStore::new(root.clone(), 8).unwrap();
        let mut store_b = DiskSpillStore::new(root.clone(), 8).unwrap();
        // Distinct private subdirectories — proving the file namespaces don't alias.
        assert_ne!(store_a.scratch_dir(), store_b.scratch_dir());

        let got_a = spilled(&keys_a, &vals_a, 3, &mut store_a);
        drop(store_a); // wipes only store_a's private subdir
        let got_b = spilled(&keys_b, &vals_b, 3, &mut store_b);

        assert_eq!(want_a, to_map(&got_a.group_columns[0], &got_a.agg_columns));
        assert_eq!(want_b, to_map(&got_b.group_columns[0], &got_b.agg_columns));
    }

    #[test]
    fn a_spill_directory_is_not_readable_by_other_local_users() {
        use std::os::unix::fs::PermissionsExt;

        // Spilled data is the query's actual rows, written to a shared scratch path.
        // Created with the default mode it is world-readable, so a co-tenant on the node
        // could read a spilled join or aggregate straight off disk.
        let root = std::env::temp_dir();
        let store = DiskSpillStore::new(root, 2).unwrap();
        let mode = std::fs::metadata(store.scratch_dir())
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o700, "spill dir mode was {:o}", mode & 0o777);
    }
}
