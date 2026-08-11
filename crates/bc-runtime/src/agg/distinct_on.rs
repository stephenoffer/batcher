//! `DISTINCT ON` — keep one whole row per distinct key, mergeably.
//!
//! `DISTINCT` collapses rows that agree on *every* column, so its output carries no payload
//! and the group representative is immaterial. Deduplicating on a *subset* of the columns is
//! a different operator: the key decides which rows collapse, and the surviving row still
//! carries every other column. That is the shape of nearly every real deduplication — one
//! row per user, per order, per document id — and expressing it as a window
//! (`row_number() OVER (PARTITION BY key ORDER BY ts) = 1`) costs a full per-partition sort
//! and a materialized rank column over the whole relation, when the answer is a single
//! reduction: per key, the minimum row under the ordering.
//!
//! The reduction is a `min` under the ordering, which is why this file is small and why the
//! operator distributes for free. Its state is *a row of the input schema*, so `partial` and
//! `combine` are the **same function** ([`distinct_on`]): reducing a relation and reducing the
//! concatenation of its already-reduced parts give the same rows, because a minimum over a
//! union is the minimum of the per-part minima. Associativity and commutativity follow from
//! `min`'s, so a morsel fold, a per-core fan-out and a cluster shuffle all compose out of one
//! implementation — invariant #7, with no distinct-on-specific distributed semantics.
//!
//! With no ordering (`keep="any"`) the reduction keeps the first row seen for the key. That is
//! still an associative fold over the concatenation — the first row of `A ++ B` is `A`'s if
//! the key occurs there and `B`'s otherwise — so everything above holds. What it is *not* is a
//! function of the input relation alone: "first" depends on the order rows arrive, which
//! partitioning and a shuffle both change, so the surviving row's payload may differ between
//! runs. Callers who need a defined row pass an ordering. Every engine's `DISTINCT ON` /
//! `unique(keep="any")` behaves this way.

use arrow::array::{Array, ArrayRef, RecordBatch};
use arrow::compute::SortOptions;
use arrow::row::{RowConverter, SortField};
use rayon::prelude::*;

use super::assign_groups;
use crate::error::RuntimeError;

/// One ordering term of a `DISTINCT ON`: which column of the batch orders it, and how.
///
/// An index rather than an array, because an ordering term may be a computed expression and
/// this crate holds no expressions. The caller evaluates it and appends it as a column, which
/// also makes the ordering travel with its rows through every gather below — see
/// `bc_interp::ops::distinct_on_widen`.
pub type OrderKey = (usize, SortOptions);

/// The ordering as `(array, options)`, resolved against `batch`.
fn order_arrays(batch: &RecordBatch, order: &[OrderKey]) -> Vec<(ArrayRef, SortOptions)> {
    order
        .iter()
        .map(|&(i, o)| (batch.column(i).clone(), o))
        .collect()
}

/// Rows below which fanning the reduce across cores costs more than it saves.
const PARALLEL_MIN_ROWS: usize = 1 << 16;

/// Per-morsel reduction below which the pre-reduce pass is not worth its hash.
///
/// The pre-reduce exists to shrink a low-cardinality relation before the partitioned reduce,
/// and it is measured on the first morsel because that is exactly what it buys: a morsel whose
/// rows are already almost all distinct will not shrink however large the relation is. Below a
/// 2x local reduction the pass is a whole extra hash of every row for nothing, so
/// [`distinct_on_parts`] skips it and partitions the input directly.
const PRE_REDUCE_MIN_RATIO: f64 = 2.0;

/// Keep one row per distinct value of the `key_indices` columns: the row minimizing `order`,
/// or — when `order` is empty — the first row seen for that key.
///
/// The output has `batch`'s schema and one row per distinct key. This is both the `partial` and
/// the `combine` of the mergeable form: applying it to the concatenation of its own outputs
/// yields the same rows as applying it to the whole input (see the module docs).
pub fn distinct_on(
    batch: &RecordBatch,
    key_indices: &[usize],
    order: &[OrderKey],
) -> Result<RecordBatch, RuntimeError> {
    let n = batch.num_rows();
    if n == 0 {
        return Ok(batch.clone());
    }
    let keys: Vec<ArrayRef> = key_indices
        .iter()
        .map(|&i| batch.column(i).clone())
        .collect();
    let reps = surviving_rows(&keys, &order_arrays(batch, order), n)?;
    crate::shuffle::gather_rows(batch, &reps)
}

/// The index of the surviving row of each distinct key: the one minimizing `order`, or the
/// first seen when there is no ordering.
///
/// The whole reduction, over *arrays* rather than a batch, so the partitioned path can run it
/// on the key and ordering columns alone and gather the payload once at the end.
fn surviving_rows(
    keys: &[ArrayRef],
    order: &[(ArrayRef, SortOptions)],
    n: usize,
) -> Result<Vec<u32>, RuntimeError> {
    let (group_ids, num_groups, _) = assign_groups(keys, n)?;
    // Every group is reached by at least one row, so every slot is written before it is read.
    Ok(match order {
        [] => first_seen_reps(&group_ids, num_groups),
        _ => min_reps(&group_ids, num_groups, order, n)?,
    })
}

/// The first row index seen for each group — the representative when no ordering is given.
fn first_seen_reps(group_ids: &[u32], num_groups: usize) -> Vec<u32> {
    let mut reps = vec![u32::MAX; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        let slot = &mut reps[g as usize];
        if *slot == u32::MAX {
            *slot = i as u32;
        }
    }
    reps
}

/// The row index minimizing `order` within each group.
///
/// Ties keep the earlier row, so the reduction is a well-defined `min` over
/// `(order key, row index)` and repeating it over already-reduced parts is idempotent.
fn min_reps(
    group_ids: &[u32],
    num_groups: usize,
    order: &[(ArrayRef, SortOptions)],
    n: usize,
) -> Result<Vec<u32>, RuntimeError> {
    let mut reps = vec![u32::MAX; num_groups];
    // A single null-free primitive ordering column — the overwhelmingly common
    // `order_by="timestamp"` — maps to an order-preserving `u64`, turning the per-row
    // comparison into a register compare instead of an indirect load plus `memcmp` over
    // arrow's row encoding.
    if let Some(ranks) = u64_ranks(order) {
        for (i, &g) in group_ids.iter().enumerate() {
            let slot = &mut reps[g as usize];
            if *slot == u32::MAX || ranks[i] < ranks[*slot as usize] {
                *slot = i as u32;
            }
        }
        return Ok(reps);
    }
    // The general path: arrow's row format encodes any key type, and its byte order *is* the
    // requested sort order (direction and null placement included), so `min` over the encoded
    // rows is `min` under the ordering the caller asked for.
    let canon = crate::keys::canonicalize_float_order_keys(order);
    let order: &[(ArrayRef, SortOptions)] = canon.as_deref().unwrap_or(order);
    let fields: Vec<SortField> = order
        .iter()
        .map(|(a, o)| SortField::new_with_options(a.data_type().clone(), *o))
        .collect();
    let cols: Vec<ArrayRef> = order.iter().map(|(a, _)| a.clone()).collect();
    let rows = RowConverter::new(fields)?.convert_columns(&cols)?;
    debug_assert_eq!(rows.num_rows(), n);
    for (i, &g) in group_ids.iter().enumerate() {
        let slot = &mut reps[g as usize];
        if *slot == u32::MAX || rows.row(i) < rows.row(*slot as usize) {
            *slot = i as u32;
        }
    }
    Ok(reps)
}

/// An order-preserving `u64` per row for a single null-free primitive ordering column.
///
/// Declines (`None`) for anything else: several ordering terms, a nullable column, or a type
/// with no order-preserving 64-bit image. `descending` is folded in by complementing the rank,
/// which reverses the order without changing which values are equal.
fn u64_ranks(order: &[(ArrayRef, SortOptions)]) -> Option<Vec<u64>> {
    let [(arr, opts)] = order else { return None };
    if arr.null_count() != 0 {
        return None;
    }
    let canon = crate::keys::canonicalize_float_keys(std::slice::from_ref(arr));
    let arr = canon.as_ref().map_or(arr, |c| &c[0]);
    let mut ranks = crate::keys::u64_order_keys(arr, false)?;
    if opts.descending {
        ranks.iter_mut().for_each(|r| *r = !*r);
    }
    Some(ranks)
}

/// `DISTINCT ON` over a relation held as morsels, across cores.
///
/// Two stages, both parallel and each skippable when it would not pay:
///
/// 1. **Pre-reduce** each morsel independently. On a low-cardinality key this collapses the
///    relation to a few rows per morsel, so stage 2 has almost nothing to do. Skipped when
///    the first morsel shows less than [`PRE_REDUCE_MIN_RATIO`] local reduction, where it
///    would be a whole extra hash pass over every row for nothing.
/// 2. **Partition by key and reduce each partition.** Equal keys hash to the same partition,
///    so the partitions are key-disjoint and their reductions are independent — which is what
///    makes the whole operator scale with cores rather than serializing on one hash table.
///    Skipped when stage 1 already brought the relation under [`PARALLEL_MIN_ROWS`].
///
/// Every morsel must carry its ordering columns at the same indices — see [`OrderKey`] — which
/// is what lets the ordering survive the gathers below without being tracked separately.
///
/// Returns one batch per non-empty partition (or one batch overall on the small path), each with
/// the input schema. The union of the returned rows is the same relation [`distinct_on`] over the
/// concatenated input would produce; only row order differs, which `DISTINCT` does not define.
pub fn distinct_on_parts(
    parts: &[RecordBatch],
    key_indices: &[usize],
    order: &[OrderKey],
    partitions: usize,
) -> Result<Vec<RecordBatch>, RuntimeError> {
    let Some(first) = parts.first() else {
        return Ok(Vec::new());
    };
    if parts.len() == 1 {
        return Ok(vec![distinct_on(first, key_indices, order)?]);
    }
    let reduce = |b: &RecordBatch| distinct_on(b, key_indices, order);

    let pre: Vec<RecordBatch> = if pre_reduce_pays(first, key_indices)? {
        parts.par_iter().map(reduce).collect::<Result<_, _>>()?
    } else {
        parts.to_vec()
    };
    let pre_rows: usize = pre.iter().map(|b| b.num_rows()).sum();
    if pre_rows < PARALLEL_MIN_ROWS || partitions <= 1 {
        return Ok(vec![reduce(&concat_batches(&pre)?)?]);
    }

    reduce_partitioned(&pre, key_indices, order, partitions)
}

/// Reduce across cores by hash-partitioning on the key: equal keys share a partition, so the
/// partitions are key-disjoint and their reductions are independent.
///
/// The data movement is what this is careful about, because on a wide relation it *is* the
/// operator. Only the key and ordering columns are gathered into partitions; the reduction runs
/// on those, and the payload is gathered exactly once, for the surviving rows alone. Both
/// gathers are `interleave` across the morsels rather than a per-morsel scatter followed by a
/// concatenation — the same rows, in one pass instead of two, and without cutting a 600-morsel
/// relation into 600xP fragments that then have to be stitched back together (which measured
/// *slower* than the two-copy version it was meant to replace: 610 morsels x 16 partitions x 9
/// columns is 88,000 `concat` calls on ~600-row arrays).
fn reduce_partitioned(
    parts: &[RecordBatch],
    key_indices: &[usize],
    order: &[OrderKey],
    partitions: usize,
) -> Result<Vec<RecordBatch>, RuntimeError> {
    let schema = parts[0].schema();
    let ncols = parts[0].num_columns();
    // Which morsel and which row inside it — the coordinates `interleave` takes.
    let csr: Vec<(Vec<u32>, Vec<u32>)> = parts
        .par_iter()
        .map(|b| {
            let keys: Vec<ArrayRef> = key_indices.iter().map(|&i| b.column(i).clone()).collect();
            let part_of = crate::shuffle::bucket_of_rows(&keys, b.num_rows(), partitions)?;
            Ok::<_, RuntimeError>(crate::shuffle::bucket_csr(&part_of, partitions))
        })
        .collect::<Result<_, _>>()?;

    let column_of =
        |c: usize| -> Vec<&dyn Array> { parts.iter().map(|b| b.column(c).as_ref()).collect() };
    (0..partitions)
        .into_par_iter()
        .map(|p| {
            let mut coords: Vec<(usize, usize)> = Vec::new();
            for (m, (rows, offsets)) in csr.iter().enumerate() {
                let span = offsets[p] as usize..offsets[p + 1] as usize;
                coords.extend(rows[span].iter().map(|&r| (m, r as usize)));
            }
            if coords.is_empty() {
                return Ok(None);
            }
            let keys: Vec<ArrayRef> = key_indices
                .iter()
                .map(|&c| arrow::compute::interleave(&column_of(c), &coords))
                .collect::<Result<_, _>>()?;
            let ord: Vec<(ArrayRef, SortOptions)> = order
                .iter()
                .map(|&(c, o)| Ok((arrow::compute::interleave(&column_of(c), &coords)?, o)))
                .collect::<Result<_, arrow::error::ArrowError>>()?;
            let reps = surviving_rows(&keys, &ord, coords.len())?;
            // The payload, gathered once and only for the rows that survived.
            let kept: Vec<(usize, usize)> = reps.iter().map(|&r| coords[r as usize]).collect();
            let cols: Vec<ArrayRef> = (0..ncols)
                .map(|c| arrow::compute::interleave(&column_of(c), &kept))
                .collect::<Result<_, _>>()?;
            Ok(Some(RecordBatch::try_new(schema.clone(), cols)?))
        })
        .collect::<Result<Vec<_>, RuntimeError>>()
        .map(|v| v.into_iter().flatten().collect())
}

/// Concatenate batches of one schema, a column at a time across cores.
fn concat_batches(parts: &[RecordBatch]) -> Result<RecordBatch, RuntimeError> {
    if let [one] = parts {
        return Ok(one.clone());
    }
    let cols: Vec<ArrayRef> = (0..parts[0].num_columns())
        .into_par_iter()
        .map(|c| {
            let arrs: Vec<&dyn Array> = parts.iter().map(|b| b.column(c).as_ref()).collect();
            Ok::<_, RuntimeError>(arrow::compute::concat(&arrs)?)
        })
        .collect::<Result<_, _>>()?;
    Ok(RecordBatch::try_new(parts[0].schema(), cols)?)
}

/// Whether pre-reducing each morsel pays, probed on the first morsel's local reduction.
///
/// The probe is local *because the quantity it decides is local*: the pre-reduce buys exactly
/// the per-morsel collapse, and a morsel whose rows are nearly all distinct will not collapse
/// however large the relation is. Nothing here needs, or infers, the global key cardinality.
fn pre_reduce_pays(probe: &RecordBatch, key_indices: &[usize]) -> Result<bool, RuntimeError> {
    let rows = probe.num_rows();
    if rows == 0 {
        return Ok(false);
    }
    let keys: Vec<ArrayRef> = key_indices
        .iter()
        .map(|&i| probe.column(i).clone())
        .collect();
    let (_ids, groups, _) = assign_groups(&keys, rows)?;
    Ok((rows as f64) >= PRE_REDUCE_MIN_RATIO * groups.max(1) as f64)
}

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::*;

    fn batch(k: Vec<Option<i64>>, ts: Vec<Option<i64>>, v: Vec<Option<&str>>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("ts", DataType::Int64, true),
            Field::new("v", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(k)) as ArrayRef,
                Arc::new(Int64Array::from(ts)) as ArrayRef,
                Arc::new(StringArray::from(v)) as ArrayRef,
            ],
        )
        .unwrap()
    }

    /// Ascending, nulls last — SQL's `ORDER BY <col>`. Spelled out rather than taken from
    /// `SortOptions::default()`, whose `nulls_first` is `true`: the IR carries the placement
    /// explicitly for exactly this reason, and a test that leans on arrow's default is
    /// asserting a different ordering than the one the engine's callers ask for.
    fn asc(col: usize) -> Vec<OrderKey> {
        vec![(col, SQL_ASC)]
    }

    /// Ascending, nulls last — SQL's `ORDER BY <col>`.
    const SQL_ASC: SortOptions = SortOptions {
        descending: false,
        nulls_first: false,
    };

    /// `(key -> (ts, v))` of a result, for comparing rows irrespective of their order.
    fn rows(b: &RecordBatch) -> HashMap<Option<i64>, (Option<i64>, Option<String>)> {
        let k = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let ts = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let v = b.column(2).as_any().downcast_ref::<StringArray>().unwrap();
        (0..b.num_rows())
            .map(|i| {
                (
                    (!k.is_null(i)).then(|| k.value(i)),
                    (
                        (!ts.is_null(i)).then(|| ts.value(i)),
                        (!v.is_null(i)).then(|| v.value(i).to_string()),
                    ),
                )
            })
            .collect()
    }

    /// The minimum under the ordering wins, and it brings its OWN payload — the property a
    /// per-column `min` aggregate cannot give, because it would pair key 2's smallest `ts`
    /// with key 2's smallest `v` even though they sit in different rows.
    #[test]
    fn keeps_the_whole_minimum_row() {
        let b = batch(
            vec![Some(1), Some(1), Some(2), Some(2), Some(2)],
            vec![Some(10), Some(20), Some(30), Some(5), Some(15)],
            vec![Some("a"), Some("b"), Some("c"), Some("d"), Some("e")],
        );
        let ord = asc(1);
        let out = distinct_on(&b, &[0], &ord).unwrap();
        assert_eq!(
            rows(&out),
            HashMap::from([
                (Some(1), (Some(10), Some("a".into()))),
                // ts=5 is row 3, whose v is "d" — NOT min(v) = "c".
                (Some(2), (Some(5), Some("d".into()))),
            ])
        );
    }

    /// Descending order keeps the maximum row (the `keep="last"` lowering).
    #[test]
    fn descending_keeps_the_maximum_row() {
        let b = batch(
            vec![Some(1), Some(1), Some(2)],
            vec![Some(10), Some(20), Some(5)],
            vec![Some("a"), Some("b"), Some("c")],
        );
        let ord = vec![(
            1,
            SortOptions {
                descending: true,
                nulls_first: false,
            },
        )];
        let out = distinct_on(&b, &[0], &ord).unwrap();
        assert_eq!(rows(&out).get(&Some(1)).unwrap().0, Some(20));
    }

    /// A null key is a group of its own (it compares equal to itself), and a null ordering
    /// value sorts where `nulls_first` says — not "always last" and not "row dropped".
    #[test]
    fn nulls_group_and_order_as_specified() {
        let b = batch(
            vec![None, None, Some(1), Some(1)],
            vec![Some(9), Some(4), None, Some(7)],
            vec![Some("a"), Some("b"), Some("c"), Some("d")],
        );
        let last = distinct_on(&b, &[0], &asc(1)).unwrap();
        assert_eq!(rows(&last).len(), 2, "null key must be one group");
        assert_eq!(rows(&last).get(&None).unwrap().0, Some(4));
        // nulls_first=false puts NULL after every value, so ts=7 wins for key 1.
        assert_eq!(rows(&last).get(&Some(1)).unwrap().0, Some(7));

        let first = distinct_on(
            &b,
            &[0],
            &[(
                1,
                SortOptions {
                    descending: false,
                    nulls_first: true,
                },
            )],
        )
        .unwrap();
        assert_eq!(first.num_rows(), 2);
        assert_eq!(rows(&first).get(&Some(1)).unwrap().0, None);
    }

    /// `-0.0` and `0.0` are one key, and every NaN is one key — the engine's float identity
    /// contract, which `DISTINCT` on a float column must not split.
    #[test]
    fn float_keys_follow_the_engine_identity_contract() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("f", DataType::Float64, true),
            Field::new("t", DataType::Int64, true),
        ]));
        let f: ArrayRef = Arc::new(Float64Array::from(vec![
            0.0,
            -0.0,
            f64::NAN,
            -f64::NAN,
            1.5,
        ]));
        let t: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 2, 3, 4, 5]));
        let b = RecordBatch::try_new(schema, vec![f, t]).unwrap();
        let out = distinct_on(&b, &[0], &asc(1)).unwrap();
        assert_eq!(out.num_rows(), 3, "zeros are one key and NaNs are one key");
    }

    /// **The mergeability invariant.** Reducing every partition and then reducing the
    /// concatenation of those results must equal reducing the whole relation — the statement
    /// that makes `partial` and `combine` the same function, and the reason the distributed
    /// path needs no semantics of its own. Checked with the ordering (a real `min`, so the
    /// answer is unique) over partitionings that split keys across parts.
    #[test]
    fn combine_of_partials_equals_the_single_node_reduction() {
        let mut s: u64 = 17;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            (s >> 33) as i64
        };
        let n = 4_000;
        let k: Vec<Option<i64>> = (0..n).map(|_| Some(next() % 97)).collect();
        let ts: Vec<Option<i64>> = (0..n).map(|_| Some(next() % 1_000_003)).collect();
        let v: Vec<Option<String>> = (0..n).map(|i| Some(format!("v{i}"))).collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("ts", DataType::Int64, true),
            Field::new("v", DataType::Utf8, true),
        ]));
        let whole = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(k.clone())) as ArrayRef,
                Arc::new(Int64Array::from(ts.clone())) as ArrayRef,
                Arc::new(StringArray::from(v.clone())) as ArrayRef,
            ],
        )
        .unwrap();
        let want = rows(&distinct_on(&whole, &[0], &asc(1)).unwrap());

        for chunks in [1usize, 3, 7, 64, 501] {
            let parts: Vec<RecordBatch> = (0..n as usize)
                .step_by(chunks)
                .map(|start| {
                    let len = chunks.min(n as usize - start);
                    whole.slice(start, len)
                })
                .collect();
            // partial on each part, then combine the concatenation of the partials.
            let partials: Vec<RecordBatch> = parts
                .iter()
                .map(|p| distinct_on(p, &[0], &asc(1)).unwrap())
                .collect();
            let merged = concat_batches(&partials).unwrap();
            let got = rows(&distinct_on(&merged, &[0], &asc(1)).unwrap());
            assert_eq!(
                got, want,
                "combine(partials) != single-node at {chunks} rows/part"
            );

            // And the parallel driver over the same morsels, at every partition width.
            for p in [1usize, 2, 16] {
                let out = distinct_on_parts(&parts, &[0], &asc(1), p).unwrap();
                let joined = concat_batches(&out).unwrap();
                assert_eq!(
                    rows(&joined),
                    want,
                    "distinct_on_parts({chunks} rows/part, {p} partitions) != single-node"
                );
            }
        }
    }

    /// With no ordering the operator still returns exactly one row per distinct key, and each
    /// returned row is one that was actually in the input.
    #[test]
    fn unordered_returns_one_real_row_per_key() {
        let b = batch(
            vec![Some(1), Some(2), Some(1), Some(2), Some(3)],
            vec![Some(10), Some(20), Some(30), Some(40), Some(50)],
            vec![Some("a"), Some("b"), Some("c"), Some("d"), Some("e")],
        );
        let want: std::collections::HashSet<(i64, i64, String)> = (0..5)
            .map(|i| {
                let r = rows(&b.slice(i, 1));
                let (k, (ts, v)) = r.into_iter().next().unwrap();
                (k.unwrap(), ts.unwrap(), v.unwrap())
            })
            .collect();
        let out = distinct_on(&b, &[0], &[]).unwrap();
        assert_eq!(out.num_rows(), 3);
        for (k, (ts, v)) in rows(&out) {
            assert!(
                want.contains(&(k.unwrap(), ts.unwrap(), v.unwrap())),
                "returned a row that is not in the input"
            );
        }
    }

    /// A composite key, and a two-term ordering that needs the second term to break a tie on
    /// the first — the shape the single-`u64` fast path must decline.
    #[test]
    fn composite_key_and_multi_term_ordering() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
            Field::new("t1", DataType::Int64, false),
            Field::new("t2", DataType::Int64, false),
        ]));
        let b = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1i64, 1, 1, 2])) as ArrayRef,
                Arc::new(StringArray::from(vec!["x", "x", "y", "x"])) as ArrayRef,
                Arc::new(Int64Array::from(vec![5i64, 5, 1, 9])) as ArrayRef,
                Arc::new(Int64Array::from(vec![7i64, 3, 1, 1])) as ArrayRef,
            ],
        )
        .unwrap();
        let ord = vec![(2, SQL_ASC), (3, SQL_ASC)];
        let out = distinct_on(&b, &[0, 1], &ord).unwrap();
        assert_eq!(out.num_rows(), 3, "(a, b) has three distinct values");
        // (1, "x") ties on t1=5, so t2 decides: row 1 (t2=3) beats row 0 (t2=7).
        let t2 = out.column(3).as_any().downcast_ref::<Int64Array>().unwrap();
        let a = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let bb = out
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let idx = (0..out.num_rows())
            .find(|&i| a.value(i) == 1 && bb.value(i) == "x")
            .unwrap();
        assert_eq!(t2.value(idx), 3);
    }

    /// An empty relation reduces to an empty relation of the same schema, not an error.
    #[test]
    fn empty_input_is_empty_output() {
        let b = batch(vec![], vec![], vec![]);
        let out = distinct_on(&b, &[0], &[]).unwrap();
        assert_eq!(out.num_rows(), 0);
        assert_eq!(out.schema(), b.schema());
        assert!(distinct_on_parts(&[], &[0], &[], 4).unwrap().is_empty());
    }
}
