//! ARG_MIN / ARG_MAX — the value at the row with the extreme (min/max) ordering key.
//!
//! Two-input aggregate: `arg_max(value, key)` returns `value` from the row whose
//! `key` is largest in the group (`arg_min` the smallest). Mergeable with a
//! **2-column state** — the winning `(key, value)` pair — so partials compose:
//! `combine` compares the per-partition winning keys and keeps the extreme pair.
//!
//! Ties on the key are broken by the **smallest value** (values are encoded into
//! arrow's order-preserving row format, so any value type is comparable), making
//! the result deterministic and partition-independent regardless of merge order.

use std::cmp::Ordering;

use arrow::array::{Array, ArrayRef, UInt32Array};
use arrow::compute::take;
use arrow::row::{RowConverter, SortField};

use crate::error::RuntimeError;

/// Pick, per group, the `(key, value)` pair at the extreme key — the shared core of
/// the partial step (over input columns) and the merge step (over partial winners).
/// Returns two columns: `[winning_key, winning_value]`. Rows where **either** the key
/// or the value is null are ignored (matching DuckDB `arg_max`/`arg_min`, which skip a
/// row with a null in either argument); an all-ignored group yields a null pair.
pub(crate) fn arg_extreme_pick(
    keys: &ArrayRef,
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
    is_max: bool,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    // Rank on the engine's float identity, not arrow's raw-bit row order: a *negative*
    // NaN key must rank greatest (not below -inf) so `arg_max(v, -NaN)` can win, and the
    // value tie-break must not split `-0.0` from `0.0`. Canonicalize the key/value copies
    // fed to the RowConverter, but `take` the *original* rows below (a `-0.0`/`-NaN` in,
    // the same value out) — matching `list.arg_max`, `MIN`/`MAX`, and `GROUP BY`.
    let keys_c = crate::keys::canonicalize_float_keys(std::slice::from_ref(keys));
    let keys_k = keys_c.as_ref().map_or(keys, |v| &v[0]);
    let values_c = crate::keys::canonicalize_float_keys(std::slice::from_ref(values));
    let values_k = values_c.as_ref().map_or(values, |v| &v[0]);
    let kconv = RowConverter::new(vec![SortField::new(keys_k.data_type().clone())])?;
    let krows = kconv.convert_columns(std::slice::from_ref(keys_k))?;
    let vconv = RowConverter::new(vec![SortField::new(values_k.data_type().clone())])?;
    let vrows = vconv.convert_columns(std::slice::from_ref(values_k))?;

    let mut best: Vec<Option<usize>> = vec![None; num_groups];
    // Both `keys` and `values` are `Arc<dyn Array>`, so the two validity checks below were
    // two virtual calls per row. Resolved once here; `None` means "no nulls", which the
    // closure then answers without touching memory at all.
    let (knulls, vnulls) = (keys.nulls(), values.nulls());
    let live =
        |i: usize| knulls.is_none_or(|n| n.is_valid(i)) && vnulls.is_none_or(|n| n.is_valid(i));
    for (i, &g) in group_ids.iter().enumerate() {
        // A null key can't be an extreme, and a null value can't be selected — DuckDB
        // ignores the whole row if either is null, so `arg_max(v, k)` returns the value
        // at the largest key *among rows with a non-null value* (not NULL because the
        // absolute-max-key row happened to have a null value).
        if !live(i) {
            continue;
        }
        let g = g as usize;
        let take_it = match best[g] {
            None => true,
            Some(b) => match krows.row(i).cmp(&krows.row(b)) {
                Ordering::Greater => is_max,
                Ordering::Less => !is_max,
                // Equal key → keep the smaller value (deterministic tiebreak).
                Ordering::Equal => vrows.row(i) < vrows.row(b),
            },
        };
        if take_it {
            best[g] = Some(i);
        }
    }
    let idx = UInt32Array::from(best.iter().map(|o| o.map(|i| i as u32)).collect::<Vec<_>>());
    let key_state = take(keys.as_ref(), &idx, None)?;
    let value_state = take(values.as_ref(), &idx, None)?;
    Ok(vec![key_state, value_state])
}

/// Partial state for arg_min/arg_max: `[winning_key, winning_value]` per group.
pub(crate) fn arg_extreme_state(
    values: &ArrayRef,
    keys: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
    is_max: bool,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    arg_extreme_pick(keys, values, group_ids, num_groups, is_max)
}

/// Merge arg_min/arg_max state across partitions: keep the extreme-key pair among
/// the partial winners routed to each group. `state[0]` is the key column,
/// `state[1]` the value column.
pub(crate) fn merge_arg_extreme(
    state: &[ArrayRef],
    group_ids: &[u32],
    num_groups: usize,
    is_max: bool,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    arg_extreme_pick(&state[0], &state[1], group_ids, num_groups, is_max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{AsArray, Int64Array};
    use arrow::datatypes::Int64Type;
    use std::sync::Arc;

    #[test]
    fn arg_extreme_picks_value_at_extreme_key() {
        // group 0: keys [1,3,2] vals [10,20,30] → max key 3 → val 20; min key 1 → 10.
        // group 1: keys [5,4]   vals [40,50]    → max key 5 → val 40; min key 4 → 50.
        let vals: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30, 40, 50]));
        let keys: ArrayRef = Arc::new(Int64Array::from(vec![1, 3, 2, 5, 4]));
        let gids = [0u32, 0, 0, 1, 1];

        let amax = arg_extreme_pick(&keys, &vals, &gids, 2, true).unwrap();
        let amax_v = amax[1].as_primitive::<Int64Type>();
        assert_eq!((amax_v.value(0), amax_v.value(1)), (20, 40));

        let amin = arg_extreme_pick(&keys, &vals, &gids, 2, false).unwrap();
        let amin_v = amin[1].as_primitive::<Int64Type>();
        assert_eq!((amin_v.value(0), amin_v.value(1)), (10, 50));
    }

    #[test]
    fn arg_extreme_skips_null_values() {
        // group 0: v=[10, NULL, 30], k=[1, 9, 5]. The absolute-max key is 9 (row 1), but
        // its value is NULL, so DuckDB ignores that row: arg_max = value at the next-
        // highest key among non-null values = key 5 → 30 (NOT null). arg_min = key 1 → 10.
        let vals: ArrayRef = Arc::new(Int64Array::from(vec![Some(10), None, Some(30)]));
        let keys: ArrayRef = Arc::new(Int64Array::from(vec![1, 9, 5]));
        let gids = [0u32, 0, 0];

        let amax = arg_extreme_pick(&keys, &vals, &gids, 1, true).unwrap();
        assert_eq!(amax[1].as_primitive::<Int64Type>().value(0), 30);
        let amin = arg_extreme_pick(&keys, &vals, &gids, 1, false).unwrap();
        assert_eq!(amin[1].as_primitive::<Int64Type>().value(0), 10);

        // A group whose only rows have null values yields a null pair.
        let v2: ArrayRef = Arc::new(Int64Array::from(vec![None, None]));
        let k2: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
        let r = arg_extreme_pick(&k2, &v2, &[0u32, 0], 1, true).unwrap();
        assert!(r[1].is_null(0), "all-null-value group must yield null");
    }

    #[test]
    fn arg_extreme_ranks_float_key_on_engine_identity() {
        use arrow::array::Float64Array;
        use arrow::datatypes::Float64Type;
        // A *negative* NaN key (what `0.0/0.0` yields on x86) must rank as the engine's
        // greatest float, so `arg_max(v, -NaN)` returns that row's value — like `MAX(k)`,
        // `GROUP BY k`, and `list.arg_max`. With the raw RowConverter it ranked below -inf
        // and never won. Also: `-0.0` and `0.0` are one key, so a value tie-break must not
        // split them.
        let neg_nan = f64::from_bits(0xfff8_0000_0000_0000);
        let keys: ArrayRef = Arc::new(Float64Array::from(vec![1.0, neg_nan, 2.0]));
        let vals: ArrayRef = Arc::new(Int64Array::from(vec![10, 20, 30]));
        let gids = [0u32, 0, 0];
        let amax = arg_extreme_pick(&keys, &vals, &gids, 1, true).unwrap();
        assert_eq!(
            amax[1].as_primitive::<Int64Type>().value(0),
            20,
            "arg_max must select the -NaN-keyed row (NaN ranks greatest)"
        );
        let amin = arg_extreme_pick(&keys, &vals, &gids, 1, false).unwrap();
        assert_eq!(
            amin[1].as_primitive::<Int64Type>().value(0),
            10,
            "arg_min must select the smallest finite key, never the NaN"
        );

        // A `-0.0`/`0.0` value tie-break on an equal key: both are one value, so the
        // smaller-value tiebreak treats them as equal and is stable, not a spurious split.
        let keys2: ArrayRef = Arc::new(Int64Array::from(vec![7, 7]));
        let vals2: ArrayRef = Arc::new(Float64Array::from(vec![-0.0, 0.0]));
        let r = arg_extreme_pick(&keys2, &vals2, &[0u32, 0], 1, true).unwrap();
        let picked = r[1].as_primitive::<Float64Type>().value(0);
        assert_eq!(picked, 0.0, "signed zeros are one value on the tie-break");
    }

    #[test]
    fn arg_extreme_null_heavy_merge_equals_single_node() {
        // The exact scenario `test_diff_agg_arg_extreme::test_arg_extreme_null_value_...`
        // exercises: two groups, both with nulls scattered through the value column, split
        // across two partitions. The two-phase (partial-per-partition, then merge) result
        // MUST equal the single-node pick over the whole input — the mergeability invariant
        // that a distributed run depends on. A null value confined to one partition must not
        // change which non-null row wins globally.
        // group a: v=[10,-,30,-,40,5] k=[1,9,5,8,3,2] → amax=30 (k5), amin=10 (k1)
        // group b: v=[-,1,-,9,2,-]    k=[4,1,7,6,2,5] → amax=9  (k6), amin=1  (k1)
        let full_v: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(10),
            None,
            Some(30),
            None,
            Some(40),
            Some(5), // group a
            None,
            Some(1),
            None,
            Some(9),
            Some(2),
            None, // group b
        ]));
        let full_k: ArrayRef = Arc::new(Int64Array::from(vec![1, 9, 5, 8, 3, 2, 4, 1, 7, 6, 2, 5]));
        let full_g = [0u32, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

        for is_max in [true, false] {
            let single = arg_extreme_state(&full_v, &full_k, &full_g, 2, is_max).unwrap();

            // Split each group's rows across two partitions (first 3 rows of each group in p1,
            // last 3 in p2), so a group's non-null winner and its null rows can land apart.
            let p1_v: ArrayRef = Arc::new(Int64Array::from(vec![
                Some(10),
                None,
                Some(30),
                None,
                Some(1),
                None,
            ]));
            let p1_k: ArrayRef = Arc::new(Int64Array::from(vec![1, 9, 5, 4, 1, 7]));
            let p2_v: ArrayRef = Arc::new(Int64Array::from(vec![
                None,
                Some(40),
                Some(5),
                Some(9),
                Some(2),
                None,
            ]));
            let p2_k: ArrayRef = Arc::new(Int64Array::from(vec![8, 3, 2, 6, 2, 5]));
            let split_g = [0u32, 0, 0, 1, 1, 1];

            let s1 = arg_extreme_state(&p1_v, &p1_k, &split_g, 2, is_max).unwrap();
            let s2 = arg_extreme_state(&p2_v, &p2_k, &split_g, 2, is_max).unwrap();
            // Shuffle brings both partials of each group together: rows [g0,g1,g0,g1].
            let kcat: ArrayRef = arrow::compute::concat(&[s1[0].as_ref(), s2[0].as_ref()]).unwrap();
            let vcat: ArrayRef = arrow::compute::concat(&[s1[1].as_ref(), s2[1].as_ref()]).unwrap();
            let merged = merge_arg_extreme(&[kcat, vcat], &[0u32, 1, 0, 1], 2, is_max).unwrap();

            let sv = single[1].as_primitive::<Int64Type>();
            let mv = merged[1].as_primitive::<Int64Type>();
            assert_eq!(
                (mv.value(0), mv.value(1)),
                (sv.value(0), sv.value(1)),
                "two-phase arg_{} must equal single-node with nulls split across partitions",
                if is_max { "max" } else { "min" }
            );
        }
    }

    #[test]
    fn arg_extreme_merges_across_partitions() {
        // Split the group across two partials, then merge: the global extreme wins.
        let v1: ArrayRef = Arc::new(Int64Array::from(vec![10, 20]));
        let k1: ArrayRef = Arc::new(Int64Array::from(vec![1, 3]));
        let v2: ArrayRef = Arc::new(Int64Array::from(vec![30, 40]));
        let k2: ArrayRef = Arc::new(Int64Array::from(vec![2, 9]));
        let p1 = arg_extreme_pick(&k1, &v1, &[0u32, 0], 1, true).unwrap(); // key3→20
        let p2 = arg_extreme_pick(&k2, &v2, &[0u32, 0], 1, true).unwrap(); // key9→40
                                                                           // Merge the two partial winners (each one row): global max key 9 → val 40.
        let kcat: ArrayRef = arrow::compute::concat(&[p1[0].as_ref(), p2[0].as_ref()]).unwrap();
        let vcat: ArrayRef = arrow::compute::concat(&[p1[1].as_ref(), p2[1].as_ref()]).unwrap();
        let merged = merge_arg_extreme(&[kcat, vcat], &[0u32, 0], 1, true).unwrap();
        assert_eq!(merged[1].as_primitive::<Int64Type>().value(0), 40);
    }
}
