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
/// Returns two columns: `[winning_key, winning_value]`. Rows with a null key are
/// ignored; an all-null-key group yields a null pair.
pub(crate) fn arg_extreme_pick(
    keys: &ArrayRef,
    values: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
    is_max: bool,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    let kconv = RowConverter::new(vec![SortField::new(keys.data_type().clone())])?;
    let krows = kconv.convert_columns(std::slice::from_ref(keys))?;
    let vconv = RowConverter::new(vec![SortField::new(values.data_type().clone())])?;
    let vrows = vconv.convert_columns(std::slice::from_ref(values))?;

    let mut best: Vec<Option<usize>> = vec![None; num_groups];
    for (i, &g) in group_ids.iter().enumerate() {
        if !keys.is_valid(i) {
            continue; // a null key can't be an extreme
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
