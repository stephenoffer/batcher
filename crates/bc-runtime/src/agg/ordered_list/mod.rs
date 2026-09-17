//! `array_agg(x ORDER BY k)`: a per-group list whose element order is a property of the rows.
//!
//! The unordered `list_agg` appends in arrival order, and arrival order is a scheduling
//! decision: a morselized scan, a spilled partition and a distributed shuffle each deliver a
//! group's rows differently, so the same query returned `[1, 2]` on one path and `[2, 1]` on
//! another. That is fine for a caller that only wants the multiset. It is not fine for one
//! that reads a position, and "which element is first" is the whole point of an ordered
//! aggregate.
//!
//! So the ordered form carries the order *with* the values instead of trusting the rows to
//! arrive in it:
//!
//! * **partial** keeps two aligned lists per group: the values (nulls kept, as `list_agg`
//!   keeps them) and each value's sort key, pre-encoded by [`encode_order_keys`] into arrow's
//!   row format with every key's direction and null placement baked in. Encoded keys compare
//!   by `memcmp`, so the state is one `LargeBinary` list however many keys the query named.
//! * **combine** concatenates both lists with the same group ids. The bucketing is stable, so
//!   the two lists stay aligned element for element; their order is still arrival order, and
//!   nothing downstream depends on it.
//! * **finalize** sorts each group by the encoded key and, on a tie, by the value itself
//!   (ascending, nulls last). Row encoding is injective, so after that tiebreak the only
//!   elements left unordered are byte-identical ones, and their order cannot be observed.
//!
//! The answer is therefore a function of the group's multiset of `(key, value)` pairs, which
//! is exactly what every partitioning of the input shares. That is the same resolution
//! `bc_interp::ops::reshape::sample_n_batches` uses for its hash ties, for the same reason.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, LargeBinaryArray, ListArray};
use arrow::buffer::OffsetBuffer;
use arrow::compute::{take, SortOptions};
use arrow::datatypes::DataType;
use arrow::row::{RowConverter, Rows, SortField};

use super::median::{finalize_list_agg, listagg_state, merge_median};
use crate::error::RuntimeError;

/// The name errors report this aggregate by: the one a user wrote.
const NAME: &str = "array_agg(order_by=...)";

/// Encode each row's `ORDER BY` keys into one `LargeBinary` value that sorts by `memcmp`.
///
/// `keys` pairs each evaluated key column with its direction and null placement. The caller
/// normalizes the columns the way every other sort in the engine does (an all-null key to a
/// constant, a float key's `-0.0`/NaN canonicalized), so an ordered aggregate ranks keys
/// exactly as `ORDER BY` does.
///
/// # Errors
///
/// A key type arrow's row format cannot encode, or columns of unequal length.
pub fn encode_order_keys(keys: &[(ArrayRef, SortOptions)]) -> Result<ArrayRef, RuntimeError> {
    let fields = keys
        .iter()
        .map(|(col, opts)| SortField::new_with_options(col.data_type().clone(), *opts))
        .collect();
    let converter = RowConverter::new(fields)?;
    let columns: Vec<ArrayRef> = keys.iter().map(|(col, _)| Arc::clone(col)).collect();
    let rows = converter.convert_columns(&columns)?;
    Ok(Arc::new(LargeBinaryArray::from_iter_values(rows.iter())))
}

/// Partial state: `[values, encoded keys]`, one aligned pair of lists per group.
pub(crate) fn ordered_list_state(
    values: &ArrayRef,
    keys: &ArrayRef,
    group_ids: &[u32],
    num_groups: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    if !matches!(keys.data_type(), DataType::LargeBinary) || keys.len() != values.len() {
        return Err(RuntimeError::UnsupportedAggregate {
            func: NAME.to_string(),
            dtype: format!(
                "an order key of type {} (expected encoded keys)",
                keys.data_type()
            ),
        });
    }
    Ok(vec![
        listagg_state(values, group_ids, num_groups)?,
        listagg_state(keys, group_ids, num_groups)?,
    ])
}

/// Combine: concatenate both lists under the same group ids, which keeps them aligned.
pub(crate) fn merge_ordered_list(
    state: &[ArrayRef],
    group_ids: &[u32],
    num_groups: usize,
) -> Result<Vec<ArrayRef>, RuntimeError> {
    Ok(vec![
        merge_median(&state[0], group_ids, num_groups)?,
        merge_median(&state[1], group_ids, num_groups)?,
    ])
}

/// Finalize: each group's values sorted by their encoded key, ties broken by the value.
///
/// A group that saw no rows is NULL, as for the unordered form.
pub(crate) fn finalize_ordered_list(state: &[ArrayRef]) -> Result<ArrayRef, RuntimeError> {
    let values = state[0].as_list::<i32>();
    let keys = state[1].as_list::<i32>();
    let key_bytes = keys.values().as_binary::<i64>();
    let value_child = values.values();
    let v_off = values.value_offsets();
    let k_off = keys.value_offsets();

    let mut perm: Vec<u32> = Vec::with_capacity(value_child.len());
    let mut offsets: Vec<i32> = Vec::with_capacity(values.len() + 1);
    offsets.push(0);
    // Built on the first tie only: most ordered aggregates order by a key that is unique
    // within the group, and encoding every value to break ties that never occur is wasted.
    let mut value_rows: Option<Rows> = None;
    for row in 0..values.len() {
        let (vs, ve) = (v_off[row] as usize, v_off[row + 1] as usize);
        let ks = k_off[row] as usize;
        let n = ve - vs;
        if k_off[row + 1] as usize - ks != n {
            return Err(RuntimeError::MalformedPartial {
                expected: n,
                got: k_off[row + 1] as usize - ks,
            });
        }
        let key = |i: usize| key_bytes.value(ks + i);
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_unstable_by(|&a, &b| key(a).cmp(key(b)));
        if idx.windows(2).any(|w| key(w[0]) == key(w[1])) {
            if value_rows.is_none() {
                value_rows = Some(encode_tiebreak(value_child)?);
            }
            let tie = value_rows.as_ref().expect("encoded just above");
            idx.sort_unstable_by(|&a, &b| {
                key(a)
                    .cmp(key(b))
                    .then_with(|| tie.row(vs + a).cmp(&tie.row(vs + b)))
            });
        }
        perm.extend(idx.into_iter().map(|i| (vs + i) as u32));
        offsets.push(perm.len() as i32);
    }
    let ordered = take(
        value_child.as_ref(),
        &arrow::array::UInt32Array::from(perm),
        None,
    )?;
    let field = match values.data_type() {
        DataType::List(f) => Arc::clone(f),
        other => {
            return Err(RuntimeError::UnsupportedAggregate {
                func: NAME.to_string(),
                dtype: other.to_string(),
            })
        }
    };
    let list = ListArray::try_new(
        field,
        OffsetBuffer::new(offsets.into()),
        ordered,
        values.nulls().cloned(),
    )?;
    finalize_list_agg(&(Arc::new(list) as ArrayRef))
}

/// The values encoded ascending with nulls last: the tiebreak between equal keys.
fn encode_tiebreak(values: &ArrayRef) -> Result<Rows, RuntimeError> {
    let field = SortField::new_with_options(
        values.data_type().clone(),
        SortOptions {
            descending: false,
            nulls_first: false,
        },
    );
    let converter =
        RowConverter::new(vec![field]).map_err(|_| RuntimeError::UnsupportedAggregate {
            func: NAME.to_string(),
            dtype: format!(
                "{} values with tied order keys (the tie cannot be broken by value; add an \
             order key that is unique within the group)",
                values.data_type()
            ),
        })?;
    Ok(converter.convert_columns(&[Arc::clone(values)])?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agg::{combine, finalize, partial, AggCall, AggFunc, Partial};
    use arrow::array::{Int64Array, StringArray};

    fn opts(descending: bool, nulls_first: bool) -> SortOptions {
        SortOptions {
            descending,
            nulls_first,
        }
    }

    /// Groups `g`, values `v`, keys `k` (a tied key, a null key, a null value).
    fn rows() -> (Vec<i64>, Vec<Option<&'static str>>, Vec<Option<i64>>) {
        (
            vec![1, 2, 1, 1, 2, 1, 2, 1, 1, 2],
            vec![
                Some("c"),
                Some("x"),
                Some("a"),
                None,
                Some("y"),
                Some("b"),
                Some("z"),
                Some("a"),
                Some("d"),
                None,
            ],
            vec![
                Some(3),
                Some(2),
                Some(1),
                Some(2),
                None,
                Some(1),
                Some(2),
                Some(2),
                None,
                Some(0),
            ],
        )
    }

    fn call_for(
        range: std::ops::Range<usize>,
        o: SortOptions,
    ) -> (Vec<ArrayRef>, Vec<AggCall>, usize) {
        let (g, v, k) = rows();
        let g: ArrayRef = Arc::new(Int64Array::from(g[range.clone()].to_vec()));
        let v: ArrayRef = Arc::new(StringArray::from(v[range.clone()].to_vec()));
        let k: ArrayRef = Arc::new(Int64Array::from(k[range.clone()].to_vec()));
        let encoded = encode_order_keys(&[(k, o)]).unwrap();
        let n = range.len();
        (
            vec![g],
            vec![AggCall::with_key(
                AggFunc::ListAggOrdered,
                Some(v),
                Some(encoded),
            )],
            n,
        )
    }

    fn partial_of(range: std::ops::Range<usize>, o: SortOptions) -> Partial {
        let (keys, calls, n) = call_for(range, o);
        partial(&keys, &calls, n).unwrap()
    }

    /// `group -> list` rendered, so two results compare exactly, element order included.
    fn render(p: &Partial) -> Vec<(i64, String)> {
        let out = finalize(&[AggFunc::ListAggOrdered], p).unwrap();
        let groups = p.group_columns[0].as_primitive::<arrow::datatypes::Int64Type>();
        let fmt = arrow::util::display::ArrayFormatter::try_new(
            out[0].as_ref(),
            &arrow::util::display::FormatOptions::default().with_null("null"),
        )
        .unwrap();
        let mut rendered: Vec<(i64, String)> = (0..groups.len())
            .map(|i| (groups.value(i), fmt.value(i).to_string()))
            .collect();
        rendered.sort();
        rendered
    }

    #[test]
    fn orders_by_key_then_value_with_null_placement() {
        let asc = render(&partial_of(0..10, opts(false, false)));
        // Group 1 keys: c=3, a=1, null-value=2, b=1, a=2, d=null. Ties on 1 -> a, b; on 2 ->
        // a, then the null value last; the null key sorts last.
        assert_eq!(
            asc,
            vec![
                (1, "[a, b, a, null, c, d]".to_string()),
                (2, "[null, x, z, y]".to_string()),
            ]
        );
        let desc_nulls_first = render(&partial_of(0..10, opts(true, true)));
        // Descending reverses the keys but not the value tiebreak, which stays ascending.
        assert_eq!(
            desc_nulls_first,
            vec![
                (1, "[d, c, a, null, a, b]".to_string()),
                (2, "[y, x, z, null]".to_string()),
            ]
        );
    }

    /// The mergeability invariant: any partitioning, combined in any order, equals the
    /// single-node list exactly -- element order included.
    #[test]
    fn combine_over_any_partitioning_and_order_equals_single_node() {
        for o in [opts(false, false), opts(true, false), opts(false, true)] {
            let want = render(&partial_of(0..10, o));
            let splits: [&[usize]; 4] = [&[0, 10], &[0, 3, 10], &[0, 1, 4, 7, 10], &[0, 5, 10]];
            for cuts in splits {
                let parts: Vec<Partial> =
                    cuts.windows(2).map(|w| partial_of(w[0]..w[1], o)).collect();
                let funcs = [AggFunc::ListAggOrdered];
                let forward = combine(&parts, &funcs).unwrap();
                assert_eq!(render(&forward), want, "cuts {cuts:?}");
                let mut reversed = parts;
                reversed.reverse();
                let backward = combine(&reversed, &funcs).unwrap();
                assert_eq!(render(&backward), want, "reversed cuts {cuts:?}");
                // Combining an already-combined partial with nothing new is the identity.
                let again = combine(&[backward], &funcs).unwrap();
                assert_eq!(render(&again), want, "re-combined cuts {cuts:?}");
            }
        }
    }

    #[test]
    fn a_global_aggregate_over_no_rows_is_null() {
        let v: ArrayRef = Arc::new(StringArray::from(Vec::<Option<&str>>::new()));
        let k: ArrayRef = Arc::new(Int64Array::from(Vec::<i64>::new()));
        let encoded = encode_order_keys(&[(k, opts(false, false))]).unwrap();
        let call = AggCall::with_key(AggFunc::ListAggOrdered, Some(v), Some(encoded));
        let p = partial(&[], &[call], 0).unwrap();
        let out = finalize(&[AggFunc::ListAggOrdered], &p).unwrap();
        assert_eq!(out[0].len(), 1);
        assert!(out[0].is_null(0));
    }

    #[test]
    fn an_unencoded_key_is_refused() {
        let v: ArrayRef = Arc::new(Int64Array::from(vec![1, 2]));
        let k: ArrayRef = Arc::new(Int64Array::from(vec![2, 1]));
        let call = AggCall::with_key(AggFunc::ListAggOrdered, Some(v), Some(k));
        assert!(partial(&[], &[call], 2).is_err());
    }
}
