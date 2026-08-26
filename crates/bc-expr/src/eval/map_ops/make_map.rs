//! Map construction for `Expr::MakeMap` — SQL `map(keys, values)`, Spark's
//! `map_from_arrays` — pairing two `List` columns into one Arrow `Map` column.
//!
//! The read side of maps was already complete: `Expr::Map` serves `map_keys`,
//! `map_values`, `map_entries`, `element_at` and `cardinality` over a `Map` column that
//! arrived from Arrow. What was missing was any way to *build* one, so every `map_*` call
//! over a constructed map failed on its **argument** rather than on the function, and one
//! absent constructor blocked the whole family.
//!
//! An Arrow `Map` is a `List<Struct<key, value>>` whose key field is **non-nullable**, which
//! is what forces the validation below rather than making it a style choice.
//!
//! Three inputs are refused rather than coerced, matching DuckDB, because each one has a
//! plausible wrong answer a silent implementation would return instead:
//!
//! | input | DuckDB, and here | the silent alternative |
//! |---|---|---|
//! | a null key | error | Arrow cannot represent it; the row would have to be dropped |
//! | a duplicate key | error | keeping first or last is a guess, and they differ |
//! | lists of unequal length | error | truncating to the shorter one drops data |
//!
//! A null *value* is fine (`map(['a'], [NULL])` is `{'a': NULL}`), and a null *list* on
//! either side yields a null map, which is what `map(NULL, NULL)` returns.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanBufferBuilder, StructArray};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::datatypes::{DataType, Field, Fields};
use arrow::row::{RowConverter, SortField};

use super::super::list_ops::as_var_list;
use crate::ExprError;

/// The field names Arrow's `Map` layout uses. `arrow-rs` and DuckDB both spell them this
/// way, and `map_entries` already returns structs with these names, so a constructed map
/// round-trips through the read side unchanged.
const ENTRIES: &str = "entries";
const KEY: &str = "key";
const VALUE: &str = "value";

/// Build a `Map` column from a column of key lists and a column of value lists.
pub(crate) fn eval_make_map(keys: &ArrayRef, values: &ArrayRef) -> Result<ArrayRef, ExprError> {
    use arrow::array::AsArray;

    let name = "map";
    let kl = as_var_list(keys, "map (keys)")?;
    let vl = as_var_list(values, "map (values)")?;
    let (ka, va) = (kl.as_list::<i32>(), vl.as_list::<i32>());
    let rows = ka.len();

    let (ko, vo) = (ka.value_offsets(), va.value_offsets());
    let (kv, vv) = (ka.values(), va.values());

    // One pass to validate and to build the output offsets. A row that is null on either
    // side contributes no entries, so its offset does not advance — the same convention the
    // list kernels use, and what makes the null map distinct from the empty one.
    let mut offsets: Vec<i32> = Vec::with_capacity(rows + 1);
    offsets.push(0);
    let mut nulls = BooleanBufferBuilder::new(rows);
    let mut keep: Vec<(usize, usize)> = Vec::with_capacity(rows);
    let mut total: i32 = 0;
    let mut seen: HashSet<Vec<u8>> = HashSet::new();

    for i in 0..rows {
        if ka.is_null(i) || va.is_null(i) {
            nulls.append(false);
            offsets.push(total);
            continue;
        }
        let (ks, ke) = (ko[i] as usize, ko[i + 1] as usize);
        let (vs, ve) = (vo[i] as usize, vo[i + 1] as usize);
        let (klen, vlen) = (ke - ks, ve - vs);
        if klen != vlen {
            return Err(ExprError::InvalidArgument {
                func: name.to_string(),
                reason: format!(
                    "the map key list does not align with the map value list: \
                     {klen} key(s) against {vlen} value(s)"
                ),
            });
        }
        // Arrow's map key field is non-nullable, so a null key has nowhere to go.
        for k in ks..ke {
            if !kv.is_valid(k) {
                return Err(ExprError::InvalidArgument {
                    func: name.to_string(),
                    reason: "map keys can not be NULL".to_string(),
                });
            }
        }
        // Uniqueness is checked on the key's *encoded bytes* rather than on a typed
        // comparison, so one implementation covers every key type the row encoder accepts
        // without restating equality per type — which is where a subtle disagreement with
        // the grouping path would otherwise creep in.
        seen.clear();
        if klen > 1 {
            let slice = kv.slice(ks, klen);
            let conv = RowConverter::new(vec![SortField::new(slice.data_type().clone())])?;
            let encoded = conv.convert_columns(&[slice])?;
            for r in &encoded {
                if !seen.insert(r.as_ref().to_vec()) {
                    return Err(ExprError::InvalidArgument {
                        func: name.to_string(),
                        reason: "map keys must be unique".to_string(),
                    });
                }
            }
        }
        nulls.append(true);
        keep.push((ks, klen));
        total += klen as i32;
        offsets.push(total);
    }

    // Gather the surviving entries. Rows are contiguous runs in the child arrays, so this is
    // a concat of slices rather than a per-element copy.
    let key_child = gather_runs(kv, &keep)?;
    let val_child = gather_runs(vv, &keep)?;

    let fields = Fields::from(vec![
        Arc::new(Field::new(KEY, key_child.data_type().clone(), false)),
        Arc::new(Field::new(VALUE, val_child.data_type().clone(), true)),
    ]);
    let entries = StructArray::new(fields.clone(), vec![key_child, val_child], None);
    let entry_field = Arc::new(Field::new(ENTRIES, DataType::Struct(fields), false));

    let map = arrow::array::MapArray::try_new(
        entry_field,
        OffsetBuffer::new(offsets.into()),
        entries,
        Some(NullBuffer::new(nulls.finish())),
        false,
    )?;
    Ok(Arc::new(map))
}

/// Concatenate the `(start, len)` runs of `child` in order.
fn gather_runs(child: &ArrayRef, runs: &[(usize, usize)]) -> Result<ArrayRef, ExprError> {
    if runs.is_empty() {
        return Ok(child.slice(0, 0));
    }
    let slices: Vec<ArrayRef> = runs.iter().map(|&(s, n)| child.slice(s, n)).collect();
    let refs: Vec<&dyn Array> = slices.iter().map(std::convert::AsRef::as_ref).collect();
    Ok(arrow::compute::concat(&refs)?)
}

#[cfg(test)]
mod tests {
    use arrow::array::{ArrayRef, Int64Builder, ListBuilder, StringBuilder};
    use std::sync::Arc;

    use super::*;

    /// `[[a, b], null, []]`-shaped string list, from an explicit row spec.
    fn str_lists(rows: &[Option<Vec<Option<&str>>>]) -> ArrayRef {
        let mut b = ListBuilder::new(StringBuilder::new());
        for row in rows {
            match row {
                None => b.append(false),
                Some(vals) => {
                    for v in vals {
                        match v {
                            Some(s) => b.values().append_value(s),
                            None => b.values().append_null(),
                        }
                    }
                    b.append(true);
                }
            }
        }
        Arc::new(b.finish())
    }

    fn int_lists(rows: &[Option<Vec<Option<i64>>>]) -> ArrayRef {
        let mut b = ListBuilder::new(Int64Builder::new());
        for row in rows {
            match row {
                None => b.append(false),
                Some(vals) => {
                    for v in vals {
                        match v {
                            Some(n) => b.values().append_value(*n),
                            None => b.values().append_null(),
                        }
                    }
                    b.append(true);
                }
            }
        }
        Arc::new(b.finish())
    }

    #[test]
    fn a_map_is_built_with_nulls_and_empties_where_duckdb_puts_them() {
        let k = str_lists(&[Some(vec![Some("a"), Some("b")]), None, Some(vec![])]);
        let v = int_lists(&[Some(vec![Some(1), Some(2)]), None, Some(vec![])]);
        let out = eval_make_map(&k, &v).expect("valid map");
        let m = out
            .as_any()
            .downcast_ref::<arrow::array::MapArray>()
            .expect("Map");
        assert_eq!(m.len(), 3);
        // Row 0 has two entries, row 1 is null, row 2 is the *empty* map — which is a
        // different thing from null and is the case an offsets bug collapses.
        assert!(m.is_valid(0) && m.is_null(1) && m.is_valid(2));
        assert_eq!(m.value_length(0), 2);
        assert_eq!(m.value_length(2), 0);
    }

    /// A null *value* is legal where a null *key* is not — the asymmetry Arrow's
    /// non-nullable key field forces, and the one a permissive implementation loses.
    #[test]
    fn a_null_value_is_kept_and_a_null_key_is_refused() {
        let ok = eval_make_map(
            &str_lists(&[Some(vec![Some("a")])]),
            &int_lists(&[Some(vec![None])]),
        );
        assert!(ok.is_ok(), "a null value is legal: {ok:?}");

        let err = eval_make_map(
            &str_lists(&[Some(vec![None])]),
            &int_lists(&[Some(vec![Some(1)])]),
        )
        .expect_err("a null key must be refused");
        assert!(format!("{err}").contains("can not be NULL"), "{err}");
    }

    #[test]
    fn duplicate_keys_are_refused_rather_than_resolved() {
        let err = eval_make_map(
            &str_lists(&[Some(vec![Some("a"), Some("a")])]),
            &int_lists(&[Some(vec![Some(1), Some(2)])]),
        )
        .expect_err("duplicate keys must be refused");
        assert!(format!("{err}").contains("must be unique"), "{err}");
    }

    /// Truncating to the shorter list is the silent wrong answer this guards.
    #[test]
    fn mismatched_list_lengths_are_refused_rather_than_truncated() {
        let err = eval_make_map(
            &str_lists(&[Some(vec![Some("a"), Some("b")])]),
            &int_lists(&[Some(vec![Some(1)])]),
        )
        .expect_err("a length mismatch must be refused");
        assert!(format!("{err}").contains("does not align"), "{err}");
    }

    /// Duplicate detection is per row, not across the column: the same key in two
    /// different rows is ordinary data, and a shared `seen` set would reject it.
    #[test]
    fn the_same_key_in_two_rows_is_not_a_duplicate() {
        let out = eval_make_map(
            &str_lists(&[Some(vec![Some("a")]), Some(vec![Some("a")])]),
            &int_lists(&[Some(vec![Some(1)]), Some(vec![Some(2)])]),
        )
        .expect("per-row uniqueness only");
        assert_eq!(out.len(), 2);
    }
}
