//! Late materialization for the Parquet driving scan: the plan's own `Filter`, and the runtime
//! join filters the executor places on the scan, as `bc_io::RowPredicate` stages the decode
//! evaluates before it materializes the rest of each row.

use std::sync::Arc;

use arrow::array::{Array, BooleanArray, DictionaryArray, RecordBatch};
use arrow::datatypes::{DataType, Int32Type, Schema};

use crate::normalize::normalize_batch;

/// The predicates of the stack of `Filter`s directly over `Scan(driving)`, innermost first, or
/// none.
///
/// Kyber can leave a scan under two stacked filters -- the part of a `WHERE` it pushed toward
/// the scan, and the rest above it. TPC-H q12 is the shape: a `l_receiptdate` range under
/// `l_shipmode IN (..) AND l_commitdate < l_receiptdate AND ...`, and with only the range as its
/// stage the late read kept a seventh of `lineitem` where the whole clause keeps 0.5%.
fn scan_filters(plan: &bc_ir::RelOp, driving: usize) -> Vec<&bc_expr::Expr> {
    let mut stack = Vec::new();
    let mut node = plan;
    while let bc_ir::RelOp::Filter { input, predicate } = node {
        stack.push(predicate);
        node = input;
    }
    if !stack.is_empty()
        && matches!(node, bc_ir::RelOp::Scan { source_id } if *source_id == driving)
    {
        stack.reverse();
        return stack;
    }
    plan.children()
        .into_iter()
        .map(|child| scan_filters(child, driving))
        .find(|found| !found.is_empty())
        .unwrap_or_default()
}

/// The plan's `Filter` over the driving scan, as the late-materialization stages of its reads.
///
/// The decode evaluates exactly what the `Filter` evaluates — the same expression, by the same
/// evaluator, over the same columns normalized the same way ([`filter_mask`]) — so the rows it
/// keeps are the rows that `Filter` keeps, and the `Filter` still runs over them.
///
/// A conjunction is split into stages, cheapest first, so a wide column is decoded only for the
/// rows the narrow ones kept. That evaluates a later conjunct over fewer rows, which is what the
/// engine's own `Filter` already does (`Expr::short_circuit_filter_mask`), and is sound on the
/// same condition: every conjunct is [infallible](bc_expr::Expr::is_infallible_predicate), so a
/// row skipped can never have been the one that raised. Its one remaining hazard is a
/// schema-driven error that only fires on a non-empty input, which a stage after one that kept
/// nothing would never see; so each conjunct is first evaluated over one all-null row of the
/// scan's schema, and a conjunction any of whose conjuncts fails that is pushed whole.
///
/// Empty when the plan has no such `Filter`. [`late_of`] turns the stages into the filter.
pub(super) fn plan_stages(
    plan: &bc_ir::RelOp,
    driving: usize,
    carrier: &[RecordBatch],
) -> Vec<bc_io::RowPredicate> {
    let filters = scan_filters(plan, driving);
    let (Some(&innermost), Some(first)) = (filters.first(), carrier.first()) else {
        return Vec::new();
    };
    let schema = first.schema();
    // Every filter of the stack, as one conjunction: a row the stack keeps passes them all. An
    // outer filter's conjuncts would otherwise be evaluated over rows an inner one removes, so
    // they are only added when every conjunct is infallible -- the condition splitting needs
    // anyway -- and the innermost filter alone is staged otherwise, as before.
    let conjuncts: Vec<&bc_expr::Expr> = filters.iter().flat_map(|p| p.and_conjuncts()).collect();
    let split = conjuncts.len() > 1
        && conjuncts
            .iter()
            .all(|c| c.is_infallible_predicate(&schema) && evaluates_on_a_null_row(c, &schema));
    if split {
        grouped_stages(conjuncts, &schema)
    } else {
        vec![stage(innermost, &schema)]
    }
}

/// Infallible `conjuncts` as few stages as keep their dictionary reads: one per string column
/// whose conjuncts read only it (that stage takes the column as a `Dictionary`), and one for all
/// the rest, each stage's conjuncts cheapest first.
///
/// One stage per conjunct made the decode a chain of row filters, each narrowing the selection
/// the next decodes under: TPC-H q12's six conjuncts (a date range, three column-to-column date
/// comparisons, `l_shipmode IN (..)`) measured 86 ns a row against 74 ns for no filter at all,
/// though they keep 0.5% of `lineitem`. A stage evaluates its own conjunction by short circuit
/// (`Expr::short_circuit_filter_mask`), so grouping loses none of the narrowing within a stage.
/// The rest stage runs first: it holds the conjuncts that need no dictionary to be cheap.
fn grouped_stages(conjuncts: Vec<&bc_expr::Expr>, schema: &Schema) -> Vec<bc_io::RowPredicate> {
    let mut rest: Vec<&bc_expr::Expr> = Vec::new();
    let mut by_column: Vec<(String, Vec<&bc_expr::Expr>)> = Vec::new();
    for c in conjuncts {
        match lone_string_column(c, schema) {
            Some(name) => match by_column.iter_mut().find(|(n, _)| *n == name) {
                Some((_, group)) => group.push(c),
                None => by_column.push((name, vec![c])),
            },
            None => rest.push(c),
        }
    }
    let mut groups: Vec<Vec<&bc_expr::Expr>> = Vec::new();
    if !rest.is_empty() {
        groups.push(rest);
    }
    groups.extend(by_column.into_iter().map(|(_, group)| group));
    groups
        .into_iter()
        .map(|mut group| {
            group.sort_by_key(|c| c.eval_cost());
            stage(&conjunction(&group), schema)
        })
        .collect()
}

/// The one string column `expr` reads, when that is all it reads.
fn lone_string_column(expr: &bc_expr::Expr, schema: &Schema) -> Option<String> {
    let mut names: Vec<&str> = Vec::new();
    expr.collect_columns(&mut names);
    names.sort_unstable();
    names.dedup();
    match names.as_slice() {
        [only]
            if schema
                .field_with_name(only)
                .is_ok_and(|f| matches!(f.data_type(), DataType::Utf8 | DataType::LargeUtf8)) =>
        {
            Some((*only).to_string())
        }
        _ => None,
    }
}

/// `parts` joined with `AND`, left to right.
fn conjunction(parts: &[&bc_expr::Expr]) -> bc_expr::Expr {
    let mut iter = parts.iter();
    let first = (*iter.next().expect("a group holds a conjunct")).clone();
    iter.fold(first, |acc, &next| bc_expr::Expr::Binary {
        op: bc_expr::BinaryOp::And,
        left: Box::new(acc),
        right: Box::new(next.clone()),
    })
}

/// The columns a read of the driving scan decodes: `columns`, or every column it carries.
pub(super) fn read_columns(carrier: &[RecordBatch], columns: Option<&[String]>) -> Vec<String> {
    match (columns, carrier.first()) {
        (Some(cols), _) => cols.to_vec(),
        (None, Some(first)) => first
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect(),
        (None, None) => Vec::new(),
    }
}

/// `stages` as the filter a read installs, or `None` when there are none, or when they would
/// decode every column in `read`: there is then nothing left to defer, and the filter could only
/// add its own evaluation to the read.
pub(super) fn late_of(
    stages: Vec<bc_io::RowPredicate>,
    read: &[String],
) -> Option<Arc<bc_io::LateFilter>> {
    let staged = |name: &String| stages.iter().any(|s| s.columns.contains(name));
    if !read.iter().any(|c| !staged(c)) {
        return None;
    }
    bc_io::LateFilter::new(stages).map(Arc::new)
}

/// The filter a read keyed by runtime join filters installs: one stage per key the scan reads,
/// ahead of the plan's own `plan` stages, or `None` when no key is among the columns in `read`.
///
/// The keys go first because they are what makes the read worth keying: a join that placed a
/// filter here did so because its build side is small against this probe side
/// (`runtime_filter::worth_filtering`), so the key stage is usually the one that removes most
/// rows. Every stage still runs over only the rows the earlier ones kept, and a [`key_mask`]
/// cannot raise, so the order changes the work and never the rows. Whether reading this way
/// pays at all is left to `bc_io::LateFilter`, which times it both ways.
pub(super) fn keyed_late(
    keys: &[bc_interp::ScanKeyFilter],
    plan: &[bc_io::RowPredicate],
    read: &[String],
) -> Option<Arc<bc_io::LateFilter>> {
    let mut stages: Vec<bc_io::RowPredicate> = keys
        .iter()
        .filter(|(column, _)| read.contains(column))
        .map(|(column, filter)| key_stage(column, Arc::clone(filter)))
        .collect();
    if stages.is_empty() {
        return None;
    }
    stages.extend(plan.iter().cloned());
    late_of(stages, read)
}

/// One late stage testing `column` against a runtime join filter's key digest.
fn key_stage(column: &str, filter: Arc<bc_runtime::join::KeyFilter>) -> bc_io::RowPredicate {
    let name = column.to_string();
    bc_io::RowPredicate {
        columns: vec![name.clone()],
        mask: Arc::new(move |batch: &RecordBatch| key_mask(&filter, &name, batch)),
        dictionary: false,
    }
}

/// The rows of `batch` whose `column` the digest does not refute: the mask
/// `runtime_filter::apply` computes over the same column once decoded, normalized the same way
/// ([`normalize_batch`], as [`filter_mask`] does). Every row is kept when the column cannot be
/// tested -- a missing column, or a key type the digest does not take -- so a key it cannot
/// judge is never dropped.
fn key_mask(
    filter: &bc_runtime::join::KeyFilter,
    column: &str,
    batch: &RecordBatch,
) -> BooleanArray {
    let keep_all = || {
        BooleanArray::new(
            arrow::buffer::BooleanBuffer::new_set(batch.num_rows()),
            None,
        )
    };
    let Ok(batch) = normalize_batch(batch) else {
        return keep_all();
    };
    batch
        .column_by_name(column)
        .and_then(|keys| filter.mask(keys))
        .unwrap_or_else(keep_all)
}

/// One late-materialization stage evaluating `expr`.
///
/// A stage over one string column that cannot raise accepts that column dictionary-encoded
/// (`dictionary`): its mask is then a pure function of each row's value, so it is computed once
/// per distinct value and looked up by key ([`dictionary_mask`]).
fn stage(expr: &bc_expr::Expr, schema: &Schema) -> bc_io::RowPredicate {
    let mut names: Vec<&str> = Vec::new();
    expr.collect_columns(&mut names);
    names.sort_unstable();
    names.dedup();
    let dictionary = matches!(names.as_slice(), [only] if schema
        .field_with_name(only)
        .is_ok_and(|f| matches!(f.data_type(), DataType::Utf8 | DataType::LargeUtf8)))
        && expr.is_infallible_predicate(schema);
    let owned = expr.clone();
    bc_io::RowPredicate {
        columns: names.into_iter().map(str::to_string).collect(),
        mask: Arc::new(move |batch: &RecordBatch| {
            dictionary_mask(&owned, batch).unwrap_or_else(|| filter_mask(&owned, batch))
        }),
        dictionary,
    }
}

/// [`filter_mask`] over a batch whose one column is a `Dictionary`, computed per distinct value.
///
/// The predicate is evaluated over the dictionary's values, plus one null standing for the rows
/// whose key is null, and each row takes its key's answer. That is the mask the per-row
/// evaluation computes because the stage's predicate reads only this column and cannot raise
/// (see [`stage`]), so its answer for a row is a function of the row's value alone. `None` for
/// any other batch, and for a dictionary larger than the batch, where evaluating every value
/// would cost more than evaluating every row.
fn dictionary_mask(predicate: &bc_expr::Expr, batch: &RecordBatch) -> Option<BooleanArray> {
    let [column] = batch.columns() else {
        return None;
    };
    let dict = column
        .as_any()
        .downcast_ref::<DictionaryArray<Int32Type>>()?;
    let values = dict.values();
    if values.len() > batch.num_rows() {
        return None;
    }
    let keys = dict.keys();
    let null_slot = values.len();
    let candidates = if keys.null_count() > 0 {
        let null = arrow::array::new_null_array(values.data_type(), 1);
        arrow::compute::concat(&[values.as_ref(), null.as_ref()]).ok()?
    } else {
        values.clone()
    };
    let field = batch
        .schema()
        .field(0)
        .clone()
        .with_data_type(values.data_type().clone())
        .with_nullable(true);
    let candidates =
        RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![candidates]).ok()?;
    let answers = filter_mask(predicate, &candidates);
    if answers.len() != candidates.num_rows() {
        return None;
    }
    let (codes, nulls) = (keys.values(), keys.nulls());
    let mask = arrow::buffer::BooleanBuffer::collect_bool(codes.len(), |i| {
        let slot = if nulls.is_some_and(|n| n.is_null(i)) {
            null_slot
        } else {
            usize::try_from(codes[i]).unwrap_or(null_slot)
        };
        answers.value(slot)
    });
    Some(BooleanArray::new(mask, None))
}

/// Whether `expr` evaluates to a boolean over one all-null row of `schema`'s columns.
fn evaluates_on_a_null_row(expr: &bc_expr::Expr, schema: &Schema) -> bool {
    let fields: Vec<_> = schema
        .fields()
        .iter()
        .map(|f| f.as_ref().clone().with_nullable(true))
        .collect();
    let columns = fields
        .iter()
        .map(|f| arrow::array::new_null_array(f.data_type(), 1))
        .collect();
    let Ok(row) = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns) else {
        return false;
    };
    expr.eval(&row)
        .is_ok_and(|a| a.as_any().downcast_ref::<BooleanArray>().is_some())
}

/// The keep mask the engine's `Filter` computes for `predicate` over `batch`: null-free, and
/// every row kept when it cannot be computed, so the `Filter` above raises whatever it raises.
fn filter_mask(predicate: &bc_expr::Expr, batch: &RecordBatch) -> BooleanArray {
    let keep_all = || BooleanArray::from(vec![true; batch.num_rows()]);
    // The columns as the engine sees them: decoded straight from Parquet they may still be
    // narrow or dictionary-encoded, which `normalize_batch` undoes on the way to the `Filter`.
    let Ok(batch) = normalize_batch(batch) else {
        return keep_all();
    };
    let mask = match predicate.short_circuit_filter_mask(&batch) {
        Ok(Some(mask)) => mask,
        _ => match predicate.eval(&batch) {
            Ok(array) => match array.as_any().downcast_ref::<BooleanArray>() {
                Some(mask) => mask.clone(),
                None => return keep_all(),
            },
            Err(_) => return keep_all(),
        },
    };
    if mask.null_count() > 0 {
        arrow::compute::prep_null_mask_filter(&mask)
    } else {
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::Field;

    fn expr(json: &str) -> bc_expr::Expr {
        serde_json::from_str(json).unwrap()
    }

    /// A one-column batch `m` holding `values` indexed by `keys` (a `None` key is a null row).
    fn dict_batch(values: &[Option<&str>], keys: &[Option<i32>]) -> RecordBatch {
        let dict = DictionaryArray::<Int32Type>::try_new(
            Int32Array::from(keys.to_vec()),
            Arc::new(StringArray::from(values.to_vec())),
        )
        .unwrap();
        let field = Field::new("m", dict.data_type().clone(), true);
        RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![Arc::new(dict)]).unwrap()
    }

    /// Per distinct value, the mask is the one the per-row evaluation of the decoded column
    /// computes — for membership, equality, a pattern and a null test, with null keys and a
    /// null dictionary value both present.
    #[test]
    fn a_dictionary_mask_equals_the_decoded_mask() {
        let values = [
            Some("MAIL"),
            Some("SHIP"),
            Some("AIR"),
            None,
            Some("AIR REG"),
        ];
        let keys: Vec<Option<i32>> = (0..64).map(|i| (i % 9 != 0).then_some(i * 7 % 5)).collect();
        let batch = dict_batch(&values, &keys);
        let decoded = normalize_batch(&batch).unwrap();
        assert_eq!(decoded.schema().field(0).data_type(), &DataType::Utf8);
        let col = r#"{"e":"col","name":"m"}"#;
        let air = r#"{"e":"lit","value":{"str":"AIR"}}"#;
        for predicate in [
            format!(r#"{{"e":"in_list","input":{col},"set":[{{"str":"MAIL"}},{{"str":"SHIP"}}]}}"#),
            format!(r#"{{"e":"binary","op":"eq","left":{col},"right":{air}}}"#),
            format!(r#"{{"e":"str","fn":"like","input":{col},"pattern":"AIR%"}}"#),
            format!(r#"{{"e":"is_null","input":{col}}}"#),
        ] {
            let e = expr(&predicate);
            let got = dictionary_mask(&e, &batch).expect("a dictionary batch takes this path");
            assert_eq!(got, filter_mask(&e, &decoded), "{predicate}");
        }
    }

    /// A dictionary with more values than the batch has rows, and a plain column, take the
    /// per-row path.
    #[test]
    fn a_large_dictionary_or_a_plain_column_is_declined() {
        let e = expr(r#"{"e":"is_null","input":{"e":"col","name":"m"}}"#);
        let values = [Some("a"), Some("b"), Some("c")];
        assert!(dictionary_mask(&e, &dict_batch(&values, &[Some(0), Some(1)])).is_none());
        let plain = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("m", DataType::Utf8, true)])),
            vec![Arc::new(StringArray::from(vec![Some("a"), None]))],
        )
        .unwrap();
        assert!(dictionary_mask(&e, &plain).is_none());
    }
}
