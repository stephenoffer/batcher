//! `UNION ALL` as a pipeline operator: yield each branch's morsels in turn, hold none of them.
//!
//! The sequential oracle's `UNION ALL` is `concat(branch_0, branch_1, …)` with the branches
//! coerced to a common column type first, and it returns that concatenation as a `Vec` — so it
//! holds **every branch's entire output at once**, which the streaming executor then inherited
//! by deferring the operator to it. On the shape people actually write it (`bt.concat` over a
//! year of daily partitions, a backfill beside a live table) that is the whole relation resident
//! for an operator whose semantics need nothing resident at all.
//!
//! Chaining the branches' streams is identical, row for row and in order, on one condition: the
//! coercion must be settled *before* the first morsel is emitted, because a morsel already handed
//! to the consumer cannot be re-typed. [`coerce_union_branches`](crate::coerce_union_branches)
//! settles it from every batch of every branch, which is exactly what this path refuses to hold.
//!
//! So it is settled from **one peeked morsel per branch** instead, which is sound precisely when
//! a branch cannot change schema partway through its own stream. Every `RelOp` derives one output
//! schema from one input schema, so the only way a branch can is if a `Scan`'s source holds
//! batches of differing schemas — which [`sources_are_homogeneous`] checks, on schemas alone,
//! before committing. When it does not hold, this declines and the operator stays on the
//! materializing path that reads every batch. Declining is free; guessing would be a wrong type.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use bc_ir::RelOp;

use super::{build_with, Ctx, Morsels};
use crate::union_coerce::{cast_to_union_schema, union_target_schema};
use crate::InterpError;

/// Compose a lazy `UNION ALL` over `inputs`, or `None` when it cannot be proved identical to
/// the materialized concatenation and the caller should keep its existing path.
///
/// Peeks one morsel from each branch — bounded by the branch count, not the relation — to settle
/// the common column types, then yields branch 0's morsels, then branch 1's, and so on. A branch
/// that yields nothing contributes nothing, which is what the oracle's `Vec::extend` does with an
/// empty branch too.
pub(super) fn build_union_all<'a>(
    inputs: &'a [RelOp],
    ctx: Ctx<'a>,
    id: Option<u32>,
) -> Result<Option<Morsels<'a>>, InterpError> {
    if !sources_are_homogeneous(inputs, ctx) {
        return Ok(None);
    }

    // One peeked morsel per branch, kept to be re-emitted: nothing is consumed twice or lost.
    let mut streams: Vec<Morsels<'a>> = Vec::with_capacity(inputs.len());
    let mut heads: Vec<Option<RecordBatch>> = Vec::with_capacity(inputs.len());
    for input in inputs {
        let mut stream = build_with(input, ctx)?;
        let head = stream.next().transpose()?;
        heads.push(head);
        streams.push(stream);
    }

    // The target is folded over the branches that produced a morsel, in branch order — the same
    // order, and so the same fold, as the oracle's `coerce_union_branches` over the concatenated
    // `Vec`. A branch with no morsel at all contributed no schema there either.
    let present: Vec<RecordBatch> = heads.iter().flatten().cloned().collect();
    let target: Option<SchemaRef> = union_target_schema(&present)?;

    let stream = streams
        .into_iter()
        .zip(heads)
        .flat_map(move |(rest, head)| head.map(Ok).into_iter().chain(rest));
    let target_for_map = target.clone();
    Ok(Some(Box::new(stream.map(move |b| {
        let b = b?;
        let rows_in = b.num_rows() as u64;
        let t = std::time::Instant::now();
        // `None` is the single-type union, where the oracle returns the branches untouched —
        // including their own per-branch nullability flags. Passing them through unchanged is
        // therefore not merely cheaper, it is what identity requires.
        let out = match target_for_map.as_ref() {
            None => b,
            Some(schema) => cast_to_union_schema(&b, schema)?,
        };
        ctx.morsel(id, rows_in, &out, t);
        Ok(out)
    }))))
}

/// Whether every source any of these branches scans holds batches of a single schema.
///
/// This is the whole soundness condition for settling the union's column types from one peeked
/// morsel per branch (see the module comment). It reads schemas only — no column, no row — so it
/// costs a walk of the plan plus a pointer comparison per batch, and it is checked before the
/// first morsel is emitted, when declining is still free.
fn sources_are_homogeneous(inputs: &[RelOp], ctx: Ctx<'_>) -> bool {
    let mut ids: BTreeSet<usize> = BTreeSet::new();
    let mut stack: Vec<&RelOp> = inputs.iter().collect();
    while let Some(node) = stack.pop() {
        if let RelOp::Scan { source_id } = node {
            ids.insert(*source_id);
        }
        stack.extend(node.children());
    }
    ids.iter().all(|id| match ctx.sources.get(*id) {
        None => false, // an unknown source: let the ordinary path raise its typed error
        Some(batches) => {
            let mut schemas = batches.iter().map(|b| b.schema());
            match schemas.next() {
                None => true, // no batches, so no disagreement
                // `Arc::ptr_eq` first: batches of one relation overwhelmingly share one `Arc`,
                // so the common case never compares a field.
                Some(first) => schemas.all(|s| Arc::ptr_eq(&first, &s) || first == s),
            }
        }
    })
}

#[cfg(test)]
mod union_all_tests {
    use super::*;
    use arrow::array::{ArrayRef, Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};

    fn batch(ty: DataType, col: ArrayRef) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("x", ty, true)]));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    fn ints(vals: Vec<i64>) -> RecordBatch {
        batch(DataType::Int64, Arc::new(Int64Array::from(vals)))
    }

    fn floats(vals: Vec<f64>) -> RecordBatch {
        batch(DataType::Float64, Arc::new(Float64Array::from(vals)))
    }

    fn scan(id: usize) -> RelOp {
        RelOp::Scan { source_id: id }
    }

    /// The streamed union yields exactly the oracle's rows, in the oracle's order.
    #[test]
    fn streamed_union_all_matches_the_oracle() {
        let sources = vec![vec![ints(vec![1, 2]), ints(vec![3])], vec![ints(vec![4])]];
        let plan = RelOp::Union {
            inputs: vec![scan(0), scan(1)],
            distinct: false,
        };
        let streamed = crate::execute_streaming(&plan, &sources, 0).unwrap();
        let oracle = crate::execute(&plan, &sources).unwrap();
        let flat = |bs: &[RecordBatch]| -> Vec<i64> {
            bs.iter()
                .flat_map(|b| {
                    b.column(0)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .values()
                        .to_vec()
                })
                .collect()
        };
        assert_eq!(flat(&streamed), flat(&oracle));
        assert_eq!(flat(&streamed), vec![1, 2, 3, 4]);
    }

    /// A promotable mismatch across branches is coerced on the streamed path exactly as the
    /// oracle coerces it — settled from the peeked heads, before anything is emitted.
    #[test]
    fn a_promotable_mismatch_still_widens_when_streamed() {
        let sources = vec![vec![ints(vec![1, 2])], vec![floats(vec![3.5])]];
        let plan = RelOp::Union {
            inputs: vec![scan(0), scan(1)],
            distinct: false,
        };
        let streamed = crate::execute_streaming(&plan, &sources, 0).unwrap();
        for b in &streamed {
            assert_eq!(b.column(0).data_type(), &DataType::Float64);
        }
        let total: usize = streamed.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 3);
    }

    /// A source whose batches disagree about their schema is the one shape a peeked head
    /// cannot speak for, so the streaming union declines and the materializing path answers.
    #[test]
    fn a_heterogeneous_source_declines() {
        let sources = vec![vec![ints(vec![1]), floats(vec![2.5])], vec![ints(vec![3])]];
        let inputs = vec![scan(0), scan(1)];
        let cache = super::super::builds::prebuild_joins(
            &RelOp::Union {
                inputs: inputs.clone(),
                distinct: false,
            },
            &sources,
            None,
            0,
            1,
        )
        .unwrap();
        let ctx = Ctx::new(&sources, &cache, None, 0);
        assert!(!sources_are_homogeneous(&inputs, ctx));
        // …and the query still answers, on the path that reads every batch.
        let plan = RelOp::Union {
            inputs,
            distinct: false,
        };
        let streamed = crate::execute_streaming(&plan, &sources, 0).unwrap();
        let total: usize = streamed.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 3);
    }

    /// An empty branch contributes no rows and no schema, matching the oracle's `extend`.
    #[test]
    fn an_empty_branch_contributes_nothing() {
        let sources = vec![vec![ints(vec![1, 2])], vec![]];
        let plan = RelOp::Union {
            inputs: vec![scan(0), scan(1)],
            distinct: false,
        };
        let streamed = crate::execute_streaming(&plan, &sources, 0).unwrap();
        let total: usize = streamed.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2);
    }
}
