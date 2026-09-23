//! Whether anything above a stream stage can observe the *order* of its output rows.
//!
//! This executor's contract is the oracle's rows **in the oracle's order**, because a `LIMIT`
//! over, say, a semi join keeps whichever rows arrive first. That is why the materializing join
//! arm stays on one core (see `materialized_join_from`): both parallel joins emit the right rows
//! in a different order. But most joins are not under a `LIMIT`. A global `COUNT(*)` over a
//! join reads a multiset; so does a grouped aggregate whose groups are then sorted by every
//! group key. There the order is unobservable and the parallel join is free to run.
//!
//! The flag travels down [`super::Ctx`]: an operator that only reads a multiset of its input
//! (an order-insensitive aggregate) sets it for its child, the row-wise operators that keep
//! their input's order one-for-one (`Filter`, `Project`, a join's two sides) pass it through,
//! and every other operator clears it. Clearing is the default, so an operator added later is
//! order-*sensitive* until someone argues otherwise.
//!
//! What "the same answer" means for the float reductions admitted here is the contract's own
//! float exception: identical up to reassociation. The streaming fold already partials per
//! morsel and combines, so it never summed in the oracle's single left-to-right pass.

use bc_expr::Expr;
use bc_ir::{AggFunc, AggregateItem, RelOp, SortKey};

/// Whether every aggregate's result is a function of the multiset of its input rows alone.
///
/// Excluded are the ones whose answer names *which* row: `list_agg` (element order),
/// `arg_min`/`arg_max` (the first of tied keys), `mode` (the first of tied counts), `histogram`
/// (entry order), the order-sensitive TDigest behind `approx_quantile`, and `product`, which the
/// global-window algebra also refuses on numerical grounds (an `inf * 0` whose sign depends on
/// the association order). Any aggregate with its own `ORDER BY` is excluded too.
pub(super) fn aggregates_ignore_order(aggregates: &[AggregateItem]) -> bool {
    aggregates.iter().all(|a| {
        a.order_by.is_empty()
            && matches!(
                a.func,
                AggFunc::CountStar
                    | AggFunc::Count
                    | AggFunc::CountDistinct
                    | AggFunc::Sum
                    | AggFunc::Min
                    | AggFunc::Max
                    | AggFunc::Mean
                    | AggFunc::Var
                    | AggFunc::Stddev
                    | AggFunc::Median
                    | AggFunc::Quantile
                    | AggFunc::BoolAnd
                    | AggFunc::BoolOr
                    | AggFunc::ApproxCountDistinct
                    | AggFunc::NLength
                    | AggFunc::LCount
                    | AggFunc::AuN
                    | AggFunc::BitAnd
                    | AggFunc::BitOr
                    | AggFunc::BitXor
                    | AggFunc::CovarPop
                    | AggFunc::CovarSamp
                    | AggFunc::Corr
                    | AggFunc::Skewness
                    | AggFunc::Kurtosis
                    | AggFunc::AnyValue
            )
    })
}

/// Whether sorting `input` by `keys` is a **total** order: `input` is an aggregate (reached
/// through filters and renaming projections) and every one of its group keys is a sort key.
///
/// Groups are unique on their keys, so no two rows tie, and a stable sort's tie-break -- the
/// only place its input order shows -- never runs. `ORDER BY o_orderpriority` over
/// `GROUP BY o_orderpriority` (TPC-H q4) is the shape.
pub(super) fn sort_is_total_over_groups(keys: &[SortKey], input: &RelOp) -> bool {
    let mut names: Vec<&str> = Vec::with_capacity(keys.len());
    for k in keys {
        match &k.expr {
            Expr::Col { name } => names.push(name),
            _ => return false,
        }
    }
    let mut node = input;
    loop {
        match node {
            RelOp::Filter { input, .. } => node = input,
            RelOp::Project { input, exprs } => {
                // Follow a sort key through a rename; a key the projection computes is not a
                // column of the aggregate, and stops the walk.
                let mut renamed = Vec::with_capacity(names.len());
                for n in &names {
                    match exprs.iter().find(|p| p.alias == *n).map(|p| &p.expr) {
                        Some(Expr::Col { name }) => renamed.push(name.as_str()),
                        _ => return false,
                    }
                }
                names = renamed;
                node = input;
            }
            RelOp::Aggregate { group_keys, .. } => {
                return group_keys.iter().all(|g| names.contains(&g.alias.as_str()));
            }
            _ => return false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bc_ir::ProjectionItem;

    fn col(name: &str) -> Expr {
        Expr::Col { name: name.into() }
    }

    fn agg(func: AggFunc) -> AggregateItem {
        AggregateItem {
            func,
            input: None,
            input2: None,
            param: None,
            interpolation: None,
            order_by: vec![],
            alias: "a".into(),
        }
    }

    fn grouped(keys: &[&str]) -> RelOp {
        RelOp::Aggregate {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            group_keys: keys
                .iter()
                .map(|k| ProjectionItem {
                    expr: col(k),
                    alias: (*k).into(),
                })
                .collect(),
            aggregates: vec![],
        }
    }

    fn by(names: &[&str]) -> Vec<SortKey> {
        names
            .iter()
            .map(|n| SortKey {
                expr: col(n),
                descending: false,
                nulls_first: false,
            })
            .collect()
    }

    #[test]
    fn a_first_seen_aggregate_is_order_sensitive() {
        assert!(aggregates_ignore_order(&[agg(AggFunc::CountStar), agg(AggFunc::Sum)]));
        for f in [
            AggFunc::ListAgg,
            AggFunc::ArgMin,
            AggFunc::Mode,
            AggFunc::ApproxQuantile,
            AggFunc::Product,
        ] {
            assert!(!aggregates_ignore_order(&[agg(AggFunc::Count), agg(f)]), "{f:?}");
        }
    }

    #[test]
    fn only_a_sort_on_every_group_key_is_total() {
        let plan = grouped(&["g", "h"]);
        assert!(sort_is_total_over_groups(&by(&["h", "g"]), &plan));
        assert!(sort_is_total_over_groups(&by(&["x", "g", "h"]), &plan));
        assert!(!sort_is_total_over_groups(&by(&["g"]), &plan));
        assert!(!sort_is_total_over_groups(
            &by(&["g"]),
            &RelOp::Scan { source_id: 0 }
        ));
        // Through a renaming projection; a computed one stops the walk.
        let renamed = RelOp::Project {
            input: Box::new(grouped(&["g"])),
            exprs: vec![ProjectionItem {
                expr: col("g"),
                alias: "k".into(),
            }],
        };
        assert!(sort_is_total_over_groups(&by(&["k"]), &renamed));
        assert!(!sort_is_total_over_groups(&by(&["g"]), &renamed));
    }

    /// End to end: a global count and a group-key-sorted grouped count over a `FULL` join are
    /// the oracle's answer on the parallel streaming executor, which is where this flag lets the
    /// join leave one core. The join is large enough to take the partitioned path's buckets.
    #[test]
    fn an_order_free_full_join_is_the_oracles_answer() {
        use arrow::array::{Array, Int64Array};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use bc_ir::{JoinOutputCol, JoinSide, JoinStrategy, JoinType};
        use std::sync::Arc;

        let side = |name: &str, keys: Vec<i64>| {
            let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int64, true)]));
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(keys))]).unwrap()
        };
        let left: Vec<i64> = (0..60_000).filter(|k| k % 3 == 0).collect();
        let right: Vec<i64> = (0..60_000).filter(|k| k % 2 == 0).map(|k| k % 40_000).collect();
        let sources = vec![vec![side("l", left)], vec![side("r", right)]];
        let join = RelOp::HashJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(RelOp::Scan { source_id: 1 }),
            left_keys: vec!["l".into()],
            right_keys: vec!["r".into()],
            join_type: JoinType::Full,
            output: vec![
                JoinOutputCol { side: JoinSide::Left, name: "l".into(), alias: "l".into() },
                JoinOutputCol { side: JoinSide::Right, name: "r".into(), alias: "r".into() },
            ],
            strategy: JoinStrategy::Hash,
        };
        let count = |func: AggFunc, input: Option<Expr>, alias: &str| AggregateItem {
            func,
            input,
            input2: None,
            param: None,
            interpolation: None,
            order_by: vec![],
            alias: alias.into(),
        };
        let global = RelOp::Aggregate {
            input: Box::new(join.clone()),
            group_keys: vec![],
            aggregates: vec![
                count(AggFunc::CountStar, None, "n"),
                count(AggFunc::Count, Some(col("l")), "nl"),
                count(AggFunc::Count, Some(col("r")), "nr"),
            ],
        };
        let sorted = RelOp::Sort {
            input: Box::new(RelOp::Aggregate {
                input: Box::new(join),
                group_keys: vec![ProjectionItem { expr: col("r"), alias: "r".into() }],
                aggregates: vec![count(AggFunc::CountStar, None, "n")],
            }),
            keys: by(&["r"]),
            limit: None,
        };
        for plan in [global, sorted] {
            let oracle = crate::execute(&plan, &sources).unwrap();
            let got = super::super::execute_streaming_parallel(&plan, &sources, 8, 0, None).unwrap();
            let rows = |b: &[RecordBatch]| {
                let t = crate::ops::materialize(b).unwrap();
                (0..t.num_columns())
                    .map(|c| {
                        let a = t.column(c).as_any().downcast_ref::<Int64Array>().unwrap();
                        (0..a.len()).map(|i| a.is_valid(i).then(|| a.value(i))).collect()
                    })
                    .collect::<Vec<Vec<Option<i64>>>>()
            };
            assert_eq!(rows(&oracle), rows(&got));
        }
    }
}
