//! Restrict a join's build-side aggregate to the keys its (already materialized) probe side holds.
//!
//! A correlated subquery decorrelates to `Join(outer, Aggregate(inner, group_keys=[k]))`, and the
//! aggregate is computed over the *whole* inner relation even though only the outer side's few
//! keys are ever read back. TPC-H q21 is the canonical shape: a `GROUP BY l_orderkey` over all
//! 6M `lineitem` rows builds ~1.5M groups, and the ~72k orders the outer spine produces read ~5%
//! of them. The rest are built, finalized, hashed into the join's table and discarded.
//!
//! The materializing executor evaluates a join's left input before its right one, so by the time
//! the right side runs the probe keys are a known constant. This module turns them into a
//! [`KeyFilter`] and applies it to the **source** the aggregate reads, before the aggregate runs.
//! That is the same algebra as [`crate::stream::runtime_filter`]'s placement through a group key,
//! on the executor the streaming path cannot serve (a plan that scans a source twice, such as
//! q21's self-joins, never shards and runs here instead).
//!
//! ## Why it is sound
//!
//! 1. **The join discards unmatched right rows.** `Inner`, `Left`, `Semi` and `Anti` emit a right
//!    row only when a left row matched it (`Semi`/`Anti` emit no right row at all and only ask
//!    whether a match exists). So a right row whose key is absent from the left side contributes
//!    nothing, and neither does a group made of such rows. `Right`/`Full` preserve them and are
//!    refused.
//! 2. **The key is traced, not guessed.** From the right input down to a `Scan`, the key column
//!    may pass only through a `Filter` (changes rows, not columns), a `Project` that forwards it as
//!    a bare column, and an `Aggregate` that groups on it as a bare column. At an aggregate, every
//!    output key value appeared in its input, and deleting every input row of a key deletes
//!    exactly that key's group and nothing else — surviving groups keep all their rows, so their
//!    aggregate values are untouched. Anything else ends the trace and nothing is filtered.
//! 3. **The source is scanned exactly once in the right subtree.** The filtered relation replaces
//!    that source for the whole subtree, so a second scan of it (reached by some untraced path)
//!    would see rows removed that it was entitled to. `sources` for the left side, and for
//!    everything outside this join, are untouched.
//! 4. **The filter is exact.** [`KeyFilter`] never rejects a key the left side holds, and a NULL
//!    probe key is rejected — correct, because an equi-join never matches NULL.
//!
//! Applying it is optional, so every guard below may decline freely: declining runs the original
//! plan over the original sources.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch};
use bc_ir::{JoinType, RelOp};
use bc_runtime::join::KeyFilter;
use rayon::prelude::*;

use crate::error::InterpError;

/// The smallest source worth filtering. Below this the aggregate is a few milliseconds at most,
/// and the digest plus one mask pass over the source is a comparable cost.
const MIN_SOURCE_ROWS: usize = 262_144;

/// How many source rows per probe row, at least, before filtering is attempted.
///
/// Each probe row can keep at most a handful of source rows alive (the ones sharing its key), so
/// the ratio bounds the reduction on offer. Below 4x the mask would keep most of the source and
/// its copy would cost more than the groups it saves; the per-batch keep check below catches the
/// rest.
const MIN_SOURCE_PER_PROBE: usize = 4;

/// Keep fraction, in 1/256ths, above which the filtered copy is not worth making.
const MAX_KEEP_256: usize = 128;

/// `sources` with the right side's traced source restricted to `left_batches`' join keys, or
/// `None` when the join does not qualify or the restriction would not pay.
pub(crate) fn restrict_right_sources(
    join_type: JoinType,
    left_keys: &[String],
    right_keys: &[String],
    left_batches: &[RecordBatch],
    right: &RelOp,
    sources: &[Vec<RecordBatch>],
) -> Result<Option<Vec<Vec<RecordBatch>>>, InterpError> {
    if !matches!(
        join_type,
        JoinType::Inner | JoinType::Left | JoinType::Semi | JoinType::Anti
    ) {
        return Ok(None);
    }
    let ([left_key], [right_key]) = (left_keys, right_keys) else {
        return Ok(None);
    };
    let Some((source_id, column)) = trace_to_scan(right, right_key, needs_aggregate(join_type))
    else {
        return Ok(None);
    };
    if count_scans_of(right, source_id) != 1 {
        return Ok(None);
    }
    let Some(batches) = sources.get(source_id) else {
        return Ok(None);
    };
    let source_rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
    let probe_rows: usize = left_batches.iter().map(RecordBatch::num_rows).sum();
    if source_rows < MIN_SOURCE_ROWS
        || probe_rows == 0
        || source_rows < probe_rows.saturating_mul(MIN_SOURCE_PER_PROBE)
    {
        return Ok(None);
    }
    let Some(filter) = key_filter(left_batches, left_key)? else {
        return Ok(None);
    };
    let Some(filtered) = filter_source(batches, &column, &filter)? else {
        return Ok(None);
    };
    let mut out = sources.to_vec();
    out[source_id] = filtered;
    Ok(Some(out))
}

/// Whether `plan` holds a join whose build side [`restrict_right_sources`] could restrict: a
/// qualifying join type, one key traced to a source scanned once on that side (through an
/// aggregate unless the join is a semi or anti join), and a source of at least
/// [`MIN_SOURCE_ROWS`].
///
/// Read before execution, so it cannot know the probe side's size and the restriction may still
/// decline at run time; it answers only whether routing the plan to the executor that can
/// restrict (the materializing one) has anything to gain. TPC-H q21 at sf10 streams in ~1,040 ms
/// and runs materialized in ~670 ms, the difference being the 60M-row `GROUP BY l_orderkey`
/// this restriction cuts to the orders the outer query can match.
#[must_use]
pub fn sideways_candidate(plan: &RelOp, sources: &[Vec<RecordBatch>]) -> bool {
    if let RelOp::HashJoin {
        join_type: join_type @ (JoinType::Inner | JoinType::Left | JoinType::Semi | JoinType::Anti),
        right_keys,
        right,
        ..
    } = plan
    {
        if let [key] = right_keys.as_slice() {
            if let Some((source_id, _)) = trace_to_scan(right, key, needs_aggregate(*join_type)) {
                let rows: usize = sources
                    .get(source_id)
                    .map_or(0, |b| b.iter().map(RecordBatch::num_rows).sum());
                if count_scans_of(right, source_id) == 1 && rows >= MIN_SOURCE_ROWS {
                    return true;
                }
            }
        }
    }
    plan.children()
        .into_iter()
        .any(|c| sideways_candidate(c, sources))
}

/// The scan the key column of `plan` comes from and its name there, provided the path crosses an
/// `Aggregate` grouping on it when `need_aggregate` is set.
///
/// An inner or left join's build side needs one: a plain filtered scan there is already served
/// by the join's own probe bloom. A semi or anti join's does not, because what it builds is the
/// build side's key set and that set is the cost — TPC-H q4 builds 38M `lineitem` keys at sf10
/// to answer for the 574k orders its date filter keeps ([`needs_aggregate`]).
fn trace_to_scan(plan: &RelOp, key: &str, need_aggregate: bool) -> Option<(usize, String)> {
    let mut node = plan;
    let mut name = key.to_string();
    let mut crossed_aggregate = false;
    loop {
        match node {
            RelOp::Scan { source_id } => {
                return (crossed_aggregate || !need_aggregate).then_some((*source_id, name));
            }
            RelOp::Filter { input, .. } => node = input,
            RelOp::Project { input, exprs } => {
                let item = exprs.iter().find(|p| p.alias == name)?;
                let bc_expr::Expr::Col { name: source } = &item.expr else {
                    return None;
                };
                name = source.clone();
                node = input;
            }
            RelOp::Aggregate {
                input, group_keys, ..
            } => {
                let item = group_keys.iter().find(|k| k.alias == name)?;
                let bc_expr::Expr::Col { name: source } = &item.expr else {
                    return None;
                };
                name = source.clone();
                node = input;
                crossed_aggregate = true;
            }
            _ => return None,
        }
    }
}

/// Whether restricting `join_type`'s build side pays only through an aggregate (see
/// [`trace_to_scan`]).
fn needs_aggregate(join_type: JoinType) -> bool {
    !matches!(join_type, JoinType::Semi | JoinType::Anti)
}

fn count_scans_of(plan: &RelOp, source_id: usize) -> usize {
    let own = usize::from(matches!(plan, RelOp::Scan { source_id: s } if *s == source_id));
    own + plan
        .children()
        .into_iter()
        .map(|c| count_scans_of(c, source_id))
        .sum::<usize>()
}

/// Digest the left side's key column. `None` when it is absent, not `Int64`, or too large for the
/// exact set [`KeyFilter`] keeps.
fn key_filter(batches: &[RecordBatch], key: &str) -> Result<Option<KeyFilter>, InterpError> {
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(batches.len());
    for b in batches {
        let Some(c) = b.column_by_name(key) else {
            return Ok(None);
        };
        cols.push(Arc::clone(c));
    }
    let refs: Vec<&dyn Array> = cols.iter().map(AsRef::as_ref).collect();
    let keys = arrow::compute::concat(&refs)?;
    Ok(KeyFilter::build_once(&keys))
}

/// Every batch of the source masked by `filter`, or `None` when the column is missing or the
/// mask keeps too much of the source to be worth copying.
fn filter_source(
    batches: &[RecordBatch],
    column: &str,
    filter: &KeyFilter,
) -> Result<Option<Vec<RecordBatch>>, InterpError> {
    let masks: Option<Vec<_>> = batches
        .par_iter()
        .map(|b| b.column_by_name(column).and_then(|c| filter.mask(c)))
        .collect();
    let Some(masks) = masks else {
        return Ok(None);
    };
    let total: usize = batches.iter().map(RecordBatch::num_rows).sum();
    let kept: usize = masks.iter().map(|m| m.true_count()).sum();
    if kept * 256 > total * MAX_KEEP_256 {
        return Ok(None);
    }
    let out: Result<Vec<RecordBatch>, InterpError> = batches
        .par_iter()
        .zip(masks.par_iter())
        .map(|(b, m)| Ok(arrow::compute::filter_record_batch(b, m)?))
        .collect();
    let out = out?;
    // Drop the emptied batches, but keep one if every batch emptied: a source is read for its
    // schema as well as its rows, and a relation with no batches has none.
    let mut kept: Vec<RecordBatch> = out.iter().filter(|b| b.num_rows() > 0).cloned().collect();
    if kept.is_empty() {
        kept.extend(out.into_iter().take(1));
    }
    Ok(Some(kept))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Array, Int64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use bc_expr::Expr;
    use bc_ir::{AggFunc, AggregateItem, JoinOutputCol, JoinSide, JoinType, ProjectionItem, RelOp};

    use super::{restrict_right_sources, sideways_candidate};
    use crate::par::{execute_parallel_with_metrics, ExecOptions};

    fn batch(cols: &[(&str, Vec<Option<i64>>)]) -> RecordBatch {
        let schema = Schema::new(
            cols.iter()
                .map(|(n, _)| Field::new(*n, DataType::Int64, true))
                .collect::<Vec<_>>(),
        );
        let arrays = cols
            .iter()
            .map(|(_, v)| Arc::new(Int64Array::from(v.clone())) as Arc<dyn Array>)
            .collect();
        RecordBatch::try_new(Arc::new(schema), arrays).unwrap()
    }

    /// An outer side of 2,000 keys (with NULLs and duplicates) joined to a 400,000-row inner
    /// relation grouped on the key: large enough to clear every gate, so the restriction runs.
    fn sources() -> Vec<Vec<RecordBatch>> {
        let outer: Vec<Option<i64>> = (0..2_000)
            .map(|i| {
                if i % 97 == 0 {
                    None
                } else {
                    Some((i % 1_800) * 37)
                }
            })
            .collect();
        let tag: Vec<Option<i64>> = (0..2_000).map(Some).collect();
        let inner: Vec<RecordBatch> = (0..4)
            .map(|chunk| {
                let k = (0..100_000)
                    .map(|i| {
                        let r = chunk * 100_000 + i;
                        if r % 1_009 == 0 {
                            None
                        } else {
                            Some(r % 90_000)
                        }
                    })
                    .collect();
                let v = (0..100_000).map(|i| Some(chunk * 7 + i % 13)).collect();
                batch(&[("k", k), ("v", v)])
            })
            .collect();
        vec![vec![batch(&[("ok", outer), ("tag", tag)])], inner]
    }

    fn plan(join_type: JoinType) -> RelOp {
        let aggregate = RelOp::Aggregate {
            input: Box::new(RelOp::Filter {
                input: Box::new(RelOp::Scan { source_id: 1 }),
                predicate: Expr::Lit {
                    value: bc_expr::Literal::Bool(true),
                },
            }),
            group_keys: vec![ProjectionItem {
                expr: Expr::Col { name: "k".into() },
                alias: "gk".into(),
            }],
            aggregates: vec![
                AggregateItem {
                    func: AggFunc::Sum,
                    input: Some(Expr::Col { name: "v".into() }),
                    input2: None,
                    order_by: Vec::new(),
                    alias: "s".into(),
                    param: None,
                    interpolation: None,
                },
                AggregateItem {
                    func: AggFunc::CountStar,
                    input: None,
                    input2: None,
                    order_by: Vec::new(),
                    alias: "n".into(),
                    param: None,
                    interpolation: None,
                },
            ],
        };
        let mut output = vec![
            JoinOutputCol {
                side: JoinSide::Left,
                name: "ok".into(),
                alias: "ok".into(),
            },
            JoinOutputCol {
                side: JoinSide::Left,
                name: "tag".into(),
                alias: "tag".into(),
            },
        ];
        if !matches!(join_type, JoinType::Semi | JoinType::Anti) {
            output.push(JoinOutputCol {
                side: JoinSide::Right,
                name: "s".into(),
                alias: "s".into(),
            });
            output.push(JoinOutputCol {
                side: JoinSide::Right,
                name: "n".into(),
                alias: "n".into(),
            });
        }
        RelOp::HashJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(aggregate),
            left_keys: vec!["ok".into()],
            right_keys: vec!["gk".into()],
            join_type,
            output,
            strategy: bc_ir::JoinStrategy::Hash,
        }
    }

    fn rows(batches: &[RecordBatch]) -> Vec<Vec<Option<i64>>> {
        let mut out = Vec::new();
        for b in batches {
            for i in 0..b.num_rows() {
                out.push(
                    (0..b.num_columns())
                        .map(|c| {
                            let a = b.column(c).as_any().downcast_ref::<Int64Array>().unwrap();
                            a.is_valid(i).then(|| a.value(i))
                        })
                        .collect(),
                );
            }
        }
        out.sort_unstable();
        out
    }

    /// Every join type the restriction admits returns the sequential oracle's multiset, and the
    /// restriction demonstrably ran (the gate is not what made the comparison pass).
    #[test]
    fn the_restricted_join_matches_the_oracle() {
        let src = sources();
        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            let p = plan(jt);
            let RelOp::HashJoin { right, .. } = &p else {
                unreachable!()
            };
            let restricted = restrict_right_sources(
                jt,
                &["ok".to_string()],
                &["gk".to_string()],
                &src[0],
                right,
                &src,
            )
            .unwrap()
            .expect("the fixture must clear every gate");
            let kept: usize = restricted[1].iter().map(RecordBatch::num_rows).sum();
            assert!(kept > 0 && kept < 400_000 / 4, "{jt:?} kept {kept}");
            let oracle = rows(&crate::execute(&p, &src).unwrap());
            let (par, metrics) =
                execute_parallel_with_metrics(&p, &src, &ExecOptions::default()).unwrap();
            let par = rows(&par);
            // The restricted subtree reports to a scratch collector, so an aggregate metric in
            // the result means the executor took some other route and this test proved nothing.
            assert!(
                metrics.ops.iter().all(|o| o.kind != "aggregate"),
                "{jt:?}: the restricted path did not run"
            );
            assert!(!oracle.is_empty(), "{jt:?} oracle is empty");
            assert_eq!(par, oracle, "{jt:?}");
        }
    }

    /// A join that keeps unmatched right rows is refused: filtering its build side would delete
    /// output rows.
    #[test]
    fn a_right_preserving_join_is_not_restricted() {
        let src = sources();
        for jt in [JoinType::Right, JoinType::Full] {
            let p = plan(jt);
            let RelOp::HashJoin { right, .. } = &p else {
                unreachable!()
            };
            let r =
                restrict_right_sources(jt, &["ok".into()], &["gk".into()], &src[0], right, &src)
                    .unwrap();
            assert!(r.is_none(), "{jt:?}");
        }
    }

    /// The routing check sees exactly the shape the restriction serves, and declines the rest:
    /// a right/full join, a source under the row floor, and a build side with no aggregate.
    #[test]
    fn the_routing_check_sees_the_shape_the_restriction_serves() {
        let srcs = sources();
        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            assert!(sideways_candidate(&plan(jt), &srcs), "{jt:?}");
        }
        assert!(!sideways_candidate(&plan(JoinType::Full), &srcs));
        let small: Vec<Vec<RecordBatch>> = vec![srcs[0].clone(), vec![srcs[1][0].slice(0, 1_000)]];
        assert!(
            !sideways_candidate(&plan(JoinType::Inner), &small),
            "under the row floor"
        );
        let RelOp::HashJoin {
            left,
            left_keys,
            join_type,
            output,
            strategy,
            ..
        } = plan(JoinType::Inner)
        else {
            unreachable!()
        };
        let no_aggregate = RelOp::HashJoin {
            left,
            right: Box::new(RelOp::Scan { source_id: 1 }),
            left_keys,
            right_keys: vec!["k".into()],
            join_type,
            output,
            strategy,
        };
        assert!(
            !sideways_candidate(&no_aggregate, &srcs),
            "no aggregate to restrict"
        );
    }

    /// A semi or anti join is restricted with no aggregate on its build side — its build is the
    /// right side's key set, which is the cost the restriction removes — and still returns the
    /// oracle's rows. An inner join over the same plain scan is left alone (the probe bloom
    /// already serves it).
    #[test]
    fn a_semi_or_anti_build_is_restricted_without_an_aggregate() {
        let src = sources();
        let plain = |jt: JoinType| RelOp::HashJoin {
            left: Box::new(RelOp::Scan { source_id: 0 }),
            right: Box::new(RelOp::Filter {
                input: Box::new(RelOp::Scan { source_id: 1 }),
                predicate: Expr::Lit {
                    value: bc_expr::Literal::Bool(true),
                },
            }),
            left_keys: vec!["ok".into()],
            right_keys: vec!["k".into()],
            join_type: jt,
            output: vec![
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "ok".into(),
                    alias: "ok".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "tag".into(),
                    alias: "tag".into(),
                },
            ],
            strategy: bc_ir::JoinStrategy::Hash,
        };
        for jt in [JoinType::Semi, JoinType::Anti] {
            let p = plain(jt);
            assert!(sideways_candidate(&p, &src), "{jt:?}");
            let RelOp::HashJoin { right, .. } = &p else {
                unreachable!()
            };
            let restricted =
                restrict_right_sources(jt, &["ok".into()], &["k".into()], &src[0], right, &src)
                    .unwrap()
                    .expect("the fixture must clear every gate");
            let kept: usize = restricted[1].iter().map(RecordBatch::num_rows).sum();
            assert!(kept > 0 && kept < 400_000 / 4, "{jt:?} kept {kept}");
            let oracle = rows(&crate::execute(&p, &src).unwrap());
            let (par, _) =
                execute_parallel_with_metrics(&p, &src, &ExecOptions::default()).unwrap();
            assert!(!oracle.is_empty(), "{jt:?} oracle is empty");
            assert_eq!(rows(&par), oracle, "{jt:?}");
        }
        assert!(!sideways_candidate(&plain(JoinType::Inner), &src));
    }
}
