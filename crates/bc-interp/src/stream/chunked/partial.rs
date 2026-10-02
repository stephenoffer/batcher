//! The map side of a distributed aggregate, with the workers reading their own units.
//!
//! A distributed `GROUP BY` maps each partition to *partial* aggregate state, which the
//! reducers combine (`dist::combine_finalize`). The map used to be three native calls per
//! byte-bounded chunk, driven from Python: decode the chunk, run the map prefix over it
//! (handing every mapped row back across the FFI), then `partial_aggregate` and `combine`.
//! Decoding and computing only overlapped through a read-ahead queue, and every call started
//! its own parallel pass over one chunk: an SF1000 TPC-H worker held its sixteen cores at 57%
//! with the object cache warm and nothing left to wait on the network for.
//!
//! Here the partition's units -- Parquet row groups, for the caller in `bc-py` -- are split
//! into ranges and each engine worker reads a range one unit at a time, pushes it through the
//! map prefix and folds it into its own partial ([`super::Run::fold_units`], the same fold the
//! single-node unit executor finalizes). The worker partials are combined once and returned in
//! `partial_aggregate`'s wire format, so the reducers see exactly what they always did:
//! `combine` is associative and commutative, which is what makes one partial per range the
//! same state as one per chunk (invariant #7).

use arrow::array::RecordBatch;
use bc_ir::{AggregateItem, ProjectionItem, RelOp};
use bc_runtime::agg;

use super::orient::orient_counted;
use super::{oriented_core, unit_ranges, units, Core, Run};
use crate::error::InterpError;
use crate::ops;
use crate::par::ExecOptions;

/// Fold `sources[driving]`, read unit by unit from `src` by the workers, through `map` into one
/// partial aggregate state batch -- the batch `dist::partial_aggregate` returns for `map`'s rows.
///
/// `map` is the breaker-free prefix a distributed aggregate runs on its map side, scanning
/// `driving`; `sources[driving]` is the zero-row schema carrier. The metrics number `map`'s
/// operators as the control plane numbers the map plan, and are empty when the prefix had to be
/// re-oriented (a count filed under another operator's id teaches worse than none).
///
/// # Errors
/// [`InterpError::NotChunkable`] when `map` is not a spine over `driving` that the unit executor
/// can stream -- the caller reads the partition itself instead; anything a unit read or an
/// operator reports.
#[allow(clippy::too_many_arguments)]
pub fn partial_aggregate_units(
    map: &RelOp,
    group_keys: &[ProjectionItem],
    aggregates: &[AggregateItem],
    sources: &[Vec<RecordBatch>],
    driving: usize,
    src: &dyn units::UnitSource,
    workers: usize,
    opts: &ExecOptions,
) -> Result<(RecordBatch, crate::ExecMetrics), InterpError> {
    let (oriented, swaps) = orient_counted(map, driving);
    let node = RelOp::Aggregate {
        input: Box::new(oriented),
        group_keys: group_keys.to_vec(),
        aggregates: aggregates.to_vec(),
    };
    // The aggregate must be the plan's root: a map prefix holds nothing above it.
    match oriented_core(&node, driving) {
        Some(Core::Aggregate { path, .. }) if path.is_empty() => {}
        _ => return Err(InterpError::NotChunkable),
    }
    let RelOp::Aggregate { input, .. } = &node else {
        unreachable!("built as an aggregate just above")
    };
    let workers = workers.max(1);
    // Numbered over the aggregate's own input box, so the addresses the fold meters are the
    // addresses it numbered and the ids are the map plan's pre-order ids.
    let meter = (swaps == 0).then(|| crate::stream::Meter::new(input, workers as u32));
    let run = Run {
        driving,
        workers,
        // No budget: the map side holds one partial per range, bounded by the groups, and the
        // reducers' combine is where a group-heavy aggregate spills.
        budget: 0,
        opts,
        carrier: sources[driving].clone(),
        driving_rows: src.rows(),
        pool: crate::par::pool_for(workers)?,
    };
    let mut srcs: Vec<Vec<RecordBatch>> = sources.to_vec();
    srcs[driving] = run.carrier.clone();
    let ranges = unit_ranges(src.units(), run.workers);
    let folded = run.fold_units(&node, &srcs, src, &ranges, meter.as_ref())?;
    let batch = if folded.partials.is_empty() {
        // No unit held a row: the empty partial, typed by the map prefix over the carrier.
        let mapped = crate::execute(input, &srcs)?;
        crate::dist::partial_aggregate(group_keys, aggregates, &mapped)?
    } else {
        let funcs = ops::agg_funcs(aggregates);
        let merged = run
            .pool
            .install(|| agg::combine(&folded.partials, &funcs))?;
        crate::dist::partial_to_batch(group_keys, aggregates, &merged)?
    };
    let metrics = meter.map(|m| m.finish()).unwrap_or_default();
    Ok((batch, metrics))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Array, Float64Array, Int64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use bc_expr::{BinaryOp, Expr, Literal};
    use bc_ir::{AggFunc, AggregateItem, ProjectionItem, RelOp};

    use super::partial_aggregate_units;
    use crate::error::InterpError;
    use crate::par::ExecOptions;

    fn fact(lo: i64, hi: i64) -> RecordBatch {
        let k: Vec<Option<i64>> = (lo..hi).map(|i| (i % 41 != 0).then_some(i % 97)).collect();
        let v: Vec<f64> = (lo..hi).map(|i| (i % 13) as f64 + 0.25).collect();
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("k", DataType::Int64, true),
                Field::new("v", DataType::Float64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(k)),
                Arc::new(Float64Array::from(v)),
            ],
        )
        .unwrap()
    }

    fn item(func: AggFunc, col: Option<&str>, alias: &str) -> AggregateItem {
        AggregateItem {
            func,
            input: col.map(|c| Expr::Col { name: c.into() }),
            input2: None,
            order_by: Vec::new(),
            alias: alias.into(),
            param: None,
            interpolation: None,
        }
    }

    /// `Filter(v > threshold)` over the driving scan: the shape of a TPC-H map prefix.
    fn map(threshold: f64) -> RelOp {
        RelOp::Filter {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            predicate: Expr::Binary {
                op: BinaryOp::Gt,
                left: Box::new(Expr::Col { name: "v".into() }),
                right: Box::new(Expr::Lit {
                    value: Literal::Float(threshold),
                }),
            },
        }
    }

    struct Units(Vec<Vec<RecordBatch>>);

    impl crate::stream::UnitSource for Units {
        fn units(&self) -> usize {
            self.0.len()
        }
        fn read(&self, unit: usize) -> Result<Vec<RecordBatch>, InterpError> {
            Ok(self.0[unit].clone())
        }
    }

    fn keys() -> Vec<ProjectionItem> {
        vec![ProjectionItem {
            expr: Expr::Col { name: "k".into() },
            alias: "k".into(),
        }]
    }

    fn aggs() -> Vec<AggregateItem> {
        vec![
            item(AggFunc::Sum, Some("v"), "s"),
            item(AggFunc::CountStar, None, "n"),
            item(AggFunc::Mean, Some("v"), "m"),
            item(AggFunc::Min, Some("v"), "lo"),
        ]
    }

    /// Finalized rows, sorted by group key (NULL first), as comparable text.
    fn finalize(partial: &RecordBatch) -> Vec<String> {
        let out =
            crate::dist::combine_finalize(&keys(), &aggs(), std::slice::from_ref(partial)).unwrap();
        let mut rows: Vec<String> = (0..out.num_rows())
            .map(|r| {
                (0..out.num_columns())
                    .map(|c| {
                        let col = out.column(c);
                        if col.is_null(r) {
                            "null".to_string()
                        } else if let Some(f) = col.as_any().downcast_ref::<Float64Array>() {
                            format!("{:.9}", f.value(r))
                        } else {
                            arrow::util::display::array_value_to_string(col, r).unwrap()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("|")
            })
            .collect();
        rows.sort();
        rows
    }

    /// The oracle: the map prefix over the whole relation, then `partial_aggregate`.
    fn oracle(whole: &[RecordBatch], threshold: f64) -> Vec<String> {
        let mapped = crate::execute(&map(threshold), &[whole.to_vec()]).unwrap();
        let partial = crate::dist::partial_aggregate(&keys(), &aggs(), &mapped).unwrap();
        finalize(&partial)
    }

    /// The source list the fold runs over: only the driving relation's zero-row carrier.
    fn carrier() -> Vec<Vec<RecordBatch>> {
        vec![vec![RecordBatch::new_empty(fact(0, 1).schema())]]
    }

    #[test]
    fn the_unit_fold_is_the_partial_the_two_call_path_builds() {
        let whole: Vec<RecordBatch> = (0..9).map(|c| fact(c * 40_000, (c + 1) * 40_000)).collect();
        let layouts = [
            vec![whole.clone()],                             // one unit
            whole.iter().map(|b| vec![b.clone()]).collect(), // a unit per batch
            vec![whole[..4].to_vec(), whole[4..].to_vec()],  // uneven units
        ];
        for threshold in [1.0, 99.0] {
            // 99.0 keeps no row: every unit folds to nothing and the empty partial must still
            // carry the aggregate's types.
            let want = oracle(&whole, threshold);
            for layout in &layouts {
                for workers in [1, 3, 8] {
                    let (partial, _) = partial_aggregate_units(
                        &map(threshold),
                        &keys(),
                        &aggs(),
                        &carrier(),
                        0,
                        &Units(layout.clone()),
                        workers,
                        &ExecOptions::default(),
                    )
                    .unwrap();
                    assert_eq!(
                        finalize(&partial),
                        want,
                        "threshold {threshold} workers {workers}"
                    );
                }
            }
        }
    }

    #[test]
    fn no_units_is_the_typed_empty_partial() {
        let (partial, _) = partial_aggregate_units(
            &map(1.0),
            &keys(),
            &aggs(),
            &carrier(),
            0,
            &Units(Vec::new()),
            4,
            &ExecOptions::default(),
        )
        .unwrap();
        assert_eq!(partial.num_rows(), 0);
        assert_eq!(finalize(&partial), oracle(&carrier()[0], 1.0));
    }

    #[test]
    fn the_metrics_number_the_map_prefix_as_its_plan_does() {
        let whole: Vec<RecordBatch> = (0..3).map(|c| fact(c * 1_000, (c + 1) * 1_000)).collect();
        let units = Units(whole.iter().map(|b| vec![b.clone()]).collect());
        let (_, metrics) = partial_aggregate_units(
            &map(1.0),
            &keys(),
            &aggs(),
            &carrier(),
            0,
            &units,
            2,
            &ExecOptions::default(),
        )
        .unwrap();
        let json = metrics.to_json();
        // Op 0 is the filter (the map plan's root), op 1 its scan: 3,000 rows read.
        assert!(json.contains("\"op_id\":0"), "{json}");
        assert!(
            !json.contains("\"op_id\":2"),
            "the aggregate is not the map plan's: {json}"
        );
    }
}
