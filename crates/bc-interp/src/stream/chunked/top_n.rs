//! A top-N over the driving scan, materialized late: sort the narrow columns, fetch the winners.
//!
//! `SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10` keeps ten rows,
//! and the spine under its sort decodes every column of every row the filter keeps to find
//! them — 105 columns of ClickBench's `hits`, which is where its time goes: the pages that hold
//! a survivor are decompressed whole, and a scattered survivor is in most pages. DuckDB
//! answers it by sorting row identifiers and fetching ten rows at the end.
//!
//! So does this. When the post ops over a spine of filters begin with a `Sort` that keeps at
//! most [`MAX_ROWS`] rows, and the source can [narrow](UnitSource::narrowed) itself, the spine
//! runs over only the columns its filters and the sort keys read plus a [`LOCATOR`]; the sort
//! picks the winners there; the source [fetches](UnitSource::fetch) just those rows whole, in
//! the sort's order; and the post ops above the sort run over them.
//!
//! # Why this returns the rows the plan does
//!
//! The narrow spine keeps exactly the rows the wide one keeps — its filters read the same
//! columns of the same rows — and the units are read in the same contiguous, in-order ranges,
//! so the collected rows arrive in the same order. The sort is the plan's own, run as the post
//! ops are, and it computes what the sequential oracle does (seq == par): a stable sort by its
//! keys alone, ties resolved to input order. Its permutation is therefore a function of the key
//! columns and their order, both of which the narrow rows share with the wide ones, so the
//! winners are the same rows in the same order, and the fetch returns them as they are in the
//! source.

use std::sync::Arc;

use arrow::array::{Array, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use bc_ir::RelOp;

use super::units::{UnitSource, LOCATOR};
use super::Run;
use crate::error::InterpError;

/// The largest top-N fetched late. Every winner can cost a row group's decode in the fetch,
/// so past a few thousand the fetch approaches the read it was meant to avoid.
pub(super) const MAX_ROWS: usize = 4_096;

impl Run<'_> {
    /// The plan's rows by way of a late-materialized top-N, or `None` when the shape or the
    /// source does not allow one. `spine` is the plan's node at `depth` zeros below its root.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn top_n_late(
        &self,
        plan: &RelOp,
        depth: usize,
        spine: &RelOp,
        srcs: &[Vec<RecordBatch>],
        src: &dyn UnitSource,
        ranges: &[std::ops::Range<usize>],
        meter: Option<&crate::stream::Meter>,
    ) -> Result<Option<Vec<RecordBatch>>, InterpError> {
        let Some(sort_depth) = depth.checked_sub(1) else {
            return Ok(None);
        };
        let sort_path = vec![0; sort_depth];
        let RelOp::Sort {
            keys,
            limit: Some(limit),
            ..
        } = node(plan, &sort_path)
        else {
            return Ok(None);
        };
        // Each winner is fetched apart from the rest, so the late form only pays when there are
        // fewer of them than units to skip.
        if *limit > MAX_ROWS || limit.saturating_mul(2) > src.units() {
            return Ok(None);
        }
        let Some(columns) = needed(spine, keys, self.driving) else {
            return Ok(None);
        };
        let Some(schema) = self.carrier.first().map(RecordBatch::schema) else {
            return Ok(None);
        };
        let Some(carrier) = narrow_carrier(&schema, &columns) else {
            return Ok(None);
        };
        let Some(narrow) = src.narrowed(&columns) else {
            return Ok(None);
        };
        let mut narrow_srcs = srcs.to_vec();
        narrow_srcs[self.driving] = vec![carrier];
        let collected = self.collect_units(spine, &narrow_srcs, narrow.as_ref(), ranges, meter)?;
        // The winners, by the plan's own sort over the narrow rows.
        let sort = RelOp::Sort {
            input: Box::new(RelOp::Scan {
                source_id: narrow_srcs.len(),
            }),
            keys: keys.clone(),
            limit: Some(*limit),
        };
        narrow_srcs[self.driving] = Vec::new();
        narrow_srcs.push(collected);
        let winners = crate::par::execute_parallel_with(&sort, &narrow_srcs, self.opts)?;
        let locators = locators_of(&winners)?;
        let rows = if locators.is_empty() {
            self.carrier.clone()
        } else {
            src.fetch(&locators)?
        };
        if sort_depth == 0 {
            return Ok(Some(rows));
        }
        // The post ops above the sort, over the winners already in its order.
        let mut post = plan.clone();
        *super::node_at(&mut post, &sort_path) = RelOp::Scan {
            source_id: srcs.len(),
        };
        let mut post_srcs = srcs.to_vec();
        post_srcs[self.driving] = Vec::new();
        post_srcs.push(rows);
        crate::par::execute_parallel_with(&post, &post_srcs, self.opts).map(Some)
    }
}

/// The node reached from `plan` by following the child indices in `path`.
fn node<'a>(plan: &'a RelOp, path: &[usize]) -> &'a RelOp {
    path.iter().fold(plan, |n, &i| {
        n.children()
            .into_iter()
            .nth(i)
            .expect("the path was recorded on this plan's shape")
    })
}

/// The scan columns the spine's filters and the sort keys read, in first-read order, when the
/// spine is filters over the driving scan and nothing else.
fn needed(spine: &RelOp, keys: &[bc_ir::SortKey], driving: usize) -> Option<Vec<String>> {
    let mut names: Vec<&str> = Vec::new();
    let mut node = spine;
    loop {
        match node {
            RelOp::Filter { input, predicate } => {
                predicate.collect_columns(&mut names);
                node = input;
            }
            RelOp::Scan { source_id } if *source_id == driving => break,
            _ => return None,
        }
    }
    for key in keys {
        key.expr.collect_columns(&mut names);
    }
    let mut out: Vec<String> = Vec::new();
    for name in names {
        if !out.iter().any(|n| n == name) {
            out.push(name.to_string());
        }
    }
    Some(out)
}

/// The zero-row carrier of the narrowed relation: `columns` of `schema`, then the locator.
fn narrow_carrier(schema: &Schema, columns: &[String]) -> Option<RecordBatch> {
    let mut fields: Vec<Field> = Vec::with_capacity(columns.len() + 1);
    for name in columns {
        fields.push(schema.field_with_name(name).ok()?.clone());
    }
    fields.push(Field::new(LOCATOR, DataType::UInt64, false));
    Some(RecordBatch::new_empty(Arc::new(Schema::new(fields))))
}

/// The locator column of the sorted winners, in order.
fn locators_of(winners: &[RecordBatch]) -> Result<Vec<u64>, InterpError> {
    let mut out = Vec::new();
    for batch in winners {
        let column = batch
            .column_by_name(LOCATOR)
            .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
            .ok_or_else(|| InterpError::ChunkSource("a narrowed read lost its locator".into()))?;
        if column.null_count() > 0 {
            return Err(InterpError::ChunkSource(
                "a narrowed read has a null locator".into(),
            ));
        }
        out.extend_from_slice(column.values());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use bc_expr::{BinaryOp, Expr, Literal};
    use bc_ir::{ProjectionItem, RelOp, SortKey};

    use super::super::execute_units;
    use super::super::units::{UnitSource, LOCATOR};
    use crate::error::InterpError;
    use crate::par::ExecOptions;

    /// Rows `lo..hi`: `k` with many ties and some nulls, `v` with more ties, `w` a wide text
    /// column the sort never reads, `id` the row's own number.
    fn wide(lo: i64, hi: i64) -> RecordBatch {
        let ids = lo..hi;
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("k", DataType::Int64, true),
                Field::new("v", DataType::Float64, false),
                Field::new("w", DataType::Utf8, true),
                Field::new("id", DataType::Int64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(
                    ids.clone()
                        .map(|i| (i % 29 != 0).then_some(i % 11))
                        .collect::<Vec<_>>(),
                )) as ArrayRef,
                Arc::new(Float64Array::from(
                    ids.clone().map(|i| (i % 5) as f64).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    ids.clone()
                        .map(|i| (i % 3 != 0).then(|| format!("w{i}-{}", "x".repeat(40))))
                        .collect::<Vec<_>>(),
                )),
                Arc::new(Int64Array::from(ids.collect::<Vec<_>>())),
            ],
        )
        .unwrap()
    }

    /// Units in memory that can narrow and fetch, counting both.
    struct Located {
        units: Vec<RecordBatch>,
        narrowed: AtomicUsize,
        fetched: AtomicUsize,
    }

    struct Narrow<'a> {
        base: &'a Located,
        columns: Vec<String>,
    }

    impl UnitSource for Located {
        fn units(&self) -> usize {
            self.units.len()
        }
        fn read(&self, unit: usize) -> Result<Vec<RecordBatch>, InterpError> {
            Ok(vec![self.units[unit].clone()])
        }
        fn narrowed(&self, columns: &[String]) -> Option<Box<dyn UnitSource + '_>> {
            self.narrowed.fetch_add(1, Ordering::Relaxed);
            Some(Box::new(Narrow {
                base: self,
                columns: columns.to_vec(),
            }))
        }
        fn fetch(&self, locators: &[u64]) -> Result<Vec<RecordBatch>, InterpError> {
            self.fetched.fetch_add(locators.len(), Ordering::Relaxed);
            let rows: Vec<RecordBatch> = locators
                .iter()
                .map(|l| self.units[(l >> 32) as usize].slice((l & 0xffff_ffff) as usize, 1))
                .collect();
            Ok(vec![arrow::compute::concat_batches(
                &rows[0].schema(),
                &rows,
            )
            .unwrap()])
        }
    }

    impl UnitSource for Narrow<'_> {
        fn units(&self) -> usize {
            self.base.units.len()
        }
        fn read(&self, unit: usize) -> Result<Vec<RecordBatch>, InterpError> {
            let b = &self.base.units[unit];
            let mut fields = Vec::new();
            let mut cols: Vec<ArrayRef> = Vec::new();
            for c in &self.columns {
                let i = b.schema().index_of(c).unwrap();
                fields.push(b.schema().field(i).clone());
                cols.push(b.column(i).clone());
            }
            fields.push(Field::new(LOCATOR, DataType::UInt64, false));
            cols.push(Arc::new(UInt64Array::from(
                (0..b.num_rows() as u64)
                    .map(|r| ((unit as u64) << 32) | r)
                    .collect::<Vec<_>>(),
            )));
            Ok(vec![RecordBatch::try_new(
                Arc::new(Schema::new(fields)),
                cols,
            )
            .unwrap()])
        }
    }

    fn col(name: &str) -> Expr {
        Expr::Col { name: name.into() }
    }

    fn key(name: &str, descending: bool) -> SortKey {
        SortKey {
            expr: col(name),
            descending,
            nulls_first: true,
        }
    }

    fn top(limit: usize, below: i64) -> RelOp {
        RelOp::Sort {
            input: Box::new(RelOp::Filter {
                input: Box::new(RelOp::Scan { source_id: 0 }),
                predicate: Expr::Binary {
                    op: BinaryOp::Lt,
                    left: Box::new(col("id")),
                    right: Box::new(Expr::Lit {
                        value: Literal::Int(below),
                    }),
                },
            }),
            keys: vec![key("k", false), key("v", true)],
            limit: Some(limit),
        }
    }

    fn strings(batches: &[RecordBatch]) -> Vec<String> {
        let mut out = Vec::new();
        for b in batches {
            let shown =
                arrow::util::pretty::pretty_format_batches(std::slice::from_ref(b)).unwrap();
            out.extend(shown.to_string().lines().skip(3).map(str::to_string));
        }
        out.retain(|l| !l.starts_with('+'));
        out
    }

    /// The late top-N returns the oracle's rows in the oracle's order — through ties the sort
    /// breaks by input order, under a select list and a limit above it, at several widths —
    /// and it really did narrow and fetch only the winners.
    #[test]
    fn a_late_top_n_returns_the_oracles_rows_in_order() {
        let units: Vec<RecordBatch> = (0..24).map(|u| wide(u * 500, (u + 1) * 500)).collect();
        let whole = arrow::compute::concat_batches(&units[0].schema(), &units).unwrap();
        let sources = vec![vec![wide(0, 0)]];
        let selected = RelOp::Project {
            input: Box::new(top(7, 9_000)),
            exprs: ["w", "id", "k"]
                .iter()
                .map(|c| ProjectionItem {
                    expr: col(c),
                    alias: (*c).into(),
                })
                .collect(),
        };
        let limited = RelOp::Limit {
            input: Box::new(top(9, 12_000)),
            n: 4,
            offset: 3,
        };
        for (plan, winners) in [(top(10, 11_000), 10), (selected, 7), (limited, 9)] {
            let want = strings(&crate::execute(&plan, &[vec![whole.clone()]]).unwrap());
            for workers in [1, 3, 8] {
                let src = Located {
                    units: units.clone(),
                    narrowed: AtomicUsize::new(0),
                    fetched: AtomicUsize::new(0),
                };
                let got = execute_units(
                    &plan,
                    &sources,
                    0,
                    &src,
                    workers,
                    0,
                    &ExecOptions::default(),
                )
                .unwrap();
                assert_eq!(strings(&got), want, "workers {workers}");
                assert_eq!(src.narrowed.load(Ordering::Relaxed), 1);
                assert_eq!(src.fetched.load(Ordering::Relaxed), winners);
            }
        }
    }

    /// A filter that keeps nothing fetches nothing and still returns the plan's empty result,
    /// and a limit too large to pay is served by the ordinary path without narrowing.
    #[test]
    fn an_empty_or_too_large_top_n_takes_the_safe_path() {
        let units: Vec<RecordBatch> = (0..8).map(|u| wide(u * 100, (u + 1) * 100)).collect();
        let whole = arrow::compute::concat_batches(&units[0].schema(), &units).unwrap();
        let sources = vec![vec![wide(0, 0)]];
        for (plan, narrows) in [(top(3, 0), 1), (top(5, 800), 0)] {
            let src = Located {
                units: units.clone(),
                narrowed: AtomicUsize::new(0),
                fetched: AtomicUsize::new(0),
            };
            let got =
                execute_units(&plan, &sources, 0, &src, 4, 0, &ExecOptions::default()).unwrap();
            let want = crate::execute(&plan, &[vec![whole.clone()]]).unwrap();
            assert_eq!(strings(&got), strings(&want));
            assert_eq!(src.narrowed.load(Ordering::Relaxed), narrows);
            assert_eq!(src.fetched.load(Ordering::Relaxed), 0);
        }
    }
}
