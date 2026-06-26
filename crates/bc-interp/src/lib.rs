//! `bc-interp` — the Tier-0 interpreter.
//!
//! Two executors share one set of operator primitives (`ops`):
//!
//! * [`execute`] — the **sequential reference**. A straightforward, deterministic
//!   walk of the IR; the correctness oracle the parallel path and (later) the JIT
//!   are checked against.
//! * [`par::execute_parallel`] — the **multi-core executor**. Same operator
//!   semantics, but it morselizes the input and runs filters/projections,
//!   partial→combine aggregation, and hash-partitioned joins across a rayon
//!   thread pool. The hash-shuffle it uses to parallelize joins is the very same
//!   mechanism the distributed layer uses across actors.
//!
//! Both are flying starts: execution begins immediately, so JIT compilation is
//! never on the critical path.

use arrow::array::RecordBatch;
use bc_ir::RelOp;

pub mod dist;
mod error;
mod join_par;
pub mod metrics;
mod ops;
pub mod par;
mod window_spill;

pub use error::InterpError;
pub use metrics::{ExecMetrics, OpMetric};
pub use par::{
    execute_parallel, execute_parallel_with, execute_parallel_with_metrics, ExecOptions,
};

use metrics::{IdGen, Stopwatch};

/// Total rows across a set of morsels.
pub(crate) fn count_rows(batches: &[RecordBatch]) -> u64 {
    batches.iter().map(|b| b.num_rows() as u64).sum()
}

/// Total Arrow buffer bytes across a set of morsels (coarse working-set proxy).
pub(crate) fn batch_bytes(batches: &[RecordBatch]) -> u64 {
    batches
        .iter()
        .map(|b| b.get_array_memory_size() as u64)
        .sum()
}

/// Execute a plan sequentially (the reference executor).
///
/// `sources[i]` is the relation referenced by `Scan { source_id: i }`.
pub fn execute(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
) -> Result<Vec<RecordBatch>, InterpError> {
    let (out, _metrics) = execute_metered(plan, sources)?;
    Ok(out)
}

/// Execute sequentially and also return per-operator [`ExecMetrics`]. The result
/// batches are identical to [`execute`]; the metrics are a pure side-channel.
pub fn execute_metered(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
) -> Result<(Vec<RecordBatch>, ExecMetrics), InterpError> {
    let mut m = ExecMetrics::default();
    let mut ids = IdGen::new();
    let out = exec_seq(plan, sources, &mut m, &mut ids)?;
    Ok((out, m))
}

fn exec_seq(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    m: &mut ExecMetrics,
    ids: &mut IdGen,
) -> Result<Vec<RecordBatch>, InterpError> {
    // Pre-order id: numbered before recursing into children so parents precede
    // children (matches the Python control plane's `_annotate_ops` numbering).
    let op_id = ids.next();
    match plan {
        RelOp::Scan { source_id } => {
            let t0 = Stopwatch::start();
            let batches = sources
                .get(*source_id)
                .cloned()
                .ok_or(InterpError::UnknownSource {
                    source_id: *source_id,
                    available: sources.len(),
                })?;
            let rows = count_rows(&batches);
            m.record(OpMetric {
                op_id,
                kind: "scan",
                rows_in: rows,
                rows_out: rows,
                elapsed_ns: t0.elapsed_ns(),
                cpu_ns: t0.cpu_ns(),
                threads: 1,
                peak_bytes: batch_bytes(&batches),
                spilled: false,
                backend: "interp",
            });
            Ok(batches)
        }

        RelOp::Filter { input, predicate } => {
            let batches = exec_seq(input, sources, m, ids)?;
            let rows_in = count_rows(&batches);
            let t0 = Stopwatch::start();
            let out: Vec<RecordBatch> = batches
                .iter()
                .map(|b| ops::filter_batch(b, predicate))
                .collect::<Result<_, _>>()?;
            record_op(m, op_id, "filter", rows_in, &out, t0, false);
            Ok(out)
        }

        RelOp::Project { input, exprs } => {
            let batches = exec_seq(input, sources, m, ids)?;
            let rows_in = count_rows(&batches);
            let t0 = Stopwatch::start();
            let out: Vec<RecordBatch> = batches
                .iter()
                .map(|b| ops::project_batch(b, exprs))
                .collect::<Result<_, _>>()?;
            record_op(m, op_id, "project", rows_in, &out, t0, false);
            Ok(out)
        }

        RelOp::Unnest {
            input,
            column,
            alias,
        } => {
            let batches = exec_seq(input, sources, m, ids)?;
            let rows_in = count_rows(&batches);
            let t0 = Stopwatch::start();
            let out: Vec<RecordBatch> = batches
                .iter()
                .map(|b| ops::unnest_batch(b, column, alias))
                .collect::<Result<_, _>>()?;
            record_op(m, op_id, "unnest", rows_in, &out, t0, false);
            Ok(out)
        }

        RelOp::RowId {
            input,
            alias,
            offset,
        } => {
            let batches = exec_seq(input, sources, m, ids)?;
            let rows_in = count_rows(&batches);
            let t0 = Stopwatch::start();
            let out = ops::add_row_ids(&batches, alias, *offset)?;
            record_op(m, op_id, "row_id", rows_in, &out, t0, false);
            Ok(out)
        }

        RelOp::Unpivot {
            input,
            index,
            on,
            variable_name,
            value_name,
        } => {
            let batches = exec_seq(input, sources, m, ids)?;
            let rows_in = count_rows(&batches);
            let t0 = Stopwatch::start();
            let out: Vec<RecordBatch> = batches
                .iter()
                .map(|b| ops::unpivot_batch(b, index, on, variable_name, value_name))
                .collect::<Result<_, _>>()?;
            record_op(m, op_id, "unpivot", rows_in, &out, t0, false);
            Ok(out)
        }

        RelOp::Sample {
            input,
            fraction,
            seed,
            n,
        } => {
            let batches = exec_seq(input, sources, m, ids)?;
            let rows_in = count_rows(&batches);
            let t0 = Stopwatch::start();
            let out: Vec<RecordBatch> = match n {
                // Fixed-count: keep the n smallest-hash rows (a breaker).
                Some(k) => ops::sample_n_batches(&batches, *k, *seed)?,
                None => batches
                    .iter()
                    .map(|b| ops::sample_batch(b, *fraction, *seed))
                    .collect::<Result<_, _>>()?,
            };
            record_op(m, op_id, "sample", rows_in, &out, t0, false);
            Ok(out)
        }

        RelOp::Aggregate {
            input,
            group_keys,
            aggregates,
        } => {
            let batches = exec_seq(input, sources, m, ids)?;
            let rows_in = count_rows(&batches);
            let t0 = Stopwatch::start();
            let combined =
                ops::materialize(&batches).map_err(|_| InterpError::EmptyAggregateInput)?;
            let funcs = ops::agg_funcs(aggregates);
            let partial = ops::eval_partial(&combined, group_keys, aggregates)?;
            let agg_cols = bc_runtime::agg::finalize(&funcs, &partial)?;
            let out = vec![ops::build_agg_batch(
                group_keys,
                aggregates,
                &partial.group_columns,
                &agg_cols,
            )?];
            record_op(m, op_id, "aggregate", rows_in, &out, t0, false);
            Ok(out)
        }

        RelOp::Sort { input, keys, limit } => {
            let batches = exec_seq(input, sources, m, ids)?;
            let rows_in = count_rows(&batches);
            let t0 = Stopwatch::start();
            let out = match ops::materialize(&batches) {
                Ok(combined) => vec![ops::sort_batch(&combined, keys, *limit)?],
                Err(_) => Vec::new(),
            };
            record_op(m, op_id, "sort", rows_in, &out, t0, false);
            Ok(out)
        }

        RelOp::Window {
            input,
            partition_keys,
            order_keys,
            functions,
            rank_limit,
        } => {
            let batches = exec_seq(input, sources, m, ids)?;
            let rows_in = count_rows(&batches);
            let t0 = Stopwatch::start();
            let out = match ops::materialize(&batches) {
                Ok(combined) => vec![ops::window_batch(
                    &combined,
                    partition_keys,
                    order_keys,
                    functions,
                    *rank_limit,
                )?],
                Err(_) => Vec::new(),
            };
            record_op(m, op_id, "window", rows_in, &out, t0, false);
            Ok(out)
        }

        RelOp::Limit { input, n, offset } => {
            let batches = exec_seq(input, sources, m, ids)?;
            let rows_in = count_rows(&batches);
            let t0 = Stopwatch::start();
            let out = ops::limit(batches, *n, *offset);
            record_op(m, op_id, "limit", rows_in, &out, t0, false);
            Ok(out)
        }

        RelOp::HashJoin {
            left,
            right,
            left_keys,
            right_keys,
            join_type,
            output,
            // The sequential reference is the oracle: it always uses the plain hash
            // join regardless of the planner's physical strategy hint (every
            // strategy must produce this exact relation).
            strategy: _,
        } => {
            let left_batches = exec_seq(left, sources, m, ids)?;
            let right_batches = exec_seq(right, sources, m, ids)?;
            let rows_in = count_rows(&left_batches) + count_rows(&right_batches);
            let t0 = Stopwatch::start();
            let left = ops::materialize(&left_batches)?;
            let right = ops::materialize(&right_batches)?;
            // The sequential reference is the oracle: always the plain hash join,
            // regardless of the planner's physical strategy (which other tiers honor).
            let out = vec![ops::join_batches(
                &left,
                &right,
                left_keys,
                right_keys,
                *join_type,
                output,
                bc_ir::JoinStrategy::Hash,
            )?];
            record_op(m, op_id, "hash_join", rows_in, &out, t0, false);
            Ok(out)
        }

        RelOp::AsofJoin {
            left,
            right,
            left_on,
            right_on,
            left_by,
            right_by,
            backward,
            output,
        } => {
            let left_batches = exec_seq(left, sources, m, ids)?;
            let right_batches = exec_seq(right, sources, m, ids)?;
            let rows_in = count_rows(&left_batches) + count_rows(&right_batches);
            let t0 = Stopwatch::start();
            let left = ops::materialize(&left_batches)?;
            let right = ops::materialize(&right_batches)?;
            let out = vec![ops::asof_join_batches(
                &left, &right, left_on, right_on, left_by, right_by, *backward, output,
            )?];
            record_op(m, op_id, "asof_join", rows_in, &out, t0, false);
            Ok(out)
        }

        RelOp::Distinct { input } => {
            let batches = exec_seq(input, sources, m, ids)?;
            let rows_in = count_rows(&batches);
            let t0 = Stopwatch::start();
            let out = vec![distinct(&batches)?];
            record_op(m, op_id, "distinct", rows_in, &out, t0, false);
            Ok(out)
        }

        RelOp::Union {
            inputs,
            distinct: dedup,
        } => {
            let mut all = Vec::new();
            for inp in inputs {
                all.extend(exec_seq(inp, sources, m, ids)?);
            }
            let rows_in = count_rows(&all);
            let t0 = Stopwatch::start();
            let out = if *dedup { vec![distinct(&all)?] } else { all };
            record_op(m, op_id, "union", rows_in, &out, t0, false);
            Ok(out)
        }
    }
}

/// Record one sequential-interpreter operator metric from its result batches.
fn record_op(
    m: &mut ExecMetrics,
    op_id: u32,
    kind: &'static str,
    rows_in: u64,
    out: &[RecordBatch],
    t0: Stopwatch,
    spilled: bool,
) {
    m.record(OpMetric {
        op_id,
        kind,
        rows_in,
        rows_out: count_rows(out),
        elapsed_ns: t0.elapsed_ns(),
        cpu_ns: t0.cpu_ns(),
        threads: 1,
        peak_bytes: batch_bytes(out),
        spilled,
        backend: "interp",
    });
}

fn distinct(batches: &[RecordBatch]) -> Result<RecordBatch, InterpError> {
    let combined = ops::materialize(batches).map_err(|_| InterpError::EmptyAggregateInput)?;
    let partial = ops::distinct_partial(&combined)?;
    Ok(RecordBatch::try_new(
        combined.schema(),
        partial.group_columns,
    )?)
}
