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

use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use bc_ir::RelOp;

mod agg_par;
pub mod dist;
mod error;
mod join_par;
pub mod metrics;
mod ops;
pub mod par;
mod rusage;
mod spill_split;
pub mod stream;
mod window_spill;

pub use error::InterpError;
pub use metrics::{ExecMetrics, OpMetric};
pub use par::{
    execute_parallel, execute_parallel_with, execute_parallel_with_metrics, ExecOptions,
};
pub use stream::{
    execute_streaming, execute_streaming_metered, execute_streaming_parallel,
    execute_streaming_parallel_metered, execute_streaming_parallel_metered_or_hand_off,
    execute_streaming_parallel_or_hand_off, streaming_parallelizes,
};

use metrics::{IdGen, Stopwatch};

/// Total rows across a set of morsels.
pub(crate) fn count_rows(batches: &[RecordBatch]) -> u64 {
    batches.iter().map(|b| b.num_rows() as u64).sum()
}

/// Total Arrow buffer bytes across a set of morsels.
///
/// Uses each column's **slice** size, not `get_array_memory_size()`. The latter reports the
/// whole parent buffer for a sliced array, so morselizing one 32 MB table into 122 morsels
/// made this report 3.9 GB — every morsel re-counting the entire buffer. Carbonite fits its
/// memory model on this figure, so the over-count would have it budget ~100x the real
/// footprint and spill (or reject) plans that fit comfortably.
///
/// A **shared dictionary** is the same bug one level down, and the slice fix does not reach
/// it. Morsels of a dictionary-encoded column share one values array, but each morsel's slice
/// size includes that whole dictionary — so the relation is counted as `morsels x dictionary`
/// instead of `dictionary + indices`. Measured on 600 morsels over a 50,000-entry string
/// dictionary: **609 MB reported for 40 MB resident, a 15x over-count**. Dictionary encoding
/// exists precisely for low-cardinality string columns, so this is the common case for the
/// most common encoding, and the consequence is that the engine spills a sort, join, or
/// window that would have fitted in memory.
///
/// Each distinct dictionary is therefore counted once, identified by its buffer address —
/// unambiguous because every morsel being measured is alive, so two live dictionaries cannot
/// share an address. Relations with no dictionary column skip the bookkeeping entirely.
pub(crate) fn batch_bytes(batches: &[RecordBatch]) -> u64 {
    let gross: u64 = batches
        .iter()
        .flat_map(|b| b.columns().iter())
        .map(|c| c.to_data().get_slice_memory_size().unwrap_or(0) as u64)
        .sum();
    // Fast path: no dictionary anywhere in the schema, so nothing can be double-counted and
    // the walk below would only cost allocations. This is every relation of plain numeric,
    // string, or nested-non-dictionary columns.
    if !batches.first().is_some_and(|b| {
        b.schema()
            .fields()
            .iter()
            .any(|f| has_dictionary(f.data_type()))
    }) {
        return gross;
    }
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let recounted: u64 = batches
        .iter()
        .flat_map(|b| b.columns().iter())
        .map(|c| recounted_dictionary_bytes(&c.to_data(), &mut seen))
        .sum();
    gross.saturating_sub(recounted)
}

/// Whether `dt` contains a dictionary anywhere in its nesting.
fn has_dictionary(dt: &arrow::datatypes::DataType) -> bool {
    use arrow::datatypes::DataType;
    match dt {
        DataType::Dictionary(..) => true,
        DataType::List(f)
        | DataType::LargeList(f)
        | DataType::FixedSizeList(f, _)
        | DataType::Map(f, _) => has_dictionary(f.data_type()),
        DataType::Struct(fs) => fs.iter().any(|f| has_dictionary(f.data_type())),
        DataType::Union(fs, _) => fs.iter().any(|(_, f)| has_dictionary(f.data_type())),
        DataType::RunEndEncoded(_, f) => has_dictionary(f.data_type()),
        _ => false,
    }
}

/// Bytes of `data` that [`batch_bytes`]'s gross sum counted for a dictionary it had already
/// seen — the amount to subtract so a shared dictionary is charged once for the relation.
///
/// On the first sighting of a dictionary, its values are kept in the total and the walk
/// descends *into* them, so a dictionary nested inside a dictionary's values is deduplicated
/// too. On a later sighting the whole values subtree is subtracted, which is exactly what the
/// gross sum added for it.
fn recounted_dictionary_bytes(
    data: &arrow::array::ArrayData,
    seen: &mut std::collections::HashSet<usize>,
) -> u64 {
    use arrow::datatypes::DataType;
    if matches!(data.data_type(), DataType::Dictionary(..)) {
        let Some(values) = data.child_data().first() else {
            return 0;
        };
        // A buffer address identifies the allocation. With no buffers there is nothing to
        // identify it by (and nothing meaningful to save), so treat it as distinct.
        let Some(id) = values.buffers().first().map(|b| b.as_ptr() as usize) else {
            return 0;
        };
        if seen.insert(id) {
            return recounted_dictionary_bytes_of_children(values, seen);
        }
        return values.get_slice_memory_size().unwrap_or(0) as u64;
    }
    recounted_dictionary_bytes_of_children(data, seen)
}

/// [`recounted_dictionary_bytes`] over every child of `data`.
fn recounted_dictionary_bytes_of_children(
    data: &arrow::array::ArrayData,
    seen: &mut std::collections::HashSet<usize>,
) -> u64 {
    data.child_data()
        .iter()
        .map(|c| recounted_dictionary_bytes(c, seen))
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
    // children (matches the Python control plane's `annotate_ops` numbering).
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
            let bytes = batch_bytes(&batches);
            let (cpu_ns, peak_rss_bytes, hw) = t0.measure();
            m.record(OpMetric {
                op_id,
                kind: "scan",
                rows_in: rows,
                rows_build: 0,
                rows_out: rows,
                elapsed_ns: t0.elapsed_ns(),
                wall_span_ns: 0,
                cpu_ns,
                threads: 1,
                peak_bytes: bytes,
                result_bytes: bytes,
                spilled: false,
                spill_bytes: 0,
                peak_rss_bytes,
                backend: "interp",
                hw,
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
            outer,
            index_alias,
        } => {
            let batches = exec_seq(input, sources, m, ids)?;
            let rows_in = count_rows(&batches);
            let t0 = Stopwatch::start();
            let out: Vec<RecordBatch> = batches
                .iter()
                .map(|b| ops::unnest_batch(b, column, alias, *outer, index_alias.as_deref()))
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
            let in_bytes = batch_bytes(&batches);
            let t0 = Stopwatch::start();
            match n {
                // Fixed-count: a global n-smallest-hash pass holds the whole input, so it
                // is a breaker (its peak is that input), not a ~0-peak streaming op.
                Some(k) => {
                    let out = ops::sample_n_batches(&batches, *k, *seed)?;
                    record_breaker(m, op_id, "sample", rows_in, 0, in_bytes, &out, t0, false);
                    Ok(out)
                }
                // Fractional: streaming per-batch; peak is the sampled result alone.
                None => {
                    let out: Vec<RecordBatch> = batches
                        .iter()
                        .map(|b| ops::sample_batch(b, *fraction, *seed))
                        .collect::<Result<_, _>>()?;
                    record_op(m, op_id, "sample", rows_in, &out, t0, false);
                    Ok(out)
                }
            }
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
            record_breaker(
                m,
                op_id,
                "aggregate",
                rows_in,
                0,
                batch_bytes(&batches),
                &out,
                t0,
                false,
            );
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
            record_breaker(
                m,
                op_id,
                "sort",
                rows_in,
                0,
                batch_bytes(&batches),
                &out,
                t0,
                false,
            );
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
            record_breaker(
                m,
                op_id,
                "window",
                rows_in,
                0,
                batch_bytes(&batches),
                &out,
                t0,
                false,
            );
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
            // The probe side (left) drives the per-row probe cost; the build side (right)
            // drives the hash table's memory. Reporting their sum made both meaningless.
            let rows_in = count_rows(&left_batches);
            let rows_build = count_rows(&right_batches);
            let in_bytes = batch_bytes(&left_batches) + batch_bytes(&right_batches);
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
            record_breaker(
                m,
                op_id,
                "hash_join",
                rows_in,
                rows_build,
                in_bytes,
                &out,
                t0,
                false,
            );
            Ok(out)
        }

        RelOp::RangeJoin {
            left,
            right,
            conditions,
            join_type,
            output,
        } => {
            let left_batches = exec_seq(left, sources, m, ids)?;
            let right_batches = exec_seq(right, sources, m, ids)?;
            let rows_in = count_rows(&left_batches);
            let rows_build = count_rows(&right_batches);
            let in_bytes = batch_bytes(&left_batches) + batch_bytes(&right_batches);
            let t0 = Stopwatch::start();
            let left = ops::materialize(&left_batches)?;
            let right = ops::materialize(&right_batches)?;
            let out = vec![ops::range_join_batches(
                &left, &right, conditions, *join_type, output,
            )?];
            record_breaker(
                m,
                op_id,
                "range_join",
                rows_in,
                rows_build,
                in_bytes,
                &out,
                t0,
                false,
            );
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
            // The probe side (left) drives the per-row probe cost; the build side (right)
            // drives the hash table's memory. Reporting their sum made both meaningless.
            let rows_in = count_rows(&left_batches);
            let rows_build = count_rows(&right_batches);
            let in_bytes = batch_bytes(&left_batches) + batch_bytes(&right_batches);
            let t0 = Stopwatch::start();
            let left = ops::materialize(&left_batches)?;
            let right = ops::materialize(&right_batches)?;
            let out = vec![ops::asof_join_batches(
                &left, &right, left_on, right_on, left_by, right_by, *backward, output,
            )?];
            record_breaker(
                m,
                op_id,
                "asof_join",
                rows_in,
                rows_build,
                in_bytes,
                &out,
                t0,
                false,
            );
            Ok(out)
        }

        RelOp::Distinct { input } => {
            let batches = exec_seq(input, sources, m, ids)?;
            let rows_in = count_rows(&batches);
            let t0 = Stopwatch::start();
            let out = vec![distinct(&batches)?];
            record_breaker(
                m,
                op_id,
                "distinct",
                rows_in,
                0,
                batch_bytes(&batches),
                &out,
                t0,
                false,
            );
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
            // Promotable-but-different branch types (`int64 ∪ float64`) are coerced to the
            // union's advertised supertype before concat/dedup, matching DuckDB.
            let all = coerce_union_branches(all)?;
            let rows_in = count_rows(&all);
            let in_bytes = batch_bytes(&all);
            let t0 = Stopwatch::start();
            if *dedup {
                // A deduplicating UNION materializes and hashes its input — a breaker whose
                // peak is that input plus the deduped result.
                let out = vec![distinct(&all)?];
                record_breaker(m, op_id, "union", rows_in, 0, in_bytes, &out, t0, false);
                Ok(out)
            } else {
                // UNION ALL streams: it holds only the concatenated result.
                record_op(m, op_id, "union", rows_in, &all, t0, false);
                Ok(all)
            }
        }
    }
}

/// Record one sequential-interpreter operator metric from its result batches.
/// Record a **streaming** operator: it holds only its result, so peak == result bytes.
fn record_op(
    m: &mut ExecMetrics,
    op_id: u32,
    kind: &'static str,
    rows_in: u64,
    out: &[RecordBatch],
    t0: Stopwatch,
    spilled: bool,
) {
    let bytes = batch_bytes(out);
    let (cpu_ns, peak_rss_bytes, hw) = t0.measure();
    m.record(OpMetric {
        op_id,
        kind,
        rows_in,
        rows_build: 0,
        rows_out: count_rows(out),
        elapsed_ns: t0.elapsed_ns(),
        wall_span_ns: 0,
        cpu_ns,
        threads: 1,
        peak_bytes: bytes,
        result_bytes: bytes,
        spilled,
        spill_bytes: 0,
        peak_rss_bytes,
        backend: "interp",
        hw,
    });
}

/// Record a **pipeline breaker**: it materializes `in_bytes` of input and builds its
/// result at the same time, so both are live at its peak. `rows_build` is the join's
/// build-side rows (0 elsewhere). The sequential oracle does not spill, so `spill_bytes`
/// is always 0 here (the field carries a magnitude only on the parallel/spilling paths).
#[allow(clippy::too_many_arguments)]
fn record_breaker(
    m: &mut ExecMetrics,
    op_id: u32,
    kind: &'static str,
    rows_in: u64,
    rows_build: u64,
    in_bytes: u64,
    out: &[RecordBatch],
    t0: Stopwatch,
    spilled: bool,
) {
    let result_bytes = batch_bytes(out);
    let (cpu_ns, peak_rss_bytes, hw) = t0.measure();
    m.record(OpMetric {
        op_id,
        kind,
        rows_in,
        rows_build,
        rows_out: count_rows(out),
        elapsed_ns: t0.elapsed_ns(),
        wall_span_ns: 0,
        cpu_ns,
        threads: 1,
        peak_bytes: in_bytes.saturating_add(result_bytes),
        result_bytes,
        spilled,
        spill_bytes: 0,
        peak_rss_bytes,
        backend: "interp",
        hw,
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

/// Coerce the branches of a set operation (UNION / INTERSECT / EXCEPT — all of which lower
/// to `RelOp::Union`) to one common column type before they are concatenated / deduped.
///
/// A set op's branches may carry promotable-but-different numeric types — `int64 ∪
/// float64` is the canonical case — and the union's advertised output schema is already
/// the promoted supertype (`promote(int64, float64) = float64`, per the Python type
/// lattice). Arrow's `concat`/`materialize`, however, reject a type mismatch outright, so
/// the branches must first be cast up to the supertype here. DuckDB likewise coerces both
/// sides to DOUBLE and returns a result; without this an ordinary `A UNION B` errored even
/// though `Dataset.schema` promised the promoted type.
///
/// A no-op that returns the input untouched when every branch already shares a column type
/// (the overwhelmingly common single-type union), so it pays only one scan of the schemas.
pub(crate) fn coerce_union_branches(
    batches: Vec<RecordBatch>,
) -> Result<Vec<RecordBatch>, InterpError> {
    let Some(first) = batches.first() else {
        return Ok(batches);
    };
    let ncols = first.num_columns();
    // Fold the per-column supertype across every branch's schema.
    let mut target: Vec<DataType> = first
        .schema()
        .fields()
        .iter()
        .map(|f| f.data_type().clone())
        .collect();
    let mut mismatch = false;
    for b in batches.iter().skip(1) {
        for (c, t) in target.iter_mut().enumerate().take(ncols) {
            let bt = b.column(c).data_type();
            if bt != t {
                // Only coerce when the two branch types have a SAFE common supertype
                // (a numeric widening). For a genuinely incompatible pair — e.g. int64
                // vs string — there is none, so fail fast with a typed error rather than
                // arrow's *lenient* string→int64 cast silently nulling the non-numeric
                // values (data corruption), or a downstream downcast panic on the
                // still-mismatched schema.
                match promote_union_type(t, bt) {
                    Some(common) => {
                        *t = common;
                        mismatch = true;
                    }
                    None => {
                        return Err(InterpError::IncompatibleSetOpTypes {
                            col: c,
                            left: t.to_string(),
                            right: bt.to_string(),
                        })
                    }
                }
            }
        }
    }
    if !mismatch {
        return Ok(batches);
    }
    // Rebuild a schema carrying the promoted types (a column is nullable if any branch's is).
    let base = first.schema();
    let fields: Vec<Field> = (0..ncols)
        .map(|c| {
            let nullable = batches.iter().any(|b| b.schema().field(c).is_nullable());
            base.field(c)
                .clone()
                .with_data_type(target[c].clone())
                .with_nullable(nullable)
        })
        .collect();
    let schema = Arc::new(Schema::new_with_metadata(fields, base.metadata().clone()));
    batches
        .into_iter()
        .map(|b| {
            let cols: Vec<ArrayRef> = (0..ncols)
                .map(|c| {
                    if b.column(c).data_type() == &target[c] {
                        Ok(Arc::clone(b.column(c)))
                    } else {
                        Ok(arrow::compute::cast(b.column(c), &target[c])?)
                    }
                })
                .collect::<Result<_, InterpError>>()?;
            Ok(RecordBatch::try_new(Arc::clone(&schema), cols)?)
        })
        .collect()
}

/// The common type two set-operation branch columns must both widen to, so neither side is
/// narrowed — or `None` when there is no safe numeric supertype. Mirrors the Python type
/// lattice's `promote`: a float on either side wins (→ `Float64`, as DuckDB promotes int∪float
/// to DOUBLE), otherwise two integers meet at `Int64`. Any non-numeric pairing (e.g. int64 vs
/// string) returns `None`, so the caller declines to coerce and the mismatch surfaces as a
/// clean error — never a lossy string→int cast that silently nulls the incompatible branch.
fn promote_union_type(a: &DataType, b: &DataType) -> Option<DataType> {
    use DataType::*;
    let is_float = |t: &DataType| matches!(t, Float16 | Float32 | Float64);
    let is_int = |t: &DataType| {
        matches!(
            t,
            Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64
        )
    };
    let numeric = |t: &DataType| is_float(t) || is_int(t);
    if numeric(a) && numeric(b) {
        // A float on either side wins (→ Float64, as DuckDB promotes int∪float to
        // DOUBLE); otherwise two integers meet at Int64. No branch is ever narrowed.
        if is_float(a) || is_float(b) {
            Some(Float64)
        } else {
            Some(Int64)
        }
    } else {
        // No safe common numeric supertype (e.g. int64 vs string). Return None so the
        // caller declines to coerce and the mismatch surfaces as a clean error rather
        // than a lossy cast — never silently null out the incompatible branch.
        None
    }
}

#[cfg(test)]
mod union_coerce_tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};

    fn batch(name: &str, ty: DataType, col: ArrayRef) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(name, ty, true)]));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    /// An `int64 ∪ float64` union must coerce both branches to Float64 so `concat`/`distinct`
    /// accept them — matching DuckDB (which promotes to DOUBLE) and the union's own advertised
    /// schema, instead of erroring on the type mismatch.
    #[test]
    fn int64_and_float64_branches_coerce_to_double() {
        let left = batch(
            "x",
            DataType::Int64,
            Arc::new(Int64Array::from(vec![1i64, 2])),
        );
        let right = batch(
            "x",
            DataType::Float64,
            Arc::new(Float64Array::from(vec![3.5f64, 4.5])),
        );
        let out = coerce_union_branches(vec![left, right]).unwrap();
        assert_eq!(out.len(), 2);
        for b in &out {
            assert_eq!(b.column(0).data_type(), &DataType::Float64);
        }
        // The int branch's values survive the widening exactly.
        let l = out[0]
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(l.value(0), 1.0);
        assert_eq!(l.value(1), 2.0);
        // Concatenation of the coerced branches now succeeds (it would error pre-coercion).
        let cols: Vec<&dyn arrow::array::Array> =
            out.iter().map(|b| b.column(0).as_ref()).collect();
        let cat = arrow::compute::concat(&cols).unwrap();
        assert_eq!(cat.len(), 4);
    }

    /// A same-type union is returned untouched (identity), paying only the schema scan.
    #[test]
    fn matching_types_are_untouched() {
        let a = batch("x", DataType::Int64, Arc::new(Int64Array::from(vec![1i64])));
        let b = batch("x", DataType::Int64, Arc::new(Int64Array::from(vec![2i64])));
        let out = coerce_union_branches(vec![a, b]).unwrap();
        assert_eq!(out[0].column(0).data_type(), &DataType::Int64);
        assert_eq!(out[1].column(0).data_type(), &DataType::Int64);
    }
}

/// How big a relation *is* — the figure every spill decision is made from.
///
/// [`batch_bytes`] is what Carbonite fits its memory model on, and what the sort, join,
/// window, and grace fan-outs size themselves against. An over-count there does not fail
/// anything: it makes the engine spill a query that would have fitted in memory, which looks
/// like slowness rather than like a bug — so it needs its own assertions.
#[cfg(test)]
mod relation_sizing_tests {
    use super::batch_bytes;
    use arrow::array::{Array, DictionaryArray, Int32Array, RecordBatch, StringArray};
    use arrow::datatypes::Int32Type;
    use std::sync::Arc;

    const ENTRIES: i32 = 20_000;
    const MORSELS: usize = 200;
    const ROWS: usize = 4_096;

    fn dictionary(prefix: &str) -> Arc<dyn Array> {
        let vals: Vec<String> = (0..ENTRIES).map(|i| format!("{prefix}-{i:06}")).collect();
        Arc::new(StringArray::from(
            vals.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))
    }

    fn keys(rows: usize) -> Int32Array {
        Int32Array::from((0..rows as i32).map(|i| i % ENTRIES).collect::<Vec<_>>())
    }

    fn dict_morsel(values: &Arc<dyn Array>, rows: usize) -> RecordBatch {
        let d = DictionaryArray::<Int32Type>::try_new(keys(rows), values.clone()).expect("dict");
        RecordBatch::try_from_iter(vec![("k", Arc::new(d) as Arc<dyn Array>)]).expect("batch")
    }

    fn size_of(a: &Arc<dyn Array>) -> u64 {
        a.to_data().get_slice_memory_size().unwrap_or(0) as u64
    }

    /// A shared dictionary must be charged once for the relation, not once per morsel.
    ///
    /// Morsels of a dictionary-encoded column all point at the same values array, but each
    /// morsel's slice size includes that whole dictionary. Measured before this fix, on 600
    /// morsels over a 50,000-entry string dictionary: **609 MB reported for 40 MB resident, a
    /// 15x over-count**. Dictionary encoding exists precisely for low-cardinality string
    /// columns, so this is the common case for the most common encoding — and the consequence
    /// is a sort, join, or window spilling when it would have fitted.
    #[test]
    fn a_shared_dictionary_is_counted_once_for_the_relation() {
        let values = dictionary("category");
        let batches: Vec<RecordBatch> = (0..MORSELS).map(|_| dict_morsel(&values, ROWS)).collect();

        let dict_bytes = size_of(&values);
        let index_bytes = (MORSELS * ROWS * 4) as u64;
        let measured = batch_bytes(&batches);

        // The honest figure is one dictionary plus the indices. The old sum was
        // `MORSELS * dict_bytes` on top of that.
        assert!(
            measured <= dict_bytes + index_bytes,
            "measured {measured} for a relation of {} resident bytes — the shared dictionary \
             is still being counted per morsel ({MORSELS} x {dict_bytes})",
            dict_bytes + index_bytes
        );
        assert!(
            measured >= index_bytes,
            "measured {measured} but the indices alone are {index_bytes} — the deduplication \
             subtracted more than the dictionary"
        );
        // And the over-count it replaces was an order of magnitude, not a rounding difference.
        let naive: u64 = batches
            .iter()
            .flat_map(|b| b.columns().iter())
            .map(|c| c.to_data().get_slice_memory_size().unwrap_or(0) as u64)
            .sum();
        assert!(
            naive > measured * 5,
            "the naive sum ({naive}) should be many times the deduplicated one ({measured}); \
             if it is not, this test is no longer measuring the shape it was written for"
        );
    }

    /// Two *distinct* dictionaries must both be counted. Deduplicating by buffer address is
    /// sound only because every morsel being measured is alive, so two live dictionaries
    /// cannot share an address — this is the assertion that the dedup does not collapse them.
    #[test]
    fn two_distinct_dictionaries_are_both_counted() {
        let a = dictionary("category");
        let b = dictionary("other");
        let shared_twice = batch_bytes(&[dict_morsel(&a, ROWS), dict_morsel(&a, ROWS)]);
        let two_distinct = batch_bytes(&[dict_morsel(&a, ROWS), dict_morsel(&b, ROWS)]);
        // `b`'s size, not `a`'s: the two prefixes differ in length, so the dictionaries do
        // too, and the second one is what the extra bytes should account for.
        assert!(
            two_distinct >= shared_twice + size_of(&b),
            "two distinct dictionaries measured {two_distinct}, barely more than the \
             {shared_twice} of one shared twice — the dedup is collapsing dictionaries it \
             should count"
        );
    }

    /// A relation with no dictionary must measure exactly what it did before, and take the
    /// fast path rather than paying for bookkeeping that cannot apply.
    #[test]
    fn a_relation_without_dictionaries_is_unchanged() {
        let batches: Vec<RecordBatch> = (0..MORSELS)
            .map(|_| {
                RecordBatch::try_from_iter(vec![("k", Arc::new(keys(ROWS)) as Arc<dyn Array>)])
                    .expect("batch")
            })
            .collect();
        let naive: u64 = batches
            .iter()
            .flat_map(|b| b.columns().iter())
            .map(|c| c.to_data().get_slice_memory_size().unwrap_or(0) as u64)
            .sum();
        assert_eq!(batch_bytes(&batches), naive);
    }

    /// A dictionary *nested* inside a struct is deduplicated too — the walk is over the whole
    /// array tree, not just top-level columns, because a multimodal or semi-structured schema
    /// is where dictionary columns most often sit one level down.
    #[test]
    fn a_dictionary_nested_in_a_struct_is_deduplicated() {
        use arrow::array::StructArray;
        use arrow::datatypes::{DataType, Field};

        let values = dictionary("category");
        let mk = || {
            let d =
                DictionaryArray::<Int32Type>::try_new(keys(ROWS), values.clone()).expect("dict");
            let field = Field::new(
                "k",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                false,
            );
            let s = StructArray::from(vec![(Arc::new(field), Arc::new(d) as Arc<dyn Array>)]);
            RecordBatch::try_from_iter(vec![("s", Arc::new(s) as Arc<dyn Array>)]).expect("batch")
        };
        let batches: Vec<RecordBatch> = (0..MORSELS).map(|_| mk()).collect();

        let measured = batch_bytes(&batches);
        let naive: u64 = batches
            .iter()
            .flat_map(|b| b.columns().iter())
            .map(|c| c.to_data().get_slice_memory_size().unwrap_or(0) as u64)
            .sum();
        assert!(
            naive > measured * 5,
            "a struct-nested dictionary was not deduplicated: naive {naive}, measured {measured}"
        );
    }
}
