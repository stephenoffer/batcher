//! Tier-0 **streaming** executor: pull morsels through the linear runs, materialize only at
//! breakers.
//!
//! `execute`/`par::exec` are tree-walks that return the full `Vec<RecordBatch>` of **every**
//! node — breaker or not. A `scan → filter → project → join → join` chain therefore holds the
//! scan's output, the filter's output, the project's output, and every join's output in RAM at
//! once. That is the single biggest structural gap to DuckDB, and it is not a tuning problem:
//! at TPC-H sf100 the deep join trees (q3/q4/q5) peak at **133 GB and are OOM-killed**, while
//! DuckDB streams the same queries in a few GB. Projection pushdown already works there, so
//! this is intermediate blow-up, not a wide scan.
//!
//! This module implements the pipeline/breaker model `docs/internals/execution.md` has
//! documented all along: *"A pipeline is a maximal chain of operators that can run a batch
//! straight through without materializing. A breaker is an operator that has to collect its
//! input."*
//!
//! * **Pipeline operators** — `Scan`, `Filter`, `Project`, `Unnest`, `Unpivot`, `RowId`,
//!   `Limit`, and a hash join's **probe** side — are lazy adapters over their child's stream.
//!   They transform one morsel and yield it. A linear run's peak memory is *one morsel per
//!   stage*, not the whole relation.
//! * **Breakers** — `Aggregate`, `Sort`, `Distinct`, `Window`, `Sample`, `AsofJoin`, `Union`,
//!   and a hash join's **build** side — collect, because their semantics require it. They stay
//!   breakers *on purpose*: they are the points where the adaptive layer measures actual
//!   cardinalities and re-plans (CLAUDE.md invariant #10, the moat). Streaming the linear runs
//!   *between* them is the whole point; the breakers are not the enemy, the incidental
//!   materialization of everything else was.
//!
//! Two operators get more than a scheduling change, because their kernels already supported
//! better and only the driver was in the way:
//!
//! * **The hash-join probe streams** (`bc_runtime::join::BroadcastProbe`): the build side is
//!   hashed once, then each probe morsel is probed and gathered on its own. This is what
//!   actually kills the q3/q4/q5 OOM — a left-deep chain of joins now threads one morsel
//!   through every probe, holding the (small) build tables and nothing else, instead of
//!   materializing each join's full output to feed the next.
//! * **The aggregate folds incrementally** (`partial` → `combine`): its state is bounded by the
//!   *group count*, not the input size. This is the mergeable algebra (invariant #7) applied to
//!   the one place that was still reading its whole input into RAM first.
//!
//! **Identical to the oracle.** This is a new *scheduling* of the same operator semantics —
//! exactly as `par` is to `execute`. It calls the same `ops::` and `bc-runtime` kernels, and it
//! is required to produce the same rows in the same order as [`crate::execute`]. Two facts make
//! that sound rather than hopeful: a morsel is a contiguous, in-order row range, so probing
//! morsels in order emits what slicing the concatenated relation would; and `partial`/`combine`
//! is associative by construction, so folding per morsel finalizes to what one big `partial`
//! would. `stream_matches_the_sequential_oracle` pins it over every operator.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::RecordBatch;
use bc_ir::{JoinType, RelOp};
use bc_runtime::agg;
use bc_runtime::join::{streaming_supported, BroadcastProbe};

use crate::ops;
use crate::InterpError;

mod breaker;
mod meter;
mod parallel;
mod pipeline;

pub use parallel::{execute_streaming_parallel, execute_streaming_parallel_metered};

use breaker::{drain, exec_breaker};
pub(crate) use meter::Meter;
use pipeline::{limit_stream, scan_stream};

/// Everything a stream stage needs, in one `Copy` handle so an iterator closure can capture it
/// by value rather than borrowing a struct that has to outlive the stream.
///
/// `sources` differs per worker on the parallel path (each sees its own shard); `cache` and
/// `meter` are shared by all of them.
#[derive(Clone, Copy)]
pub(crate) struct Ctx<'a> {
    sources: &'a [Vec<RecordBatch>],
    cache: &'a BuildCache,
    /// `None` when the caller did not ask for metrics — the counters are not free (an atomic add
    /// and a clock read per morsel), and a query nobody is measuring should not pay for them.
    meter: Option<&'a Meter>,
    /// The memory envelope a breaker's state must stay inside, or `0` for "unbounded".
    ///
    /// The streaming executor's breakers do **not** spill — the out-of-core paths (`agg::spill`,
    /// the external sort, the grace join) live in the materializing executor. Rather than OOM
    /// where that executor would have spilled, a breaker that finds its state over budget stops
    /// and returns [`InterpError::MemoryBudgetExceeded`], and the caller re-runs the query on the
    /// executor that can spill. Streaming keeps the fast, bounded-intermediate path for the
    /// queries it fits, and gives way on the ones it does not — instead of quietly turning a
    /// spill into a crash.
    budget: usize,
}

impl<'a> Ctx<'a> {
    pub(crate) fn new(
        sources: &'a [Vec<RecordBatch>],
        cache: &'a BuildCache,
        meter: Option<&'a Meter>,
        budget: usize,
    ) -> Self {
        Self {
            sources,
            cache,
            meter,
            budget,
        }
    }

    /// Stop if `needed` bytes of breaker state would exceed the envelope.
    fn check_budget(&self, needed: u64, reason: &'static str) -> Result<(), InterpError> {
        if self.budget > 0 && needed as usize > self.budget {
            return Err(InterpError::MemoryBudgetExceeded {
                needed: needed as usize,
                budget: self.budget,
                reason,
            });
        }
        Ok(())
    }

    /// This node's pre-order `op_id`, when metrics are being collected.
    fn id(&self, plan: &RelOp) -> Option<u32> {
        self.meter.map(|m| m.id(plan))
    }

    /// Record one morsel through a pipeline operator.
    fn morsel(&self, id: Option<u32>, rows_in: u64, out: &RecordBatch, t: std::time::Instant) {
        if let (Some(m), Some(id)) = (self.meter, id) {
            m.morsel(id, rows_in, out, t.elapsed().as_nanos() as u64);
        }
    }
}

/// A hash join's build side, prepared once and shared by every worker that probes it.
///
/// Rebuilding this per worker would be `workers x` the build cost, and on a chain of joins that
/// is the dominant term — the thing that would make a "parallel" streaming executor slower than
/// the materializing one it replaces.
pub(crate) struct JoinBuild {
    /// The materialized build relation (small by construction — it is the broadcast side).
    side: RecordBatch,
    /// The hash table over it, or `None` when this join's shape cannot be probed per morsel and
    /// the materialized fallback must be used.
    probe: Option<BroadcastProbe>,
}

/// Prepared build sides, keyed by the identity of their `HashJoin` node.
///
/// The key is the node's address. The plan is borrowed for the whole execution and never moves,
/// so the address is a stable identity — and it distinguishes two structurally identical joins in
/// the same plan, which a structural key would conflate.
pub(crate) type BuildCache = HashMap<usize, Arc<JoinBuild>>;

/// Execute (and hash) every hash-join build side in `plan`, once.
///
/// The build sides are themselves run on the streaming path, so preparing them never materializes
/// their subtrees either.
pub(crate) fn prebuild_joins(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    meter: Option<&Meter>,
    budget: usize,
) -> Result<Arc<BuildCache>, InterpError> {
    let mut cache = BuildCache::new();
    collect_builds(plan, sources, &mut cache, meter, budget)?;
    Ok(Arc::new(cache))
}

fn collect_builds(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    cache: &mut BuildCache,
    meter: Option<&Meter>,
    budget: usize,
) -> Result<(), InterpError> {
    // Children first. A join's build side may itself contain joins, and draining it below runs
    // the streaming path over that subtree — which consults this cache. Post-order guarantees
    // the inner joins are prepared before an outer one asks to probe through them.
    for child in plan.children() {
        collect_builds(child, sources, cache, meter, budget)?;
    }
    if let RelOp::HashJoin {
        right,
        right_keys,
        join_type,
        ..
    } = plan
    {
        let ctx = Ctx::new(sources, cache, meter, budget);
        let batches = drain(build_with(right, ctx)?)?;
        if let Ok(side) = ops::materialize(&batches) {
            let probe = make_probe(&side, right_keys, *join_type)?;
            cache.insert(node_key(plan), Arc::new(JoinBuild { side, probe }));
        }
    }
    Ok(())
}

/// Identity of a plan node — its address in the (borrowed, immobile) plan tree.
fn node_key(plan: &RelOp) -> usize {
    plan as *const RelOp as usize
}

/// The per-morsel probe table over `side`, or `None` when this join's shape cannot be served
/// per morsel (`Right`/`Full`, or a non-integer key) and the materialized path must take over.
fn make_probe(
    side: &RecordBatch,
    right_keys: &[String],
    join_type: JoinType,
) -> Result<Option<BroadcastProbe>, InterpError> {
    let build_keys = ops::columns_by_name(side, right_keys)?;
    let key_types: Vec<&arrow::datatypes::DataType> =
        build_keys.iter().map(|k| k.data_type()).collect();
    let rt = ops::map_join_type(join_type);
    if !streaming_supported(rt, &key_types, side.num_rows()) {
        return Ok(None);
    }
    let tuning = bc_arrow::RuntimeTuning::default();
    // `probe_rows` only decides whether the probe-side bloom pays for itself, and the bloom is a
    // pure short-circuit with no false negatives — the emitted rows are identical either way. A
    // streamed probe is by definition the large side, and its exact row count is not knowable
    // without materializing it, which is the thing this executor exists to avoid.
    Ok(BroadcastProbe::new(
        &build_keys,
        rt,
        usize::MAX,
        tuning.bloom_fp_rate,
        tuning.bloom_min_build_rows,
    ))
}

/// A lazily-produced stream of morsels. `Box<dyn Iterator>` rather than a bespoke trait: a
/// pipeline operator *is* an iterator adapter, and Rust's iterators already give the pull-based
/// composition (`map`, `chain`, short-circuit) this executor is made of — for free, and lazily.
pub(crate) type Morsels<'a> = Box<dyn Iterator<Item = Result<RecordBatch, InterpError>> + 'a>;

/// How many partials the aggregate lets pile up before folding them together.
///
/// The fold has to be bounded or the "streaming" aggregate quietly re-materializes its input as
/// a heap of per-morsel partials. Combining on *every* morsel would instead re-hash the whole
/// running state once per morsel. Batching the fold keeps state at `O(groups)` while paying the
/// combine only every `N` morsels.
const AGG_FOLD_EVERY: usize = 32;

/// Execute `plan` by streaming morsels through its linear runs.
///
/// Returns the same rows, in the same order, as [`crate::execute`] — the sequential oracle.
/// The *final* result is collected (a caller asked for a relation), but nothing *between*
/// operators is, which is the whole point: peak memory is the breakers' state plus one morsel
/// in flight, not the sum of every operator's output.
pub fn execute_streaming(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    budget: usize,
) -> Result<Vec<RecordBatch>, InterpError> {
    let cache = prebuild_joins(plan, sources, None, budget)?;
    let ctx = Ctx::new(sources, &cache, None, budget);
    let out: Vec<RecordBatch> = build_with(plan, ctx)?.collect::<Result<_, _>>()?;
    Ok(strip_empties(out))
}

/// [`execute_streaming`], with per-operator metrics. Results are identical; the metrics are a
/// side-channel.
pub fn execute_streaming_metered(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    budget: usize,
) -> Result<(Vec<RecordBatch>, crate::ExecMetrics), InterpError> {
    let m = Meter::new(plan, 1);
    let cache = prebuild_joins(plan, sources, Some(&m), budget)?;
    let ctx = Ctx::new(sources, &cache, Some(&m), budget);
    let out: Vec<RecordBatch> = build_with(plan, ctx)?.collect::<Result<_, _>>()?;
    Ok((strip_empties(out), m.finish()))
}

/// Drop the zero-row morsels a filter naturally produces, but keep one if that is *all* there
/// is — a downstream caller (and the oracle) still needs the schema over an empty relation.
pub(crate) fn strip_empties(batches: Vec<RecordBatch>) -> Vec<RecordBatch> {
    if batches.iter().any(|b| b.num_rows() > 0) {
        return batches.into_iter().filter(|b| b.num_rows() > 0).collect();
    }
    batches.into_iter().take(1).collect()
}

/// Compose the stream for `plan`. Pipeline operators wrap their child lazily; breakers drain it.
pub(crate) fn build_with<'a>(plan: &'a RelOp, ctx: Ctx<'a>) -> Result<Morsels<'a>, InterpError> {
    let id = ctx.id(plan);
    match plan {
        RelOp::Scan { source_id } => {
            let batches = ctx
                .sources
                .get(*source_id)
                .ok_or(InterpError::UnknownSource {
                    source_id: *source_id,
                    available: ctx.sources.len(),
                })?;
            Ok(Box::new(scan_stream(batches).map(move |b| {
                let t = std::time::Instant::now();
                let b = b?;
                // A scan's rows in and out are the same rows — it produces them.
                ctx.morsel(id, b.num_rows() as u64, &b, t);
                Ok(b)
            })))
        }

        // ---- linear pipeline operators: one morsel in, one morsel out --------------------
        RelOp::Filter { input, predicate } => {
            let child = build_with(input, ctx)?;
            Ok(Box::new(child.map(move |b| {
                let b = b?;
                let rows_in = b.num_rows() as u64;
                let t = std::time::Instant::now();
                let out = ops::filter_batch(&b, predicate)?;
                ctx.morsel(id, rows_in, &out, t);
                Ok(out)
            })))
        }

        RelOp::Project { input, exprs } => {
            let child = build_with(input, ctx)?;
            Ok(Box::new(child.map(move |b| {
                let b = b?;
                let rows_in = b.num_rows() as u64;
                let t = std::time::Instant::now();
                let out = ops::project_batch(&b, exprs)?;
                ctx.morsel(id, rows_in, &out, t);
                Ok(out)
            })))
        }

        RelOp::Unnest {
            input,
            column,
            alias,
        } => {
            let child = build_with(input, ctx)?;
            Ok(Box::new(child.map(move |b| {
                let b = b?;
                let rows_in = b.num_rows() as u64;
                let t = std::time::Instant::now();
                let out = ops::unnest_batch(&b, column, alias)?;
                ctx.morsel(id, rows_in, &out, t);
                Ok(out)
            })))
        }

        RelOp::Unpivot {
            input,
            index,
            on,
            variable_name,
            value_name,
        } => {
            let child = build_with(input, ctx)?;
            Ok(Box::new(child.map(move |b| {
                let b = b?;
                let rows_in = b.num_rows() as u64;
                let t = std::time::Instant::now();
                let out = ops::unpivot_batch(&b, index, on, variable_name, value_name)?;
                ctx.morsel(id, rows_in, &out, t);
                Ok(out)
            })))
        }

        RelOp::RowId {
            input,
            alias,
            offset,
        } => {
            let child = build_with(input, ctx)?;
            // Row ids are a running sequence over the *relation*, so the counter carries across
            // morsels; the kernel is the same one, handed one morsel at a time.
            let mut seen: i64 = 0;
            Ok(Box::new(child.map(move |b| {
                let b = b?;
                let rows_in = b.num_rows() as i64;
                let t = std::time::Instant::now();
                let out = ops::materialize(&ops::add_row_ids(&[b], alias, *offset + seen)?)?;
                seen += rows_in;
                ctx.morsel(id, rows_in as u64, &out, t);
                Ok(out)
            })))
        }

        RelOp::Limit { input, n, offset } => {
            let child = build_with(input, ctx)?;
            // The one operator whose streaming form changes *complexity*, not just memory: it
            // stops pulling once it has `n` rows, so `LIMIT 10` over a billion-row scan reads ten
            // rows' worth of morsels instead of the billion the materializing path read before
            // throwing them away.
            let limited = limit_stream(child, *n, *offset);
            Ok(Box::new(limited.map(move |b| {
                let b = b?;
                let t = std::time::Instant::now();
                ctx.morsel(id, b.num_rows() as u64, &b, t);
                Ok(b)
            })))
        }

        // ---- the hash join: build once, stream the probe --------------------------------
        RelOp::HashJoin {
            left,
            left_keys,
            right_keys,
            join_type,
            output,
            strategy,
            ..
        } => build_join(
            plan, left, left_keys, right_keys, *join_type, output, *strategy, ctx,
        ),

        // ---- breakers: collect, because the semantics require it -------------------------
        _ => {
            let out = exec_breaker(plan, ctx)?;
            Ok(Box::new(out.into_iter().map(Ok)))
        }
    }
}

/// A hash join: probe the (already-hashed, shared) build side with the probe side's morsels.
///
/// The build table comes from [`prebuild_joins`], so it is hashed exactly once no matter how many
/// workers probe it. Falling back is safe and silent — a join type or key shape `BroadcastProbe`
/// cannot serve per morsel keeps the materialized path, which is what the oracle does anyway.
#[allow(clippy::too_many_arguments)]
fn build_join<'a>(
    plan: &'a RelOp,
    left: &'a RelOp,
    left_keys: &'a [String],
    right_keys: &'a [String],
    join_type: JoinType,
    output: &'a [bc_ir::JoinOutputCol],
    strategy: bc_ir::JoinStrategy,
    ctx: Ctx<'a>,
) -> Result<Morsels<'a>, InterpError> {
    let id = ctx.id(plan);
    let Some(prepared) = ctx.cache.get(&node_key(plan)) else {
        // No prepared build side: the build relation had no batches at all (not even a schema).
        // The oracle owns what a join against nothing yields.
        let probe = build_with(left, ctx)?;
        return materialized_join_from(
            probe,
            left_keys,
            right_keys,
            join_type,
            output,
            strategy,
            &[],
        );
    };
    let build_rows = prepared.side.num_rows() as u64;

    let Some(table) = prepared.probe.as_ref() else {
        // `Right`/`Full`, or a key shape the per-morsel probe cannot serve.
        let probe = build_with(left, ctx)?;
        return materialized_join_from(
            probe,
            left_keys,
            right_keys,
            join_type,
            output,
            strategy,
            std::slice::from_ref(&prepared.side),
        );
    };

    // Peek one morsel to settle the probe's key shape and the output schema. Every morsel of a
    // relation shares a schema, so one look answers both for all of them — and, crucially, it
    // lets a shape the table cannot serve **fall back** rather than error. A query `execute`
    // answers must never be one `execute_streaming` refuses; that would be a divergence, not an
    // optimization. The peeked morsel is pushed back onto the front of the stream, so nothing is
    // consumed twice or lost.
    let mut probe = build_with(left, ctx)?;
    let Some(first) = probe.next().transpose()? else {
        return materialized_join_from(
            Box::new(std::iter::empty()),
            left_keys,
            right_keys,
            join_type,
            output,
            strategy,
            std::slice::from_ref(&prepared.side),
        );
    };

    let first_keys = ops::columns_by_name(&first, left_keys)?;
    if !table.accepts(&first_keys) {
        let rest: Morsels<'a> = Box::new(std::iter::once(Ok(first)).chain(probe));
        return materialized_join_from(
            rest,
            left_keys,
            right_keys,
            join_type,
            output,
            strategy,
            std::slice::from_ref(&prepared.side),
        );
    }

    let out_schema = ops::join_output_schema(&first, &prepared.side, output)?;
    let stream = std::iter::once(Ok(first)).chain(probe);
    let side = &prepared.side;

    Ok(Box::new(stream.map(move |morsel| {
        let morsel = morsel?;
        let rows_in = morsel.num_rows() as u64;
        let t = std::time::Instant::now();
        let probe_keys = ops::columns_by_name(&morsel, left_keys)?;
        // `accepts` vetted this relation's key shape above, so the probe is infallible from
        // here — the `expect` documents an invariant, not a hope.
        let idx = table
            .probe(&probe_keys)
            .expect("accepts() vetted this relation's key shape");
        let out =
            ops::gather_join_output_with(&morsel, side, &idx, output, Arc::clone(&out_schema))?;
        if let (Some(m), Some(id)) = (ctx.meter, id) {
            // `rows_in` is the *probe* side only, and `rows_build` the build side — the split the
            // metric contract requires, so a join's fan-out (`rows_out / rows_in`) means what
            // Kyber thinks it means. The build rows are recorded once, on the first morsel, not
            // once per morsel: the table was built once.
            m.morsel(id, rows_in, &out, t.elapsed().as_nanos() as u64);
            m.record_build_rows_once(id, build_rows);
        }
        Ok(out)
    })))
}

/// The oracle's join: materialize the probe side and join once. Used whenever the per-morsel
/// probe cannot serve the shape — a fallback, never a divergence.
#[allow(clippy::too_many_arguments)]
fn materialized_join_from<'a>(
    probe: Morsels<'a>,
    left_keys: &'a [String],
    right_keys: &'a [String],
    join_type: JoinType,
    output: &'a [bc_ir::JoinOutputCol],
    strategy: bc_ir::JoinStrategy,
    build_batches: &[RecordBatch],
) -> Result<Morsels<'a>, InterpError> {
    let probe_batches = drain(probe)?;
    let (Ok(probe_side), Ok(build_side)) = (
        ops::materialize(&probe_batches),
        ops::materialize(build_batches),
    ) else {
        // A side with no batches at all (not even a schema) — the oracle yields nothing.
        return Ok(Box::new(std::iter::empty()));
    };
    let out = ops::join_batches(
        &probe_side,
        &build_side,
        left_keys,
        right_keys,
        join_type,
        output,
        strategy,
    )?;
    Ok(Box::new(std::iter::once(Ok(out))))
}

/// Fold an aggregate's input **incrementally** into one `Partial` — `partial` each morsel,
/// `combine` them — without ever holding the input.
///
/// `None` means the stream held no rows at all. That is not the same as "the aggregate is
/// empty": a global `COUNT` over nothing is still one row containing `0`. Only the caller knows
/// whether it is looking at a whole relation (defer to the oracle for that row) or at one shard
/// of a parallel run (contribute nothing and let the other shards speak).
///
/// **Why a `Partial` and not a finished aggregate.** A finalized aggregate cannot be merged: two
/// shards' `mean`s do not average to the relation's `mean`, and two `count`s do not compose with
/// two `sum`s once the shape is lost. `partial`/`combine`/`finalize` is the mergeable algebra
/// (invariant #7) — the *same* fold the distributed path runs across nodes — and it is
/// associative, so folding morsel by morsel, or shard by shard, finalizes to exactly what one
/// `partial` over the concatenated input would.
pub(crate) fn fold_partial(
    input: Morsels<'_>,
    group_keys: &[bc_ir::ProjectionItem],
    aggregates: &[bc_ir::AggregateItem],
) -> Result<(Option<agg::Partial>, u64), InterpError> {
    let funcs = ops::agg_funcs(aggregates);
    let mut partials: Vec<agg::Partial> = Vec::new();
    let mut folded: Option<agg::Partial> = None;
    let mut rows_in: u64 = 0;

    for morsel in input {
        let morsel = morsel?;
        if morsel.num_rows() == 0 {
            continue;
        }
        rows_in += morsel.num_rows() as u64;
        partials.push(ops::eval_partial(&morsel, group_keys, aggregates)?);
        // Bounded: without this the "streaming" aggregate quietly re-materializes its input as a
        // heap of per-morsel partials. Combining on *every* morsel would instead re-hash the
        // whole running state once per morsel; batching the fold keeps state at O(groups).
        if partials.len() >= AGG_FOLD_EVERY {
            if let Some(prev) = folded.take() {
                partials.push(prev);
            }
            folded = Some(agg::combine(&partials, &funcs)?);
            partials.clear();
        }
    }

    if let Some(prev) = folded.take() {
        partials.push(prev);
    }
    if partials.is_empty() {
        return Ok((None, rows_in));
    }
    Ok((Some(agg::combine(&partials, &funcs)?), rows_in))
}

/// `finalize` one (already combined) partial into the aggregate's output batch.
pub(crate) fn finalize_partial(
    merged: &agg::Partial,
    group_keys: &[bc_ir::ProjectionItem],
    aggregates: &[bc_ir::AggregateItem],
) -> Result<Vec<RecordBatch>, InterpError> {
    let funcs = ops::agg_funcs(aggregates);
    let agg_cols = agg::finalize(&funcs, merged)?;
    Ok(vec![ops::build_agg_batch(
        group_keys,
        aggregates,
        &merged.group_columns,
        &agg_cols,
    )?])
}

/// Combine shard-level partials into one, then finalize — the parallel aggregate's tail.
pub(crate) fn combine_and_finalize(
    partials: &[agg::Partial],
    group_keys: &[bc_ir::ProjectionItem],
    aggregates: &[bc_ir::AggregateItem],
) -> Result<Vec<RecordBatch>, InterpError> {
    let funcs = ops::agg_funcs(aggregates);
    let merged = agg::combine(partials, &funcs)?;
    finalize_partial(&merged, group_keys, aggregates)
}
