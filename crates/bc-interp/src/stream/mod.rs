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

use std::sync::Arc;

use arrow::array::RecordBatch;
use bc_ir::{JoinType, RelOp};
use bc_runtime::agg;

use crate::ops;
use crate::InterpError;

mod breaker;
mod builds;
mod meter;
mod parallel;
mod pipeline;
mod probe_chunks;
mod runtime_filter;

pub use parallel::{
    execute_streaming_parallel, execute_streaming_parallel_metered,
    execute_streaming_parallel_metered_or_hand_off, execute_streaming_parallel_or_hand_off,
    streaming_parallelizes,
};

use breaker::{drain, exec_breaker};
pub(crate) use builds::{node_key, prebuild_joins, BuildCache, MatCache};
pub(crate) use meter::Meter;
pub(crate) use pipeline::limit_stream;
use pipeline::scan_stream;
use probe_chunks::{PendingProbe, ProbeSlicer};

/// Everything a stream stage needs, in one `Copy` handle so an iterator closure can capture it
/// by value rather than borrowing a struct that has to outlive the stream.
///
/// `sources` differs per worker on the parallel path (each sees its own shard); `cache` and
/// `meter` are shared by all of them.
#[derive(Clone, Copy)]
pub(crate) struct Ctx<'a> {
    sources: &'a [Vec<RecordBatch>],
    cache: &'a BuildCache,
    /// Workers available to a stage that is allowed to fan out. **1 inside a sharded worker**:
    /// its pipeline is already one of `workers` running in a rayon loop, and spawning there would
    /// nest rayon and duplicate work. Only the un-sharded (`fallback`) path passes the real count,
    /// which is exactly where a stage can still be serial and would like not to be.
    workers: usize,
    /// Spine breakers already evaluated, in parallel, over the **unsharded** sources.
    ///
    /// `None` on the sequential path, which materializes its breakers inline. See [`MatCache`].
    mats: Option<&'a MatCache>,
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
        Self::with_workers(sources, cache, meter, budget, 1)
    }

    /// [`Ctx::new`], for a caller that is *not* itself inside a rayon worker and so may fan out.
    pub(crate) fn with_workers(
        sources: &'a [Vec<RecordBatch>],
        cache: &'a BuildCache,
        meter: Option<&'a Meter>,
        budget: usize,
        workers: usize,
    ) -> Self {
        Self {
            sources,
            cache,
            mats: None,
            meter,
            budget,
            workers,
        }
    }

    /// This context, reading the caller's already-materialized spine breakers.
    ///
    /// Threaded rather than rebuilt: a worker must see the *same* materialized relation every
    /// other worker sees, because it was computed once from the unsharded sources and is the very
    /// thing that licenses sharding the spine above it.
    pub(crate) fn with_mats(mut self, mats: Option<&'a MatCache>) -> Self {
        self.mats = mats;
        self
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
    let cache = prebuild_joins(plan, sources, None, budget, 1)?;
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
    let cache = prebuild_joins(plan, sources, Some(&m), budget, 1)?;
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

/// Compose the stream for `plan`, then apply any runtime join filter placed on its output.
///
/// The filter is attached *outside* [`build_node`] so it lands on the node's finished morsels,
/// after that node's own metrics are recorded — the operator did the work the meter says it
/// did, and what the filter removes is charged to the consumer, not hidden from the producer.
pub(crate) fn build_with<'a>(plan: &'a RelOp, ctx: Ctx<'a>) -> Result<Morsels<'a>, InterpError> {
    let stream = build_node(plan, ctx)?;
    match ctx.cache.filters_for(node_key(plan)) {
        None => Ok(stream),
        Some(filters) => Ok(Box::new(
            stream.map(move |b| runtime_filter::apply(filters, b?)),
        )),
    }
}

/// Compose the stream for `plan`. Pipeline operators wrap their child lazily; breakers drain it.
fn build_node<'a>(plan: &'a RelOp, ctx: Ctx<'a>) -> Result<Morsels<'a>, InterpError> {
    // An already-materialized spine breaker is a leaf: yield what it produced. Checked before the
    // match, so it wins over every arm below — including the breaker arm that would otherwise
    // re-execute this subtree once per worker, and (crucially) over any arm that would execute it
    // against this worker's *shard*. Metrics are not re-recorded: the run that filled the cache
    // was metered, and counting it again in every worker would inflate the very cardinalities
    // Kyber learns from.
    if let Some(batches) = ctx.mats.and_then(|m| m.get(&node_key(plan))) {
        let batches = Arc::clone(batches);
        let n = batches.len();
        return Ok(Box::new((0..n).map(move |i| Ok(batches[i].clone()))));
    }

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
        // Filter and Project stay on the interpreter here, and that is a measured choice rather
        // than an oversight: wiring the Tier-1 JIT into this path (compile once per operator on
        // the first morsel, reuse across the rest) was tried and measured 1.01x over TPC-H in an
        // interleaved A/B, with five queries slower. Arrow's compare/boolean kernels are already
        // SIMD, so a scalar Cranelift loop has nothing to win on these predicates, and the real
        // cost on this path is in the joins and aggregates rather than the scalar expressions.
        // `par.rs` still compiles, which is where the fused-pipeline shapes make it pay.
        RelOp::Filter { input, predicate } => {
            let child = build_with(input, ctx)?;
            // Per-operator conjunct order, built once and captured by the per-morsel
            // closure. This path is the engine default and never carries a JIT (see the
            // note above), so it is the one that most wants a measured order rather than a
            // static-cost guess. Result-invariant: the conjuncts of an `AND` commute.
            let order = bc_expr::ConjunctOrder::new(predicate);
            Ok(Box::new(child.map(move |b| {
                let b = b?;
                let rows_in = b.num_rows() as u64;
                let t = std::time::Instant::now();
                let out = ops::filter_batch_jit(&b, predicate, &None, order.as_ref())?;
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
            outer,
            index_alias,
        } => {
            let child = build_with(input, ctx)?;
            Ok(Box::new(child.map(move |b| {
                let b = b?;
                let rows_in = b.num_rows() as u64;
                let t = std::time::Instant::now();
                let out = ops::unnest_batch(&b, column, alias, *outer, index_alias.as_deref())?;
                ctx.morsel(id, rows_in, &out, t);
                Ok(out)
            })))
        }

        // A *fractional* sample keeps a row iff a seeded hash of its values falls under the
        // fraction — a per-row predicate, so it is a pipeline operator, not a breaker. The
        // sequential oracle already treats it exactly this way (`lib.rs`: it maps
        // `sample_batch` over each batch independently and records it as streaming, while
        // only the fixed-`n` arm is a breaker), so this is the same kernel on the same
        // per-batch boundary — a scheduling change, not a second semantics.
        //
        // A *fixed-count* sample (`n = Some(k)`) is NOT included: it keeps the k
        // smallest-hash rows of the WHOLE relation, so a per-morsel draw would keep k rows
        // from every morsel — a different sample, not a faster one. It stays deferred.
        RelOp::Sample {
            input,
            fraction,
            seed,
            n: None,
        } => {
            let child = build_with(input, ctx)?;
            let (fraction, seed) = (*fraction, *seed);
            Ok(Box::new(child.map(move |b| {
                let b = b?;
                let rows_in = b.num_rows() as u64;
                let t = std::time::Instant::now();
                let out = ops::sample_batch(&b, fraction, seed)?;
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
        // `Right`/`Full`, a build past the cache-radix cliff, or a key shape the per-morsel probe
        // cannot serve. `spine_is_shardable` refuses to shard through such a join, so reaching
        // here means the *whole plan* took the un-sharded path — which is right (sharding would
        // re-join the entire build in every worker) but leaves the probe side serial. It is the
        // one relation still on one core in TPC-H q4 (`orders SEMI lineitem`: 1.5M orders scanned
        // and filtered to 57k). Run it across the workers instead; we are not inside a rayon loop
        // here, exactly because this join stopped the sharding.
        if ctx.workers > 1 {
            // Both caches go with it. The probe subtree holds joins whose build sides `ctx.cache`
            // already contains (`collect_builds` descends the probe spine) and possibly a spine
            // breaker `ctx.mats` already evaluated; re-entering without them re-executes every one
            // of those, which is invisible in the result and doubles the work.
            let batches = parallel::run_reusing(
                left,
                ctx.sources,
                ctx.workers,
                ctx.meter,
                ctx.budget,
                Some(ctx.cache),
                ctx.mats,
                None,
            )?;
            return materialized_join_from(
                Box::new(batches.into_iter().map(Ok)),
                left_keys,
                right_keys,
                join_type,
                output,
                strategy,
                std::slice::from_ref(&prepared.side),
            );
        }
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

    // One probe morsel does NOT imply one output morsel: a join *multiplies* rows, and a
    // high-fan-out one multiplies them a lot. Against a build side holding `f` rows per key, a
    // 16,384-row probe morsel yields `16,384 x f` output rows, and emitting that as a single
    // `RecordBatch` is what broke the streaming executor's constant-memory property — measured at
    // 13.1 GB for a cartesian join over two 20,000-row tables. It is not a cartesian-only
    // problem: an ordinary equi-join on a skewed key with 100,000 build-side duplicates does the
    // same. So the *output* is morselized, not the input, and the probe emits as many morsels as
    // its fan-out requires. `ops::remorselize`'s doc comment names this exact hazard for
    // unnest/unpivot; a join is the operator that multiplies rows most.
    let mut pending: Option<PendingProbe> = None;
    // The probe morsel currently being consumed, and how far into it we have got. A morsel is
    // probed in slices, not whole, so the *index* buffers stay morsel-scale too — output
    // morselization alone bounds the gathered batch but not the two `u32` arrays behind it.
    let mut current: Option<(RecordBatch, usize)> = None;
    let mut slicer = ProbeSlicer::new();
    let mut source = stream;
    Ok(Box::new(std::iter::from_fn(move || {
        loop {
            // Drain the morsel being emitted before pulling another from the probe side.
            if let Some(p) = pending.as_mut() {
                match p.next_chunk(side, output, &out_schema) {
                    Some(Ok((out, rows_in))) => {
                        if let (Some(m), Some(id)) = (ctx.meter, id) {
                            // `rows_in` is the *probe* side only, and `rows_build` the build side
                            // — the split the metric contract requires, so a join's fan-out
                            // (`rows_out / rows_in`) means what Kyber thinks it means. It is
                            // carried by the FIRST chunk of a morsel and zero thereafter, so
                            // chunking cannot inflate the input count; `rows_out` accumulates
                            // across chunks, which is what it should do. The build rows are
                            // recorded once, not once per morsel: the table was built once.
                            m.morsel(id, rows_in, &out, p.take_elapsed());
                            m.record_build_rows_once(id, build_rows);
                        }
                        return Some(Ok(out));
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => pending = None,
                }
                continue;
            }

            // Pull a fresh morsel only once the previous one has been probed to its end.
            let (morsel, offset) = match current.take() {
                Some(pair) => pair,
                None => match source.next()? {
                    Ok(m) => (m, 0),
                    Err(e) => return Some(Err(e)),
                },
            };
            let remaining = morsel.num_rows() - offset;
            let take = slicer.slice_rows().min(remaining);
            // Slicing costs a `RecordBatch` rebuild (cheap, `Arc`-shared buffers), so skip it
            // when the slice *is* the whole morsel — the steady state for any join whose fan-out
            // is 1, which is most of them.
            let slice = if offset == 0 && take == morsel.num_rows() {
                morsel.clone()
            } else {
                morsel.slice(offset, take)
            };
            if offset + take < morsel.num_rows() {
                current = Some((morsel, offset + take));
            }

            let t = std::time::Instant::now();
            let probe_keys = match ops::columns_by_name(&slice, left_keys) {
                Ok(k) => k,
                Err(e) => return Some(Err(e)),
            };
            // `accepts` vetted this relation's key shape above, so the probe is infallible from
            // here — the `expect` documents an invariant, not a hope.
            let idx = table
                .probe(&probe_keys)
                .expect("accepts() vetted this relation's key shape");
            // What this slice actually fanned out to sizes the next one.
            slicer.observe(take, idx.left.len());
            pending = Some(PendingProbe::new(slice, idx, t.elapsed().as_nanos() as u64));
        }
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
    // `ops::join_batches`, NOT the parallel `par::join_partitioned` — even though this arm has
    // already materialized both sides and running it on one core is a real cost (it is ~55% of
    // TPC-H q4). `join_partitioned` buckets by `rayon::current_num_threads()` where
    // `join_batches`'s radix buckets by `radix_parts(build_rows)`, so it emits the same rows in a
    // **different order** — and this executor's contract is the same rows in the *same order* as
    // `crate::execute` (a `LIMIT` over a semi join would otherwise return different rows on
    // different executors). Swapping it in measured q3 120→34.5 ms and q4 169→120 ms and was
    // reverted for exactly that: `a_semi_join_with_a_huge_build_matches_the_oracle` fails on it.
    // Making this parallel means making it *order-preserving*, not just parallel.
    let out = ops::join_batches(
        &probe_side,
        &build_side,
        left_keys,
        right_keys,
        join_type,
        output,
        strategy,
    )?;
    // Emitted whole, deliberately, even though it can be relation-sized.
    //
    // Splitting it into morsels here looks like the tidy thing to do — everything downstream is
    // written against morsels — and it is a trap. The consumer of this stream is very often
    // *another* join's fallback, or a sort, and both begin by `ops::materialize`-ing what they are
    // given: handed one batch that is a no-op, handed N slices of it that is a full concatenating
    // copy of the whole relation. On a chain of joins each link pays that copy, which is how a
    // TPC-H q5 that streams in a few GB was measured at 99 GB and OOM-killed.
    //
    // The one consumer that genuinely wants morsels — the aggregate, which folds them across the
    // pool — slices this itself, at zero copy, in `fold_partial_parallel`.
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
    jit: &std::sync::OnceLock<ops::AggJit>,
) -> Result<(Option<agg::Partial>, u64), InterpError> {
    let funcs = ops::agg_funcs(aggregates);
    let mut partials: Vec<agg::Partial> = Vec::new();
    let mut folded: Option<agg::Partial> = None;
    let mut rows_in: u64 = 0;
    // Compile the computed group-key and aggregate-input expressions once, from the first
    // morsel that carries rows — the JIT fast path the materializing executor already uses,
    // so arithmetic in aggregate inputs (`SUM(price * (1 - discount) * (1 + tax))`, the whole
    // TPC-H q1 shape) is compiled once and reused across morsels instead of interpreted per
    // row. `eval_jit` is bit-identical to the interpreter on its supported subset and falls
    // back to it otherwise, so this changes throughput only — the streaming-oracle tests pin
    // it against the same interpreter.
    //
    // The `OnceLock` is **shared across the shards** by the caller: `compile_agg` is a pure
    // function of the plan and the schema, and every shard's post-child morsel has the same
    // schema, so one compile serves all of them. Compiling per shard instead paid Cranelift's
    // per-expression cost once per core (~90× on a big box), which measured as a real fraction
    // of a low-cardinality aggregate — this hoists it to exactly one compile per query.

    for morsel in input {
        let morsel = morsel?;
        if morsel.num_rows() == 0 {
            continue;
        }
        rows_in += morsel.num_rows() as u64;
        let jit = jit.get_or_init(|| ops::compile_agg(group_keys, aggregates, &morsel));
        partials.push(ops::eval_partial_jit(&morsel, group_keys, aggregates, jit)?);
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
    // Keep the merge's hash-radix partitions as separate morsels rather than concatenating
    // them into one. They are key-disjoint, so the rows and their order are exactly what one
    // combined `Partial` would finalize to — but the concat is a second full copy of the
    // grouped relation (on a high-cardinality string key, the largest term in the merge), and
    // the next operator gets a batch per partition to fan back out over instead of one.
    let merged = agg::combine_partitioned(partials, &funcs, 0)?;
    let mut out = Vec::with_capacity(merged.len());
    for part in &merged {
        out.extend(finalize_partial(part, group_keys, aggregates)?);
    }
    Ok(out)
}
