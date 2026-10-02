//! Stream the driving relation into the executor chunk by chunk, instead of all at once.
//!
//! Every other entry point takes `sources: &[Vec<RecordBatch>]` — each input already decoded and
//! resident. For a scan-heavy query over a table larger than memory that is the whole problem:
//! TPC-H sf100 `lineitem` projected to q5's four columns is ~19 GB, the query's answer is five
//! rows, and the control plane had to decode every one of those bytes before the engine could
//! start — or, when they would not fit, route the query to an out-of-core path that ran the plan
//! once per 8 MiB morsel from Python. q5 took 3 s on a 240 GB box and 233 s on a 92 GB one.
//!
//! Here one source — the **driving** one — arrives as a sequence of chunks from a caller-supplied
//! producer (`bc-py` reads Parquet a group of files at a time, the next group decoding while this
//! one computes). Everything else is prepared once:
//!
//! 1. The plan must be `post ops → Aggregate → spine`, where the post ops are row-wise, sort or
//!    limit nodes over the aggregate's *result*, and the spine reaches the driving scan through
//!    filters, projections and the **probe** side of joins that emit each probe row
//!    independently (`Inner`, `Left`, `Semi`, `Anti`). The driving source is scanned exactly
//!    once, so no build side reads it. See [`chunkable`].
//! 2. Every build side is prepared once, from the non-driving sources, exactly as
//!    [`super::prebuild_joins`] prepares it for a resident run.
//! 3. Each chunk is sharded across the workers, run through the spine, and folded into partial
//!    aggregate state — the same `partial → combine → finalize` algebra (invariant #7) the
//!    sharded executor and the cluster already use. A probe row lives in exactly one chunk, so
//!    each chunk's partial covers disjoint input and the partials combine to the whole answer.
//! 4. The combined partial is finalized once and the post ops run over that small result.
//!
//! Peak memory is the build sides, the aggregate state and two chunks (the one folding and the
//! one the producer is decoding), independent of the driving relation's size.

mod orient;
mod partial;
mod top_n;

pub use partial::partial_aggregate_units;
pub(crate) mod units;

use arrow::array::RecordBatch;
use bc_ir::RelOp;
use bc_runtime::agg;
use rayon::prelude::*;

use super::parallel::{effective_shard_count, shard};
use super::{build_with, combine_and_finalize, fold_partial, prebuild_joins_for_chunks, Ctx};
use crate::error::InterpError;
use crate::ops;
use crate::par::ExecOptions;
use orient::{orient, orient_counted, original_ids, probe_spine_reaches, scans_of};

/// A chunk producer: the next chunk of the driving relation, `None` once it is exhausted.
pub type NextChunk<'a> = dyn FnMut() -> Option<Result<Vec<RecordBatch>, InterpError>> + 'a;

/// Whether `plan` can be executed with `driving` streamed in chunks. See the module note.
#[must_use]
pub fn chunkable(plan: &RelOp, driving: usize) -> bool {
    oriented_core(&orient(plan, driving), driving).is_some()
}

/// What the chunks feed, below the post ops that run once over its result.
enum Core<'a> {
    /// An aggregate over the driving spine, wherever it sits in the plan (`path` is the child
    /// indices from the root down to it): each chunk folds into partial state, finalized once.
    Aggregate { path: Vec<usize>, node: &'a RelOp },
    /// A streamable spine (an adaptive stage whose root is a join, say): each chunk's output
    /// rows are collected in order.
    Spine { depth: usize, node: &'a RelOp },
}

fn oriented_core(plan: &RelOp, driving: usize) -> Option<Core<'_>> {
    if scans_of(plan, driving) != 1 {
        return None;
    }
    // The aggregate whose input is the probe spine over the driving scan, wherever it sits: the
    // driving source is scanned once, so there is at most one. Everything above it — post ops,
    // an aggregate over it, or a join to another input — reads only its (small) result, and runs
    // on the ordinary executor once it is known.
    //
    // That placement matters beyond memory. TPC-H q15 compares a grouped `sum` with a `max` of
    // the same sums that the control plane evaluates first and folds in as a literal, and in the
    // main query the `sum` sits under a join to `supplier`. If the two evaluations took
    // different executors they would sum in different orders, disagree in the last bit, and the
    // equality would keep nothing; streaming the aggregate wherever it sits sends both through
    // the same chunks, shards and combine.
    let mut path = Vec::new();
    if let Some(node) = find_aggregate(plan, driving, &mut path) {
        return Some(Core::Aggregate { path, node });
    }
    // No aggregate over the spine: the spine starts below any global `Sort`/`Limit`, which must
    // see every chunk's rows at once and so run as post ops over the collected result. A
    // `Project` *above* one of them runs there too — it reads only the post ops' output — which
    // is the shape SQL gives every `SELECT a, b ... ORDER BY c LIMIT n`: the sort is planned
    // under the select list, and stopping at the `Project` left the whole query resident.
    // A `Project` with no `Sort`/`Limit` beneath it stays in the spine, where it streams.
    let mut node = plan;
    let mut depth = 0;
    let (mut spine, mut spine_depth) = (plan, 0);
    loop {
        match node {
            RelOp::Sort { input, .. } | RelOp::Limit { input, .. } => {
                node = input;
                depth += 1;
                (spine, spine_depth) = (node, depth);
            }
            RelOp::Project { input, .. } => {
                node = input;
                depth += 1;
            }
            _ => break,
        }
    }
    probe_spine_reaches(spine, driving).then_some(Core::Spine {
        depth: spine_depth,
        node: spine,
    })
}

/// The aggregate whose input is the probe spine reaching `driving`, recording its child-index
/// path from `plan` into `path`.
fn find_aggregate<'a>(plan: &'a RelOp, driving: usize, path: &mut Vec<usize>) -> Option<&'a RelOp> {
    if let RelOp::Aggregate { input, .. } = plan {
        if probe_spine_reaches(input, driving) {
            return Some(plan);
        }
    }
    for (i, child) in plan.children().into_iter().enumerate() {
        if scans_of(child, driving) == 0 {
            continue;
        }
        path.push(i);
        if let Some(found) = find_aggregate(child, driving, path) {
            return Some(found);
        }
        path.pop();
    }
    None
}

/// Execute `plan` with `sources[driving]` supplied by `next_chunk` rather than resident.
///
/// `sources[driving]` carries only the driving relation's schema — a zero-row batch — and is
/// what the plan runs over when the chunks hold no rows at all. Returns the same rows the
/// resident executors return for the same plan over the concatenated chunks.
///
/// # Errors
/// [`InterpError::NotChunkable`] when the plan is not [`chunkable`] — the caller should run it
/// resident instead; [`InterpError::MemoryBudgetExceeded`] when the aggregate state (or the
/// collected output of a spine) outgrows `budget` — the caller routes the query to an executor
/// that spills; anything a chunk producer or an operator reports.
pub fn execute_chunked(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    driving: usize,
    next_chunk: &mut NextChunk<'_>,
    workers: usize,
    budget: usize,
    opts: &ExecOptions,
) -> Result<Vec<RecordBatch>, InterpError> {
    let oriented = orient(plan, driving);
    let plan = &oriented;
    let Some(core) = oriented_core(plan, driving) else {
        return Err(InterpError::NotChunkable);
    };
    let run = Run {
        driving,
        workers: workers.max(1),
        budget,
        opts,
        carrier: sources[driving].clone(),
        driving_rows: None,
        pool: crate::par::pool_for(workers.max(1))?,
    };
    let mut srcs: Vec<Vec<RecordBatch>> = sources.to_vec();
    srcs[driving] = next_chunk().transpose()?.unwrap_or_default();
    match core {
        Core::Aggregate { path, node } => {
            let result = run.aggregate(node, &mut srcs, next_chunk)?;
            if path.is_empty() {
                return Ok(result);
            }
            // The rest of the plan reads the aggregate's result as a new source.
            let mut rest = plan.clone();
            *node_at(&mut rest, &path) = RelOp::Scan {
                source_id: srcs.len(),
            };
            let mut rest_srcs = srcs;
            rest_srcs[driving] = Vec::new();
            rest_srcs.push(result);
            crate::par::execute_parallel_with(&rest, &rest_srcs, opts)
        }
        Core::Spine { depth, node } => {
            let result = run.collect(node, &mut srcs, next_chunk)?;
            if depth == 0 {
                return Ok(result);
            }
            // The global sort/limit above the spine reads its collected rows as a new source.
            let path = vec![0; depth];
            let mut post = plan.clone();
            *node_at(&mut post, &path) = RelOp::Scan {
                source_id: srcs.len(),
            };
            let mut post_srcs = srcs;
            post_srcs[driving] = Vec::new();
            post_srcs.push(result);
            crate::execute(&post, &post_srcs)
        }
    }
}

/// Execute `plan` with `sources[driving]` read unit by unit from `src` by the workers themselves.
///
/// [`execute_chunked`] alternates: the producer decodes a chunk on every core, then the workers
/// run it on every core. Here there is no producer. The units are split into contiguous ranges,
/// and each worker decodes its range one unit at a time and pushes each unit's morsels straight
/// through its pipeline, so decoding overlaps computing across the pool and a worker holds one
/// decoded unit rather than a chunk. The same plans qualify ([`chunkable`]) and the same rows
/// come back: ranges are contiguous and taken in order, so a spine's output concatenates in the
/// relation's order, and an aggregate folds one partial per range.
///
/// # Errors
/// As [`execute_chunked`].
pub fn execute_units(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    driving: usize,
    src: &dyn units::UnitSource,
    workers: usize,
    budget: usize,
    opts: &ExecOptions,
) -> Result<Vec<RecordBatch>, InterpError> {
    units_with(plan, sources, driving, src, workers, budget, opts, None)
}

/// [`execute_units`], with per-operator metrics for the operators it runs itself.
///
/// The worker-read pass sees every driving row exactly once, so its counts are the query's real
/// cardinalities and the learning loop can take them — unlike [`execute_chunked`], whose
/// operators run once per chunk. The operators it hands to another executor (the post ops over
/// an aggregate's result) are not measured and are simply absent from the metrics. When the
/// plan had to be re-oriented the metrics are empty: the operators are then numbered by a tree
/// the control plane never built, and a count attributed to the wrong operator teaches worse
/// than none.
///
/// # Errors
/// As [`execute_units`].
#[allow(clippy::too_many_arguments)]
pub fn execute_units_metered(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    driving: usize,
    src: &dyn units::UnitSource,
    workers: usize,
    budget: usize,
    opts: &ExecOptions,
) -> Result<(Vec<RecordBatch>, crate::ExecMetrics), InterpError> {
    let (oriented, swaps) = orient_counted(plan, driving);
    let meter = super::Meter::new(&oriented, workers.max(1) as u32);
    let out = units_with(
        &oriented,
        sources,
        driving,
        src,
        workers,
        budget,
        opts,
        Some(&meter),
    )?;
    // The meter numbers the plan it ran, and a swapped join reorders its children in that
    // numbering. The control plane files each metric under the op id of the plan *it* built,
    // so a swapped plan's ids are translated back rather than dropped: dropping them is what
    // left every scan-mode query that joins onto its driving table with no measured
    // cardinality at all, so TPC-H q18 at sf10 planned its 60M-row hash build from defaults
    // on every run and never learned otherwise.
    let metrics = if swaps == 0 {
        meter.finish()
    } else {
        meter.finish_renumbered(&original_ids(plan, &oriented, driving))
    };
    Ok((out, metrics))
}

#[allow(clippy::too_many_arguments)]
fn units_with(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    driving: usize,
    src: &dyn units::UnitSource,
    workers: usize,
    budget: usize,
    opts: &ExecOptions,
    meter: Option<&super::Meter>,
) -> Result<Vec<RecordBatch>, InterpError> {
    let oriented = orient(plan, driving);
    // Measure against the plan the meter numbered: `execute_units_metered` passes an already
    // oriented plan, which orienting again leaves unchanged, so its node addresses are the
    // meter's only if the tree walked is that very plan.
    let plan = if meter.is_some() { plan } else { &oriented };
    let Some(core) = oriented_core(plan, driving) else {
        return Err(InterpError::NotChunkable);
    };
    let run = Run {
        driving,
        workers: workers.max(1),
        budget,
        opts,
        carrier: sources[driving].clone(),
        driving_rows: src.rows(),
        pool: crate::par::pool_for(workers.max(1))?,
    };
    let mut srcs: Vec<Vec<RecordBatch>> = sources.to_vec();
    srcs[driving] = run.carrier.clone();
    let ranges = unit_ranges(src.units(), run.workers);
    match core {
        Core::Aggregate { path, node } => {
            let result = run.aggregate_units(node, &srcs, src, &ranges, meter)?;
            if path.is_empty() {
                return Ok(result);
            }
            let mut rest = plan.clone();
            *node_at(&mut rest, &path) = RelOp::Scan {
                source_id: srcs.len(),
            };
            srcs[driving] = Vec::new();
            srcs.push(result);
            run_post(plan, &rest, &path, &srcs, opts, meter)
        }
        Core::Spine { depth, node } => {
            if let Some(out) = run.top_n_late(plan, depth, node, &srcs, src, &ranges, meter)? {
                return Ok(out);
            }
            let result = run.collect_units(node, &srcs, src, &ranges, meter)?;
            if depth == 0 {
                return Ok(result);
            }
            let mut post = plan.clone();
            *node_at(&mut post, &vec![0; depth]) = RelOp::Scan {
                source_id: srcs.len(),
            };
            srcs[driving] = Vec::new();
            srcs.push(result);
            // The parallel executor, not the sequential oracle: a top-N over a spine that kept
            // a million rows (`SELECT s ... ORDER BY s LIMIT 10`) is the whole query's work, and
            // sorting it on one core made it 3.9x slower than the resident read it replaced.
            run_post(plan, &post, &vec![0; depth], &srcs, opts, meter)
        }
    }
}

/// Run the operators above the streamed core — `post` is `plan` with the subtree at `path`
/// replaced by a `Scan` of the collected result — on the parallel executor, filing each of
/// their measurements under the operator of `plan` it belongs to.
///
/// They were run unmetered, so a `Sort` over a Parquet-driven aggregate executed and was never
/// reported: `stats()` listed the scan, the join and the aggregate of
/// `lineitem JOIN orders GROUP BY o_orderpriority ORDER BY o_orderpriority` and no sort.
fn run_post(
    plan: &RelOp,
    post: &RelOp,
    path: &[usize],
    srcs: &[Vec<RecordBatch>],
    opts: &ExecOptions,
    meter: Option<&super::Meter>,
) -> Result<Vec<RecordBatch>, InterpError> {
    let (out, metrics) = crate::par::execute_parallel_with_metrics(post, srcs, opts)?;
    if let Some(m) = meter {
        m.absorb_mapped(plan, &post_ids(plan, path), &metrics);
    }
    Ok(out)
}

/// For each operator of the post plan (pre-order), the pre-order id in `plan` of the operator it
/// is — `None` for the `Scan` standing in for the subtree at `path`, which is that subtree's
/// result rather than any one of its operators. Every operator after the subtree sits at an id
/// shifted by the subtree's size, which is why the post plan cannot be absorbed by offset.
fn post_ids(plan: &RelOp, path: &[usize]) -> Vec<Option<u32>> {
    fn size(node: &RelOp) -> u32 {
        1 + node.children().into_iter().map(size).sum::<u32>()
    }
    fn walk(node: &RelOp, path: Option<&[usize]>, next: &mut u32, out: &mut Vec<Option<u32>>) {
        if path == Some(&[]) {
            out.push(None);
            *next += size(node);
            return;
        }
        out.push(Some(*next));
        *next += 1;
        for (i, child) in node.children().into_iter().enumerate() {
            let sub = match path {
                Some([head, rest @ ..]) if *head == i => Some(rest),
                _ => None,
            };
            walk(child, sub, next, out);
        }
    }
    let mut out = Vec::new();
    walk(plan, Some(path), &mut 0, &mut out);
    out
}

/// `units` split into contiguous, in-order ranges for `workers` workers.
///
/// Two ranges per worker rather than one: units differ in cost (a row group a predicate
/// empties decodes one column, a full one every column), and with one range each the slowest
/// worker sets the time. Rayon hands the spare ranges to whichever worker finishes first.
fn unit_ranges(units: usize, workers: usize) -> Vec<std::ops::Range<usize>> {
    let pieces = (workers * 2).clamp(1, units.max(1));
    (0..pieces)
        .map(|k| (k * units / pieces)..((k + 1) * units / pieces))
        .filter(|r| !r.is_empty())
        .collect()
}

/// The node reached from `plan` by following the child indices in `path`.
fn node_at<'a>(plan: &'a mut RelOp, path: &[usize]) -> &'a mut RelOp {
    let mut node = plan;
    for &i in path {
        node = node
            .children_mut()
            .into_iter()
            .nth(i)
            .expect("the path was recorded on this plan's shape");
    }
    node
}

/// What [`Run::fold_units`] leaves: the partials, the driving rows they cover, and the compiled
/// aggregate inputs, for the meter.
struct Folded {
    partials: Vec<agg::Partial>,
    rows_in: u64,
    jit: std::sync::OnceLock<ops::AggJit>,
}

/// One chunked execution's settings.
struct Run<'o> {
    driving: usize,
    workers: usize,
    budget: usize,
    opts: &'o ExecOptions,
    /// The zero-row batch carrying the driving relation's schema.
    carrier: Vec<RecordBatch>,
    /// The driving relation's total rows, when the source can say (a Parquet footer can; a
    /// chunk producer cannot). The carrier stands in for the relation everywhere else, so
    /// this is the only place its size is known to the runtime-filter placement.
    driving_rows: Option<usize>,
    /// The producer is not `Send` (in `bc-py` it calls back into Python), so it is only ever
    /// called on the driver thread, between the parallel steps that run inside this pool.
    pool: std::sync::Arc<rayon::ThreadPool>,
}

impl Run<'_> {
    /// The streamed relation's `(source id, rows)` for [`prebuild_joins_for_chunks`].
    fn driving_hint(&self) -> Option<(usize, usize)> {
        self.driving_rows.map(|rows| (self.driving, rows))
    }

    /// One view of `srcs` per shard of the current chunk, or none when the chunk is empty.
    fn shard_views(&self, srcs: &[Vec<RecordBatch>]) -> Vec<Vec<Vec<RecordBatch>>> {
        let rows: usize = srcs[self.driving].iter().map(RecordBatch::num_rows).sum();
        if rows == 0 {
            return Vec::new();
        }
        shard(
            &srcs[self.driving],
            effective_shard_count(self.workers, rows),
        )
        .into_iter()
        .map(|s| {
            let mut view = srcs.to_vec();
            view[self.driving] = s;
            view
        })
        .collect()
    }

    fn over_budget(&self, bytes: usize, reason: &'static str) -> Result<(), InterpError> {
        if self.budget > 0 && bytes > self.budget {
            return Err(InterpError::MemoryBudgetExceeded {
                needed: bytes,
                budget: self.budget,
                reason,
            });
        }
        Ok(())
    }

    /// Fold every chunk into partial aggregate state; finalize once.
    fn aggregate(
        &self,
        node: &RelOp,
        srcs: &mut [Vec<RecordBatch>],
        next_chunk: &mut NextChunk<'_>,
    ) -> Result<Vec<RecordBatch>, InterpError> {
        let RelOp::Aggregate {
            input,
            group_keys,
            aggregates,
        } = node
        else {
            unreachable!("Core::Aggregate holds an aggregate")
        };
        let funcs = ops::agg_funcs(aggregates);
        // The build sides read no driving rows (`chunkable`), so preparing them against the first
        // chunk prepares exactly what a resident run prepares.
        let cache = self.pool.install(|| {
            prebuild_joins_for_chunks(
                input,
                srcs,
                None,
                self.budget,
                self.workers,
                Some(self.opts),
                self.driving_hint(),
            )
        })?;
        let jit = std::sync::OnceLock::new();
        let mut partials: Vec<agg::Partial> = Vec::new();
        let mut state_bytes = 0usize;
        loop {
            self.opts.check_cancelled()?;
            let views = self.shard_views(srcs);
            let folded = self.pool.install(|| {
                views
                    .par_iter()
                    .map(|view| {
                        let ctx = Ctx::new(view, &cache, None, self.budget);
                        Ok(fold_partial(build_with(input, ctx)?, group_keys, aggregates, &jit)?.0)
                    })
                    .collect::<Result<Vec<_>, InterpError>>()
            })?;
            let folded: Vec<agg::Partial> = folded.into_iter().flatten().collect();
            if !folded.is_empty() {
                // One partial per chunk: the shards' states merged while they are small, so what
                // is held across chunks is bounded by the groups, not by chunks times workers.
                let merged = self.pool.install(|| agg::combine(&folded, &funcs))?;
                state_bytes += crate::column_bytes(
                    merged
                        .group_columns
                        .iter()
                        .chain(merged.states.iter().flatten()),
                ) as usize;
                self.over_budget(state_bytes, "the chunked aggregate does not spill")?;
                partials.push(merged);
            }
            match next_chunk().transpose()? {
                Some(chunk) => srcs[self.driving] = chunk,
                None => break,
            }
        }
        if partials.is_empty() {
            // No chunk held a row. The aggregate over nothing is the oracle's to answer — a
            // global aggregate still owes its one identity row — so run it over the empty
            // relation.
            let mut empty = srcs.to_vec();
            empty[self.driving] = self.carrier.clone();
            return crate::execute(node, &empty);
        }
        self.pool
            .install(|| combine_and_finalize(&partials, group_keys, aggregates))
    }

    /// Run every chunk through the spine and keep its rows, in chunk and shard order.
    fn collect(
        &self,
        node: &RelOp,
        srcs: &mut [Vec<RecordBatch>],
        next_chunk: &mut NextChunk<'_>,
    ) -> Result<Vec<RecordBatch>, InterpError> {
        let cache = self.pool.install(|| {
            prebuild_joins_for_chunks(
                node,
                srcs,
                None,
                self.budget,
                self.workers,
                Some(self.opts),
                self.driving_hint(),
            )
        })?;
        let mut out: Vec<RecordBatch> = Vec::new();
        let mut held = 0usize;
        loop {
            self.opts.check_cancelled()?;
            let views = self.shard_views(srcs);
            let pieces = self.pool.install(|| {
                views
                    .par_iter()
                    .map(|view| {
                        let ctx = Ctx::new(view, &cache, None, self.budget);
                        build_with(node, ctx)?.collect::<Result<Vec<_>, InterpError>>()
                    })
                    .collect::<Result<Vec<_>, InterpError>>()
            })?;
            for batch in pieces.into_iter().flatten() {
                if batch.num_rows() > 0 {
                    held += crate::column_bytes(batch.columns()) as usize;
                    out.push(batch);
                }
            }
            self.over_budget(held, "the chunked spine's output is held in memory")?;
            match next_chunk().transpose()? {
                Some(chunk) => srcs[self.driving] = chunk,
                None => break,
            }
        }
        if out.is_empty() {
            // Nothing survived: the oracle over the empty relation gives the schema.
            let mut empty = srcs.to_vec();
            empty[self.driving] = self.carrier.clone();
            return crate::execute(node, &empty);
        }
        Ok(out)
    }
}

impl Run<'_> {
    /// One lazily-read driving scan per unit range.
    fn lazies<'s>(
        &self,
        src: &'s dyn units::UnitSource,
        ranges: &[std::ops::Range<usize>],
    ) -> Vec<units::LazyScan<'s>> {
        ranges
            .iter()
            .map(|units| units::LazyScan {
                source_id: self.driving,
                src,
                units: units.clone(),
            })
            .collect()
    }

    /// Fold each unit range into partial aggregate state, on the worker that reads it.
    fn aggregate_units(
        &self,
        node: &RelOp,
        srcs: &[Vec<RecordBatch>],
        src: &dyn units::UnitSource,
        ranges: &[std::ops::Range<usize>],
        meter: Option<&super::Meter>,
    ) -> Result<Vec<RecordBatch>, InterpError> {
        let t = std::time::Instant::now();
        let RelOp::Aggregate {
            group_keys,
            aggregates,
            ..
        } = node
        else {
            unreachable!("Core::Aggregate holds an aggregate")
        };
        let folded = self.fold_units(node, srcs, src, ranges, meter)?;
        let partials = folded.partials;
        if partials.is_empty() {
            return crate::execute(node, srcs);
        }
        let state = crate::column_bytes(
            partials
                .iter()
                .flat_map(|p| p.group_columns.iter().chain(p.states.iter().flatten())),
        ) as usize;
        self.over_budget(state, "the chunked aggregate does not spill")?;
        let out = self
            .pool
            .install(|| combine_and_finalize(&partials, group_keys, aggregates))?;
        if let Some(m) = meter {
            if let Some(compiled) = folded.jit.get() {
                m.note_backend(m.id(node), compiled.backend_tag());
            }
            m.breaker(
                m.id(node),
                folded.rows_in,
                0,
                state as u64,
                &out,
                t.elapsed().as_nanos() as u64,
            );
        }
        Ok(out)
    }

    /// Fold each unit range through `node`'s input into one partial per non-empty range, on the
    /// worker that reads the range. `meter` numbers the input's operators; `node` itself is not
    /// looked up in it, so a caller may meter the input alone.
    fn fold_units(
        &self,
        node: &RelOp,
        srcs: &[Vec<RecordBatch>],
        src: &dyn units::UnitSource,
        ranges: &[std::ops::Range<usize>],
        meter: Option<&super::Meter>,
    ) -> Result<Folded, InterpError> {
        let RelOp::Aggregate {
            input,
            group_keys,
            aggregates,
        } = node
        else {
            unreachable!("Core::Aggregate holds an aggregate")
        };
        // The build sides read no driving rows (`chunkable`), so the carrier suffices.
        let cache = self.pool.install(|| {
            prebuild_joins_for_chunks(
                input,
                srcs,
                meter,
                self.budget,
                self.workers,
                Some(self.opts),
                self.driving_hint(),
            )
        })?;
        self.opts.check_cancelled()?;
        let lazies = self.lazies(src, ranges);
        let jit = std::sync::OnceLock::new();
        let rows_in = std::sync::atomic::AtomicU64::new(0);
        let partials: Vec<agg::Partial> = self
            .pool
            .install(|| {
                lazies
                    .par_iter()
                    .map(|lazy| {
                        let ctx = Ctx::new(srcs, &cache, meter, self.budget).with_lazy(lazy);
                        let (partial, n) =
                            fold_partial(build_with(input, ctx)?, group_keys, aggregates, &jit)?;
                        rows_in.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
                        Ok(partial)
                    })
                    .collect::<Result<Vec<_>, InterpError>>()
            })?
            .into_iter()
            .flatten()
            .collect();
        Ok(Folded {
            partials,
            rows_in: rows_in.into_inner(),
            jit,
        })
    }

    /// Run each unit range through the spine on the worker that reads it; keep the rows in order.
    fn collect_units(
        &self,
        node: &RelOp,
        srcs: &[Vec<RecordBatch>],
        src: &dyn units::UnitSource,
        ranges: &[std::ops::Range<usize>],
        meter: Option<&super::Meter>,
    ) -> Result<Vec<RecordBatch>, InterpError> {
        let cache = self.pool.install(|| {
            prebuild_joins_for_chunks(
                node,
                srcs,
                meter,
                self.budget,
                self.workers,
                Some(self.opts),
                self.driving_hint(),
            )
        })?;
        self.opts.check_cancelled()?;
        let held = std::sync::atomic::AtomicUsize::new(0);
        let lazies = self.lazies(src, ranges);
        let pieces = self.pool.install(|| {
            lazies
                .par_iter()
                .map(|lazy| {
                    let ctx = Ctx::new(srcs, &cache, meter, self.budget).with_lazy(lazy);
                    let keep = |batch: Result<RecordBatch, InterpError>| {
                        let batch = batch?;
                        let bytes = crate::column_bytes(batch.columns()) as usize;
                        let total =
                            held.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed) + bytes;
                        self.over_budget(total, "the chunked spine's output is held in memory")?;
                        Ok(batch)
                    };
                    let out: Vec<RecordBatch> = build_with(node, ctx)?
                        .map(keep)
                        .filter(|b| b.as_ref().map_or(true, |b| b.num_rows() > 0))
                        .collect::<Result<_, InterpError>>()?;
                    Ok(out)
                })
                .collect::<Result<Vec<_>, InterpError>>()
        })?;
        let out: Vec<RecordBatch> = pieces.into_iter().flatten().collect();
        if out.is_empty() {
            return crate::execute(node, srcs);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Array, Float64Array, Int64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use bc_expr::{BinaryOp, Expr, Literal};
    use bc_ir::{AggFunc, AggregateItem, JoinOutputCol, JoinSide, JoinType, ProjectionItem, RelOp};

    use super::orient::{orient_counted, original_ids};
    use super::{chunkable, execute_chunked, execute_units, post_ids};
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

    fn dim() -> RecordBatch {
        let d: Vec<i64> = (0..60).collect();
        let g: Vec<i64> = (0..60).map(|i| i % 7).collect();
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("d", DataType::Int64, false),
                Field::new("g", DataType::Int64, false),
            ])),
            vec![Arc::new(Int64Array::from(d)), Arc::new(Int64Array::from(g))],
        )
        .unwrap()
    }

    fn agg_item(func: AggFunc, col: Option<&str>, alias: &str) -> AggregateItem {
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

    /// `post(Aggregate(Join(Filter(Scan 0), Scan 1)))`, grouped by `g` when the join keeps it.
    fn plan(join_type: JoinType, grouped: bool) -> RelOp {
        let keeps_right = matches!(join_type, JoinType::Inner | JoinType::Left);
        let mut output = vec![
            JoinOutputCol {
                side: JoinSide::Left,
                name: "k".into(),
                alias: "k".into(),
            },
            JoinOutputCol {
                side: JoinSide::Left,
                name: "v".into(),
                alias: "v".into(),
            },
        ];
        if keeps_right {
            output.push(JoinOutputCol {
                side: JoinSide::Right,
                name: "g".into(),
                alias: "g".into(),
            });
        }
        let join = RelOp::HashJoin {
            left: Box::new(RelOp::Filter {
                input: Box::new(RelOp::Scan { source_id: 0 }),
                predicate: Expr::Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(Expr::Col { name: "v".into() }),
                    right: Box::new(Expr::Lit {
                        value: Literal::Float(1.0),
                    }),
                },
            }),
            right: Box::new(RelOp::Scan { source_id: 1 }),
            left_keys: vec!["k".into()],
            right_keys: vec!["d".into()],
            join_type,
            output,
            strategy: bc_ir::JoinStrategy::Hash,
        };
        let group_col = if grouped && keeps_right { "g" } else { "k" };
        let aggregate = RelOp::Aggregate {
            input: Box::new(join),
            group_keys: if grouped {
                vec![ProjectionItem {
                    expr: Expr::Col {
                        name: group_col.into(),
                    },
                    alias: "gk".into(),
                }]
            } else {
                Vec::new()
            },
            aggregates: vec![
                agg_item(AggFunc::Sum, Some("v"), "s"),
                agg_item(AggFunc::CountStar, None, "n"),
                agg_item(AggFunc::Mean, Some("v"), "m"),
            ],
        };
        if !grouped {
            return aggregate;
        }
        RelOp::Limit {
            input: Box::new(RelOp::Sort {
                input: Box::new(aggregate),
                keys: vec![bc_ir::SortKey {
                    expr: Expr::Col { name: "gk".into() },
                    descending: false,
                    nulls_first: true,
                }],
                limit: None,
            }),
            n: 50,
            offset: 0,
        }
    }

    fn rows(batches: &[RecordBatch]) -> Vec<String> {
        let mut out = Vec::new();
        for b in batches {
            for i in 0..b.num_rows() {
                let cells: Vec<String> = b
                    .columns()
                    .iter()
                    .map(|c| {
                        if c.is_null(i) {
                            "null".into()
                        } else if let Some(a) = c.as_any().downcast_ref::<Float64Array>() {
                            format!("{:.6}", a.value(i))
                        } else if let Some(a) = c.as_any().downcast_ref::<Int64Array>() {
                            a.value(i).to_string()
                        } else {
                            format!("{c:?}")
                        }
                    })
                    .collect();
                out.push(cells.join("|"));
            }
        }
        out
    }

    fn run(plan: &RelOp, chunks: Vec<Vec<RecordBatch>>) -> Vec<RecordBatch> {
        let sources = vec![vec![fact(0, 0)], vec![dim()]];
        let mut it = chunks.into_iter();
        let mut next = || it.next().map(Ok::<_, InterpError>);
        execute_chunked(plan, &sources, 0, &mut next, 8, 0, &ExecOptions::default()).unwrap()
    }

    /// Every admitted join type, grouped (with a sort and limit above) and global, returns what
    /// the sequential oracle returns over the concatenated chunks — for uneven chunks, an empty
    /// chunk, one chunk, and none at all.
    #[test]
    fn chunked_matches_the_oracle() {
        let whole: Vec<RecordBatch> = (0..6).map(|c| fact(c * 70_000, (c + 1) * 70_000)).collect();
        let splits: Vec<Vec<Vec<RecordBatch>>> = vec![
            vec![whole.clone()],
            vec![
                whole[..1].to_vec(),
                Vec::new(),
                whole[1..4].to_vec(),
                whole[4..].to_vec(),
            ],
            whole.iter().map(|b| vec![b.clone()]).collect(),
            Vec::new(),
        ];
        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            for grouped in [true, false] {
                let p = plan(jt, grouped);
                assert!(chunkable(&p, 0), "{jt:?} grouped={grouped}");
                for (i, split) in splits.iter().enumerate() {
                    let input: Vec<RecordBatch> = split.iter().flatten().cloned().collect();
                    let input = if input.is_empty() {
                        vec![fact(0, 0)]
                    } else {
                        input
                    };
                    let oracle = crate::execute(&p, &[input, vec![dim()]]).unwrap();
                    let mut want = rows(&oracle);
                    let mut got = rows(&run(&p, split.clone()));
                    if !grouped {
                        want.sort();
                        got.sort();
                    }
                    // Grouped results are sorted by the plan, so the order is compared too.
                    assert_eq!(got, want, "{jt:?} grouped={grouped} split={i}");
                }
            }
        }
    }

    /// Units served from memory, one `Vec` of batches per unit, optionally failing one of them.
    struct Units(Vec<Vec<RecordBatch>>, Option<usize>);

    impl crate::stream::UnitSource for Units {
        fn units(&self) -> usize {
            self.0.len()
        }
        fn read(&self, unit: usize) -> Result<Vec<RecordBatch>, InterpError> {
            if self.1 == Some(unit) {
                return Err(InterpError::ChunkSource(format!("unit {unit} failed")));
            }
            Ok(self.0[unit].clone())
        }
    }

    /// The worker-read path returns the oracle's rows — in the oracle's order for a spine, which
    /// its contiguous ranges preserve — for every admitted join type, at every pool width from
    /// one worker up, for more units than workers and fewer, and for a relation with no rows.
    #[test]
    fn units_match_the_oracle_at_every_width() {
        let whole: Vec<RecordBatch> = (0..6).map(|c| fact(c * 70_000, (c + 1) * 70_000)).collect();
        let splits: Vec<Vec<Vec<RecordBatch>>> = vec![
            vec![whole.clone()],
            whole.iter().map(|b| vec![b.clone()]).collect(),
            vec![whole[..2].to_vec(), Vec::new(), whole[2..].to_vec()],
            Vec::new(),
        ];
        let sources = vec![vec![fact(0, 0)], vec![dim()]];
        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            for grouped in [true, false] {
                let p = plan(jt, grouped);
                let RelOp::Aggregate { input: spine, .. } = plan(jt, false) else {
                    unreachable!()
                };
                for split in &splits {
                    let input: Vec<RecordBatch> = split.iter().flatten().cloned().collect();
                    let input = if input.is_empty() {
                        vec![fact(0, 0)]
                    } else {
                        input
                    };
                    let src = Units(split.clone(), None);
                    for workers in [1, 3, 8] {
                        let opts = ExecOptions::default();
                        let oracle = crate::execute(&p, &[input.clone(), vec![dim()]]).unwrap();
                        let got = execute_units(&p, &sources, 0, &src, workers, 0, &opts).unwrap();
                        let (mut want, mut got) = (rows(&oracle), rows(&got));
                        if !grouped {
                            want.sort();
                            got.sort();
                        }
                        assert_eq!(got, want, "{jt:?} grouped={grouped} workers={workers}");
                        let oracle = crate::execute(&spine, &[input.clone(), vec![dim()]]).unwrap();
                        let got =
                            execute_units(&spine, &sources, 0, &src, workers, 0, &opts).unwrap();
                        assert_eq!(rows(&got), rows(&oracle), "spine {jt:?} workers={workers}");
                    }
                }
            }
        }
    }

    /// A unit that fails to read fails the query with its own error rather than dropping rows.
    #[test]
    fn a_failed_unit_fails_the_query() {
        let whole: Vec<Vec<RecordBatch>> = (0..4)
            .map(|c| vec![fact(c * 70_000, (c + 1) * 70_000)])
            .collect();
        let sources = vec![vec![fact(0, 0)], vec![dim()]];
        let src = Units(whole, Some(2));
        let err = execute_units(
            &plan(JoinType::Inner, true),
            &sources,
            0,
            &src,
            4,
            0,
            &ExecOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("unit 2 failed"), "{err}");
    }

    /// A plan with no aggregate — a join spine, bare or under a projection and a sort — collects
    /// each chunk's rows and returns the oracle's rows (in the oracle's order, under the sort).
    #[test]
    fn a_spine_without_an_aggregate_collects_the_oracle_rows() {
        let RelOp::Aggregate { input: join, .. } = plan(JoinType::Inner, false) else {
            unreachable!()
        };
        let projected = RelOp::Project {
            input: join.clone(),
            exprs: vec![
                ProjectionItem {
                    expr: Expr::Col { name: "k".into() },
                    alias: "k".into(),
                },
                ProjectionItem {
                    expr: Expr::Col { name: "v".into() },
                    alias: "v".into(),
                },
                ProjectionItem {
                    expr: Expr::Col { name: "g".into() },
                    alias: "g".into(),
                },
            ],
        };
        let sorted = RelOp::Sort {
            input: Box::new(projected.clone()),
            keys: ["k", "v"]
                .iter()
                .map(|c| bc_ir::SortKey {
                    expr: Expr::Col { name: (*c).into() },
                    descending: false,
                    nulls_first: true,
                })
                .collect(),
            limit: Some(5_000),
        };
        // The select list above the sort, which is how SQL plans `SELECT v, k ... ORDER BY`:
        // the `Project` runs as a post op over the sorted rows, not as part of the spine.
        let selected = RelOp::Project {
            input: Box::new(sorted.clone()),
            exprs: ["v", "k"]
                .iter()
                .map(|c| ProjectionItem {
                    expr: Expr::Col { name: (*c).into() },
                    alias: (*c).into(),
                })
                .collect(),
        };
        let whole: Vec<RecordBatch> = (0..4).map(|c| fact(c * 80_000, (c + 1) * 80_000)).collect();
        let sources = vec![vec![fact(0, 0)], vec![dim()]];
        let src = Units(whole.iter().map(|b| vec![b.clone()]).collect(), None);
        for (p, ordered) in [
            (*join, false),
            (projected, false),
            (sorted, true),
            (selected, true),
        ] {
            assert!(chunkable(&p, 0));
            let mut want = rows(&crate::execute(&p, &[whole.clone(), vec![dim()]]).unwrap());
            let mut got = rows(&run(&p, whole.iter().map(|b| vec![b.clone()]).collect()));
            let mut units =
                rows(&execute_units(&p, &sources, 0, &src, 3, 0, &ExecOptions::default()).unwrap());
            if !ordered {
                want.sort();
                got.sort();
                units.sort();
            }
            assert_eq!(got, want, "ordered={ordered}");
            assert_eq!(units, want, "units, ordered={ordered}");
        }
    }

    /// An aggregate over the chunked aggregate (`max` of grouped sums, TPC-H q15's subquery) runs
    /// as a post op over the inner aggregate's result and matches the oracle.
    #[test]
    fn an_aggregate_over_the_chunked_aggregate_is_a_post_op() {
        let RelOp::Limit { input: sort, .. } = plan(JoinType::Inner, true) else {
            unreachable!()
        };
        let RelOp::Sort { input: grouped, .. } = *sort else {
            unreachable!()
        };
        let nested = RelOp::Aggregate {
            input: grouped,
            group_keys: Vec::new(),
            aggregates: vec![
                agg_item(AggFunc::Max, Some("s"), "mx"),
                agg_item(AggFunc::CountStar, None, "groups"),
            ],
        };
        assert!(chunkable(&nested, 0));
        let whole: Vec<RecordBatch> = (0..4).map(|c| fact(c * 90_000, (c + 1) * 90_000)).collect();
        let want = rows(&crate::execute(&nested, &[whole.clone(), vec![dim()]]).unwrap());
        let got = rows(&run(
            &nested,
            whole.iter().map(|b| vec![b.clone()]).collect(),
        ));
        assert_eq!(got, want);
    }

    /// An aggregate under a join to another input (TPC-H q15's main query) is streamed where it
    /// sits, and the join above it runs over its result.
    #[test]
    fn an_aggregate_under_a_join_is_streamed_in_place() {
        let RelOp::Limit { input: sort, .. } = plan(JoinType::Inner, true) else {
            unreachable!()
        };
        let RelOp::Sort { input: grouped, .. } = *sort else {
            unreachable!()
        };
        let over = RelOp::HashJoin {
            left: Box::new(RelOp::Scan { source_id: 1 }),
            right: Box::new(RelOp::Filter {
                input: grouped,
                predicate: Expr::Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(Expr::Col { name: "n".into() }),
                    right: Box::new(Expr::Lit {
                        value: Literal::Int(10),
                    }),
                },
            }),
            left_keys: vec!["g".into()],
            right_keys: vec!["gk".into()],
            join_type: JoinType::Inner,
            output: vec![
                JoinOutputCol {
                    side: JoinSide::Left,
                    name: "d".into(),
                    alias: "d".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Right,
                    name: "s".into(),
                    alias: "s".into(),
                },
                JoinOutputCol {
                    side: JoinSide::Right,
                    name: "n".into(),
                    alias: "n".into(),
                },
            ],
            strategy: bc_ir::JoinStrategy::Hash,
        };
        assert!(chunkable(&over, 0));
        let whole: Vec<RecordBatch> = (0..4).map(|c| fact(c * 90_000, (c + 1) * 90_000)).collect();
        let mut want = rows(&crate::execute(&over, &[whole.clone(), vec![dim()]]).unwrap());
        let mut got = rows(&run(&over, whole.iter().map(|b| vec![b.clone()]).collect()));
        want.sort();
        got.sort();
        assert!(!want.is_empty());
        assert_eq!(got, want);
    }

    /// The driving scan on the *build* side of an inner join is swapped onto the probe side, and
    /// the swapped plan still returns the oracle's rows with the columns in the right places.
    #[test]
    fn a_driving_scan_on_the_build_side_is_reoriented() {
        let RelOp::Limit { input: sort, .. } = plan(JoinType::Inner, true) else {
            unreachable!()
        };
        let RelOp::Sort {
            input: aggregate, ..
        } = *sort
        else {
            unreachable!()
        };
        let RelOp::Aggregate {
            input,
            group_keys,
            aggregates,
        } = *aggregate
        else {
            unreachable!()
        };
        let RelOp::HashJoin {
            left,
            right,
            left_keys,
            right_keys,
            output,
            ..
        } = *input
        else {
            unreachable!()
        };
        let flipped = RelOp::Aggregate {
            input: Box::new(RelOp::HashJoin {
                left: right,
                right: left,
                left_keys: right_keys,
                right_keys: left_keys,
                join_type: JoinType::Inner,
                output: output
                    .into_iter()
                    .map(|c| JoinOutputCol {
                        side: match c.side {
                            JoinSide::Left => JoinSide::Right,
                            JoinSide::Right => JoinSide::Left,
                        },
                        ..c
                    })
                    .collect(),
                strategy: bc_ir::JoinStrategy::Broadcast,
            }),
            group_keys,
            aggregates,
        };
        assert!(chunkable(&flipped, 0));
        let whole: Vec<RecordBatch> = (0..4).map(|c| fact(c * 90_000, (c + 1) * 90_000)).collect();
        let mut want = rows(&crate::execute(&flipped, &[whole.clone(), vec![dim()]]).unwrap());
        let mut got = rows(&run(
            &flipped,
            whole.iter().map(|b| vec![b.clone()]).collect(),
        ));
        want.sort();
        got.sort();
        assert_eq!(got, want);
    }

    /// A plan scanning the driving source twice, or through a join that keeps unmatched build
    /// rows, or with no aggregate, is not chunkable — and `execute_chunked` says so up front.
    #[test]
    fn unsupported_shapes_are_declined() {
        let RelOp::Aggregate {
            group_keys,
            aggregates,
            ..
        } = plan(JoinType::Inner, false)
        else {
            unreachable!()
        };
        let self_join = RelOp::Aggregate {
            input: Box::new(RelOp::HashJoin {
                left: Box::new(RelOp::Scan { source_id: 0 }),
                right: Box::new(RelOp::Scan { source_id: 0 }),
                left_keys: vec!["k".into()],
                right_keys: vec!["k".into()],
                join_type: JoinType::Inner,
                output: vec![JoinOutputCol {
                    side: JoinSide::Left,
                    name: "v".into(),
                    alias: "v".into(),
                }],
                strategy: bc_ir::JoinStrategy::Hash,
            }),
            group_keys,
            aggregates,
        };
        assert!(!chunkable(&self_join, 0));
        for jt in [JoinType::Right, JoinType::Full] {
            assert!(!chunkable(&plan(jt, false), 0), "{jt:?}");
        }
        assert!(!chunkable(
            &RelOp::Distinct {
                input: Box::new(RelOp::Scan { source_id: 0 }),
                keys: Vec::new(),
                order: Vec::new(),
                limit: None
            },
            0
        ));
        let mut none = || None;
        let r = execute_chunked(
            &self_join,
            &[vec![fact(0, 0)]],
            0,
            &mut none,
            2,
            0,
            &ExecOptions::default(),
        );
        assert!(matches!(r, Err(InterpError::NotChunkable)));
    }

    /// The post plan's operators map to their ids in the full plan, across the subtree the
    /// collected result replaced: `Limit > Sort > Aggregate > Join(Filter > Scan0, Scan1)` is
    /// 0..=6 in pre-order.
    #[test]
    fn post_plan_operators_map_to_the_full_plans_ids() {
        let full = plan(JoinType::Inner, true);
        // The aggregate core: only the `Limit` and `Sort` above it remain, then its result.
        assert_eq!(post_ids(&full, &[0, 0]), vec![Some(0), Some(1), None]);
        // A core inside the join's left side: `Scan1` follows the two-node `Filter > Scan0`
        // subtree, so it is 6 in the full plan though it is 5th in the post plan. Absorbing by
        // offset would have filed it under 5, the scan the subtree swallowed.
        assert_eq!(
            post_ids(&full, &[0, 0, 0, 0]),
            vec![Some(0), Some(1), Some(2), Some(3), None, Some(6)]
        );
        // No core replaced at the root's own position: every operator is itself.
        assert_eq!(post_ids(&full, &[]), vec![None]);
    }

    /// A swapped join reorders its children in the numbering the meter uses, and every metric
    /// must still be filed under the op id the control plane gave that node in *its* plan.
    #[test]
    fn a_swapped_plans_metrics_are_numbered_as_the_original_plan() {
        fn preorder<'a>(node: &'a RelOp, out: &mut Vec<&'a RelOp>) {
            out.push(node);
            for c in node.children() {
                preorder(c, out);
            }
        }
        fn same_node(a: &RelOp, b: &RelOp) -> bool {
            match (a, b) {
                (RelOp::Scan { source_id: x }, RelOp::Scan { source_id: y }) => x == y,
                _ => std::mem::discriminant(a) == std::mem::discriminant(b),
            }
        }
        let original = plan(JoinType::Inner, true);
        let (oriented, swaps) = orient_counted(&original, 1);
        assert_eq!(
            swaps, 1,
            "driving from the right-hand scan must swap the join"
        );
        let ids = original_ids(&original, &oriented, 1);
        let (mut orig_nodes, mut ran_nodes) = (Vec::new(), Vec::new());
        preorder(&original, &mut orig_nodes);
        preorder(&oriented, &mut ran_nodes);
        assert_eq!(ids.len(), ran_nodes.len());
        for (ran, &id) in ran_nodes.iter().zip(&ids) {
            assert!(
                same_node(ran, orig_nodes[id as usize]),
                "{ran:?} filed under {id}"
            );
        }
        // The swap really reordered the numbering: identity would have filed the scans wrongly.
        assert_ne!(ids, (0..ids.len() as u32).collect::<Vec<_>>());
        // Positive control: an unswapped plan maps every node to itself.
        let (same, none) = orient_counted(&original, 0);
        assert_eq!(none, 0);
        let identity = original_ids(&original, &same, 0);
        assert_eq!(identity, (0..identity.len() as u32).collect::<Vec<_>>());
    }
}
