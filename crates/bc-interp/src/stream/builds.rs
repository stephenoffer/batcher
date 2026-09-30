//! Preparing a hash join's build side once, for every worker that will probe it.
//!
//! Split out of `stream::mod` along its clearest seam: everything here runs **before** the probe
//! pipeline is composed, over the *unsharded* sources, and produces the constants that pipeline
//! then reads — the hashed build tables, the spine breakers already evaluated, and the runtime
//! key filters derived from them. `mod.rs` composes streams; this module supplies what they close
//! over.
//!
//! Rebuilding any of it per worker would be `workers x` the cost, and on a chain of joins that is
//! the dominant term — the thing that would make a "parallel" streaming executor slower than the
//! materializing one it replaces.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::RecordBatch;
use bc_ir::{JoinType, RelOp};
use bc_runtime::join::{streaming_shape_supported, BroadcastProbe};

use super::{parallel, runtime_filter, Meter};
use crate::error::InterpError;
use crate::ops;

/// A hash join's build side, prepared once and shared by every worker that probes it.
///
/// Rebuilding this per worker would be `workers x` the build cost, and on a chain of joins that
/// is the dominant term — the thing that would make a "parallel" streaming executor slower than
/// the materializing one it replaces.
pub(crate) struct JoinBuild {
    /// The materialized build relation (small by construction — it is the broadcast side).
    pub(crate) side: RecordBatch,
    /// The hash table over it, or `None` when this join's shape cannot be probed per morsel and
    /// the materialized fallback must be used.
    pub(crate) probe: Option<BroadcastProbe>,
}

impl JoinBuild {
    /// Whether this join can be probed one morsel at a time. `false` means every worker that
    /// probes it would re-join the whole build side, so the driver must not shard through it.
    pub(crate) fn has_morsel_probe(&self) -> bool {
        self.probe.is_some()
    }
}

/// Prepared build sides, keyed by the identity of their `HashJoin` node — plus the runtime
/// filters those build sides imply about the probe sides.
///
/// The key is the node's address. The plan is borrowed for the whole execution and never moves,
/// so the address is a stable identity — and it distinguishes two structurally identical joins in
/// the same plan, which a structural key would conflate.
///
/// The filters live here rather than beside the cache because they are *derived from it* and
/// have exactly its lifetime and its sharing: every path that probes a prepared build side is a
/// path that may also apply that side's key filter, so carrying them together means no executor
/// entry point has to learn about them (see [`runtime_filter`]).
pub(crate) struct BuildCache {
    joins: HashMap<usize, Arc<JoinBuild>>,
    filters: runtime_filter::RuntimeFilters,
    /// Cumulative materialized bytes of every build side in `joins`.
    ///
    /// Every side prepared into this cache stays resident until the query ends — that is the
    /// point of prebuilding them — so the quantity that has to fit in the envelope is this
    /// sum, not any one side. Each side is "small by construction" relative to the relation
    /// it broadcasts, which is what made checking them one at a time look sufficient; a plan
    /// with several joins then holds up to `joins.len() * budget` while no single check ever
    /// fires, and the process is killed at exactly the point the handoff exists to prevent.
    /// TPC-H q9 at sf100 is the shape that shows it: five build sides, an 82 GB envelope, and
    /// a 184 GB machine.
    ///
    /// `bc-py` already draws this distinction one level up — its pool is process-wide
    /// "(per-query pools would let N concurrent queries each hold `budget` and OOM)". This is
    /// the same argument one level down, for N build sides inside a single query.
    bytes: u64,
    /// Probe sides evaluated ahead of their join to restrict its build side (see
    /// [`collect_builds`]), keyed by the probe subtree's `node_key`. Each is a finished relation
    /// the executor yields in place of re-running the subtree, exactly as it yields a
    /// [`MatCache`] entry, and it is resident, so it is counted in `bytes` too.
    probe_leaves: MatCache,
}

impl BuildCache {
    fn new() -> Self {
        Self {
            joins: HashMap::new(),
            filters: runtime_filter::RuntimeFilters::new(),
            bytes: 0,
            probe_leaves: MatCache::new(),
        }
    }

    /// The probe side evaluated ahead of its join for this node, if it has one.
    pub(crate) fn probe_leaf(&self, key: usize) -> Option<&Arc<Vec<RecordBatch>>> {
        self.probe_leaves.get(&key)
    }

    /// Whether any probe side was evaluated ahead of its join.
    pub(crate) fn has_probe_leaves(&self) -> bool {
        !self.probe_leaves.is_empty()
    }

    /// `mats` with this cache's probe leaves added: the view every shardability check reads.
    pub(crate) fn merged_with(&self, mats: Option<&MatCache>) -> MatCache {
        let mut out = mats.cloned().unwrap_or_default();
        for (k, v) in &self.probe_leaves {
            out.insert(*k, Arc::clone(v));
        }
        out
    }

    fn insert(&mut self, key: usize, build: Arc<JoinBuild>) {
        // The probe table counts too. It is ~9 bytes per build row on top of the relation, and
        // since a large build side now gets one (see [`Admission`]) leaving it out understates
        // what this cache holds by roughly half on exactly the joins the envelope is about.
        self.bytes += build.side.get_array_memory_size() as u64
            + build.probe.as_ref().map_or(0, |p| p.heap_bytes()) as u64;
        self.joins.insert(key, build);
    }

    /// Stop if the build sides prepared so far have outgrown `budget` (`0` is unbounded).
    ///
    /// Returning here is the same handoff a breaker makes, and sound for the same reason: the
    /// caller re-runs on the materializing executor, which spills, and the two are checked
    /// against one sequential oracle — so this changes peak memory and speed, never the answer.
    fn check_total(&self, budget: usize) -> Result<(), InterpError> {
        if budget > 0 && self.bytes as usize > budget {
            return Err(InterpError::MemoryBudgetExceeded {
                needed: self.bytes as usize,
                budget,
                reason: "the streaming executor's join build sides do not spill",
            });
        }
        Ok(())
    }

    /// The prepared build side for a `HashJoin` node, if it has one.
    pub(crate) fn get(&self, key: &usize) -> Option<&Arc<JoinBuild>> {
        self.joins.get(key)
    }

    /// The runtime filters to apply to this node's output, if any.
    pub(crate) fn filters_for(&self, key: usize) -> Option<&[runtime_filter::PendingFilter]> {
        self.filters.get(&key).map(Vec::as_slice)
    }
}

/// Spine breakers that have already been evaluated, keyed the same way (`node_key`).
///
/// A breaker sitting between the plan root and the driving scan used to force the *whole* query
/// onto the sequential path: sharding cannot cross it (a breaker handed one shard answers for one
/// shard), and `spine_is_shardable` refuses the plan rather than risk that. But its own subtree is
/// usually the expensive half and is very often perfectly shardable on its own — TPC-H q17's
/// decorrelated aggregate over 6M rows of `lineitem` is exactly that, and it ran on one core.
///
/// So the breaker is evaluated **up front, in parallel, over the unsharded sources** — the same
/// treatment [`prebuild_joins`] already gives a join's build side — and its result is stored here.
/// From then on it is a materialized *leaf*: [`build_with`] yields the stored batches instead of
/// executing the subtree, and the spine above it becomes shardable because nothing on that spine
/// is a breaker any more.
///
/// The soundness argument is the one that matters, because this is a wrong-answer-shaped change:
/// sharding is never extended *through* a breaker. The breaker is fully evaluated first, over
/// every row of its input, and what the workers then share is a finished relation — identical in
/// every worker, and never itself sharded.
pub(crate) type MatCache = HashMap<usize, Arc<Vec<RecordBatch>>>;

/// Execute (and hash) every hash-join build side in `plan`, once, across `workers`.
///
/// Each build side is run on the streaming path too, so preparing it never materializes its
/// subtree either — and it is *sharded* like any other streamed relation, because a build side is
/// not always the small one (see `collect_builds`).
pub(crate) fn prebuild_joins(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    meter: Option<&Meter>,
    budget: usize,
    workers: usize,
    opts: Option<&crate::par::ExecOptions>,
) -> Result<Arc<BuildCache>, InterpError> {
    // A plan with exactly one hash join can decline the per-morsel probe and still be run
    // across every core, by handing off to the materializing executor
    // (`parallel::unshardable_join_reason`) — which keeps the partitioned radix join too.
    // With more than one join that hand-off declines by construction, so declining here
    // selects the sequential pipeline instead. `flat_probe_pays` needs to know which.
    let admission = Admission {
        driving: driving_rows(plan, sources),
        can_hand_off: parallel::count_hash_joins(plan) == 1,
        chunked: false,
        under_probe: false,
    };
    prebuild_with(plan, sources, meter, budget, workers, opts, admission, None)
}

/// [`prebuild_joins`] for [`super::chunked`], which streams the driving relation in chunks.
///
/// Every build that *can* take a per-morsel probe gets one, whatever its size or join type.
/// The admission rule's two refusals both rest on an alternative this caller does not have:
/// handing the plan to the materializing executor (which needs the probe side resident), and,
/// for a large `Semi`/`Anti` build, the partitioned radix join (which, run per morsel, rebuilds
/// its partition tables on every call — measured on TPC-H sf100 q3, `join_partition_into` was
/// 46% of the query). A flat table built once and probed by every chunk is the only shape here
/// that hashes each build row once.
///
/// `driving` is the streamed relation's `(source id, total rows)` when the caller knows it (see
/// [`runtime_filter::plan_filters`]); its entry in `sources` is only a schema carrier.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prebuild_joins_for_chunks(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    meter: Option<&Meter>,
    budget: usize,
    workers: usize,
    opts: Option<&crate::par::ExecOptions>,
    driving: Option<(usize, usize)>,
) -> Result<Arc<BuildCache>, InterpError> {
    let admission = Admission {
        driving: 0,
        can_hand_off: false,
        chunked: true,
        under_probe: false,
    };
    prebuild_with(
        plan, sources, meter, budget, workers, opts, admission, driving,
    )
}

#[allow(clippy::too_many_arguments)]
fn prebuild_with(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    meter: Option<&Meter>,
    budget: usize,
    workers: usize,
    opts: Option<&crate::par::ExecOptions>,
    admission: Admission,
    driving: Option<(usize, usize)>,
) -> Result<Arc<BuildCache>, InterpError> {
    let mut cache = BuildCache::new();
    collect_builds(
        plan, sources, &mut cache, meter, budget, workers, admission, opts,
    )?;
    // Every build side now exists, so every reducible join's key set is a known constant and can
    // be placed over the probe pipeline that is about to run. One pass per build key column; no
    // execution. See [`runtime_filter`] for which joins qualify and why it cannot regress.
    cache.filters = runtime_filter::plan_filters(plan, sources, &cache, driving, meter);
    Ok(Arc::new(cache))
}

#[allow(clippy::too_many_arguments)]
/// Whether to evaluate this join's probe side first and restrict its build side to the probe keys.
///
/// A decorrelated subquery joins its outer rows to `Aggregate(inner, group_keys=[k])`, and the
/// streaming executor computes that aggregate over the *whole* inner relation before a probe row
/// exists. TPC-H q21 at sf10 builds all 15M `l_orderkey` groups of `lineitem` for the ~700k orders
/// the outer side keeps. The materializing executor evaluates the probe side first and restricts
/// (`join_par::sideways`); here the same happens only on Kyber's verdict that the build side reads
/// far more than the probe can match (`ExecOptions::prefer_sideways`), because evaluating the
/// probe side first materializes it, which this executor otherwise never does.
///
/// Not on the chunked executor: its driving relation is a zero-row carrier until the chunks
/// arrive, so a probe side evaluated now would be evaluated over nothing. And not for a join on
/// another join's probe side: the stored probe side cannot be sharded through, so everything
/// above it runs on one core. TPC-H q21's join has only filters and a small aggregate above it
/// (648 -> 215 ms at sf10); q18's has a join against `lineitem`, and lost 276 -> 310 ms.
fn sideways_first(
    opts: Option<&crate::par::ExecOptions>,
    admission: Admission,
    join_type: JoinType,
    right_keys: &[String],
    right: &RelOp,
    sources: &[Vec<RecordBatch>],
) -> bool {
    opts.is_some_and(|o| o.prefer_sideways)
        && !admission.chunked
        && !admission.under_probe
        && crate::join_par::sideways::restrictable(join_type, right_keys, right, sources)
}

/// Prepare `plan`'s build side from sources restricted to its probe side's keys, and keep the
/// probe side, already evaluated, as a leaf (see [`sideways_first`]).
///
/// The build side's operators run over a filtered source, so their row counts describe this
/// query's restriction, not the subplan: they are not metered, for the reason the materializing
/// executor gives in the same place (Kyber would learn a cardinality the subplan does not have).
/// The probe side runs on its own nodes with the meter, so its counts are its own, and it is not
/// run again: [`super::build_with`] yields the stored rows. When the restriction declines (the
/// probe side turned out too large to cut the source), the build side is prepared unrestricted.
#[allow(clippy::too_many_arguments)]
fn build_sideways(
    plan: &RelOp,
    left: &RelOp,
    right: &RelOp,
    left_keys: &[String],
    right_keys: &[String],
    join_type: JoinType,
    sources: &[Vec<RecordBatch>],
    cache: &mut BuildCache,
    meter: Option<&Meter>,
    budget: usize,
    workers: usize,
    admission: Admission,
    opts: Option<&crate::par::ExecOptions>,
) -> Result<(), InterpError> {
    let probe = parallel::run(left, sources, workers, meter, budget, false, None, opts)?;
    let restricted = crate::join_par::sideways::restrict_right_sources(
        join_type, left_keys, right_keys, &probe, right, sources,
    )?;
    let batches = match &restricted {
        Some(restricted) => {
            parallel::run(right, restricted, workers, None, budget, false, None, opts)?
        }
        None => parallel::run(right, sources, workers, meter, budget, false, None, opts)?,
    };
    cache.bytes += probe
        .iter()
        .map(|b| b.get_array_memory_size() as u64)
        .sum::<u64>();
    cache.probe_leaves.insert(node_key(left), Arc::new(probe));
    if let Ok(side) = ops::materialize(&batches) {
        let probe_table = make_probe(&side, right_keys, join_type, admission)?;
        cache.insert(
            node_key(plan),
            Arc::new(JoinBuild {
                side,
                probe: probe_table,
            }),
        );
    }
    drop_across_pool(batches);
    cache.check_total(budget)
}

/// Drop a build side's morsels across the pool rather than on the calling thread.
///
/// `ops::materialize` has copied them into one batch, so they are garbage the moment it
/// returns, and every one of their buffers was allocated by a pool worker. Freeing a block from
/// a thread other than the one that allocated it is mimalloc's slow path, and doing all of them
/// on one thread was the longest serial stretch of TPC-H q7 at sf10: ~8 ms between two fully
/// parallel phases. Each worker frees the morsels it takes instead.
fn drop_across_pool(batches: Vec<RecordBatch>) {
    use rayon::prelude::*;
    batches.into_par_iter().for_each(drop);
}

/// Options for running this build side on the **materializing** executor, or `None` to stream it.
///
/// A build side is prepared by `parallel::run` — the streaming executor — whatever it contains.
/// When it is a join-free grouped aggregate, that is the wrong executor by a wide margin, and the
/// margin is the whole of TPC-H q18's gap: its `SEMI` join builds from `GROUP BY l_orderkey` over
/// 60M rows to 15M groups, which `explain(analyze=True)` attributes **99% of the query's operator
/// time** to. Measured standing alone, both orders, minimum of three: **4,660 ms streaming
/// against 736 ms materializing**, with read (347 ms) plus aggregate (505 ms) accounting for the
/// second figure exactly.
///
/// The shape test is `materializing_aggregate_is_faster`, the same guard `bc-py` applies to a
/// whole plan, and it admits only a **join-free** grouped aggregate — so what this materializes is
/// the subtree's input and one hash table over it, never a join's intermediates. That is the
/// distinction that makes it safe here: routing *whole* plans on a permissive affordability test
/// is what took q17/q18/q20/q21 from completing to `SIGKILL` on a 30 GiB box.
///
/// **`op_budgets` is dropped and everything else kept.** The map is keyed by pre-order `op_id`
/// over the *whole* plan, so handing it to a subtree would budget the wrong operators; without it
/// each falls back to the global envelope, which is what an unkeyed operator already does. The
/// pool, the spill directory, the codec and the cancel token all carry over — a build side that
/// materializes must still be able to spill where the caller configured and to be cancelled.
///
/// `None` whenever the caller gave us no options (the sequential `execute_streaming` entries, and
/// the tests), so those paths stream exactly as they always have.
pub(super) fn build_materializes_faster(
    right: &RelOp,
    opts: Option<&crate::par::ExecOptions>,
) -> Option<crate::par::ExecOptions> {
    let opts = opts?;
    if !parallel::materializing_aggregate_is_faster(right) {
        return None;
    }
    let mut sub = opts.clone();
    sub.op_budgets = Default::default();
    Some(sub)
}

/// One plain build side found on the probe spine, waiting to be prepared.
struct PendingBuild<'a> {
    key: usize,
    right: &'a RelOp,
    right_keys: &'a [String],
    join_type: JoinType,
    admission: Admission,
}

#[allow(clippy::too_many_arguments)]
fn collect_builds(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    cache: &mut BuildCache,
    meter: Option<&Meter>,
    budget: usize,
    workers: usize,
    admission: Admission,
    opts: Option<&crate::par::ExecOptions>,
) -> Result<(), InterpError> {
    let mut pending = Vec::new();
    gather_builds(
        plan,
        sources,
        cache,
        meter,
        budget,
        workers,
        admission,
        opts,
        &mut pending,
    )?;
    // The build sides of one spine are independent of one another, so they are prepared
    // together. One at a time, a build that cannot fill the machine left it idle: at TPC-H
    // sf100 `supplier` is one row group and `customer` fifteen, and q5 and q10 spent the first
    // ~35% of their wall time below half the machine's cores while those builds ran in turn,
    // then the probe pipeline ran at 94%. Together they share the pool by work stealing, so a
    // build that can use every core still does.
    use rayon::prelude::*;
    let built: Vec<Result<Option<(usize, JoinBuild)>, InterpError>> =
        crate::par::pool_for(workers.max(1))?.install(|| {
            pending
                .par_iter()
                .map(|p| prepare_build(p, sources, meter, budget, workers, opts))
                .collect()
        });
    for build in built {
        if let Some((key, join_build)) = build? {
            cache.insert(key, Arc::new(join_build));
            // After the insert, not before: the check is on what is *resident*, and this side
            // is resident now. Declining before building it would be the thing the comment
            // in `prepare_build` rules out -- refusing a plan on a fact that does not exist yet.
            cache.check_total(budget)?;
        }
    }
    Ok(())
}

/// Walk the probe spine, preparing sideways joins in place and listing every plain build side.
#[allow(clippy::too_many_arguments)]
fn gather_builds<'a>(
    plan: &'a RelOp,
    sources: &[Vec<RecordBatch>],
    cache: &mut BuildCache,
    meter: Option<&Meter>,
    budget: usize,
    workers: usize,
    admission: Admission,
    opts: Option<&crate::par::ExecOptions>,
    pending: &mut Vec<PendingBuild<'a>>,
) -> Result<(), InterpError> {
    if let RelOp::HashJoin {
        left,
        right,
        left_keys,
        right_keys,
        join_type,
        ..
    } = plan
    {
        if sideways_first(opts, admission, *join_type, right_keys, right, sources) {
            return build_sideways(
                plan, left, right, left_keys, right_keys, *join_type, sources, cache, meter,
                budget, workers, admission, opts,
            );
        }
        // Only the probe spine draws on *this* cache. The build side is executed as one
        // self-contained unit, which prepares whatever joins it holds itself, so descending into
        // it here would build them twice.
        let probe_side = Admission {
            under_probe: true,
            ..admission
        };
        gather_builds(
            left, sources, cache, meter, budget, workers, probe_side, opts, pending,
        )?;
        pending.push(PendingBuild {
            key: node_key(plan),
            right,
            right_keys,
            join_type: *join_type,
            admission,
        });
        return Ok(());
    }
    for child in plan.children() {
        gather_builds(
            child, sources, cache, meter, budget, workers, admission, opts, pending,
        )?;
    }
    Ok(())
}

/// Execute one build side and hash it, or `None` when it cannot be materialized.
fn prepare_build(
    p: &PendingBuild<'_>,
    sources: &[Vec<RecordBatch>],
    meter: Option<&Meter>,
    budget: usize,
    workers: usize,
    opts: Option<&crate::par::ExecOptions>,
) -> Result<Option<(usize, JoinBuild)>, InterpError> {
    // Shard the build side across the workers, exactly as the probe side is sharded. This
    // was the streaming executor's worst asymmetry: the probe ran on every core while the
    // build -- the *whole* other relation -- ran on one. It is hashed into a table either way,
    // so single-threading it bought no memory and cost the entire build serially. TPC-H q4
    // (`orders SEMI lineitem`) is the shape that exposes it: a semi join's build is always
    // the right input (it is not commutative, so Kyber cannot swap it), so the 3.8M-row side
    // is built and probed by 57k rows -- 279 ms streaming vs 45 ms materializing. Recursion
    // terminates because each build subtree is strictly smaller than the plan.
    // Never hands off: a build side is prepared *for* a decision the caller has not made yet,
    // so declining here would abort the plan before the fact that decides it exists.
    let batches = match build_materializes_faster(p.right, opts) {
        Some(sub) => {
            let (out, metrics) = crate::par::execute_parallel_with_metrics(p.right, sources, &sub)?;
            if let Some(meter) = meter {
                meter.absorb(p.right, &metrics);
            }
            out
        }
        None => parallel::run(p.right, sources, workers, meter, budget, false, None, opts)?,
    };
    let built = match ops::materialize(&batches) {
        Ok(side) => {
            let probe = make_probe(&side, p.right_keys, p.join_type, p.admission)?;
            Some((p.key, JoinBuild { side, probe }))
        }
        Err(_) => None,
    };
    drop_across_pool(batches);
    Ok(built)
}

/// Rows in the relation the probe spine will be sharded over, or 0 when there is none.
///
/// The probe side's size is not knowable without materializing it — the thing this executor
/// exists to avoid — but its *driving scan* bounds it: every row the probe pipeline sees comes
/// from there, and the operators between only drop rows. That bound is what
/// [`flat_probe_pays`] weighs the build against.
fn driving_rows(plan: &RelOp, sources: &[Vec<RecordBatch>]) -> usize {
    let spine = match plan {
        RelOp::Aggregate { input, .. } | RelOp::Distinct { input, .. } => input.as_ref(),
        other => other,
    };
    parallel::leftmost_scan(spine, None)
        .and_then(|sid| sources.get(sid))
        .map_or(0, |b| b.iter().map(|b| b.num_rows()).sum())
}

/// Whether a build above the ceiling may still take a per-morsel probe.
///
/// Declining is not free: a spine join with no per-morsel probe makes the whole probe spine
/// un-shardable (`parallel::spine_is_shardable`), and a multi-join plan cannot hand off to the
/// materializing executor either (`unshardable_join_reason` declines), so what the ceiling
/// selects on those plans is **the sequential streaming pipeline** — 11-18 cores of 96 on TPC-H
/// sf10 against DuckDB's 30-57, and the whole reason Batcher won sf1 and lost sf10 on every
/// multi-way join.
///
/// Two conditions remain, and both are structural rather than fitted:
///
/// * **`Semi` / `Anti` are never admitted.** Their build is the side being *tested against*
///   rather than the side being emitted, so a flat table over it is the whole cost of the join
///   — there is no output gather for it to amortize against. q4 builds 37.9M rows for a `Semi`
///   and q22 15M for an `Anti`.
/// * **Never where the plan could hand off instead.** One hash join and a probe bigger than
///   the build is the case `unshardable_join_reason` gives to the materializing executor,
///   which spreads the probe across every core *and* keeps the cache-resident radix join.
///   Pinned by
///   `stream_oracle::a_large_probe_against_a_huge_build_is_handed_back_only_when_the_caller_asks`.
///
/// **A size ratio used to stand here as well, and it was measuring something else.** Two
/// fitted constants admitted a large build only when the probe dominated it by 6x or when the
/// probe was the smaller side — because past those bands a flat build measured slower than the
/// sequential spine it replaced. What made it slower was not the probe's cache misses: it was
/// the *build*, whose probe-side bloom allocated one full-size filter per shard and folded the
/// 64 of them together in a serial bit-OR. At sf10 q9's 3.26M-row build that merge alone was
/// **34.3 ms, more than the 31.3 ms parallel hash insert beside it**, and it was charged twice
/// per query. `JoinTable::bloom` now shards the filter the way the heads are sharded and merges
/// nothing, and with that cost gone the ratio bands invert: admitting every non-semi build took
/// TPC-H sf10 **q9 468 -> 248 ms** and the suite geomean against DuckDB **1.044 -> 0.993**, with
/// q4 (`Semi`, still refused) and q13 unmoved. The constants are gone rather than re-fitted —
/// what they were compensating for is fixed at its source.
///
/// Below the ceiling this admits everything, exactly as before.
#[derive(Clone, Copy)]
struct Admission {
    /// Rows in the relation the probe spine is sharded over — the probe side's upper bound.
    driving: usize,
    /// Whether declining still leaves this plan a parallel path (the materializing hand-off).
    can_hand_off: bool,
    /// The chunked executor's builds: admit every shape a flat probe supports.
    chunked: bool,
    /// Whether this join sits on another hash join's probe side. The sideways prepass stores
    /// its probe side as a finished leaf, and a spine cannot be sharded through one, so it is
    /// taken only by a join with no hash join above it on the spine ([`sideways_first`]).
    under_probe: bool,
}

impl Admission {
    fn admits(self, build_rows: usize, join_type: JoinType) -> bool {
        let ceiling = bc_runtime::join::RADIX_MIN_BUILD_ROWS_BROADCAST;
        if self.chunked || build_rows <= ceiling {
            return true; // unchanged: this is the shape the ceiling was drawn for
        }
        if matches!(join_type, JoinType::Semi | JoinType::Anti) {
            return false;
        }
        !(self.can_hand_off && self.driving > build_rows)
    }
}

/// Identity of a plan node — its address in the (borrowed, immobile) plan tree.
pub(crate) fn node_key(plan: &RelOp) -> usize {
    plan as *const RelOp as usize
}

/// The per-morsel probe table over `side`, or `None` when this join's shape cannot be served
/// per morsel (`Right`/`Full`, or a non-integer key) and the materialized path must take over.
///
/// **The build side's size is deliberately not a condition here**, and that is the difference
/// between this executor using the machine and using one core of it. `streaming_supported`'s row
/// ceiling (`RADIX_MIN_BUILD_ROWS_BROADCAST`, ~2M rows ≈ an L3-sized table) compares a flat probe
/// against the *partitioned radix* join, and for that comparison it is right. It is not the
/// comparison this caller is making. A join with no per-morsel probe makes the whole probe spine
/// un-shardable (`parallel::spine_is_shardable`), and a plan with more than one join cannot hand
/// off to the materializing executor either (`unshardable_join_reason` declines) — so what the
/// ceiling actually selects, on a multi-join plan, is **the sequential streaming pipeline**.
///
/// Measured on TPC-H sf10, where the filtered `orders` build side is 2,275,919 rows and the
/// ceiling is 2,097,152 — 8% over, and the whole query changes character:
///
/// | query | before | after | cores before → after |
/// |---|---:|---:|---|
/// | q5  | 445 ms | (see BENCHMARK_RESULTS) | 11.5 → |
///
/// The cache misses a flat >L3 table pays are real, and they are bounded by the build side; the
/// parallelism they buy is bounded by the machine. At sf1 the same build is 228k rows, under the
/// ceiling, which is exactly why Batcher won sf1 and lost sf10 on every multi-way join.
fn make_probe(
    side: &RecordBatch,
    right_keys: &[String],
    join_type: JoinType,
    admission: Admission,
) -> Result<Option<BroadcastProbe>, InterpError> {
    let build_keys = ops::columns_by_name(side, right_keys)?;
    let key_types: Vec<&arrow::datatypes::DataType> =
        build_keys.iter().map(|k| k.data_type()).collect();
    let rt = ops::map_join_type(join_type);
    if !streaming_shape_supported(rt, &key_types) {
        return Ok(None);
    }
    if !admission.admits(side.num_rows(), join_type) {
        return Ok(None);
    }
    let tuning = bc_arrow::RuntimeTuning::default();
    // `probe_rows` only decides whether the probe-side bloom pays for itself, and the bloom is a
    // pure short-circuit with no false negatives — the emitted rows are identical either way. A
    // streamed probe is by definition the large side, and its exact row count is not knowable
    // without materializing it, which is the thing this executor exists to avoid.
    Ok(BroadcastProbe::over_any_build(
        &build_keys,
        rt,
        usize::MAX,
        tuning.bloom_fp_rate,
        tuning.bloom_min_build_rows,
    ))
}
