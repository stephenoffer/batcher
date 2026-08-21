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
}

impl BuildCache {
    fn new() -> Self {
        Self {
            joins: HashMap::new(),
            filters: runtime_filter::RuntimeFilters::new(),
            bytes: 0,
        }
    }

    fn insert(&mut self, key: usize, build: Arc<JoinBuild>) {
        self.bytes += build.side.get_array_memory_size() as u64;
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
) -> Result<Arc<BuildCache>, InterpError> {
    let mut cache = BuildCache::new();
    // A plan with exactly one hash join can decline the per-morsel probe and still be run
    // across every core, by handing off to the materializing executor
    // (`parallel::unshardable_join_reason`) — which keeps the partitioned radix join too.
    // With more than one join that hand-off declines by construction, so declining here
    // selects the sequential pipeline instead. `flat_probe_pays` needs to know which.
    let admission = Admission {
        driving: driving_rows(plan, sources),
        can_hand_off: parallel::count_hash_joins(plan) == 1,
    };
    collect_builds(plan, sources, &mut cache, meter, budget, workers, admission)?;
    // Every build side now exists, so every reducible join's key set is a known constant and can
    // be placed over the probe pipeline that is about to run. One pass per build key column; no
    // execution. See [`runtime_filter`] for which joins qualify and why it cannot regress.
    cache.filters = runtime_filter::plan_filters(plan, sources, &cache);
    Ok(Arc::new(cache))
}

fn collect_builds(
    plan: &RelOp,
    sources: &[Vec<RecordBatch>],
    cache: &mut BuildCache,
    meter: Option<&Meter>,
    budget: usize,
    workers: usize,
    admission: Admission,
) -> Result<(), InterpError> {
    if let RelOp::HashJoin {
        left,
        right,
        right_keys,
        join_type,
        ..
    } = plan
    {
        // Only the probe spine draws on *this* cache. The build side is executed below as one
        // self-contained unit, which prepares whatever joins it holds itself, so descending into
        // it here would build them twice.
        collect_builds(left, sources, cache, meter, budget, workers, admission)?;
        // Shard the build side across the workers, exactly as the probe side is sharded. This
        // was the streaming executor's worst asymmetry: the probe ran on every core while the
        // build — the *whole* other relation — ran on one. It is hashed into a table either way,
        // so single-threading it bought no memory and cost the entire build serially. TPC-H q4
        // (`orders SEMI lineitem`) is the shape that exposes it: a semi join's build is always
        // the right input (it is not commutative, so Kyber cannot swap it), so the 3.8M-row side
        // is built and probed by 57k rows — 279 ms streaming vs 45 ms materializing. Recursion
        // terminates because each build subtree is strictly smaller than the plan.
        // Never hands off: a build side is prepared *for* a decision the caller has not made yet,
        // so declining here would abort the plan before the fact that decides it exists.
        let batches = parallel::run(right, sources, workers, meter, budget, false, None)?;
        if let Ok(side) = ops::materialize(&batches) {
            let probe = make_probe(&side, right_keys, *join_type, admission)?;
            cache.insert(node_key(plan), Arc::new(JoinBuild { side, probe }));
            // After the insert, not before: the check is on what is *resident*, and this side
            // is resident now. Declining before building it would be the thing the comment
            // above rules out — refusing a plan on a fact that does not exist yet.
            cache.check_total(budget)?;
        }
        return Ok(());
    }
    for child in plan.children() {
        collect_builds(child, sources, cache, meter, budget, workers, admission)?;
    }
    Ok(())
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
        .map(|b| b.iter().map(|x| x.num_rows()).sum())
        .unwrap_or(0)
}

/// How far past its cache-sized ceiling a build may go and still be worth probing flat when the
/// *probe* is the smaller side. ~16.8M rows is a ~270 MB table: past that, building it flat is a
/// pathology on its own terms rather than a cache trade — TPC-H sf10 q18 builds 60M rows (~1 GB)
/// and admitting it cost 378 -> 559 ms.
const SMALL_PROBE_BUILD_CAP: usize = 8;

/// How much larger the probe must be than the build before a flat probe past cache pays.
///
/// The flat probe pays about one cache miss per probe row; what it buys is the probe spine
/// running on every core instead of one. So it wins at the *extremes* of the size ratio and
/// loses in the middle, where the radix path's partition pass is cheap against both sides. Six
/// is fitted to the measurements below rather than derived, and the band it sits in is narrow:
/// TPC-H sf10 q9 wins at 7.5x and q21 loses at 4.0x.
const PROBE_DOMINANCE: usize = 6;

/// Whether a build above the ceiling may still take a per-morsel probe.
///
/// Declining is not free: a spine join with no per-morsel probe makes the whole probe spine
/// un-shardable (`parallel::spine_is_shardable`), and a multi-join plan cannot hand off to the
/// materializing executor either (`unshardable_join_reason` declines), so what the ceiling
/// selects on those plans is **the sequential streaming pipeline** — 11-18 cores of 96 on TPC-H
/// sf10 against DuckDB's 30-57, and the whole reason Batcher won sf1 and lost sf10 on every
/// multi-way join.
///
/// The conditions are **fitted to measurement, not derived**, and are stated that way on
/// purpose. Each has a mechanism, and each was measured in-process with the arms alternating by
/// round (seven rounds, best of two, TPC-H sf10):
///
/// * **`Semi` / `Anti` are never admitted.** Their build is the side being *tested against*
///   rather than the side being emitted, and a flat table over it is the whole cost of the
///   join. q4 builds 37.9M rows for a `Semi` and q22 15M for an `Anti`: admitting them cost
///   **146 -> 305 ms** and **43 -> 214 ms**. q13's `Left` join over a near-identical 14.8M-row
///   build and the same 1.5M probe *gains* 515 -> 309, which is what rules out build size as
///   the explanation.
/// * **A probe that dominates the build** ([`PROBE_DOMINANCE`]) — the misses are bounded by the
///   build while the serial time saved scales with the probe. q5 probes 60M against 2.28M
///   (26x): **350 -> 177 ms**. q9, at 7.5x: **541 -> 449**.
/// * **A probe smaller than the build**, up to [`SMALL_PROBE_BUILD_CAP`] — total misses are
///   bounded by the small probe. q13 probes 1.5M against 14.8M: **515 -> 309 ms**.
/// * **Never where the plan could hand off instead.** One hash join and a probe bigger than the
///   build is the case `unshardable_join_reason` gives to the materializing executor, which
///   spreads the probe across every core *and* keeps the cache-resident radix join. Pinned by
///   `stream_oracle::a_large_probe_against_a_huge_build_is_handed_back_only_when_the_caller_asks`.
///
/// Below the ceiling this admits everything, exactly as before.
#[derive(Clone, Copy)]
struct Admission {
    /// Rows in the relation the probe spine is sharded over — the probe side's upper bound.
    driving: usize,
    /// Whether declining still leaves this plan a parallel path (the materializing hand-off).
    can_hand_off: bool,
}

impl Admission {
    fn admits(self, build_rows: usize, join_type: JoinType) -> bool {
        let ceiling = bc_runtime::join::RADIX_MIN_BUILD_ROWS_BROADCAST;
        if build_rows <= ceiling {
            return true; // unchanged: this is the shape the ceiling was drawn for
        }
        if matches!(join_type, JoinType::Semi | JoinType::Anti) {
            return false;
        }
        if self.can_hand_off && self.driving > build_rows {
            return false;
        }
        let probe_dominates = self.driving >= build_rows.saturating_mul(PROBE_DOMINANCE);
        let small_probe = self.driving < build_rows
            && build_rows <= ceiling.saturating_mul(SMALL_PROBE_BUILD_CAP);
        probe_dominates || small_probe
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
