//! Which executor a plan runs on, and the two different affordability tests behind that.
//!
//! Split out of `lib.rs` because the routing decision is a concern of its own — it reads the
//! plan, the bound sources and the memory envelope, and answers one boolean — and because
//! `lib.rs` is `bc-py`'s FFI assembly point, which the size limit holds at 800 code lines.
//! Nothing here is FFI: it is pure arithmetic over `RelOp` and Arrow batches, which is also
//! what makes it testable without a Python interpreter.

use arrow::array::{Array, RecordBatch};

use crate::EngineConfig;

/// The bytes a batch's rows actually occupy, not the bytes its buffers were allocated with.
///
/// `get_array_memory_size` reports the *capacity* of every buffer an array points into, and a
/// sliced array points into its parent's. The parquet reader decodes a row group once and hands
/// the engine morsel-sized slices of it, so every slice reports the whole row group: measured on
/// sf10 `lineitem`, 17,092 two-column batches holding **0.89 GiB** of rows reported **15.71 GiB**,
/// a 17.6x over-count. `materialize_fits` multiplies that figure by eight before comparing it to
/// the envelope, so a 0.89 GiB input asked for 126 GiB of headroom against a 20 GiB budget and the
/// materializing route was refused on every scan-sourced plan.
///
/// What that cost is not marginal. On a join-free `GROUP BY l_orderkey` over the same 60M rows —
/// the aggregate that is 99% of TPC-H q18, the largest single-node gap against DuckDB — the
/// refusal is **4,660 ms streaming against 774 ms materializing**, measured in both orders. Kyber
/// had already set `prefer_materializing_aggregate`, and the engine's own shape guard had already
/// admitted the plan; this figure was the only thing saying no.
///
/// `get_slice_memory_size` is arrow's slice-aware counterpart and answers the question the guard
/// is actually asking. It errors only on a shape whose slice size cannot be computed, where the
/// capacity figure is the right conservative fallback — it can only over-state, which keeps the
/// bounded streaming path.
pub(crate) fn batch_footprint(batch: &RecordBatch) -> usize {
    batch
        .columns()
        .iter()
        .map(|c| {
            let data = c.to_data();
            data.get_slice_memory_size()
                .unwrap_or_else(|_| c.get_array_memory_size())
        })
        .sum()
}

/// The rows-occupied byte total over the sources this plan actually scans.
///
/// The capacity-based counterpart lives at the call site: the two differ by 17.6x on a scan, and
/// which one is right depends on which question is being asked, so both are kept.
pub(crate) fn scanned_rows_bytes(
    sources: &[Vec<RecordBatch>],
    scanned: &std::collections::HashSet<usize>,
) -> usize {
    sources
        .iter()
        .enumerate()
        .filter(|(i, _)| scanned.contains(i))
        .flat_map(|(_, relation)| relation.iter())
        .map(batch_footprint)
        .sum()
}

/// Whether the materializing executor is both safe and faster for this plan.
///
/// **The two reasons do not get the same affordability test, and the difference is measured.**
///
/// The first — a plan streaming cannot shard — materializes *join intermediates*, which are not
/// bounded by the input at all, and its `x8` headroom was calibrated against
/// `get_array_memory_size`. Feeding it the true (17.6x smaller) footprint makes it far more
/// permissive than it was ever tuned to be, and on a 30 GiB box that is not a theoretical risk:
/// at TPC-H sf10 it took **q17, q18, q20 and q21 from completing to SIGKILL**, all four
/// join-heavy. So it keeps reading the capacity figure, which is conservative by construction.
///
/// The second — Kyber's grouped-aggregate verdict — is **join-free by construction** on both
/// sides (`_prefers_materializing_aggregate` returns false for any plan containing a `Join`, and
/// `materializing_aggregate_is_faster` requires `!has_join`), so what it materializes is the
/// input and one hash table over it, and the input's true size is exactly the right question.
/// Measured on the 15M-group aggregate that is 99% of q18: **4,660 ms against 736 ms**, unlocked
/// by nothing but reading the size correctly.
pub(crate) fn materialize_is_safe_and_faster(
    plan: &bc_ir::RelOp,
    prefer_aggregate: bool,
    materialize_fits: bool,
    aggregate_materialize_fits: bool,
) -> bool {
    (!bc_interp::streaming_parallelizes(plan) && materialize_fits)
        || (prefer_aggregate
            && bc_interp::materializing_aggregate_is_faster(plan)
            && aggregate_materialize_fits)
}

/// Print the route's inputs when `BATCHER_DEBUG_ROUTE` is set.
///
/// The route is decided from five independent inputs and reported as one boolean, so a plan that
/// streams when every input says it should not is indistinguishable from one that was never
/// asked. Printing them costs an env lookup per query and is the difference between measuring the
/// decision and inferring it. Same shape as `agg_par`'s `BATCHER_DEBUG_AGGSPLIT`.
pub(crate) fn trace(
    plan: &bc_ir::RelOp,
    streaming: bool,
    cfg_streaming: bool,
    prefer_aggregate: bool,
    aggregate_materialize_fits: bool,
    src_rows_bytes: usize,
    budget: usize,
) {
    if std::env::var("BATCHER_DEBUG_ROUTE").is_err() {
        return;
    }
    eprintln!(
        "ROUTE streaming={} cfg_streaming={} parallelizes={} prefer={} shape_ok={} \
         fits={} src_bytes={} budget={}",
        streaming,
        cfg_streaming,
        bc_interp::streaming_parallelizes(plan),
        prefer_aggregate,
        bc_interp::materializing_aggregate_is_faster(plan),
        aggregate_materialize_fits,
        src_rows_bytes,
        budget,
    );
}

/// The allocated-capacity byte total over the sources this plan actually scans.
///
/// The conservative counterpart to [`scanned_rows_bytes`], and the one the join-intermediate
/// test keeps reading — its `x8` headroom was calibrated against this figure, and feeding it the
/// true footprint took four join-heavy sf10 queries from completing to SIGKILL.
///
/// **The sources this plan scans, not every source bound to the session**
/// (`RelOp::scanned_source_ids`). Judged by the catalog, a session holding the 24 TPC-DS tables
/// made every query look like 1.76 GB, which the `x8` turns into 14.1 GB against a 7.73 GB
/// envelope — so `materialize_fits` was false for *every* TPC-DS query at every size and what it
/// gates, mostly Kyber's grouped-aggregate verdict, could not fire at all. Suite geomeans against
/// the three rounds before it (0.840/0.859/0.879 and 1.184/1.201/1.189): **TPC-H 0.826,
/// TPC-DS 1.157**, each below every one of them.
pub(crate) fn scanned_capacity_bytes(
    sources: &[Vec<RecordBatch>],
    scanned: &std::collections::HashSet<usize>,
) -> usize {
    sources
        .iter()
        .enumerate()
        .filter(|(i, _)| scanned.contains(i))
        .flat_map(|(_, relation)| relation.iter())
        .map(|b| b.get_array_memory_size())
        .sum()
}

/// Whether this query runs on the streaming executor. `true` by default.
///
/// Streaming pulls morsels through the linear runs and materializes only at breakers, so its peak
/// memory is a constant rather than the sum of every operator's output — and on the shapes where
/// that matters it is also *faster*, because the copies it stops making were not free.
///
/// **The two executors do not dominate one another, and that is why this is not a simple swap.**
/// Streaming bounds the *intermediates* but its breakers fold in memory; the materializing
/// executor has unbounded intermediates but breakers that spill out of core. A plan whose
/// aggregate state exceeds the envelope is one the materializing executor survives and this one
/// would OOM on. So the streaming breakers check their state against `memory_budget_bytes` and
/// return `MemoryBudgetExceeded` instead of dying — and `execute_plan` catches exactly that and
/// re-runs on the executor that can spill. Streaming takes the queries it fits (the
/// overwhelming majority, and every one whose intermediates were the problem) and gives way on
/// the ones it does not, rather than quietly turning a spill into a crash.
///
/// Set `streaming = false` to force the materializing executor — a bisecting escape hatch, not a
/// tuning knob.
pub(crate) fn use_streaming(cfg: &EngineConfig) -> bool {
    cfg.streaming
}
