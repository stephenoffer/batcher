//! Per-operator metrics for the streaming executor.
//!
//! The metrics are not telemetry — they are load-bearing. Kyber calibrates its cost
//! coefficients from `rows_in` and learns cardinalities from `rows_out`; Carbonite fits its
//! per-family memory model on `peak_bytes` and sizes admission, spill routing and buffer
//! reservation from it. A streaming executor that reported *nothing* would blind both; one that
//! reported *plausible-looking rubbish* would quietly corrupt what they learn, which is worse.
//! So this measures rather than guesses.
//!
//! Two things genuinely change meaning on this tier, and both are changes toward the truth:
//!
//! * **A join's `peak_bytes` no longer includes its probe side.** `OpMetric` documents the old
//!   behaviour as a consequence of the engine's shape — *"this is a batch engine, so a join
//!   materializes its probe input too rather than streaming it"* — and that is precisely the
//!   sentence this executor makes obsolete. The probe is a morsel at a time now, so a streamed
//!   join's peak is its build table plus one morsel. Reporting the old, larger number would make
//!   Carbonite reserve memory for a materialization that no longer happens.
//! * **An aggregate's `peak_bytes` no longer includes its input.** It folds incrementally, so
//!   its peak is its state (bounded by the group count) plus the result.
//!
//! `elapsed_ns` is the time spent *inside* an operator's own transform, summed over every morsel
//! and every worker that ran it — which is the honest reading of "this operator's own work" in a
//! pipelined model, where operators interleave and no wall-clock interval belongs to one alone.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow::array::RecordBatch;
use bc_ir::RelOp;

use crate::metrics::{ExecMetrics, OpMetric};

/// One operator's running counters. Atomics because every rayon worker running a shard of the
/// pipeline increments the *same* operator's counters.
#[derive(Default)]
struct Counters {
    rows_in: AtomicU64,
    rows_build: AtomicU64,
    rows_out: AtomicU64,
    elapsed_ns: AtomicU64,
    /// A high-water mark, not a sum: peak is the most this operator ever held at once.
    peak_bytes: AtomicU64,
    result_bytes: AtomicU64,
}

/// Per-operator counters for one plan, indexed by the pre-order `op_id` the control plane uses.
pub(crate) struct Meter {
    kinds: Vec<&'static str>,
    counters: Vec<Counters>,
    /// Plan-node address → `op_id`. The plan is borrowed and immobile for the whole execution,
    /// so its nodes' addresses are stable identities.
    ids: HashMap<usize, u32>,
    threads: u32,
}

impl Meter {
    /// Number every node pre-order — parents before children, children left to right — which is
    /// exactly the numbering `exec_seq`, `par::exec` and the Python control plane's
    /// `annotate_ops` all use. A metric keyed by a different id is a metric attributed to the
    /// wrong operator.
    pub(crate) fn new(plan: &RelOp, threads: u32) -> Self {
        let mut kinds = Vec::new();
        let mut ids = HashMap::new();
        assign(plan, &mut kinds, &mut ids);
        let counters = (0..kinds.len()).map(|_| Counters::default()).collect();
        Self {
            kinds,
            counters,
            ids,
            threads,
        }
    }

    /// This node's `op_id`.
    pub(crate) fn id(&self, plan: &RelOp) -> u32 {
        self.ids[&(plan as *const RelOp as usize)]
    }

    /// Record one morsel through a pipeline operator: rows in, rows out, and the nanoseconds
    /// its own transform took.
    pub(crate) fn morsel(&self, op: u32, rows_in: u64, out: &RecordBatch, elapsed_ns: u64) {
        let c = &self.counters[op as usize];
        c.rows_in.fetch_add(rows_in, Ordering::Relaxed);
        c.rows_out
            .fetch_add(out.num_rows() as u64, Ordering::Relaxed);
        c.elapsed_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        let bytes = crate::batch_bytes(std::slice::from_ref(out));
        c.result_bytes.fetch_add(bytes, Ordering::Relaxed);
        // A pipeline operator holds one morsel at a time, so its peak *is* the largest morsel it
        // ever produced — not the sum of them, which is the whole point of streaming.
        c.peak_bytes.fetch_max(bytes, Ordering::Relaxed);
    }

    /// A streamed join's build side is hashed **once**, no matter how many morsels probe it, so
    /// its `rows_build` must be recorded once too. Adding it per morsel would multiply the build
    /// cardinality by the morsel count — and Kyber calibrates the join's asymmetric build and
    /// probe coefficients from exactly these two numbers.
    pub(crate) fn record_build_rows_once(&self, op: u32, rows: u64) {
        let c = &self.counters[op as usize];
        // `fetch_max`, not `fetch_add`: idempotent under any number of morsels and any number of
        // workers racing on the same join.
        c.rows_build.fetch_max(rows, Ordering::Relaxed);
    }

    /// Record a breaker: what it consumed, what it held at once, and what it produced.
    pub(crate) fn breaker(
        &self,
        op: u32,
        rows_in: u64,
        rows_build: u64,
        peak_bytes: u64,
        out: &[RecordBatch],
        elapsed_ns: u64,
    ) {
        let c = &self.counters[op as usize];
        c.rows_in.fetch_add(rows_in, Ordering::Relaxed);
        c.rows_build.fetch_add(rows_build, Ordering::Relaxed);
        c.rows_out
            .fetch_add(crate::count_rows(out), Ordering::Relaxed);
        c.elapsed_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        let bytes = crate::batch_bytes(out);
        c.result_bytes.fetch_add(bytes, Ordering::Relaxed);
        c.peak_bytes
            .fetch_max(peak_bytes.max(bytes), Ordering::Relaxed);
    }

    /// The metrics document, in pre-order — the order every other executor emits.
    ///
    /// An operator that never ran (a subtree the limit short-circuited, say) is omitted rather
    /// than reported as a zero-row operator, because "did not run" and "ran and produced nothing"
    /// are different facts and the learned cardinality model must not confuse them.
    pub(crate) fn finish(self) -> ExecMetrics {
        let mut m = ExecMetrics::default();
        for (id, c) in self.counters.iter().enumerate() {
            let elapsed_ns = c.elapsed_ns.load(Ordering::Relaxed);
            let rows_out = c.rows_out.load(Ordering::Relaxed);
            let rows_in = c.rows_in.load(Ordering::Relaxed);
            if elapsed_ns == 0 && rows_in == 0 && rows_out == 0 {
                continue;
            }
            m.record(OpMetric {
                op_id: id as u32,
                kind: self.kinds[id],
                rows_in,
                rows_build: c.rows_build.load(Ordering::Relaxed),
                rows_out,
                elapsed_ns,
                // Summed across workers, which is what a CPU-time reading means; the streaming
                // tier does not sample the OS clock per morsel (it would cost more than the
                // transform), so this is the wall-clock work summed over threads.
                cpu_ns: elapsed_ns,
                threads: self.threads,
                peak_bytes: c.peak_bytes.load(Ordering::Relaxed),
                result_bytes: c.result_bytes.load(Ordering::Relaxed),
                spilled: false,
                spill_bytes: 0,
                peak_rss_bytes: 0,
                backend: "interp",
            });
        }
        m
    }
}

/// The JSON IR `op` tag for a node — the same `snake_case` name Kyber buckets cost calibration
/// by, so a streamed operator's measurements land in the same bucket as a materialized one's.
fn kind_of(plan: &RelOp) -> &'static str {
    match plan {
        RelOp::Scan { .. } => "scan",
        RelOp::Filter { .. } => "filter",
        RelOp::Project { .. } => "project",
        RelOp::Aggregate { .. } => "aggregate",
        RelOp::Sort { .. } => "sort",
        RelOp::Limit { .. } => "limit",
        RelOp::HashJoin { .. } => "hash_join",
        RelOp::AsofJoin { .. } => "asof_join",
        RelOp::Distinct { .. } => "distinct",
        RelOp::Window { .. } => "window",
        RelOp::Union { .. } => "union",
        RelOp::Unnest { .. } => "unnest",
        RelOp::Unpivot { .. } => "unpivot",
        RelOp::RowId { .. } => "row_id",
        RelOp::Sample { .. } => "sample",
    }
}

fn assign(plan: &RelOp, kinds: &mut Vec<&'static str>, ids: &mut HashMap<usize, u32>) {
    let id = kinds.len() as u32;
    kinds.push(kind_of(plan));
    ids.insert(plan as *const RelOp as usize, id);
    for c in plan.children() {
        assign(c, kinds, ids);
    }
}
