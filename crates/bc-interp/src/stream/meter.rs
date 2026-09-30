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
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::time::Instant;

use arrow::array::RecordBatch;
use bc_ir::RelOp;

use crate::metrics::{ExecMetrics, OpMetric};

/// One operator's running counters. Atomics because every rayon worker running a shard of the
/// pipeline increments the *same* operator's counters.
struct Counters {
    /// A runtime join filter reduced this operator's input: see `ExecMetrics::runtime_filtered`.
    runtime_filtered: AtomicBool,
    rows_in: AtomicU64,
    rows_build: AtomicU64,
    rows_out: AtomicU64,
    elapsed_ns: AtomicU64,
    /// A high-water mark, not a sum: peak is the most this operator ever held at once.
    peak_bytes: AtomicU64,
    result_bytes: AtomicU64,
    /// Nanoseconds after the meter's epoch at which this operator's *first* morsel began, across
    /// every worker (a running `fetch_min`). Starts at `u64::MAX` so the first sample wins.
    span_start_ns: AtomicU64,
    /// Nanoseconds after the meter's epoch at which this operator's *last* morsel ended, across
    /// every worker (a running `fetch_max`).
    span_end_ns: AtomicU64,
    /// Which expression backend ran this operator, as an index into [`BACKENDS`]. Written once
    /// by an operator that actually compiles something; the default is `"interp"`, which is
    /// what every operator on this tier that does not JIT genuinely runs on.
    backend: AtomicU8,
}

/// The backend tags, indexed by the code stored in [`Counters::backend`]. Index 0 is the
/// default so an operator that never reports one reads as `"interp"`.
const BACKENDS: [&str; 3] = ["interp", "jit", "interp+jit"];

fn backend_code(tag: &str) -> u8 {
    BACKENDS.iter().position(|t| *t == tag).unwrap_or(0) as u8
}

impl Default for Counters {
    fn default() -> Self {
        Self {
            runtime_filtered: AtomicBool::new(false),
            rows_in: AtomicU64::new(0),
            rows_build: AtomicU64::new(0),
            rows_out: AtomicU64::new(0),
            elapsed_ns: AtomicU64::new(0),
            peak_bytes: AtomicU64::new(0),
            result_bytes: AtomicU64::new(0),
            // The identity for `fetch_min`: any real timestamp replaces it. `0` would make every
            // operator appear to have started at the query's first instant.
            span_start_ns: AtomicU64::new(u64::MAX),
            span_end_ns: AtomicU64::new(0),
            backend: AtomicU8::new(0),
        }
    }
}

/// Per-operator counters for one plan, indexed by the pre-order `op_id` the control plane uses.
pub(crate) struct Meter {
    kinds: Vec<&'static str>,
    counters: Vec<Counters>,
    /// Plan-node address → `op_id`. The plan is borrowed and immobile for the whole execution,
    /// so its nodes' addresses are stable identities.
    ids: HashMap<usize, u32>,
    threads: u32,
    /// The instant the meter was created — the origin every operator's span is measured from.
    /// Spans need a *shared* origin to be comparable across workers, and an `Instant` per
    /// operator would give each its own.
    epoch: Instant,
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
            epoch: Instant::now(),
        }
    }

    /// Widen an operator's wall span to cover a unit of work that just finished after
    /// `elapsed_ns` of transform time.
    ///
    /// The span is what makes an occupancy figure possible on this tier. `elapsed_ns` alone is
    /// *busy* time summed over every worker and every morsel; dividing it by itself yields
    /// `1/threads` for every operator of every query, which is the constant this tier used to
    /// report as CPU utilization. Dividing it by the wall interval the operator actually
    /// occupied, times the worker count, yields the real figure.
    ///
    /// The end is read now and the start derived by subtraction, because the callers already
    /// time their own transforms and threading a start instant through every one of them would
    /// buy nothing: `Instant::now()` is a vDSO read, so the cost here is nanoseconds per
    /// morsel against transforms measured in microseconds and up.
    fn mark_span(&self, op: u32, elapsed_ns: u64) {
        let c = &self.counters[op as usize];
        let end = self.epoch.elapsed().as_nanos() as u64;
        c.span_start_ns
            .fetch_min(end.saturating_sub(elapsed_ns), Ordering::Relaxed);
        c.span_end_ns.fetch_max(end, Ordering::Relaxed);
    }

    /// This node's `op_id`.
    /// Mark `plan`'s node as fed rows a runtime join filter reduced. A node this meter does not
    /// number is ignored.
    pub(crate) fn mark_runtime_filtered(&self, plan: &RelOp) {
        if let Some(&id) = self.ids.get(&(plan as *const RelOp as usize)) {
            self.counters[id as usize]
                .runtime_filtered
                .store(true, Ordering::Relaxed);
        }
    }

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
        self.mark_span(op, elapsed_ns);
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

    /// Record which expression backend ran an operator.
    ///
    /// This tier reported a hardcoded `"interp"` for every operator, which was right for filter
    /// and project — they genuinely stay on the interpreter here, a measured choice documented
    /// in `stream::build_with` — and wrong for the aggregate, which compiles its computed group
    /// keys and inputs through `ops::compile_agg`. So `ds.stats()` showed `interp` on TPC-H q1,
    /// whose `sum(l_extendedprice * (1 - l_discount))` is compiled, and the one column a user
    /// profiling a query would read to see whether Tier-1 fired could never say yes.
    ///
    /// `store`, not a running combine: the tag is a property of the compiled plan, so every
    /// worker and every fold round reports the same value.
    pub(crate) fn note_backend(&self, op: u32, tag: &str) {
        self.counters[op as usize]
            .backend
            .store(backend_code(tag), Ordering::Relaxed);
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
        self.mark_span(op, elapsed_ns);
    }

    /// The metrics document, in pre-order — the order every other executor emits.
    ///
    /// An operator that never ran (a subtree the limit short-circuited, say) is omitted rather
    /// than reported as a zero-row operator, because "did not run" and "ran and produced nothing"
    /// are different facts and the learned cardinality model must not confuse them.
    /// Fold a sub-execution's metrics into this meter, for a subtree of the metered plan that
    /// ran on another executor.
    ///
    /// A join's build side can run on the materializing executor
    /// (`builds::build_materializes_faster`), which numbers its own operators from 0 at the
    /// subtree's root. The subtree is contiguous in this meter's pre-order, so each id is offset
    /// by the root's; a metric whose kind disagrees with the node at that id is dropped rather
    /// than filed under the wrong operator. Without this every build side that took that path
    /// reported nothing: TPC-H q18's `HAVING sum(l_quantity) > 300` subquery, which keeps 57 of
    /// 1.5M orders, stayed at its default one-third estimate on every run.
    pub(crate) fn absorb(&self, root: &RelOp, sub: &ExecMetrics) {
        let Some(&base) = self.ids.get(&(root as *const RelOp as usize)) else {
            return;
        };
        for op in &sub.ops {
            let id = (base + op.op_id) as usize;
            let (Some(c), Some(kind)) = (self.counters.get(id), self.kinds.get(id)) else {
                continue;
            };
            if *kind != op.kind {
                continue;
            }
            c.rows_in.fetch_add(op.rows_in, Ordering::Relaxed);
            c.rows_build.fetch_add(op.rows_build, Ordering::Relaxed);
            c.rows_out.fetch_add(op.rows_out, Ordering::Relaxed);
            c.elapsed_ns.fetch_add(op.elapsed_ns, Ordering::Relaxed);
            c.peak_bytes.fetch_max(op.peak_bytes, Ordering::Relaxed);
            c.result_bytes.fetch_max(op.result_bytes, Ordering::Relaxed);
        }
    }

    /// [`Self::finish`], with each metric's `op_id` translated through `original` (indexed by
    /// this meter's own id) — for a plan executed in a reordered form but reported against the
    /// numbering of the plan it was built from.
    pub(crate) fn finish_renumbered(self, original: &[u32]) -> ExecMetrics {
        let mut m = self.finish();
        for op in m.ops.iter_mut() {
            if let Some(&id) = original.get(op.op_id as usize) {
                op.op_id = id;
            }
        }
        for op_id in m.runtime_filtered.iter_mut() {
            if let Some(&id) = original.get(*op_id as usize) {
                *op_id = id;
            }
        }
        m.ops.sort_by_key(|op| op.op_id);
        m
    }

    pub(crate) fn finish(self) -> ExecMetrics {
        let mut m = ExecMetrics::default();
        for (id, c) in self.counters.iter().enumerate() {
            let elapsed_ns = c.elapsed_ns.load(Ordering::Relaxed);
            let rows_out = c.rows_out.load(Ordering::Relaxed);
            let rows_in = c.rows_in.load(Ordering::Relaxed);
            if elapsed_ns == 0 && rows_in == 0 && rows_out == 0 {
                continue;
            }
            if c.runtime_filtered.load(Ordering::Relaxed) {
                m.runtime_filtered.push(id as u32);
            }
            // The interval from the first worker entering this operator to the last leaving it.
            // `span_start_ns` still holding its `u64::MAX` sentinel means no unit of work was
            // ever marked, so there is no interval and `0` (unmeasured) is the honest report.
            let start = c.span_start_ns.load(Ordering::Relaxed);
            let end = c.span_end_ns.load(Ordering::Relaxed);
            let wall_span_ns = if start == u64::MAX {
                0
            } else {
                end.saturating_sub(start)
            };
            m.record(OpMetric {
                op_id: id as u32,
                kind: self.kinds[id],
                rows_in,
                rows_build: c.rows_build.load(Ordering::Relaxed),
                rows_out,
                elapsed_ns,
                wall_span_ns,
                // The numerator of this tier's utilization: transform time summed over every
                // morsel and every worker that ran the operator. It is *busy* time rather than
                // CPU time from `getrusage` — a worker inside a transform is running, but for an
                // I/O-bound transform some of that wall time is spent waiting rather than
                // computing, so this reads slightly high on a scan and is exact on a compute
                // kernel. That is an approximation with a known direction, which is a different
                // thing from the constant it replaces.
                //
                // What it replaces: this field was `0`, and before that it was `elapsed_ns`.
                // Reporting `elapsed_ns` was not a conservative approximation but a fabricated
                // constant — the consumer divides `cpu_ns` by `elapsed_ns * threads`
                // (`plan.feedback.cpu_utilization`), so handing it the same number twice yields
                // exactly `1 / threads` for *every* operator of *every* query: a hardcoded 6.25%
                // on a 16-core box. `explain(analyze=True)` printed that as "CPU idle, not
                // CPU-limited" on queries measured at 8-10x parallelism, confidently backwards,
                // and the learned CPU-share model was fed the same constant. Zeroing it stopped
                // the corruption; `wall_span_ns` is what makes a real figure possible, by giving
                // the division an interval to divide by instead of the work itself.
                cpu_ns: if wall_span_ns > 0 { elapsed_ns } else { 0 },
                threads: self.threads,
                peak_bytes: c.peak_bytes.load(Ordering::Relaxed),
                result_bytes: c.result_bytes.load(Ordering::Relaxed),
                spilled: false,
                spill_bytes: 0,
                peak_rss_bytes: 0,
                backend: BACKENDS[c.backend.load(Ordering::Relaxed) as usize % BACKENDS.len()],
                // Unmeasured, for the same reason `cpu_ns` is: the OS counters are
                // process-wide, and attributing them to one operator is only sound when that
                // operator owns an exclusive wall interval. In this tier operators interleave
                // across a pipeline and none does, so any per-operator split here would be a
                // fabricated number rather than an approximate one. The control plane reads
                // zeros as "unmeasured" and keeps its prior.
                hw: Default::default(),
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
        RelOp::RangeJoin { .. } => "range_join",
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use bc_expr::{BinaryOp, Expr, Literal};
    use bc_ir::{ProjectionItem, RelOp};

    use super::Meter;
    use crate::par::{execute_parallel_with_metrics, ExecOptions};

    fn batch() -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)])),
            vec![Arc::new(Int64Array::from((0..100).collect::<Vec<i64>>()))],
        )
        .unwrap()
    }

    fn filtered() -> RelOp {
        RelOp::Filter {
            input: Box::new(RelOp::Scan { source_id: 0 }),
            predicate: Expr::Binary {
                op: BinaryOp::Lt,
                left: Box::new(Expr::Col { name: "x".into() }),
                right: Box::new(Expr::Lit {
                    value: Literal::Int(10),
                }),
            },
        }
    }

    /// A subtree run on another executor numbers its operators from 0 at its own root; folded
    /// into the whole plan's meter they must land at that subtree's ids, not at 0 and 1.
    #[test]
    fn an_absorbed_subtree_is_filed_under_its_ids_in_the_whole_plan() {
        let plan = RelOp::Project {
            input: Box::new(filtered()),
            exprs: vec![ProjectionItem {
                expr: Expr::Col { name: "x".into() },
                alias: "x".into(),
            }],
        };
        let RelOp::Project { input: sub, .. } = &plan else {
            unreachable!()
        };
        let sources = vec![vec![batch()]];
        let (_out, metrics) =
            execute_parallel_with_metrics(sub, &sources, &ExecOptions::default()).unwrap();
        let meter = Meter::new(&plan, 1);
        meter.absorb(sub, &metrics);
        let ops = meter.finish().ops;
        let filter = ops
            .iter()
            .find(|o| o.kind == "filter")
            .expect("filter metered");
        let scan = ops.iter().find(|o| o.kind == "scan").expect("scan metered");
        assert_eq!((filter.op_id, filter.rows_out), (1, 10));
        assert_eq!((scan.op_id, scan.rows_out), (2, 100));
        assert!(
            ops.iter().all(|o| o.kind != "project"),
            "the project never ran here"
        );
    }

    /// Metrics whose numbering does not line up with the meter's are dropped, never misfiled.
    #[test]
    fn a_metric_under_the_wrong_kind_is_dropped() {
        let sources = vec![vec![batch()]];
        let (_out, metrics) =
            execute_parallel_with_metrics(&filtered(), &sources, &ExecOptions::default()).unwrap();
        // A meter over a bare scan: id 0 is a scan, so the filter's metric (id 0) mismatches.
        let scan_only = RelOp::Scan { source_id: 0 };
        let meter = Meter::new(&scan_only, 1);
        meter.absorb(&scan_only, &metrics);
        let ops = meter.finish().ops;
        assert!(ops.iter().all(|o| o.kind == "scan"));
        assert!(ops.iter().all(|o| o.rows_out != 10));
    }
}
