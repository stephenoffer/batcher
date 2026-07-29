//! Per-operator execution metrics — the measure half of the adaptive loop.
//!
//! The interpreter is the one place that *is* the operator walk, so it is where
//! runtime facts are measured: how many rows each operator consumed and produced,
//! how long its own work took, how much memory its result held, whether it spilled,
//! and which backend (interpreter vs JIT) ran it. These are a pure **side-channel** —
//! collecting them never changes a result batch, so the seq == par == JIT oracle is
//! unaffected. The control plane (`core`) transcribes them into `OperatorFeedback`
//! and Kyber calibrates its cost model from them on the next run.
//!
//! Operators are identified by a pre-order DFS index (`op_id`): the root is 0, then
//! its children left-to-right, recursively. The Python control plane numbers the
//! optimized plan the same way (`kyber.annotate.annotate_ops` over `plan.walk()`),
//! so an `op_id` measured here lines up with the operator the planner annotated.

use std::time::Instant;

use serde::Serialize;

use crate::rusage::{self, ResourceSample};

/// The hardware an operator consumed, beyond time and rows.
///
/// Separated from the timing fields because these answer a different question — not "how long
/// did it take" but "what did it cost the machine, and was the machine actually available".
/// Flattened into the metrics JSON, so the control plane reads them as ordinary top-level
/// keys alongside `cpu_ns`.
///
/// Every field is `0` when the platform cannot report it, and `0` means **unmeasured**, not
/// zero. A control-plane consumer must keep its prior on a zero rather than concluding the
/// operator faulted no pages or read no disk.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct HwCounters {
    /// Page faults served from memory during this operator — first touch of newly committed
    /// pages. Multiplied by the page size it estimates how much memory the operator actually
    /// *materialized*, which is the figure `peak_bytes` models and this one measures.
    pub minor_faults: u64,
    /// Page faults that required disk I/O. Non-trivial counts mean the operator was waiting
    /// on storage for memory it believed it held — the machine is paging against the query,
    /// and no amount of extra parallelism will help. Invisible in wall time, which shows only
    /// the symptom.
    pub major_faults: u64,
    /// Times the operator gave up the CPU to wait on something. High against low CPU time is
    /// the signature of an I/O- or lock-bound operator, which looks identical to an
    /// under-parallelized one in a utilization figure alone.
    pub vol_ctx_switches: u64,
    /// Times the scheduler took a CPU away from this operator. The per-operator measurement
    /// of core contention: it is nonzero exactly when something else on the box wanted the
    /// cores this process was told it had. Divided by wall time it gives a preemption rate
    /// the control plane gates fan-out on.
    pub invol_ctx_switches: u64,
    /// Bytes this operator actually pulled from a block device, excluding page-cache hits.
    /// A warm scan and a cold scan issue identical syscalls and differ by two orders of
    /// magnitude in cost; calibrating an I/O coefficient across both without this field
    /// learns a number true of neither.
    pub io_read_bytes: u64,
    /// Bytes this operator sent to a block device, spill writes included. The measured
    /// counterpart to `spill_bytes`, which counts the logical volume an operator *decided* to
    /// route to disk rather than what reached it.
    pub io_write_bytes: u64,
}

impl HwCounters {
    /// These counters split evenly across `n` fused stages.
    ///
    /// A fused pipeline runs several logical operators in one pass, so the OS counters cover
    /// all of them together and cannot be attributed exactly. Splitting evenly is the same
    /// convention the wall-time and CPU-time fields already use at those sites, and it keeps
    /// the per-kind sums the control plane calibrates from correct even though no individual
    /// stage's figure is exact. `n == 0` is treated as 1 rather than dividing by zero.
    pub(crate) fn split(self, n: u64) -> Self {
        let n = n.max(1);
        Self {
            minor_faults: self.minor_faults / n,
            major_faults: self.major_faults / n,
            vol_ctx_switches: self.vol_ctx_switches / n,
            invol_ctx_switches: self.invol_ctx_switches / n,
            io_read_bytes: self.io_read_bytes / n,
            io_write_bytes: self.io_write_bytes / n,
        }
    }

    /// The hardware deltas carried by a `ResourceSample` difference.
    fn from_delta(delta: &ResourceSample) -> Self {
        Self {
            minor_faults: delta.minor_faults,
            major_faults: delta.major_faults,
            vol_ctx_switches: delta.vol_ctx_switches,
            invol_ctx_switches: delta.invol_ctx_switches,
            io_read_bytes: delta.io_read_bytes,
            io_write_bytes: delta.io_write_bytes,
        }
    }
}

/// One operator's measured execution facts.
#[derive(Debug, Clone, Serialize)]
pub struct OpMetric {
    /// Pre-order DFS index of this operator in the plan (matches Python numbering).
    pub op_id: u32,
    /// Operator tag — the same `snake_case` name as the JSON IR `op` tag
    /// (`scan`, `filter`, `aggregate`, `hash_join`, ...). Kyber buckets cost
    /// calibration by this.
    pub kind: &'static str,
    /// Rows fed into this operator. For a single-input operator this is its child's
    /// output; for a scan it equals `rows_out`. For a **join** it is the *probe* side
    /// only (the left input) — not the sum of both sides. The probe rows are what the
    /// per-row probe cost scales with, and `rows_out / rows_in` is then the join's
    /// fan-out. Summing both sides made `selectivity` mean nothing and conflated the
    /// asymmetric build and probe costs into one calibrated coefficient.
    pub rows_in: u64,
    /// Rows fed into a join's *build* side (the right input, over which the hash table is
    /// constructed). `0` for every non-join operator. Separating it from `rows_in` is what
    /// gives a join a meaningful `selectivity` (fan-out) and lets the cost model calibrate
    /// the asymmetric build and probe coefficients independently. Note that `peak_bytes`
    /// accounts for *both* sides: this is a batch engine, so a join materializes its probe
    /// input too rather than streaming it.
    pub rows_build: u64,
    /// Rows this operator produced.
    pub rows_out: u64,
    /// Wall-clock nanoseconds spent in this operator's *own* work (excludes the
    /// time spent producing its children's inputs).
    pub elapsed_ns: u64,
    /// The wall interval this operator actually occupied — from the first worker entering it to
    /// the last worker leaving — or `0` where the executor does not track one.
    ///
    /// Exists because `elapsed_ns` means two different things on the two tiers. On a
    /// materializing executor an operator runs alone, so its wall interval *is* its elapsed
    /// time and this field stays `0`. On the streaming executor operators interleave, and
    /// `elapsed_ns` is transform time summed over every morsel and every worker — a quantity of
    /// *work*, not an interval. Any utilization figure needs an interval as its denominator, and
    /// dividing the summed work by itself is what made this tier report a hardcoded `1/threads`
    /// for every operator of every query.
    ///
    /// Consumers should divide by this when it is non-zero and by `elapsed_ns` otherwise.
    pub wall_span_ns: u64,
    /// CPU-time nanoseconds (user + system, summed across *all* worker threads)
    /// consumed during this operator's own work. Divided by `elapsed_ns x threads`
    /// it gives the per-core utilization the control plane learns from to size each
    /// task's `num_cpus`. `0` when the platform can't report process CPU time.
    pub cpu_ns: u64,
    /// Worker threads this operator's pool actually ran across (rayon's live count;
    /// `1` for the sequential oracle). The control plane uses this as the exact
    /// denominator for per-core utilization instead of guessing the host core count —
    /// which is wrong under a cgroup CPU quota (a container sees host cores but rayon
    /// sizes to the quota), the common case in a Kubernetes deployment.
    pub threads: u32,
    /// Bytes simultaneously live for this operator — its **peak working set**.
    ///
    /// For a pipeline breaker (aggregate / sort / distinct / window / join) that is the
    /// materialized input it holds *plus* the result it is building, because both exist
    /// at once. For a streaming operator it is the result alone.
    ///
    /// This used to be `batch_bytes(out)` — the operator's *output* size — for every
    /// operator, which is a catastrophic under-count for exactly the cardinality-reducing
    /// breakers that spill: a 60M-row aggregate over 4 groups reported ~0 peak. Carbonite
    /// fits its per-family memory model on this field and drives admission, spill routing,
    /// buffer reservation and per-worker sizing from it, so the under-count systematically
    /// under-provisioned the operators most likely to exhaust memory.
    pub peak_bytes: u64,
    /// Bytes held by this operator's *result* alone (Arrow `get_array_memory_size`).
    /// What `peak_bytes` used to contain. The profiler reports this as `result_bytes`;
    /// nothing sizes memory from it.
    pub result_bytes: u64,
    /// Whether the operator engaged its out-of-core spill path.
    pub spilled: bool,
    /// Logical bytes this operator routed to its spill path (the summed in-memory size of
    /// the batches it wrote to disk), or `0` when it did not spill (or the spill volume was
    /// not measured). A `spilled: bool` cannot distinguish a 1 GB spill from a 100 GB one;
    /// Carbonite sizes spill scratch, disk bandwidth, and partition counts from this
    /// measured magnitude. Best-effort: exact where the operator owns its `SpillStore`,
    /// `0` (unmeasured) for a spill nested inside a helper that does not surface its volume.
    pub spill_bytes: u64,
    /// Growth in the process's **peak resident set** (bytes) during this operator's work —
    /// a measured high-water mark from `getrusage(ru_maxrss)`, complementing the Arrow-size
    /// `peak_bytes` estimate (which cannot see scratch, fragmentation, or off-pool buffers).
    /// `0` when the op set no new high-water or the platform can't report RSS (unmeasured).
    pub peak_rss_bytes: u64,
    /// Which execution backend ran the per-row work: `"interp"`, `"jit"`, or
    /// `"interp+jit"` (some sub-expressions compiled, others fell back).
    pub backend: &'static str,
    /// The hardware this operator consumed: faults, preemption, and real block-device I/O.
    /// Flattened into the JSON, so these arrive as top-level keys the control plane reads
    /// beside `cpu_ns` rather than as a nested object.
    #[serde(flatten)]
    pub hw: HwCounters,
}

/// All per-operator metrics gathered during one plan execution.
///
/// Built up as the interpreter walks the plan; serialized to JSON at the FFI
/// boundary so it can ride back alongside the (still zero-copy) result batches.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ExecMetrics {
    pub ops: Vec<OpMetric>,
}

impl ExecMetrics {
    /// Append one operator's metric.
    pub fn record(&mut self, m: OpMetric) {
        self.ops.push(m);
    }

    /// Serialize to the JSON document the FFI returns to the control plane.
    /// Infallible in practice (the struct is plain data); a serialization error
    /// degrades to an empty-metrics document rather than failing the query.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{\"ops\":[]}".to_string())
    }
}

/// A pre-order operator-id allocator. Each `next()` hands out the id for the
/// operator about to be entered, so parents are numbered before their children.
///
/// `Clone` so a speculative execution (the fused join pipeline, which can only tell whether
/// it applies *after* materializing the build sides) can number its operators off a copy and
/// commit the advanced counter only if it succeeds — a bail leaves the real numbering
/// untouched, so the fallback path assigns exactly the ids it would have.
#[derive(Clone)]
pub(crate) struct IdGen {
    next: u32,
}

impl IdGen {
    pub(crate) fn new() -> Self {
        Self { next: 0 }
    }

    /// An allocator positioned at `next` — the id a subtree starting there would receive.
    ///
    /// With `RelOp::node_count` this lets an executor run a node's children out of order and
    /// still hand each subtree exactly the ids a recursive pre-order walk would (the fused
    /// join pipeline runs its build sides before its probe side).
    pub(crate) fn at(next: u32) -> Self {
        Self { next }
    }

    /// The id this allocator will hand out next.
    pub(crate) fn peek(&self) -> u32 {
        self.next
    }

    /// Allocate the id for the operator being entered (pre-order).
    pub(crate) fn next(&mut self) -> u32 {
        let id = self.next;
        self.next += 1;
        id
    }
}

/// A wall clock paired with a full OS resource snapshot, captured at the start of an
/// operator's own work.
///
/// `Copy` so it threads through the metric helpers exactly as the bare `Instant` it replaces
/// did. Everything but wall time is sampled process-wide, which is sound because the
/// interpreter runs operators one at a time and fully joins each before recording it: the
/// delta over an operator's window is that operator's own consumption across every rayon
/// thread.
#[derive(Clone, Copy)]
pub(crate) struct Stopwatch {
    wall: Instant,
    start: ResourceSample,
}

impl Stopwatch {
    /// Capture the wall clock and every OS counter now (an operator's start).
    pub(crate) fn start() -> Self {
        Self {
            wall: Instant::now(),
            start: rusage::sample(),
        }
    }

    /// Wall-clock nanoseconds since [`start`](Self::start).
    pub(crate) fn elapsed_ns(&self) -> u64 {
        self.wall.elapsed().as_nanos() as u64
    }

    /// The CPU-time delta, **peak-RSS growth**, and hardware counters since
    /// [`start`](Self::start), from **one** coherent snapshot — the reading recorded at an
    /// operator's end. Prefer this over sampling each separately, which would straddle
    /// several instants and cost a syscall apiece.
    ///
    /// `ru_maxrss` is a monotonic high-water mark, so the RSS delta is how much *new* physical
    /// memory this operator's work forced resident — a ground-truth complement to the Arrow-size
    /// `peak_bytes` estimate, which cannot see transient scratch, allocator fragmentation, or
    /// off-pool buffers. Any component is `0` when this op stayed under a prior high-water (or
    /// the platform can't report it): the control plane treats `0` as "unmeasured".
    pub(crate) fn measure(&self) -> (u64, u64, HwCounters) {
        let delta = rusage::sample().since(&self.start);
        (
            delta.cpu_ns,
            delta.peak_rss_bytes,
            HwCounters::from_delta(&delta),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The CPU stopwatch registers measurable CPU time for a busy span (the signal
    /// the adaptive CPU-share loop learns from). On a unix host CPU time advances;
    /// elsewhere it reports 0 and the control plane treats that as "unmeasured".
    #[test]
    fn stopwatch_measures_cpu_and_wall_time() {
        let sw = Stopwatch::start();
        // CPU-bound busy work the optimizer can't elide (black_box the accumulator).
        let mut acc: u64 = 0;
        for i in 0..50_000_000u64 {
            acc = acc.wrapping_add(i).wrapping_mul(2_654_435_761);
        }
        std::hint::black_box(acc);
        assert!(sw.elapsed_ns() > 0, "wall time must advance");
        if cfg!(unix) {
            let (cpu_ns, _, _) = sw.measure();
            assert!(cpu_ns > 0, "a busy span must register CPU time on unix");
        }
    }

    /// `measure` reads every counter from one coherent snapshot — the paired reader the
    /// op-record sites use. CPU advances on a busy span;
    /// RSS growth is `>= 0` (a monotonic high-water delta, saturating to 0).
    #[test]
    fn measure_pairs_a_coherent_snapshot() {
        let sw = Stopwatch::start();
        let mut acc: u64 = 0;
        for i in 0..50_000_000u64 {
            acc = acc.wrapping_add(i).wrapping_mul(2_654_435_761);
        }
        std::hint::black_box(acc);
        let (cpu, _rss, hw) = sw.measure();
        if cfg!(unix) {
            assert!(cpu > 0, "a busy span must register CPU time on unix");
            // A span this long is preempted or yields at least once on any real scheduler,
            // so the switch counters must be wired up rather than left at their default.
            assert!(
                hw.vol_ctx_switches + hw.invol_ctx_switches + hw.minor_faults > 0,
                "the hardware counters must be populated, not defaulted"
            );
        }
        // A later reading of the same stopwatch cannot report less than an earlier one: the
        // underlying counters are monotonic, and a regression here would mean the snapshots
        // are straddling instants rather than being read coherently.
        let (later, _, _) = sw.measure();
        assert!(
            cpu <= later,
            "the earlier snapshot cannot exceed a later one"
        );
    }

    /// The serialized metrics document carries the `cpu_ns` key the control plane
    /// reads (`core.record_exec_metrics`) — a guard on the wire contract.
    #[test]
    fn to_json_includes_cpu_ns() {
        let mut m = ExecMetrics::default();
        m.record(OpMetric {
            op_id: 0,
            kind: "scan",
            rows_in: 1,
            rows_build: 0,
            rows_out: 1,
            elapsed_ns: 10,
            wall_span_ns: 0,
            cpu_ns: 7,
            threads: 4,
            peak_bytes: 0,
            result_bytes: 0,
            spilled: false,
            spill_bytes: 0,
            peak_rss_bytes: 0,
            backend: "interp",
            hw: HwCounters {
                minor_faults: 11,
                major_faults: 2,
                vol_ctx_switches: 5,
                invol_ctx_switches: 3,
                io_read_bytes: 4096,
                io_write_bytes: 8192,
            },
        });
        let json = m.to_json();
        assert!(
            json.contains("\"cpu_ns\":7"),
            "cpu_ns must serialize: {json}"
        );
        // The hardware counters flatten to top-level keys, which is the shape the control
        // plane reads. A nested object here would silently deliver zeros to every consumer.
        for key in [
            "\"minor_faults\":11",
            "\"major_faults\":2",
            "\"vol_ctx_switches\":5",
            "\"invol_ctx_switches\":3",
            "\"io_read_bytes\":4096",
            "\"io_write_bytes\":8192",
        ] {
            assert!(json.contains(key), "{key} must serialize flat: {json}");
        }
        assert!(
            json.contains("\"threads\":4"),
            "threads must serialize: {json}"
        );
        assert!(
            json.contains("\"spill_bytes\":0"),
            "spill_bytes must serialize (the control plane reads it): {json}"
        );
        assert!(
            json.contains("\"peak_rss_bytes\":0"),
            "peak_rss_bytes must serialize (the control plane reads it): {json}"
        );
    }
}
