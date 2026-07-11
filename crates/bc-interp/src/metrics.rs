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
pub(crate) struct IdGen {
    next: u32,
}

impl IdGen {
    pub(crate) fn new() -> Self {
        Self { next: 0 }
    }

    /// Allocate the id for the operator being entered (pre-order).
    pub(crate) fn next(&mut self) -> u32 {
        let id = self.next;
        self.next += 1;
        id
    }
}

/// A paired wall + CPU stopwatch, captured at the start of an operator's own work.
///
/// `Copy` so it threads through the metric helpers exactly as the bare `Instant` it
/// replaces did. `cpu_ns` is sampled against process-wide CPU time (`getrusage`):
/// because the interpreter runs operators one at a time and fully joins each before
/// recording it, the delta over an operator's window is that operator's CPU work
/// across every rayon thread — the numerator of its per-core utilization.
#[derive(Clone, Copy)]
pub(crate) struct Stopwatch {
    wall: Instant,
    cpu_ns_start: u64,
    rss_start: u64,
}

impl Stopwatch {
    /// Capture the wall and CPU clocks now (an operator's start).
    pub(crate) fn start() -> Self {
        let (cpu, rss) = process_rusage();
        Self {
            wall: Instant::now(),
            cpu_ns_start: cpu,
            rss_start: rss,
        }
    }

    /// Wall-clock nanoseconds since [`start`](Self::start).
    pub(crate) fn elapsed_ns(&self) -> u64 {
        self.wall.elapsed().as_nanos() as u64
    }

    /// Process CPU-time nanoseconds consumed since [`start`](Self::start).
    pub(crate) fn cpu_ns(&self) -> u64 {
        process_cpu_ns().saturating_sub(self.cpu_ns_start)
    }

    /// The CPU-time delta and **peak-RSS growth** (bytes) since [`start`](Self::start), from
    /// **one** `getrusage` snapshot — the coherent, single-syscall pairing recorded at an op's
    /// end. Prefer this over sampling CPU and RSS separately, which would straddle two instants
    /// and cost two syscalls.
    ///
    /// `ru_maxrss` is a monotonic high-water mark, so the RSS delta is how much *new* physical
    /// memory this operator's work forced resident — a ground-truth complement to the Arrow-size
    /// `peak_bytes` estimate, which cannot see transient scratch, allocator fragmentation, or
    /// off-pool buffers. Either component is `0` when this op stayed under a prior high-water (or
    /// the platform can't report it): the control plane treats `0` as "unmeasured".
    pub(crate) fn cpu_and_rss(&self) -> (u64, u64) {
        let (cpu, rss) = process_rusage();
        (
            cpu.saturating_sub(self.cpu_ns_start),
            rss.saturating_sub(self.rss_start),
        )
    }
}

/// One coherent `getrusage(RUSAGE_SELF)` snapshot as `(cpu_ns, peak_rss_bytes)`.
///
/// A single syscall reads both counters from the *same* instant, so a paired CPU+RSS
/// sample is internally consistent (two separate `getrusage` calls would straddle any
/// work done between them) and costs one syscall instead of two at every op boundary.
/// `cpu_ns` is process-wide CPU time (user + system, all threads); `peak_rss_bytes` is
/// `ru_maxrss` normalized to bytes. Both are `0` when the platform can't report them —
/// the control plane treats `0` as "unmeasured" and keeps its prior.
#[cfg(unix)]
fn process_rusage() -> (u64, u64) {
    use std::mem::MaybeUninit;

    let mut usage = MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: `getrusage` fully initializes the `rusage` out-param and returns 0 on
    // success; the initialized value is read only on that success path.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return (0, 0);
    }
    let usage = unsafe { usage.assume_init() };
    let tv_ns = |t: &libc::timeval| (t.tv_sec as u64) * 1_000_000_000 + (t.tv_usec as u64) * 1_000;
    let cpu_ns = tv_ns(&usage.ru_utime) + tv_ns(&usage.ru_stime);
    let max_rss = usage.ru_maxrss as u64;
    #[cfg(target_os = "linux")]
    let rss_bytes = max_rss.saturating_mul(1024); // Linux reports KiB
    #[cfg(not(target_os = "linux"))]
    let rss_bytes = max_rss; // the BSDs / macOS already report bytes
    (cpu_ns, rss_bytes)
}

#[cfg(not(unix))]
fn process_rusage() -> (u64, u64) {
    (0, 0)
}

/// Process-wide CPU time (user + system, all threads) in nanoseconds, or `0` when
/// unavailable — the control plane treats `0` as "unmeasured" and keeps its prior.
fn process_cpu_ns() -> u64 {
    process_rusage().0
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
            assert!(
                sw.cpu_ns() > 0,
                "a busy span must register CPU time on unix"
            );
        }
    }

    /// `cpu_and_rss` returns the same CPU delta a lone `cpu_ns()` would, from one
    /// coherent snapshot — the paired reader the op-record sites use. CPU advances on a
    /// busy span; RSS growth is `>= 0` (a monotonic high-water delta, saturating to 0).
    #[test]
    fn cpu_and_rss_pairs_a_coherent_snapshot() {
        let sw = Stopwatch::start();
        let mut acc: u64 = 0;
        for i in 0..50_000_000u64 {
            acc = acc.wrapping_add(i).wrapping_mul(2_654_435_761);
        }
        std::hint::black_box(acc);
        let (cpu, _rss) = sw.cpu_and_rss();
        if cfg!(unix) {
            assert!(cpu > 0, "a busy span must register CPU time on unix");
        }
        // The paired reader must agree with the lone CPU reader to within the tiny drift
        // of the extra work between the two calls — never wildly apart or zero after busy work.
        let lone = sw.cpu_ns();
        assert!(
            cpu <= lone,
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
            cpu_ns: 7,
            threads: 4,
            peak_bytes: 0,
            result_bytes: 0,
            spilled: false,
            spill_bytes: 0,
            peak_rss_bytes: 0,
            backend: "interp",
        });
        let json = m.to_json();
        assert!(
            json.contains("\"cpu_ns\":7"),
            "cpu_ns must serialize: {json}"
        );
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
