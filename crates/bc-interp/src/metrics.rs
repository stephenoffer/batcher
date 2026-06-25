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
//! optimized plan the same way (`kyber.optimizer._annotate_ops` over `plan.walk()`),
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
    /// Rows fed into this operator (sum over child outputs; = `rows_out` for a scan).
    pub rows_in: u64,
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
    /// Bytes held by this operator's result (Arrow `get_array_memory_size`). A
    /// coarse proxy for peak working-set used to calibrate the memory cost axis.
    pub peak_bytes: u64,
    /// Whether the operator engaged its out-of-core spill path.
    pub spilled: bool,
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
}

impl Stopwatch {
    /// Capture the wall and CPU clocks now (an operator's start).
    pub(crate) fn start() -> Self {
        Self {
            wall: Instant::now(),
            cpu_ns_start: process_cpu_ns(),
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
}

/// Process-wide CPU time (user + system, all threads) in nanoseconds, or `0` when
/// unavailable — the control plane treats `0` as "unmeasured" and keeps its prior.
#[cfg(unix)]
fn process_cpu_ns() -> u64 {
    use std::mem::MaybeUninit;

    let mut usage = MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: `getrusage` fully initializes the `rusage` out-param and returns 0 on
    // success; the initialized value is read only on that success path.
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return 0;
    }
    let usage = unsafe { usage.assume_init() };
    let tv_ns = |t: &libc::timeval| (t.tv_sec as u64) * 1_000_000_000 + (t.tv_usec as u64) * 1_000;
    tv_ns(&usage.ru_utime) + tv_ns(&usage.ru_stime)
}

#[cfg(not(unix))]
fn process_cpu_ns() -> u64 {
    0
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

    /// The serialized metrics document carries the `cpu_ns` key the control plane
    /// reads (`core.record_exec_metrics`) — a guard on the wire contract.
    #[test]
    fn to_json_includes_cpu_ns() {
        let mut m = ExecMetrics::default();
        m.record(OpMetric {
            op_id: 0,
            kind: "scan",
            rows_in: 1,
            rows_out: 1,
            elapsed_ns: 10,
            cpu_ns: 7,
            threads: 4,
            peak_bytes: 0,
            spilled: false,
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
    }
}
