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
