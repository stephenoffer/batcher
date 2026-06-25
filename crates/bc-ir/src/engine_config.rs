//! Execution tunables shipped from the Python control plane alongside the plan.
//!
//! `EngineConfig` is the wire-contract mirror of the Rust-relevant slice of
//! Python's `Config` (`ExecutionConfig`). The control plane is the source of truth
//! at runtime — Core serializes the live config and ships it next to the plan JSON —
//! so the data plane no longer bakes morsel size / parallelism into constants and
//! silently drifts from Python. The `Default` here (sourced from the engine's own
//! `bc_arrow` consts) is what a standalone `cargo test` or an absent/empty config
//! falls back to; a Python↔Rust parity test pins the two default sets together.

use std::collections::HashMap;

use serde::Deserialize;

use crate::IrError;

/// The execution knobs the data plane consumes, deserialized from the control
/// plane's JSON. Unknown fields are ignored and missing fields take their default,
/// so adding a Python-side knob never breaks an older engine.
// `Eq` is intentionally omitted: `bloom_fp_rate` is an `f64`, which is only
// `PartialEq`. Every test compares configs with `assert_eq!`, which needs only
// `PartialEq`, so dropping `Eq` is source-compatible.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case", default)]
pub struct EngineConfig {
    /// Rows per morsel — the unit of parallel scheduling (§1.4). Mirrors
    /// `ExecutionConfig.morsel_rows`.
    pub morsel_rows: usize,
    /// Byte budget per morsel — the byte-aware companion to `morsel_rows`. A
    /// morsel is split when it reaches *either* bound, so wide/variable-width
    /// data (large strings, embeddings, blob handles) cannot balloon a morsel's
    /// working set even though its row count is small. Mirrors
    /// `ExecutionConfig.morsel_bytes`. For narrow data the row bound trips first,
    /// so this changes nothing.
    pub morsel_bytes: usize,
    /// Worker threads for the parallel executor; `0` means "all available cores"
    /// (the rayon global pool). Mirrors `ExecutionConfig.parallelism`.
    pub parallelism: usize,
    /// Soft cap (bytes) on a stateful operator's in-memory state before it spills
    /// to disk. `0` (the default) means "unbounded" — the engine stays fully
    /// in-memory, so a small query pays nothing. A positive budget makes the main
    /// `execute_plan` path able to spill (aggregate / distinct / sort / join /
    /// window grace-partition out of core). Derived by the control plane from
    /// `MemoryConfig` (`max_memory_bytes`/`default_total_bytes` × `hard_limit`).
    pub memory_budget_bytes: usize,
    /// Scratch directory for spill files (one Arrow-IPC file per hash partition).
    /// `None`/absent falls back to the OS temp dir. Mirrors `MemoryConfig.spill_dir`.
    pub spill_dir: Option<String>,
    /// Per-operator spill budget (bytes), keyed by the pre-order `op_id` Kyber
    /// assigns in `_annotate_ops` (the same numbering the metrics side-channel uses).
    /// This is the *byte-true, per-operator* envelope (Kyber's `m_max_bytes`) that
    /// the global `memory_budget_bytes` collapses to one number for every operator;
    /// when an operator has an entry here the executor budgets *it* against this
    /// instead of the shared global, so a small operator no longer spills while a
    /// large neighbour assumes the whole budget. JSON object keys are strings
    /// (`{"3": 1048576}`); serde_json parses them back to `u32`. Empty (the default,
    /// e.g. an older control plane or an ad-hoc IR with no `PhysicalOp` DAG) ⇒ every
    /// operator falls back to `memory_budget_bytes`, so behavior is unchanged.
    pub op_budgets: HashMap<u32, usize>,
    /// Fuse runs of linear, per-morsel streaming operators (Filter/Project) into a
    /// single pass over the input's morsels in the parallel executor. A relation-level
    /// no-op (same rows, same order — verified against the sequential oracle); it only
    /// changes morsel boundaries and the number of rayon dispatches. `false` (the
    /// default) keeps the staged operator-at-a-time path. Mirrors
    /// `ExecutionConfig.fuse_linear`.
    pub fuse_linear: bool,
    // --- Performance-threshold knobs (mirror `bc_arrow::RuntimeTuning`) ----------
    // These are performance-only: they change *how* the parallel executor runs an
    // operator, never the relation it produces. Each default equals the historical
    // `const` it replaced, so an absent override is bit-identical to the old engine.
    // Only the parallel hot path (`par.rs`) threads them; the sequential oracle keeps
    // the defaults.
    /// False-positive rate for the hash-join probe-side bloom pre-filter.
    pub bloom_fp_rate: f64,
    /// Build-row floor above which the probe bloom pays for itself.
    pub bloom_min_build_rows: usize,
    /// Window row count above which per-partition sorts run across cores.
    pub window_parallel_row_threshold: usize,
    /// Concatenated-input row count above which `combine` regroups via parallel
    /// hash-radix partitioning.
    pub radix_parallel_threshold: usize,
    /// Maximum runs merged per pass in the external (spilling) sort's k-way merge.
    pub sort_merge_fanin: usize,
    /// A join bucket is "hot" when it exceeds this multiple of the average bucket.
    pub skew_bucket_factor: usize,
    /// Absolute row floor below which a bucket is never treated as skewed.
    pub skew_min_bucket_rows: usize,
    /// Absolute byte floor below which a bucket is never treated as skewed.
    pub skew_min_bucket_bytes: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            morsel_rows: bc_arrow::DEFAULT_MORSEL_ROWS,
            morsel_bytes: bc_arrow::DEFAULT_MORSEL_BYTES,
            parallelism: 0,
            memory_budget_bytes: 0,
            spill_dir: None,
            op_budgets: HashMap::new(),
            fuse_linear: true,
            // Mirror `bc_arrow::RuntimeTuning::default()` field-for-field; the skew
            // floors reference the morsel consts the tuning struct derives them from.
            bloom_fp_rate: 0.01,
            bloom_min_build_rows: 1 << 16,
            window_parallel_row_threshold: 1 << 15,
            radix_parallel_threshold: 200_000,
            sort_merge_fanin: 16,
            skew_bucket_factor: 4,
            skew_min_bucket_rows: 4 * bc_arrow::DEFAULT_MORSEL_ROWS,
            skew_min_bucket_bytes: 4 * bc_arrow::DEFAULT_MORSEL_BYTES,
        }
    }
}

impl EngineConfig {
    /// The performance-threshold knobs as a [`bc_arrow::RuntimeTuning`] the data
    /// plane consumes — the bridge from the wire config to the runtime hot path.
    pub fn runtime_tuning(&self) -> bc_arrow::RuntimeTuning {
        bc_arrow::RuntimeTuning {
            bloom_fp_rate: self.bloom_fp_rate,
            bloom_min_build_rows: self.bloom_min_build_rows,
            window_parallel_row_threshold: self.window_parallel_row_threshold,
            radix_parallel_threshold: self.radix_parallel_threshold,
            sort_merge_fanin: self.sort_merge_fanin,
            skew_bucket_factor: self.skew_bucket_factor,
            skew_min_bucket_rows: self.skew_min_bucket_rows,
            skew_min_bucket_bytes: self.skew_min_bucket_bytes,
        }
    }

    /// Parse from the JSON the control plane ships. A blank document yields
    /// defaults, so callers that have no config to pass send `""`.
    pub fn from_json(s: &str) -> Result<Self, IrError> {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Ok(Self::default());
        }
        serde_json::from_str(trimmed).map_err(IrError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_json_is_default() {
        assert_eq!(
            EngineConfig::from_json("").unwrap(),
            EngineConfig::default()
        );
        assert_eq!(
            EngineConfig::from_json("   ").unwrap(),
            EngineConfig::default()
        );
    }

    #[test]
    fn default_morsel_matches_engine_const() {
        assert_eq!(
            EngineConfig::default().morsel_rows,
            bc_arrow::DEFAULT_MORSEL_ROWS
        );
        assert_eq!(
            EngineConfig::default().morsel_bytes,
            bc_arrow::DEFAULT_MORSEL_BYTES
        );
        assert_eq!(EngineConfig::default().parallelism, 0);
        // Unbounded by default: the in-memory fast path spills only when the
        // control plane ships a positive budget.
        assert_eq!(EngineConfig::default().memory_budget_bytes, 0);
        assert_eq!(EngineConfig::default().spill_dir, None);
    }

    #[test]
    fn spill_envelope_overlays_from_json() {
        let c =
            EngineConfig::from_json(r#"{"memory_budget_bytes": 1048576, "spill_dir": "/scratch"}"#)
                .unwrap();
        assert_eq!(c.memory_budget_bytes, 1_048_576);
        assert_eq!(c.spill_dir.as_deref(), Some("/scratch"));
        // Unspecified knobs still take their defaults.
        assert_eq!(c.parallelism, 0);
    }

    #[test]
    fn partial_json_overlays_onto_defaults() {
        let c = EngineConfig::from_json(r#"{"morsel_rows": 4096}"#).unwrap();
        assert_eq!(c.morsel_rows, 4096);
        assert_eq!(c.parallelism, 0); // unspecified → default
    }

    #[test]
    fn op_budgets_parse_from_string_keyed_object() {
        // The control plane ships per-operator budgets as a JSON object whose keys
        // are stringified pre-order op_ids; serde_json parses them back to u32.
        let c = EngineConfig::from_json(
            r#"{"memory_budget_bytes": 4096, "op_budgets": {"0": 1048576, "3": 2048}}"#,
        )
        .unwrap();
        assert_eq!(c.op_budgets.get(&0), Some(&1_048_576));
        assert_eq!(c.op_budgets.get(&3), Some(&2_048));
        assert_eq!(c.op_budgets.get(&1), None);
        // The global budget still parses alongside the side map.
        assert_eq!(c.memory_budget_bytes, 4096);
    }

    #[test]
    fn op_budgets_default_empty() {
        // Absent (older control plane / ad-hoc IR) ⇒ empty, so every operator falls
        // back to the global budget and behavior is unchanged.
        let c = EngineConfig::from_json(r#"{"memory_budget_bytes": 4096}"#).unwrap();
        assert!(c.op_budgets.is_empty());
    }

    #[test]
    fn runtime_tuning_defaults_match_bc_arrow() {
        let cfg = EngineConfig::default();
        let t = cfg.runtime_tuning();
        let d = bc_arrow::RuntimeTuning::default();
        assert_eq!(t.bloom_fp_rate, d.bloom_fp_rate);
        assert_eq!(t.bloom_min_build_rows, d.bloom_min_build_rows);
        assert_eq!(
            t.window_parallel_row_threshold,
            d.window_parallel_row_threshold
        );
        assert_eq!(t.radix_parallel_threshold, d.radix_parallel_threshold);
        assert_eq!(t.sort_merge_fanin, d.sort_merge_fanin);
        assert_eq!(t.skew_bucket_factor, d.skew_bucket_factor);
        assert_eq!(t.skew_min_bucket_rows, d.skew_min_bucket_rows);
        assert_eq!(t.skew_min_bucket_bytes, d.skew_min_bucket_bytes);
        assert_eq!(t, d);
    }

    #[test]
    fn tuning_knobs_overlay_from_json_into_runtime_tuning() {
        let c = EngineConfig::from_json(
            r#"{"bloom_fp_rate": 0.05, "radix_parallel_threshold": 50000}"#,
        )
        .unwrap();
        let t = c.runtime_tuning();
        assert_eq!(t.bloom_fp_rate, 0.05);
        assert_eq!(t.radix_parallel_threshold, 50_000);
        // Unspecified tuning knobs still take their defaults.
        let d = bc_arrow::RuntimeTuning::default();
        assert_eq!(t.sort_merge_fanin, d.sort_merge_fanin);
        assert_eq!(t.bloom_min_build_rows, d.bloom_min_build_rows);
    }

    #[test]
    fn unknown_fields_are_ignored() {
        // A newer control plane may ship knobs this engine doesn't consume yet.
        let c = EngineConfig::from_json(r#"{"morsel_rows": 8192, "future_knob": 1}"#).unwrap();
        assert_eq!(c.morsel_rows, 8192);
    }
}
