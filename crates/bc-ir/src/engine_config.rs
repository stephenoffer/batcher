//! Execution tunables shipped from the Python control plane alongside the plan.
//!
//! `EngineConfig` is the wire-contract mirror of the Rust-relevant slice of
//! Python's `Config` (`ExecutionConfig`). The control plane is the source of truth
//! at runtime — Core serializes the live config and ships it next to the plan JSON —
//! so the data plane no longer bakes morsel size / parallelism into constants and
//! silently drifts from Python. The `Default` here (sourced from the engine's own
//! `bc_arrow` consts) is what a standalone `cargo test` or an absent/empty config
//! falls back to; a Python↔Rust parity test pins the two default sets together.

use serde::Deserialize;

use crate::IrError;

/// The execution knobs the data plane consumes, deserialized from the control
/// plane's JSON. Unknown fields are ignored and missing fields take their default,
/// so adding a Python-side knob never breaks an older engine.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
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
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            morsel_rows: bc_arrow::DEFAULT_MORSEL_ROWS,
            morsel_bytes: bc_arrow::DEFAULT_MORSEL_BYTES,
            parallelism: 0,
            memory_budget_bytes: 0,
            spill_dir: None,
        }
    }
}

impl EngineConfig {
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
    fn unknown_fields_are_ignored() {
        // A newer control plane may ship knobs this engine doesn't consume yet.
        let c = EngineConfig::from_json(r#"{"morsel_rows": 8192, "future_knob": 1}"#).unwrap();
        assert_eq!(c.morsel_rows, 8192);
    }
}
