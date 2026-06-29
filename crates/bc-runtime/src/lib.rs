//! `bc-runtime` — the engine's runtime library.
//!
//! These are the stateful, branchy, memory-managing building blocks that the
//! interpreter calls into and that (later) generated code invokes through a
//! stable ABI: hash aggregation, hash join, sorting, and their spillable
//! variants. Keeping them here — separate from the operator orchestration —
//! is what lets compiled pipelines own no relational state: the state lives in
//! these structures, so an artifact can be swapped without losing progress.
//!
//! The bootstrap implementations are correct-first (they lean on arrow's typed
//! kernels) and single-threaded; the SIMD/NUMA/spillable rewrites land behind
//! the same function signatures.

pub mod agg;
mod error;
pub mod join;
pub mod shuffle;
pub mod window;
pub mod window_frame;
mod window_partition_agg;

pub use error::RuntimeError;
