//! `bc-runtime` — the engine's runtime library.
//!
//! These are the stateful, branchy, memory-managing building blocks that the
//! interpreter calls into and that (later) generated code invokes through a
//! stable ABI: hash aggregation, hash join, hash shuffle, window functions, and
//! their spillable variants. Keeping them here — separate from the operator
//! orchestration — is what lets compiled pipelines own no relational state: the
//! state lives in these structures, so an artifact can be swapped without
//! losing progress.
//!
//! **Sorting is not here.** All four sort implementations (radix, sample-sort,
//! stable string, and the out-of-core merge) live in `bc_interp::ops`, because a
//! sort carries no state *between* morsels the way an aggregate or join build
//! does — it is a whole-input operation the executor drives directly. Window
//! functions do straddle the two crates: the kernels are here, while the
//! grace-partitioned out-of-core path is `bc_interp::window_spill`.
//!
//! The bootstrap implementations are correct-first (they lean on arrow's typed
//! kernels) and single-threaded; the SIMD/NUMA/spillable rewrites land behind
//! the same function signatures.

pub mod agg;
mod error;
pub mod gather;
pub mod join;
pub(crate) mod keys;
mod measure;
pub mod shuffle;
pub mod topn;
pub mod window;

pub use error::RuntimeError;
