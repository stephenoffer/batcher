//! Process-wide singletons the FFI layer shares across calls.
//!
//! A worker process makes many `execute_plan` / Flight calls over its life; these
//! must draw on *one* tokio runtime and *one* memory pool, not a fresh one per call.
//! Keeping both here (a single responsibility — process-lifetime shared state) keeps
//! `lib.rs` focused on the PyO3 surface.

use std::sync::{Arc, OnceLock};

use bc_resource::MemoryPool;

/// The one process-wide tokio runtime that drives every Flight server and client.
///
/// A runtime owns OS threads; building one per `FlightShuffleServer`/`ShuffleClient`
/// means a worker that instantiates many of them across a run accumulates (and then
/// drops, blocking-to-drain) dozens of runtimes — thread churn and GC pauses. One
/// lazily-built shared runtime keeps the thread pool bounded no matter how many
/// servers/clients a worker process creates.
pub(crate) fn shared_runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("build shared tokio runtime")
    })
}

/// The one process-wide [`MemoryPool`] backing the runtime spill backstop. Shared
/// across every `execute_plan` so the budget is a real ceiling on this process's
/// live operator state, not a per-query allowance N concurrent queries could each
/// blow. The limit only grows (`max(current, budget)`) so a smaller-budget query
/// can't shrink the envelope below a larger concurrent query's live reservations;
/// reservations are RAII, so `used()` returns to 0 between queries.
pub(crate) fn shared_memory_pool(budget: usize) -> Arc<MemoryPool> {
    static POOL: OnceLock<Arc<MemoryPool>> = OnceLock::new();
    let pool = POOL.get_or_init(|| MemoryPool::new(budget));
    if budget > pool.limit() {
        pool.set_limit(budget);
    }
    Arc::clone(pool)
}
