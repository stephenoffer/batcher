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
///
/// **Thread count governs shuffle-reduce throughput.** A reducer gathers many mapper
/// streams concurrently, and each stream's Arrow IPC decode is CPU-bound work that runs
/// on these worker threads; too few threads pins concurrent decode to a couple of cores
/// and caps inbound shuffle bandwidth well below the NIC. So the pool is sized to the
/// host's parallelism (clamped), not a fixed 2. It stays *bounded* — idle tokio workers
/// park rather than spin, so headroom is cheap — and is overridable via
/// `BATCHER_SHUFFLE_RT_THREADS` for hosts packing many actors per node (where a smaller
/// per-process pool avoids oversubscription across co-resident actors).
pub(crate) fn shared_runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(shuffle_runtime_threads())
            .enable_all()
            .build()
            .expect("build shared tokio runtime")
    })
}

/// Worker-thread count for the shuffle runtime: `BATCHER_SHUFFLE_RT_THREADS` when set
/// (and > 0), else the host's available parallelism clamped by the transfer's CPU cost.
///
/// The `[4, 8]` bound comes from measurement on a real cross-node cluster: a reducer
/// gathering many mapper streams of *uncompressed* IPC saturates its NIC at ~4–8
/// concurrent decode threads; below that it is decode-CPU-starved, and above 8 lightweight
/// decode regresses on context-switch/cache contention.
///
/// **Wire compression changes that calculus.** Compress (producer) + decompress (consumer)
/// is several-fold heavier per stream than plain decode, and it is what lets effective
/// throughput exceed line rate — but only if enough threads run it in parallel; at 8
/// threads ZSTD is CPU-starved and can't realize its ratio. Batcher runs one worker per
/// node (it owns every core, so there is no co-resident-actor oversubscription), so when a
/// codec is active the pool is sized to the cores (capped at 32) to parallelize
/// compression. `BATCHER_SHUFFLE_RT_THREADS` still overrides for unusual node shapes.
fn shuffle_runtime_threads() -> usize {
    if let Ok(v) = std::env::var("BATCHER_SHUFFLE_RT_THREADS") {
        if let Ok(n) = v.trim().parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    let cores = bc_arrow::usable_cores();
    // A codec is active by default; size to the cores for parallel (de)compression. Plain
    // uncompressed decode keeps the measured [4, 8] sweet spot.
    if bc_transport::compression().is_some() {
        cores.clamp(4, 32)
    } else {
        cores.clamp(4, 8)
    }
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
