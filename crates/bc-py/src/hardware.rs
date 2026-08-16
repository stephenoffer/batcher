//! What the engine's own process knows about its hardware and its allocator.
//!
//! Two facts the Python control plane cannot get any other way.
//!
//! **The hardware the *engine* sees.** Carbonite and Kyber size decisions from
//! `_internal/hardware/`, which reads `/sys` from the Python process. That is the same
//! machine, so the numbers usually agree — but "usually" is doing real work there: the
//! distributed path runs the engine inside a Ray worker whose cgroup was applied after the
//! interpreter started, and a heterogeneous cluster runs the same driver against workers with
//! different caches and ISAs. Reading the figures back *from the engine* is the only way to
//! know what the data plane actually sized itself to.
//!
//! **The allocator's real state.** Every buffer the data plane allocates comes from mimalloc,
//! which is invisible to `psutil` in the ways that matter: freed pages it retains still count
//! as RSS, and committed-but-untouched arena is reserved without being resident. A memory
//! envelope computed from process RSS therefore over-counts the engine and spills too early.
//! [`allocator_stats`] reads mimalloc's own accounting, and [`allocator_collect`] is the lever
//! that makes the over-count *actionable*: it returns retained free pages to the OS, which is
//! the cheap thing to try under pressure before spilling a hash table to disk.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// The CPU topology, ISA, and thread-placement plan the **engine process** detected.
///
/// Keys mirror the `bc_arrow` structs they come from: `logical_cores`, `physical_cores`,
/// `smt_width`, `numa_nodes`, `l1d_bytes`, `l2_bytes`, `l3_bytes`, `cache_line`, plus
/// `isa_tier` (the widest vector family: `"avx512"` / `"avx2"` / `"neon"` / …),
/// `isa_features` (every detected capability, sorted), `vector_bytes`, and
/// `avx512_is_cheap` (the down-clock heuristic — see `bc_arrow::isa`).
///
/// Returns:
///     A dict of hardware facts as detected inside the engine.
#[pyfunction]
pub(crate) fn engine_hardware(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let topo = bc_arrow::CpuTopology::detect();
    let isa = bc_arrow::IsaFeatures::detect();
    let d = PyDict::new(py);
    d.set_item("logical_cores", topo.logical_cores)?;
    d.set_item("physical_cores", topo.physical_cores)?;
    d.set_item("smt_width", topo.smt_width())?;
    d.set_item("has_smt", topo.has_smt())?;
    d.set_item("numa_nodes", topo.numa_nodes)?;
    d.set_item("is_numa", topo.is_numa())?;
    d.set_item("compute_threads", topo.compute_threads())?;
    d.set_item("l1d_bytes", topo.l1d_bytes)?;
    d.set_item("l2_bytes", topo.l2_bytes)?;
    d.set_item("l3_bytes", topo.l3_bytes)?;
    d.set_item("cache_line", topo.cache_line)?;
    d.set_item("isa_tier", isa.tier())?;
    d.set_item("isa_features", PyList::new(py, isa.names())?)?;
    d.set_item("vector_bytes", isa.vector_bytes())?;
    d.set_item("avx512_is_cheap", isa.avx512_is_cheap())?;
    Ok(d.unbind())
}

/// The CPU ids the engine would pin worker threads to, in worker order.
///
/// Empty when the topology is unreadable, which is the engine's "do not pin" state. Exposed
/// so a placement decision made in the control plane (a Ray actor's CPU set, a NUMA-aware
/// partition count) can be checked against what the data plane will actually do, rather than
/// both sides guessing independently.
///
/// Returns:
///     CPU ids in the order worker 0, 1, 2, ... would be pinned to.
#[pyfunction]
pub(crate) fn engine_pinning_order() -> Vec<usize> {
    bc_arrow::pinning_order()
}

/// The engine's NUMA node map: node id to the usable CPU ids on it.
///
/// Empty when NUMA is not exposed, which callers read as "one node". This is the map a
/// NUMA-aware placement needs — how many workers to put on each memory controller, and
/// therefore how to split a build side so each node probes a local copy.
///
/// Returns:
///     A list of `(node_id, [cpu_id, ...])` pairs, ordered by node id.
#[pyfunction]
pub(crate) fn engine_numa_map() -> Vec<(usize, Vec<usize>)> {
    bc_arrow::numa_node_cpus()
}

/// mimalloc's own view of the engine's memory: what is resident, committed, and peaked.
///
/// Keys, all in bytes except `page_faults`: `rss`, `peak_rss`, `commit`, `peak_commit`,
/// `reserved`, `page_faults`. On Linux `rss` is mimalloc's estimate from its commit
/// accounting rather than a `/proc` read, so it describes *the allocator's* footprint and not
/// the interpreter's — which is the number a data-plane memory envelope wants.
///
/// The gap between `commit` and `rss` is the retained-but-free arena
/// [`allocator_collect`] can hand back.
///
/// Returns:
///     A dict of allocator counters.
#[pyfunction]
pub(crate) fn allocator_stats(py: Python<'_>) -> PyResult<Py<PyDict>> {
    // No thread-local merge is needed (and mimalloc v3 no longer offers one): `mi_process_info`
    // reports process-level commit and RSS, not the per-thread counters `mi_stats_print` walks.
    let (mut elapsed, mut user, mut sys) = (0usize, 0usize, 0usize);
    let (mut rss, mut commit, mut peak_rss, mut peak_commit, mut faults) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    // SAFETY: every pointer is to a live local `usize` for the duration of the call, which is
    // exactly the out-parameter contract `mi_process_info` documents.
    unsafe {
        libmimalloc_sys::mi_process_info(
            &mut elapsed,
            &mut user,
            &mut sys,
            &mut commit,
            &mut peak_commit,
            &mut rss,
            &mut peak_rss,
            &mut faults,
        );
    }
    let d = PyDict::new(py);
    d.set_item("rss", rss)?;
    d.set_item("peak_rss", peak_rss)?;
    d.set_item("commit", commit)?;
    d.set_item("peak_commit", peak_commit)?;
    d.set_item("page_faults", faults)?;
    d.set_item("elapsed_ms", elapsed)?;
    d.set_item("user_ms", user)?;
    d.set_item("system_ms", sys)?;
    Ok(d.unbind())
}

/// Return the allocator's retained free pages to the operating system.
///
/// mimalloc keeps freed pages in per-thread heaps rather than unmapping them — that retention
/// is the whole reason it beats glibc here (see the `#[global_allocator]` note in `lib.rs`:
/// each `munmap` broadcasts a TLB-shootdown IPI to every core, which serializes an otherwise
/// parallel scan). The cost is that a query's peak footprint stays charged to the process
/// after the query ends.
///
/// This is the release valve, for the one situation where that trade inverts: a memory
/// envelope about to force a spill, where handing back tens of gigabytes of already-free arena
/// is far cheaper than writing a hash table to disk. Call it *before* spilling, not in a hot
/// path — it walks every heap, and forcing it re-imposes exactly the unmap cost the allocator
/// exists to avoid.
///
/// Args:
///     force: Also release pages from other threads' heaps. Thorough and considerably
///         more expensive; leave it false for a routine trim.
///
/// Returns:
///     Bytes of resident memory released, as the difference in mimalloc's own RSS
///     accounting. Zero means there was nothing retained to give back.
#[pyfunction]
#[pyo3(signature = (force = false))]
pub(crate) fn allocator_collect(py: Python<'_>, force: bool) -> u64 {
    let before = allocator_rss();
    // A forced collect walks every thread's heap and can take milliseconds; holding the GIL
    // across it would stall every other Python thread in the worker for no reason.
    py.allow_threads(|| {
        // SAFETY: `mi_collect` takes only a bool and is safe to call from any thread at any
        // point after the allocator is initialized, which it is — it served this frame.
        unsafe { libmimalloc_sys::mi_collect(force) }
    });
    before.saturating_sub(allocator_rss())
}

/// How long mimalloc holds a freed region before handing its pages back to the OS.
///
/// mimalloc's own default is 10 ms, which is tuned for a process whose allocation sizes sit in
/// its per-thread caches. An analytical engine's do not: an operator's output buffer is tens or
/// hundreds of megabytes, so a purge returns it to the kernel and the *next* query's identical
/// buffer arrives as fresh zero pages that the kernel must clear on first touch. That clearing
/// is not a rounding error — `clear_page_erms` was **9.3% of the whole query** on a 9M-row,
/// 13-column hash join (h2o-join q5), a query that allocates and frees ~1 GB of output buffers
/// per run and then does it again.
///
/// Ten seconds is long enough that consecutive queries reuse the same regions and short enough
/// that an idle process still gives its peak back. Measured on that join, alternating arms over
/// three rounds: **median 489 / 473 ms at the default against 422 / 432 ms at ten seconds**, and
/// neutral on TPC-H sf1 and the operator mix (0.840/0.844 against 0.853/0.836), whose buffers are
/// small enough never to reach the OS either way. `-1` (never purge) measured the same as this,
/// so nothing is bought by giving up the release entirely.
///
/// The retention is safe *because the release valve already exists*: [`allocator_collect`] hands
/// the arena back on demand, and Carbonite calls it under pressure before it spills. A user who
/// wants mimalloc's own tuning sets `MIMALLOC_PURGE_DELAY` and this defers to it.
const PURGE_DELAY_MS: std::ffi::c_long = 10_000;

/// mimalloc's `purge_delay` option, named by its position rather than by a symbol.
///
/// `libmimalloc-sys` does not re-export this one, so it has to be given as an ordinal — and an
/// ordinal into someone else's enum is exactly the kind of constant that silently comes to mean
/// something different. It is therefore *derived* from a neighbour the crate does export
/// unconditionally, and cross-checked against a second one: in both the v2 and v3 headers this
/// crate can link, `purge_delay` sits immediately before `use_numa_nodes` and six past
/// `reserve_os_memory`. If a future vendored mimalloc reorders them the assertion fails the
/// **build**, rather than leaving the engine quietly setting some other option at run time.
const MI_OPTION_PURGE_DELAY: std::ffi::c_int = libmimalloc_sys::mi_option_use_numa_nodes - 1;
const _: () = assert!(
    MI_OPTION_PURGE_DELAY == libmimalloc_sys::mi_option_reserve_os_memory + 6,
    "mimalloc's option enum moved: purge_delay is no longer the option before use_numa_nodes, \
     so this ordinal now names a different option"
);

/// Give the allocator the retention an analytical engine wants, unless the user set it.
///
/// Called once from the module initializer, which is the first moment the engine controls and
/// is already after mimalloc has served allocations — `purge_delay` is read on each purge
/// rather than latched at startup, so setting it here takes effect for everything that follows.
pub(crate) fn tune_allocator() {
    if std::env::var_os("MIMALLOC_PURGE_DELAY").is_some() {
        return;
    }
    // SAFETY: `mi_option_set` takes two integers and is safe to call from any thread once the
    // allocator is initialized, which it is — it served this frame.
    unsafe { libmimalloc_sys::mi_option_set(MI_OPTION_PURGE_DELAY, PURGE_DELAY_MS) };
}

/// mimalloc's current RSS estimate in bytes, for the before/after in [`allocator_collect`].
fn allocator_rss() -> u64 {
    let (mut elapsed, mut user, mut sys) = (0usize, 0usize, 0usize);
    let (mut rss, mut commit, mut peak_rss, mut peak_commit, mut faults) =
        (0usize, 0usize, 0usize, 0usize, 0usize);
    // SAFETY: as in `allocator_stats` — out-parameters to live locals.
    unsafe {
        libmimalloc_sys::mi_process_info(
            &mut elapsed,
            &mut user,
            &mut sys,
            &mut commit,
            &mut peak_commit,
            &mut rss,
            &mut peak_rss,
            &mut faults,
        );
    }
    rss as u64
}

/// Register the hardware and allocator introspection surface on the extension module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(engine_hardware, m)?)?;
    m.add_function(wrap_pyfunction!(engine_pinning_order, m)?)?;
    m.add_function(wrap_pyfunction!(engine_numa_map, m)?)?;
    m.add_function(wrap_pyfunction!(allocator_stats, m)?)?;
    m.add_function(wrap_pyfunction!(allocator_collect, m)?)?;
    Ok(())
}
