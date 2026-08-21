//! The machine's memory and core topology, as the data plane needs to see it.
//!
//! [`HardwareProfile`](crate::HardwareProfile) answers "what instructions can I emit and how
//! many threads may I run". That is only half of what a columnar engine has to know. The
//! other half is *where the memory is and how much of it stays close*: the cache sizes that
//! decide how big a morsel or a radix partition may get before it stops being resident, the
//! SMT ratio that decides whether the second thread on a core buys anything, and the NUMA
//! node map that decides whether a hash table probed from every core is a local read or a
//! cross-socket round trip.
//!
//! Those facts already existed on the Python side (`_internal/hardware/`), which is the wrong
//! place for them: the control plane cannot use a cache size to pick a partition fan-out
//! inside a `combine`, and shipping the numbers in `EngineConfig` would bake the *driver's*
//! machine into a plan that runs on a heterogeneous worker. Detection therefore happens
//! locally, per process, exactly like the ISA probe next door.
//!
//! Linux-only in substance (`/sys/devices/system/cpu`, `/sys/devices/system/node`). Every
//! probe has a defined answer elsewhere — the conservative one that reproduces the behavior
//! the engine had before this module existed: one NUMA node, no SMT, and the
//! `DEFAULT_*` cache estimates.

use std::sync::OnceLock;

use crate::hardware::usable_cores;

/// Cache-line size assumed when the platform does not report one.
///
/// 64 B on every x86_64 part and on the great majority of aarch64 ones. Apple silicon uses
/// 128 B, which detection reports where it can; assuming 64 there costs a little padding,
/// never correctness.
pub const DEFAULT_CACHE_LINE: usize = 64;

/// L1 data cache assumed when undetectable. 32 KiB is the x86_64 value from Nehalem through
/// Zen 4 and the common aarch64 one.
pub const DEFAULT_L1D_BYTES: usize = 32 << 10;

/// L2 cache assumed when undetectable. 512 KiB is deliberately conservative: it is the
/// smallest per-core L2 among parts the engine targets, so a partition sized to it is
/// resident everywhere rather than only on the machine the constant was measured on.
pub const DEFAULT_L2_BYTES: usize = 512 << 10;

/// Last-level cache assumed when undetectable. 8 MiB is a small-server figure; the same
/// conservatism as [`DEFAULT_L2_BYTES`] applies.
pub const DEFAULT_L3_BYTES: usize = 8 << 20;

/// The fraction of a cache level a blocking decision may plan to occupy.
///
/// A partition sized to *all* of L2 evicts itself: the hash table, the input morsel being
/// scanned, and the output buffers all share the level. Half is the standard blocking
/// headroom (it is what the classic radix-join literature uses), and it leaves room for the
/// streaming side without needing to model it.
const CACHE_OCCUPANCY_NUMERATOR: usize = 1;
const CACHE_OCCUPANCY_DENOMINATOR: usize = 2;

/// The host's cache hierarchy, SMT ratio, and NUMA layout.
///
/// Detected once per process and cached. Every field has a defined, conservative value on a
/// platform that cannot report it, so callers never branch on "did detection work" — they
/// read the number and size to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuTopology {
    /// Logical CPUs this process may use — identical to [`usable_cores`], repeated here so a
    /// caller sizing from topology reads one struct.
    pub logical_cores: usize,
    /// Physical cores backing those logical CPUs (SMT siblings collapsed). Never 0.
    ///
    /// The right denominator for compute-bound fan-out: a second thread on a core that is
    /// already saturating its execution units adds almost nothing while halving the L1/L2
    /// each thread sees.
    pub physical_cores: usize,
    /// Hardware threads sharing one physical core: 1 without SMT, 2 with it, 4 on POWER.
    ///
    /// Read from the sibling lists rather than computed as `logical / physical`, and that
    /// distinction is not cosmetic. `logical_cores` is capped by the cgroup CFS *bandwidth*
    /// quota while `physical_cores` is derived from the *affinity mask*, so the two have
    /// different denominators the moment a quota is set. On a 96-CPU 2-way-SMT host limited to
    /// 92 cores, the ratio is `92 / 48 = 1` under integer division — the machine reports as
    /// having no SMT at all, and every "is the second sibling worth a thread" decision
    /// silently inverts. Detecting the width directly cannot drift that way.
    pub threads_per_core: usize,
    /// NUMA nodes owning at least one usable CPU. Never 0; 1 means "not a NUMA problem".
    ///
    /// A process pinned to one socket of a two-socket host reports 1, which is correct: all
    /// its memory is local, and partitioning for a locality problem it cannot have would
    /// cost work for nothing.
    pub numa_nodes: usize,
    /// L1 data cache bytes for one core.
    pub l1d_bytes: usize,
    /// L2 cache bytes for one core (or one SMT pair, where L2 is shared).
    pub l2_bytes: usize,
    /// Last-level cache bytes for this core's cache domain.
    ///
    /// Read from `cpu0`, so on a chiplet part this is the *per-CCX* figure — the residency a
    /// table probed by the cores sharing that cache actually gets, not the socket total a
    /// summed figure would overstate by the chiplet count.
    pub l3_bytes: usize,
    /// Coherency line size in bytes — the granularity every false-sharing and prefetch
    /// decision is expressed in.
    pub cache_line: usize,
}

impl Default for CpuTopology {
    /// The conservative profile: one node, no SMT, default cache estimates.
    fn default() -> Self {
        let cores = usable_cores();
        Self {
            logical_cores: cores,
            physical_cores: cores,
            threads_per_core: 1,
            numa_nodes: 1,
            l1d_bytes: DEFAULT_L1D_BYTES,
            l2_bytes: DEFAULT_L2_BYTES,
            l3_bytes: DEFAULT_L3_BYTES,
            cache_line: DEFAULT_CACHE_LINE,
        }
    }
}

impl CpuTopology {
    /// The detected host topology (cached after the first call).
    pub fn detect() -> &'static CpuTopology {
        static TOPOLOGY: OnceLock<CpuTopology> = OnceLock::new();
        TOPOLOGY.get_or_init(detect_raw)
    }

    /// Hardware threads per physical core — see [`Self::threads_per_core`].
    ///
    /// A ratio rather than a boolean because 4-way SMT exists (POWER, some SPARC) and a
    /// caller weighing "how much of my logical count is real throughput" needs the factor.
    pub fn smt_width(&self) -> usize {
        self.threads_per_core.max(1)
    }

    /// Whether this host runs more than one hardware thread per core.
    pub fn has_smt(&self) -> bool {
        self.threads_per_core > 1
    }

    /// Whether this host has more than one usable NUMA node.
    pub fn is_numa(&self) -> bool {
        self.numa_nodes > 1
    }

    /// How many rows of `row_bytes` fit in the working half of a cache level.
    ///
    /// The single place the engine turns "this cache is N bytes" into "so process M rows at a
    /// time". Returns at least 1 so a caller never divides by zero on a pathologically wide
    /// row, and saturates rather than overflowing on a zero width.
    pub fn rows_in(&self, cache_bytes: usize, row_bytes: usize) -> usize {
        let budget = cache_bytes / CACHE_OCCUPANCY_DENOMINATOR * CACHE_OCCUPANCY_NUMERATOR;
        (budget / row_bytes.max(1)).max(1)
    }

    /// Rows of `row_bytes` that keep an operator's working set inside L2.
    ///
    /// The morsel-sizing question: a morsel exists to be small enough that the operator
    /// touching it does not evict itself between passes. L2 is the right level — L1 is too
    /// small to amortize per-morsel scheduling, L3 is shared and so not a per-thread budget.
    pub fn l2_resident_rows(&self, row_bytes: usize) -> usize {
        self.rows_in(self.l2_bytes, row_bytes)
    }

    /// Threads to run for work that saturates execution units rather than stalling on memory.
    ///
    /// Compute-bound kernels get the physical core count; the SMT sibling of a saturated core
    /// contributes almost nothing and halves its cache. Memory-bound work should use
    /// [`Self::logical_cores`] instead, which is what hides the stalls SMT exists for.
    pub fn compute_threads(&self) -> usize {
        self.physical_cores.max(1)
    }
}

/// Parse a Linux CPU list (`"0-3,8,10-11"`) into an ascending, deduplicated id vector.
///
/// The format `/sys` and `/proc` publish every CPU set in. Exposed because
/// [`placement`](crate::placement) reads `thread_siblings_list` files this module does not,
/// and two parsers for one kernel format would be one parser too many.
///
/// Malformed components are skipped rather than failing the whole parse: a partially readable
/// topology beats none, and the kernel occasionally emits fields this was not written for.
pub fn parse_cpu_list_public(raw: &str) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for part in raw.trim().split(',') {
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            Some((lo, hi)) => {
                if let (Ok(lo), Ok(hi)) = (lo.trim().parse(), hi.trim().parse::<usize>()) {
                    out.extend(lo..=hi);
                }
            }
            None => {
                if let Ok(c) = part.trim().parse() {
                    out.push(c);
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
mod sysfs {
    use std::collections::BTreeSet;

    /// A Linux CPU list as a set, for the membership and retain operations detection does.
    ///
    /// The parse itself lives in [`parse_cpu_list_public`](super::parse_cpu_list_public) —
    /// one parser for one kernel format; this only chooses the collection shape.
    pub(super) fn parse_cpu_list(raw: &str) -> BTreeSet<usize> {
        super::parse_cpu_list_public(raw).into_iter().collect()
    }

    /// Parse a `/sys` cache size (`"32K"`, `"1M"`, `"2048"`) into bytes, or 0.
    pub(super) fn parse_cache_size(raw: &str) -> usize {
        let raw = raw.trim();
        let (digits, mult) = match raw.chars().last() {
            Some('K') | Some('k') => (&raw[..raw.len() - 1], 1 << 10),
            Some('M') | Some('m') => (&raw[..raw.len() - 1], 1 << 20),
            Some('G') | Some('g') => (&raw[..raw.len() - 1], 1 << 30),
            _ => (raw, 1),
        };
        digits
            .trim()
            .parse::<usize>()
            .unwrap_or(0)
            .saturating_mul(mult)
    }

    pub(super) fn read_trimmed(path: &str) -> Option<String> {
        std::fs::read_to_string(path)
            .ok()
            .map(|s| s.trim().to_string())
    }

    pub(super) fn read_usize(path: &str) -> Option<usize> {
        read_trimmed(path)?.parse().ok()
    }

    pub(super) fn read_cpu_list(path: &str) -> BTreeSet<usize> {
        read_trimmed(path)
            .map(|s| parse_cpu_list(&s))
            .unwrap_or_default()
    }
}

/// The CPUs in this process's affinity mask, or `None` when it cannot be read.
///
/// `None` means "no restriction known", which callers read as "every CPU is allowed" — the
/// same answer the engine gave before affinity was consulted at all.
#[cfg(target_os = "linux")]
pub fn affinity_cpus() -> Option<Vec<usize>> {
    // /proc/self/status exposes the mask as a hex bitmap in `Cpus_allowed_list` (a cpulist),
    // which is exactly the format `parse_cpu_list` already reads. Going through /proc rather
    // than `sched_getaffinity` keeps this crate free of a libc dependency: `bc-arrow` is the
    // base of the DAG and every crate links it, so a C binding here is a cost everything pays.
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status
        .lines()
        .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))?;
    let cpus = sysfs::parse_cpu_list(line);
    (!cpus.is_empty()).then(|| cpus.into_iter().collect())
}

#[cfg(not(target_os = "linux"))]
pub fn affinity_cpus() -> Option<Vec<usize>> {
    None
}

/// Usable CPU ids grouped by NUMA node, ordered by node id.
///
/// The map a NUMA-aware placement needs: which cores to bind a per-node worker set to, and
/// therefore how to split a build side so each node probes its own copy. Empty when NUMA is
/// not exposed, which callers read as "one node".
#[cfg(target_os = "linux")]
pub fn numa_node_cpus() -> Vec<(usize, Vec<usize>)> {
    let allowed: Option<std::collections::BTreeSet<usize>> =
        affinity_cpus().map(|v| v.into_iter().collect());
    let mut nodes: Vec<(usize, Vec<usize>)> = Vec::new();
    let Ok(entries) = std::fs::read_dir("/sys/devices/system/node") else {
        return nodes;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(id) = name
            .strip_prefix("node")
            .and_then(|n| n.parse::<usize>().ok())
        else {
            continue;
        };
        let mut cpus = sysfs::read_cpu_list(&format!("/sys/devices/system/node/{name}/cpulist"));
        if let Some(allowed) = &allowed {
            cpus.retain(|c| allowed.contains(c));
        }
        if !cpus.is_empty() {
            nodes.push((id, cpus.into_iter().collect()));
        }
    }
    nodes.sort_by_key(|(id, _)| *id);
    nodes
}

#[cfg(not(target_os = "linux"))]
pub fn numa_node_cpus() -> Vec<(usize, Vec<usize>)> {
    Vec::new()
}

/// Physical cores backing this process's usable CPUs, and the SMT width, as
/// `(physical_cores, threads_per_core)`.
///
/// Both come from one walk of the sibling lists because they answer one question and must not
/// disagree — deriving the width by dividing two independently-sourced counts is exactly the
/// drift [`CpuTopology::threads_per_core`] documents.
///
/// Falls back to `(logical, 1)` when `thread_siblings_list` is absent, which preserves the
/// behavior of every platform that does not publish it.
#[cfg(target_os = "linux")]
fn physical_cores(logical: usize) -> (usize, usize) {
    use std::collections::BTreeSet;
    let allowed: Option<BTreeSet<usize>> = affinity_cpus().map(|v| v.into_iter().collect());
    let Ok(entries) = std::fs::read_dir("/sys/devices/system/cpu") else {
        return (logical, 1);
    };
    let mut cores: BTreeSet<Vec<usize>> = BTreeSet::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(id) = name
            .strip_prefix("cpu")
            .and_then(|n| n.parse::<usize>().ok())
        else {
            continue;
        };
        if allowed.as_ref().is_some_and(|a| !a.contains(&id)) {
            continue;
        }
        let mut siblings = sysfs::read_cpu_list(&format!(
            "/sys/devices/system/cpu/{name}/topology/thread_siblings_list"
        ));
        if let Some(allowed) = &allowed {
            siblings.retain(|c| allowed.contains(c));
        }
        if siblings.is_empty() {
            siblings.insert(id);
        }
        cores.insert(siblings.into_iter().collect());
    }
    if cores.is_empty() {
        return (logical, 1);
    }
    // The *widest* sibling group is the machine's SMT width. Taking the max rather than an
    // average is what keeps this right on a hybrid part (Alder Lake's P-cores have a sibling,
    // its E-cores do not) and under a mask that happens to admit only one sibling of some
    // core — in both cases the hardware is still SMT, and a caller deciding whether the second
    // thread of a core is worth using needs to know that.
    let width = cores.iter().map(|c| c.len()).max().unwrap_or(1).max(1);
    (cores.len().min(logical).max(1), width)
}

#[cfg(not(target_os = "linux"))]
fn physical_cores(logical: usize) -> (usize, usize) {
    (logical, 1)
}

/// `cpu0`'s data-cache hierarchy as `(l1d, l2, l3, line)`, zero for any level not reported.
#[cfg(target_os = "linux")]
fn cache_sizes() -> (usize, usize, usize, usize) {
    let (mut l1d, mut l2, mut l3, mut line) = (0usize, 0usize, 0usize, 0usize);
    let Ok(entries) = std::fs::read_dir("/sys/devices/system/cpu/cpu0/cache") else {
        return (l1d, l2, l3, line);
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with("index") {
            continue;
        }
        let base = format!("/sys/devices/system/cpu/cpu0/cache/{name}");
        let kind = sysfs::read_trimmed(&format!("{base}/type")).unwrap_or_default();
        // An instruction cache constrains no data-plane decision.
        if kind == "Instruction" {
            continue;
        }
        let Some(level) = sysfs::read_usize(&format!("{base}/level")) else {
            continue;
        };
        let size = sysfs::read_trimmed(&format!("{base}/size"))
            .map(|s| sysfs::parse_cache_size(&s))
            .unwrap_or(0);
        // A "Unified" L1 (some ARM cores) is the cache a data working set contends for, so
        // it counts as the d-cache rather than being skipped.
        match level {
            1 => l1d = l1d.max(size),
            2 => l2 = l2.max(size),
            // Levels past 3 (a victim L4 / eDRAM) are reported as the last level: that is the
            // cache a residency decision should actually plan against.
            n if n >= 3 => l3 = l3.max(size),
            _ => {}
        }
        if let Some(l) = sysfs::read_usize(&format!("{base}/coherency_line_size")) {
            line = line.max(l);
        }
    }
    (l1d, l2, l3, line)
}

#[cfg(not(target_os = "linux"))]
fn cache_sizes() -> (usize, usize, usize, usize) {
    (0, 0, 0, 0)
}

fn detect_raw() -> CpuTopology {
    let logical = usable_cores();
    let (l1d, l2, l3, line) = cache_sizes();
    let numa = numa_node_cpus().len().max(1);
    let (physical, threads_per_core) = physical_cores(logical);
    CpuTopology {
        logical_cores: logical,
        physical_cores: physical.clamp(1, logical),
        threads_per_core,
        numa_nodes: numa,
        // A zero from detection means "not reported", not "no cache" — fall back to the
        // conservative default rather than propagating a zero into a division.
        l1d_bytes: if l1d > 0 { l1d } else { DEFAULT_L1D_BYTES },
        l2_bytes: if l2 > 0 { l2 } else { DEFAULT_L2_BYTES },
        l3_bytes: if l3 > 0 { l3 } else { DEFAULT_L3_BYTES },
        cache_line: if line > 0 { line } else { DEFAULT_CACHE_LINE },
    }
}

// ---------------------------------------------------------------------------
// Prefetch
// ---------------------------------------------------------------------------

/// Hint the CPU to pull `addr` into cache for reading, without stalling on it.
///
/// The point of a prefetch is a hash probe: the bucket index is known one iteration before the
/// bucket is read, and issuing the load early hides the ~200-cycle miss behind the current
/// iteration's work. Without it a probe loop is a dependent chain of cache misses, which is
/// the single largest cost in a hash join over a table that does not fit in L2.
///
/// Safe to call with any address, including a dangling or unmapped one: a prefetch is a hint,
/// architecturally defined not to fault and not to change program state. It is a no-op on
/// targets without a prefetch intrinsic.
#[inline(always)]
pub fn prefetch_read<T>(addr: *const T) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `_mm_prefetch` is a hint instruction. It never faults, never dereferences
        // architecturally, and has no effect on program state — only on cache residency.
        unsafe {
            core::arch::x86_64::_mm_prefetch(addr as *const i8, core::arch::x86_64::_MM_HINT_T0);
        }
    }
    // AArch64 is not a fallback target here — Graviton, Ampere Altra and Grace are ordinary
    // cloud instances, and a Grace-Hopper host is an aarch64 machine feeding an H100. Leaving
    // the prefetch out on those was a silent loss on exactly the loops it exists for: the
    // gather kernels issue it against an index the loop will reach several iterations later,
    // which is where a random-access DRAM stall is otherwise unavoidable.
    //
    // Written as inline assembly rather than through `core::arch::aarch64::_prefetch`, which
    // is still unstable; `prfm` is a two-operand hint instruction and the asm is as stable as
    // the intrinsic would be.
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `prfm` is architecturally a hint. It never faults, never dereferences
        // architecturally, and has no effect on program state — only on cache residency.
        // `pldl1keep` is the read-for-reuse variant, matching `_MM_HINT_T0` above.
        unsafe {
            core::arch::asm!(
                "prfm pldl1keep, [{addr}]",
                addr = in(reg) addr,
                options(nostack, preserves_flags),
            );
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = addr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_is_internally_consistent() {
        let t = CpuTopology::detect();
        assert!(t.logical_cores >= 1);
        assert!(t.physical_cores >= 1);
        assert!(
            t.physical_cores <= t.logical_cores,
            "SMT collapse can only reduce the count ({} > {})",
            t.physical_cores,
            t.logical_cores
        );
        assert!(t.numa_nodes >= 1);
        assert!(t.cache_line.is_power_of_two());
        assert!(t.l1d_bytes > 0 && t.l2_bytes > 0 && t.l3_bytes > 0);
        assert!(t.smt_width() >= 1);
        assert_eq!(t.smt_width(), t.threads_per_core);
        assert_eq!(t.has_smt(), t.threads_per_core > 1);
        assert_eq!(t.compute_threads(), t.physical_cores);
        // The SMT width must survive a CFS quota. A quota caps `logical_cores` (bandwidth)
        // without changing the affinity mask `physical_cores` is derived from, so on a
        // 2-way-SMT host limited to 92 of 96 cores the ratio `92 / 48` floors to 1 and the
        // machine reports as having no SMT. Detection reads the sibling lists instead, so a
        // host with more logical CPUs than physical cores always reports a width above 1.
        if t.logical_cores > t.physical_cores {
            assert!(
                t.threads_per_core > 1,
                "{} logical CPUs on {} physical cores must report SMT, got width {}",
                t.logical_cores,
                t.physical_cores,
                t.threads_per_core
            );
        }
    }

    #[test]
    fn detection_is_cached_and_stable() {
        // Two reads must be the same object: callers size hot-path decisions from this, and a
        // topology that changed between two operators in one query would size them apart.
        let a = CpuTopology::detect();
        let b = CpuTopology::detect();
        assert!(std::ptr::eq(a, b));
    }

    #[test]
    fn rows_in_never_returns_zero_or_divides_by_zero() {
        let t = CpuTopology::default();
        assert!(t.rows_in(t.l2_bytes, 8) > 1);
        // A row wider than the whole cache still yields one row, not zero.
        assert_eq!(t.rows_in(1024, 1 << 20), 1);
        // A zero-width row must not panic.
        assert!(t.rows_in(t.l2_bytes, 0) >= 1);
    }

    #[test]
    fn l2_residency_shrinks_as_rows_widen() {
        let t = CpuTopology::default();
        let narrow = t.l2_resident_rows(8);
        let wide = t.l2_resident_rows(256);
        assert!(
            narrow > wide,
            "a wider row must fit fewer times in the same cache ({narrow} !> {wide})"
        );
        // Half of a 512 KiB L2 at 8 B/row is 32768 rows.
        assert_eq!(narrow, (DEFAULT_L2_BYTES / 2) / 8);
    }

    #[test]
    fn smt_width_is_the_detected_sibling_count_not_a_ratio() {
        let smt2 = CpuTopology {
            logical_cores: 16,
            physical_cores: 8,
            threads_per_core: 2,
            ..CpuTopology::default()
        };
        assert_eq!(smt2.smt_width(), 2);
        assert!(smt2.has_smt());
        assert_eq!(smt2.compute_threads(), 8);

        let none = CpuTopology {
            logical_cores: 8,
            physical_cores: 8,
            threads_per_core: 1,
            ..CpuTopology::default()
        };
        assert_eq!(none.smt_width(), 1);
        assert!(!none.has_smt());

        // The regression this field exists for: a CFS quota caps the logical count below
        // `physical x SMT`, so `92 / 48` floors to 1 and a 2-way-SMT host would report as
        // having none. The detected width is unaffected.
        let quota_capped = CpuTopology {
            logical_cores: 92,
            physical_cores: 48,
            threads_per_core: 2,
            ..CpuTopology::default()
        };
        assert_eq!(
            quota_capped.smt_width(),
            2,
            "a CFS quota must not make SMT disappear"
        );
        assert!(quota_capped.has_smt());

        // 4-way SMT (POWER) is a width, not a boolean.
        let smt4 = CpuTopology {
            threads_per_core: 4,
            ..CpuTopology::default()
        };
        assert_eq!(smt4.smt_width(), 4);
    }

    #[test]
    fn numa_flag_matches_the_node_count() {
        let one = CpuTopology {
            numa_nodes: 1,
            ..CpuTopology::default()
        };
        assert!(!one.is_numa());
        let two = CpuTopology {
            numa_nodes: 2,
            ..CpuTopology::default()
        };
        assert!(two.is_numa());
    }

    #[test]
    fn numa_map_agrees_with_the_detected_node_count() {
        let map = numa_node_cpus();
        let t = CpuTopology::detect();
        if map.is_empty() {
            assert_eq!(t.numa_nodes, 1, "an unexposed topology reads as one node");
        } else {
            assert_eq!(t.numa_nodes, map.len());
            // Node ids are sorted and each node owns at least one usable CPU.
            let ids: Vec<usize> = map.iter().map(|(id, _)| *id).collect();
            let mut sorted = ids.clone();
            sorted.sort_unstable();
            assert_eq!(ids, sorted);
            assert!(map.iter().all(|(_, cpus)| !cpus.is_empty()));
        }
    }

    #[test]
    fn affinity_is_a_subset_of_the_machine() {
        if let Some(cpus) = affinity_cpus() {
            assert!(!cpus.is_empty());
            // A cpulist is parsed sorted and deduplicated.
            let mut sorted = cpus.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(cpus, sorted);
        }
    }

    #[test]
    fn prefetch_is_a_no_op_on_any_address() {
        // The contract is that a prefetch never faults and never changes state. Exercising it
        // on a live slice, past its end, and on a null pointer is what pins that.
        let v = [1u64, 2, 3, 4];
        prefetch_read(v.as_ptr());
        prefetch_read(std::ptr::null::<u64>());
        assert_eq!(v[0], 1, "a prefetch must not modify the data");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_list_parsing_covers_the_shapes_sysfs_emits() {
        use super::sysfs::parse_cpu_list;
        assert_eq!(
            parse_cpu_list("0-3,8,10-11")
                .into_iter()
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 8, 10, 11]
        );
        assert_eq!(parse_cpu_list("5").into_iter().collect::<Vec<_>>(), vec![5]);
        assert!(parse_cpu_list("").is_empty());
        // Garbage components are dropped, not fatal.
        assert_eq!(
            parse_cpu_list("0,,x,2-1,3").into_iter().collect::<Vec<_>>(),
            vec![0, 3]
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cache_size_parsing_covers_the_suffixes_sysfs_emits() {
        use super::sysfs::parse_cache_size;
        assert_eq!(parse_cache_size("32K"), 32 << 10);
        assert_eq!(parse_cache_size("1M"), 1 << 20);
        assert_eq!(parse_cache_size("2G"), 2 << 30);
        assert_eq!(parse_cache_size("2048"), 2048);
        assert_eq!(parse_cache_size(""), 0);
        assert_eq!(parse_cache_size("junk"), 0);
    }
}
