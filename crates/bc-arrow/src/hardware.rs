//! Host CPU capability detection for adaptive execution.
//!
//! [`HardwareProfile::detect`] probes the running CPU's SIMD ISA and core count
//! once (cached in a `OnceLock`) so the data plane can adapt *per process* — the
//! JIT picks a vector width/unroll, the scheduler sizes thread placement. This is
//! detected **locally on each worker**, never shipped in `EngineConfig`: a profile
//! baked into the driver's config would be wrong on a heterogeneous worker, and
//! single-node == distributed depends on the shipped config being host-independent.
//! A [`SimdOverride`] (force a width, disable SIMD, opt into AVX-512 width) is layered on
//! top of detection by [`HardwareProfile::resolved`]; it is not an `EngineConfig` field,
//! so production always compiles with the default and only the tests pin it.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// Detected host CPU capabilities plus the SIMD width/unroll the JIT should use.
///
/// The `simd_lanes_f64` / `simd_unroll` fields are the *resolved* plan: detection
/// caps the auto-selected f64 lane count at AVX2-equivalent (4) even on AVX-512
/// hosts, because 512-bit code can down-clock the core — AVX-512 width is reachable only
/// through an explicit [`SimdOverride`]. The unroll factor defaults to 1 (the historical single
/// vector chain); widening it trades code size for instruction-level parallelism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HardwareProfile {
    /// f64 lanes per emitted SIMD vector: 2 (SSE2/NEON), 4 (AVX2), 8 (AVX-512).
    pub simd_lanes_f64: usize,
    /// Independent vector chains emitted per loop iteration (ILP unroll factor, ≥ 1).
    pub simd_unroll: usize,
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_neon: bool,
    /// Logical CPU count (≥ 1).
    pub logical_cores: usize,
}

/// A host-independent policy override for the SIMD plan, applied by
/// [`HardwareProfile::resolved`]. All-default means "use detection".
///
/// Not user-configurable: no `EngineConfig` field carries it, and the engine's operators
/// always pass the default. It exists so the codegen parity tests can prove every width
/// computes the same result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SimdOverride {
    /// Force the f64 lane count (`0` = auto/detected). Set to 2/4/8 to pin a width
    /// (e.g. opt into AVX-512's 8 lanes, which detection won't auto-select).
    pub lanes: usize,
    /// Force the unroll factor (`0` = auto/detected, currently 1).
    pub unroll: usize,
    /// Disable the SIMD JIT path entirely (the scalar JIT / interpreter still run).
    pub force_scalar: bool,
}

/// Whole cores the cgroup CFS bandwidth quota permits, or `None` when unlimited or
/// unreadable. cgroup v2 (`cpu.max`, the tightest across the process's whole cgroup
/// ancestry) then v1 (`cpu.cfs_quota_us` / `cpu.cfs_period_us`).
///
/// The quota is enforced at *every* level of a v2 hierarchy, so a limit set on a parent
/// slice — a Ray worker under a systemd scope, a nested container — binds even when the
/// leaf is unlimited. Taking the minimum over the chain is correct for any topology.
#[cfg(target_os = "linux")]
fn cfs_quota_cores() -> Option<usize> {
    fn quota_at(dir: &str) -> Option<usize> {
        let raw = std::fs::read_to_string(format!("{dir}/cpu.max")).ok()?;
        let mut parts = raw.split_whitespace();
        let quota: usize = parts.next()?.parse().ok()?; // "max" fails to parse ⇒ unlimited
        let period: usize = parts.next().unwrap_or("100000").parse().ok()?;
        (quota > 0 && period > 0).then(|| quota.div_ceil(period).max(1))
    }
    let own = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    let dirs = cgroup_v2_dirs(std::path::Path::new("/sys/fs/cgroup"), &own);
    let v2 = dirs.iter().filter_map(|d| quota_at(d)).min();
    if v2.is_some() {
        return v2;
    }
    let quota: i64 = std::fs::read_to_string("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        .ok()?
        .trim()
        .parse()
        .ok()?;
    let period: i64 = std::fs::read_to_string("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        .ok()?
        .trim()
        .parse()
        .ok()?;
    (quota > 0 && period > 0).then(|| (quota as usize).div_ceil(period as usize).max(1))
}

/// The cgroup v2 directories that bind this process: the mount at `root`, the process's own
/// cgroup, and every level between.
///
/// `/proc/self/cgroup` names the cgroup relative to the root the *kernel* sees, which inside a
/// container is not the mount: an Anyscale node reports `/anyscale/ctr_<id>/activities` for a
/// cgroup mounted at `/sys/fs/cgroup/activities`, whose quota is 75% of the container's. Joined
/// onto the mount that path names nothing, so leading components are dropped until it exists.
/// The same resolution as `bc_resource::headroom`'s, kept in step with it by hand: that crate
/// sits below this one and takes no Arrow, so neither can borrow the other's.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn cgroup_v2_dirs(root: &std::path::Path, proc_cgroup: &str) -> Vec<String> {
    let mut dirs = vec![root.to_string_lossy().into_owned()];
    let Some(sub) = proc_cgroup.lines().find_map(|l| l.strip_prefix("0::")) else {
        return dirs;
    };
    let parts: Vec<&str> = sub.trim().split('/').filter(|p| !p.is_empty()).collect();
    let skip = (0..parts.len())
        .find(|&k| {
            parts[k..]
                .iter()
                .fold(root.to_path_buf(), |d, p| d.join(p))
                .is_dir()
        })
        .unwrap_or(0);
    for i in skip + 1..=parts.len() {
        let dir = parts[skip..i]
            .iter()
            .fold(root.to_path_buf(), |d, p| d.join(p));
        dirs.push(dir.to_string_lossy().into_owned());
    }
    dirs
}

#[cfg(not(target_os = "linux"))]
fn cfs_quota_cores() -> Option<usize> {
    None
}

/// Environment variables in which a batch scheduler publishes its per-node core grant.
///
/// The same vocabulary the control plane reads (`_internal.site.scheduler.allocated_cpus`), so
/// the two planes agree about the machine. `SLURM_CPUS_PER_TASK` is set when a job asked with
/// `--cpus-per-task` and `SLURM_CPUS_ON_NODE` is the node's whole share; `PBS_NCPUS` is PBS'
/// submitted request; `NSLOTS` is Grid Engine's slot grant. Each name belongs to exactly one
/// scheduler, so its presence is enough. LSF publishes a per-host breakdown instead and is read
/// separately.
const CPU_GRANT_VARS: [&str; 4] = [
    "SLURM_CPUS_PER_TASK",
    "SLURM_CPUS_ON_NODE",
    "PBS_NCPUS",
    "NSLOTS",
];

/// Grant variables whose *name* belongs to nobody in particular, with the marker that makes one
/// this scheduler's.
///
/// `NCPUS` is what PBS sets on the execution host and is the sharper of its two figures — but
/// it is also a name unrelated tooling sets, and this bound narrows every thread pool the
/// process starts, so it is believed only inside a PBS job.
const GATED_CPU_GRANT_VARS: [(&str, [&str; 2]); 1] = [("NCPUS", ["PBS_JOBID", "PBS_NODEFILE"])];

/// Cores this process's batch allocation granted it on this node, or `None` when unscheduled.
///
/// A container is confined by cgroups, which the affinity mask and the CFS quota above already
/// report. A batch allocation is not, unless the site configured cgroup confinement — and
/// plenty of HPC sites do not. There the affinity mask reports every core on a shared node, so
/// a job granted 8 cores starts a thread per host core: it oversubscribes the node, steals from
/// the co-tenants the scheduler placed there, and at a site with enforcement is what gets the
/// job killed.
///
/// This is the bound the Python control plane applies and the data plane did not, so the two
/// planes disagreed about the machine on exactly these nodes: the planner sized a fan-out to
/// the grant while the executor sized its rayon pool and its tokio runtime to the whole node.
/// The data plane is the half that actually spawns the threads, so it is the half where the gap
/// bites.
///
/// **The smallest present grant wins, and no scheduler detection is consulted.** A job
/// submitted through a compatibility wrapper carries two schedulers' variables at once, and the
/// weakest bound is the one that keeps the co-tenants whole — the same reason
/// [`slurm_expansion_min`] takes the minimum within a single variable.
fn scheduler_granted_cores() -> Option<usize> {
    let gated = GATED_CPU_GRANT_VARS.iter().filter_map(|(var, markers)| {
        markers
            .iter()
            .any(|m| std::env::var(m).is_ok_and(|v| !v.trim().is_empty()))
            .then_some(*var)
    });
    CPU_GRANT_VARS
        .iter()
        .copied()
        .chain(gated)
        .filter_map(|var| {
            std::env::var(var)
                .ok()
                .and_then(|raw| slurm_expansion_min(raw.trim()))
        })
        .chain(lsf_min_slots())
        .min()
}

/// The smallest per-host slot count LSF published, or `None`.
///
/// LSF's `LSB_DJOB_NUMPROC` is the *job-wide* slot total, so using it as a per-node bound
/// over-counts by the number of hosts — the opposite of what a bound is for. `LSB_MCPU_HOSTS`
/// is the per-host breakdown (`"hostA 8 hostB 8"`), and its minimum is the weakest safe bound
/// when this host's own entry cannot be matched by name.
fn lsf_min_slots() -> Option<usize> {
    let raw = std::env::var("LSB_MCPU_HOSTS").ok()?;
    raw.split_whitespace()
        .skip(1)
        .step_by(2)
        .filter_map(|count| count.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .min()
}

/// The smallest per-node count in a scheduler CPU-count value, or `None` if it does not parse.
///
/// `SLURM_CPUS_ON_NODE` is a run-length list on a heterogeneous job (`"4(x2),8"`). Which entry
/// describes *this* node is not derivable from the variable, so the smallest is taken: under-
/// parallelizing costs throughput, where over-parallelizing on the node that got the small
/// grant is the failure this bound exists to prevent.
///
/// Split out from [`scheduler_granted_cores`] so the parse is testable as a pure function: the
/// lookup around it reads process-global environment, which no test can exercise without
/// racing every other test in the binary.
fn slurm_expansion_min(raw: &str) -> Option<usize> {
    raw.split(',')
        .map(|part| {
            part.split('(')
                .next()
                .unwrap_or("")
                .trim()
                .parse::<usize>()
                .ok()
        })
        // An unrecognized shape yields `None` for the whole value: no bound beats a wrong
        // one, and a missing bound is exactly the behavior that held before.
        .collect::<Option<Vec<usize>>>()?
        .into_iter()
        .filter(|n| *n > 0)
        .min()
}

/// Cores this process may actually use: `available_parallelism` capped by the cgroup CFS
/// quota and by any batch scheduler's allocation. Never fewer than 1.
///
/// `available_parallelism` honors the CPU *affinity mask* (a cpuset pin) but not the CFS
/// *bandwidth* quota, and Kubernetes' `cpu` limit is the latter — a pod limited to 15 cores
/// on a 16-core node reports 16 and sizes every pool one thread too wide. Oversubscription
/// does not merely waste a thread: exceeding the quota gets the whole cgroup throttled for
/// the rest of the CFS period, so the extra worker buys stalls for *all* the others. It
/// honors no scheduler grant either; see [`scheduler_granted_cores`]. This is the figure to size
/// thread pools and shard counts from.
pub fn usable_cores() -> usize {
    let now = process_nanos();
    let last = CORE_COUNT_TAKEN_AT.load(Ordering::Relaxed);
    let cached = CORE_COUNT.load(Ordering::Relaxed);
    if cached != 0 && now.saturating_sub(last) < CORE_COUNT_TTL_NANOS {
        return cached;
    }
    let fresh = measure_usable_cores();
    CORE_COUNT.store(fresh, Ordering::Relaxed);
    CORE_COUNT_TAKEN_AT.store(now, Ordering::Relaxed);
    fresh
}

/// How long a reading of [`usable_cores`] is reused before it is taken again.
///
/// The reading is not free: on cgroup v2 the quota is enforced at every level of the
/// hierarchy, so [`cfs_quota_cores`] reads `/proc/self/cgroup` and then one `cpu.max` per
/// ancestor. Traced on a query with a five-deep cgroup path, one `execute_plan` cost **six
/// `/proc/self/cgroup` reads and twelve `cpu.max` opens** — the figure is asked for once per
/// pool sizing, once per shard-count decision, and once per profile consult, and each answer
/// costs a handful of syscalls.
///
/// A tenth of a second is short enough that the freshness this function exists for still
/// holds: a Ray worker whose CPU affinity is applied *after* the process starts (the hazard
/// `ExecOptions::workers` documents) is picked up within one tick, and nothing sizes a pool
/// more often than that in a way a stale-by-100 ms answer would get wrong.
const CORE_COUNT_TTL_NANOS: u64 = 100_000_000;

/// The last reading, and when it was taken. Zero means "never read".
///
/// Two relaxed atomics rather than a lock: a racing pair of readers may both measure and
/// store, which costs one extra reading and cannot produce a wrong one, since every writer
/// stores a value it just measured.
static CORE_COUNT: AtomicUsize = AtomicUsize::new(0);
static CORE_COUNT_TAKEN_AT: AtomicU64 = AtomicU64::new(0);

/// Nanoseconds since the first call, on a monotonic clock that cannot jump backwards.
fn process_nanos() -> u64 {
    static START: OnceLock<Instant> = OnceLock::new();
    let start = START.get_or_init(Instant::now);
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// [`usable_cores`] with no caching — the reading itself.
fn measure_usable_cores() -> usize {
    let affinity = std::thread::available_parallelism().map_or(1, |n| n.get());
    [cfs_quota_cores(), scheduler_granted_cores()]
        .into_iter()
        .flatten()
        .fold(affinity, usize::min)
        .max(1)
}

/// Threads to run a **relational operator pipeline**: every physical core this process may
/// use, plus a third of the SMT siblings among them.
///
/// [`usable_cores`] is the wrong figure for a query, and running all of it is the worse end of
/// the SMT trade. A plan is not one kernel: it interleaves work that stalls on memory (hash
/// build and probe, group assignment) with work that saturates bandwidth (gather, scan,
/// concat). SMT hides the stalls of the first and doubles the cache pressure of the second, so
/// the best width sits between the physical core count and the logical one.
///
/// The SMT *ratio* comes from the cached topology (a machine property that cannot change) and
/// the *count* from a fresh [`usable_cores`] — deliberately, so this stays correct on a Ray
/// worker whose CPU affinity is applied after the process starts, which is the hazard
/// `ExecOptions::workers` documents. On a host with no SMT it is every usable core, and it
/// never exceeds what a cgroup quota grants.
///
/// **A small host runs every sibling.** The trade above was measured on a 96-CPU box, where
/// dozens of cores saturate the socket's memory bandwidth and the second sibling mostly adds
/// cache pressure. Four or eight physical cores do not saturate it, so the sibling's latency
/// hiding is what remains. Measured on an 8-CPU (4 physical) Xeon 8259CL, two interleaved
/// rounds, `b/duckdb` geomean at the old width (5) against all 8:
///
/// | suite | this rule's old width | all logical |
/// |---|---:|---:|
/// | TPC-H sf1 (22) | 0.64 / 0.63 | **0.53 / 0.53** |
/// | ClickBench (43) | 0.52 / 0.51 | **0.47 / 0.46** |
/// | H2O groupby (10) | 0.64 / 0.63 | **0.59 / 0.58** |
///
/// Stated rather than hidden: H2O q3 and q7 (`GROUP BY id3`, 100k string groups) are 11-17%
/// slower at the full width in both rounds, the large-box pattern on its two most
/// bandwidth-bound shapes, and still 0.62-0.68x DuckDB.
#[must_use]
pub fn operator_cores() -> usize {
    operator_width(usable_cores(), crate::CpuTopology::detect().smt_width())
}

/// [`operator_cores`]' arithmetic, for `usable` logical cores at `smt` siblings per core.
fn operator_width(usable: usize, smt: usize) -> usize {
    if smt <= 1 {
        return usable;
    }
    let physical = usable.div_ceil(smt);
    if physical <= SMALL_HOST_PHYSICAL_CORES {
        return usable;
    }
    (physical + (usable - physical) / 3).clamp(1, usable)
}

/// Physical cores at or below which [`operator_cores`] runs every SMT sibling — see its doc.
const SMALL_HOST_PHYSICAL_CORES: usize = 8;

fn detect_raw() -> HardwareProfile {
    let logical_cores = usable_cores();

    #[cfg(target_arch = "x86_64")]
    {
        let has_avx2 = std::is_x86_feature_detected!("avx2");
        let has_avx512f = std::is_x86_feature_detected!("avx512f");
        // Cap the auto width at AVX2 (4 lanes); AVX-512's 8 lanes are opt-in because
        // 512-bit execution can down-clock the core and lose the net win.
        let simd_lanes_f64 = if has_avx2 || has_avx512f { 4 } else { 2 };
        return HardwareProfile {
            simd_lanes_f64,
            simd_unroll: 1,
            has_avx2,
            has_avx512f,
            has_neon: false,
            logical_cores,
        };
    }
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is baseline on aarch64; it is 128-bit, so 2 f64 lanes.
        return HardwareProfile {
            simd_lanes_f64: 2,
            simd_unroll: 1,
            has_avx2: false,
            has_avx512f: false,
            has_neon: true,
            logical_cores,
        };
    }
    #[allow(unreachable_code)]
    HardwareProfile {
        simd_lanes_f64: 2,
        simd_unroll: 1,
        has_avx2: false,
        has_avx512f: false,
        has_neon: false,
        logical_cores,
    }
}

impl HardwareProfile {
    /// The detected host profile (cached after the first call).
    pub fn detect() -> &'static HardwareProfile {
        static PROFILE: OnceLock<HardwareProfile> = OnceLock::new();
        PROFILE.get_or_init(detect_raw)
    }

    /// The detected profile with a policy override applied: a non-zero `lanes`/
    /// `unroll` pins that field; `force_scalar` collapses to a single scalar lane so
    /// the JIT never takes the vector path. Lane/unroll counts are clamped to ≥ 1.
    #[must_use]
    pub fn resolved(over: SimdOverride) -> HardwareProfile {
        let base = *Self::detect();
        if over.force_scalar {
            return HardwareProfile {
                simd_lanes_f64: 1,
                simd_unroll: 1,
                ..base
            };
        }
        HardwareProfile {
            simd_lanes_f64: if over.lanes > 0 {
                over.lanes
            } else {
                base.simd_lanes_f64
            },
            simd_unroll: if over.unroll > 0 {
                over.unroll.max(1)
            } else {
                base.simd_unroll
            },
            ..base
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small SMT host runs every sibling; a large one keeps physical + a third of the
    /// siblings; a host with no SMT runs every core either way.
    #[test]
    fn operator_width_runs_every_sibling_only_on_a_small_host() {
        assert_eq!(operator_width(8, 2), 8, "4 physical cores: every sibling");
        assert_eq!(operator_width(16, 2), 16, "8 physical cores: every sibling");
        assert_eq!(operator_width(18, 2), 12, "9 physical cores: 9 + 9/3");
        assert_eq!(operator_width(96, 2), 64, "48 physical cores: 48 + 48/3");
        assert_eq!(operator_width(96, 1), 96, "no SMT: every core");
        assert_eq!(operator_width(1, 2), 1);
    }

    #[test]
    fn detect_is_internally_consistent() {
        let p = HardwareProfile::detect();
        assert!(p.logical_cores >= 1);
        assert!(p.simd_lanes_f64 == 2 || p.simd_lanes_f64 == 4 || p.simd_lanes_f64 == 8);
        assert!(p.simd_unroll >= 1);
        // AVX-512 width is never auto-selected (opt-in only).
        assert!(p.simd_lanes_f64 <= 4);
        #[cfg(target_arch = "aarch64")]
        assert!(p.has_neon && p.simd_lanes_f64 == 2);
    }

    #[test]
    fn override_pins_and_force_scalar_collapses() {
        let pinned = HardwareProfile::resolved(SimdOverride {
            lanes: 8,
            unroll: 2,
            force_scalar: false,
        });
        assert_eq!(pinned.simd_lanes_f64, 8);
        assert_eq!(pinned.simd_unroll, 2);

        let scalar = HardwareProfile::resolved(SimdOverride {
            lanes: 8,
            unroll: 4,
            force_scalar: true,
        });
        assert_eq!(scalar.simd_lanes_f64, 1);
        assert_eq!(scalar.simd_unroll, 1);

        // All-default override == detection.
        assert_eq!(
            HardwareProfile::resolved(SimdOverride::default()),
            *HardwareProfile::detect()
        );
    }
}

#[cfg(test)]
mod usable_cores_tests {
    use super::*;

    /// `usable_cores` must never exceed what the affinity mask allows, never be 0, and must
    /// agree with the detected profile — the profile is what the JIT and scheduler read, so a
    /// divergence between the two would size pools differently from the reported hardware.
    #[test]
    fn a_container_cgroup_path_resolves_under_its_mount() {
        let root = std::env::temp_dir().join(format!("bc-arrow-cg-{}", std::process::id()));
        std::fs::create_dir_all(root.join("activities").join("x")).unwrap();
        let s = |p: &std::path::Path| p.to_string_lossy().into_owned();
        let dirs = cgroup_v2_dirs(&root, "0::/anyscale/ctr_abc/activities/x\n");
        assert_eq!(
            dirs,
            vec![
                s(&root),
                s(&root.join("activities")),
                s(&root.join("activities/x"))
            ]
        );
        assert_eq!(cgroup_v2_dirs(&root, "0::/\n"), vec![s(&root)]);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn usable_cores_is_bounded_and_consistent() {
        let affinity = std::thread::available_parallelism().map_or(1, |n| n.get());
        let usable = usable_cores();
        assert!(usable >= 1, "must never be zero");
        assert!(
            usable <= affinity,
            "quota may only narrow the affinity mask, never widen it ({usable} > {affinity})"
        );
        assert_eq!(usable, HardwareProfile::detect().logical_cores);
    }

    /// The cached reading must be the reading. A cache that returned a stale, zero or
    /// otherwise invented figure would size every pool in the process from it, silently, so
    /// this holds the memoized answer against a fresh measurement rather than against itself.
    ///
    /// The TTL's *expiry* is deliberately not asserted here: a test that sleeps past it would
    /// be a wall-clock assertion in a unit suite, and what it would prove — that a constant is
    /// finite — is visible in the constant. What matters and is checked is that the value
    /// served is the measured one.
    #[test]
    fn the_memoized_reading_equals_a_fresh_measurement() {
        let fresh = measure_usable_cores();
        assert_eq!(usable_cores(), fresh);
        // Warm, so this call is served from the cache rather than by measuring again.
        assert_eq!(usable_cores(), fresh);
        assert!(fresh >= 1);
    }

    /// A quota, when present, is a whole-core ceiling ≥ 1 — `cpu.max` of "50000 100000"
    /// (half a core) must round *up* to 1 rather than to a pool of zero threads.
    #[test]
    fn a_quota_is_a_positive_whole_core_count() {
        if let Some(q) = cfs_quota_cores() {
            assert!(q >= 1, "a sub-core quota must round up to one usable core");
        }
    }
}

#[cfg(test)]
mod slurm_grant_tests {
    use super::*;

    /// A heterogeneous job's `SLURM_CPUS_ON_NODE` is a run-length list (`"4(x2),8"`), and the
    /// *smallest* grant in it binds.
    ///
    /// Which entry describes this node is not derivable from the variable, and the asymmetry is
    /// what decides the direction: under-parallelizing costs throughput, where over-parallelizing
    /// on the node that got the small grant oversubscribes a shared HPC node and, at a site with
    /// enforcement, gets the job killed.
    #[test]
    fn an_expansion_binds_to_its_smallest_grant() {
        assert_eq!(slurm_expansion_min("4(x2),8"), Some(4));
        assert_eq!(slurm_expansion_min("8,4(x2)"), Some(4));
        assert_eq!(slurm_expansion_min("16"), Some(16));
        assert_eq!(slurm_expansion_min("32(x4)"), Some(32));
    }

    /// An unrecognized shape must yield no bound at all. A wrong bound silently
    /// under-parallelizes every query for the life of the job; a missing one is exactly the
    /// behavior that held before this existed.
    #[test]
    fn an_unparseable_value_yields_no_bound() {
        assert_eq!(slurm_expansion_min("weird"), None);
        assert_eq!(slurm_expansion_min("4,weird"), None);
        assert_eq!(slurm_expansion_min(""), None);
        assert_eq!(slurm_expansion_min("0"), None);
    }

    /// LSF publishes a per-host breakdown rather than a single count, and its job-wide total
    /// (`LSB_DJOB_NUMPROC`) over-counts a per-node bound by the number of hosts — the opposite
    /// of what a bound is for. The smallest per-host entry is the weakest safe answer.
    #[test]
    fn an_lsf_breakdown_binds_to_its_smallest_host() {
        // Parsed as a pure function over the value, for the reason `slurm_expansion_min` is:
        // the lookup around it reads process-global environment.
        fn min_slots(raw: &str) -> Option<usize> {
            raw.split_whitespace()
                .skip(1)
                .step_by(2)
                .filter_map(|c| c.parse::<usize>().ok())
                .filter(|n| *n > 0)
                .min()
        }
        assert_eq!(min_slots("gpu07 16 gpu08 8"), Some(8));
        assert_eq!(min_slots("gpu07 16"), Some(16));
        assert_eq!(min_slots(""), None);
    }

    /// The scheduler grant may only ever *narrow* the core budget, never widen it past what the
    /// affinity mask and the cgroup quota already allow.
    #[test]
    fn usable_cores_is_still_bounded_by_the_affinity_mask() {
        let affinity = std::thread::available_parallelism().map_or(1, |n| n.get());
        assert!(usable_cores() <= affinity.max(1));
        assert!(usable_cores() >= 1);
    }
}
