//! Host CPU capability detection for adaptive execution.
//!
//! [`HardwareProfile::detect`] probes the running CPU's SIMD ISA and core count
//! once (cached in a `OnceLock`) so the data plane can adapt *per process* — the
//! JIT picks a vector width/unroll, the scheduler sizes thread placement. This is
//! detected **locally on each worker**, never shipped in `EngineConfig`: a profile
//! baked into the driver's config would be wrong on a heterogeneous worker, and
//! single-node == distributed depends on the shipped config being host-independent.
//! `EngineConfig` carries only host-independent *policy overrides* (force a width,
//! disable SIMD, opt into AVX-512 width), which [`HardwareProfile::resolved`] layers
//! on top of detection.

use std::sync::OnceLock;

/// Detected host CPU capabilities plus the SIMD width/unroll the JIT should use.
///
/// The `simd_lanes_f64` / `simd_unroll` fields are the *resolved* plan: detection
/// caps the auto-selected f64 lane count at AVX2-equivalent (4) even on AVX-512
/// hosts, because 512-bit code can down-clock the core — AVX-512 width is opt-in via
/// the [`SimdOverride`]. The unroll factor defaults to 1 (the historical single
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

/// A host-independent policy override for the SIMD plan, carried in `EngineConfig`
/// and applied by [`HardwareProfile::resolved`]. All-default means "use detection".
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
    let mut dirs = vec!["/sys/fs/cgroup".to_string()];
    if let Ok(own) = std::fs::read_to_string("/proc/self/cgroup") {
        if let Some(sub) = own.lines().find_map(|l| l.strip_prefix("0::")) {
            let parts: Vec<&str> = sub.trim().split('/').filter(|p| !p.is_empty()).collect();
            for i in 1..=parts.len() {
                dirs.push(format!("/sys/fs/cgroup/{}", parts[..i].join("/")));
            }
        }
    }
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

#[cfg(not(target_os = "linux"))]
fn cfs_quota_cores() -> Option<usize> {
    None
}

/// Cores this Slurm allocation granted on this node, or `None` off Slurm.
///
/// A container is confined by cgroups, which the affinity mask and the CFS quota above already
/// report. A Slurm allocation is not, unless the site configured `task/cgroup` confinement —
/// and plenty of HPC sites do not. There the affinity mask reports every core on a shared node,
/// so a job granted 8 cores starts a thread per host core: it oversubscribes the node, steals
/// from the co-tenants Slurm placed there, and at a site with enforcement is what gets the job
/// killed.
///
/// This is the bound the Python control plane has always applied
/// (`_internal.hardware.cpu._slurm_cpu_count`) and the data plane did not, so the two planes
/// disagreed about the machine on exactly these nodes: the planner sized a fan-out to the grant
/// while the executor sized its rayon pool and its tokio runtime to the whole node. The data
/// plane is the half that actually spawns the threads, so it is the half where the gap bites.
///
/// `SLURM_CPUS_ON_NODE` is a run-length list on a heterogeneous job (`"4(x2),8"`). Which entry
/// describes *this* node is not derivable from the variable, so the smallest is taken: under-
/// parallelizing costs throughput, where over-parallelizing on the node that got the small
/// grant is the failure this bound exists to prevent.
fn slurm_granted_cores() -> Option<usize> {
    // Most specific first: `SLURM_CPUS_PER_TASK` is set when the job asked with
    // `--cpus-per-task`; `SLURM_CPUS_ON_NODE` is the node's whole share of the allocation and
    // is the fallback for a job that did not.
    ["SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"]
        .iter()
        .find_map(|var| {
            std::env::var(var)
                .ok()
                .and_then(|raw| slurm_expansion_min(raw.trim()))
        })
}

/// The smallest per-node count in a Slurm CPU-count value, or `None` if it does not parse.
///
/// Split out from [`slurm_granted_cores`] so the parse is testable as a pure function: the
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
/// quota and by any Slurm allocation. Never fewer than 1.
///
/// `available_parallelism` honors the CPU *affinity mask* (a cpuset pin) but not the CFS
/// *bandwidth* quota, and Kubernetes' `cpu` limit is the latter — a pod limited to 15 cores
/// on a 16-core node reports 16 and sizes every pool one thread too wide. Oversubscription
/// does not merely waste a thread: exceeding the quota gets the whole cgroup throttled for
/// the rest of the CFS period, so the extra worker buys stalls for *all* the others. It
/// honors no scheduler grant either; see [`slurm_granted_cores`]. This is the figure to size
/// thread pools and shard counts from.
pub fn usable_cores() -> usize {
    let affinity = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    [cfs_quota_cores(), slurm_granted_cores()]
        .into_iter()
        .flatten()
        .fold(affinity, usize::min)
        .max(1)
}

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
    fn usable_cores_is_bounded_and_consistent() {
        let affinity = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let usable = usable_cores();
        assert!(usable >= 1, "must never be zero");
        assert!(
            usable <= affinity,
            "quota may only narrow the affinity mask, never widen it ({usable} > {affinity})"
        );
        assert_eq!(usable, HardwareProfile::detect().logical_cores);
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

    /// The Slurm grant may only ever *narrow* the core budget, never widen it past what the
    /// affinity mask and the cgroup quota already allow.
    #[test]
    fn usable_cores_is_still_bounded_by_the_affinity_mask() {
        let affinity = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        assert!(usable_cores() <= affinity.max(1));
        assert!(usable_cores() >= 1);
    }
}
