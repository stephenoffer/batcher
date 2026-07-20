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

/// Cores this process may actually use: `available_parallelism` capped by the cgroup CFS
/// quota. Never fewer than 1.
///
/// `available_parallelism` honors the CPU *affinity mask* (a cpuset pin) but not the CFS
/// *bandwidth* quota, and Kubernetes' `cpu` limit is the latter — a pod limited to 15 cores
/// on a 16-core node reports 16 and sizes every pool one thread too wide. Oversubscription
/// does not merely waste a thread: exceeding the quota gets the whole cgroup throttled for
/// the rest of the CFS period, so the extra worker buys stalls for *all* the others. This is
/// the figure to size thread pools and shard counts from.
pub fn usable_cores() -> usize {
    let affinity = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    match cfs_quota_cores() {
        Some(q) => affinity.min(q).max(1),
        None => affinity.max(1),
    }
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
