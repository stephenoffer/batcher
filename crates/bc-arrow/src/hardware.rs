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

fn detect_raw() -> HardwareProfile {
    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

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
