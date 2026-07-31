//! The host's instruction-set capabilities, in full.
//!
//! [`HardwareProfile`](crate::HardwareProfile) carries the three ISA facts the JIT's lane
//! selection needs (AVX2, AVX-512F, NEON). That is the right surface for *codegen*, and the
//! wrong one for everything else: a kernel choosing between a scalar and a vector path wants
//! to know about FMA; a bit-manipulation loop wants BMI2; a report or a bug triage wants the
//! whole list. Widening `HardwareProfile` for each of those would put fields in the struct the
//! JIT reads that the JIT does not use.
//!
//! So the full probe lives here, detected once and cached, and `HardwareProfile` keeps its
//! narrow shape. Both read the same `is_x86_feature_detected!` / `is_aarch64_feature_detected!`
//! machinery, which is a runtime `cpuid`/`HWCAP` query rather than a compile-time
//! `target_feature` — the distinction that matters, because the engine ships one binary that
//! must adapt to whatever it lands on.
//!
//! ## The AVX-512 width policy, and why detection alone does not settle it
//!
//! It is tempting to read `avx512f` and emit 512-bit code. On Skylake-SP and Cascade Lake that
//! is frequently a *loss*: heavy 512-bit use drops the core (and, on some parts, the whole
//! socket) into a lower license frequency, and the clock loss outweighs the width gain on any
//! kernel that is not purely vector-bound. Ice Lake-SP onward and Zen 4 onward largely removed
//! that penalty.
//!
//! There is no CPUID bit for "does this part down-clock". [`IsaFeatures::avx512_is_cheap`] uses
//! the closest available proxy — `avx512vpopcntdq`, which arrived with Ice Lake and is present
//! on Zen 4, and is absent on every Skylake-derived part — and is documented as a heuristic
//! rather than a fact. The engine still does **not** auto-widen on it: the default stays at
//! AVX2-equivalent width and 512-bit is opt-in through [`SimdOverride`](crate::SimdOverride),
//! because widening a default is a performance claim, and a performance claim needs a
//! benchmark on the part in question. What this flag buys is that the opt-in can be made
//! knowingly instead of blindly.

use std::sync::OnceLock;

/// Every ISA capability the data plane can make a decision from.
///
/// Fields are `false` on an architecture where the question does not apply, so a caller reads
/// the flag it cares about without arch-gating its own code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IsaFeatures {
    // ---- x86_64 ----
    /// SSE4.2 — `pcmpestri` and friends, plus the CRC32 instruction.
    pub sse4_2: bool,
    /// AVX — 256-bit float vectors.
    pub avx: bool,
    /// AVX2 — 256-bit *integer* vectors and per-lane variable shifts.
    pub avx2: bool,
    /// FMA3 — fused multiply-add. Halves the latency of a dot-product chain.
    pub fma: bool,
    /// AVX-512 Foundation — 512-bit vectors and mask registers.
    pub avx512f: bool,
    /// AVX-512 Byte/Word — 8- and 16-bit lanes, which is what makes AVX-512 useful for
    /// string and dictionary work rather than only for wide numerics.
    pub avx512bw: bool,
    /// AVX-512 Doubleword/Quadword — 32/64-bit integer ops on 512-bit vectors.
    pub avx512dq: bool,
    /// AVX-512 Vector Length — the same instructions on 128/256-bit vectors, which is what
    /// lets AVX-512's *mask registers* be used without paying the 512-bit clock penalty.
    pub avx512vl: bool,
    /// AVX-512 VPOPCNTDQ — vectorized population count. Also the generation marker behind
    /// [`Self::avx512_is_cheap`].
    pub avx512vpopcntdq: bool,
    /// AVX-512 VNNI — integer dot-product, the quantized-inference instruction.
    pub avx512vnni: bool,
    /// BMI1 — `andn`, `blsr`, `tzcnt`.
    pub bmi1: bool,
    /// BMI2 — `pdep`/`pext` (bit gather/scatter) and `mulx`. `pext` is the fast path for
    /// extracting a selection vector from a bitmap, which is the shape of every Arrow filter.
    pub bmi2: bool,
    /// POPCNT — scalar population count, the null-count and selectivity primitive.
    pub popcnt: bool,
    /// AES-NI — hardware AES rounds, which is the backend `ahash` picks when compiled for it.
    pub aes: bool,

    // ---- aarch64 ----
    /// NEON / ASIMD — baseline 128-bit SIMD on aarch64.
    pub neon: bool,
    /// SVE — scalable vectors (Graviton 3+, Neoverse V1/V2).
    pub sve: bool,
    /// SVE2 — the second-generation scalable set (Graviton 4, Neoverse V2).
    pub sve2: bool,
    /// SDOT/UDOT — integer dot product, aarch64's VNNI equivalent.
    pub dotprod: bool,
    /// FP16 arithmetic (not just conversion).
    pub fp16: bool,
    /// Hardware CRC32.
    pub crc: bool,
}

impl IsaFeatures {
    /// The detected host capabilities (cached after the first call).
    pub fn detect() -> &'static IsaFeatures {
        static FEATURES: OnceLock<IsaFeatures> = OnceLock::new();
        FEATURES.get_or_init(detect_raw)
    }

    /// The widest vector register the host can use, in bytes.
    ///
    /// 64 with AVX-512, 32 with AVX2/AVX, 16 with SSE2 or NEON, 8 (a scalar word) otherwise.
    /// SVE is reported as 16 because its register width is implementation-defined and not
    /// discoverable through this interface — sizing to the guaranteed minimum is the correct
    /// conservative answer.
    pub fn vector_bytes(&self) -> usize {
        if self.avx512f {
            64
        } else if self.avx2 || self.avx {
            32
        } else if cfg!(target_arch = "x86_64") || self.neon || self.sve {
            16
        } else {
            8
        }
    }

    /// Whether 512-bit code is likely to be a win rather than a clock-frequency loss.
    ///
    /// A **heuristic**, not a capability bit — see the module docs. `avx512vpopcntdq` is used
    /// as an Ice Lake-SP / Zen 4 generation marker: those parts largely removed the AVX-512
    /// license-frequency penalty, and every Skylake-derived part (which has it at its worst)
    /// lacks the bit. Treat a `true` here as "worth benchmarking the 512-bit override on this
    /// host", never as "the engine should widen automatically".
    pub fn avx512_is_cheap(&self) -> bool {
        self.avx512f && self.avx512vpopcntdq
    }

    /// A short, stable label for the widest ISA family available, for logs and telemetry.
    ///
    /// Deliberately coarse: it names the dispatch tier a kernel would pick, not the full
    /// feature vector, so it stays comparable across hosts in an aggregated metric.
    pub fn tier(&self) -> &'static str {
        if self.avx512f {
            "avx512"
        } else if self.avx2 {
            "avx2"
        } else if self.avx {
            "avx"
        } else if self.sve2 {
            "sve2"
        } else if self.sve {
            "sve"
        } else if self.neon {
            "neon"
        } else if cfg!(target_arch = "x86_64") {
            "sse2"
        } else {
            "scalar"
        }
    }

    /// Every detected capability by name, sorted, for reporting.
    ///
    /// Sorted so two hosts' lists diff cleanly and a golden test can pin one.
    pub fn names(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        for (present, name) in [
            (self.aes, "aes"),
            (self.avx, "avx"),
            (self.avx2, "avx2"),
            (self.avx512bw, "avx512bw"),
            (self.avx512dq, "avx512dq"),
            (self.avx512f, "avx512f"),
            (self.avx512vl, "avx512vl"),
            (self.avx512vnni, "avx512vnni"),
            (self.avx512vpopcntdq, "avx512vpopcntdq"),
            (self.bmi1, "bmi1"),
            (self.bmi2, "bmi2"),
            (self.crc, "crc"),
            (self.dotprod, "dotprod"),
            (self.fma, "fma"),
            (self.fp16, "fp16"),
            (self.neon, "neon"),
            (self.popcnt, "popcnt"),
            (self.sse4_2, "sse4_2"),
            (self.sve, "sve"),
            (self.sve2, "sve2"),
        ] {
            if present {
                out.push(name);
            }
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
fn detect_raw() -> IsaFeatures {
    IsaFeatures {
        sse4_2: std::is_x86_feature_detected!("sse4.2"),
        avx: std::is_x86_feature_detected!("avx"),
        avx2: std::is_x86_feature_detected!("avx2"),
        fma: std::is_x86_feature_detected!("fma"),
        avx512f: std::is_x86_feature_detected!("avx512f"),
        avx512bw: std::is_x86_feature_detected!("avx512bw"),
        avx512dq: std::is_x86_feature_detected!("avx512dq"),
        avx512vl: std::is_x86_feature_detected!("avx512vl"),
        avx512vpopcntdq: std::is_x86_feature_detected!("avx512vpopcntdq"),
        avx512vnni: std::is_x86_feature_detected!("avx512vnni"),
        bmi1: std::is_x86_feature_detected!("bmi1"),
        bmi2: std::is_x86_feature_detected!("bmi2"),
        popcnt: std::is_x86_feature_detected!("popcnt"),
        aes: std::is_x86_feature_detected!("aes"),
        ..IsaFeatures::default()
    }
}

#[cfg(target_arch = "aarch64")]
fn detect_raw() -> IsaFeatures {
    IsaFeatures {
        // NEON is architecturally mandatory on aarch64, so it is reported unconditionally
        // rather than probed — the probe exists but can only ever answer `true`.
        neon: true,
        sve: std::arch::is_aarch64_feature_detected!("sve"),
        sve2: std::arch::is_aarch64_feature_detected!("sve2"),
        dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
        fp16: std::arch::is_aarch64_feature_detected!("fp16"),
        crc: std::arch::is_aarch64_feature_detected!("crc"),
        aes: std::arch::is_aarch64_feature_detected!("aes"),
        ..IsaFeatures::default()
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect_raw() -> IsaFeatures {
    IsaFeatures::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_is_cached() {
        assert!(std::ptr::eq(IsaFeatures::detect(), IsaFeatures::detect()));
    }

    #[test]
    fn feature_implications_hold() {
        let f = IsaFeatures::detect();
        // The ISA extensions are strictly cumulative: a part with the wider set always has
        // the narrower one. A violation means detection is reading the wrong bit.
        if f.avx2 {
            assert!(f.avx, "AVX2 implies AVX");
        }
        if f.avx512f {
            assert!(f.avx2 && f.avx, "AVX-512F implies AVX2");
        }
        if f.avx512vl || f.avx512bw || f.avx512dq {
            assert!(f.avx512f, "an AVX-512 subset implies the foundation");
        }
        if f.avx512_is_cheap() {
            assert!(f.avx512f && f.avx512vpopcntdq);
        }
        if f.sve2 {
            assert!(f.sve, "SVE2 implies SVE");
        }
        #[cfg(target_arch = "aarch64")]
        assert!(f.neon, "NEON is baseline on aarch64");
        #[cfg(target_arch = "x86_64")]
        assert!(
            f.sse4_2,
            "SSE4.2 is baseline on every x86_64 part the engine targets"
        );
    }

    #[test]
    fn vector_width_matches_the_widest_detected_set() {
        let f = IsaFeatures::detect();
        let bytes = f.vector_bytes();
        assert!(bytes.is_power_of_two() && (8..=64).contains(&bytes));
        if f.avx512f {
            assert_eq!(bytes, 64);
        } else if f.avx2 {
            assert_eq!(bytes, 32);
        }
    }

    #[test]
    fn synthetic_widths_are_computed_not_guessed() {
        let avx512 = IsaFeatures {
            avx: true,
            avx2: true,
            avx512f: true,
            ..IsaFeatures::default()
        };
        assert_eq!(avx512.vector_bytes(), 64);
        assert_eq!(avx512.tier(), "avx512");

        let avx2 = IsaFeatures {
            avx: true,
            avx2: true,
            ..IsaFeatures::default()
        };
        assert_eq!(avx2.vector_bytes(), 32);
        assert_eq!(avx2.tier(), "avx2");

        let neon = IsaFeatures {
            neon: true,
            ..IsaFeatures::default()
        };
        assert_eq!(neon.vector_bytes(), 16);
    }

    #[test]
    fn the_downclock_heuristic_rejects_skylake_derived_parts() {
        // Cascade Lake: AVX-512F + VNNI, no VPOPCNTDQ. This is exactly the shape that must
        // NOT be reported as cheap — it is the part with the worst license-frequency penalty,
        // and a VNNI-based heuristic would have called it cheap.
        let cascade_lake = IsaFeatures {
            avx: true,
            avx2: true,
            avx512f: true,
            avx512bw: true,
            avx512dq: true,
            avx512vl: true,
            avx512vnni: true,
            avx512vpopcntdq: false,
            ..IsaFeatures::default()
        };
        assert!(!cascade_lake.avx512_is_cheap());
        // Ice Lake-SP / Zen 4: VPOPCNTDQ present.
        let ice_lake = IsaFeatures {
            avx512vpopcntdq: true,
            ..cascade_lake
        };
        assert!(ice_lake.avx512_is_cheap());
    }

    #[test]
    fn names_are_sorted_and_only_list_present_features() {
        let f = IsaFeatures::detect();
        let names = f.names();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(
            names, sorted,
            "names must be sorted so two hosts diff cleanly"
        );
        assert_eq!(names.contains(&"avx2"), f.avx2);
        assert_eq!(names.contains(&"avx512f"), f.avx512f);
        assert!(IsaFeatures::default().names().is_empty());
    }

    #[test]
    fn tier_is_the_widest_family_present() {
        let f = IsaFeatures::detect();
        let tier = f.tier();
        assert!(!tier.is_empty());
        if f.avx512f {
            assert_eq!(tier, "avx512");
        }
        assert_eq!(
            IsaFeatures::default().tier(),
            if cfg!(target_arch = "x86_64") {
                "sse2"
            } else {
                "scalar"
            }
        );
    }
}
