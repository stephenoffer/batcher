//! `bc_arrow::xxhash64` against an independent implementation, over every length class.
//!
//! `bc-arrow` writes xxHash64 out by hand rather than depending on a crate, because the
//! value chooses which reducer a shuffled row lands on and therefore must not be able to
//! change when a dependency bumps a version. The cost of writing it out is that a subtle
//! error — a wrong tail branch, a mis-transcribed prime — would be invisible: it would be
//! self-consistent, deterministic, and wrong, and every same-host test would pass.
//!
//! The specification vectors in `bc-arrow` guard the common cases. This guards the rest,
//! by differential-testing all 256 lengths across four seeds against `twox-hash`. It lives
//! in `bc-expr` rather than `bc-arrow` because that is where `twox-hash` is already a
//! dependency, and `bc-arrow` should not gain one for a test — it sits at the root of the
//! crate DAG, so anything added there lands in every build.

use std::hash::Hasher;

fn reference(bytes: &[u8], seed: u64) -> u64 {
    let mut hasher = twox_hash::XxHash64::with_seed(seed);
    hasher.write(bytes);
    hasher.finish()
}

/// Every length from 0 to 255 crosses all four tail branches (the 32-byte main loop, the
/// 8-byte, the 4-byte, and the single-byte remainder) and every combination of them.
#[test]
fn agrees_with_an_independent_implementation() {
    let data: Vec<u8> = (0..=255u8).collect();
    for seed in [0_u64, 1, 42, u64::MAX] {
        for len in 0..=data.len() {
            assert_eq!(
                bc_arrow::xxhash64(&data[..len], seed),
                reference(&data[..len], seed),
                "diverged at len={len} seed={seed}"
            );
        }
    }
}
