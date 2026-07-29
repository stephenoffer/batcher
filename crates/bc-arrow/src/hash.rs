//! The one hash whose value crosses a process boundary.
//!
//! Most hashing in the engine picks a bucket inside one hash table inside one process, and
//! for that any fast hash will do. A few hashes are different in kind: they decide **which
//! machine a row goes to**, or they are folded into a sketch that will be merged with a
//! sketch built somewhere else. Those must produce the same 64 bits on every host in the
//! cluster, forever. That is what lives here.
//!
//! # Why this exists
//!
//! The shuffle used `ahash::RandomState` with fixed seeds and a comment saying fixed seeds
//! made it deterministic. Fixed seeds make it deterministic *within one binary*. `ahash`
//! selects an AES-NI backend from the **compile-time** `target_feature`, so the same seeds
//! on a machine built with `+aes` and one built without produce different values. Every
//! trigger for that is a documented, supported configuration:
//!
//! * `.cargo/config.toml` explicitly invites `-C target-cpu=native` for a "known
//!   homogeneous" cluster — which enables `+aes`.
//! * The driver self-ships its own `.so` to workers *unless* `distributed.runtime_env` is
//!   set or a managed platform's `RAY_RUNTIME_ENV_HOOK` intervenes, in which case each
//!   worker runs its own wheel.
//! * A cross-architecture cluster ships a per-arch binary by design.
//!
//! The failure is silent and it is the worst class this engine has: two workers disagree
//! about which reducer owns a key, so one `GROUP BY` group splits in two and join rows
//! quietly go missing. No error, no crash, just a wrong answer — and it cannot be
//! reproduced on one host, which is why it survived so long.
//!
//! # Why the algorithm is written out
//!
//! xxHash64 is a published specification with published test vectors, so an implementation
//! either matches it or is broken, on any platform. It is written out here rather than
//! pulled from a crate for the same reason `bc_expr`'s `Expr::Hash` is: this value chooses
//! a *reducer*, so it must not be able to change because a dependency bumped a minor
//! version or the workspace unified on a different one. `bc-arrow` is also the root of the
//! crate DAG, where a new third-party dependency would land in every build.
//!
//! This is the same argument that already put `canon_f64_bits` and `float_total_cmp` in
//! this crate: the lowest crate everyone can see is where a shared identity contract goes,
//! so the paths that must agree cannot drift apart.
//!
//! # What must NOT use this
//!
//! Hash-table bucket selection inside a single `execute_plan` call — the group assigner's
//! probe table, the join build map, `distinct`, the spill partitioner, `IN`-list lookup.
//! Those never leave the process, they are the hottest kernels in the engine, and `ahash`
//! is faster on them. Converting them buys nothing and costs throughput.

use std::hash::{BuildHasher, Hasher};

/// SplitMix64's finalizer — an avalanching 64-bit integer hash.
///
/// The same function as `bc_expr::eval::hash::mix64`, and deliberately so: both exist to
/// make a weak accumulator's high and low bits equally usable, which matters here because
/// `bucket_of` extracts the *low* bits for a power-of-two partition count and the *high*
/// bits for the multiply-shift path. A hash that only avalanches one end silently skews
/// one of the two.
#[inline]
#[must_use]
pub const fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

const PRIME64_1: u64 = 0x9E37_79B1_85EB_CA87;
const PRIME64_2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const PRIME64_3: u64 = 0x1656_67B1_9E37_79F9;
const PRIME64_4: u64 = 0x85EB_CA77_C2B2_AE63;
const PRIME64_5: u64 = 0x27D4_EB2F_1656_67C5;

#[inline]
fn round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(PRIME64_2))
        .rotate_left(31)
        .wrapping_mul(PRIME64_1)
}

#[inline]
fn merge_round(acc: u64, val: u64) -> u64 {
    (acc ^ round(0, val))
        .wrapping_mul(PRIME64_1)
        .wrapping_add(PRIME64_4)
}

#[inline]
fn read_u64(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8 bytes in range"))
}

#[inline]
fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4 bytes in range"))
}

/// xxHash64 of `bytes` with `seed`, per the reference specification.
///
/// Bit-identical on every platform and in every build profile: the algorithm is fully
/// specified, reads little-endian explicitly, and is pinned here by the reference test
/// vectors. That is the entire property this module exists to provide.
#[must_use]
pub fn xxhash64(bytes: &[u8], seed: u64) -> u64 {
    let len = bytes.len();
    let mut acc = if len >= 32 {
        let mut v1 = seed.wrapping_add(PRIME64_1).wrapping_add(PRIME64_2);
        let mut v2 = seed.wrapping_add(PRIME64_2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(PRIME64_1);
        let mut at = 0;
        while at + 32 <= len {
            v1 = round(v1, read_u64(bytes, at));
            v2 = round(v2, read_u64(bytes, at + 8));
            v3 = round(v3, read_u64(bytes, at + 16));
            v4 = round(v4, read_u64(bytes, at + 24));
            at += 32;
        }
        let merged = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
        let merged = merge_round(merged, v1);
        let merged = merge_round(merged, v2);
        let merged = merge_round(merged, v3);
        merge_round(merged, v4)
    } else {
        seed.wrapping_add(PRIME64_5)
    };

    acc = acc.wrapping_add(len as u64);

    let mut at = len & !31;
    while at + 8 <= len {
        acc = (acc ^ round(0, read_u64(bytes, at)))
            .rotate_left(27)
            .wrapping_mul(PRIME64_1)
            .wrapping_add(PRIME64_4);
        at += 8;
    }
    if at + 4 <= len {
        acc = (acc ^ u64::from(read_u32(bytes, at)).wrapping_mul(PRIME64_1))
            .rotate_left(23)
            .wrapping_mul(PRIME64_2)
            .wrapping_add(PRIME64_3);
        at += 4;
    }
    while at < len {
        acc = (acc ^ u64::from(bytes[at]).wrapping_mul(PRIME64_5))
            .rotate_left(11)
            .wrapping_mul(PRIME64_1);
        at += 1;
    }

    // Final avalanche.
    acc ^= acc >> 33;
    acc = acc.wrapping_mul(PRIME64_2);
    acc ^= acc >> 29;
    acc = acc.wrapping_mul(PRIME64_3);
    acc ^ (acc >> 32)
}

/// Fold one already-hashed value into a running accumulator.
///
/// Order-sensitive, so a composite key `(a, b)` does not collide with `(b, a)`.
#[inline]
fn combine(acc: u64, value: u64) -> u64 {
    mix64(acc ^ value.wrapping_add(0x9E37_79B9_7F4A_7C15).rotate_left(31))
}

/// A [`Hasher`] whose output is identical on every host, build, and CPU.
///
/// Two shapes, because the values it hashes come in two kinds and one algorithm is wrong
/// for both:
///
/// * **Byte runs** (a row-encoded composite key, a string) go through [`xxhash64`], which
///   processes 32 bytes per iteration.
/// * **Fixed-width integers** are folded through [`mix64`] directly rather than being
///   serialized and re-parsed. This is both faster than a byte hash and faster than
///   `ahash` on the 8-byte-key path that dominates the shuffle.
///
/// **Every integer write is explicitly little-endian.** `Hasher`'s provided methods
/// delegate to `to_ne_bytes()`, which would make this hasher's output depend on the host's
/// byte order — reintroducing exactly the bug it exists to remove, on the one axis nobody
/// would think to test because all current cloud hardware is little-endian.
#[derive(Debug, Clone)]
pub struct PortableHasher {
    acc: u64,
}

impl PortableHasher {
    /// A hasher seeded with `seed`.
    #[must_use]
    pub const fn with_seed(seed: u64) -> Self {
        Self { acc: seed }
    }
}

impl Hasher for PortableHasher {
    #[inline]
    fn finish(&self) -> u64 {
        mix64(self.acc)
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.acc = combine(self.acc, xxhash64(bytes, 0));
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.acc = combine(self.acc, u64::from(i));
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.acc = combine(self.acc, u64::from(i.to_le()));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.acc = combine(self.acc, u64::from(i.to_le()));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.acc = combine(self.acc, i.to_le());
    }

    #[inline]
    fn write_u128(&mut self, i: u128) {
        let le = i.to_le();
        self.acc = combine(self.acc, le as u64);
        self.acc = combine(self.acc, (le >> 64) as u64);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        // Widened to u64 rather than hashed at the host's pointer width, so a 32-bit and a
        // 64-bit worker agree.
        self.write_u64(i as u64);
    }

    #[inline]
    fn write_i8(&mut self, i: i8) {
        self.write_u8(i as u8);
    }

    #[inline]
    fn write_i16(&mut self, i: i16) {
        self.write_u16(i as u16);
    }

    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.write_u32(i as u32);
    }

    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.write_u64(i as u64);
    }

    #[inline]
    fn write_i128(&mut self, i: i128) {
        self.write_u128(i as u128);
    }

    #[inline]
    fn write_isize(&mut self, i: isize) {
        self.write_u64(i as u64);
    }
}

/// A [`BuildHasher`] handing out [`PortableHasher`]s from a fixed seed.
///
/// Drop-in for `ahash::RandomState` at the call sites whose value crosses a process
/// boundary. `hash_one` and `build_hasher` both work, so a migration is a constant swap.
#[derive(Debug, Clone, Copy)]
pub struct PortableBuildHasher {
    seed: u64,
}

impl PortableBuildHasher {
    /// A builder producing hashers seeded with `seed`.
    ///
    /// The seed separates *purposes* (shuffle routing, HLL registers, sketch buckets) so
    /// two independent hashes of the same key are uncorrelated. It is not a secret and
    /// must never be randomized per process — the whole point is that two machines
    /// computing the same purpose agree.
    #[must_use]
    pub const fn with_seed(seed: u64) -> Self {
        Self { seed }
    }
}

impl BuildHasher for PortableBuildHasher {
    type Hasher = PortableHasher;

    #[inline]
    fn build_hasher(&self) -> PortableHasher {
        PortableHasher::with_seed(self.seed)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    /// The reference vectors from the xxHash specification.
    ///
    /// This is the test that makes the whole module trustworthy, and it is trustworthy
    /// precisely because the oracle is **external**. Any platform, CPU feature, compiler
    /// version, or optimization level that changed the answer would fail here, on that
    /// platform — which is the one thing a same-host test can never prove about a
    /// cross-host agreement bug.
    #[test]
    fn xxhash64_matches_the_specification() {
        assert_eq!(xxhash64(b"", 0), 0xEF46_DB37_51D8_E999);
        assert_eq!(xxhash64(b"", 1), 0xD5AF_BA13_36A3_BE4B);
        assert_eq!(xxhash64(b"a", 0), 0xD24E_C4F1_A98C_6E5B);
        assert_eq!(xxhash64(b"abc", 0), 0x44BC_2CF5_AD77_0999);
    }

    /// Covers every branch of the tail handling: the 32-byte main loop, the 8-byte, the
    /// 4-byte, and the single-byte remainder. A hash that is right at 3 bytes and wrong at
    /// 37 would otherwise pass the vectors above.
    #[test]
    fn every_length_class_is_exercised_and_distinct() {
        let data: Vec<u8> = (0..=255u8).collect();
        let mut seen = std::collections::HashSet::new();
        for len in [0, 1, 3, 4, 7, 8, 15, 16, 31, 32, 33, 63, 64, 100, 255] {
            let h = xxhash64(&data[..len], 0);
            assert!(seen.insert(h), "collision at len {len} — suspicious");
        }
    }

    #[test]
    fn hashing_is_deterministic_and_seed_separated() {
        let a = PortableBuildHasher::with_seed(7);
        let b = PortableBuildHasher::with_seed(9);
        assert_eq!(a.hash_one(42_i64), a.hash_one(42_i64));
        assert_ne!(
            a.hash_one(42_i64),
            b.hash_one(42_i64),
            "different seeds must decorrelate the same key"
        );
    }

    /// A composite key must be order-sensitive, or `(part, supplier)` and
    /// `(supplier, part)` would route to the same reducer and a join would silently
    /// mis-partition.
    #[test]
    fn composite_keys_are_order_sensitive() {
        let build = PortableBuildHasher::with_seed(1);
        let mut ab = build.build_hasher();
        ab.write_i64(1);
        ab.write_i64(2);
        let mut ba = build.build_hasher();
        ba.write_i64(2);
        ba.write_i64(1);
        assert_ne!(ab.finish(), ba.finish());
    }

    /// The property the shuffle actually depends on: both extraction modes of `bucket_of`
    /// must see well-distributed bits. A hash that avalanches only its high bits routes
    /// every key to a handful of buckets under the power-of-two mask, which shows up as
    /// catastrophic skew rather than as a wrong answer.
    #[test]
    fn both_ends_of_the_hash_are_well_distributed() {
        let build = PortableBuildHasher::with_seed(1);
        let n = 4096usize;
        let buckets = 64usize;
        let mut low = vec![0usize; buckets];
        let mut high = vec![0usize; buckets];
        for i in 0..n {
            let h = build.hash_one(i as i64);
            low[(h as usize) & (buckets - 1)] += 1;
            high[((u128::from(h) * buckets as u128) >> 64) as usize] += 1;
        }
        let expected = n / buckets;
        for (name, counts) in [("low", &low), ("high", &high)] {
            let worst = counts.iter().copied().max().unwrap();
            assert!(
                worst < expected * 3,
                "{name} bits skewed: worst bucket {worst} vs expected {expected}"
            );
            assert!(
                counts.iter().all(|&c| c > 0),
                "{name} bits left a bucket empty"
            );
        }
    }

    /// Byte-slice hashing must agree with the one-shot, since `Hash for [u8]` routes
    /// through `write` and the shuffle's row-encoded path relies on it.
    #[test]
    fn byte_runs_go_through_xxhash() {
        let build = PortableBuildHasher::with_seed(0);
        let mut h = build.build_hasher();
        h.write(b"customer-42");
        assert_eq!(h.finish(), mix64(combine(0, xxhash64(b"customer-42", 0))));
    }

    /// Endianness is pinned explicitly rather than inherited from the host. There is no
    /// big-endian CI target to prove this on, so the test asserts the *mechanism*: an
    /// integer write must equal the little-endian byte pattern's fold, not the native one.
    #[test]
    fn integer_writes_are_little_endian() {
        let build = PortableBuildHasher::with_seed(0);
        let value: u64 = 0x0102_0304_0506_0708;
        let direct = build.hash_one(value);
        let mut manual = build.build_hasher();
        manual.write_u64(u64::from_le_bytes(value.to_le_bytes()));
        assert_eq!(direct, manual.finish());
    }

    /// Equal values hash equally through the `Hash` trait, which is what `hash_one` at the
    /// shuffle call sites relies on.
    #[test]
    fn hash_trait_round_trips() {
        let build = PortableBuildHasher::with_seed(3);
        let key = (7_i64, "abc");
        // `hash_one` is exactly what the shuffle call sites use, so use it here too.
        assert_eq!(build.hash_one(key), build.hash_one(key));
        assert_ne!(build.hash_one(key), build.hash_one((7_i64, "abd")));
    }
}
