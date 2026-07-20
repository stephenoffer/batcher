//! Parallel hash-table build — shard the heads by hash so every core builds at once.
//!
//! The build loop was the join's sequential prefix. It is not a small one: profiling TPC-H q5
//! (sf1, 48 cores) found the *streaming* broadcast join spending **10.7 ms building a
//! 227,597-row table on one thread** and only 6.0 ms probing it with 1.2M rows across all 48
//! — ~46 ns per build row, with 47 cores idle. Across q5's joins that is ~17.6 ms of a 64.5 ms
//! query, and it is pure Amdahl: the probe already scales, so the build is what caps the join.
//!
//! (The *partitioned* path hides this by building one small table per bucket in parallel — but
//! it buys that by scattering the whole probe **payload** into buckets first, measured at 20 ms
//! for the same join. Each path paid for exactly what the other avoided. Sharding the build
//! here is what lets the streaming path keep its no-copy probe *and* build in parallel.)
//!
//! The shard is a slice of the hash, so a key's shard is a pure function of its hash: build and
//! probe agree without communicating, and a chain never spans two shards. Each shard therefore
//! owns a disjoint set of build rows and can be built with no synchronization at all.
//!
//! **Bit-identical to the serial build.** [`radix::partition_side`] hands each shard its rows in
//! ascending `abs_row` order, and the chain prepends — exactly as the serial loop did, walking
//! rows in increasing order and prepending. So every key's chain comes out in the same
//! (descending-row) order, every probe walks it in the same order, and the join emits the same
//! rows in the same sequence. The `seq == par` oracle sees no difference; only the clock does.

use bc_sketches::{BloomFilter, Mergeable};
use hashbrown::hash_table::Entry;
use hashbrown::HashTable;
use rayon::prelude::*;

use super::radix;
use super::JoinKeys;

/// Below this many build rows the serial loop wins: the table is small (a few hundred µs) and
/// the partition pass plus rayon fan-out would cost more than they save. It also keeps the
/// *bucketed* join — which already builds one table per bucket inside a `par_iter`, each bucket
/// a fraction of the build — on the single-shard path, so this never nests a parallel build
/// inside a parallel one.
const PARALLEL_BUILD_MIN_ROWS: usize = 1 << 14; // 16,384 — one morsel

/// Cap on shards. Past this the tables get too small to amortize their own allocation, and the
/// probe's shard indirection starts to cost more than the build saves.
const MAX_SHARDS: usize = 64;

/// The shard holding `hash`. Reads the **high** bits: hashbrown indexes its buckets with the
/// low bits, so sharding on those would hand every shard a table whose keys all collide into
/// the same region. `shards` is always a power of two, so this is a mask.
#[inline]
pub(super) fn shard_of(hash: u64, shards: usize) -> usize {
    ((hash >> 32) as usize) & (shards - 1)
}

/// How many shards to build `rows` across — 1 (the serial path) when the build is small.
pub(super) fn shard_count(rows: usize) -> usize {
    if rows < PARALLEL_BUILD_MIN_ROWS {
        return 1;
    }
    let threads = rayon::current_num_threads().max(1);
    // A power of two so `shard_of` is a mask; never below 2 (we are past the threshold, so
    // there is real work to split) and never above the cap.
    threads
        .min(MAX_SHARDS)
        .next_power_of_two()
        .clamp(2, MAX_SHARDS)
}

/// One shard's build output: its head table, the `(row, next)` chain links it stitched, and its
/// share of the probe bloom. Merged into the whole table by [`build_sharded`].
type ShardBuild = (HashTable<u32>, Vec<(u32, u32)>, Option<BloomFilter>);

/// The chained hash table, built across every core.
///
/// Returns the per-shard heads (each holding **absolute** build-row indices), the `next` chain
/// (also absolute, `u32::MAX` terminating), and the optional probe bloom. Null-key rows are
/// never inserted — NULL never matches — so a chain head is always a real match.
pub(super) fn build_sharded<K: JoinKeys + Sync>(
    keys: &K,
    state: &ahash::RandomState,
    right_rows: usize,
    right_null: &[bool],
    shards: usize,
    bloom: Option<BloomFilter>,
) -> (Vec<HashTable<u32>>, Vec<u32>, Option<BloomFilter>, bool) {
    // `partition_side` carries the hash itself as the partition's key, so it is computed once
    // here and reused by both the insert below and the bloom — the serial build hashed each
    // row exactly once too.
    let parts = radix::partition_side(
        |i| keys.hash_right(state, i),
        right_null,
        shards,
        |h| shard_of(*h, shards),
    );

    // Each shard builds its own table and its own slice of the chain. The `(row, next)` writes
    // are collected rather than written through a shared `&mut [u32]`: the shards are disjoint
    // by construction, but applying them in one serial pass afterwards costs ~0.2 ms on a
    // 228k-row build against the ~10 ms this removes, and needs no `unsafe` to justify.
    let built: Vec<ShardBuild> = parts
        .par_iter()
        .map(|rows| {
            let mut heads: HashTable<u32> = HashTable::with_capacity(rows.len());
            let mut chain: Vec<(u32, u32)> = Vec::new();
            // Same dimensions as the caller's, so the union below is a plain bit-OR.
            let mut shard_bloom = bloom
                .as_ref()
                .map(|b| BloomFilter::new(b.num_bits(), b.num_hashes()));
            for &(hash, abs) in rows {
                if let Some(b) = shard_bloom.as_mut() {
                    b.add_hash(hash);
                }
                match heads.entry(
                    hash,
                    |&h| keys.right_eq_right(h as usize, abs as usize),
                    |&h| keys.hash_right(state, h as usize),
                ) {
                    // Prepend, exactly as the serial loop did — see the module docs on why the
                    // resulting chain order is what keeps this bit-identical.
                    Entry::Occupied(mut e) => {
                        chain.push((abs, *e.get()));
                        *e.get_mut() = abs;
                    }
                    Entry::Vacant(e) => {
                        e.insert(abs);
                    }
                }
            }
            (heads, chain, shard_bloom)
        })
        .collect();

    // A shard pushes to `chain` only on an `Occupied` entry, i.e. only when a key repeats. So
    // "every chain is empty" is exactly "the build key is unique", decided here for free from
    // work already done. On a unique build (any join to a primary key — `orders.o_orderkey`,
    // `part.p_partkey`, every dimension table) `next` would then be `u32::MAX` in every slot and
    // never read, so allocating and zeroing it is pure waste: 24 MB of memset at 6M build rows,
    // serial, on the critical path. See `JoinTable::unique` for the probe-side half.
    let unique = built.iter().all(|(_, chain, _)| chain.is_empty());
    let mut next: Vec<u32> = if unique {
        Vec::new()
    } else {
        vec![u32::MAX; right_rows]
    };
    let mut heads = Vec::with_capacity(shards);
    let mut merged = bloom;
    for (shard_heads, chain, shard_bloom) in built {
        // Empty for every shard when `unique` — so this never indexes the empty `next`.
        for (row, nxt) in chain {
            next[row as usize] = nxt;
        }
        if let (Some(m), Some(s)) = (merged.as_mut(), shard_bloom.as_ref()) {
            m.merge(s);
        }
        heads.push(shard_heads);
    }
    (heads, next, merged, unique)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::join::{hash_join_indices, JoinIndices, JoinType};
    use arrow::array::{Array, ArrayRef, Int64Array};
    use std::sync::Arc;

    /// Distinct build keys, each repeated 3x below: 36,000 build rows. That lands in the
    /// window this table actually serves — past `PARALLEL_BUILD_MIN_ROWS` (so it *shards*) but
    /// under `RADIX_MIN_BUILD_ROWS` (so the flat path runs, and the output stays in probe-row
    /// order, which is what makes the strict sequence assertion below meaningful).
    /// `the_test_is_actually_on_the_sharded_flat_path` pins both ends so a future threshold
    /// change cannot quietly slide this test onto a path it does not mean to test.
    const KEYS: i64 = 12_000;
    const BUILD_ROWS: usize = 3 * KEYS as usize;

    /// One key per row, no duplicates — 50,000 rows, also inside the window.
    const BIG: i64 = 50_000;

    #[test]
    fn the_test_is_actually_on_the_sharded_flat_path() {
        for rows in [BUILD_ROWS, BIG as usize] {
            assert!(
                shard_count(rows) >= 2,
                "must shard, else this tests nothing new"
            );
            assert!(
                rows < crate::join::RADIX_MIN_BUILD_ROWS,
                "must stay on the flat path, whose output is in probe-row order"
            );
        }
    }

    fn keys(v: Vec<Option<i64>>) -> Vec<ArrayRef> {
        vec![Arc::new(Int64Array::from(v)) as ArrayRef]
    }

    fn pairs(idx: &JoinIndices) -> Vec<(Option<u32>, Option<u32>)> {
        (0..idx.left.len())
            .map(|i| {
                let l = (!idx.left.is_null(i)).then(|| idx.left.value(i));
                let r = (!idx.right.is_null(i)).then(|| idx.right.value(i));
                (l, r)
            })
            .collect()
    }

    #[test]
    fn the_build_shards_once_it_is_worth_sharding() {
        assert_eq!(
            shard_count(PARALLEL_BUILD_MIN_ROWS - 1),
            1,
            "small builds stay flat"
        );
        assert!(
            shard_count(BIG as usize) >= 2,
            "a large build must parallelize"
        );
        assert!(
            shard_count(BIG as usize).is_power_of_two(),
            "shard_of masks"
        );
    }

    /// A key's shard is a pure function of its hash, so build and probe agree with no
    /// coordination. If this ever drifts, a probe looks in the wrong shard and silently
    /// finds nothing — a wrong answer, not a crash.
    #[test]
    fn every_hash_lands_in_a_real_shard() {
        for shards in [1usize, 2, 4, 16, 64] {
            for h in [0u64, 1, u64::MAX, 0x1234_5678_9abc_def0] {
                assert!(shard_of(h, shards) < shards);
            }
        }
        assert_eq!(shard_of(u64::MAX, 1), 0, "one shard is always shard zero");
    }

    /// The claim the whole change rests on: sharding the build is invisible in the result.
    /// A sharded build with duplicate keys must emit the same pairs, in the same order, as
    /// the flat one — same chain order, same rows. Duplicates are the sharp edge: they are
    /// what the `next` chain exists for, and a mis-stitched chain would reorder or drop them.
    #[test]
    fn a_sharded_build_emits_exactly_what_a_flat_build_emits() {
        // Every build key appears 3x, so each chain is 3 long and its order is observable.
        // Duplicates are the sharp edge: they are what the `next` chain exists for, and a
        // mis-stitched chain would reorder, drop, or duplicate them.
        let build: Vec<Option<i64>> = (0..KEYS)
            .flat_map(|k| [Some(k), Some(k), Some(k)])
            .collect();
        assert_eq!(build.len(), BUILD_ROWS);
        let probe: Vec<Option<i64>> = (0..KEYS)
            .map(|k| if k % 7 == 0 { None } else { Some(k) })
            .chain([Some(KEYS + 1)]) // a probe key nothing matches
            .collect();

        let got = pairs(
            &hash_join_indices(&keys(probe.clone()), &keys(build), JoinType::Inner).expect("join"),
        );

        // The serial build walks rows in increasing order and prepending, so a key's chain is
        // its build rows in DESCENDING order, and the flat probe emits in probe-row order.
        // The sharded build must reproduce that sequence exactly — see the module docs.
        let mut want: Vec<(Option<u32>, Option<u32>)> = Vec::new();
        for (l, k) in probe.iter().enumerate() {
            let Some(k) = k else { continue };
            if *k >= KEYS {
                continue;
            }
            let base = (*k as u32) * 3;
            for r in [base + 2, base + 1, base] {
                want.push((Some(l as u32), Some(r)));
            }
        }
        assert_eq!(got.len(), want.len(), "row count");
        assert_eq!(
            got, want,
            "the sharded build must emit the flat build's exact sequence"
        );
    }

    /// The probe-side bloom is a pure performance short-circuit even on the **sharded**
    /// build path, where each shard fills its own bloom and they are bit-OR merged. A
    /// merge that dropped a set bit would be a false negative — a silently dropped match.
    /// Force the bloom on over a build large enough to shard and assert the relation is
    /// identical to the bloom-off run, for every left-driven join type.
    #[test]
    fn sharded_build_bloom_never_drops_a_match() {
        use crate::join::hash_join_indices_impl;
        // 40k distinct build keys (each once) -> shards, and every probe key is present so
        // any false-negative bloom rejection would delete a real match.
        let build: Vec<Option<i64>> = (0..40_000i64).map(Some).collect();
        assert!(shard_count(build.len()) >= 2, "must shard");
        // probe: every build key plus some misses, and some nulls.
        let probe: Vec<Option<i64>> = (0..40_000i64)
            .map(|k| if k % 11 == 0 { None } else { Some(k) })
            .chain((40_000..40_050).map(Some))
            .collect();
        for jt in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Semi,
            JoinType::Anti,
        ] {
            let with =
                hash_join_indices_impl(&keys(probe.clone()), &keys(build.clone()), jt, true, 0.01)
                    .unwrap();
            let without =
                hash_join_indices_impl(&keys(probe.clone()), &keys(build.clone()), jt, false, 0.01)
                    .unwrap();
            let mut a = pairs(&with);
            let mut b = pairs(&without);
            a.sort();
            b.sort();
            assert_eq!(a, b, "sharded bloom-on disagrees with bloom-off for {jt:?}");
        }
    }

    /// Nulls never join (NULL != NULL), and a `Left` join must still emit their probe rows
    /// unmatched. Sharding must not lose that: a null-key build row is never inserted into
    /// any shard.
    #[test]
    fn nulls_never_match_across_a_sharded_build() {
        let build: Vec<Option<i64>> = (0..BIG)
            .map(|k| if k % 3 == 0 { None } else { Some(k) })
            .collect();
        let probe: Vec<Option<i64>> = vec![None, Some(1), Some(3), Some(4)];
        let idx = hash_join_indices(&keys(probe), &keys(build), JoinType::Left).expect("join");
        let got = pairs(&idx);
        assert_eq!(
            got,
            vec![
                (Some(0), None), // NULL probe: unmatched
                (Some(1), Some(1)),
                (Some(2), None), // build row 3 has a NULL key — never inserted
                (Some(3), Some(4)),
            ]
        );
    }
}
