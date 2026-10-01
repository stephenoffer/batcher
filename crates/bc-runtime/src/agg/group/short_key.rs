//! Short byte-string keys (≤ 7 bytes) as one `u64` each, and the small table that groups them.
//!
//! A status, a mode, a region code or an `id001`-style label is a handful of bytes, and as a
//! group key it is worth far more as a register than as a slice: comparing two packed keys is
//! one instruction where comparing slices is a length check and a `memcmp` call. This module
//! is that representation — [`pack_short_bytes`] — and the open-addressing table that groups
//! and ranks packed keys without `ahash` or `hashbrown`'s generic probe, both of which the
//! low-cardinality shapes paid per row.
//!
//! Everything here is injective on the values (distinct strings, distinct keys), so the group
//! ids and first-seen representatives match the byte-slice oracle in [`super::assign`] exactly.

use arrow::array::GenericByteArray;
use arrow::datatypes::ArrowNativeType;

/// Pack each null-free byte-string ≤ 7 bytes into a `u64` group key, or `None` if any value
/// exceeds 7 bytes (in which case the caller keeps the byte-slice hash path).
///
/// The key is `(len << 56) | little_endian(bytes)`: the length occupies the high byte and the
/// ≤ 7 payload bytes the low 56 bits, so two values collide iff they have the same length and
/// the same bytes — i.e. iff the strings are equal. That injectivity is what lets the integer
/// grouping produce the exact same groups as hashing the slices directly. One linear pass over
/// the offsets bails out the moment a value is too long, so a long-string column pays only a
/// cheap scan before falling back.
///
/// One unaligned 8-byte load per value, masked to its length, rather than a byte loop with a
/// data-dependent trip count: the loop measured as the single hottest frame of a grouped
/// `MEDIAN` by `l_shipmode` (~20 % of the query), ahead of the hash it feeds. Bytes past the
/// value are masked off, so the key is the same either way; a value within 8 bytes of the
/// buffer's end, where the load would overrun, takes the byte loop.
///
/// ## Seven bytes, and why widening it to fifteen does not pay
///
/// Seven is a low ceiling for a categorical key — an ISO code, a SKU, a `YYYY-MM-DD`, most real
/// identifiers are wider — and the byte-slice hash path beyond it is the engine's worst measured
/// group-by shape. On the H2O db-benchmark's group-by table at its 1e7-row tier, over 100,000
/// groups and the identical `sum(v1)`:
///
/// | key | bytes | Batcher | DuckDB |
/// |---|---|---:|---:|
/// | `id6` (`int32`) | — | 26.0 ms | 31.6 ms |
/// | `id3` (`'id0000039083'`) | 12 | 54.5 ms | 32.0 ms |
///
/// Same cardinality, same aggregate: the string key costs **2.1x** what the integer one does,
/// while DuckDB pays the same either way. So the obvious move is to pack 8-15 bytes into an
/// `i128` and route it through the integer grouper exactly as this does for `u64`.
///
/// **Measured, it is 1.13x *slower*** (`id3` 49.7/53.0 ms -> 57.6/59.2 ms
/// over two interleaved rounds, same tree, two `.so`s differing only in this). Two costs swamp
/// the saving, and both are properties of the wide key rather than of the implementation: the
/// packing is a second full pass that materializes a 160 MB `Vec<i128>` the hash path never
/// allocates, and the inline-key table becomes 20 bytes a slot against the byte path's 4, so it
/// loses far more to cache misses on a 100,000-group probe than it gains by comparing registers
/// instead of slices. The byte path is not naive: it already keeps each group's representative
/// *slice* beside its id, so its comparison costs no indirection either.
///
/// The gap is real and still open; a wider pack of the same shape is not the way to close it.
/// What the measurement points at is the representation, not the key width — a `StringView`
/// leaf with an inline prefix, so the comparison never leaves the array that was scanned
/// (`competitor_technique_review.md` item 2).
pub(super) fn pack_short_bytes<T>(a: &GenericByteArray<T>, num_rows: usize) -> Option<Vec<u64>>
where
    T: arrow::array::types::ByteArrayType,
{
    let offsets = a.value_offsets();
    let data = a.value_data();
    let mut out = Vec::with_capacity(num_rows);
    for w in offsets[..=num_rows].windows(2) {
        let (lo, hi) = (w[0].as_usize(), w[1].as_usize());
        let len = hi - lo;
        if len > 7 {
            return None;
        }
        let payload = match data.get(lo..lo + 8) {
            Some(word) => {
                let word = u64::from_le_bytes(word.try_into().expect("an 8-byte slice"));
                word & ((1u64 << (8 * len)) - 1)
            }
            None => data[lo..hi]
                .iter()
                .enumerate()
                .fold(0u64, |k, (j, &b)| k | u64::from(b) << (8 * j)),
        };
        out.push(((len as u64) << 56) | payload);
    }
    Some(out)
}

/// No packed key is this value: a key's top byte is its length, at most 7.
const EMPTY: u64 = u64::MAX;

/// An open-addressing map from packed short keys to dense ids, in first-seen order.
///
/// Linear probing over two flat arrays, kept at most half full, with a multiply-shift hash of
/// the key. A low-cardinality key — the shape short strings almost always are — sits in a few
/// cache lines, so a lookup is a multiply, a load and a compare. That is what it replaces:
/// `ahash` plus `hashbrown`'s closure-driven `entry` probe in the integer grouper, and an
/// `AHashMap<&[u8], _>` (hash a slice, `memcmp` it) in the ranker.
struct ShortKeyTable {
    keys: Vec<u64>,
    ids: Vec<u32>,
    len: usize,
    shift: u32,
}

impl ShortKeyTable {
    fn new() -> Self {
        Self::with_slots(64)
    }

    fn with_slots(slots: usize) -> Self {
        debug_assert!(slots.is_power_of_two());
        ShortKeyTable {
            keys: vec![EMPTY; slots],
            ids: vec![0; slots],
            len: 0,
            shift: 64 - slots.trailing_zeros(),
        }
    }

    #[inline(always)]
    fn slot(&self, key: u64) -> usize {
        // Fold the high half in first: two keys differing only in their low bytes must still
        // reach different top bits, which is where the shift reads the slot from.
        ((key ^ (key >> 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    /// The id of `key`, inserting it as `next` when absent. Returns `(id, inserted)`.
    #[inline(always)]
    fn id_or_insert(&mut self, key: u64, next: u32) -> (u32, bool) {
        let mask = self.keys.len() - 1;
        let mut i = self.slot(key);
        loop {
            let k = self.keys[i];
            if k == key {
                return (self.ids[i], false);
            }
            if k == EMPTY {
                self.keys[i] = key;
                self.ids[i] = next;
                self.len += 1;
                if self.len * 2 > self.keys.len() {
                    self.grow();
                }
                return (next, true);
            }
            i = (i + 1) & mask;
        }
    }

    fn grow(&mut self) {
        let mut bigger = ShortKeyTable::with_slots(self.keys.len() * 2);
        for (&k, &id) in self.keys.iter().zip(&self.ids) {
            if k != EMPTY {
                let mask = bigger.keys.len() - 1;
                let mut i = bigger.slot(k);
                while bigger.keys[i] != EMPTY {
                    i = (i + 1) & mask;
                }
                bigger.keys[i] = k;
                bigger.ids[i] = id;
            }
        }
        bigger.len = self.len;
        *self = bigger;
    }
}

/// Dense group ids for packed keys, and each group's first-seen row: the `(group_ids, reps)`
/// pair every integer grouper in [`super::assign`] returns.
pub(super) fn short_group_ids(packed: &[u64]) -> (Vec<u32>, Vec<u32>) {
    let mut table = ShortKeyTable::new();
    let mut reps: Vec<u32> = Vec::new();
    let mut group_ids = vec![0u32; packed.len()];
    for (i, (out, &key)) in group_ids.iter_mut().zip(packed).enumerate() {
        let (id, inserted) = table.id_or_insert(key, reps.len() as u32);
        if inserted {
            reps.push(i as u32);
        }
        *out = id;
    }
    (group_ids, reps)
}

/// Packed keys as dense `Int64` codes in first-seen order, or `None` once more than `cap`
/// distinct keys have been seen (ranking a high-cardinality column does not pay).
pub(super) fn rank_short(packed: &[u64], cap: usize) -> Option<Vec<i64>> {
    let mut table = ShortKeyTable::new();
    let mut codes = vec![0i64; packed.len()];
    for (out, &key) in codes.iter_mut().zip(packed) {
        let (id, inserted) = table.id_or_insert(key, table.len as u32);
        if inserted && table.len > cap {
            return None;
        }
        *out = i64::from(id);
    }
    Some(codes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First-seen order oracle over plain `u64`s.
    fn oracle(keys: &[u64]) -> (Vec<u32>, Vec<u32>) {
        let mut seen: Vec<u64> = Vec::new();
        let mut reps = Vec::new();
        let ids = keys
            .iter()
            .enumerate()
            .map(|(i, k)| match seen.iter().position(|s| s == k) {
                Some(g) => g as u32,
                None => {
                    seen.push(*k);
                    reps.push(i as u32);
                    (seen.len() - 1) as u32
                }
            })
            .collect();
        (ids, reps)
    }

    /// Ids and representatives match the first-seen oracle through several growths, on keys
    /// that differ only in a low byte, only in the length tag, and across the whole word.
    #[test]
    fn group_ids_match_first_seen_order_through_growth() {
        let mut keys: Vec<u64> = Vec::new();
        for i in 0..3_000u64 {
            keys.push((3 << 56) | (i % 997)); // low-byte differences
            keys.push(((i % 8) << 56) | 0x61); // length-tag differences
            keys.push((7 << 56) | (i.wrapping_mul(0x0001_0203_0405) & ((1 << 56) - 1)));
        }
        assert_eq!(short_group_ids(&keys), oracle(&keys));
        let (ids, _) = oracle(&keys);
        let codes = rank_short(&keys, usize::MAX).unwrap();
        assert_eq!(codes, ids.iter().map(|&g| i64::from(g)).collect::<Vec<_>>());
    }

    /// The ranker gives up exactly past its cap, and not at it.
    #[test]
    fn rank_declines_past_the_cap() {
        let keys: Vec<u64> = (0..100).collect();
        assert!(rank_short(&keys, 100).is_some());
        assert!(rank_short(&keys, 99).is_none());
    }
}
