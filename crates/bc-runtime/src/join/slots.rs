//! The hash join's chain-head table: open addressing over packed 64-bit slots, laid out so a
//! probe can prefetch where a key lives before it looks.
//!
//! The table it replaces was a `hashbrown::HashTable<u32>`, which is an excellent hash table
//! and the wrong one for this loop. A join probe is a long run of *independent* lookups into a
//! table that, past a few hundred thousand build rows, no longer fits in L2: every probe row
//! pays a control-group miss, a bucket miss and a build-key miss, one after another. The
//! lookups do not depend on each other, so the misses could all be in flight at once — but
//! hashbrown does not say where a key's group is, so nothing can be fetched ahead of it.
//! Profiling a 10M-row probe against a 1M-row build put 87% of the query in that lookup.
//!
//! Here the slot a key starts at is a pure function of its hash ([`SlotTable::start`]), so the
//! probe loop computes a block of hashes, prefetches each start slot, and only then walks them —
//! by which point the lines have arrived ([`super::JoinTable::probe_range`]).
//!
//! ## Layout
//!
//! One `u64` per slot: the hash's high 32 bits as a **tag** above `row + 1`, with `0` meaning
//! empty. The tag settles almost every non-matching slot without reading a build key, which
//! would be one more random access. The start slot is taken from the hash's *low* 32 bits by a
//! multiply-shift range reduction, so it shares no bits with the tag and needs no power-of-two
//! capacity; the shard a key is built in ([`super::build::shard_of`]) reads bits 32 and up,
//! which the tag still discriminates beyond.
//!
//! Capacity is twice the row count, so the load factor is at most one half and linear probing
//! stays short: a hit takes ~1.5 slots on average, a miss ~2.5. That is ~16 bytes per build row
//! against hashbrown's ~5 — the price of a table whose probes can overlap
//! ([`super::estimate_build_bytes`] counts it).
//!
//! ## What a slot means
//!
//! Exactly what a hashbrown entry meant: one slot per **distinct** key, holding the *latest*
//! build row inserted for it; the rows before it hang off `next`, prepended. [`SlotTable::upsert`]
//! returns the row it displaced so the caller links the chain exactly as before, so every key's
//! chain — and therefore every join's output order — is unchanged.

/// The empty slot. A real slot stores `row + 1`, which is never zero.
const EMPTY: u64 = 0;

/// The tag a hash leaves in its slot: its high half, independent of the start slot.
#[inline(always)]
fn tag_of(hash: u64) -> u32 {
    (hash >> 32) as u32
}

/// A probed slot's build row, or `None` when the slot is empty.
#[inline(always)]
fn row_of(slot: u64) -> Option<u32> {
    // `row + 1` sits in the low half; `wrapping_sub` turns an empty slot's `0` into `u32::MAX`,
    // which is filtered by the `EMPTY` test before it is ever used.
    (slot != EMPTY).then(|| (slot as u32).wrapping_sub(1))
}

/// Chain heads keyed by hash, one slot per distinct build key. See the module docs.
pub(super) struct SlotTable {
    slots: Vec<u64>,
}

impl SlotTable {
    /// A table for up to `rows` distinct keys, at a load factor of at most one half.
    pub(super) fn with_rows(rows: usize) -> Self {
        Self {
            slots: vec![EMPTY; rows.saturating_mul(2).max(16)],
        }
    }

    /// The slot a probe for `hash` begins at — what [`Self::prefetch`] fetches.
    ///
    /// A multiply-shift reduction of the hash's low 32 bits onto `[0, capacity)`: uniform for
    /// any capacity, so the table is sized to the build rather than to a power of two.
    #[inline(always)]
    fn start(&self, hash: u64) -> usize {
        (((hash & 0xFFFF_FFFF) * self.slots.len() as u64) >> 32) as usize
    }

    /// Hint the cache to fetch the slot a probe for `hash` begins at. A hint only: it never
    /// changes what [`Self::find`] returns, only how long it waits.
    #[inline(always)]
    pub(super) fn prefetch(&self, hash: u64) {
        bc_arrow::prefetch_read(self.slots.as_ptr().wrapping_add(self.start(hash)));
    }

    /// The head row of the key hashing to `hash` for which `eq(row)` holds, if one is present.
    ///
    /// `eq` is asked only of rows whose tag matches, so it compares the full key; a tag alone
    /// is never taken as a match.
    #[inline(always)]
    pub(super) fn find(&self, hash: u64, mut eq: impl FnMut(u32) -> bool) -> Option<u32> {
        let tag = tag_of(hash);
        let len = self.slots.len();
        let mut i = self.start(hash);
        loop {
            let slot = self.slots[i];
            let row = row_of(slot)?;
            if (slot >> 32) as u32 == tag && eq(row) {
                return Some(row);
            }
            i += 1;
            if i == len {
                i = 0;
            }
        }
    }

    /// Make `row` the head for its key: returns the head it displaced if the key was already
    /// present (`eq` matched a slot), or `None` when `row` opened a new slot.
    ///
    /// The caller prepends the displaced head to `row`'s chain, which is the order the
    /// hashbrown table this replaces produced — see the module docs.
    #[inline]
    pub(super) fn upsert(
        &mut self,
        hash: u64,
        row: u32,
        mut eq: impl FnMut(u32) -> bool,
    ) -> Option<u32> {
        debug_assert!(row < u32::MAX, "row + 1 must fit the slot");
        let tag = tag_of(hash);
        let packed = (u64::from(tag) << 32) | u64::from(row + 1);
        let len = self.slots.len();
        let mut i = self.start(hash);
        loop {
            let slot = self.slots[i];
            let Some(head) = row_of(slot) else {
                self.slots[i] = packed;
                return None;
            };
            if (slot >> 32) as u32 == tag && eq(head) {
                self.slots[i] = packed;
                return Some(head);
            }
            i += 1;
            if i == len {
                i = 0;
            }
        }
    }

    /// Heap bytes the slots hold.
    pub(super) fn heap_bytes(&self) -> usize {
        self.slots.capacity() * std::mem::size_of::<u64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every key inserted is found, at its latest row; every key not inserted is not — over a
    /// table driven to its full one-half load, with a hash that collides on purpose (the start
    /// slot and the tag both repeat) so linear probing and the tag test are both exercised.
    #[test]
    fn finds_every_present_key_at_its_latest_row_and_no_absent_one() {
        let rows = 5_000usize;
        let keys: Vec<u64> = (0..rows as u64).map(|i| i % 3_000).collect();
        // Weak on purpose: 64 distinct start slots and 16 distinct tags.
        let hash = |k: u64| ((k % 16) << 32) | (k % 64);
        let mut t = SlotTable::with_rows(rows);
        let mut next = vec![u32::MAX; rows];
        for (row, &k) in keys.iter().enumerate() {
            if let Some(prev) = t.upsert(hash(k), row as u32, |r| keys[r as usize] == k) {
                next[row] = prev;
            }
        }
        for k in 0..3_000u64 {
            let head = t.find(hash(k), |r| keys[r as usize] == k);
            // The latest row carrying k, then the chain descends through every earlier one.
            let mut want: Vec<u32> = (0..rows as u32)
                .filter(|&r| keys[r as usize] == k)
                .collect();
            want.reverse();
            let mut got = Vec::new();
            let mut r = head;
            while let Some(row) = r {
                got.push(row);
                r = Some(next[row as usize]).filter(|&n| n != u32::MAX);
            }
            assert_eq!(got, want, "key {k}");
        }
        for k in 3_000..4_000u64 {
            assert_eq!(
                t.find(hash(k), |r| keys[r as usize] == k),
                None,
                "absent key {k}"
            );
        }
    }

    /// Row 0 is a real row, not the empty marker, and an empty table finds nothing.
    #[test]
    fn row_zero_is_found_and_an_empty_table_is_empty() {
        let mut t = SlotTable::with_rows(1);
        assert_eq!(t.find(7, |_| true), None);
        assert_eq!(t.upsert(7, 0, |_| true), None);
        assert_eq!(t.find(7, |r| r == 0), Some(0));
        t.prefetch(7);
        assert_eq!(t.heap_bytes(), 16 * 8);
    }

    /// The start slot covers the whole table and never leaves it, whatever the hash.
    #[test]
    fn the_start_slot_is_always_in_range() {
        for rows in [1usize, 9, 100, 12_345] {
            let t = SlotTable::with_rows(rows);
            for h in [
                0u64,
                1,
                u64::MAX,
                0xFFFF_FFFF,
                0x1_0000_0000,
                0xDEAD_BEEF_CAFE_F00D,
            ] {
                assert!(t.start(h) < t.slots.len());
            }
            assert_eq!(t.start(0), 0);
            assert_eq!(t.start(0xFFFF_FFFF), t.slots.len() - 1);
        }
    }
}
