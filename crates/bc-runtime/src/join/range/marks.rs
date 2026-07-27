//! The mark bitmap the IEJoin sweep reads, and the levels that make reading it cheap.
//!
//! Split from the operator because it is a self-contained data structure with its own
//! invariant (each level agrees with the one beneath it) and its own tests.

/// A hierarchical bitmap over first-axis ranks: the marks, then a bit per word of the level
/// below, repeated until one word covers everything.
///
/// This scan is the inner loop of the IEJoin sweep and the whole reason the operator's cost
/// is not simply `O(n log n + k)`: every left row walks the axis-1 suffix from its own bound
/// to the end of the array, and most of that suffix holds no marks. Each level multiplies the
/// span that one word read can dismiss by 64, so walking an *empty* suffix costs `O(levels)`
/// rather than `O(n / 64)`.
///
/// The level count is derived from the size rather than fixed, and that is not tidiness. A
/// single summary level (span 4,096) measured **no different at all** at two million rows —
/// the term was not the bottleneck there — and dominated at five million, where the suffix
/// walk is `5,000,000 x 10,000,000 / 262,144` word reads. A fixed level count tunes for one
/// size; deriving it does not.
pub(super) struct MarkSet {
    /// `levels[0]` is the marks. For `L > 0`, bit `b` of `levels[L]` is set when word `b` of
    /// `levels[L - 1]` holds any set bit.
    levels: Vec<Vec<u64>>,
}

impl MarkSet {
    pub(super) fn new(bits: usize) -> Self {
        let mut levels = Vec::new();
        let mut words = bits.div_ceil(64).max(1);
        loop {
            levels.push(vec![0u64; words]);
            if words == 1 {
                break;
            }
            words = words.div_ceil(64);
        }
        Self { levels }
    }

    /// Mark `bit`, and every level's summary of it.
    ///
    /// The loop is why the levels are a `Vec`: a bit's position at level `L + 1` is its
    /// position at level `L` divided by 64, so the same two lines serve every depth.
    #[inline]
    pub(super) fn set(&mut self, bit: usize) {
        let mut b = bit;
        for level in &mut self.levels {
            level[b / 64] |= 1u64 << (b % 64);
            b /= 64;
        }
    }

    /// Lowest set bit at or after `from`, or `None`.
    ///
    /// Climbs while a level's current word is exhausted and descends once a level offers a
    /// candidate, so an empty span costs one word read per level rather than one per word.
    #[inline]
    pub(super) fn next_set(&self, from: usize) -> Option<usize> {
        let mut lvl = 0usize;
        let mut idx = from;
        loop {
            let bits = &self.levels[lvl];
            let w = idx / 64;
            if w < bits.len() {
                let masked = bits[w] & (u64::MAX << (idx % 64));
                if masked != 0 {
                    // A set bit at level `lvl` names a non-empty *word* of the level below,
                    // so descending is one trailing-zeros step per level.
                    let mut pos = w * 64 + masked.trailing_zeros() as usize;
                    while lvl > 0 {
                        lvl -= 1;
                        pos = pos * 64 + self.levels[lvl][pos].trailing_zeros() as usize;
                    }
                    return Some(pos);
                }
            }
            if lvl + 1 >= self.levels.len() {
                return None;
            }
            // Word `w` of this level is bit `w` of the next one, so resume above at `w + 1`.
            idx = w + 1;
            lvl += 1;
        }
    }
}
