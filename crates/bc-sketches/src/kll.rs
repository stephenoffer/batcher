//! KLL — streaming quantile / rank sketch (Karnin–Lang–Liberty).
//!
//! Answers "what value sits at quantile `q`?" and "what fraction of values are
//! ≤ `x`?" in `O(k)` space with rank error ~`O(1/k)`. The optimizer uses this two
//! ways: **selectivity** (a range predicate `x < c` keeps about `rank(c)` of the
//! rows) and **histograms** (quantile boundaries → equi-depth buckets). It also
//! backs approximate `median`/`quantile` aggregates at scale.
//!
//! Structure: a stack of *compactors* (sorted buffers); level `h` holds items of
//! weight `2^h`. When a level fills, it sorts and promotes every other item up a
//! level (halving the count, doubling the weight) — the classic KLL compaction.
//! Min and max are tracked **exactly** so the extremes (and `q=0`/`q=1`) are precise.
//! Compaction's coin flip uses a deterministic PRNG so independent builds and
//! merges are reproducible.

use crate::Mergeable;

const DEFAULT_K: usize = 200;
// Capacity decay between adjacent levels. 2/3 is the KLL paper's choice.
const C: f64 = 2.0 / 3.0;
// Smallest a compactor may get, however tall the sketch grows. KLL's capacities decay
// geometrically *downward* from the top (`k·C^depth`), so on a large stream the level that
// every value lands in — level 0 — decays toward nothing: at 1M values it reached this
// floor, and a floor of 2 means sorting and compacting on **every other value**. That
// single constant was most of what remained of KLL's per-value cost.
//
// 8 is the minimum compactor width Apache DataSketches' KLL uses (its `m`), and it is a
// strict improvement on both axes that matter: a wider buffer compacts less often, and
// *fewer compactions is less error* — a compaction is the only step that discards
// information. It costs a few hundred bytes on a tall sketch (six extra slots per level).
const MIN_COMPACTOR: usize = 8;

/// A KLL quantile sketch over `f64` values.
#[derive(Clone)]
pub struct KllSketch {
    k: usize,
    compactors: Vec<Vec<f64>>, // level 0 = finest (weight 1)
    n: u64,
    min: f64,
    max: f64,
    rng: u64, // deterministic compaction coin
    // `capacity(h)` for every level, cached — see `refresh_caps`.
    caps: Vec<usize>,
}

impl Default for KllSketch {
    fn default() -> Self {
        Self::new(DEFAULT_K)
    }
}

impl KllSketch {
    /// Create an empty sketch. Larger `k` → smaller rank error (~`1/k`) and more
    /// memory; `k=200` gives roughly ~1% error.
    pub fn new(k: usize) -> Self {
        assert!(k >= 8, "k must be >= 8");
        let mut s = Self {
            k,
            compactors: vec![Vec::new()],
            n: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            rng: 0x9E37_79B9_7F4A_7C15, // fixed seed → reproducible compaction
            caps: Vec::new(),
        };
        s.refresh_caps();
        s
    }

    /// Recompute the cached per-level capacities. Call after **any** change to the level
    /// count — a taller sketch shrinks every level below the top, so one push rewrites the
    /// whole table.
    ///
    /// This table is why the sketch is fast. `capacity(h)` costs a float `powi`, a `ceil`
    /// and a cast, and level 0's capacity decays to the floor of 2 once the sketch is tall
    /// (`k·C^12 ≈ 1.5` at 1M values) — so `add` overflowed level 0 on roughly *every other
    /// value*, and each overflow ran `compress`, which re-derived `capacity(h)` for all
    /// ~13 levels. That put six-plus transcendentals on the path of every value in the
    /// column: KLL measured **82.7 ns/value**, against 3.0 ns for the HLL beside it, and it
    /// made sketching a 1M-row join key cost ~90 ms — more than ten times the query it was
    /// supposed to be informing. The capacities depend only on `k` and the level count, so
    /// they are derived `O(log n)` times per sketch instead of `O(n)`.
    fn refresh_caps(&mut self) {
        let levels = self.compactors.len();
        self.caps = (0..levels)
            .map(|h| {
                let depth_from_top = levels - 1 - h;
                (((self.k as f64) * C.powi(depth_from_top as i32)).ceil() as usize)
                    .max(MIN_COMPACTOR)
            })
            .collect();
    }

    /// Number of values seen.
    pub fn count(&self) -> u64 {
        self.n
    }

    /// True if no (non-NaN) value has been added.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Exact minimum / maximum seen (`None` if empty).
    pub fn min(&self) -> Option<f64> {
        (self.n > 0).then_some(self.min)
    }
    pub fn max(&self) -> Option<f64> {
        (self.n > 0).then_some(self.max)
    }

    /// Add one value. NaN is ignored (it has no place in an ordered sketch).
    pub fn add(&mut self, x: f64) {
        if x.is_nan() {
            return;
        }
        self.n += 1;
        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
        self.compactors[0].push(x);
        // Only the bottom compactor grew, so a compaction cascade can be needed only
        // when *it* overflows — checking that one length is far cheaper than walking
        // every level on every value (the hot path over millions of rows). `compress`
        // still re-checks all levels, so a level-0 compaction that overflows level 1
        // is handled in the same call. The bound is read from the cached capacity table:
        // deriving it here meant a float `powi` per value (see `refresh_caps`).
        if self.compactors[0].len() >= self.caps[0] {
            self.compress();
        }
    }

    /// Add every non-null numeric value of an Arrow array (ints, floats, dates,
    /// timestamps are cast to `f64`). Non-numeric arrays are ignored.
    pub fn add_array(&mut self, array: &arrow::array::ArrayRef) {
        use arrow::array::Array;
        use arrow::compute::cast;
        use arrow::datatypes::DataType;

        if !matches!(
            array.data_type(),
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float16
                | DataType::Float32
                | DataType::Float64
                | DataType::Date32
                | DataType::Date64
                | DataType::Timestamp(_, _)
        ) {
            return;
        }
        let Ok(f) = cast(array, &DataType::Float64) else {
            return;
        };
        let f = f
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .expect("cast to Float64");
        for i in 0..f.len() {
            if f.is_valid(i) {
                self.add(f.value(i));
            }
        }
    }

    fn next_coin(&mut self) -> usize {
        // xorshift64 — fast, deterministic, good enough for an unbiased coin.
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        (x & 1) as usize
    }

    /// Restore every level to within its capacity, promoting overflow upward.
    fn compress(&mut self) {
        let mut h = 0;
        while h < self.compactors.len() {
            if self.compactors[h].len() >= self.caps[h] {
                self.compact_level(h);
            }
            h += 1;
        }
    }

    /// Sort level `h`, promote every other item to `h + 1`, and drop the rest.
    ///
    /// **Allocation-free**, which is the difference between a usable sketch and the one
    /// this replaced. A tall sketch's level 0 holds only `caps[0]` items — which decays to
    /// the floor of 2 — so this runs on roughly every *other* value in the column, and the
    /// obvious implementation allocated twice on each call: `std::mem::take` leaves a
    /// **zero-capacity** `Vec` behind (so pushing the odd leftover back had to reallocate,
    /// and the taken buffer was then freed), and `sort_by` is a *stable* sort, which
    /// allocates a merge buffer. Two `malloc`/`free` pairs every two values is what made KLL
    /// cost 60-80 ns/value against the HLL's 3.
    ///
    /// So: sort in place with `sort_unstable_by` (pattern-defeating quicksort — no
    /// allocation, and for `f64` "stability" is meaningless because equal values are
    /// indistinguishable), promote through a split borrow, and `clear()` the level, which
    /// *keeps* its capacity. The retained items, the coin sequence, and therefore the
    /// sketch itself are bit-for-bit what the old code produced.
    fn compact_level(&mut self, h: usize) {
        if h + 1 == self.compactors.len() {
            self.compactors.push(Vec::new());
            self.refresh_caps(); // a taller sketch shrinks every level below the top
        }
        self.compactors[h].sort_unstable_by(|a, b| a.partial_cmp(b).expect("no NaN in sketch"));
        let len = self.compactors[h].len();
        // An odd leftover (the largest, as the old `pop()` took it after sorting) stays at
        // this level, so the promoted count halves exactly.
        let odd = len % 2 == 1;
        let promotable = len - usize::from(odd);
        // Promote every other item (coin picks the phase) → weight doubles.
        let start = self.next_coin();
        let (below, above) = self.compactors.split_at_mut(h + 1);
        let src = &mut below[h];
        let dst = &mut above[0];
        let mut i = start;
        while i < promotable {
            dst.push(src[i]);
            i += 2;
        }
        let leftover = odd.then(|| src[promotable]);
        src.clear(); // keeps the buffer's capacity — the whole point
        if let Some(v) = leftover {
            src.push(v);
        }
    }

    /// All retained items as `(value, weight)`, sorted by value.
    fn weighted_items(&self) -> Vec<(f64, u64)> {
        let n_retained: usize = self.compactors.iter().map(|c| c.len()).sum();
        let mut items: Vec<(f64, u64)> = Vec::with_capacity(n_retained);
        for (h, comp) in self.compactors.iter().enumerate() {
            let w = 1u64 << h;
            items.extend(comp.iter().map(|&v| (v, w)));
        }
        items.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("no NaN in sketch"));
        items
    }

    /// Approximate fraction of values ≤ `x`, in `[0, 1]` — i.e. the selectivity of
    /// `col <= x`. Returns 0 for an empty sketch.
    pub fn rank(&self, x: f64) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        let items = self.weighted_items();
        let total: u64 = items.iter().map(|(_, w)| w).sum();
        let below: u64 = items.iter().filter(|(v, _)| *v <= x).map(|(_, w)| w).sum();
        below as f64 / total as f64
    }

    /// Approximate value at quantile `q ∈ [0, 1]` (`None` if empty). `q=0`/`q=1`
    /// return the exact min/max.
    pub fn quantile(&self, q: f64) -> Option<f64> {
        if self.n == 0 {
            return None;
        }
        let items = self.weighted_items();
        let total: u64 = items.iter().map(|(_, w)| w).sum();
        self.quantile_from(&items, total, q)
    }

    /// Batch quantile lookup: answers every `q` from a **single** sorted pass over
    /// the retained items, instead of rebuilding and re-sorting per call. Order is
    /// preserved (output `i` is the quantile for `qs[i]`). This is what an
    /// equi-depth histogram (many quantiles off one sketch) should use.
    pub fn quantiles(&self, qs: &[f64]) -> Vec<Option<f64>> {
        if self.n == 0 {
            return vec![None; qs.len()];
        }
        let items = self.weighted_items();
        let total: u64 = items.iter().map(|(_, w)| w).sum();
        qs.iter()
            .map(|&q| self.quantile_from(&items, total, q))
            .collect()
    }

    /// Resolve one quantile against pre-sorted `(value, weight)` items.
    fn quantile_from(&self, items: &[(f64, u64)], total: u64, q: f64) -> Option<f64> {
        let q = q.clamp(0.0, 1.0);
        if q <= 0.0 {
            return Some(self.min);
        }
        if q >= 1.0 {
            return Some(self.max);
        }
        let target = (q * total as f64).ceil() as u64;
        let mut cum = 0u64;
        for &(v, w) in items {
            cum += w;
            if cum >= target {
                return Some(v);
            }
        }
        Some(self.max)
    }

    /// Convenience: the median.
    pub fn median(&self) -> Option<f64> {
        self.quantile(0.5)
    }

    /// Serialize to a byte blob. Layout (all little-endian):
    /// `[k: u64][n: u64][min: f64][max: f64][level_count: u64]` then per level
    /// `[len: u64][values: len × f64]`. The rng is *not* stored: it only seeds a
    /// reproducible coin and has no effect on the retained quantile estimates.
    pub fn to_bytes(&self) -> Vec<u8> {
        // Exact wire size: 5×u64 header + per level (u64 len + len×f64) — pre-size so the
        // shuffle/persist serialization fills one allocation instead of doubling as it grows.
        let n_values: usize = self.compactors.iter().map(|c| c.len()).sum();
        let mut out = Vec::with_capacity(40 + self.compactors.len() * 8 + n_values * 8);
        out.extend_from_slice(&(self.k as u64).to_le_bytes());
        out.extend_from_slice(&self.n.to_le_bytes());
        out.extend_from_slice(&self.min.to_le_bytes());
        out.extend_from_slice(&self.max.to_le_bytes());
        out.extend_from_slice(&(self.compactors.len() as u64).to_le_bytes());
        for level in &self.compactors {
            out.extend_from_slice(&(level.len() as u64).to_le_bytes());
            for &v in level {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    /// Reconstruct from [`to_bytes`](Self::to_bytes). Returns `None` on truncated
    /// or otherwise malformed input. The rng is reset to its default seed.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut c = Cursor::new(bytes);
        let k = c.u64()? as usize;
        if k < 8 {
            return None;
        }
        let n = c.u64()?;
        let min = c.f64()?;
        let max = c.f64()?;
        let level_count = c.u64()? as usize;
        // A KLL always has at least one (level-0) compactor.
        if level_count == 0 {
            return None;
        }
        // Never trust the length fields for the reservation: each level costs at
        // least an 8-byte length prefix, and each value 8 bytes, so cap both by the
        // bytes that actually remain. A corrupt/foreign blob claiming a huge count
        // then hits the short-read `None` path instead of `capacity overflow`.
        let mut compactors = Vec::with_capacity(level_count.min(c.remaining() / 8));
        for _ in 0..level_count {
            let len = c.u64()? as usize;
            let mut level = Vec::with_capacity(len.min(c.remaining() / 8));
            for _ in 0..len {
                let v = c.f64()?;
                // `add` filters NaN, so a well-formed sketch never stores one; the
                // compaction/query sorts rely on that (`partial_cmp` on NaN panics).
                // Reject a NaN from a corrupt/foreign blob here rather than letting a
                // later `quantile`/`merge` panic on the shuffle/spill path. (±inf is a
                // value `add` legitimately accepts and sorts fine, so it is allowed.)
                if v.is_nan() {
                    return None;
                }
                level.push(v);
            }
            compactors.push(level);
        }
        if !c.is_done() {
            return None; // trailing garbage → reject
        }
        let mut s = Self {
            k,
            compactors,
            n,
            min,
            max,
            rng: 0x9E37_79B9_7F4A_7C15,
            caps: Vec::new(),
        };
        s.refresh_caps();
        Some(s)
    }
}

/// Minimal little-endian read cursor that returns `None` instead of panicking on
/// short reads.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let end = self.pos.checked_add(N)?;
        let slice = self.bytes.get(self.pos..end)?;
        self.pos = end;
        slice.try_into().ok()
    }

    fn u64(&mut self) -> Option<u64> {
        self.take::<8>().map(u64::from_le_bytes)
    }

    fn f64(&mut self) -> Option<f64> {
        self.take::<8>().map(f64::from_le_bytes)
    }

    fn is_done(&self) -> bool {
        self.pos == self.bytes.len()
    }

    /// Bytes not yet consumed. Used to bound `Vec::with_capacity` against an
    /// untrusted length field so a crafted blob cannot request a giant allocation.
    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }
}

impl Mergeable for KllSketch {
    /// Concatenate level-wise then re-compress. `k` must match.
    fn merge(&mut self, other: &KllSketch) {
        assert_eq!(self.k, other.k, "k mismatch");
        if other.n == 0 {
            return;
        }
        if self.compactors.len() < other.compactors.len() {
            self.compactors
                .resize_with(other.compactors.len(), Vec::new);
            self.refresh_caps(); // a taller sketch shrinks every level below the top
        }
        for (h, comp) in other.compactors.iter().enumerate() {
            self.compactors[h].extend_from_slice(comp);
        }
        self.n += other.n;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.compress();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merge_all;

    // Deterministic shuffle so insertion order isn't sorted (exercises compaction).
    fn shuffled(n: u64) -> Vec<f64> {
        let mut v: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut state = 0x1234_5678u64;
        for i in (1..v.len()).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let j = (state as usize) % (i + 1);
            v.swap(i, j);
        }
        v
    }

    #[test]
    fn quantiles_within_rank_error() {
        let n = 100_000u64;
        let mut s = KllSketch::new(200);
        for x in shuffled(n) {
            s.add(x);
        }
        assert_eq!(s.count(), n);
        // Values are 0..n, so the true value at quantile q is ≈ q·n. Allow 2%
        // rank error (k=200 → ε≈1%, plus slack).
        for &q in &[0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let est = s.quantile(q).unwrap();
            let err = (est - q * n as f64).abs() / n as f64;
            assert!(err < 0.02, "q={q} est={est} rank-error={err}");
        }
    }

    #[test]
    fn min_max_exact() {
        let mut s = KllSketch::new(200);
        for x in shuffled(50_000) {
            s.add(x);
        }
        assert_eq!(s.min(), Some(0.0));
        assert_eq!(s.max(), Some(49_999.0));
        assert_eq!(s.quantile(0.0), Some(0.0));
        assert_eq!(s.quantile(1.0), Some(49_999.0));
    }

    #[test]
    fn rank_is_selectivity() {
        let mut s = KllSketch::new(200);
        for x in shuffled(100_000) {
            s.add(x);
        }
        // ~30% of values are < 30_000.
        let sel = s.rank(30_000.0);
        assert!((sel - 0.3).abs() < 0.02, "selectivity {sel}");
    }

    #[test]
    fn merge_matches_single() {
        let n = 100_000u64;
        let vals = shuffled(n);
        let mut whole = KllSketch::new(200);
        for &x in &vals {
            whole.add(x);
        }
        let mut a = KllSketch::new(200);
        let mut b = KllSketch::new(200);
        for (i, &x) in vals.iter().enumerate() {
            if i % 2 == 0 {
                a.add(x);
            } else {
                b.add(x);
            }
        }
        a.merge(&b);
        assert_eq!(a.count(), n);
        for &q in &[0.25, 0.5, 0.75] {
            let merged = a.quantile(q).unwrap();
            let single = whole.quantile(q).unwrap();
            let err = (merged - single).abs() / n as f64;
            assert!(
                err < 0.03,
                "q={q} merged={merged} single={single} err={err}"
            );
        }
    }

    #[test]
    fn bytes_roundtrip_preserves_quantiles() {
        let mut s = KllSketch::new(200);
        for x in shuffled(100_000) {
            s.add(x);
        }
        let bytes = s.to_bytes();
        let back = KllSketch::from_bytes(&bytes).expect("valid blob");
        assert_eq!(back.count(), s.count());
        assert_eq!(back.min(), s.min());
        assert_eq!(back.max(), s.max());
        // Retained items are identical, so every quantile/rank matches exactly.
        for &q in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            assert_eq!(back.quantile(q), s.quantile(q));
        }
        assert_eq!(back.quantile(0.5), s.quantile(0.5));
        assert_eq!(back.rank(30_000.0), s.rank(30_000.0));
    }

    #[test]
    fn empty_sketch_roundtrips() {
        let s = KllSketch::new(200);
        let back = KllSketch::from_bytes(&s.to_bytes()).expect("valid blob");
        assert!(back.is_empty());
        assert_eq!(back.quantile(0.5), None);
    }

    #[test]
    fn from_bytes_rejects_malformed() {
        assert!(KllSketch::from_bytes(&[]).is_none());
        assert!(KllSketch::from_bytes(&[0; 7]).is_none()); // truncated header
                                                           // Valid blob with one extra trailing byte → rejected.
        let mut s = KllSketch::new(200);
        s.add(1.0);
        let mut bytes = s.to_bytes();
        bytes.push(0);
        assert!(KllSketch::from_bytes(&bytes).is_none());
        // k < 8 is invalid.
        let mut bad = 4u64.to_le_bytes().to_vec();
        bad.extend_from_slice(&0u64.to_le_bytes()); // n
        bad.extend_from_slice(&f64::INFINITY.to_le_bytes()); // min
        bad.extend_from_slice(&f64::NEG_INFINITY.to_le_bytes()); // max
        bad.extend_from_slice(&1u64.to_le_bytes()); // level_count
        bad.extend_from_slice(&0u64.to_le_bytes()); // level 0 len
        assert!(KllSketch::from_bytes(&bad).is_none());
    }

    #[test]
    fn empty_sketch() {
        let s = KllSketch::new(200);
        assert!(s.is_empty());
        assert_eq!(s.quantile(0.5), None);
        assert_eq!(s.rank(0.0), 0.0);
    }

    #[test]
    fn from_bytes_rejects_nan_level_value() {
        // A corrupt/foreign blob carrying a NaN must be rejected, not deserialized
        // into a sketch that panics ("no NaN in sketch") on the next quantile/merge.
        let mut b = Vec::new();
        b.extend_from_slice(&8u64.to_le_bytes()); // k
        b.extend_from_slice(&2u64.to_le_bytes()); // n
        b.extend_from_slice(&1.0f64.to_le_bytes()); // min
        b.extend_from_slice(&2.0f64.to_le_bytes()); // max
        b.extend_from_slice(&1u64.to_le_bytes()); // level_count
        b.extend_from_slice(&2u64.to_le_bytes()); // level 0 len
        b.extend_from_slice(&f64::NAN.to_le_bytes());
        b.extend_from_slice(&1.5f64.to_le_bytes());
        assert!(KllSketch::from_bytes(&b).is_none());
    }

    #[test]
    fn from_bytes_rejects_absurd_level_count_without_panic() {
        // A crafted/corrupt blob with a valid header but an enormous `level_count`
        // must be rejected with `None`, not abort the process by pre-allocating a
        // `Vec` of that many levels (`Vec::with_capacity(huge)` → capacity overflow).
        let mut b = Vec::new();
        b.extend_from_slice(&8u64.to_le_bytes()); // k
        b.extend_from_slice(&0u64.to_le_bytes()); // n
        b.extend_from_slice(&f64::INFINITY.to_le_bytes()); // min
        b.extend_from_slice(&f64::NEG_INFINITY.to_le_bytes()); // max
        b.extend_from_slice(&u64::MAX.to_le_bytes()); // level_count = absurd
        assert!(KllSketch::from_bytes(&b).is_none());
    }

    #[test]
    fn from_bytes_rejects_absurd_level_len_without_panic() {
        // Same hazard one layer down: a valid level_count but an enormous per-level
        // `len` must be rejected, not pre-allocate `len` f64s and blow the allocator.
        let mut b = Vec::new();
        b.extend_from_slice(&8u64.to_le_bytes()); // k
        b.extend_from_slice(&2u64.to_le_bytes()); // n
        b.extend_from_slice(&1.0f64.to_le_bytes()); // min
        b.extend_from_slice(&2.0f64.to_le_bytes()); // max
        b.extend_from_slice(&1u64.to_le_bytes()); // level_count = 1
        b.extend_from_slice(&u64::MAX.to_le_bytes()); // level 0 len = absurd
        assert!(KllSketch::from_bytes(&b).is_none());
    }

    #[test]
    fn from_bytes_preserves_infinities() {
        // ±inf is a value `add` accepts and sorts fine, so a roundtrip must keep it.
        let mut s = KllSketch::new(8);
        s.add(f64::INFINITY);
        s.add(1.0);
        s.add(f64::NEG_INFINITY);
        let back = KllSketch::from_bytes(&s.to_bytes()).expect("inf roundtrips");
        assert_eq!(back.count(), 3);
        assert_eq!(back.min(), Some(f64::NEG_INFINITY));
        assert_eq!(back.max(), Some(f64::INFINITY));
    }

    // ---- Property / fuzz tests (deterministic xorshift64, fixed seed) -------

    /// Minimal deterministic PRNG so trials are reproducible without `rand`.
    struct XorShift64(u64);
    impl XorShift64 {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, bound: u64) -> u64 {
            self.next_u64() % bound
        }
        /// A finite f64 in `[-1e6, 1e6)` (never NaN/inf — KLL ignores NaN).
        fn finite(&mut self) -> f64 {
            // 53-bit mantissa fraction in [0, 1), then map to the range.
            let frac = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            frac * 2_000_000.0 - 1_000_000.0
        }
    }

    #[test]
    fn prop_serialize_roundtrip() {
        const TRIALS: usize = 200;
        let mut rng = XorShift64::new(0x11C1_7A55_9000_0001);

        for trial in 0..TRIALS {
            let k = 8 + rng.below(400) as usize; // k in 8..=407
            let n = rng.below(4_000); // up to ~4k values

            let mut s = KllSketch::new(k);
            let mut sample_xs: Vec<f64> = Vec::new();
            for _ in 0..n {
                let x = rng.finite();
                if sample_xs.len() < 8 {
                    sample_xs.push(x);
                }
                s.add(x);
            }

            let back = KllSketch::from_bytes(&s.to_bytes()).expect("valid blob");

            // Retained items are byte-identical → every quantile/rank is exactly equal.
            for &q in &[0.0, 0.25, 0.5, 0.75, 1.0] {
                assert_eq!(
                    back.quantile(q),
                    s.quantile(q),
                    "trial {trial}: k={k} n={n} q={q} quantile mismatch after roundtrip"
                );
            }
            for &x in &sample_xs {
                assert_eq!(
                    back.rank(x),
                    s.rank(x),
                    "trial {trial}: k={k} n={n} x={x} rank mismatch after roundtrip"
                );
            }
            assert_eq!(back.count(), s.count());
            assert_eq!(back.min(), s.min());
            assert_eq!(back.max(), s.max());
        }
    }

    #[test]
    fn prop_merge_matches_single() {
        const TRIALS: usize = 150;
        let mut rng = XorShift64::new(0x7E57_C0DE_2222_3333);

        for trial in 0..TRIALS {
            let k = 100 + rng.below(200) as usize; // k in 100..=299 → ε ≈ 1/k
            let n = 2_000 + rng.below(2_000); // 2k..4k values

            let vals: Vec<f64> = (0..n).map(|_| rng.finite()).collect();

            let mut whole = KllSketch::new(k);
            for &x in &vals {
                whole.add(x);
            }

            // Split the stream across a random number of partitions and merge.
            let parts = 2 + rng.below(4) as usize; // 2..=5 partitions
            let mut sketches: Vec<KllSketch> = (0..parts).map(|_| KllSketch::new(k)).collect();
            for &x in &vals {
                let p = rng.below(parts as u64) as usize;
                sketches[p].add(x);
            }
            let merged = merge_all(sketches.into_iter()).unwrap();
            assert_eq!(merged.count(), n);

            // Compare estimated quantiles by *rank error*: the merged quantile value
            // must sit within KLL's rank tolerance of the single-pass value. We
            // measure error as the difference in true rank (fraction ≤ value) using
            // the single-pass sketch as the reference distribution.
            let eps = 1.0 / k as f64;
            for &q in &[0.1, 0.25, 0.5, 0.75, 0.9] {
                let merged_v = merged.quantile(q).unwrap();
                let single_v = whole.quantile(q).unwrap();
                // Rank of each estimate within the full reference distribution.
                let r_merged = whole.rank(merged_v);
                let r_single = whole.rank(single_v);
                let rank_err = (r_merged - r_single).abs();
                // Two compactions stack (merge then query) → allow ~3ε plus a small
                // absolute floor for discreteness on the random domain.
                assert!(
                    rank_err < 3.0 * eps + 0.02,
                    "trial {trial}: k={k} n={n} q={q} merged={merged_v} single={single_v} rank_err={rank_err} (eps={eps})"
                );
            }
        }
    }
}
