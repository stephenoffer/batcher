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

/// A KLL quantile sketch over `f64` values.
#[derive(Clone)]
pub struct KllSketch {
    k: usize,
    compactors: Vec<Vec<f64>>, // level 0 = finest (weight 1)
    n: u64,
    min: f64,
    max: f64,
    rng: u64, // deterministic compaction coin
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
        Self {
            k,
            compactors: vec![Vec::new()],
            n: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            rng: 0x9E37_79B9_7F4A_7C15, // fixed seed → reproducible compaction
        }
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
        // is handled in the same call.
        if self.compactors[0].len() >= self.capacity(0) {
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

    /// Capacity of level `h`: top level holds `k`, each level down scales by `C`,
    /// floored at 2. Lower levels shrink as the sketch grows taller (KLL).
    fn capacity(&self, h: usize) -> usize {
        let depth_from_top = self.compactors.len() - 1 - h;
        (((self.k as f64) * C.powi(depth_from_top as i32)).ceil() as usize).max(2)
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
            if self.compactors[h].len() >= self.capacity(h) {
                self.compact_level(h);
            }
            h += 1;
        }
    }

    fn compact_level(&mut self, h: usize) {
        if h + 1 == self.compactors.len() {
            self.compactors.push(Vec::new());
        }
        let mut items = std::mem::take(&mut self.compactors[h]);
        items.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in sketch"));
        // Odd leftover stays at this level (keeps the count exactly halving).
        let leftover = (items.len() % 2 == 1).then(|| items.pop().unwrap());
        // Promote every other item (coin picks the phase) → weight doubles.
        let start = self.next_coin();
        let mut i = start;
        while i < items.len() {
            self.compactors[h + 1].push(items[i]);
            i += 2;
        }
        if let Some(v) = leftover {
            self.compactors[h].push(v);
        }
    }

    /// All retained items as `(value, weight)`, sorted by value.
    fn weighted_items(&self) -> Vec<(f64, u64)> {
        let mut items: Vec<(f64, u64)> = Vec::new();
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
        let mut out = Vec::new();
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
        let mut compactors = Vec::with_capacity(level_count);
        for _ in 0..level_count {
            let len = c.u64()? as usize;
            let mut level = Vec::with_capacity(len);
            for _ in 0..len {
                level.push(c.f64()?);
            }
            compactors.push(level);
        }
        if !c.is_done() {
            return None; // trailing garbage → reject
        }
        Some(Self {
            k,
            compactors,
            n,
            min,
            max,
            rng: 0x9E37_79B9_7F4A_7C15,
        })
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
}

impl Mergeable for KllSketch {
    /// Concatenate level-wise then re-compress. `k` must match.
    fn merge(&mut self, other: &KllSketch) {
        assert_eq!(self.k, other.k, "k mismatch");
        if other.n == 0 {
            return;
        }
        while self.compactors.len() < other.compactors.len() {
            self.compactors.push(Vec::new());
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
