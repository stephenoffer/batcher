//! T-Digest — tail-accurate quantile sketch (Dunning).
//!
//! A centroid summary that is *adaptively* accurate: buckets near the tails
//! (`q→0`, `q→1`) are kept small so extreme quantiles (p99, p999) are precise,
//! while the bulk (around the median) is summarized coarsely. The optimizer uses
//! this where the *tails* drive cost — latency SLAs, skew/outlier detection, the
//! heaviest keys in a join — and where KLL's uniform rank error would smear the
//! very quantiles that matter most.
//!
//! Structure: a sorted `Vec` of centroids `(mean, weight)`. Incoming values are
//! buffered and, when the buffer fills, merged in sorted order. A centroid may
//! absorb weight only up to the scale-function limit `4·n·δ·q·(1−q)` (with `q`
//! its cumulative position) — which shrinks toward the tails, giving them more,
//! finer centroids. Min/max are tracked exactly so `q=0`/`q=1` are precise.

use crate::Mergeable;

/// Default compression parameter `δ`. Larger → more centroids, tighter quantiles.
const DEFAULT_COMPRESSION: f64 = 100.0;

#[derive(Clone, Copy)]
struct Centroid {
    mean: f64,
    weight: f64,
}

/// A t-digest over `f64` values with adaptive (tail-accurate) quantile error.
#[derive(Clone)]
pub struct TDigest {
    compression: f64,
    /// Always kept sorted by `mean` after a `flush`.
    centroids: Vec<Centroid>,
    /// Unmerged values awaiting the next `flush`.
    buffer: Vec<f64>,
    n: f64,
    min: f64,
    max: f64,
}

impl Default for TDigest {
    fn default() -> Self {
        Self::new(DEFAULT_COMPRESSION)
    }
}

impl TDigest {
    /// Create an empty digest with the given `compression` (`δ`, e.g. 100). Must
    /// be ≥ 1; larger values keep more centroids and yield tighter quantiles.
    pub fn new(compression: f64) -> Self {
        assert!(
            compression >= 1.0,
            "compression must be >= 1, got {compression}"
        );
        Self {
            compression,
            centroids: Vec::new(),
            buffer: Vec::new(),
            n: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Number of values seen.
    pub fn count(&self) -> u64 {
        self.n as u64
    }

    /// True if no (non-NaN) value has been added.
    pub fn is_empty(&self) -> bool {
        self.n == 0.0
    }

    /// Exact minimum / maximum seen (`None` if empty).
    pub fn min(&self) -> Option<f64> {
        (self.n > 0.0).then_some(self.min)
    }
    pub fn max(&self) -> Option<f64> {
        (self.n > 0.0).then_some(self.max)
    }

    /// Add one value. NaN/±inf are ignored (no place in an ordered sketch).
    pub fn add(&mut self, v: f64) {
        if !v.is_finite() {
            return;
        }
        if v < self.min {
            self.min = v;
        }
        if v > self.max {
            self.max = v;
        }
        self.n += 1.0;
        self.buffer.push(v);
        // Flush when the buffer grows large relative to the centroid budget.
        if self.buffer.len() as f64 >= self.compression * 10.0 {
            self.flush();
        }
    }

    /// Inverse scale function: the maximum weight a centroid at cumulative
    /// quantile `q` may hold. `4·n·δ·q·(1−q)` shrinks toward the tails so they
    /// stay finely resolved. `total` is the full weight after the pending merge.
    #[inline]
    fn weight_limit(compression: f64, total: f64, q: f64) -> f64 {
        4.0 * total * q * (1.0 - q) / compression
    }

    /// Fold buffered values into the sorted centroid list, then re-compress so
    /// every centroid respects its scale-function weight limit.
    fn flush(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        // Seed the merge list with existing centroids plus each buffered value as
        // a unit-weight centroid, all sorted by mean.
        let mut incoming: Vec<Centroid> =
            Vec::with_capacity(self.centroids.len() + self.buffer.len());
        incoming.append(&mut self.centroids);
        for &v in &self.buffer {
            incoming.push(Centroid {
                mean: v,
                weight: 1.0,
            });
        }
        self.buffer.clear();
        incoming.sort_by(|a, b| a.mean.partial_cmp(&b.mean).expect("no NaN in digest"));

        self.centroids = Self::merge_sorted(self.compression, incoming, self.n);
    }

    /// Merge a sorted list of centroids into a compressed list whose centroids
    /// each respect the scale-function weight limit. `total` is the sum of all
    /// weights (== n).
    fn merge_sorted(compression: f64, sorted: Vec<Centroid>, total: f64) -> Vec<Centroid> {
        let mut out: Vec<Centroid> = Vec::with_capacity(sorted.len());
        let mut cum = 0.0; // weight strictly before the current accumulator
        let mut iter = sorted.into_iter();
        let Some(mut acc) = iter.next() else {
            return out;
        };
        for c in iter {
            // Quantile at the centre of the *combined* mass if we absorbed c.
            let proposed = acc.weight + c.weight;
            let q = (cum + proposed / 2.0) / total;
            if proposed <= Self::weight_limit(compression, total, q) {
                // Absorb: weighted-mean update.
                acc.mean = (acc.mean * acc.weight + c.mean * c.weight) / proposed;
                acc.weight = proposed;
            } else {
                cum += acc.weight;
                out.push(acc);
                acc = c;
            }
        }
        out.push(acc);
        out
    }

    /// Approximate value at quantile `q ∈ [0, 1]` (`None` if empty). `q=0`/`q=1`
    /// return the exact min/max; interior quantiles interpolate across centroid
    /// cumulative weights.
    pub fn quantile(&mut self, q: f64) -> Option<f64> {
        self.flush();
        self.quantile_ref(q)
    }

    /// Quantile without flushing — usable on a `&self`. Call after a `flush` (or
    /// via [`quantile`](Self::quantile)) for the buffered values to be included.
    fn quantile_ref(&self, q: f64) -> Option<f64> {
        if self.n == 0.0 {
            return None;
        }
        let q = q.clamp(0.0, 1.0);
        if q <= 0.0 {
            return Some(self.min);
        }
        if q >= 1.0 {
            return Some(self.max);
        }
        if self.centroids.len() == 1 {
            return Some(self.centroids[0].mean);
        }
        let target = q * self.n; // target cumulative rank
        let cs = &self.centroids;

        // Cumulative weight at the *centre* of each centroid; interpolate between
        // adjacent centres. Below the first centre → interpolate from min; above
        // the last centre → interpolate to max.
        let mut cum = 0.0;
        for (i, c) in cs.iter().enumerate() {
            let centre = cum + c.weight / 2.0;
            if target <= centre {
                return Some(if i == 0 {
                    // Between min and the first centroid centre.
                    let lo_rank = c.weight / 2.0;
                    if lo_rank <= 0.0 {
                        c.mean
                    } else {
                        let t = (target / lo_rank).clamp(0.0, 1.0);
                        self.min + t * (c.mean - self.min)
                    }
                } else {
                    let prev = &cs[i - 1];
                    let prev_centre = cum - prev.weight / 2.0;
                    let span = centre - prev_centre;
                    let t = if span > 0.0 {
                        (target - prev_centre) / span
                    } else {
                        0.0
                    };
                    prev.mean + t * (c.mean - prev.mean)
                });
            }
            cum += c.weight;
        }
        // Past the last centre → interpolate to max.
        let last = cs[cs.len() - 1];
        let last_centre = self.n - last.weight / 2.0;
        let span = self.n - last_centre;
        let t = if span > 0.0 {
            ((target - last_centre) / span).clamp(0.0, 1.0)
        } else {
            1.0
        };
        Some(last.mean + t * (self.max - last.mean))
    }

    /// Convenience: the median.
    pub fn median(&mut self) -> Option<f64> {
        self.quantile(0.5)
    }

    /// Approximate fraction of values ≤ `x`, in `[0, 1]` (the selectivity of
    /// `col <= x`). Flushes buffered values first. Returns 0 for empty.
    pub fn rank(&mut self, x: f64) -> f64 {
        self.flush();
        if self.n == 0.0 {
            return 0.0;
        }
        if x < self.min {
            return 0.0;
        }
        if x >= self.max {
            return 1.0;
        }
        let cs = &self.centroids;
        let mut cum = 0.0;
        for (i, c) in cs.iter().enumerate() {
            let centre = cum + c.weight / 2.0;
            if x < c.mean {
                // Interpolate between previous centre and this one.
                return if i == 0 {
                    let lo_rank = c.weight / 2.0;
                    let span = c.mean - self.min;
                    let t = if span > 0.0 {
                        (x - self.min) / span
                    } else {
                        0.0
                    };
                    (t * lo_rank) / self.n
                } else {
                    let prev = &cs[i - 1];
                    let prev_centre = cum - prev.weight / 2.0;
                    let span = c.mean - prev.mean;
                    let t = if span > 0.0 {
                        (x - prev.mean) / span
                    } else {
                        0.0
                    };
                    (prev_centre + t * (centre - prev_centre)) / self.n
                };
            }
            cum += c.weight;
        }
        1.0
    }

    /// Serialize to a byte blob. Layout (all little-endian). Buffered values are
    /// flushed into centroids first so the blob fully captures the state.
    /// `[compression: f64][n: f64][min: f64][max: f64][len: u64]`
    /// then `len × ([mean: f64][weight: f64])`.
    pub fn to_bytes(&self) -> Vec<u8> {
        // Flush a clone so `&self` stays immutable and the on-wire form is canonical.
        let mut canon = self.clone();
        canon.flush();
        let mut out = Vec::new();
        out.extend_from_slice(&canon.compression.to_le_bytes());
        out.extend_from_slice(&canon.n.to_le_bytes());
        out.extend_from_slice(&canon.min.to_le_bytes());
        out.extend_from_slice(&canon.max.to_le_bytes());
        out.extend_from_slice(&(canon.centroids.len() as u64).to_le_bytes());
        for c in &canon.centroids {
            out.extend_from_slice(&c.mean.to_le_bytes());
            out.extend_from_slice(&c.weight.to_le_bytes());
        }
        out
    }

    /// Reconstruct from [`to_bytes`](Self::to_bytes). Returns `None` on truncated
    /// or otherwise malformed input.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut c = Cursor::new(bytes);
        let compression = c.f64()?;
        if !compression.is_finite() || compression < 1.0 {
            return None;
        }
        let n = c.f64()?;
        if !n.is_finite() || n < 0.0 {
            return None;
        }
        let min = c.f64()?;
        let max = c.f64()?;
        let len = c.u64()? as usize;
        let mut centroids = Vec::with_capacity(len);
        let mut wsum = 0.0;
        for _ in 0..len {
            let mean = c.f64()?;
            let weight = c.f64()?;
            if !weight.is_finite() || weight <= 0.0 || !mean.is_finite() {
                return None;
            }
            wsum += weight;
            centroids.push(Centroid { mean, weight });
        }
        if !c.is_done() {
            return None; // trailing garbage → reject
        }
        // Centroid means must be non-decreasing, and weights must sum to n.
        for w in centroids.windows(2) {
            if w[0].mean > w[1].mean {
                return None;
            }
        }
        if (wsum - n).abs() > 1e-6 * n.max(1.0) {
            return None;
        }
        Some(Self {
            compression,
            centroids,
            buffer: Vec::new(),
            n,
            min,
            max,
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

impl Mergeable for TDigest {
    /// Concatenate centroids (and buffers) then re-compress. `compression` must
    /// match — the scale function is parameterized by it.
    fn merge(&mut self, other: &TDigest) {
        assert_eq!(
            self.compression, other.compression,
            "compression mismatch: {} vs {}",
            self.compression, other.compression
        );
        if other.n == 0.0 {
            return;
        }
        // Pull in other's flushed centroids and any buffered (unit-weight) values.
        for c in &other.centroids {
            self.centroids.push(*c);
        }
        for &v in &other.buffer {
            self.centroids.push(Centroid {
                mean: v,
                weight: 1.0,
            });
        }
        self.n += other.n;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        // Re-compress the combined (now unsorted) centroid set.
        let mut all = std::mem::take(&mut self.centroids);
        all.sort_by(|a, b| a.mean.partial_cmp(&b.mean).expect("no NaN in digest"));
        self.centroids = Self::merge_sorted(self.compression, all, self.n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merge_all;

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
    fn quantiles_accurate() {
        let n = 100_000u64;
        let mut t = TDigest::new(200.0);
        for x in shuffled(n) {
            t.add(x);
        }
        assert_eq!(t.count(), n);
        // True value at quantile q over 0..n is ≈ q·(n-1). Bulk within ~1%·n,
        // tails tighter (t-digest's adaptive resolution).
        let cases: &[(f64, f64)] = &[
            (0.1, 0.01),
            (0.25, 0.01),
            (0.5, 0.01),
            (0.75, 0.01),
            (0.9, 0.01),
            (0.99, 0.005), // tail: tighter
            (0.999, 0.002),
        ];
        for &(q, tol) in cases {
            let est = t.quantile(q).unwrap();
            let truth = q * (n - 1) as f64;
            let err = (est - truth).abs() / n as f64;
            assert!(
                err < tol,
                "q={q} est={est} truth={truth} err={err} tol={tol}"
            );
        }
    }

    #[test]
    fn tails_more_accurate_than_bulk() {
        // The scale function gives the extreme tails finer resolution; verify p999
        // lands very close in absolute rank terms.
        let n = 100_000u64;
        let mut t = TDigest::new(200.0);
        for x in shuffled(n) {
            t.add(x);
        }
        let p999 = t.quantile(0.999).unwrap();
        let truth = 0.999 * (n - 1) as f64;
        assert!(
            (p999 - truth).abs() / (n as f64) < 0.002,
            "p999={p999} truth={truth}"
        );
    }

    #[test]
    fn min_max_exact() {
        let mut t = TDigest::new(100.0);
        for x in shuffled(50_000) {
            t.add(x);
        }
        assert_eq!(t.min(), Some(0.0));
        assert_eq!(t.max(), Some(49_999.0));
        assert_eq!(t.quantile(0.0), Some(0.0));
        assert_eq!(t.quantile(1.0), Some(49_999.0));
    }

    #[test]
    fn rank_is_selectivity() {
        let mut t = TDigest::new(200.0);
        for x in shuffled(100_000) {
            t.add(x);
        }
        let sel = t.rank(30_000.0);
        assert!((sel - 0.3).abs() < 0.02, "selectivity {sel}");
        assert_eq!(t.rank(-1.0), 0.0);
        assert_eq!(t.rank(200_000.0), 1.0);
    }

    #[test]
    fn merge_matches_single() {
        let n = 100_000u64;
        let vals = shuffled(n);
        let mut whole = TDigest::new(200.0);
        for &x in &vals {
            whole.add(x);
        }
        let mut a = TDigest::new(200.0);
        let mut b = TDigest::new(200.0);
        for (i, &x) in vals.iter().enumerate() {
            if i % 2 == 0 {
                a.add(x);
            } else {
                b.add(x);
            }
        }
        a.merge(&b);
        assert_eq!(a.count(), n);
        for &q in &[0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let merged = a.quantile(q).unwrap();
            let single = whole.quantile(q).unwrap();
            let err = (merged - single).abs() / n as f64;
            assert!(
                err < 0.02,
                "q={q} merged={merged} single={single} err={err}"
            );
        }
    }

    #[test]
    fn merge_all_partitions() {
        let n = 80_000u64;
        let vals = shuffled(n);
        let mut whole = TDigest::new(200.0);
        for &x in &vals {
            whole.add(x);
        }
        let parts = (0..4)
            .map(|p| {
                let mut t = TDigest::new(200.0);
                for (i, &x) in vals.iter().enumerate() {
                    if i % 4 == p {
                        t.add(x);
                    }
                }
                t
            })
            .collect::<Vec<_>>();
        let mut merged = merge_all(parts.into_iter()).unwrap();
        assert_eq!(merged.count(), n);
        for &q in &[0.25, 0.5, 0.75, 0.9] {
            let m = merged.quantile(q).unwrap();
            let s = whole.quantile(q).unwrap();
            assert!((m - s).abs() / (n as f64) < 0.02, "q={q} m={m} s={s}");
        }
    }

    #[test]
    #[should_panic(expected = "compression mismatch")]
    fn merge_rejects_compression_mismatch() {
        let mut a = TDigest::new(100.0);
        let b = TDigest::new(200.0);
        a.add(1.0);
        a.merge(&b);
    }

    #[test]
    fn bytes_roundtrip_preserves_quantiles() {
        let mut t = TDigest::new(200.0);
        for x in shuffled(100_000) {
            t.add(x);
        }
        // Materialize quantiles before serializing.
        let before: Vec<f64> = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
            .iter()
            .map(|&q| t.quantile(q).unwrap())
            .collect();
        let bytes = t.to_bytes();
        let mut back = TDigest::from_bytes(&bytes).expect("valid blob");
        assert_eq!(back.count(), t.count());
        assert_eq!(back.min(), t.min());
        assert_eq!(back.max(), t.max());
        for (i, &q) in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
            .iter()
            .enumerate()
        {
            assert_eq!(back.quantile(q).unwrap(), before[i], "q={q} roundtrip");
        }
    }

    #[test]
    fn empty_sketch() {
        let mut t = TDigest::new(100.0);
        assert!(t.is_empty());
        assert_eq!(t.count(), 0);
        assert_eq!(t.quantile(0.5), None);
        assert_eq!(t.min(), None);
        assert_eq!(t.max(), None);
        assert_eq!(t.rank(0.0), 0.0);
    }

    #[test]
    fn empty_sketch_roundtrips() {
        let t = TDigest::new(100.0);
        let mut back = TDigest::from_bytes(&t.to_bytes()).expect("valid blob");
        assert!(back.is_empty());
        assert_eq!(back.quantile(0.5), None);
    }

    #[test]
    fn single_value() {
        let mut t = TDigest::new(100.0);
        t.add(42.0);
        assert_eq!(t.count(), 1);
        assert_eq!(t.quantile(0.0), Some(42.0));
        assert_eq!(t.quantile(0.5), Some(42.0));
        assert_eq!(t.quantile(1.0), Some(42.0));
    }

    #[test]
    fn from_bytes_rejects_malformed() {
        assert!(TDigest::from_bytes(&[]).is_none());
        assert!(TDigest::from_bytes(&[0; 7]).is_none());
        let mut t = TDigest::new(100.0);
        t.add(1.0);
        let mut bytes = t.to_bytes();
        bytes.push(0);
        assert!(TDigest::from_bytes(&bytes).is_none());
        // compression < 1 invalid.
        let mut bad = 0.5f64.to_le_bytes().to_vec();
        bad.extend_from_slice(&0.0f64.to_le_bytes()); // n
        bad.extend_from_slice(&f64::INFINITY.to_le_bytes());
        bad.extend_from_slice(&f64::NEG_INFINITY.to_le_bytes());
        bad.extend_from_slice(&0u64.to_le_bytes()); // len
        assert!(TDigest::from_bytes(&bad).is_none());
    }

    #[test]
    fn skewed_distribution() {
        // Exponential-ish skew: many small, few huge. Tail quantiles must track.
        let mut t = TDigest::new(200.0);
        let mut vals: Vec<f64> = Vec::new();
        let mut state = 0xABCD_1234u64;
        for _ in 0..100_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
                                                                // Inverse-CDF of exp(1): -ln(1-u).
            vals.push(-(1.0 - u).ln());
        }
        for &x in &vals {
            t.add(x);
        }
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for &q in &[0.5, 0.9, 0.99] {
            let est = t.quantile(q).unwrap();
            let idx = ((q * (sorted.len() - 1) as f64).round() as usize).min(sorted.len() - 1);
            let truth = sorted[idx];
            // Relative error on a skewed positive distribution; generous but real.
            let rel = (est - truth).abs() / truth.max(1e-9);
            assert!(rel < 0.1, "q={q} est={est} truth={truth} rel={rel}");
        }
    }

    // ---- Property tests ----------------------------------------------------

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
        fn finite(&mut self) -> f64 {
            let frac = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            frac * 2_000_000.0 - 1_000_000.0
        }
    }

    #[test]
    fn prop_serialize_roundtrip() {
        const TRIALS: usize = 100;
        let mut rng = XorShift64::new(0x5151_7A55_9000_0001);
        for trial in 0..TRIALS {
            let compression = 20.0 + rng.below(300) as f64;
            let n = rng.below(4_000);
            let mut t = TDigest::new(compression);
            for _ in 0..n {
                t.add(rng.finite());
            }
            // Canonicalize before comparing (to_bytes flushes a clone).
            let qs_before: Vec<Option<f64>> = [0.0, 0.25, 0.5, 0.75, 1.0]
                .iter()
                .map(|&q| t.quantile(q))
                .collect();
            let mut back = TDigest::from_bytes(&t.to_bytes()).expect("valid blob");
            assert_eq!(back.count(), t.count(), "trial {trial}");
            assert_eq!(back.min(), t.min());
            assert_eq!(back.max(), t.max());
            for (i, &q) in [0.0, 0.25, 0.5, 0.75, 1.0].iter().enumerate() {
                assert_eq!(back.quantile(q), qs_before[i], "trial {trial} q={q}");
            }
        }
    }

    #[test]
    fn prop_merge_close_to_single() {
        const TRIALS: usize = 60;
        let mut rng = XorShift64::new(0x9A9A_C0DE_2222_3333);
        for trial in 0..TRIALS {
            let compression = 100.0 + rng.below(200) as f64;
            let n = 5_000 + rng.below(5_000);
            let vals: Vec<f64> = (0..n).map(|_| rng.finite()).collect();
            let mut whole = TDigest::new(compression);
            for &x in &vals {
                whole.add(x);
            }
            let parts = 2 + rng.below(4) as usize;
            let mut sketches: Vec<TDigest> =
                (0..parts).map(|_| TDigest::new(compression)).collect();
            for &x in &vals {
                let p = rng.below(parts as u64) as usize;
                sketches[p].add(x);
            }
            let mut merged = merge_all(sketches.into_iter()).unwrap();
            assert_eq!(merged.count(), n);
            // Domain spans 2e6; allow ~2% of range for merge vs single divergence.
            for &q in &[0.1, 0.25, 0.5, 0.75, 0.9] {
                let m = merged.quantile(q).unwrap();
                let s = whole.quantile(q).unwrap();
                let err = (m - s).abs() / 2_000_000.0;
                assert!(err < 0.02, "trial {trial} q={q} m={m} s={s} err={err}");
            }
        }
    }
}
