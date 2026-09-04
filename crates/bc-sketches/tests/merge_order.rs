//! What "mergeable in any order" actually means, per sketch.
//!
//! `combine` must be associative and commutative, because a shuffle's reduce receives
//! partials in an order nobody chooses. Which sketches here satisfy that *exactly* is not
//! uniform, and it was not written down: three reach a bit-identical state whatever the
//! order, and the quantile sketches do not, because their merge re-clusters or compacts and
//! that is order-sensitive by construction rather than by defect.
//!
//! Both halves are pinned. The exact ones so a regression is caught. The approximate ones on
//! the property that actually matters to a caller -- that two merge orders agree to within
//! the sketch's own rank error -- rather than on an equality they were never going to hold,
//! and rather than on their *inequality*, which would be a test demanding they never improve.
//!
//! Measured in rank space, not value space, so the bound does not depend on how the data
//! happens to be distributed.

use bc_sketches::merge_all;

/// Deterministic pseudo-random partitions over `0..1000`.
fn parts(n: usize, per: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            (0..per)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((state >> 11) as f64) / ((1u64 << 53) as f64) * 1000.0
                })
                .collect()
        })
        .collect()
}

/// Forward, reverse, and a shuffle: three orders a reduce could plausibly see.
const ORDERS: [[usize; 8]; 3] = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [7, 6, 5, 4, 3, 2, 1, 0],
    [3, 0, 6, 1, 7, 2, 5, 4],
];

/// The fraction of `all` at or below `v` -- where a quantile estimate actually landed.
fn true_rank(all: &[f64], v: f64) -> f64 {
    all.iter().filter(|&&x| x <= v).count() as f64 / all.len() as f64
}

#[test]
fn hll_countmin_and_bloom_reach_the_same_state_in_any_merge_order() {
    // These three merge by register-wise max, cell-wise sum and bitwise OR. Each of those
    // is associative and commutative on the nose, so the merged state is identical, not
    // merely close -- and that is worth pinning, because it is what lets a caller compare
    // two runs' distinct counts directly.
    let parts = parts(8, 400, 7);

    let hll = |order: &[usize]| {
        merge_all(order.iter().map(|&i| {
            let mut sketch = bc_sketches::HyperLogLog::new(12);
            for &x in &parts[i] {
                sketch.add(&(x as u64));
            }
            sketch
        }))
        .unwrap()
        .estimate()
    };
    let countmin = |order: &[usize]| {
        let merged = merge_all(order.iter().map(|&i| {
            let mut sketch = bc_sketches::CountMinSketch::new(256, 4);
            for &x in &parts[i] {
                sketch.add(&(x as u64));
            }
            sketch
        }))
        .unwrap();
        (0..200u64).map(|k| merged.estimate(&k)).collect::<Vec<_>>()
    };
    let bloom = |order: &[usize]| {
        let merged = merge_all(order.iter().map(|&i| {
            let mut sketch = bc_sketches::BloomFilter::new(8192, 4);
            for &x in &parts[i] {
                sketch.add(&(x as u64));
            }
            sketch
        }))
        .unwrap();
        (0..400u64).map(|k| merged.contains(&k)).collect::<Vec<_>>()
    };

    for order in &ORDERS[1..] {
        assert_eq!(
            hll(&ORDERS[0]),
            hll(order),
            "HyperLogLog moved with the merge order"
        );
        assert_eq!(
            countmin(&ORDERS[0]),
            countmin(order),
            "CountMin moved with the merge order"
        );
        assert_eq!(
            bloom(&ORDERS[0]),
            bloom(order),
            "Bloom moved with the merge order"
        );
    }
}

#[test]
fn column_stats_scalars_are_identical_whatever_the_merge_order() {
    // min/max/count/ndv fold by min, max, sum and a HyperLogLog, so they are exact. The
    // quantile grid it also carries is not, which is why only the scalars are asserted here
    // -- see the quantile test below for what holds of the rest of it.
    use arrow::array::{ArrayRef, Float64Array};
    use std::sync::Arc;

    let parts = parts(8, 400, 41);
    let arrays: Vec<ArrayRef> = parts
        .iter()
        .map(|v| Arc::new(Float64Array::from(v.clone())) as ArrayRef)
        .collect();

    let scalars = |order: &[usize]| {
        let merged = merge_all(
            order
                .iter()
                .map(|&i| bc_sketches::ColumnStats::from_array(&arrays[i])),
        )
        .unwrap();
        (merged.min(), merged.max(), merged.distinct_estimate())
    };

    for order in &ORDERS[1..] {
        assert_eq!(
            scalars(&ORDERS[0]),
            scalars(order),
            "a ColumnStats scalar moved with the merge order"
        );
    }
}

#[test]
fn the_quantile_sketches_agree_within_their_rank_error_in_any_merge_order() {
    // KLL compacts and TDigest re-clusters centroids, and both are order-sensitive doing so,
    // so two merge orders do not produce the same estimate. What must hold is that they land
    // within the accuracy the sketch already promises, because a caller reading a p99 across
    // two runs is entitled to the sketch's error and nothing worse.
    //
    // Bounds are the algorithms', with headroom. KLL at k=200 has a rank error near 1/k, and
    // two independently merged copies can each be off by that, so ~2/k is the theoretical
    // gap; measured over 39 seeds and three orders the worst was 0.0097 against that 0.01.
    // TDigest at compression 100 measured 0.0050. Both are asserted at roughly 3x the
    // measurement so a real regression fails while seed noise does not.
    const KLL_RANK_GAP: f64 = 0.03;
    const TDIGEST_RANK_GAP: f64 = 0.02;

    for seed in 1..12u64 {
        let parts = parts(8, 400, seed);
        let all: Vec<f64> = parts.iter().flatten().copied().collect();

        let kll_ranks = |order: &[usize]| {
            let merged = merge_all(order.iter().map(|&i| {
                let mut sketch = bc_sketches::KllSketch::new(200);
                for &x in &parts[i] {
                    sketch.add(x);
                }
                sketch
            }))
            .unwrap();
            (1..20)
                .map(|q| true_rank(&all, merged.quantile(f64::from(q) / 20.0).unwrap()))
                .collect::<Vec<_>>()
        };
        let tdigest_ranks = |order: &[usize]| {
            let mut merged = merge_all(order.iter().map(|&i| {
                let mut sketch = bc_sketches::TDigest::new(100.0);
                for &x in &parts[i] {
                    sketch.add(x);
                }
                sketch
            }))
            .unwrap();
            (1..20)
                .map(|q| true_rank(&all, merged.quantile(f64::from(q) / 20.0).unwrap()))
                .collect::<Vec<_>>()
        };

        let base_kll = kll_ranks(&ORDERS[0]);
        let base_tdigest = tdigest_ranks(&ORDERS[0]);
        for order in &ORDERS[1..] {
            for (a, b) in base_kll.iter().zip(&kll_ranks(order)) {
                assert!(
                    (a - b).abs() <= KLL_RANK_GAP,
                    "seed {seed}: KLL ranks {a} and {b} are {} apart, past its rank error",
                    (a - b).abs()
                );
            }
            for (a, b) in base_tdigest.iter().zip(&tdigest_ranks(order)) {
                assert!(
                    (a - b).abs() <= TDIGEST_RANK_GAP,
                    "seed {seed}: TDigest ranks {a} and {b} are {} apart, past its rank error",
                    (a - b).abs()
                );
            }
        }
    }
}
