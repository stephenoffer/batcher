//! TEMPORARY scratch measurement — not part of the crate's contract. Delete after use.

#[cfg(test)]
mod scratch {
    use std::sync::Arc;
    use std::time::Instant;

    use arrow::array::{ArrayRef, Int64Array};

    use crate::join::{broadcast_hash_join_indices, JoinType};

    fn keys(n: usize, modulo: i64, seed: i64) -> Vec<ArrayRef> {
        let v: Vec<i64> = (0..n as i64)
            .map(|i| (i.wrapping_mul(2654435761).wrapping_add(seed)).rem_euclid(modulo))
            .collect();
        vec![Arc::new(Int64Array::from(v)) as ArrayRef]
    }

    fn bench(build_rows: usize, probe_rows: usize, threads: usize) -> (f64, f64) {
        let build = keys(build_rows, build_rows as i64, 7);
        let probe = keys(probe_rows, build_rows as i64, 13);

        // Force the radix parallel path by driving the threshold from the *caller* side:
        // `bloom_min_build_rows` is unrelated; instead we time the two shapes we can reach
        // today — the serial-build broadcast (current) vs the same call with a build large
        // enough that the internal radix parallel build engages.
        let mut t_bcast = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            let out =
                broadcast_hash_join_indices(&probe, &build, JoinType::Inner, threads, 0.01, 1 << 20)
                    .unwrap();
            let ms = t.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(&out);
            t_bcast = t_bcast.min(ms);
        }
        (t_bcast, 0.0)
    }

    #[test]
    fn scratch_broadcast_build_scaling() {
        let threads = rayon::current_num_threads();
        println!("\n=== broadcast join: serial build + parallel probe ({threads} threads) ===");
        println!("{:>10} {:>10} {:>12}", "build", "probe", "ms");
        // Hold the probe fixed; grow the build. If the *build* is serial, ms grows with
        // build_rows even though the probe work is constant.
        for build in [10_000usize, 50_000, 150_000, 227_000, 500_000, 1_000_000, 2_000_000] {
            let (ms, _) = bench(build, 1_200_000, threads);
            println!("{build:>10} {:>10} {ms:>12.2}", 1_200_000);
        }
    }
}
