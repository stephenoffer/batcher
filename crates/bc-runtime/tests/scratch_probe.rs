//! TEMPORARY scratch measurement of the hash-join bloom pre-filter. Delete after use.
//!
//! Every shard of `build_sharded` allocates a FULL-SIZE bloom, then they are OR-merged, so the
//! bloom's cost scales with `shards x full_size`: 34% of a 147k build, 69% of a 2M build. The
//! bloom is supposed to buy that back by letting the probe reject non-matching rows without
//! touching the hash table. This asks whether it does, END TO END (build + parallel probe),
//! across the selectivities that decide it.

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, Int64Array};
use bc_runtime::join::{BroadcastProbe, JoinType};
use rayon::prelude::*;

fn keys(n: usize, modulo: i64, seed: i64) -> Vec<ArrayRef> {
    let v: Vec<i64> = (0..n as i64)
        .map(|i| (i.wrapping_mul(2654435761).wrapping_add(seed)).rem_euclid(modulo))
        .collect();
    vec![Arc::new(Int64Array::from(v)) as ArrayRef]
}

const MORSEL: usize = 16_384;

/// (build_ms, probe_ms) best-of-N. `bloom` off is expressed by lifting the min-build threshold
/// above the build, which is exactly what `use_probe_bloom_with` keys on — nothing else changes.
fn run(build: &[ArrayRef], morsels: &[Vec<ArrayRef>], probe_rows: usize, bloom: bool) -> (f64, f64) {
    let rows = build[0].len();
    let min_build = if bloom { 1 << 12 } else { rows + 1 };
    let mut b_ms = f64::MAX;
    let mut p_ms = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        let table =
            BroadcastProbe::new(build, JoinType::Inner, probe_rows, 0.01, min_build).unwrap();
        b_ms = b_ms.min(t.elapsed().as_secs_f64() * 1e3);

        let t = Instant::now();
        let m: usize = morsels
            .par_iter()
            .map(|k| table.probe(k).unwrap().left.len())
            .sum();
        p_ms = p_ms.min(t.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(m);
    }
    (b_ms, p_ms)
}

#[test]
fn scratch_is_the_bloom_worth_it() {
    let probe_rows = 3_241_776usize;
    println!("\n  threads: {}", rayon::current_num_threads());
    println!("  probe rows: {probe_rows}   (build+probe, ms; lower is better)");
    println!(
        "\n  {:>9} {:>7} | {:>16} | {:>16} | {:>8}",
        "build", "match%", "WITH bloom (b+p)", "NO bloom (b+p)", "verdict"
    );

    for &build_rows in &[147_126usize, 500_000, 2_000_000] {
        let build = keys(build_rows, build_rows as i64, 7);
        // `spread` widens the probe's key domain past the build's, so most probe rows MISS --
        // the case the bloom exists for. spread=1 => nearly every probe row matches.
        for &spread in &[1i64, 4, 15] {
            let col = keys(probe_rows, (build_rows as i64) * spread, 13)[0].clone();
            let morsels: Vec<Vec<ArrayRef>> = (0..probe_rows)
                .step_by(MORSEL)
                .map(|off| vec![col.slice(off, MORSEL.min(probe_rows - off))])
                .collect();

            let (b1, p1) = run(&build, &morsels, probe_rows, true);
            let (b0, p0) = run(&build, &morsels, probe_rows, false);
            let tot1 = b1 + p1;
            let tot0 = b0 + p0;
            let win = if tot0 < tot1 { "NO BLOOM" } else { "bloom" };
            println!(
                "  {:>9} {:>6.0}% | {:>5.2}+{:>5.2}={:>5.2} | {:>5.2}+{:>5.2}={:>5.2} | {:>8} {:+.0}%",
                build_rows,
                100.0 / spread as f64,
                b1,
                p1,
                tot1,
                b0,
                p0,
                tot0,
                win,
                100.0 * (tot0 - tot1) / tot1,
            );
        }
    }
    println!();
}
