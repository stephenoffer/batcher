//! `row_sort::stable_lexsort_indices` must produce the *same permutation* as the path it
//! replaces, at a scale the unit tests do not reach, and be faster.
//!
//! The unit tests beside the implementation cover the shapes: directions, nulls, ties,
//! mixed types, the crossover gate. This file covers the one thing they cannot, which is
//! whether the equivalence survives a realistic row count with realistic tie density — the
//! regime where a sort's tie handling actually decides the answer, and where the encoding
//! saving is worth measuring.
//!
//! ```text
//! cargo test --release -p bc-arrow --test row_sort_equivalence -- --nocapture
//! ```
//!
//! Timings are printed, never asserted: a ratio threshold in a test is a flake on a shared
//! machine, and the equality assertions above it are the point.

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{ArrayRef, Int64Array, StringArray, UInt32Array};
use arrow::compute::{lexsort_to_indices, SortColumn, SortOptions};
use bc_arrow::row_sort::{stable_lexsort_indices, MIN_KEYS_FOR_ROW_ENCODING};

/// Enough rows for ties to be dense and for the sort to leave cache, small enough to run in
/// a debug build. The timing study below raises it.
const ROWS: usize = 1 << 16;

fn opts(descending: bool, nulls_first: bool) -> SortOptions {
    SortOptions {
        descending,
        nulls_first,
    }
}

/// The permutation the engine produced before this existed: the keys plus an appended
/// ascending row-index column, through arrow's unstable `lexsort_to_indices`.
fn reference(columns: &[ArrayRef], options: &[SortOptions]) -> UInt32Array {
    let mut sort_columns: Vec<SortColumn> = columns
        .iter()
        .zip(options)
        .map(|(values, o)| SortColumn {
            values: Arc::clone(values),
            options: Some(*o),
        })
        .collect();
    let n = columns[0].len();
    sort_columns.push(SortColumn {
        values: Arc::new(UInt32Array::from_iter_values(0..n as u32)),
        options: Some(opts(false, false)),
    });
    lexsort_to_indices(&sort_columns, None).expect("reference lexsort")
}

/// Two integer keys with heavy ties (`ROWS / 1000` and `ROWS / 100` distinct values) plus a
/// low-cardinality string, so a fully-tied group is common and the tie-break is load-bearing.
/// Nulls are sprinkled through both integer keys.
fn columns(rows: usize) -> (Vec<ArrayRef>, Vec<&'static str>) {
    let mix = |i: usize, k: usize| (i.wrapping_mul(2_654_435_761)) % k;
    let a: Int64Array = (0..rows)
        .map(|i| (i % 13 != 0).then(|| mix(i, 1_000) as i64))
        .collect();
    let b: Int64Array = (0..rows)
        .map(|i| (i % 17 != 0).then(|| mix(i, 100) as i64))
        .collect();
    let modes = ["AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB", "REG AIR"];
    let s: StringArray = (0..rows).map(|i| Some(modes[mix(i, 7)])).collect();
    let d: Int64Array = (0..rows).map(|i| Some(mix(i, 31) as i64)).collect();
    let e: Int64Array = (0..rows).map(|i| Some(mix(i, 3) as i64)).collect();
    (
        vec![
            Arc::new(a),
            Arc::new(b),
            Arc::new(s),
            Arc::new(d),
            Arc::new(e),
        ],
        vec!["int", "int", "utf8", "int", "int"],
    )
}

fn time<T>(reps: usize, mut f: impl FnMut() -> T) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        std::hint::black_box(f());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

#[test]
fn stable_row_sort_equals_the_reference_permutation_at_scale() {
    let (cols, _) = columns(ROWS);
    for width in MIN_KEYS_FOR_ROW_ENCODING..=cols.len() {
        let subset = &cols[..width];
        for descending in [false, true] {
            for nulls_first in [false, true] {
                let options = vec![opts(descending, nulls_first); width];
                let expected = reference(subset, &options);
                let got = stable_lexsort_indices(subset, &options)
                    .expect("integer and utf8 keys are row-encodable");
                assert_eq!(
                    got, expected,
                    "width={width} descending={descending} nulls_first={nulls_first}: \
                     the row-encoded sort must be the same permutation"
                );
            }
        }
    }
}

/// The source of the crossover table in `row_sort`'s module docs. Ignored by default: a
/// million rows sorted every width, twice over, is minutes in a debug build, and the number
/// is only meaningful in release anyway.
#[test]
#[ignore = "timing study behind MIN_KEYS_FOR_ROW_ENCODING; run with --release -- --ignored"]
fn report_the_row_encoding_crossover() {
    const BIG: usize = 1 << 20;
    let (cols, kinds) = columns(BIG);
    println!("\n=== multi-key sort: {BIG} rows, keys {kinds:?} ===");

    // Every width, including those below the gate, because the point of the table is *why*
    // the gate sits where it does. Below it `stable_lexsort_indices` declines, so the encoder
    // is timed through a local copy that skips only the gate.
    for width in 1..=cols.len() {
        let subset = &cols[..width];
        let options = vec![opts(false, false); width];
        let old = time(3, || reference(subset, &options));
        let new = time(3, || encode_and_sort(subset, &options));
        println!(
            "{width} key(s): reference {old:>8.2} ms -> row-encoded {new:>8.2} ms ({:>5.2}x)",
            old / new
        );
    }
}

/// `stable_lexsort_indices` without its key-count gate, so the timing study can show what
/// the encoding costs at one and two keys — the measurement that puts the gate where it is.
/// Kept to the test binary; nothing in the engine may skip the gate.
fn encode_and_sort(columns: &[ArrayRef], options: &[SortOptions]) -> UInt32Array {
    use arrow::row::{RowConverter, SortField};
    let fields: Vec<SortField> = columns
        .iter()
        .zip(options)
        .map(|(c, o)| SortField::new_with_options(c.data_type().clone(), *o))
        .collect();
    let converter = RowConverter::new(fields).expect("row-encodable");
    let rows = converter.convert_columns(columns).expect("encodes");
    let mut permutation: Vec<u32> = (0..columns[0].len() as u32).collect();
    permutation.sort_unstable_by(|&a, &b| {
        rows.row(a as usize)
            .cmp(&rows.row(b as usize))
            .then(a.cmp(&b))
    });
    UInt32Array::from(permutation)
}
