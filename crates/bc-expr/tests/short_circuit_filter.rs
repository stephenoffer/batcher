//! `Expr::short_circuit_filter_mask` must equal whole-batch evaluation on realistic
//! data, and be faster.
//!
//! The unit tests beside the implementation cover its edges — nulls, an empty result,
//! a declined predicate, the fallible classification. This file covers the thing those
//! cannot: a lineitem-shaped table streamed as many morsels, with the predicates that
//! actually appear in the benchmark suite, asserting the two paths agree row for row
//! on every one of them. That is the assertion that matters, because the optimization's
//! entire claim is that it changes nothing but the time.
//!
//! It also prints the timings, so the speedups quoted elsewhere have a named command
//! behind them rather than a remembered number:
//!
//! ```text
//! cargo test --release -p bc-expr --test short_circuit_filter -- --nocapture
//! ```
//!
//! The timings are printed, never asserted. A ratio threshold in a test is a flake on
//! a shared or throttled machine, and the correctness assertions above it are what
//! this file is for.

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Date32Array, Float64Array, Int64Array, RecordBatch, StringArray,
};
use arrow::compute::filter_record_batch;
use arrow::datatypes::{DataType, Field, Schema};
use bc_expr::{BinaryOp, Expr, Literal, StrFunc};

const MORSEL: usize = 16_384;
const MORSELS: usize = 64;

fn col(name: &str) -> Box<Expr> {
    Box::new(Expr::Col { name: name.into() })
}

fn and(left: Expr, right: Expr) -> Expr {
    Expr::Binary {
        op: BinaryOp::And,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn binary(op: BinaryOp, left: Box<Expr>, value: Literal) -> Expr {
    Expr::Binary {
        op,
        left,
        right: Box::new(Expr::Lit { value }),
    }
}

/// A lineitem-shaped morsel: the four columns TPC-H's selective filters read, a
/// low-cardinality shipmode, and twelve payload columns a real fact table carries and
/// a compacting filter must therefore *not* gather.
///
/// Every eleventh row is null in `l_discount` and every seventh in `l_partkey`, so the
/// null-drop half of the equivalence argument is exercised rather than assumed.
fn morsel(seed: i64) -> RecordBatch {
    let n = MORSEL as i64;
    let mix = |i: i64, k: i64| (i.wrapping_add(seed).wrapping_mul(2_654_435_761)).rem_euclid(k);

    let shipdate: Date32Array = (0..n).map(|i| Some(8_766 + mix(i, 2_557) as i32)).collect();
    let discount: Float64Array = (0..n)
        .map(|i| (i % 11 != 0).then(|| mix(i, 11) as f64 / 100.0))
        .collect();
    let quantity: Float64Array = (0..n).map(|i| Some(mix(i, 50) as f64 + 1.0)).collect();
    let partkey: Int64Array = (0..n)
        .map(|i| (i % 7 != 0).then(|| mix(i, 200_000)))
        .collect();
    let modes = ["AIR", "RAIL", "SHIP", "TRUCK", "MAIL", "FOB", "REG AIR"];
    let shipmode: StringArray = (0..n).map(|i| Some(modes[mix(i, 7) as usize])).collect();

    let mut fields = vec![
        Field::new("l_shipdate", DataType::Date32, true),
        Field::new("l_discount", DataType::Float64, true),
        Field::new("l_quantity", DataType::Float64, true),
        Field::new("l_partkey", DataType::Int64, true),
        Field::new("l_shipmode", DataType::Utf8, true),
    ];
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(shipdate),
        Arc::new(discount),
        Arc::new(quantity),
        Arc::new(partkey),
        Arc::new(shipmode),
    ];
    for p in 0..12 {
        fields.push(Field::new(format!("pay{p}"), DataType::Int64, true));
        let payload: Int64Array = (0..n).map(|i| Some(i * (p + 1))).collect();
        columns.push(Arc::new(payload));
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).expect("morsel")
}

/// TPC-H q6: a date range, a discount range and a quantity bound — five cheap
/// conjuncts, where the saving comes from skipping rather than from ordering.
fn q6() -> Expr {
    and(
        and(
            binary(BinaryOp::Ge, col("l_shipdate"), Literal::Date(9_131)),
            binary(BinaryOp::Lt, col("l_shipdate"), Literal::Date(9_496)),
        ),
        and(
            and(
                binary(BinaryOp::Ge, col("l_discount"), Literal::Float(0.05)),
                binary(BinaryOp::Le, col("l_discount"), Literal::Float(0.07)),
            ),
            binary(BinaryOp::Lt, col("l_quantity"), Literal::Float(24.0)),
        ),
    )
}

fn str_pred(func: StrFunc, name: &str, pattern: &str) -> Expr {
    Expr::Str {
        func,
        input: col(name),
        pattern: Some(pattern.into()),
        replacement: None,
        start: None,
        length: None,
    }
}

/// A cheap comparison guarding an expensive string predicate, written expensive-first
/// so that only the *reordering* can save anything. This is the ClickBench shape.
fn guarded_string_scan(func: StrFunc, pattern: &str) -> Expr {
    and(
        str_pred(func, "l_shipmode", pattern),
        binary(BinaryOp::Lt, col("l_quantity"), Literal::Float(3.0)),
    )
}

/// An `IN` list beside two comparisons — TPC-H q12/q19's shape.
fn in_list_and_ranges() -> Expr {
    and(
        and(
            Expr::InList {
                input: col("l_shipmode"),
                set: vec![
                    Literal::Str("AIR".into()),
                    Literal::Str("REG AIR".into()),
                    Literal::Str("RAIL".into()),
                ],
            },
            binary(BinaryOp::Lt, col("l_quantity"), Literal::Float(11.0)),
        ),
        binary(BinaryOp::Ge, col("l_partkey"), Literal::Int(100_000)),
    )
}

/// A predicate whose conjunct can fail on a row. It must be *declined*, and the
/// declined result must still be the right one via the ordinary path.
fn fallible() -> Expr {
    and(
        binary(BinaryOp::Lt, col("l_quantity"), Literal::Float(10.0)),
        binary(
            BinaryOp::Gt,
            Box::new(Expr::Binary {
                op: BinaryOp::Div,
                left: col("l_partkey"),
                right: col("l_partkey"),
            }),
            Literal::Int(0),
        ),
    )
}

/// The whole-batch path: evaluate the predicate, then gather. The reference.
fn whole_batch(pred: &Expr, batch: &RecordBatch) -> RecordBatch {
    let mask = pred.eval(batch).expect("whole-batch eval");
    let mask = mask
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("boolean predicate");
    filter_record_batch(batch, mask).expect("whole-batch filter")
}

/// What `bc-interp`'s Filter operator does: take the short-circuited mask when there
/// is one, and fall back otherwise.
fn short_circuit(pred: &Expr, batch: &RecordBatch) -> RecordBatch {
    match pred
        .short_circuit_filter_mask(batch)
        .expect("short-circuit must not error")
    {
        Some(mask) => filter_record_batch(batch, &mask).expect("short-circuit filter"),
        None => whole_batch(pred, batch),
    }
}

/// Assert the two paths agree on every morsel, then report what each cost.
fn compare(label: &str, pred: &Expr, morsels: &[RecordBatch]) {
    let mut kept = 0usize;
    for (i, batch) in morsels.iter().enumerate() {
        let expected = whole_batch(pred, batch);
        let got = short_circuit(pred, batch);
        assert_eq!(
            format!("{expected:?}"),
            format!("{got:?}"),
            "{label}: morsel {i} diverged from whole-batch evaluation"
        );
        kept += expected.num_rows();
    }
    let total: usize = morsels.iter().map(RecordBatch::num_rows).sum();
    assert!(
        kept > 0 && kept < total,
        "{label}: a predicate that keeps everything or nothing tests neither path \
         (kept {kept} of {total})"
    );

    // A declined predicate returns immediately, so timing its "mask" would print a
    // four-digit ratio for having done no work at all — the most misleading number
    // this file could emit. Say declined instead.
    if pred
        .short_circuit_filter_mask(&morsels[0])
        .expect("mask")
        .is_none()
    {
        let mut whole = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            for batch in morsels {
                std::hint::black_box(whole_batch(pred, batch));
            }
            whole = whole.min(t.elapsed().as_secs_f64() * 1e3);
        }
        println!(
            "{label:<26} kept {:>5.2}%   declined — whole-batch path only, {whole:>7.2} ms",
            100.0 * kept as f64 / total as f64,
        );
        return;
    }

    let mut mask_whole = f64::MAX;
    let mut mask_short = f64::MAX;
    let mut full_whole = f64::MAX;
    let mut full_short = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        for batch in morsels {
            std::hint::black_box(pred.eval(batch).expect("eval"));
        }
        mask_whole = mask_whole.min(t.elapsed().as_secs_f64() * 1e3);

        let t = Instant::now();
        for batch in morsels {
            std::hint::black_box(pred.short_circuit_filter_mask(batch).expect("mask"));
        }
        mask_short = mask_short.min(t.elapsed().as_secs_f64() * 1e3);

        let t = Instant::now();
        for batch in morsels {
            std::hint::black_box(whole_batch(pred, batch));
        }
        full_whole = full_whole.min(t.elapsed().as_secs_f64() * 1e3);

        let t = Instant::now();
        for batch in morsels {
            std::hint::black_box(short_circuit(pred, batch));
        }
        full_short = full_short.min(t.elapsed().as_secs_f64() * 1e3);
    }
    println!(
        "{label:<26} kept {:>5.2}%   mask {mask_whole:>7.2} -> {mask_short:>7.2} ms ({:>5.2}x)   \
         mask+gather {full_whole:>7.2} -> {full_short:>7.2} ms ({:>5.2}x)",
        100.0 * kept as f64 / total as f64,
        mask_whole / mask_short,
        full_whole / full_short,
    );
}

#[test]
fn short_circuit_equals_whole_batch_on_a_lineitem_shape() {
    let morsels: Vec<RecordBatch> = (0..MORSELS as i64).map(|s| morsel(s * 7)).collect();
    let rows: usize = morsels.iter().map(RecordBatch::num_rows).sum();
    println!("\n=== short-circuit conjunctive filter: {MORSELS} morsels, {rows} rows ===");

    compare("tpch-q6 (5 cheap)", &q6(), &morsels);
    compare("in-list + 2 ranges", &in_list_and_ranges(), &morsels);
    for (func, pattern) in [
        (StrFunc::Contains, "AIR"),
        (StrFunc::Like, "%AIR"),
        (StrFunc::RegexpMatches, "A.R"),
    ] {
        compare(
            &format!("guarded {func:?}"),
            &guarded_string_scan(func, pattern),
            &morsels,
        );
    }
    compare("fallible (declined)", &fallible(), &morsels);
}

/// The declined shapes must genuinely be declined, not merely produce the right
/// answer by luck. If a future change let a row-fallible conjunct through, the
/// equivalence test above would keep passing on data where nothing happens to
/// overflow — this is what would go red.
#[test]
fn a_row_fallible_predicate_is_declined() {
    let batch = morsel(0);
    assert!(
        fallible()
            .short_circuit_filter_mask(&batch)
            .expect("eval")
            .is_none(),
        "a conjunct that can divide by zero must not be short-circuited"
    );
    assert!(
        q6().short_circuit_filter_mask(&batch)
            .expect("eval")
            .is_some(),
        "five infallible comparisons must be short-circuited"
    );
}
