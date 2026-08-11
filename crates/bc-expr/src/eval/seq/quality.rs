//! FASTQ quality-string decoding — Phred scores, mean quality, and expected errors.
//!
//! A FASTQ quality string is a per-base confidence encoded one ASCII character per base. Every
//! read-filtering decision in sequencing is a threshold on some reduction of that string, and
//! doing the decode outside the engine means materializing a list of integers per read in
//! Python — for a run that is hundreds of millions of reads.
//!
//! The **offset** is the one thing the data cannot tell you. Sanger and Illumina 1.8+ encode
//! `Q + 33`; the older Illumina 1.3-1.7 pipelines encoded `Q + 64`. The two ranges overlap, so
//! a file gives no reliable signal and guessing shifts every score by 31 — turning a Q40 base
//! into Q9 or the reverse. It defaults to 33 (what every current instrument writes) and is
//! stated rather than sniffed.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, Int32Builder, ListBuilder, StringArray};

/// Sanger / Illumina 1.8+ encoding, which is what every current instrument writes.
const DEFAULT_OFFSET: i64 = 33;

/// Iterate a row's Phred scores at `offset`, as `i32`.
///
/// A character below the offset yields a **negative** score rather than being clamped or
/// dropped. That is deliberate: a negative score is the unmistakable signature of the wrong
/// offset, and clamping it to zero would hide the one mistake this whole module is built to
/// make visible.
#[inline]
fn scores(seq: &str, offset: i64) -> impl Iterator<Item = i32> + '_ {
    let off = offset as i32;
    seq.bytes().map(move |b| b as i32 - off)
}

/// `phred_quality(offset)` → `List<Int32>` of per-base scores.
pub(super) fn phred_quality(s: &StringArray, offset: Option<i64>) -> ArrayRef {
    let offset = offset.unwrap_or(DEFAULT_OFFSET);
    let mut builder =
        ListBuilder::with_capacity(Int32Builder::with_capacity(s.value_data().len()), s.len());
    for i in 0..s.len() {
        if s.is_null(i) {
            builder.append_null();
            continue;
        }
        for q in scores(s.value(i), offset) {
            builder.values().append_value(q);
        }
        builder.append(true);
    }
    Arc::new(builder.finish())
}

/// `mean_quality(offset)` → the arithmetic mean of the Phred scores (→ Float64).
///
/// This is the "average quality" every FASTQ tool reports (FastQC's per-read quality,
/// `seqkit fx2tab -q`), and it is what a `mean_quality >= 20` filter means. It is **not** the
/// quality corresponding to the read's average error rate — averaging in log space
/// systematically overstates a read whose errors are concentrated in a bad tail. Use
/// [`expected_errors`] when the question is how many bases are actually likely wrong.
///
/// An empty quality string has no mean and yields null rather than 0.
pub(super) fn mean_quality(s: &StringArray, offset: Option<i64>) -> ArrayRef {
    let offset = offset.unwrap_or(DEFAULT_OFFSET);
    let values: Vec<Option<f64>> = (0..s.len())
        .map(|i| {
            if s.is_null(i) {
                return None;
            }
            let seq = s.value(i);
            let n = seq.len();
            (n > 0).then(|| scores(seq, offset).map(f64::from).sum::<f64>() / n as f64)
        })
        .collect();
    Arc::new(Float64Array::from(values))
}

/// `expected_errors(offset)` → `Σ 10^(-Q/10)`, the expected number of miscalled bases in the
/// read (→ Float64).
///
/// The quality filter that actually corresponds to a claim about the data: "this read probably
/// contains fewer than one error" is `expected_errors() < 1.0`. It is the `fastq_maxee`
/// criterion from USEARCH and VSEARCH, and it is strictly better than a mean-quality threshold
/// because it is additive over bases — one Q2 base contributes as much as sixty Q20 bases, and
/// a mean cannot see that.
///
/// An empty quality string yields 0.0: a read with no bases has no expected errors, which is
/// arithmetically true and keeps the quantity summable across a partition.
pub(super) fn expected_errors(s: &StringArray, offset: Option<i64>) -> ArrayRef {
    let offset = offset.unwrap_or(DEFAULT_OFFSET);
    // One `powf` per *distinct byte* rather than per base. A quality character is a single
    // byte, so there are only 256 possible error probabilities for a given offset; computing
    // them up front turns the inner loop into a table load.
    //
    // This was measured, not assumed: at 150 bp reads `expected_errors` ran at 48 MB/s
    // against `mean_quality`'s 739 MB/s over the same column, and the only difference
    // between them is this transcendental. The table closes essentially all of that gap.
    let mut error_prob = [0.0f64; 256];
    for (byte, slot) in error_prob.iter_mut().enumerate() {
        let q = byte as i64 - offset;
        *slot = 10f64.powf(-(q as f64) / 10.0);
    }
    let values: Vec<Option<f64>> = (0..s.len())
        .map(|i| {
            (!s.is_null(i)).then(|| {
                // Folded from an explicit `0.0` rather than `.sum()`, which starts from
                // `-0.0` — so an empty quality string reported "-0" expected errors. Every
                // term here is positive, so the two agree on every non-empty input; the
                // difference is only the identity, and a count that renders as `-0.0` is
                // a wart a reader is right to distrust. `assert_eq!(x, 0.0)` cannot see it,
                // which is why this is a comment and not just a fix.
                s.value(i)
                    .bytes()
                    .fold(0.0, |acc, b| acc + error_prob[b as usize])
            })
        })
        .collect();
    Arc::new(Float64Array::from(values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::seq::test_col;
    use arrow::array::{Int32Array, ListArray};

    fn floats(a: &ArrayRef) -> Vec<Option<f64>> {
        let f = a.as_any().downcast_ref::<Float64Array>().unwrap();
        (0..f.len())
            .map(|i| (!f.is_null(i)).then(|| f.value(i)))
            .collect()
    }

    #[test]
    fn phred_decodes_the_sanger_range() {
        // '!' is Q0 (the lowest Sanger character) and 'I' is Q40 (a typical maximum).
        let c = test_col(&[Some("!5I"), None]);
        let out = phred_quality(&c, None);
        let l = out.as_any().downcast_ref::<ListArray>().unwrap();
        let v = l.value(0);
        let q = v.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(
            (0..3).map(|i| q.value(i)).collect::<Vec<_>>(),
            vec![0, 20, 40]
        );
        assert!(l.is_null(1));
    }

    #[test]
    fn the_offset_shifts_every_score_by_thirty_one() {
        // The whole reason the offset is stated rather than sniffed: the same bytes decode to
        // two very different reads.
        let c = test_col(&[Some("IIII")]);
        assert_eq!(floats(&mean_quality(&c, Some(33))), vec![Some(40.0)]);
        assert_eq!(floats(&mean_quality(&c, Some(64))), vec![Some(9.0)]);
    }

    #[test]
    fn a_character_below_the_offset_reports_a_negative_score() {
        // Clamping to zero here would hide the one mistake worth surfacing.
        let out = phred_quality(&test_col(&[Some("!")]), Some(64));
        let l = out.as_any().downcast_ref::<ListArray>().unwrap();
        let v = l.value(0);
        assert_eq!(
            v.as_any().downcast_ref::<Int32Array>().unwrap().value(0),
            -31
        );
    }

    #[test]
    fn expected_errors_is_additive_where_a_mean_is_not() {
        // One Q2 base among Q40s: the mean stays high, the expected-error count does not.
        let good = test_col(&[Some("IIIIIIIIII")]);
        let one_bad = test_col(&[Some("IIIIIIIII#")]);
        let mean_good = floats(&mean_quality(&good, None))[0].unwrap();
        let mean_bad = floats(&mean_quality(&one_bad, None))[0].unwrap();
        let ee_good = floats(&expected_errors(&good, None))[0].unwrap();
        let ee_bad = floats(&expected_errors(&one_bad, None))[0].unwrap();
        assert!(mean_bad > 35.0, "the mean barely moves: {mean_bad}");
        // Q2 is a ~63% chance of a miscall, so this read is expected to contain an error.
        assert!(ee_bad > 0.6, "{ee_bad}");
        assert!(ee_good < 0.01, "{ee_good}");
        assert!(mean_good > mean_bad);
    }

    #[test]
    fn q20_is_one_percent_and_q30_is_one_in_a_thousand() {
        // The definition of the Phred scale, which everything else here rests on.
        let c = test_col(&[Some("5"), Some("?")]);
        let ee = floats(&expected_errors(&c, None));
        assert!((ee[0].unwrap() - 0.01).abs() < 1e-12);
        assert!((ee[1].unwrap() - 0.001).abs() < 1e-12);
    }

    #[test]
    fn an_empty_quality_string_has_no_mean_but_no_errors_either() {
        let c = test_col(&[Some("")]);
        assert_eq!(floats(&mean_quality(&c, None)), vec![None]);
        assert_eq!(floats(&expected_errors(&c, None)), vec![Some(0.0)]);
        // Positive zero specifically. `assert_eq!(x, 0.0)` passes for `-0.0` too, so the
        // sign has to be asserted on its own or the `.sum()` identity creeps back in.
        assert!(
            floats(&expected_errors(&c, None))[0]
                .unwrap()
                .is_sign_positive(),
            "an expected-error count must not render as -0"
        );
    }
}
