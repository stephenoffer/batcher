//! Duplex melting temperature — the nearest-neighbour thermodynamic model.
//!
//! `Tm` is what primer and probe design is a search over: an oligo that melts too low will not
//! anneal at the annealing step, one that melts too high binds where it should not. Screening a
//! candidate set is a filter over a computed column, which is exactly the shape this engine is
//! for — and the alternative, a per-row call into Biopython, is a control-plane row loop over
//! millions of candidates.
//!
//! The model is the SantaLucia (1998) *unified* nearest-neighbour parameter set — the one
//! primer3 and Biopython's `Tm_NN` default to — rather than the Wallace rule
//! (`2·(A+T) + 4·(G+C)`) or a GC-percentage formula. Those two are closed-form and wrong in a
//! way that matters: they read a sequence as a bag of bases, so `GCGCGC` and `GGGCCC` get the
//! same answer despite stacking very differently, and the error is several degrees.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, StringArray};

/// Nearest-neighbour ΔH (kcal/mol) and ΔS (cal/mol·K) for each 5'→3' dinucleotide, from
/// SantaLucia, *PNAS* 95:1460 (1998), Table 1.
///
/// The table lists ten pairs; the other six dinucleotides are the reverse complements of those
/// and carry the same values, which is why (for example) `TT` reads the same as `AA`. Written
/// out for all sixteen rather than derived, so a lookup is one index and cannot depend on a
/// complement routine being right.
const NN: [(&[u8; 2], f64, f64); 16] = [
    (b"AA", -7.9, -22.2),
    (b"TT", -7.9, -22.2),
    (b"AT", -7.2, -20.4),
    (b"TA", -7.2, -21.3),
    (b"CA", -8.5, -22.7),
    (b"TG", -8.5, -22.7),
    (b"GT", -8.4, -22.4),
    (b"AC", -8.4, -22.4),
    (b"CT", -7.8, -21.0),
    (b"AG", -7.8, -21.0),
    (b"GA", -8.2, -22.2),
    (b"TC", -8.2, -22.2),
    (b"CG", -10.6, -27.2),
    (b"GC", -9.8, -24.4),
    (b"GG", -8.0, -19.9),
    (b"CC", -8.0, -19.9),
];

/// Helix-initiation penalty at a terminal G·C pair: ΔH, ΔS.
const INIT_GC: (f64, f64) = (0.1, -2.8);
/// Helix-initiation penalty at a terminal A·T pair: ΔH, ΔS.
const INIT_AT: (f64, f64) = (2.3, 4.1);

/// The gas constant in cal/(mol·K), matching the units the NN table is published in.
const R: f64 = 1.987;

/// Total strand concentration, 500 nM — a typical PCR primer concentration and the value the
/// reported temperature is only meaningful relative to. Stated as a constant because `Tm` is
/// not a property of a sequence alone: halving the concentration moves it by several degrees.
const STRAND_CONC: f64 = 0.5e-6;

/// Monovalent salt concentration, 50 mM Na⁺ — standard PCR buffer, and the condition the
/// entropy correction below is applied for.
const NA_CONC: f64 = 0.05;

/// Look up one dinucleotide's contribution.
#[inline]
fn nn(pair: &[u8]) -> Option<(f64, f64)> {
    NN.iter()
        .find(|(p, _, _)| p.as_slice() == pair)
        .map(|&(_, h, s)| (h, s))
}

/// `melting_temp()` → the duplex melting temperature in °C (→ Float64).
///
/// Null for a sequence shorter than two bases (a single base stacks against nothing, so the
/// model has no terms) or one containing any character outside `ACGT`/`acgt`. An ambiguity
/// code has no defined stacking energy, and substituting one would report a specific
/// temperature the data does not support — so the row is null rather than approximate, which
/// is the same choice `translate` makes when it emits `X`.
pub(super) fn melting_temp(s: &StringArray) -> ArrayRef {
    let values: Vec<Option<f64>> = (0..s.len())
        .map(|i| {
            if s.is_null(i) {
                return None;
            }
            tm_of(s.value(i))
        })
        .collect();
    Arc::new(Float64Array::from(values))
}

fn tm_of(seq: &str) -> Option<f64> {
    let upper: Vec<u8> = seq.bytes().map(|b| b.to_ascii_uppercase()).collect();
    if upper.len() < 2 || !upper.iter().all(|b| matches!(b, b'A' | b'C' | b'G' | b'T')) {
        return None;
    }
    // Stacking terms: one per adjacent pair.
    let (mut dh, mut ds) = (0.0, 0.0);
    for pair in upper.windows(2) {
        let (h, s) = nn(pair)?;
        dh += h;
        ds += s;
    }
    // Initiation at each end, which depends on whether that end is a G·C or an A·T pair.
    for &end in [upper[0], upper[upper.len() - 1]].iter() {
        let (h, s) = if end == b'G' || end == b'C' {
            INIT_GC
        } else {
            INIT_AT
        };
        dh += h;
        ds += s;
    }
    // Salt correction on the entropy, SantaLucia (1998) eq. 7: each phosphate binds counter-
    // ions, so a longer duplex is stabilized more by the same salt concentration.
    let ds = ds + 0.368 * (upper.len() as f64 - 1.0) * NA_CONC.ln();
    // For non-self-complementary strands the effective concentration is C_T/4.
    let tm = (dh * 1000.0) / (ds + R * (STRAND_CONC / 4.0).ln()) - 273.15;
    tm.is_finite().then_some(tm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::seq::test_col;

    fn floats(a: &ArrayRef) -> Vec<Option<f64>> {
        let f = a.as_any().downcast_ref::<Float64Array>().unwrap();
        (0..f.len())
            .map(|i| (!f.is_null(i)).then(|| f.value(i)))
            .collect()
    }

    #[test]
    fn a_typical_primer_melts_in_the_range_a_primer_is_designed_for() {
        // A 20-mer at ~55% GC. Primer design targets 55-65 °C, and a model that landed far
        // outside that would be unusable whatever its internal consistency.
        let out = floats(&melting_temp(&test_col(&[Some("GTAAAACGACGGCCAGTGAA")])));
        let tm = out[0].unwrap();
        assert!((50.0..70.0).contains(&tm), "Tm was {tm}");
    }

    #[test]
    fn gc_rich_melts_hotter_than_at_rich_at_the_same_length() {
        let out = floats(&melting_temp(&test_col(&[
            Some("GCGCGCGCGCGCGCGC"),
            Some("ATATATATATATATAT"),
        ])));
        assert!(out[0].unwrap() > out[1].unwrap() + 20.0, "{out:?}");
    }

    #[test]
    fn stacking_order_matters_which_is_why_a_gc_percentage_formula_will_not_do() {
        // Same length, same base composition, different arrangement. Wallace and GC% give
        // these two identical answers; the nearest-neighbour model does not.
        let out = floats(&melting_temp(&test_col(&[
            Some("GCGCGCGCGCGC"),
            Some("GGGGGGCCCCCC"),
        ])));
        assert!(
            (out[0].unwrap() - out[1].unwrap()).abs() > 1.0,
            "the two arrangements should differ: {out:?}"
        );
    }

    #[test]
    fn longer_melts_hotter() {
        let out = floats(&melting_temp(&test_col(&[
            Some("ACGTACGT"),
            Some("ACGTACGTACGTACGT"),
            Some("ACGTACGTACGTACGTACGTACGT"),
        ])));
        assert!(out[0].unwrap() < out[1].unwrap());
        assert!(out[1].unwrap() < out[2].unwrap());
    }

    #[test]
    fn case_does_not_change_the_answer() {
        let out = floats(&melting_temp(&test_col(&[
            Some("acgtacgtacgtacgt"),
            Some("ACGTACGTACGTACGT"),
        ])));
        assert_eq!(out[0], out[1]);
    }

    #[test]
    fn an_ambiguous_or_too_short_sequence_is_null_rather_than_approximate() {
        let out = floats(&melting_temp(&test_col(&[
            Some("ACGTN"),
            Some("A"),
            Some(""),
            None,
        ])));
        assert_eq!(out, vec![None, None, None, None]);
    }

    #[test]
    fn the_reverse_complement_melts_the_same() {
        // A duplex has one melting temperature, so reading it from either strand must agree.
        let out = floats(&melting_temp(&test_col(&[
            Some("GTAAAACGACGGCCAGTGAA"),
            Some("TTCACTGGCCGTCGTTTTAC"),
        ])));
        assert!((out[0].unwrap() - out[1].unwrap()).abs() < 1e-9, "{out:?}");
    }
}
