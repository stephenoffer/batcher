//! IUPAC-degenerate motif search — finding a binding site, a restriction site, or a primer.
//!
//! A motif is written in the IUPAC degenerate alphabet: `GGWWTT` matches `GGAATT`, `GGATTT`,
//! `GGTATT`, and `GGTTTT`. That is not a substring search and it is not a regular expression
//! either — spelling it as one (`GG[AT][AT]TT`) means building and caching a regex per pattern
//! and, worse, gets the *subject* side wrong: a reference genome contains ambiguity codes too,
//! and `N` in the sequence must match every pattern base, which a character-class regex over
//! the literal text does not do.
//!
//! Matching is therefore defined on **sets of bases**: position `i` matches when the pattern
//! code's set and the sequence code's set intersect. That single rule makes the operation
//! symmetric in ambiguity and is what distinguishes this from a text search.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Array, Int64Builder, ListBuilder, StringArray};

use crate::eval::seq::bad_arg;
use crate::{ExprError, SeqFunc};

/// The set of concrete bases each IUPAC code stands for, as a 4-bit mask over A, C, G, T.
///
/// `U` shares `T`'s bit so an RNA motif matches a DNA sequence and the reverse; that is the
/// behaviour a mixed transcriptome/genome pipeline needs, and it costs nothing here.
const fn code_mask(b: u8) -> u8 {
    const A: u8 = 1;
    const C: u8 = 2;
    const G: u8 = 4;
    const T: u8 = 8;
    match b {
        b'A' | b'a' => A,
        b'C' | b'c' => C,
        b'G' | b'g' => G,
        b'T' | b't' | b'U' | b'u' => T,
        b'R' | b'r' => A | G,
        b'Y' | b'y' => C | T,
        b'S' | b's' => G | C,
        b'W' | b'w' => A | T,
        b'K' | b'k' => G | T,
        b'M' | b'm' => A | C,
        b'B' | b'b' => C | G | T,
        b'D' | b'd' => A | G | T,
        b'H' | b'h' => A | C | T,
        b'V' | b'v' => A | C | G,
        b'N' | b'n' => A | C | G | T,
        // Anything else stands for no base at all, so it matches nothing — including itself.
        // A gap character in a sequence is not a base, and a motif containing one is asking
        // for something that cannot occur.
        _ => 0,
    }
}

/// The 256-entry lookup the scan indexes, built once at compile time.
const MASKS: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = code_mask(i as u8);
        i += 1;
    }
    table
};

/// Compile a motif to its per-position masks, rejecting an empty or unmatchable one.
fn compile(func: SeqFunc, pattern: Option<&str>) -> Result<Vec<u8>, ExprError> {
    let pattern = pattern.ok_or_else(|| bad_arg(func, "a motif is required"))?;
    if pattern.is_empty() {
        return Err(bad_arg(func, "the motif must not be empty"));
    }
    let masks: Vec<u8> = pattern.bytes().map(|b| MASKS[b as usize]).collect();
    // A zero mask can never intersect anything, so a motif containing one matches nowhere. That
    // is almost always a typo (an `X`, a stray space, a `-` copied out of an alignment), and
    // reporting zero matches for every row would be a silent wrong answer.
    if let Some(pos) = masks.iter().position(|&m| m == 0) {
        let bad = pattern.as_bytes()[pos] as char;
        return Err(bad_arg(
            func,
            format!("{bad:?} at position {pos} is not an IUPAC nucleotide code"),
        ));
    }
    Ok(masks)
}

/// Whether the motif matches the sequence starting at `start`.
#[inline]
fn matches_at(seq: &[u8], start: usize, motif: &[u8]) -> bool {
    seq[start..start + motif.len()]
        .iter()
        .zip(motif)
        .all(|(&b, &m)| MASKS[b as usize] & m != 0)
}

/// `find_motif(pattern)` → `List<Int64>` of the **1-based** start positions of every match.
///
/// Matches may **overlap**: `AA` occurs three times in `AAAA`. That is the biologically
/// meaningful count — tandem repeats and overlapping binding sites are real — and it is what
/// distinguishes this from a `replace`-and-measure spelling, which counts only non-overlapping
/// occurrences.
///
/// Positions are 1-based to match every genome browser, GFF file, and VCF record a caller will
/// compare them against. A sequence with no match yields the empty list, not null.
pub(super) fn find_motif(s: &StringArray, pattern: Option<&str>) -> Result<ArrayRef, ExprError> {
    let motif = compile(SeqFunc::FindMotif, pattern)?;
    let mut builder = ListBuilder::with_capacity(Int64Builder::new(), s.len());
    for i in 0..s.len() {
        if s.is_null(i) {
            builder.append_null();
            continue;
        }
        let seq = s.value(i).as_bytes();
        if seq.len() >= motif.len() {
            for start in 0..=(seq.len() - motif.len()) {
                if matches_at(seq, start, &motif) {
                    builder.values().append_value(start as i64 + 1);
                }
            }
        }
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()))
}

/// `count_motif(pattern)` → how many (possibly overlapping) matches the sequence contains
/// (→ Int64).
///
/// The reduction of [`find_motif`], computed without building the position list — which is the
/// difference between one integer and a list allocation per row on a genome-scale scan.
pub(super) fn count_motif(s: &StringArray, pattern: Option<&str>) -> Result<ArrayRef, ExprError> {
    let motif = compile(SeqFunc::CountMotif, pattern)?;
    let values: Vec<Option<i64>> = (0..s.len())
        .map(|i| {
            if s.is_null(i) {
                return None;
            }
            let seq = s.value(i).as_bytes();
            if seq.len() < motif.len() {
                return Some(0);
            }
            Some(
                (0..=(seq.len() - motif.len()))
                    .filter(|&start| matches_at(seq, start, &motif))
                    .count() as i64,
            )
        })
        .collect();
    Ok(Arc::new(Int64Array::from(values)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::seq::test_col;
    use arrow::array::ListArray;

    fn positions(a: &ArrayRef) -> Vec<Option<Vec<i64>>> {
        let l = a.as_any().downcast_ref::<ListArray>().unwrap();
        (0..l.len())
            .map(|i| {
                (!l.is_null(i)).then(|| {
                    let v = l.value(i);
                    let q = v.as_any().downcast_ref::<Int64Array>().unwrap();
                    (0..q.len()).map(|j| q.value(j)).collect()
                })
            })
            .collect()
    }

    fn counts(a: &ArrayRef) -> Vec<Option<i64>> {
        let v = a.as_any().downcast_ref::<Int64Array>().unwrap();
        (0..v.len())
            .map(|i| (!v.is_null(i)).then(|| v.value(i)))
            .collect()
    }

    #[test]
    fn positions_are_one_based_like_every_genome_coordinate() {
        let out = find_motif(&test_col(&[Some("GGAATTCC")]), Some("GAATTC")).unwrap();
        assert_eq!(positions(&out), vec![Some(vec![2])]);
    }

    #[test]
    fn matches_overlap() {
        // Three occurrences of AA in AAAA, not two — tandem repeats are real.
        let c = test_col(&[Some("AAAA")]);
        assert_eq!(
            positions(&find_motif(&c, Some("AA")).unwrap()),
            vec![Some(vec![1, 2, 3])]
        );
        assert_eq!(counts(&count_motif(&c, Some("AA")).unwrap()), vec![Some(3)]);
    }

    #[test]
    fn a_degenerate_motif_matches_every_base_it_stands_for() {
        // W is A-or-T, so GGWWTT matches all four spellings.
        let c = test_col(&[
            Some("GGAATT"),
            Some("GGATTT"),
            Some("GGTATT"),
            Some("GGTTTT"),
            Some("GGACTT"),
        ]);
        assert_eq!(
            counts(&count_motif(&c, Some("GGWWTT")).unwrap()),
            vec![Some(1), Some(1), Some(1), Some(1), Some(0)]
        );
    }

    #[test]
    fn ambiguity_in_the_sequence_matches_too() {
        // The asymmetry a regex over the literal text gets wrong: an N in the reference is
        // consistent with any pattern base, so it matches.
        let c = test_col(&[Some("GGNATT"), Some("NNNNNN")]);
        assert_eq!(
            counts(&count_motif(&c, Some("GGAATT")).unwrap()),
            vec![Some(1), Some(1)]
        );
    }

    #[test]
    fn rna_and_dna_motifs_are_interchangeable() {
        let dna = test_col(&[Some("ACGTACGT")]);
        let rna = test_col(&[Some("ACGUACGU")]);
        for pattern in ["ACGT", "ACGU"] {
            assert_eq!(
                counts(&count_motif(&dna, Some(pattern)).unwrap()),
                vec![Some(2)],
                "{pattern} against DNA"
            );
            assert_eq!(
                counts(&count_motif(&rna, Some(pattern)).unwrap()),
                vec![Some(2)],
                "{pattern} against RNA"
            );
        }
    }

    #[test]
    fn case_is_folded_on_both_sides() {
        let c = test_col(&[Some("ggaattcc")]);
        assert_eq!(
            counts(&count_motif(&c, Some("GAATTC")).unwrap()),
            vec![Some(1)]
        );
    }

    #[test]
    fn a_motif_longer_than_the_sequence_finds_nothing_rather_than_panicking() {
        let c = test_col(&[Some("AC"), Some(""), None]);
        assert_eq!(
            counts(&count_motif(&c, Some("ACGTACGT")).unwrap()),
            vec![Some(0), Some(0), None]
        );
        assert_eq!(
            positions(&find_motif(&c, Some("ACGTACGT")).unwrap()),
            vec![Some(vec![]), Some(vec![]), None]
        );
    }

    #[test]
    fn an_unmatchable_motif_is_an_error_not_a_column_of_zeroes() {
        for bad in ["ACX T", "AC-GT", ""] {
            let err = count_motif(&test_col(&[Some("ACGT")]), Some(bad)).unwrap_err();
            assert!(
                err.to_string().contains("motif") || err.to_string().contains("IUPAC"),
                "{bad:?}: {err}"
            );
        }
        let err = count_motif(&test_col(&[Some("ACGT")]), None).unwrap_err();
        assert!(err.to_string().contains("motif is required"), "{err}");
    }

    #[test]
    fn find_and_count_always_agree() {
        let c = test_col(&[Some("GGAATTCCGGAATTCC"), Some("ACGT"), Some(""), None]);
        let found = positions(&find_motif(&c, Some("RRWWYY")).unwrap());
        let counted = counts(&count_motif(&c, Some("RRWWYY")).unwrap());
        for (f, n) in found.iter().zip(&counted) {
            assert_eq!(f.as_ref().map(|v| v.len() as i64), *n);
        }
    }
}
