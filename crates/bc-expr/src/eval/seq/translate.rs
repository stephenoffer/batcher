//! Codon translation — DNA/RNA to protein, NCBI genetic code table 1 (the standard code).
//!
//! Translation is the operation that most obviously cannot live on a `.str` surface: it reads
//! three bases at a time, and expressing that with substring arithmetic means a per-row loop
//! in the control plane. Here it is one pass with a 64-entry lookup, so translating a
//! transcriptome stays in the data plane.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, StringArray, StringBuilder};

use crate::eval::seq::bad_arg;
use crate::{ExprError, SeqFunc};

/// The standard genetic code (NCBI translation table 1), indexed by a base-4 codon number.
///
/// The index is `base4(b0) * 16 + base4(b1) * 4 + base4(b2)` with `T/U=0, C=1, A=2, G=3` —
/// the conventional TCAG ordering, which is what makes this table transcribable directly from
/// the NCBI listing and checkable against it by eye. `*` marks a stop codon.
const CODONS: &[u8; 64] = b"FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG";

/// Map a base to its TCAG index, or `None` for anything else (including every ambiguity code).
///
/// An ambiguous base makes the codon ambiguous, and this returns `None` rather than guessing:
/// a `GTN` codon really is valine whatever the third base is, but resolving that needs the
/// whole degenerate table, and quietly picking one of the four would put a specific residue
/// where the data supports none. `X` is the standard marker for exactly this, and that is what
/// the caller gets.
#[inline]
fn base4(b: u8) -> Option<usize> {
    Some(match b.to_ascii_uppercase() {
        b'T' | b'U' => 0,
        b'C' => 1,
        b'A' => 2,
        b'G' => 3,
        _ => return None,
    })
}

/// `translate(frame, to_stop)` → the amino-acid sequence (→ Utf8).
///
/// Reading starts at `frame` (0, 1, or 2) and proceeds in non-overlapping triplets. A trailing
/// partial codon is dropped, matching Biopython and EMBOSS: two leftover bases encode nothing,
/// and padding them with an invented base would fabricate a residue.
///
/// A codon containing any non-`ACGTU` byte translates to `X` (unknown residue), the IUPAC
/// marker for exactly that situation. A stop codon is `*`; with `to_stop` the sequence ends
/// at the first one, excluding the stop itself, which is what "the protein this ORF encodes"
/// means.
pub(super) fn translate(
    s: &StringArray,
    frame: Option<i64>,
    to_stop: bool,
) -> Result<ArrayRef, ExprError> {
    let frame = frame.unwrap_or(0);
    if !(0..=2).contains(&frame) {
        return Err(bad_arg(
            SeqFunc::Translate,
            format!("frame must be 0, 1, or 2, got {frame}"),
        ));
    }
    let frame = frame as usize;
    let mut out = StringBuilder::with_capacity(s.len(), s.value_data().len() / 3 + s.len());
    let mut protein: Vec<u8> = Vec::new();
    for i in 0..s.len() {
        if s.is_null(i) {
            out.append_null();
            continue;
        }
        let bytes = s.value(i).as_bytes();
        protein.clear();
        if bytes.len() > frame {
            protein.reserve((bytes.len() - frame) / 3);
            for codon in bytes[frame..].chunks_exact(3) {
                let residue = match (base4(codon[0]), base4(codon[1]), base4(codon[2])) {
                    (Some(a), Some(b), Some(c)) => CODONS[a * 16 + b * 4 + c],
                    _ => b'X',
                };
                if to_stop && residue == b'*' {
                    break;
                }
                protein.push(residue);
            }
        }
        // Every byte pushed is one of the 20 residues, `*`, or `X` — all ASCII — so this is
        // valid UTF-8 by construction.
        out.append_value(std::str::from_utf8(&protein).unwrap_or(""));
    }
    Ok(Arc::new(out.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::seq::test_col;

    fn strs(a: &ArrayRef) -> Vec<Option<String>> {
        let s = a.as_any().downcast_ref::<StringArray>().unwrap();
        (0..s.len())
            .map(|i| (!s.is_null(i)).then(|| s.value(i).to_string()))
            .collect()
    }

    #[test]
    fn the_table_is_the_standard_genetic_code() {
        // Spot-checks that would each fail on a differently-ordered table: the start codon,
        // the three stops, and one residue from each end of the index range.
        let c = test_col(&[
            Some("ATG"),
            Some("TAA"),
            Some("TAG"),
            Some("TGA"),
            Some("TTT"),
            Some("GGG"),
            Some("TGG"),
        ]);
        assert_eq!(
            strs(&translate(&c, None, false).unwrap()),
            ["M", "*", "*", "*", "F", "G", "W"]
                .iter()
                .map(|s| Some(s.to_string()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn rna_translates_the_same_as_dna() {
        let dna = test_col(&[Some("ATGGCC")]);
        let rna = test_col(&[Some("AUGGCC")]);
        assert_eq!(
            strs(&translate(&dna, None, false).unwrap()),
            strs(&translate(&rna, None, false).unwrap())
        );
    }

    #[test]
    fn a_trailing_partial_codon_is_dropped_not_padded() {
        let c = test_col(&[Some("ATGGC"), Some("AT"), Some(""), None]);
        assert_eq!(
            strs(&translate(&c, None, false).unwrap()),
            vec![Some("M".into()), Some("".into()), Some("".into()), None]
        );
    }

    #[test]
    fn an_ambiguous_codon_is_x_rather_than_a_guess() {
        let c = test_col(&[Some("ATGNNNGCC")]);
        assert_eq!(
            strs(&translate(&c, None, false).unwrap()),
            vec![Some("MXA".into())]
        );
    }

    #[test]
    fn to_stop_truncates_before_the_stop_codon() {
        let c = test_col(&[Some("ATGGCCTAAATG")]);
        assert_eq!(
            strs(&translate(&c, None, true).unwrap()),
            vec![Some("MA".into())]
        );
        assert_eq!(
            strs(&translate(&c, None, false).unwrap()),
            vec![Some("MA*M".into())]
        );
    }

    #[test]
    fn a_shifted_frame_reads_a_different_protein() {
        let c = test_col(&[Some("AATGGCC")]);
        assert_eq!(
            strs(&translate(&c, Some(1), false).unwrap()),
            vec![Some("MA".into())]
        );
        // A frame past the end of a short sequence yields the empty protein, not a panic.
        let short = test_col(&[Some("A")]);
        assert_eq!(
            strs(&translate(&short, Some(2), false).unwrap()),
            vec![Some("".into())]
        );
    }

    #[test]
    fn an_out_of_range_frame_is_a_plan_error_not_a_wrong_answer() {
        let err = translate(&test_col(&[Some("ATG")]), Some(3), false).unwrap_err();
        assert!(
            err.to_string().contains("frame must be 0, 1, or 2"),
            "{err}"
        );
    }

    #[test]
    fn lowercase_translates_like_uppercase() {
        let c = test_col(&[Some("atgGCC")]);
        assert_eq!(
            strs(&translate(&c, None, false).unwrap()),
            vec![Some("MA".into())]
        );
    }
}
