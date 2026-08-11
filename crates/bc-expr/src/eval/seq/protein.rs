//! Molecular weight, hydropathy, and isoelectric point — the physicochemical properties a
//! protein or oligo column is filtered and bucketed on.
//!
//! These are the three numbers a proteomics or protein-design pipeline computes for every
//! sequence it sees, and each is a fold over the residues. Computing them here rather than in
//! a per-row Python call is the difference between a scan and a row loop.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, StringArray};

use crate::eval::seq::bad_arg;
use crate::{ExprError, SeqFunc};

/// Average *residue* masses in daltons, indexed by the 20 standard amino-acid letters.
///
/// A residue mass is the free amino acid minus the water lost forming a peptide bond, which is
/// why the chain mass below adds one water back rather than subtracting `n-1` of them. The two
/// bookkeepings agree exactly: `Σ residue + H₂O == Σ free − (n−1)·H₂O`, the second being the
/// arrangement `Bio.SeqUtils.molecular_weight` uses. Stated because the tables look different
/// at a glance — glycine is 57.05 here and 75.07 there — and picking the wrong one costs a
/// water per residue.
const AA_MASS: [(u8, f64); 20] = [
    (b'A', 71.0788),
    (b'C', 103.1388),
    (b'D', 115.0886),
    (b'E', 129.1155),
    (b'F', 147.1766),
    (b'G', 57.0519),
    (b'H', 137.1411),
    (b'I', 113.1594),
    (b'K', 128.1741),
    (b'L', 113.1594),
    (b'M', 131.1986),
    (b'N', 114.1038),
    (b'P', 97.1167),
    (b'Q', 128.1307),
    (b'R', 156.1875),
    (b'S', 87.0782),
    (b'T', 101.1051),
    (b'V', 99.1326),
    (b'W', 186.2132),
    (b'Y', 163.1760),
];

/// Mass of a water molecule, added once to close a peptide chain and subtracted once per
/// phosphodiester bond in a nucleic acid.
const WATER: f64 = 18.0153;

/// Average masses of the DNA nucleotide monophosphates, matching
/// `Bio.Data.IUPACData.unambiguous_dna_weights`.
const DNA_MASS: [(u8, f64); 4] = [
    (b'A', 331.2218),
    (b'C', 307.1971),
    (b'G', 347.2212),
    (b'T', 322.2085),
];

/// Average masses of the RNA nucleotide monophosphates.
const RNA_MASS: [(u8, f64); 4] = [
    (b'A', 347.2212),
    (b'C', 323.1965),
    (b'G', 363.2206),
    (b'U', 324.1813),
];

/// Kyte-Doolittle hydropathy indices — the scale GRAVY is the average of.
const HYDROPATHY: [(u8, f64); 20] = [
    (b'A', 1.8),
    (b'R', -4.5),
    (b'N', -3.5),
    (b'D', -3.5),
    (b'C', 2.5),
    (b'Q', -3.5),
    (b'E', -3.5),
    (b'G', -0.4),
    (b'H', -3.2),
    (b'I', 4.5),
    (b'L', 3.8),
    (b'K', -3.9),
    (b'M', 1.9),
    (b'F', 2.8),
    (b'P', -1.6),
    (b'S', -0.8),
    (b'T', -0.7),
    (b'W', -0.9),
    (b'Y', -1.3),
    (b'V', 4.2),
];

/// Look a residue up in one of the tables above. Linear over at most 20 entries, which beats a
/// hash map at this size and needs no allocation or lazy initialization.
#[inline]
fn lookup(table: &[(u8, f64)], b: u8) -> Option<f64> {
    let b = b.to_ascii_uppercase();
    table.iter().find(|(c, _)| *c == b).map(|&(_, v)| v)
}

/// `molecular_weight(alphabet)` → the average molecular weight in daltons (→ Float64).
///
/// For `protein`, the sum of residue masses plus one water for the chain's free termini. For
/// `dna`/`rna`, the sum of nucleotide monophosphate masses minus one water per phosphodiester
/// bond — a *single* strand, which is what a sequence column holds; double it for a duplex.
///
/// Null for a sequence containing any character outside the alphabet's unambiguous set,
/// including the `*` stop marker and every ambiguity code. A weight is a physical quantity and
/// an unknown residue has none, so an approximate answer would be worse than no answer. The
/// empty sequence yields 0.0 for the nucleic acids and the mass of one water for a protein,
/// each of which is what the formula says.
pub(super) fn molecular_weight(
    s: &StringArray,
    alphabet: Option<&str>,
) -> Result<ArrayRef, ExprError> {
    let name =
        alphabet.ok_or_else(|| bad_arg(SeqFunc::MolecularWeight, "an alphabet is required"))?;
    // Rejecting the degenerate alphabets here rather than nulling every row is the more useful
    // failure: a caller who asked for the weight of a `dna_iupac` column has a question the
    // model cannot answer, and finding that out at plan time beats a column of nulls.
    let (table, is_protein) = match name {
        "dna" => (DNA_MASS.as_slice(), false),
        "rna" => (RNA_MASS.as_slice(), false),
        "protein" => (
            // Restricted to the 20 standard residues; the table has no entry for `*`, so a
            // translated sequence still carrying its stop marker is null rather than silently
            // weighed as if the stop were not there.
            AA_MASS.as_slice(),
            true,
        ),
        _ => {
            return Err(bad_arg(
                SeqFunc::MolecularWeight,
                format!(
                    "alphabet must be one of [\"dna\", \"protein\", \"rna\"] \
                     (a degenerate alphabet has no defined mass), got {name:?}"
                ),
            ))
        }
    };
    let values: Vec<Option<f64>> = (0..s.len())
        .map(|i| {
            if s.is_null(i) {
                return None;
            }
            let seq = s.value(i);
            let mut total = 0.0;
            let mut n = 0usize;
            for b in seq.bytes() {
                total += lookup(table, b)?;
                n += 1;
            }
            Some(if is_protein {
                total + WATER
            } else {
                total - WATER * n.saturating_sub(1) as f64
            })
        })
        .collect();
    Ok(Arc::new(Float64Array::from(values)))
}

/// `gravy()` → the grand average of hydropathy (→ Float64).
///
/// The mean Kyte-Doolittle index over the residues. Positive is hydrophobic (a membrane
/// protein typically sits above 0.5), negative is hydrophilic (a soluble globular protein is
/// usually below 0). It is the standard first-pass discriminator between the two, and the
/// reason it is one number rather than a window scan is that the window scan — a hydropathy
/// *plot* — is `list.transform` over this same table, which the engine can already express.
///
/// Residues outside the standard twenty are skipped rather than nulling the row: a single `X`
/// in a long predicted protein should not erase its hydropathy. A sequence with no scorable
/// residue at all yields null.
pub(super) fn gravy(s: &StringArray) -> ArrayRef {
    let values: Vec<Option<f64>> = (0..s.len())
        .map(|i| {
            if s.is_null(i) {
                return None;
            }
            let (sum, n) = s.value(i).bytes().fold((0.0, 0usize), |(sum, n), b| {
                match lookup(&HYDROPATHY, b) {
                    Some(v) => (sum + v, n + 1),
                    None => (sum, n),
                }
            });
            (n > 0).then(|| sum / n as f64)
        })
        .collect();
    Arc::new(Float64Array::from(values))
}

/// Ionizable groups: the pKa of each side chain that carries a positive charge when protonated,
/// plus the N-terminus. Bjellqvist values, as used by ExPASy and Biopython.
const POSITIVE_PKA: [(u8, f64); 3] = [(b'K', 10.0), (b'R', 12.0), (b'H', 5.98)];
/// pKa of each group that carries a negative charge when deprotonated, plus the C-terminus.
const NEGATIVE_PKA: [(u8, f64); 4] = [(b'D', 4.05), (b'E', 4.45), (b'C', 9.0), (b'Y', 10.0)];
/// The free amino terminus.
const N_TERM_PKA: f64 = 7.5;
/// The free carboxyl terminus.
const C_TERM_PKA: f64 = 3.55;

/// Net charge of the sequence at a given pH, from the Henderson-Hasselbalch equation.
fn charge_at(counts: &[(u8, usize)], ph: f64, has_termini: bool) -> f64 {
    let mut charge = 0.0;
    for &(res, pka) in POSITIVE_PKA.iter() {
        let n = counts
            .iter()
            .find(|(c, _)| *c == res)
            .map_or(0, |&(_, n)| n);
        charge += n as f64 / (1.0 + 10f64.powf(ph - pka));
    }
    for &(res, pka) in NEGATIVE_PKA.iter() {
        let n = counts
            .iter()
            .find(|(c, _)| *c == res)
            .map_or(0, |&(_, n)| n);
        charge -= n as f64 / (1.0 + 10f64.powf(pka - ph));
    }
    if has_termini {
        charge += 1.0 / (1.0 + 10f64.powf(ph - N_TERM_PKA));
        charge -= 1.0 / (1.0 + 10f64.powf(C_TERM_PKA - ph));
    }
    charge
}

/// `isoelectric_point()` → the pH at which the peptide carries no net charge (→ Float64).
///
/// The number that decides how a protein behaves in isoelectric focusing and ion-exchange
/// chromatography, and the property a purification protocol is designed around.
///
/// Solved by bisection on the net-charge curve, which is strictly decreasing in pH, so the
/// root is unique and the bisection converges from any bracket. Searched over pH 0-14 to
/// 0.001, which is finer than the model's own accuracy. A sequence with no ionizable group and
/// no termini (the empty sequence) yields null.
pub(super) fn isoelectric_point(s: &StringArray) -> ArrayRef {
    let values: Vec<Option<f64>> = (0..s.len())
        .map(|i| {
            if s.is_null(i) {
                return None;
            }
            let seq = s.value(i);
            if seq.is_empty() {
                return None;
            }
            // Count once per row; the bisection then reads the counts ~28 times rather than
            // re-scanning the sequence.
            let mut counts: Vec<(u8, usize)> = Vec::new();
            for b in seq.bytes() {
                let b = b.to_ascii_uppercase();
                match counts.iter_mut().find(|(c, _)| *c == b) {
                    Some((_, n)) => *n += 1,
                    None => counts.push((b, 1)),
                }
            }
            let (mut lo, mut hi) = (0.0f64, 14.0f64);
            while hi - lo > 0.001 {
                let mid = (lo + hi) / 2.0;
                if charge_at(&counts, mid, true) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            Some((lo + hi) / 2.0)
        })
        .collect();
    Arc::new(Float64Array::from(values))
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
    fn a_single_residue_weighs_the_free_amino_acid() {
        // Glycine's residue mass plus one water is free glycine, 75.07 Da.
        let out = floats(&molecular_weight(&test_col(&[Some("G")]), Some("protein")).unwrap());
        assert!((out[0].unwrap() - 75.0672).abs() < 0.01, "{out:?}");
    }

    #[test]
    fn a_peptide_loses_one_water_per_bond() {
        let one = floats(&molecular_weight(&test_col(&[Some("G")]), Some("protein")).unwrap())[0]
            .unwrap();
        let two = floats(&molecular_weight(&test_col(&[Some("GG")]), Some("protein")).unwrap())[0]
            .unwrap();
        assert!((two - (2.0 * one - WATER)).abs() < 1e-6, "{one} {two}");
    }

    #[test]
    fn dna_weight_matches_the_monophosphate_table() {
        // A single A is one AMP; ACGT is the four minus three waters.
        let out =
            floats(&molecular_weight(&test_col(&[Some("A"), Some("ACGT")]), Some("dna")).unwrap());
        assert!((out[0].unwrap() - 331.2218).abs() < 1e-6, "{out:?}");
        let expect = 331.2218 + 307.1971 + 347.2212 + 322.2085 - 3.0 * WATER;
        assert!((out[1].unwrap() - expect).abs() < 1e-6, "{out:?}");
    }

    #[test]
    fn rna_weighs_more_than_the_same_dna_because_of_the_extra_oxygen() {
        let dna =
            floats(&molecular_weight(&test_col(&[Some("ACG")]), Some("dna")).unwrap())[0].unwrap();
        let rna =
            floats(&molecular_weight(&test_col(&[Some("ACG")]), Some("rna")).unwrap())[0].unwrap();
        assert!(rna > dna, "{rna} should exceed {dna}");
    }

    #[test]
    fn an_unknown_residue_has_no_weight_rather_than_an_approximate_one() {
        let out = floats(
            &molecular_weight(
                &test_col(&[Some("GXG"), Some("MA*"), None]),
                Some("protein"),
            )
            .unwrap(),
        );
        assert_eq!(out, vec![None, None, None]);
    }

    #[test]
    fn a_degenerate_alphabet_is_refused_rather_than_nulling_every_row() {
        let err = molecular_weight(&test_col(&[Some("ACGT")]), Some("dna_iupac")).unwrap_err();
        assert!(err.to_string().contains("no defined mass"), "{err}");
    }

    #[test]
    fn gravy_separates_a_hydrophobic_stretch_from_a_charged_one() {
        let out = floats(&gravy(&test_col(&[Some("IIIVVVLLL"), Some("KKKRRRDDD")])));
        assert!(out[0].unwrap() > 3.0, "{out:?}");
        assert!(out[1].unwrap() < -3.0, "{out:?}");
    }

    #[test]
    fn gravy_skips_unknown_residues_instead_of_erasing_the_row() {
        let out = floats(&gravy(&test_col(&[
            Some("IXI"),
            Some("II"),
            Some("XXX"),
            Some(""),
        ])));
        assert_eq!(out[0], out[1]);
        assert_eq!(out[2], None);
        assert_eq!(out[3], None);
    }

    #[test]
    fn the_isoelectric_point_is_where_the_net_charge_crosses_zero() {
        let out = floats(&isoelectric_point(&test_col(&[Some("KKKK"), Some("DDDD")])));
        // A poly-lysine peptide is strongly basic and a poly-aspartate one strongly acidic.
        assert!(out[0].unwrap() > 9.0, "{out:?}");
        assert!(out[1].unwrap() < 4.5, "{out:?}");
    }

    #[test]
    fn the_charge_really_is_zero_at_the_reported_ph() {
        // The property the bisection is solving for, checked directly rather than against a
        // remembered number.
        for seq in ["KKKK", "DDDD", "ACDEFGHIKLMNPQRSTVWY"] {
            let pi = floats(&isoelectric_point(&test_col(&[Some(seq)])))[0].unwrap();
            let counts: Vec<(u8, usize)> = seq.bytes().fold(Vec::new(), |mut acc, b| {
                match acc.iter_mut().find(|(c, _)| *c == b) {
                    Some((_, n)) => *n += 1,
                    None => acc.push((b, 1)),
                }
                acc
            });
            let q = charge_at(&counts, pi, true);
            assert!(q.abs() < 0.01, "{seq}: charge {q} at pH {pi}");
        }
    }

    #[test]
    fn an_empty_or_null_sequence_has_no_isoelectric_point() {
        assert_eq!(
            floats(&isoelectric_point(&test_col(&[Some(""), None]))),
            vec![None, None]
        );
    }
}
