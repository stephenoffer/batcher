//! Nucleotide transforms and composition measures — the base of every genomics pipeline.
//!
//! The transforms (`complement`, `reverse_complement`, `transcribe`) are byte-table lookups
//! that preserve case, because lowercase is how a reference genome marks soft-masked repeats
//! and upper-casing would destroy the mask. The measures (`gc_content`, `gc_skew`,
//! `base_counts`, `max_homopolymer`, `is_valid`) fold case, because a repeat-masked contig is
//! not a contig of unknown bases.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray, StringBuilder,
    StructArray,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field};

use crate::eval::seq::bad_arg;
use crate::{ExprError, SeqFunc};

/// The IUPAC complement of every ASCII byte, case-preserving.
///
/// A full 256-entry table rather than a `match`: one indexed load per base with no branch,
/// which is what keeps a whole-chromosome reverse-complement memory-bandwidth-bound. Every
/// byte that is not a nucleotide code maps to itself, so a FASTA line containing a gap
/// character, an `*`, or a stray space comes back with those characters where they were
/// instead of silently becoming `N`.
///
/// The ambiguity codes complement as IUPAC defines them — `R` (A/G) ↔ `Y` (C/T), `K` (G/T) ↔
/// `M` (A/C), `B` (not-A) ↔ `V` (not-T), `D` (not-C) ↔ `H` (not-G) — while `S` (G/C), `W`
/// (A/T), and `N` are their own complements. Getting these wrong is the classic
/// reverse-complement bug: it is invisible on a test of pure ACGT and wrong on any real
/// variant call.
const COMPLEMENT: [u8; 256] = build_complement();

const fn build_complement() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = i as u8;
        i += 1;
    }
    // Pairs written once and applied in both cases and both directions, so the table cannot
    // be asymmetric — `complement(complement(x)) == x` holds by construction for every code
    // below.
    const PAIRS: [(u8, u8); 6] = [
        (b'A', b'T'),
        (b'C', b'G'),
        (b'R', b'Y'),
        (b'K', b'M'),
        (b'B', b'V'),
        (b'D', b'H'),
    ];
    let mut p = 0;
    while p < PAIRS.len() {
        let (a, b) = PAIRS[p];
        table[a as usize] = b;
        table[b as usize] = a;
        table[(a + 32) as usize] = b + 32;
        table[(b + 32) as usize] = a + 32;
        p += 1;
    }
    // `S` (G/C), `W` (A/T), `N`, and the gap character are self-complementary: the identity
    // fill above already covers them, so writing them out would only be a chance to get one
    // wrong.
    //
    // `U` is the one deliberately one-way entry. It complements to `A`, but `A` complements
    // back to `T`, because a mixed or unknown-alphabet column is far more often DNA and `T`
    // is the answer that keeps it so. Writing `U` as a symmetric pair is the subtle bug this
    // comment exists to prevent: it would silently redirect `A` to `U` and turn every DNA
    // complement into RNA. Use `transcribe` to move between the alphabets deliberately.
    table[b'U' as usize] = b'A';
    table[b'u' as usize] = b'a';
    table
}

/// Fold an ASCII byte to upper case. Non-alphabetic bytes pass through, so a gap character
/// stays a gap character rather than becoming whatever `b - 32` happens to be.
#[inline]
fn upper(b: u8) -> u8 {
    b.to_ascii_uppercase()
}

/// `complement()` / `reverse_complement()` — one table lookup per base, case preserved.
pub(super) fn complement(s: &StringArray, reverse: bool) -> ArrayRef {
    let mut out = StringBuilder::with_capacity(s.len(), s.value_data().len());
    let mut buf: Vec<u8> = Vec::new();
    for i in 0..s.len() {
        if s.is_null(i) {
            out.append_null();
            continue;
        }
        let bytes = s.value(i).as_bytes();
        buf.clear();
        buf.reserve(bytes.len());
        if reverse {
            buf.extend(bytes.iter().rev().map(|&b| COMPLEMENT[b as usize]));
        } else {
            buf.extend(bytes.iter().map(|&b| COMPLEMENT[b as usize]));
        }
        // The table maps every byte to a byte of the same ASCII class and leaves non-ASCII
        // bytes untouched, so a valid UTF-8 input stays valid UTF-8 — but a multi-byte
        // character reversed byte-wise would not be. `from_utf8` therefore decides rather
        // than an `unsafe` assumption, and a sequence that was never text yields null.
        match std::str::from_utf8(&buf) {
            Ok(text) => out.append_value(text),
            Err(_) => out.append_null(),
        }
    }
    Arc::new(out.finish())
}

/// `transcribe()` (DNA→RNA, T→U) and `back_transcribe()` (RNA→DNA, U→T), case preserved.
pub(super) fn transcribe(s: &StringArray, to_rna: bool) -> ArrayRef {
    let (from_upper, to_upper) = if to_rna { (b'T', b'U') } else { (b'U', b'T') };
    let (from_lower, to_lower) = (from_upper + 32, to_upper + 32);
    let mut out = StringBuilder::with_capacity(s.len(), s.value_data().len());
    let mut buf: Vec<u8> = Vec::new();
    for i in 0..s.len() {
        if s.is_null(i) {
            out.append_null();
            continue;
        }
        buf.clear();
        buf.extend(s.value(i).as_bytes().iter().map(|&b| {
            if b == from_upper {
                to_upper
            } else if b == from_lower {
                to_lower
            } else {
                b
            }
        }));
        match std::str::from_utf8(&buf) {
            Ok(text) => out.append_value(text),
            Err(_) => out.append_null(),
        }
    }
    Arc::new(out.finish())
}

/// The per-row tally every composition measure reads: how many of each of the five bases,
/// how many ambiguity codes, and how many bytes were none of those.
#[derive(Default, Clone, Copy)]
struct Tally {
    a: i64,
    c: i64,
    g: i64,
    t: i64,
    u: i64,
    n: i64,
    other: i64,
}

impl Tally {
    fn of(seq: &str) -> Self {
        let mut t = Tally::default();
        for &b in seq.as_bytes() {
            match upper(b) {
                b'A' => t.a += 1,
                b'C' => t.c += 1,
                b'G' => t.g += 1,
                b'T' => t.t += 1,
                b'U' => t.u += 1,
                b'N' => t.n += 1,
                _ => t.other += 1,
            }
        }
        t
    }

    /// Bases the composition measures are defined over: the four unambiguous ones plus `U`,
    /// so a DNA and an RNA column give the same answer. `N` and the ambiguity codes are
    /// excluded rather than counted as non-GC — counting them would make a run of `N`s look
    /// like an AT-rich region, which is the difference between "no data" and "a real signal".
    fn defined(self) -> i64 {
        self.a + self.c + self.g + self.t + self.u
    }
}

/// `gc_content()` → the (G+C) fraction of the *unambiguous* bases (→ Float64 in `[0, 1]`).
pub(super) fn gc_content(s: &StringArray) -> ArrayRef {
    let values: Vec<Option<f64>> = (0..s.len())
        .map(|i| {
            if s.is_null(i) {
                return None;
            }
            let t = Tally::of(s.value(i));
            let denom = t.defined();
            // A sequence of nothing but `N`s has no defined GC content, and neither does an
            // empty one. Answering 0.0 would put both at the AT-rich extreme of every
            // histogram and every filter built on one.
            (denom > 0).then(|| (t.g + t.c) as f64 / denom as f64)
        })
        .collect();
    Arc::new(Float64Array::from(values))
}

/// `gc_skew()` → `(G - C) / (G + C)` (→ Float64 in `[-1, 1]`).
///
/// The replication-origin signal: the leading and lagging strands mutate differently, so the
/// sign of this quantity flips at the origin and the terminus of a bacterial chromosome.
pub(super) fn gc_skew(s: &StringArray) -> ArrayRef {
    let values: Vec<Option<f64>> = (0..s.len())
        .map(|i| {
            if s.is_null(i) {
                return None;
            }
            let t = Tally::of(s.value(i));
            let gc = t.g + t.c;
            (gc > 0).then(|| (t.g - t.c) as f64 / gc as f64)
        })
        .collect();
    Arc::new(Float64Array::from(values))
}

/// `base_counts()` → struct `{a, c, g, t, u, n, other}`, all Int64, case-folded.
///
/// One pass yields every count, so asking for the `N` count costs nothing beyond asking for
/// the `A` count. Project the one you want with `.struct.field("n")`.
pub(super) fn base_counts(s: &StringArray) -> ArrayRef {
    let mut cols: [Vec<i64>; 7] = Default::default();
    let mut valid = Vec::with_capacity(s.len());
    for i in 0..s.len() {
        let t = if s.is_null(i) {
            valid.push(false);
            Tally::default()
        } else {
            valid.push(true);
            Tally::of(s.value(i))
        };
        for (col, v) in cols.iter_mut().zip([t.a, t.c, t.g, t.t, t.u, t.n, t.other]) {
            col.push(v);
        }
    }
    let names = ["a", "c", "g", "t", "u", "n", "other"];
    let fields: Vec<Arc<Field>> = names
        .iter()
        .map(|n| Arc::new(Field::new(*n, DataType::Int64, false)))
        .collect();
    let columns: Vec<ArrayRef> = cols
        .into_iter()
        .map(|c| Arc::new(Int64Array::from(c)) as ArrayRef)
        .collect();
    Arc::new(StructArray::new(
        fields.into(),
        columns,
        Some(NullBuffer::from(valid)),
    ))
}

/// `max_homopolymer()` → the length of the longest run of one base (→ Int64).
///
/// The nanopore and PacBio error signature: those chemistries miscount long single-base runs,
/// so a homopolymer length is what a variant filter thresholds on. Case-folded, so a
/// soft-masked run counts as one run rather than two.
pub(super) fn max_homopolymer(s: &StringArray) -> ArrayRef {
    let values: Vec<Option<i64>> = (0..s.len())
        .map(|i| {
            if s.is_null(i) {
                return None;
            }
            let bytes = s.value(i).as_bytes();
            let mut best = 0i64;
            let mut run = 0i64;
            let mut prev = 0u8;
            for &b in bytes {
                let b = upper(b);
                if b == prev {
                    run += 1;
                } else {
                    prev = b;
                    run = 1;
                }
                best = best.max(run);
            }
            Some(best)
        })
        .collect();
    Arc::new(Int64Array::from(values))
}

/// The alphabets `is_valid` and `molecular_weight` accept, as upper-case byte sets.
///
/// Named here rather than inline because both functions must agree on what "DNA" means: one
/// accepting `N` and the other not would let a column validate and then weigh as null.
pub(crate) fn alphabet_bytes(name: &str) -> Option<&'static [u8]> {
    Some(match name {
        "dna" => b"ACGT",
        "rna" => b"ACGU",
        // The full IUPAC degenerate sets, which is what a real variant or primer column
        // contains — a reference genome is full of `N`, and a degenerate primer is written
        // entirely in these codes.
        "dna_iupac" => b"ACGTRYSWKMBDHVN",
        "rna_iupac" => b"ACGURYSWKMBDHVN",
        // The 20 standard amino acids plus the stop marker `*` that `translate` emits, so a
        // translated column validates against its own producer.
        "protein" => b"ACDEFGHIKLMNPQRSTVWY*",
        _ => return None,
    })
}

/// Every alphabet name, for the error message that lists them.
pub(crate) const ALPHABETS: [&str; 5] = ["dna", "dna_iupac", "protein", "rna", "rna_iupac"];

/// `is_valid(alphabet)` → whether every character is in `alphabet` (→ Boolean).
///
/// An empty sequence is **valid**: it violates no membership rule, and reporting it invalid
/// would conflate "has a bad character" with "has no characters", which are different
/// problems with different fixes. Filter on length for the second.
pub(super) fn is_valid(s: &StringArray, alphabet: Option<&str>) -> Result<ArrayRef, ExprError> {
    let name = alphabet.ok_or_else(|| bad_arg(SeqFunc::IsValid, "an alphabet is required"))?;
    let set = alphabet_bytes(name).ok_or_else(|| {
        bad_arg(
            SeqFunc::IsValid,
            format!("alphabet must be one of {ALPHABETS:?}, got {name:?}"),
        )
    })?;
    let mut table = [false; 256];
    for &b in set {
        table[b as usize] = true;
        table[(b.to_ascii_lowercase()) as usize] = true;
    }
    let values: Vec<Option<bool>> = (0..s.len())
        .map(|i| (!s.is_null(i)).then(|| s.value(i).as_bytes().iter().all(|&b| table[b as usize])))
        .collect();
    Ok(Arc::new(BooleanArray::from(values)))
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

    fn floats(a: &ArrayRef) -> Vec<Option<f64>> {
        let f = a.as_any().downcast_ref::<Float64Array>().unwrap();
        (0..f.len())
            .map(|i| (!f.is_null(i)).then(|| f.value(i)))
            .collect()
    }

    #[test]
    fn reverse_complement_matches_the_textbook_answer() {
        let c = test_col(&[Some("ATGC"), Some("AAATTT"), Some(""), None]);
        assert_eq!(
            strs(&complement(&c, true)),
            vec![
                Some("GCAT".into()),
                Some("AAATTT".into()),
                Some("".into()),
                None
            ]
        );
    }

    #[test]
    fn complement_preserves_soft_masking() {
        // Lowercase marks a repeat-masked region; upper-casing it would destroy the mask.
        let c = test_col(&[Some("ATgcAT")]);
        assert_eq!(strs(&complement(&c, false)), vec![Some("TAcgTA".into())]);
    }

    #[test]
    fn ambiguity_codes_complement_as_iupac_defines_them() {
        // The classic reverse-complement bug is invisible on pure ACGT and wrong here.
        let c = test_col(&[Some("RYKMBDHVSWN")]);
        assert_eq!(
            strs(&complement(&c, false)),
            vec![Some("YRMKVHDBSWN".into())]
        );
    }

    #[test]
    fn complement_is_an_involution_on_dna() {
        let c = test_col(&[Some("ACGTRYKMBDHVSWNacgt")]);
        let once = complement(&c, false);
        let twice = complement(once.as_any().downcast_ref::<StringArray>().unwrap(), false);
        assert_eq!(strs(&twice), strs(&(Arc::new(c) as ArrayRef)));
    }

    #[test]
    fn transcribe_round_trips_and_keeps_case() {
        let c = test_col(&[Some("ATgc")]);
        let rna = transcribe(&c, true);
        assert_eq!(strs(&rna), vec![Some("AUgc".into())]);
        let back = transcribe(rna.as_any().downcast_ref::<StringArray>().unwrap(), false);
        assert_eq!(strs(&back), vec![Some("ATgc".into())]);
    }

    #[test]
    fn gc_content_excludes_ambiguous_bases_from_the_denominator() {
        // "GCNN" is 100% GC over the bases that are actually known, not 50%. Counting the
        // Ns would make a gap look like an AT-rich region.
        let c = test_col(&[Some("GCNN"), Some("ATAT"), Some("NNNN"), Some(""), None]);
        assert_eq!(
            floats(&gc_content(&c)),
            vec![Some(1.0), Some(0.0), None, None, None]
        );
    }

    #[test]
    fn gc_content_reads_rna_and_dna_alike() {
        let c = test_col(&[Some("GCAU"), Some("GCAT")]);
        assert_eq!(floats(&gc_content(&c)), vec![Some(0.5), Some(0.5)]);
    }

    #[test]
    fn gc_skew_is_null_without_any_gc() {
        let c = test_col(&[Some("GGGC"), Some("CCCG"), Some("ATAT")]);
        assert_eq!(floats(&gc_skew(&c)), vec![Some(0.5), Some(-0.5), None]);
    }

    #[test]
    fn base_counts_separates_the_five_bases_from_n_and_from_other() {
        let out = base_counts(&test_col(&[Some("AACGTUNR"), None]));
        let st = out.as_any().downcast_ref::<StructArray>().unwrap();
        let read = |name: &str| {
            st.column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0)
        };
        assert_eq!(
            [
                read("a"),
                read("c"),
                read("g"),
                read("t"),
                read("u"),
                read("n"),
                read("other")
            ],
            [2, 1, 1, 1, 1, 1, 1]
        );
        assert!(st.is_null(1));
    }

    #[test]
    fn max_homopolymer_folds_case_and_is_zero_for_empty() {
        let c = test_col(&[Some("AaaaTG"), Some(""), Some("ACGT"), None]);
        let out = max_homopolymer(&c);
        let v = out.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(
            (0..4)
                .map(|i| (!v.is_null(i)).then(|| v.value(i)))
                .collect::<Vec<_>>(),
            vec![Some(4), Some(0), Some(1), None]
        );
    }

    #[test]
    fn is_valid_accepts_the_empty_sequence_and_rejects_a_stray_code() {
        let c = test_col(&[Some("ACGT"), Some("ACGN"), Some(""), None]);
        let out = is_valid(&c, Some("dna")).unwrap();
        let b = out.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            (0..4)
                .map(|i| (!b.is_null(i)).then(|| b.value(i)))
                .collect::<Vec<_>>(),
            vec![Some(true), Some(false), Some(true), None]
        );
        // The IUPAC alphabet is what a reference genome actually validates against.
        let iupac = is_valid(&c, Some("dna_iupac")).unwrap();
        assert!(iupac
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .value(1));
    }

    #[test]
    fn is_valid_names_the_alphabets_when_given_a_bad_one() {
        let err = is_valid(&test_col(&[Some("A")]), Some("peptide")).unwrap_err();
        assert!(err.to_string().contains("dna_iupac"), "{err}");
    }
}
