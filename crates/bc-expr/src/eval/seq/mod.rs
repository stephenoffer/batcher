//! Biological-sequence evaluation for `Expr::Seq` — the `.seq` namespace.
//!
//! A genomics pipeline is a string pipeline with a different alphabet, and that is exactly
//! why it needs its own kernels rather than a composition of `.str` ones. Reverse-complement
//! is not `reverse` plus `translate`; it is one pass with a 256-byte lookup table. GC content
//! is not `len(replace(s,'A',''))/len(s)`; that spelling allocates two strings per row and
//! silently counts the `N`s the caller meant to exclude. Codon translation over a `.str`
//! surface is a per-row Python loop, which the control plane must never run.
//!
//! Everything here is a single pass over the row's bytes with no allocation beyond the
//! output, so a FASTQ scan stays memory-bandwidth-bound rather than allocator-bound.
//!
//! **The alphabet is ASCII by construction.** Nucleotide and amino-acid codes are single
//! ASCII bytes, so these kernels index byte tables directly instead of decoding UTF-8. A
//! sequence carrying non-ASCII bytes is not a sequence; those bytes fall through the tables
//! as "other" rather than erroring, so one malformed row cannot abort a scan.
//!
//! Case is **preserved** by the transforms (`complement`, `reverse_complement`, `transcribe`)
//! and **ignored** by the measures (`gc_content`, `base_counts`, motif search, validity).
//! That is the Biopython convention, and it matters: lowercase is how every reference genome
//! marks soft-masked repeats, so a transform that upper-cased would destroy the mask, while a
//! measure that respected it would report a repeat-rich contig as mostly-`other`.

use arrow::array::{ArrayRef, StringArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;

use crate::{ExprError, SeqFunc};

mod kmer;
mod motif;
mod nucleotide;
mod protein;
mod quality;
mod thermo;
mod translate;

/// The constant arguments a `.seq` op carries, gathered so the dispatch signature stays
/// one parameter wide instead of eight (the shape `Image`/`Audio` already use).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SeqArgs<'a> {
    /// `kmers`/`canonical_kmers`/`minimizers`: the k-mer length.
    pub k: Option<i64>,
    /// `minimizers`: how many consecutive k-mers each window spans.
    pub window: Option<i64>,
    /// `translate`: the reading frame, 0, 1, or 2.
    pub frame: Option<i64>,
    /// `phred_quality`/`mean_quality`/`expected_errors`: the FASTQ ASCII offset.
    pub offset: Option<i64>,
    /// `molecular_weight`/`is_valid`: which alphabet the column is written in.
    pub alphabet: Option<&'a str>,
    /// `find_motif`/`count_motif`: the IUPAC-degenerate pattern.
    pub pattern: Option<&'a str>,
    /// `translate`: stop at the first stop codon rather than running to the end.
    pub to_stop: bool,
}

/// Evaluate a sequence function over a Utf8 array, preserving nulls.
///
/// The input is coerced to `Utf8` first for the same reason `eval_str` does it: a FASTA or
/// FASTQ reader that had no UTF-8 logical annotation to write hands over a `Binary` column,
/// and a sequence column is textual by definition.
pub(crate) fn eval_seq(
    func: SeqFunc,
    arr: &ArrayRef,
    args: SeqArgs<'_>,
) -> Result<ArrayRef, ExprError> {
    // An all-null column arrives typed `Null` (an upstream filter that matched nothing, an
    // outer join with no partner). Every kernel here answers null for a null row, so widening
    // to an empty `Utf8` array of the same length gets the output type and null bits right
    // with no per-op special case — the same fix `media::widen_null_column` makes.
    let widened;
    let arr = if matches!(arr.data_type(), DataType::Null) {
        widened = arrow::array::new_null_array(&DataType::Utf8, arr.len());
        &widened
    } else {
        arr
    };
    let coerced;
    let arr = match arr.data_type() {
        DataType::Utf8 => arr,
        DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary | DataType::Utf8View => {
            coerced = cast(arr, &DataType::Utf8)?;
            &coerced
        }
        other => {
            return Err(ExprError::ExpectedType {
                func: format!("seq.{}", name_of(func)),
                want: "a Utf8 argument",
                got: other.to_string(),
            })
        }
    };
    let s = arr
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| ExprError::ExpectedType {
            func: format!("seq.{}", name_of(func)),
            want: "a Utf8 argument",
            got: arr.data_type().to_string(),
        })?;

    match func {
        SeqFunc::Complement => Ok(nucleotide::complement(s, false)),
        SeqFunc::ReverseComplement => Ok(nucleotide::complement(s, true)),
        SeqFunc::Transcribe => Ok(nucleotide::transcribe(s, true)),
        SeqFunc::BackTranscribe => Ok(nucleotide::transcribe(s, false)),
        SeqFunc::GcContent => Ok(nucleotide::gc_content(s)),
        SeqFunc::GcSkew => Ok(nucleotide::gc_skew(s)),
        SeqFunc::BaseCounts => Ok(nucleotide::base_counts(s)),
        SeqFunc::MaxHomopolymer => Ok(nucleotide::max_homopolymer(s)),
        SeqFunc::IsValid => nucleotide::is_valid(s, args.alphabet),
        SeqFunc::Translate => translate::translate(s, args.frame, args.to_stop),
        SeqFunc::Kmers => kmer::kmers(s, args.k, kmer::Canonical::No),
        SeqFunc::CanonicalKmers => kmer::kmers(s, args.k, kmer::Canonical::Yes),
        SeqFunc::Minimizers => kmer::minimizers(s, args.k, args.window),
        SeqFunc::MeltingTemp => Ok(thermo::melting_temp(s)),
        SeqFunc::MolecularWeight => protein::molecular_weight(s, args.alphabet),
        SeqFunc::Gravy => Ok(protein::gravy(s)),
        SeqFunc::IsoelectricPoint => Ok(protein::isoelectric_point(s)),
        SeqFunc::PhredQuality => Ok(quality::phred_quality(s, args.offset)),
        SeqFunc::MeanQuality => Ok(quality::mean_quality(s, args.offset)),
        SeqFunc::ExpectedErrors => Ok(quality::expected_errors(s, args.offset)),
        SeqFunc::FindMotif => motif::find_motif(s, args.pattern),
        SeqFunc::CountMotif => motif::count_motif(s, args.pattern),
    }
}

/// The wire name of a function, for error messages. Written out rather than derived from
/// `Debug` so the message names the function the *caller* wrote (`seq.gc_content`), not the
/// Rust variant they have never seen (`GcContent`).
fn name_of(func: SeqFunc) -> &'static str {
    match func {
        SeqFunc::Complement => "complement",
        SeqFunc::ReverseComplement => "reverse_complement",
        SeqFunc::Transcribe => "transcribe",
        SeqFunc::BackTranscribe => "back_transcribe",
        SeqFunc::GcContent => "gc_content",
        SeqFunc::GcSkew => "gc_skew",
        SeqFunc::BaseCounts => "base_counts",
        SeqFunc::MaxHomopolymer => "max_homopolymer",
        SeqFunc::IsValid => "is_valid",
        SeqFunc::Translate => "translate",
        SeqFunc::Kmers => "kmers",
        SeqFunc::CanonicalKmers => "canonical_kmers",
        SeqFunc::Minimizers => "minimizers",
        SeqFunc::MeltingTemp => "melting_temp",
        SeqFunc::MolecularWeight => "molecular_weight",
        SeqFunc::Gravy => "gravy",
        SeqFunc::IsoelectricPoint => "isoelectric_point",
        SeqFunc::PhredQuality => "phred_quality",
        SeqFunc::MeanQuality => "mean_quality",
        SeqFunc::ExpectedErrors => "expected_errors",
        SeqFunc::FindMotif => "find_motif",
        SeqFunc::CountMotif => "count_motif",
    }
}

/// An `InvalidArgument` naming the `.seq` function the caller actually wrote.
pub(crate) fn bad_arg(func: SeqFunc, reason: impl Into<String>) -> ExprError {
    ExprError::InvalidArgument {
        func: format!("seq.{}", name_of(func)),
        reason: reason.into(),
    }
}

/// Build a Utf8 test column, with `None` for a null row. Shared by every sibling module's
/// test block so they cannot drift on how a null row is spelled.
#[cfg(test)]
pub(crate) fn test_col(vals: &[Option<&str>]) -> StringArray {
    StringArray::from(vals.to_vec())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::Array;

    use super::*;

    #[test]
    fn null_typed_column_widens_instead_of_erroring() {
        let nulls = arrow::array::new_null_array(&DataType::Null, 3);
        let out = eval_seq(SeqFunc::GcContent, &nulls, SeqArgs::default()).unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out.null_count(), 3);
    }

    #[test]
    fn non_textual_input_is_rejected_by_name() {
        let ints: ArrayRef = Arc::new(arrow::array::Int64Array::from(vec![1i64, 2]));
        let err = eval_seq(SeqFunc::GcContent, &ints, SeqArgs::default()).unwrap_err();
        assert!(err.to_string().contains("seq.gc_content"), "{err}");
    }

    #[test]
    fn binary_input_is_coerced_like_str_functions_do() {
        let bin: ArrayRef = Arc::new(arrow::array::BinaryArray::from(vec![
            Some(b"GGCC".as_ref()),
            None,
        ]));
        let out = eval_seq(SeqFunc::GcContent, &bin, SeqArgs::default()).unwrap();
        let f = out
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        assert_eq!(f.value(0), 1.0);
        assert!(f.is_null(1));
    }
}
