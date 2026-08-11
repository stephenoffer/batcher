//! K-mers, canonical k-mers, and minimizers — the sketching primitives every sequence
//! comparison is built on.
//!
//! These emit `List<Utf8>` rather than packed integers on purpose: a list column composes with
//! the vocabulary the engine already has. `explode` turns k-mers into rows for a group-by
//! count, `list.n_unique()` is a cardinality estimate, `array_intersect` between two rows'
//! minimizer lists is a shared-substring count, and `minhash` over the same list is a
//! containment estimate. A packed `u64` encoding would be faster per k-mer and would compose
//! with nothing.
//!
//! All three fold to upper case, unlike the transforms in `nucleotide.rs`. That is required
//! rather than stylistic: soft-masked lowercase marks repeats, and a k-mer table that treated
//! `acgt` and `ACGT` as different strings would split every repeat-adjacent count in two —
//! and `canonical` would compare a lowercase k-mer against its uppercase reverse complement
//! and pick the wrong representative.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, ListBuilder, StringArray, StringBuilder};

use crate::eval::seq::bad_arg;
use crate::{ExprError, SeqFunc};

/// Whether a k-mer is emitted as read or folded with its reverse complement.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Canonical {
    No,
    Yes,
}

/// The largest `k` accepted. Beyond this a k-mer list is longer than the sequence it came
/// from in bytes and the operation is almost certainly a mistake (a `k` that was meant to be a
/// window, a column of the wrong thing). 256 covers every real use: assembly uses 21-127,
/// alignment seeds 15-31, and taxonomic classification 31-35.
const MAX_K: i64 = 256;

/// Read and validate the `k` argument shared by all three functions.
fn checked_k(func: SeqFunc, k: Option<i64>) -> Result<usize, ExprError> {
    let k = k.ok_or_else(|| bad_arg(func, "k is required"))?;
    if !(1..=MAX_K).contains(&k) {
        return Err(bad_arg(func, format!("k must be in 1..={MAX_K}, got {k}")));
    }
    Ok(k as usize)
}

/// Complement of every ASCII byte for the *canonical* comparison, as a lookup table.
///
/// A table rather than a `match`, for the reason the measurement showed: the `match` spelling
/// ran `canonical_kmers` at 6.5 MB/s against `reverse_complement`'s 991, and the only
/// structural difference between them was that the transform indexed a 256-byte table while
/// this branched per base. Five arms is a branch chain the predictor cannot help with when the
/// input is a random sequence, which is exactly what a genome is.
///
/// Only the four unambiguous bases complement; anything else — an `N`, an ambiguity code —
/// maps to `N`. A canonical k-mer must be comparable to its own reverse complement, and a code
/// that complemented to itself would make two different windows collide.
const CANON_COMPLEMENT: [u8; 256] = {
    let mut table = [b'N'; 256];
    table[b'A' as usize] = b'T';
    table[b'T' as usize] = b'A';
    table[b'C' as usize] = b'G';
    table[b'G' as usize] = b'C';
    table
};

/// The upper-cased reverse complement of an upper-case window, in place into `out`.
#[inline]
fn revcomp_into(window: &[u8], out: &mut Vec<u8>) {
    out.clear();
    out.extend(window.iter().rev().map(|&b| CANON_COMPLEMENT[b as usize]));
}

/// `kmers(k)` / `canonical_kmers(k)` → `List<Utf8>` of every length-`k` window, step 1.
///
/// A sequence shorter than `k` yields the **empty list**, not null: it genuinely contains no
/// k-mers, which is a different fact from "this row had no sequence". Downstream `explode`
/// then drops the row, which is what a k-mer count wants.
pub(super) fn kmers(
    s: &StringArray,
    k: Option<i64>,
    canonical: Canonical,
) -> Result<ArrayRef, ExprError> {
    let func = if canonical == Canonical::Yes {
        SeqFunc::CanonicalKmers
    } else {
        SeqFunc::Kmers
    };
    let k = checked_k(func, k)?;
    let mut builder = ListBuilder::with_capacity(
        StringBuilder::with_capacity(s.len(), s.value_data().len()),
        s.len(),
    );
    let mut upper: Vec<u8> = Vec::new();
    let mut rc: Vec<u8> = Vec::new();
    for i in 0..s.len() {
        if s.is_null(i) {
            builder.append_null();
            continue;
        }
        upper.clear();
        upper.extend(s.value(i).bytes().map(|b| b.to_ascii_uppercase()));
        for window in upper.windows(k) {
            let emitted = if canonical == Canonical::Yes {
                revcomp_into(window, &mut rc);
                // Lexicographic min of the window and its reverse complement — the standard
                // canonical representative (Jellyfish, KMC, minimap2 all use it), so a k-mer
                // table built here is comparable with one built by those tools.
                if rc.as_slice() < window {
                    rc.as_slice()
                } else {
                    window
                }
            } else {
                window
            };
            // Every byte came from `to_ascii_uppercase` of an input byte, or from the ACGTN
            // complement table, so a valid-UTF-8 input stays valid. A window that cut a
            // multi-byte character in half is not a sequence and is dropped.
            if let Ok(text) = std::str::from_utf8(emitted) {
                builder.values().append_value(text);
            }
        }
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()))
}

/// `minimizers(k, window)` → `List<Utf8>`: the distinct minimizers of the sequence.
///
/// A *minimizer* is the lexicographically smallest canonical k-mer in each window of `window`
/// consecutive k-mers. Because adjacent windows overlap, the same k-mer is usually selected
/// many times in a row; consecutive repeats are collapsed, which is what makes the output a
/// sketch — roughly `2/(window+1)` of the k-mers — rather than a re-encoding of the sequence.
///
/// This is the primitive behind seed-and-extend alignment (minimap2) and the reason two long
/// reads can be compared without comparing every k-mer: two sequences that share a substring
/// of length `window + k - 1` are guaranteed to share a minimizer, so a cheap list
/// intersection cannot miss a real overlap.
pub(super) fn minimizers(
    s: &StringArray,
    k: Option<i64>,
    window: Option<i64>,
) -> Result<ArrayRef, ExprError> {
    let func = SeqFunc::Minimizers;
    let k = checked_k(func, k)?;
    let w = window.ok_or_else(|| bad_arg(func, "window is required"))?;
    if !(1..=MAX_K).contains(&w) {
        return Err(bad_arg(
            func,
            format!("window must be in 1..={MAX_K}, got {w}"),
        ));
    }
    let w = w as usize;
    let mut builder = ListBuilder::with_capacity(StringBuilder::new(), s.len());
    let mut upper: Vec<u8> = Vec::new();
    let mut rc: Vec<u8> = Vec::new();
    let mut canon: Vec<u8> = Vec::new();
    for i in 0..s.len() {
        if s.is_null(i) {
            builder.append_null();
            continue;
        }
        upper.clear();
        upper.extend(s.value(i).bytes().map(|b| b.to_ascii_uppercase()));
        // Every canonical k-mer, packed end to end in one buffer instead of one `Vec` each.
        //
        // The obvious spelling — `canon: Vec<Vec<u8>>`, pushing `rc.clone()` or `win.to_vec()`
        // — costs a heap allocation *per k-mer*, and a 150 bp read at k=21 has 130 of them.
        // One flat buffer of `n * k` bytes removes them: the k-mers are fixed-width, so the
        // i-th starts at `i * k` and no offset table is needed.
        //
        // Worth stating plainly, because the measurement contradicted the guess that
        // prompted this: removing those allocations moved `minimizers` from 5.3 to 5.1 MB/s,
        // which is nothing. The allocator was *not* the bottleneck — `revcomp_into` was, and
        // it is fixed where it lives. This spelling is kept anyway because bounded allocation
        // per row is worth having on a genome-scale scan regardless of what it buys in
        // wall-clock here, but it is not a speed optimization and should not be cited as one.
        canon.clear();
        canon.reserve(upper.len().saturating_sub(k - 1) * k);
        for win in upper.windows(k) {
            revcomp_into(win, &mut rc);
            // Lexicographic min of the window and its reverse complement — the standard
            // canonical representative, so a sketch built here is comparable with minimap2's.
            canon.extend_from_slice(if rc.as_slice() < win { &rc } else { win });
        }
        let n_kmers = canon.len() / k;
        let gram = |idx: usize| &canon[idx * k..(idx + 1) * k];
        // The scan is a plain min over each window rather than a monotonic deque. The deque
        // is the asymptotically better structure, but `window` here is a small constant (10
        // is the minimap2 default) while the deque costs an allocation and a branch per
        // k-mer; at this size the straight-line scan wins and is obviously correct.
        //
        // A sequence with fewer than `window` k-mers still has one window — a short read is
        // not a read with no sketch, and returning nothing there would make every short
        // sequence silently unmatchable by a minimizer join.
        let windows = if n_kmers == 0 {
            0
        } else {
            n_kmers.saturating_sub(w - 1).max(1)
        };
        let mut last: Option<usize> = None;
        for start in 0..windows {
            let end = (start + w).min(n_kmers);
            let Some(best) = (start..end).min_by(|&a, &b| gram(a).cmp(gram(b))) else {
                continue;
            };
            // Collapse consecutive repeats: adjacent windows overlap by `w-1` k-mers, so the
            // same minimizer is normally selected many times running. Compared by *content*,
            // not by index — two different positions can hold the same k-mer, and emitting it
            // twice would inflate the sketch.
            if last.is_none_or(|prev| gram(prev) != gram(best)) {
                if let Ok(text) = std::str::from_utf8(gram(best)) {
                    builder.values().append_value(text);
                }
                last = Some(best);
            }
        }
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::seq::test_col;
    use arrow::array::ListArray;

    fn lists(a: &ArrayRef) -> Vec<Option<Vec<String>>> {
        let l = a.as_any().downcast_ref::<ListArray>().unwrap();
        (0..l.len())
            .map(|i| {
                (!l.is_null(i)).then(|| {
                    let v = l.value(i);
                    let s = v.as_any().downcast_ref::<StringArray>().unwrap();
                    (0..s.len()).map(|j| s.value(j).to_string()).collect()
                })
            })
            .collect()
    }

    #[test]
    fn kmers_slide_by_one_and_short_rows_yield_an_empty_list() {
        let c = test_col(&[Some("ACGTA"), Some("AC"), Some(""), None]);
        let out = kmers(&c, Some(3), Canonical::No).unwrap();
        assert_eq!(
            lists(&out),
            vec![
                Some(vec!["ACG".into(), "CGT".into(), "GTA".into()]),
                Some(vec![]),
                Some(vec![]),
                None,
            ]
        );
    }

    #[test]
    fn kmers_fold_case_so_a_masked_repeat_counts_with_its_unmasked_copy() {
        let c = test_col(&[Some("acg"), Some("ACG")]);
        let out = kmers(&c, Some(3), Canonical::No).unwrap();
        assert_eq!(lists(&out)[0], lists(&out)[1]);
    }

    #[test]
    fn a_canonical_kmer_equals_its_reverse_complements_canonical_kmer() {
        // This is the whole point of canonicalization: a read and the same read sequenced
        // from the other strand must produce the same k-mer table.
        let fwd = kmers(&test_col(&[Some("ACGTT")]), Some(3), Canonical::Yes).unwrap();
        let rev = kmers(&test_col(&[Some("AACGT")]), Some(3), Canonical::Yes).unwrap();
        let (mut f, mut r) = (
            lists(&fwd)[0].clone().unwrap(),
            lists(&rev)[0].clone().unwrap(),
        );
        f.sort();
        r.sort();
        assert_eq!(f, r);
    }

    #[test]
    fn canonical_picks_the_lexicographically_smaller_strand() {
        // TTT's reverse complement is AAA, which sorts first, so TTT canonicalizes to AAA.
        let out = kmers(
            &test_col(&[Some("TTT"), Some("AAA")]),
            Some(3),
            Canonical::Yes,
        )
        .unwrap();
        assert_eq!(lists(&out)[0], Some(vec!["AAA".to_string()]));
        assert_eq!(lists(&out)[1], Some(vec!["AAA".to_string()]));
    }

    #[test]
    fn an_ambiguous_base_canonicalizes_through_n_rather_than_colliding() {
        // `N` complements to `N`, so a window containing one is still comparable, and two
        // different windows cannot become the same k-mer.
        let out = kmers(&test_col(&[Some("ANG")]), Some(3), Canonical::Yes).unwrap();
        assert_eq!(lists(&out)[0], Some(vec!["ANG".to_string()]));
    }

    #[test]
    fn minimizers_collapse_consecutive_repeats() {
        // Every window of this poly-A run selects the same k-mer, so the sketch is one entry
        // rather than one per window.
        let out = minimizers(&test_col(&[Some("AAAAAAAA")]), Some(3), Some(4)).unwrap();
        assert_eq!(lists(&out)[0], Some(vec!["AAA".to_string()]));
    }

    #[test]
    fn minimizers_are_a_subset_of_the_canonical_kmers() {
        let seq = "ACGTTGCAAGGCTTAACG";
        let mins = minimizers(&test_col(&[Some(seq)]), Some(4), Some(5)).unwrap();
        let all = kmers(&test_col(&[Some(seq)]), Some(4), Canonical::Yes).unwrap();
        let all = lists(&all)[0].clone().unwrap();
        for m in lists(&mins)[0].clone().unwrap() {
            assert!(all.contains(&m), "{m} is not one of the canonical k-mers");
        }
    }

    #[test]
    fn two_sequences_sharing_a_long_substring_share_a_minimizer() {
        // The guarantee minimizers exist for: an overlap of `w + k - 1` bases cannot be
        // missed by a sketch comparison.
        let (k, w) = (4usize, 5usize);
        let shared = "ACGTTGCAA"; // k + w - 1 = 8 bases; this is longer.
        let a = minimizers(
            &test_col(&[Some(&format!("TTTTT{shared}GGGGG"))]),
            Some(k as i64),
            Some(w as i64),
        )
        .unwrap();
        let b = minimizers(
            &test_col(&[Some(&format!("CCCCC{shared}AAAAA"))]),
            Some(k as i64),
            Some(w as i64),
        )
        .unwrap();
        let (a, b) = (lists(&a)[0].clone().unwrap(), lists(&b)[0].clone().unwrap());
        assert!(
            a.iter().any(|m| b.contains(m)),
            "no shared minimizer between {a:?} and {b:?}"
        );
    }

    #[test]
    fn a_sequence_shorter_than_the_window_still_yields_its_minimizer() {
        // `saturating_sub` here is the difference between a sketch and a panic.
        let out = minimizers(&test_col(&[Some("ACGT")]), Some(3), Some(10)).unwrap();
        assert_eq!(lists(&out)[0].as_ref().map(|v| v.len()), Some(1));
    }

    #[test]
    fn k_is_validated_rather_than_silently_clamped() {
        for bad in [0i64, -1, 1000] {
            let err = kmers(&test_col(&[Some("ACGT")]), Some(bad), Canonical::No).unwrap_err();
            assert!(err.to_string().contains("k must be in"), "{err}");
        }
        let err = kmers(&test_col(&[Some("ACGT")]), None, Canonical::No).unwrap_err();
        assert!(err.to_string().contains("k is required"), "{err}");
    }
}
