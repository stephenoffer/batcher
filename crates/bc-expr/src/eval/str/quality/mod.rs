//! Per-document text-quality measures — the LLM pretraining-corpus filters.
//!
//! These are the heuristics a web-scraped corpus is filtered with before a model is trained
//! on it: they remove navigation boilerplate, SEO keyword spam, truncated listing pages, and
//! machine-generated repetition — the categories that survive deduplication because each
//! document is individually unique.
//!
//! **These are per-row, and that is the whole point.** `plan/functions/metrics/text/` already
//! measures the same properties across a corpus, as aggregates: `char_repetition_rate` tells
//! you how repetitive your dataset is. That answers "should I worry", not "which documents do
//! I drop", and a filter needs the second.
//!
//! Split on the seam the family actually has: `gopher` holds the published rules from Gopher
//! (Rae et al. 2021, appendix A.1.1), C4 (Raffel et al. 2020), and RefinedWeb, whose
//! thresholds are citable; `entropy` holds the character-distribution measure that is *not*
//! one of them and carries no published threshold. Keeping that distinction in the module
//! layout is what stops the second from being cited as though it were the first.
//!
//! Every ratio is in `[0, 1]` and returns null for a document with nothing to measure — no
//! words, no lines, no n-grams. The empty string is the case that matters: it is what a
//! failed extraction produces, and it must not silently pass a `< 0.1` filter.

mod builders;
mod entropy;
mod gopher;

pub(super) use entropy::char_entropy;
pub(super) use gopher::{
    alpha_word_ratio, bullet_line_ratio, duplicate_line_ratio, duplicate_ngram_ratio,
    duplicate_paragraph_ratio, ellipsis_line_ratio, mean_word_length, stopword_count, symbol_ratio,
    top_ngram_ratio, word_count,
};

#[cfg(test)]
mod tests {
    use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, StringArray};

    use super::*;

    fn col(vals: &[Option<&str>]) -> StringArray {
        StringArray::from(vals.to_vec())
    }

    fn floats(a: &ArrayRef) -> Vec<Option<f64>> {
        let f = a.as_any().downcast_ref::<Float64Array>().unwrap();
        (0..f.len())
            .map(|i| (!f.is_null(i)).then(|| f.value(i)))
            .collect()
    }

    fn ints(a: &ArrayRef) -> Vec<Option<i64>> {
        let v = a.as_any().downcast_ref::<Int64Array>().unwrap();
        (0..v.len())
            .map(|i| (!v.is_null(i)).then(|| v.value(i)))
            .collect()
    }

    #[test]
    fn word_count_is_what_it_says() {
        let c = col(&[Some("one two  three"), Some("a\n\nb\nc"), Some(""), None]);
        assert_eq!(ints(&word_count(&c)), vec![Some(3), Some(3), Some(0), None]);
    }

    #[test]
    fn mean_word_length_is_null_for_a_document_with_no_words() {
        // Not 0.0: an empty extraction must not pass a `>= 3` filter by scoring below it, and
        // it must not pass a `<= 10` one either. Null fails both, which is correct.
        let c = col(&[Some("aa bbbb"), Some("   "), Some(""), None]);
        assert_eq!(
            floats(&mean_word_length(&c)),
            vec![Some(3.0), None, None, None]
        );
    }

    #[test]
    fn symbol_ratio_counts_both_spellings_of_an_ellipsis() {
        // Which one a page uses is a function of its encoder, not its content.
        let dots = col(&[Some("a... b... c... d")]);
        let glyph = col(&[Some("a\u{2026} b\u{2026} c\u{2026} d")]);
        assert_eq!(floats(&symbol_ratio(&dots)), floats(&symbol_ratio(&glyph)));
        assert_eq!(floats(&symbol_ratio(&dots)), vec![Some(0.75)]);
    }

    #[test]
    fn alpha_word_ratio_finds_a_table_that_lost_its_structure() {
        let prose = col(&[Some("the quick brown fox")]);
        let table = col(&[Some("1.99 2.50 3.75 four")]);
        assert_eq!(floats(&alpha_word_ratio(&prose)), vec![Some(1.0)]);
        assert_eq!(floats(&alpha_word_ratio(&table)), vec![Some(0.25)]);
    }

    #[test]
    fn stopword_count_counts_distinct_words_not_occurrences() {
        // A page repeating "the" is not better evidence of English than one using two words.
        let spam = col(&[Some("the the the the the")]);
        let prose = col(&[Some("the cat sat with a hat")]);
        assert_eq!(ints(&stopword_count(&spam)), vec![Some(1)]);
        assert_eq!(ints(&stopword_count(&prose)), vec![Some(2)]);
    }

    #[test]
    fn stopword_count_ignores_surrounding_punctuation() {
        assert_eq!(
            ints(&stopword_count(&col(&[Some("(The) and,")]))),
            vec![Some(2)]
        );
    }

    #[test]
    fn bullet_and_ellipsis_line_ratios_find_a_navigation_menu_and_a_listing_page() {
        let menu = col(&[Some("- home\n- about\n- contact\nreal prose here")]);
        assert_eq!(floats(&bullet_line_ratio(&menu)), vec![Some(0.75)]);
        let listing = col(&[Some("A story about\u{2026}\nAnother one...\nplain")]);
        assert!((floats(&ellipsis_line_ratio(&listing))[0].unwrap() - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn duplicate_ratios_weigh_by_characters_not_by_unit_count() {
        // One repeated 40-character footer is a different document from fifty repeated
        // one-word lines, and a unit count cannot tell them apart.
        let long_dup = col(&[Some(
            "unique line\nxxxxxxxxxxxxxxxxxxxx\nxxxxxxxxxxxxxxxxxxxx",
        )]);
        let short_dup = col(&[Some(
            "a very long and entirely unique opening line here\nz\nz",
        )]);
        let a = floats(&duplicate_line_ratio(&long_dup))[0].unwrap();
        let b = floats(&duplicate_line_ratio(&short_dup))[0].unwrap();
        assert!(a > 0.3, "{a}");
        assert!(b < 0.1, "{b}");
    }

    #[test]
    fn duplicate_paragraph_ratio_splits_on_blank_lines() {
        let c = col(&[Some("para one\n\npara two\n\npara one")]);
        let r = floats(&duplicate_paragraph_ratio(&c))[0].unwrap();
        assert!(r > 0.3, "{r}");
    }

    #[test]
    fn top_ngram_ratio_finds_a_keyword_stuffed_page() {
        let spam = col(&[Some(
            "cheap flights cheap flights cheap flights cheap flights",
        )]);
        let prose = col(&[Some("the quick brown fox jumps over the lazy dog today")]);
        let s = floats(&top_ngram_ratio(&spam, 2))[0].unwrap();
        let p = floats(&top_ngram_ratio(&prose, 2))[0].unwrap();
        assert!(s > 0.8, "{s}");
        assert!(p < 0.3, "{p}");
    }

    #[test]
    fn duplicate_ngram_ratio_sums_over_every_repeated_gram() {
        // The definition, asserted exactly rather than by comparison: two distinct 3-grams
        // each recur once, so the measure is *both* their character lengths over the
        // document. Comparing two documents instead would confound the ratio with their
        // lengths, and the point here is that the second block is counted at all — which is
        // the whole difference from `top_ngram_ratio`, which only ever sees one gram.
        let text = "aa bb cc xx aa bb cc yy dd ee ff zz dd ee ff";
        let total = text.chars().count() as f64;
        let expected = ("aa bb cc".len() + "dd ee ff".len()) as f64 / total;
        let got = floats(&duplicate_ngram_ratio(&col(&[Some(text)]), 3))[0].unwrap();
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );

        // Not compared against `top_ngram_ratio`: the two are not ordered relative to each
        // other in general, and on this very document they coincide — `top` counts its one
        // gram's two occurrences (16 characters) while `duplicate` counts two grams once each
        // (also 16). Asserting an inequality between them would be asserting an accident.
    }

    #[test]
    fn every_ratio_stays_within_zero_and_one() {
        let texts = [
            "aaaa aaaa aaaa aaaa",
            "one two three four five six seven",
            "- a\n- b",
            "x",
        ];
        for t in texts {
            let c = col(&[Some(t)]);
            for r in [
                floats(&symbol_ratio(&c))[0],
                floats(&alpha_word_ratio(&c))[0],
                floats(&bullet_line_ratio(&c))[0],
                floats(&ellipsis_line_ratio(&c))[0],
                floats(&duplicate_line_ratio(&c))[0],
                floats(&top_ngram_ratio(&c, 2))[0],
                floats(&duplicate_ngram_ratio(&c, 3))[0],
            ]
            .into_iter()
            .flatten()
            {
                assert!((0.0..=1.0).contains(&r), "{t:?} produced {r}");
            }
        }
    }

    #[test]
    fn char_entropy_separates_prose_from_a_blob_and_from_a_repeat() {
        let repeat = floats(&char_entropy(&col(&[Some("aaaaaaaaaaaaaaaa")])))[0].unwrap();
        let prose = floats(&char_entropy(&col(&[Some(
            "The quick brown fox jumps over the lazy dog and then rests.",
        )])))[0]
            .unwrap();
        let blob = floats(&char_entropy(&col(&[Some(
            "a1B2c3D4e5F6g7H8i9J0kLmNoPqRsTuVwXyZ+/=abcdefghijklmnop",
        )])))[0]
            .unwrap();
        assert_eq!(repeat, 0.0, "one repeated character carries no information");
        // Positive zero specifically: negating a zero sum yields -0.0, and `assert_eq!`
        // against 0.0 passes for it. Only a sign check catches an entropy of "-0".
        assert!(repeat.is_sign_positive(), "entropy must not render as -0");
        assert!(prose > 3.5 && prose < 5.0, "prose was {prose}");
        assert!(
            blob > prose,
            "a near-uniform blob ({blob}) should exceed prose ({prose})"
        );
    }

    #[test]
    fn every_measure_answers_null_for_a_null_row() {
        let c = col(&[None]);
        assert_eq!(floats(&mean_word_length(&c)), vec![None]);
        assert_eq!(floats(&symbol_ratio(&c)), vec![None]);
        assert_eq!(floats(&char_entropy(&c)), vec![None]);
        assert_eq!(ints(&word_count(&c)), vec![None]);
        assert_eq!(ints(&stopword_count(&c)), vec![None]);
    }
}
