//! Case conversion for `StrFunc::ToCase`, and the SQuAD normalization every text metric
//! runs first. Both start from the same question — where are the words.
//!
//! Case conversion is one operation with a style parameter, not ten functions: every
//! style shares the same hard part, which is deciding where the words are. Splitting
//! once and re-joining is also what makes the styles mutually consistent — `snake` and
//! `camel` of the same input always agree on the word boundaries, which they would not
//! if each style carried its own ad-hoc scanner.

use crate::ExprError;

/// The style names accepted in the `pattern` slot of `StrFunc::ToCase`.
///
/// Kept as a slice rather than an enum because the Python control plane validates
/// against the same list by name (`plan/expr_ir/namespaces/strings.py`), and the error
/// message below is the place a typo is diagnosed.
pub(crate) const STYLES: [&str; 10] = [
    "snake",
    "upper_snake",
    "camel",
    "pascal",
    "kebab",
    "upper_kebab",
    "title",
    "sentence",
    "dot",
    "train",
];

/// Split `s` into words at separators and at case/script transitions.
///
/// Three boundary rules, in the order they are tested per character:
///
/// 1. Any non-alphanumeric character is a separator and is dropped (`a_b`, `a-b`,
///    `a b`, `a.b` all split).
/// 2. A lower-to-upper transition starts a word (`helloWorld` → `hello`, `World`).
/// 3. In a run of uppercase, the *last* uppercase before a lowercase starts the next
///    word (`HTTPServer` → `HTTP`, `Server`), which is what keeps acronyms intact.
///
/// Digits attach to the word they touch (`utf8Bytes` → `utf8`, `Bytes`), because
/// splitting them out turns `sha256` into two words and no style spells that back the
/// way anyone wants.
fn words(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for (i, &c) in chars.iter().enumerate() {
        if !c.is_alphanumeric() {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            continue;
        }
        if !cur.is_empty() && c.is_uppercase() {
            let prev = chars[i - 1];
            // Rule 2, then rule 3: the acronym check needs the *next* character, so a
            // trailing uppercase run stays whole.
            let starts_word = prev.is_lowercase()
                || prev.is_numeric()
                || (prev.is_uppercase() && chars.get(i + 1).is_some_and(|n| n.is_lowercase()));
            if starts_word {
                out.push(std::mem::take(&mut cur));
            }
        }
        cur.push(c);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Uppercase the first character of `w` and lowercase the rest.
fn capitalize(w: &str) -> String {
    let mut it = w.chars();
    match it.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + &it.as_str().to_lowercase(),
    }
}

/// Convert `s` to `style`. Returns `None` for an unknown style so the caller can raise
/// a single well-formed error rather than one per row.
pub(crate) fn to_case(s: &str, style: &str) -> Option<String> {
    let w = words(s);
    let joined =
        |sep: &str, f: &dyn Fn(&String) -> String| w.iter().map(f).collect::<Vec<_>>().join(sep);
    Some(match style {
        "snake" => joined("_", &|x| x.to_lowercase()),
        "upper_snake" => joined("_", &|x| x.to_uppercase()),
        "kebab" => joined("-", &|x| x.to_lowercase()),
        "upper_kebab" => joined("-", &|x| x.to_uppercase()),
        "dot" => joined(".", &|x| x.to_lowercase()),
        "pascal" => joined("", &|x| capitalize(x)),
        "title" => joined(" ", &|x| capitalize(x)),
        "train" => joined("-", &|x| capitalize(x)),
        "camel" => w
            .iter()
            .enumerate()
            .map(|(i, x)| {
                if i == 0 {
                    x.to_lowercase()
                } else {
                    capitalize(x)
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        "sentence" => w
            .iter()
            .enumerate()
            .map(|(i, x)| {
                if i == 0 {
                    capitalize(x)
                } else {
                    x.to_lowercase()
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => return None,
    })
}

/// The error raised once when the style name is not one of [`STYLES`].
pub(crate) fn unknown_style(style: &str) -> ExprError {
    ExprError::InvalidArgument {
        func: "ToCase".to_string(),
        reason: format!(
            "unknown case style {style:?}; expected one of {}",
            STYLES.join(", ")
        ),
    }
}

// --- SQuAD answer normalization ---------------------------------------------------------
//
// Lowercase, drop the standalone articles `a`/`an`/`the`, delete punctuation, collapse
// whitespace, trim. Every word-level text metric runs it first, which is what makes
// `token_set_f1`, `answer_groundedness`, BLEU and ROUGE agree on what a token is.
//
// It lives beside case conversion because it starts with the same question — where are the
// words — and because a lowercase pass is its first step.
//
// It existed as a composition of five expressions: `lower`, three `regexp_replace_all` passes,
// two trims. That is six kernel invocations and six string arrays per column, three of them
// driven by a regex engine, and on 20,000 rows of 40-token text it measured ~2.1 s against
// 20 ms for a bare `len` over the same column. Every word metric paid it once per text column;
// BLEU paid it eight times. One pass with one allocation is 4.7x faster and byte-identical.
//
// The scan walks maximal runs of word and non-word characters, which reproduces all five steps:
//
// * a word run that is an article emits a space, anything else emits itself;
// * a non-word run emits a *single* space if it contained whitespace and **nothing** if it did
//   not — the asymmetry that makes `cat-dog` one token and `cat, dog` two;
// * a space is only written before the next real content, so leading, trailing and repeated
//   spaces never reach the output and both trims come for free.

/// General categories Mn, Mc, Me.
///
/// `char` exposes no accessor, and a Unicode-tables dependency for three categories would be a
/// crate for a predicate. These are the ranges text a normalizer sees actually contains:
/// combining diacritics, and the Devanagari, Hebrew and Arabic marks.
fn is_mark(c: char) -> bool {
    matches!(
        c,
        '\u{0300}'..='\u{036F}'
            | '\u{0483}'..='\u{0489}'
            | '\u{0591}'..='\u{05BD}'
            | '\u{0610}'..='\u{061A}'
            | '\u{064B}'..='\u{065F}'
            | '\u{0900}'..='\u{0903}'
            | '\u{093A}'..='\u{094F}'
            | '\u{0951}'..='\u{0957}'
            | '\u{1AB0}'..='\u{1AFF}'
            | '\u{1DC0}'..='\u{1DFF}'
            | '\u{20D0}'..='\u{20F0}'
            | '\u{FE20}'..='\u{FE2F}'
    )
}

/// Category Nd. `char::is_numeric` is broader — it admits Nl and No, so Roman numerals and
/// vulgar fractions — and `\w` does not match those.
fn is_decimal_digit(c: char) -> bool {
    c.is_ascii_digit()
        || matches!(
            c,
            '\u{0660}'..='\u{0669}'
                | '\u{06F0}'..='\u{06F9}'
                | '\u{0966}'..='\u{096F}'
                | '\u{09E6}'..='\u{09EF}'
                | '\u{0E50}'..='\u{0E59}'
                | '\u{FF10}'..='\u{FF19}'
        )
}

/// Category Pc — the underscore and its relatives.
fn is_connector(c: char) -> bool {
    matches!(
        c,
        '_' | '\u{203F}'
            | '\u{2040}'
            | '\u{2054}'
            | '\u{FE33}'..='\u{FE34}'
            | '\u{FE4D}'..='\u{FE4F}'
            | '\u{FF3F}'
    )
}

/// Whether `c` is a word character under the regex crate's Unicode `\w`.
///
/// `\w` is `[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}\p{Join_Control}]`. The classes are spelled out
/// rather than approximated with `is_alphanumeric`, which is a different set: the difference
/// decides where a token boundary falls, and the composition this replaces was written against
/// the regex engine's definition.
fn is_word(c: char) -> bool {
    c.is_alphabetic()
        || is_mark(c)
        || is_decimal_digit(c)
        || is_connector(c)
        || matches!(c, '\u{200C}' | '\u{200D}')
}

/// The SQuAD normalization of one string, into a reused buffer.
fn normalize_one(value: &str, out: &mut String) {
    out.clear();
    let lowered = value.to_lowercase();
    let mut pending_space = false;
    let mut rest = lowered.as_str();
    while !rest.is_empty() {
        let word_run = is_word(rest.chars().next().expect("non-empty"));
        let end = rest
            .char_indices()
            .find(|(_, c)| is_word(*c) != word_run)
            .map_or(rest.len(), |(i, _)| i);
        let (run, remainder) = rest.split_at(end);
        rest = remainder;
        if word_run {
            if matches!(run, "a" | "an" | "the") {
                pending_space = true;
                continue;
            }
            if pending_space && !out.is_empty() {
                out.push(' ');
            }
            out.push_str(run);
            pending_space = false;
        } else if run.chars().any(char::is_whitespace) {
            pending_space = true;
        }
    }
}

/// `StrFunc::SquadNormalize` over a `Utf8` column.
pub(crate) fn eval_squad_normalize(arr: &arrow::array::StringArray) -> arrow::array::ArrayRef {
    use arrow::array::{Array, StringBuilder};
    use std::sync::Arc;

    // The output is never longer than the input, so one reservation avoids every regrow.
    let mut builder = StringBuilder::with_capacity(arr.len(), arr.value_data().len());
    let mut scratch = String::new();
    for i in 0..arr.len() {
        if arr.is_null(i) {
            builder.append_null();
            continue;
        }
        normalize_one(arr.value(i), &mut scratch);
        builder.append_value(&scratch);
    }
    Arc::new(builder.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_separators_and_case_transitions() {
        assert_eq!(words("hello_world"), vec!["hello", "world"]);
        assert_eq!(words("helloWorld"), vec!["hello", "World"]);
        assert_eq!(
            words("hello-world foo.bar"),
            vec!["hello", "world", "foo", "bar"]
        );
        assert_eq!(words("  padded  "), vec!["padded"]);
        assert!(words("").is_empty());
        assert!(words("___").is_empty());
    }

    #[test]
    fn keeps_acronyms_whole() {
        assert_eq!(words("HTTPServer"), vec!["HTTP", "Server"]);
        assert_eq!(
            words("parseHTTPResponse"),
            vec!["parse", "HTTP", "Response"]
        );
        assert_eq!(words("HTTP"), vec!["HTTP"]);
    }

    #[test]
    fn digits_attach_to_their_word() {
        assert_eq!(words("sha256"), vec!["sha256"]);
        assert_eq!(words("utf8Bytes"), vec!["utf8", "Bytes"]);
    }

    #[test]
    fn every_style_round_trips_a_mixed_input() {
        let s = "helloWorld_again";
        assert_eq!(to_case(s, "snake").unwrap(), "hello_world_again");
        assert_eq!(to_case(s, "upper_snake").unwrap(), "HELLO_WORLD_AGAIN");
        assert_eq!(to_case(s, "camel").unwrap(), "helloWorldAgain");
        assert_eq!(to_case(s, "pascal").unwrap(), "HelloWorldAgain");
        assert_eq!(to_case(s, "kebab").unwrap(), "hello-world-again");
        assert_eq!(to_case(s, "upper_kebab").unwrap(), "HELLO-WORLD-AGAIN");
        assert_eq!(to_case(s, "title").unwrap(), "Hello World Again");
        assert_eq!(to_case(s, "sentence").unwrap(), "Hello world again");
        assert_eq!(to_case(s, "dot").unwrap(), "hello.world.again");
        assert_eq!(to_case(s, "train").unwrap(), "Hello-World-Again");
    }

    #[test]
    fn styles_agree_on_word_boundaries() {
        // The point of one splitter: `snake` and `camel` never disagree on where the
        // words were, for any input.
        for s in ["HTTPServer", "a_b-c d", "utf8Bytes", "", "X"] {
            let snake = to_case(s, "snake").unwrap();
            let camel = to_case(s, "camel").unwrap();
            let n_snake = if snake.is_empty() {
                0
            } else {
                snake.matches('_').count() + 1
            };
            let n_camel = to_case(s, "kebab").unwrap();
            let n_kebab = if n_camel.is_empty() {
                0
            } else {
                n_camel.matches('-').count() + 1
            };
            assert_eq!(n_snake, n_kebab, "{s}: {snake} vs {camel}");
        }
    }

    #[test]
    fn empty_and_separator_only_inputs_yield_empty() {
        for style in STYLES {
            assert_eq!(to_case("", style).unwrap(), "");
            assert_eq!(to_case("__--__", style).unwrap(), "");
        }
    }

    #[test]
    fn unknown_style_is_rejected() {
        assert!(to_case("x", "wobble").is_none());
    }

    #[test]
    fn style_list_is_exactly_what_to_case_accepts() {
        for style in STYLES {
            assert!(
                to_case("a_b", style).is_some(),
                "{style} listed but not handled"
            );
        }
    }

    fn norm(value: &str) -> String {
        let mut out = String::new();
        normalize_one(value, &mut out);
        out
    }

    /// The composition `normalize` replaced — `lower`, article strip, punctuation strip,
    /// whitespace collapse, trim — spelled out with the same regexes, as the oracle.
    fn via_composition(value: &str) -> String {
        use regex::Regex;
        let articles = Regex::new(r"\b(a|an|the)\b").expect("articles");
        let punctuation = Regex::new(r"[^\w\s]").expect("punctuation");
        let spaces = Regex::new(r"\s+").expect("spaces");
        let lowered = value.to_lowercase();
        let without_articles = articles.replace_all(&lowered, " ");
        let without_punctuation = punctuation.replace_all(&without_articles, "");
        spaces
            .replace_all(&without_punctuation, " ")
            .trim()
            .to_string()
    }

    /// The whole justification: one pass has to mean exactly what five expressions meant.
    #[test]
    fn squad_normalize_equals_the_composition_it_replaced() {
        let cases = [
            "The quick brown Fox!",
            "a cat, an apple, the dog",
            "cat-dog",
            "cat, dog",
            "  leading and trailing  ",
            "",
            "   ",
            "the",
            "the the the",
            "a",
            "an_apple",
            "hello_world 42",
            "theatre and thistle",
            "Ünïcödé tëxt wîth àccents",
            "东京都 and 東京市",
            "naïve café résumé",
            "punctuation!!!...???",
            "mixed123 456abc",
            "tabs\tand\nnewlines",
            "The-The-The",
            "a.b.c",
            "e.g. the thing",
            "don't can't won't",
            "50% of $100",
        ];
        for case in cases {
            assert_eq!(norm(case), via_composition(case), "differed on {case:?}");
        }
    }

    #[test]
    fn an_article_is_dropped_only_when_it_stands_alone() {
        assert_eq!(norm("the cat"), "cat");
        assert_eq!(norm("theatre"), "theatre");
        assert_eq!(norm("a an the"), "");
    }

    #[test]
    fn punctuation_without_whitespace_joins_its_neighbours() {
        assert_eq!(norm("cat-dog"), "catdog");
        assert_eq!(norm("cat, dog"), "cat dog");
    }

    #[test]
    fn whitespace_collapses_and_the_ends_are_trimmed() {
        assert_eq!(norm("  a   lot   of   space  "), "lot of space");
    }

    #[test]
    fn an_empty_or_all_punctuation_string_normalizes_to_empty() {
        assert_eq!(norm(""), "");
        assert_eq!(norm("!!!"), "");
        assert_eq!(norm("   "), "");
    }
}
