//! Identifier case conversion for `StrFunc::ToCase` — one word splitter, ten styles.
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
}
