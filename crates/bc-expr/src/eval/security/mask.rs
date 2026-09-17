//! Character masking — the redaction primitive behind partial-disclosure policies
//! ("show only the last four digits").
//!
//! Masking is lossy and unkeyed: unlike `super::crypto`, there is nothing to reverse
//! and no key to protect. It is the right tool when the *shape* of a value must survive
//! for a human reader (a support agent confirming a card's last four) and the value
//! itself must not.

/// Replace every character of `s` outside the first `show_first` and the last
/// `show_last` with `ch`.
///
/// Counts Unicode characters, not bytes, so a masked string has the same character
/// length as its input (`.str.len()` is preserved) and multi-byte input is never split
/// mid-codepoint. When the revealed windows meet or overlap (that is, when
/// `show_first + show_last` reaches the value's length) nothing is masked and `s` is
/// returned as-is, which makes the function monotone in the reveal counts and keeps
/// short values from silently becoming cleartext-with-extra-steps.
pub(super) fn mask(s: &str, ch: char, show_first: usize, show_last: usize) -> String {
    let len = s.chars().count();
    if show_first.saturating_add(show_last) >= len {
        return s.to_string();
    }
    let keep_from = len - show_last;
    s.chars()
        .enumerate()
        .map(|(i, c)| {
            if i < show_first || i >= keep_from {
                c
            } else {
                ch
            }
        })
        .collect()
}

/// The replacement for each character class Spark's `mask` distinguishes, `None` to keep
/// that class unmasked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ClassMask {
    pub upper: Option<char>,
    pub lower: Option<char>,
    pub digit: Option<char>,
    pub other: Option<char>,
}

/// The general categories Spark's `Character.getType` switch names: `Lu`, `Ll`, `Nd`.
/// One alternation with a group per class, so a value is classified in one scan.
fn class_regex() -> &'static regex::Regex {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        regex::Regex::new(r"(\p{Lu})|(\p{Ll})|(\p{Nd})").expect("a fixed, valid pattern")
    })
}

/// Spark `mask(str, upperChar, lowerChar, digitChar, otherChar)`: replace each character
/// by its class's replacement.
///
/// Spark classifies UTF-16 code units, so a character outside the Basic Multilingual
/// Plane arrives as two surrogates of category `SURROGATE`: it is "other" whatever it is,
/// and an `other` replacement is written once per surrogate. That is reproduced rather
/// than corrected, because a result that agrees with Spark on `😀` is the point of this
/// form; it is also the one case where the output is not the input's character length.
pub(super) fn mask_by_class(s: &str, m: ClassMask) -> String {
    let mut out = String::with_capacity(s.len());
    let mut copied = 0;
    let push_other = |out: &mut String, text: &str| {
        for c in text.chars() {
            match m.other {
                Some(r) if c.len_utf16() == 2 => {
                    out.push(r);
                    out.push(r);
                }
                Some(r) => out.push(r),
                None => out.push(c),
            }
        }
    };
    for caps in class_regex().captures_iter(s) {
        let whole = caps.get(0).expect("group 0 always participates");
        push_other(&mut out, &s[copied..whole.start()]);
        let c = whole
            .as_str()
            .chars()
            .next()
            .expect("a class matches one character");
        // A letter or digit beyond the BMP is a surrogate pair to Spark, so "other".
        let replacement = if c.len_utf16() == 2 {
            push_other(&mut out, whole.as_str());
            copied = whole.end();
            continue;
        } else if caps.get(1).is_some() {
            m.upper
        } else if caps.get(2).is_some() {
            m.lower
        } else {
            m.digit
        };
        out.push(replacement.unwrap_or(c));
        copied = whole.end();
    }
    push_other(&mut out, &s[copied..]);
    out
}

#[cfg(test)]
mod tests {
    use super::{mask, mask_by_class, ClassMask};

    const SPARK_DEFAULT: ClassMask = ClassMask {
        upper: Some('X'),
        lower: Some('x'),
        digit: Some('n'),
        other: None,
    };

    #[test]
    fn class_masking_reproduces_sparks_documented_examples() {
        // Every example in `maskExpressions.scala`'s `ExpressionDescription`.
        assert_eq!(
            mask_by_class("abcd-EFGH-8765-4321", SPARK_DEFAULT),
            "xxxx-XXXX-nnnn-nnnn"
        );
        let q = ClassMask {
            upper: Some('Q'),
            ..SPARK_DEFAULT
        };
        assert_eq!(
            mask_by_class("abcd-EFGH-8765-4321", q),
            "xxxx-QQQQ-nnnn-nnnn"
        );
        assert_eq!(mask_by_class("AbCD123-@$#", q), "QxQQnnn-@$#");
        let lower_q = ClassMask {
            lower: Some('q'),
            ..q
        };
        assert_eq!(mask_by_class("AbCD123-@$#", lower_q), "QqQQnnn-@$#");
        let digit_d = ClassMask {
            digit: Some('d'),
            ..lower_q
        };
        assert_eq!(mask_by_class("AbCD123-@$#", digit_d), "QqQQddd-@$#");
        let other_o = ClassMask {
            other: Some('o'),
            ..digit_d
        };
        assert_eq!(mask_by_class("AbCD123-@$#", other_o), "QqQQdddoooo");
        assert_eq!(mask_by_class("AbCD123-@$#", SPARK_DEFAULT), "XxXXnnn-@$#");
        let keep_upper = ClassMask {
            upper: None,
            ..other_o
        };
        assert_eq!(mask_by_class("AbCD123-@$#", keep_upper), "AqCDdddoooo");
        let keep_letters = ClassMask {
            lower: None,
            ..keep_upper
        };
        assert_eq!(mask_by_class("AbCD123-@$#", keep_letters), "AbCDdddoooo");
        let other_only = ClassMask {
            digit: None,
            ..keep_letters
        };
        assert_eq!(mask_by_class("AbCD123-@$#", other_only), "AbCD123oooo");
        let keep_all = ClassMask {
            upper: None,
            lower: None,
            digit: None,
            other: None,
        };
        assert_eq!(mask_by_class("AbCD123-@$#", keep_all), "AbCD123-@$#");
    }

    #[test]
    fn class_masking_is_unicode_by_category_and_utf16_beyond_the_bmp() {
        assert_eq!(mask_by_class("Éé٣ ", SPARK_DEFAULT), "Xxn ");
        let other = ClassMask {
            other: Some('*'),
            ..SPARK_DEFAULT
        };
        assert_eq!(mask_by_class("a😀B𝐀", other), "x**X**");
        assert_eq!(mask_by_class("", other), "");
    }

    #[test]
    fn masks_everything_by_default() {
        assert_eq!(mask("secret", 'X', 0, 0), "XXXXXX");
        assert_eq!(mask("", 'X', 0, 0), "");
    }

    #[test]
    fn reveals_a_prefix_and_a_suffix() {
        assert_eq!(mask("4111111111111234", 'X', 0, 4), "XXXXXXXXXXXX1234");
        assert_eq!(mask("4111111111111234", 'X', 4, 0), "4111XXXXXXXXXXXX");
        assert_eq!(mask("4111111111111234", '*', 2, 2), "41************34");
    }

    #[test]
    fn overlapping_windows_reveal_the_whole_value_rather_than_over_masking() {
        assert_eq!(mask("abc", 'X', 2, 2), "abc");
        assert_eq!(mask("abc", 'X', 3, 0), "abc");
        assert_eq!(mask("abc", 'X', 0, 9), "abc");
    }

    #[test]
    fn character_length_is_preserved_across_multibyte_input() {
        let masked = mask("héllo", '#', 1, 1);
        assert_eq!(masked, "h###o");
        assert_eq!(masked.chars().count(), "héllo".chars().count());
    }
}
