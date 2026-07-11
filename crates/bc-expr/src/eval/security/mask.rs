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

#[cfg(test)]
mod tests {
    use super::mask;

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
