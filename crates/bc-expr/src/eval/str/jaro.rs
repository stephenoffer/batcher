//! Jaro and Jaro-Winkler string similarity (the `.str.jaro`/`.str.jaro_winkler` funcs).
//!
//! The fuzzy-match metrics for short strings (names, addresses) that drive entity
//! resolution and record linkage. Both match `DuckDB`'s `jaro_similarity` /
//! `jaro_winkler_similarity` (differential-tested), computed over Unicode **scalar values**
//! (`char`s) so multi-byte text scores by character, not by byte.

/// Jaro similarity of `a` and `b` in `[0, 1]` (1.0 = identical, 0.0 = no shared characters).
///
/// Two characters match if they are equal and no further apart than
/// `max(|a|, |b|)/2 - 1`. With `m` matches and `t` transpositions (half the number of
/// out-of-order matched pairs), the score is `(m/|a| + m/|b| + (m-t)/m) / 3`.
pub(crate) fn jaro(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (la, lb) = (a.len(), b.len());
    // An empty operand scores 0, *including* when both are empty. That reads as wrong —
    // two empty strings are identical — but it is what DuckDB answers
    // (`jaro_similarity('', '')` is 0.0), and the metric is only meaningful against the
    // engine it is compared with. Returning 1.0 for the pair, which this used to do,
    // disagreed on exactly the row an empty string reaches.
    if la == 0 || lb == 0 {
        return 0.0;
    }
    // Match window: characters farther apart than this cannot be paired.
    let window = (la.max(lb) / 2).saturating_sub(1);

    let mut a_matched = vec![false; la];
    let mut b_matched = vec![false; lb];
    let mut matches = 0usize;
    for (i, &ca) in a.iter().enumerate() {
        let lo = i.saturating_sub(window);
        let hi = (i + window + 1).min(lb);
        for j in lo..hi {
            if !b_matched[j] && b[j] == ca {
                a_matched[i] = true;
                b_matched[j] = true;
                matches += 1;
                break;
            }
        }
    }
    if matches == 0 {
        return 0.0;
    }

    // Transpositions: matched characters, taken in order, that don't line up.
    let mut transpositions = 0usize;
    let mut k = 0usize;
    for i in 0..la {
        if a_matched[i] {
            while !b_matched[k] {
                k += 1;
            }
            if a[i] != b[k] {
                transpositions += 1;
            }
            k += 1;
        }
    }
    let t = transpositions as f64 / 2.0;
    let m = matches as f64;
    (m / la as f64 + m / lb as f64 + (m - t) / m) / 3.0
}

/// Jaro-Winkler similarity: Jaro plus a bonus for a shared prefix (up to 4 characters,
/// scaling factor `0.1`), applied only when the Jaro score already clears `0.7` — matching
/// DuckDB / the classic Winkler boost threshold.
pub(crate) fn jaro_winkler(a: &str, b: &str) -> f64 {
    let j = jaro(a, b);
    if j <= 0.7 {
        return j;
    }
    let prefix = a
        .chars()
        .zip(b.chars())
        .take(4)
        .take_while(|(x, y)| x == y)
        .count();
    j + prefix as f64 * 0.1 * (1.0 - j)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(x: f64, y: f64) -> bool {
        (x - y).abs() < 1e-9
    }

    #[test]
    fn identical_and_disjoint() {
        assert!(close(jaro("abc", "abc"), 1.0));
        assert!(close(jaro("abc", "xyz"), 0.0));
        assert!(close(jaro("a", ""), 0.0));
    }

    /// Two empty strings score **0**, not 1.
    ///
    /// They are identical, so 1.0 is the reading the metric's definition suggests and the
    /// one this used to return. DuckDB answers 0.0 (`jaro_similarity('', '')`), and this
    /// kernel exists to agree with DuckDB — an entity-resolution score that disagrees on
    /// the empty row is a different metric, not a rounding difference.
    #[test]
    fn two_empty_strings_score_zero_as_duckdb_does() {
        assert!(close(jaro("", ""), 0.0));
    }

    #[test]
    fn known_jaro_values() {
        // Classic textbook cases.
        assert!((jaro("MARTHA", "MARHTA") - 0.944444444).abs() < 1e-6);
        assert!((jaro("DWAYNE", "DUANE") - 0.822222222).abs() < 1e-6);
        assert!((jaro("DIXON", "DICKSONX") - 0.766666666).abs() < 1e-6);
    }

    #[test]
    fn winkler_prefix_bonus() {
        // MARTHA/MARHTA share a 3-char prefix: 0.9444 + 3*0.1*(1-0.9444) = 0.96111.
        assert!((jaro_winkler("MARTHA", "MARHTA") - 0.961111111).abs() < 1e-6);
        // DIXON/DICKSONX Jaro is 0.7667 (> 0.7) with a 2-char shared prefix ("DI"):
        // 0.7667 + 2*0.1*(1-0.7667) = 0.81333.
        assert!((jaro_winkler("DIXON", "DICKSONX") - 0.813333333).abs() < 1e-6);
        // A pair below the 0.7 threshold gets no bonus → winkler == jaro.
        assert!(close(jaro_winkler("abc", "xbz"), jaro("abc", "xbz")));
    }

    #[test]
    fn multibyte_by_character() {
        // Scored over chars, not bytes: a 3-char accented string vs itself is 1.0.
        assert!(close(jaro("héllo", "héllo"), 1.0));
    }
}
