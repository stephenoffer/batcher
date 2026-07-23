//! A process-wide memo for compiled regexes.
//!
//! `regex::Regex::new` parses a pattern and builds an automaton — tens of microseconds for a
//! realistic pattern, and entirely fixed cost. Every regex-shaped string function
//! (`regexp_matches`, `regexp_replace`, `regexp_extract`, …) and **every `LIKE`/`ILIKE`**
//! compiles its pattern inside `eval_str`, which runs **once per morsel**. Morsels are 16,384
//! rows, so a `LIKE` over a 60M-row column rebuilt the same automaton ~3,700 times and threw it
//! away each time. DuckDB and Polars both compile the pattern once per operator; this closes
//! that gap.
//!
//! The pattern is a *literal in the plan*, so the key space is the number of distinct patterns a
//! query mentions — single digits, not one per row. Compilation is a pure function of the regex
//! source, so the artifact is reusable across every batch, operator, and `execute_plan` call.
//!
//! Keys are the full regex source compared for equality, never merely hashed, so a collision
//! cannot hand back an automaton built from a different pattern. `None` records a
//! *known-invalid* pattern, so a bad regex costs one failed parse rather than one per morsel.
//!
//! This changes throughput only: a cached `Regex` matches exactly what a freshly-compiled one
//! does, so the interpreter stays the oracle.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

/// Cap on retained automata. Each entry is small (a compiled program), and the working set is
/// the distinct patterns across the query shapes a long-lived driver sees — far below this. The
/// bound exists so a driver that generates patterns dynamically cannot grow the map without
/// limit.
const MAX_ENTRIES: usize = 1024;

/// `None` = the source is not a valid regex. Remembering the failure keeps a bad pattern from
/// re-parsing once per morsel.
type Entry = Option<Arc<regex::Regex>>;

fn cache() -> &'static RwLock<HashMap<String, Entry>> {
    static CACHE: OnceLock<RwLock<HashMap<String, Entry>>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// `Regex::new(source)`, memoized. `None` means the source does not compile.
///
/// Callers map `None` onto their own error (`ExprError::InvalidRegex`) so the message can name
/// the *user's* pattern rather than the desugared source a `LIKE` produces.
pub(super) fn compile_cached(source: &str) -> Entry {
    if let Ok(map) = cache().read() {
        if let Some(hit) = map.get(source) {
            return hit.clone();
        }
    }
    // Compile outside the lock: holding the write lock across a parse would serialize every
    // other operator's first-sight compile behind this one. Two threads racing on the same
    // source both compile; that is wasted work but never wrong, since the automata are
    // equivalent.
    let compiled = regex::Regex::new(source).ok().map(Arc::new);
    if let Ok(mut map) = cache().write() {
        // Bounded: drop the memo wholesale rather than evicting one entry. Compiles are rare
        // (once per pattern) and refilling is cheap relative to tracking LRU order. In-flight
        // users hold their own `Arc`, so clearing never frees an automaton still matching.
        if map.len() >= MAX_ENTRIES {
            map.clear();
        }
        // Hand back whatever is *in* the map, so a lost race still converges every caller onto
        // a single automaton rather than leaving each with its own copy.
        return map.entry(source.to_string()).or_insert(compiled).clone();
    }
    compiled
}

#[cfg(test)]
mod tests {
    use super::compile_cached;
    use std::sync::Arc;

    #[test]
    fn second_compile_returns_the_same_automaton() {
        let first = compile_cached(r"^a\d+$").expect("valid regex");
        let second = compile_cached(r"^a\d+$").expect("valid regex");
        assert!(
            Arc::ptr_eq(&first, &second),
            "cache must reuse the automaton, not recompile per morsel"
        );
        assert!(first.is_match("a123"));
        assert!(!first.is_match("b123"));
    }

    #[test]
    fn distinct_sources_do_not_share_an_automaton() {
        let a = compile_cached("^a").expect("valid");
        let b = compile_cached("^b").expect("valid");
        assert!(!Arc::ptr_eq(&a, &b));
        assert!(a.is_match("abc") && !a.is_match("bcd"));
        assert!(b.is_match("bcd") && !b.is_match("abc"));
    }

    /// A `LIKE` desugars to a case-insensitive source; it must not collide with the
    /// case-sensitive one, or `ILIKE` would silently become `LIKE`.
    #[test]
    fn case_insensitive_source_is_a_distinct_key() {
        let sensitive = compile_cached("(?s)^abc$").expect("valid");
        let insensitive = compile_cached("(?i)(?s)^abc$").expect("valid");
        assert!(!Arc::ptr_eq(&sensitive, &insensitive));
        assert!(!sensitive.is_match("ABC"));
        assert!(insensitive.is_match("ABC"));
    }

    #[test]
    fn an_invalid_pattern_is_remembered_as_invalid() {
        assert!(compile_cached("(unclosed").is_none());
        // And again — the negative result is cached, not re-parsed.
        assert!(compile_cached("(unclosed").is_none());
    }
}
