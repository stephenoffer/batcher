//! How deep a plan document is, measured without recursing into it.
//!
//! [`RelOp::from_json`](crate::RelOp::from_json) disables serde_json's 128-level recursion
//! guard, because a legitimately deep generated pipeline nests past it. That was the right
//! call for the arbitrary 128, and the wrong call to leave unbounded: a serde-derived
//! visitor frame for `RelOp` is large, so past a few hundred levels deserialization walks
//! off the end of the stack. The failure mode is the worst kind — not a `Result::Err` the
//! caller can turn into a message, but a guard-page `SIGSEGV` that Rust converts into an
//! **uncatchable `SIGABRT`**. On a `_FlightWorker` actor it surfaces as an opaque
//! `ActorDiedError` rather than "your plan is too deep".
//!
//! So the limit comes back, but as a *measured* bound rather than a guess, and it is
//! checked by a scan that cannot itself overflow: one pass over the bytes, counting
//! bracket nesting, allocating nothing.
//!
//! # Where the number came from
//!
//! Measured on the debug profile, which has the largest frames and is therefore the
//! conservative case (`just build` installs a debug engine, so this is a real
//! configuration, not a worst-case fiction):
//!
//! | stack | deepest that parses | first that aborts |
//! |---|---|---|
//! | 2 MiB (the rayon worker default) | 650 | 700 |
//! | 8 MiB (the thread that actually calls `from_json`) | 2500 | 3000 |
//!
//! Roughly 3.2 KiB of stack per level, consistently across both.
//!
//! # Parsing is not the expensive pass, and this limit used to be set as if it were
//!
//! The table above measures **deserialization**, and calibrating the limit against it alone
//! was wrong: a plan is not only parsed, it is *evaluated*, and `Expr::eval` — with its
//! analyses and the compiler-generated `Drop` of the recursive enum — carries Arrow arrays
//! and match temporaries in every frame. Measured on the same debug profile, evaluation
//! costs **roughly 20 KiB per level, about six times deserialization**, so on rayon's
//! 2 MiB worker default a plan **evaluated** only to ~84 levels and aborted by 104 — against
//! a limit of 512 that had admitted it.
//!
//! That gap was a live `SIGSEGV`, not a theoretical one. Everything in it parsed cleanly and
//! passed this guard before dying: `col.is_in(values)` over 100 members (318 for
//! `TfidfVectorizer(stop_words="english")`), an `IsotonicCalibrator` at its default 100 bins,
//! a 100-term arithmetic chain.
//!
//! The fix is on the other side of the pair: `bc_interp::par::WORKER_STACK_BYTES` gives each
//! worker 32 MiB, which carries evaluation to 509 measured levels — past this limit, so the
//! guard is once again the binding constraint and a too-deep plan gets this error instead of
//! a signal. **The two constants are a pair.** Raising [`MAX_PLAN_DEPTH`] without raising the
//! worker stack re-opens exactly the window described here, and it re-opens it silently,
//! because the parse measurements above will keep looking comfortable.
//!
//! [`MAX_PLAN_DEPTH`] is 512, which is chosen against three separate bounds:
//!
//! - **Above anything real.** A 100-operator `.filter(...)` chain — the shape that
//!   motivated lifting serde's guard in the first place — measures 103 levels. Ordinary
//!   analytic plans measure 5 to 6.
//! - **Above anything the control plane can emit by default.** Python's `to_ir()` is
//!   itself recursive (`plan/logical/relational.py`), burning two frames per level, so at
//!   the default `sys.getrecursionlimit()` of 1000 it raises `RecursionError` at about 500
//!   levels. A plan deeper than this limit therefore cannot be produced without the caller
//!   having deliberately raised their own recursion limit, which means **this guard
//!   rejects nothing that used to work**.
//! - **Below where the stack breaks.** 5x margin on the 8 MiB thread that really calls
//!   `from_json`, and clear of the 509 levels a worker can *evaluate* on
//!   `bc_interp::par::WORKER_STACK_BYTES`, in the build profile with the fattest frames.
//!   Evaluation is the tighter of the two bounds and therefore the one to check.

/// Maximum nesting depth accepted in a plan IR document.
///
/// See the module docs for the measurements behind the number. Raising it requires
/// re-measuring, not reasoning: the safe value depends on the build profile and on the
/// stack of whichever thread deserializes.
pub const MAX_PLAN_DEPTH: usize = 512;

/// The deepest `{`/`[` nesting in `s`, counting bracket characters only.
///
/// Bytes inside string literals are skipped, along with their backslash escapes — without
/// that, a plan containing the literal `"{{{{"` (a `LIKE` pattern, a JSON-path expression,
/// a format string) would inflate its own measured depth and be rejected for containing
/// data. That is the classic bug in this scanner and it has its own test.
///
/// Scanning bytes rather than `char`s is safe here because every byte of a multi-byte
/// UTF-8 sequence has the high bit set, so none can be mistaken for an ASCII bracket or
/// quote.
///
/// Returns the maximum depth reached. Unbalanced input is not this function's problem —
/// serde reports malformed JSON with a far better message than a depth scanner could, so
/// this only ever reports a number and lets the parse decide.
pub fn json_max_depth(s: &str) -> usize {
    let mut depth: usize = 0;
    let mut max: usize = 0;
    let mut in_string = false;
    let mut escaped = false;

    for &byte in s.as_bytes() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth += 1;
                if depth > max {
                    max = depth;
                }
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    max
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_documents() {
        assert_eq!(json_max_depth(""), 0);
        assert_eq!(json_max_depth("null"), 0);
        assert_eq!(json_max_depth("{}"), 1);
        assert_eq!(json_max_depth(r#"{"a":1,"b":2}"#), 1);
    }

    #[test]
    fn nesting_is_the_deepest_path_not_the_total() {
        // Two siblings at depth 2 is depth 2, not 4 — the stack only holds one path.
        assert_eq!(json_max_depth(r#"{"a":{"x":1},"b":{"y":2}}"#), 2);
        assert_eq!(json_max_depth(r#"{"a":{"b":{"c":1}}}"#), 3);
    }

    #[test]
    fn arrays_count_too() {
        // `Union` nests its inputs in a list, so an array level is a real stack level.
        assert_eq!(json_max_depth(r#"[[[1]]]"#), 3);
        assert_eq!(json_max_depth(r#"{"inputs":[{"op":"scan"}]}"#), 3);
    }

    #[test]
    fn brackets_inside_a_string_literal_are_data_not_depth() {
        // The bug this scanner is most likely to have. A predicate against a JSON column,
        // or a LIKE pattern, legitimately contains braces; counting them would reject a
        // valid plan for the contents of its own data.
        assert_eq!(json_max_depth(r#"{"value":"{{{{{{"}"#), 1);
        assert_eq!(json_max_depth(r#"{"value":"[[[["}"#), 1);
        assert_eq!(json_max_depth(r#"{"a":{"path":"$.x[0].y"}}"#), 2);
    }

    #[test]
    fn escaped_quote_does_not_end_the_string() {
        // `"he said \" {" ` — the escaped quote must not drop us back into structure,
        // or the brace after it would be counted.
        assert_eq!(json_max_depth(r#"{"v":"he said \" {"}"#), 1);
        // A trailing escaped backslash *does* end the string, so the following brace is
        // real structure.
        assert_eq!(json_max_depth(r#"{"v":"back\\","w":{"x":1}}"#), 2);
    }

    #[test]
    fn measures_a_deep_chain_exactly() {
        let mut json = String::from(r#"{"op":"scan","source_id":0}"#);
        for _ in 0..100 {
            json = format!(r#"{{"op":"distinct","input":{json}}}"#);
        }
        // 100 wrappers plus the innermost scan object.
        assert_eq!(json_max_depth(&json), 101);
    }

    #[test]
    fn unbalanced_input_reports_a_number_rather_than_panicking() {
        // serde gives the real error; this must not be the thing that fails first.
        assert_eq!(json_max_depth("}}}}"), 0);
        assert_eq!(json_max_depth("{{{"), 3);
    }
}
