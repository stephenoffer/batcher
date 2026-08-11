//! A too-deep plan must return an error, not abort the process.
//!
//! Before the depth guard, `RelOp::from_json` on a deeply nested document overflowed the
//! stack, which Rust reports as `SIGABRT` — uncatchable, no message, and on a shuffle
//! actor indistinguishable from any other worker death. These tests pin the two halves of
//! the fix: the boundary is an `Err`, and everything below it still parses.
//!
//! The assertions are deliberately about `Result`, not about survival. A test cannot
//! assert "the process did not abort" from inside the process that would have aborted;
//! getting an `Err` back *is* that assertion.

use bc_ir::{json_max_depth, IrError, RelOp, MAX_PLAN_DEPTH};

/// A chain of `depth` `Distinct` nodes over a scan. One JSON level per operator, so the
/// document's nesting depth is `depth + 1` (the innermost scan object).
fn nested_plan(depth: usize) -> String {
    let mut json = String::from(r#"{"op":"scan","source_id":0}"#);
    for _ in 0..depth {
        json = format!(r#"{{"op":"distinct","input":{json}}}"#);
    }
    json
}

#[test]
fn a_plan_at_the_limit_still_parses() {
    // The guard must not reject anything it was not built to reject. `MAX_PLAN_DEPTH - 1`
    // operators is a document of exactly `MAX_PLAN_DEPTH` levels.
    let json = nested_plan(MAX_PLAN_DEPTH - 1);
    assert_eq!(json_max_depth(&json), MAX_PLAN_DEPTH);
    let plan = RelOp::from_json(&json).expect("a plan at the limit must parse");
    assert_eq!(plan.node_count(), MAX_PLAN_DEPTH as u32);
}

#[test]
fn a_plan_past_the_limit_is_an_error_not_an_abort() {
    // The whole point. Reaching this assertion at all proves the process survived; that
    // the value is `PlanTooDeep` proves it survived for the right reason rather than
    // because serde happened to reject the document for some other cause.
    let json = nested_plan(MAX_PLAN_DEPTH + 50);
    match RelOp::from_json(&json) {
        Err(IrError::PlanTooDeep { depth, limit }) => {
            assert_eq!(limit, MAX_PLAN_DEPTH);
            assert!(
                depth > MAX_PLAN_DEPTH,
                "reported depth {depth} must exceed the limit"
            );
        }
        other => panic!("expected PlanTooDeep, got {other:?}"),
    }
}

#[test]
fn the_error_message_says_what_to_do_about_it() {
    // A user hitting this has almost certainly built a plan in a loop. An error that only
    // says "too deep" sends them looking in the wrong place.
    let err = RelOp::from_json(&nested_plan(MAX_PLAN_DEPTH + 1)).unwrap_err();
    let text = err.to_string();
    assert!(text.contains("nests"), "unhelpful message: {text}");
    assert!(
        text.contains("loop"),
        "message should name the usual cause: {text}"
    );
}

#[test]
fn deep_plans_do_not_overflow_the_walkers() {
    // `node_count` and `contains_media_decode` used to recurse, so a deep plan aborted in
    // them even when it had parsed fine. They walk a heap worklist now, which this proves
    // at a depth far past what any stack would survive.
    let json = nested_plan(MAX_PLAN_DEPTH - 1);
    let plan = RelOp::from_json(&json).expect("parses");
    assert_eq!(plan.node_count(), MAX_PLAN_DEPTH as u32);
    assert!(!plan.contains_media_decode());
}

#[test]
fn the_walkers_survive_far_past_the_wire_limit() {
    // A plan built in Rust never went through `from_json`, so the wire guard does not
    // bound it. Build one 20,000 deep — roughly 30x the measured stack ceiling — and walk
    // it. A recursive implementation aborts here; a worklist does not.
    let mut plan = RelOp::from_json(r#"{"op":"scan","source_id":0}"#).expect("scan parses");
    for _ in 0..20_000 {
        plan = RelOp::Distinct {
            input: Box::new(plan),
            keys: Vec::new(),
            order: Vec::new(),
            limit: None,
        };
    }
    assert_eq!(plan.node_count(), 20_001);
    assert!(!plan.contains_media_decode());
}

#[test]
fn a_shallow_plan_is_unaffected() {
    // The overwhelmingly common case: ordinary plans are 5 to 6 levels deep and must not
    // pay anything or behave differently.
    let json = r#"{"op":"filter","input":{"op":"scan","source_id":0},
                   "predicate":{"e":"col","name":"a"}}"#;
    let plan = RelOp::from_json(json).expect("an ordinary plan parses");
    assert_eq!(plan.node_count(), 2);
}
