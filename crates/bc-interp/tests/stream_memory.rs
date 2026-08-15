//! The streaming executor's peak memory must not scale with the input.
//!
//! This is the whole reason the streaming executor exists, and it is the one property a
//! correctness test cannot see. `execute` materializes every operator's output, so a left-deep
//! join chain holds each join's full result while building the next — which is why TPC-H sf100
//! q3/q4/q5 peak at **133 GB and are OOM-killed** on a machine where DuckDB streams the same
//! queries in a few GB.
//!
//! `execute_streaming` threads one morsel through every probe and folds the aggregate
//! incrementally, so its peak is the build tables + one morsel + the aggregate's state — a
//! constant, independent of how many rows flow through. Measured here on the q3/q4/q5 shape:
//!
//! ```text
//!      rows   materializing    streaming
//!   1000000       221.3 MB       1.0 MB   (233x)
//!   2000000       442.5 MB       1.0 MB   (464x)
//!   4000000       854.5 MB       1.0 MB   (891x)
//! ```
//!
//! The ratio is not the point — the *shape* is. Materializing grows linearly with the input;
//! streaming is flat. The assertion below is deliberately loose (peak must not double across a
//! 4x input growth), because it is guarding against the reintroduction of materialization, not
//! policing an allocator's exact byte count.
//!
//! A counting global allocator is installed for this test binary only, which is what lets it
//! measure peak *live heap* rather than a process-wide RSS high-water mark polluted by
//! whichever executor happened to run first.
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

/// Only one test in this binary may measure at a time.
///
/// `LIVE`/`PEAK` are process-global, and `cargo test` runs a binary's tests concurrently on
/// separate threads — so two measurements in flight at once corrupt each other in both
/// directions: the other test's multi-hundred-MB setup allocations inflate this one's
/// high-water mark, and its re-seed of `PEAK` can drop the mark *below* this test's baseline.
/// That second effect made `PEAK - base` underflow and panic with "attempt to subtract with
/// overflow". It surfaced only on a 4-core CI runner, because the interleaving on a many-core
/// dev box happens to keep the two tests apart. Every test here holds this for its whole body.
static MEASURING: Mutex<()> = Mutex::new(());

/// Take the measurement lock, ignoring poisoning.
///
/// A panic in one test must not turn every other test in the file into a confusing poison
/// error that hides the original failure.
fn measuring() -> std::sync::MutexGuard<'static, ()> {
    MEASURING.lock().unwrap_or_else(|e| e.into_inner())
}

struct Track;
unsafe impl GlobalAlloc for Track {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            let now = LIVE.fetch_add(l.size(), Ordering::Relaxed) + l.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static A: Track = Track;

use arrow::array::{ArrayRef, Int64Array};
use arrow::record_batch::RecordBatch;
use bc_interp::{execute, execute_streaming};
use bc_ir::RelOp;
use std::sync::Arc;

fn facts(n: i64, wide: usize) -> Vec<RecordBatch> {
    let mut cols: Vec<(String, ArrayRef)> = vec![
        (
            "k".into(),
            Arc::new(Int64Array::from(
                (0..n).map(|i| i % 100).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "v".into(),
            Arc::new(Int64Array::from((0..n).collect::<Vec<_>>())) as ArrayRef,
        ),
    ];
    // Extra payload columns, so each join's materialized output is genuinely large.
    for c in 0..wide {
        cols.push((
            format!("p{c}"),
            Arc::new(Int64Array::from(
                (0..n).map(|i| i + c as i64).collect::<Vec<_>>(),
            )) as ArrayRef,
        ));
    }
    vec![RecordBatch::try_from_iter(cols).unwrap()]
}

fn dim(name: &str) -> Vec<RecordBatch> {
    let k: ArrayRef = Arc::new(Int64Array::from((0..100i64).collect::<Vec<_>>()));
    let d: ArrayRef = Arc::new(Int64Array::from(
        (0..100i64).map(|i| i * 10).collect::<Vec<_>>(),
    ));
    vec![RecordBatch::try_from_iter(vec![("k", k), (name, d)]).unwrap()]
}

/// scan(facts) ⋈ d1 ⋈ d2 ⋈ d3 → aggregate. The q3/q4/q5 shape: a left-deep chain of inner
/// joins whose intermediates are what blow up.
fn deep_plan(wide: usize) -> RelOp {
    let mut out: Vec<String> = vec![
        r#"{"side":"left","name":"k","alias":"k"}"#.into(),
        r#"{"side":"left","name":"v","alias":"v"}"#.into(),
    ];
    for c in 0..wide {
        out.push(format!(r#"{{"side":"left","name":"p{c}","alias":"p{c}"}}"#));
    }
    let mut node = r#"{"op":"scan","source_id":0}"#.to_string();
    for (i, name) in ["a", "b", "c"].iter().enumerate() {
        let mut o = out.clone();
        o.push(format!(
            r#"{{"side":"right","name":"{name}","alias":"{name}"}}"#
        ));
        // carry previously joined dim cols forward
        for prev in ["a", "b", "c"].iter().take(i) {
            o.push(format!(
                r#"{{"side":"left","name":"{prev}","alias":"{prev}"}}"#
            ));
        }
        node = format!(
            r#"{{"op":"hash_join","left":{node},"right":{{"op":"scan","source_id":{}}},
                "left_keys":["k"],"right_keys":["k"],"join_type":"inner",
                "output":[{}],"strategy":"hash"}}"#,
            i + 1,
            o.join(",")
        );
    }
    let json = format!(
        r#"{{"op":"aggregate","input":{node},
            "group_keys":[{{"expr":{{"e":"col","name":"k"}},"alias":"k"}}],
            "aggregates":[{{"func":"sum","input":{{"e":"col","name":"v"}},"alias":"sv"}}]}}"#
    );
    serde_json::from_str(&json).unwrap()
}

/// Peak *additional* live heap while `f` runs.
///
/// The baseline and the high-water seed are the SAME `LIVE` reading, deliberately. Sampling
/// them separately — a `LIVE` load for the baseline, then a second one to seed `PEAK` — lets a
/// free land in between, leaving `PEAK` below the baseline and making the subtraction
/// underflow. Seeding `PEAK` with the baseline makes it total instead: `fetch_max` can only
/// ever raise the mark above the seed.
fn peak_delta(f: impl FnOnce()) -> usize {
    let base = LIVE.load(Ordering::Relaxed);
    PEAK.store(base, Ordering::Relaxed);
    f();
    PEAK.load(Ordering::Relaxed) - base
}

#[test]
fn peak_memory_streaming_vs_materializing() {
    let _measuring = measuring();
    let wide = 8;
    let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
    println!("\n  3 inner joins → aggregate (the q3/q4/q5 shape), {wide} payload cols\n");
    println!("      rows   materializing    streaming");

    let mut first = None;
    let mut last = None;
    for n in [1_000_000i64, 2_000_000, 4_000_000] {
        let srcs = vec![facts(n, wide), dim("a"), dim("b"), dim("c")];
        let p = deep_plan(wide);
        let mat = peak_delta(|| {
            std::hint::black_box(execute(&p, &srcs).unwrap());
        });
        let stream = peak_delta(|| {
            std::hint::black_box(execute_streaming(&p, &srcs, 0).unwrap());
        });
        println!(
            "  {:>8}   {:>9.1} MB   {:>7.1} MB   ({:.0}x)",
            n,
            mb(mat),
            mb(stream),
            mat as f64 / stream.max(1) as f64
        );
        if first.is_none() {
            first = Some(stream);
        }
        last = Some(stream);
    }
    // The point is not the ratio, it is the *shape*: materializing grows with the input;
    // streaming does not. Peak memory is the build tables + one morsel + the aggregate's state.
    let (f, l) = (first.unwrap(), last.unwrap());
    println!(
        "\n  streaming peak across a 4x input growth: {:.1} MB -> {:.1} MB\n",
        mb(f),
        mb(l)
    );
    assert!(
        l < f * 2,
        "streaming peak must not scale with input size: {} -> {}",
        f,
        l
    );
}

/// A breaker that cannot fit its input must give way to the spilling executor **before** it
/// has materialized that input, not after.
///
/// The streaming sort/distinct/union breakers hold their input, and when it exceeds the
/// envelope they return `MemoryBudgetExceeded` so the caller re-runs the query on the
/// executor that spills. That handoff is what turns a would-be OOM into a spill -- but it
/// only works if it happens before the allocation it is avoiding. Draining the whole input
/// and checking afterwards inverts it: an input ten times the envelope is ten envelopes of
/// resident memory before a single byte of the check runs, so the process dies at the drain
/// and the executor that could have spilled is never reached. The guard was, in exactly the
/// case it exists for, unreachable.
///
/// Measured on live heap rather than argued: the peak while refusing an input many times the
/// envelope must stay near the envelope.
#[test]
fn an_over_budget_breaker_gives_way_before_materializing_its_input() {
    let _measuring = measuring();
    // The sort's input is *computed*, not scanned. A scan over already-materialized sources
    // yields zero-copy slices, so collecting them allocates almost nothing and would hide the
    // very thing being measured; a projection allocates a fresh column per morsel, which is
    // what a real pipeline feeding a breaker does.
    let sort: RelOp = serde_json::from_str(
        r#"{"op":"sort",
            "input":{"op":"project","input":{"op":"scan","source_id":0},"exprs":[
                {"expr":{"e":"col","name":"k"},"alias":"k"},
                {"expr":{"e":"binary","op":"mul","left":{"e":"col","name":"v"},
                         "right":{"e":"lit","value":{"int":2}}},"alias":"v"}]},
            "keys":[{"expr":{"e":"col","name":"v"},"descending":false,"nulls_first":false}],
            "limit":null}"#,
    )
    .unwrap();

    // ~4 M rows over 8 payload columns: a few hundred MB, and far more than the envelope.
    let srcs = vec![facts(4_000_000, 8)];
    // What the sort would hold: the projection's two i64 columns over every row.
    let input_bytes: usize = 4_000_000 * 2 * std::mem::size_of::<i64>();
    let budget = 8 << 20; // 8 MiB

    let mut refused = false;
    let peak = peak_delta(|| {
        // `black_box` so the result cannot be optimized away.
        let r = std::hint::black_box(execute_streaming(&sort, &srcs, budget));
        refused = r.is_err();
    });

    assert!(
        refused,
        "a {input_bytes}-byte sort under an {budget}-byte envelope must give way, not run"
    );
    // Generous headroom: the bail holds up to the budget plus the morsel that crossed it,
    // plus whatever the scan pipeline holds in flight. The point is that it is bounded by the
    // *envelope* and not by the input -- an order of magnitude below the input either way.
    assert!(
        peak < input_bytes / 4,
        "refusing a {input_bytes}-byte input under an {budget}-byte envelope peaked at \
         {peak} bytes -- the whole input was materialized before the budget check ran, so \
         the handoff to the spilling executor happens after the OOM it exists to prevent"
    );
}

/// A `Project` over source 0, so every morsel allocates fresh columns.
///
/// A scan over already-materialized sources yields zero-copy slices, so an operator that held
/// all of them would allocate almost nothing and hide the very thing being measured. Every
/// memory test below feeds its operator through this for that reason.
fn computed_input() -> String {
    r#"{"op":"project","input":{"op":"scan","source_id":0},"exprs":[
           {"expr":{"e":"col","name":"k"},"alias":"k"},
           {"expr":{"e":"binary","op":"mul","left":{"e":"col","name":"v"},
                    "right":{"e":"lit","value":{"int":2}}},"alias":"v"}]}"#
        .to_string()
}

/// Peak memory for a top-N must scale with `k`, not with the relation it selects from.
///
/// `ORDER BY … LIMIT 10` keeps ten rows. The streaming executor nonetheless drained its whole
/// input first and only then reduced it, so the shape that most obviously does not need memory
/// held all of it — on a hundred-million-row scan, a hundred million rows resident to return ten.
/// `parallel_top_n` was already the mergeable top-N; only the driver was in the way.
#[test]
fn a_top_n_does_not_hold_the_relation_it_selects_from() {
    let _measuring = measuring();
    let plan: RelOp = serde_json::from_str(&format!(
        r#"{{"op":"sort","input":{},
             "keys":[{{"expr":{{"e":"col","name":"v"}},"descending":true,"nulls_first":false}}],
             "limit":10}}"#,
        computed_input()
    ))
    .unwrap();

    let mut peaks = Vec::new();
    for n in [1_000_000i64, 4_000_000] {
        let srcs = vec![facts(n, 0)];
        // The answer is checked as well as the peak: a top-N that keeps nothing is cheap and
        // wrong, and this test would otherwise be delighted by it.
        let mut rows = 0usize;
        let peak = peak_delta(|| {
            let out = std::hint::black_box(execute_streaming(&plan, &srcs, 0).unwrap());
            rows = out.iter().map(|b| b.num_rows()).sum();
        });
        assert_eq!(rows, 10, "a LIMIT 10 must return ten rows");
        peaks.push(peak);
    }
    assert!(
        peaks[1] < peaks[0] * 2,
        "a top-N's peak must not scale with its input: {} -> {} across a 4x growth",
        peaks[0],
        peaks[1]
    );
}

/// A whole-row `DISTINCT` over a low-cardinality relation holds its survivors, not its input.
///
/// `facts` repeats its key every 100 rows, so the distinct row count is fixed while the input
/// grows — which is the shape anyone writes a `DISTINCT` for, and the one where holding the
/// input was pure waste.
#[test]
fn a_reducing_distinct_does_not_hold_its_input() {
    let _measuring = measuring();
    let plan: RelOp = serde_json::from_str(&format!(
        r#"{{"op":"distinct","input":{},"keys":[],"order":[],"limit":null}}"#,
        // Only the repeating key, so the survivors are 100 rows however long the input is.
        r#"{"op":"project","input":{"op":"scan","source_id":0},"exprs":[
               {"expr":{"e":"binary","op":"add","left":{"e":"col","name":"k"},
                        "right":{"e":"lit","value":{"int":0}}},"alias":"k"}]}"#
    ))
    .unwrap();

    let mut peaks = Vec::new();
    for n in [1_000_000i64, 4_000_000] {
        let srcs = vec![facts(n, 0)];
        let mut rows = 0usize;
        let peak = peak_delta(|| {
            let out = std::hint::black_box(execute_streaming(&plan, &srcs, 0).unwrap());
            rows = out.iter().map(|b| b.num_rows()).sum();
        });
        assert_eq!(rows, 100, "the key repeats every 100 rows");
        peaks.push(peak);
    }
    assert!(
        peaks[1] < peaks[0] * 2,
        "a reducing DISTINCT's peak must not scale with its input: {} -> {} across a 4x growth",
        peaks[0],
        peaks[1]
    );
}

/// `UNION ALL` yields its branches in turn and holds none of them.
///
/// Its result *is* the concatenation, so a caller collecting it holds that much either way —
/// what this measures is that the operator itself adds nothing on top, which the deferred path
/// did: it ran the whole subtree on the materializing oracle, holding every branch at once
/// beneath its own copy of the concatenation.
#[test]
fn a_union_all_does_not_hold_its_branches() {
    let _measuring = measuring();
    let branch = |src: usize| {
        format!(
            r#"{{"op":"project","input":{{"op":"scan","source_id":{src}}},"exprs":[
                   {{"expr":{{"e":"col","name":"k"}},"alias":"k"}},
                   {{"expr":{{"e":"binary","op":"mul","left":{{"e":"col","name":"v"}},
                            "right":{{"e":"lit","value":{{"int":2}}}}}},"alias":"v"}}]}}"#
        )
    };
    // A `LIMIT` on top, so the *result* is bounded and the only thing a peak can measure is
    // what the union itself holds. Streaming, the limit also stops the pull.
    let plan: RelOp = serde_json::from_str(&format!(
        r#"{{"op":"limit","input":{{"op":"union","inputs":[{},{}],"distinct":false}},
             "n":5,"offset":0}}"#,
        branch(0),
        branch(1)
    ))
    .unwrap();

    let mut peaks = Vec::new();
    for n in [1_000_000i64, 4_000_000] {
        let srcs = vec![facts(n, 0), facts(n, 0)];
        let mut rows = 0usize;
        let peak = peak_delta(|| {
            let out = std::hint::black_box(execute_streaming(&plan, &srcs, 0).unwrap());
            rows = out.iter().map(|b| b.num_rows()).sum();
        });
        assert_eq!(rows, 5, "the limit above the union must still bind");
        peaks.push(peak);
    }
    assert!(
        peaks[1] < peaks[0] * 2,
        "a UNION ALL under a LIMIT must not hold its branches: {} -> {} across a 4x growth",
        peaks[0],
        peaks[1]
    );
}

/// `UNION DISTINCT` over branches that reduce holds its survivors, not its branches.
///
/// It is the streamed concat composed with the streamed dedup, so it inherits both bounds: the
/// branches are never held, and the dedup's state is the distinct rows. Before, every branch was
/// drained in full and only then deduped.
#[test]
fn a_union_distinct_does_not_hold_its_branches() {
    let _measuring = measuring();
    // Project to the repeating key alone, so the answer is 100 rows however long the branches are.
    let branch = |src: usize| {
        format!(
            r#"{{"op":"project","input":{{"op":"scan","source_id":{src}}},"exprs":[
                   {{"expr":{{"e":"binary","op":"add","left":{{"e":"col","name":"k"}},
                            "right":{{"e":"lit","value":{{"int":0}}}}}},"alias":"k"}}]}}"#
        )
    };
    let plan: RelOp = serde_json::from_str(&format!(
        r#"{{"op":"union","inputs":[{},{}],"distinct":true}}"#,
        branch(0),
        branch(1)
    ))
    .unwrap();

    let mut peaks = Vec::new();
    for n in [1_000_000i64, 4_000_000] {
        let srcs = vec![facts(n, 0), facts(n, 0)];
        let mut rows = 0usize;
        let peak = peak_delta(|| {
            let out = std::hint::black_box(execute_streaming(&plan, &srcs, 0).unwrap());
            rows = out.iter().map(|b| b.num_rows()).sum();
        });
        assert_eq!(rows, 100, "both branches carry the same 100 distinct keys");
        peaks.push(peak);
    }
    assert!(
        peaks[1] < peaks[0] * 2,
        "a reducing UNION DISTINCT must not hold its branches: {} -> {} across a 4x growth",
        peaks[0],
        peaks[1]
    );
}

/// A fixed-count `SAMPLE n` holds the `n` rows it is keeping, not the relation it draws from.
///
/// `ds.sample(n=1000)` over a table too large to collect is precisely what a fixed-count sample
/// is for, and it was the shape that held the whole table to return a thousand rows.
#[test]
fn a_fixed_count_sample_does_not_hold_the_relation_it_draws_from() {
    let _measuring = measuring();
    let plan: RelOp = serde_json::from_str(&format!(
        r#"{{"op":"sample","input":{},"fraction":1.0,"seed":7,"n":100}}"#,
        computed_input()
    ))
    .unwrap();

    let mut peaks = Vec::new();
    for n in [1_000_000i64, 4_000_000] {
        let srcs = vec![facts(n, 0)];
        let mut rows = 0usize;
        let peak = peak_delta(|| {
            let out = std::hint::black_box(execute_streaming(&plan, &srcs, 0).unwrap());
            rows = out.iter().map(|b| b.num_rows()).sum();
        });
        assert_eq!(rows, 100, "a SAMPLE 100 must return a hundred rows");
        peaks.push(peak);
    }
    assert!(
        peaks[1] < peaks[0] * 2,
        "a fixed-count sample's peak must not scale with its input: {} -> {} across a 4x growth",
        peaks[0],
        peaks[1]
    );
}

/// A dimension whose *build side* is large but whose *join output* is not.
///
/// Keys `0..rows`, of which only `0..100` match `facts`. The unmatched remainder inflates the
/// table the join must hold without inflating the result it produces, which is what separates
/// "the build sides do not fit" from "the query returns too much".
fn fat_dim(name: &str, rows: i64) -> Vec<RecordBatch> {
    let k: ArrayRef = Arc::new(Int64Array::from((0..rows).collect::<Vec<_>>()));
    let d: ArrayRef = Arc::new(Int64Array::from(
        (0..rows).map(|i| i * 10).collect::<Vec<_>>(),
    ));
    vec![RecordBatch::try_from_iter(vec![("k", k), (name, d)]).unwrap()]
}

/// Build sides that each fit the envelope but together do not must give way to the spilling
/// executor.
///
/// `prebuild_joins` prepares **every** join's build side before the probe pipeline runs, and
/// they all stay resident for the whole query -- so the quantity that has to fit is their sum.
/// Checking each side against the full envelope on its own is the bug this pins: three sides
/// at 40% of the envelope each are individually fine and collectively 120% of it, so no check
/// fires and the process is killed at exactly the point the handoff exists to prevent. The
/// per-side view is seductive because a build side really is "small by construction" relative
/// to the relation it broadcasts -- it is small *per join*, and there are several joins.
///
/// TPC-H q9 at sf100 is the real instance: five build sides, an auto-sensed 82 GB envelope,
/// and a 184 GB machine that the query was killed on. The same query under a 40 GB cap -- low
/// enough that one side alone crosses it -- spilled and completed, which is what identified
/// the sum rather than any single side as the thing going unchecked.
#[test]
fn build_sides_that_only_together_exceed_the_envelope_give_way() {
    let _measuring = measuring();
    let plan = deep_plan(2);
    let rows = 1_000_000;
    let one = fat_dim("a", rows)[0].get_array_memory_size();
    let srcs = vec![
        facts(100_000, 2),
        fat_dim("a", rows),
        fat_dim("b", rows),
        fat_dim("c", rows),
    ];

    // Sized off the measured side rather than a literal, so allocator padding cannot quietly
    // move a side across the line and make this test pass for the wrong reason. One side is
    // half the envelope; three are one and a half times it.
    let budget = one * 2;

    let err = execute_streaming(&plan, &srcs, budget).expect_err(
        "three {one}-byte build sides under a {budget}-byte envelope must give way: they are \
         all resident at once, so the sum is what has to fit",
    );
    assert!(
        matches!(err, bc_interp::InterpError::MemoryBudgetExceeded { .. }),
        "expected MemoryBudgetExceeded so the caller re-runs on the executor that spills, \
         got: {err:?}"
    );

    // The converse, so this cannot pass by refusing everything: the same plan under an
    // envelope that genuinely fits all three sides must run.
    execute_streaming(&plan, &srcs, one * 8)
        .expect("build sides that fit must still run on the streaming executor");
}
