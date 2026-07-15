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
//!   1000000       198.4 MB       3.4 MB   (58x)
//!   2000000       396.7 MB       3.3 MB   (119x)
//!   4000000       793.5 MB       3.3 MB   (238x)
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

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

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

fn peak_of(f: impl FnOnce()) -> usize {
    PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
    f();
    PEAK.load(Ordering::Relaxed)
}

#[test]
fn peak_memory_streaming_vs_materializing() {
    let wide = 8;
    let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
    println!("\n  3 inner joins → aggregate (the q3/q4/q5 shape), {wide} payload cols\n");
    println!("      rows   materializing    streaming");

    let mut first = None;
    let mut last = None;
    for n in [1_000_000i64, 2_000_000, 4_000_000] {
        let srcs = vec![facts(n, wide), dim("a"), dim("b"), dim("c")];
        let p = deep_plan(wide);
        let base = LIVE.load(Ordering::Relaxed);
        let mat = peak_of(|| {
            std::hint::black_box(execute(&p, &srcs).unwrap());
        }) - base;
        let stream = peak_of(|| {
            std::hint::black_box(execute_streaming(&p, &srcs, 0).unwrap());
        }) - base;
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
