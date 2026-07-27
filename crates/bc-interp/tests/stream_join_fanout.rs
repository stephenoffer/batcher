//! A join that *multiplies* rows must not put its whole result in one morsel.
//!
//! This lives in its own test binary rather than beside `stream_memory.rs` on purpose. That file
//! installs a **process-wide counting global allocator** to measure peak live heap, and its
//! measurement is only valid while nothing else in the process is allocating. A second test in
//! the same binary runs concurrently by default and corrupts those counters — which is exactly
//! what happened when this test was first written there, turning `peak - base` into an
//! arithmetic underflow. Separate binary, separate process, no interference.

use arrow::array::{ArrayRef, Int64Array, RecordBatch};
use bc_interp::{execute, execute_streaming};
use bc_ir::RelOp;
use std::sync::Arc;

/// A join that *multiplies* rows must not put its whole result in one morsel.
///
/// The streaming executor's constant-memory property rests on "one morsel in flight per
/// operator", and for row-preserving and row-reducing operators that follows from the pipeline
/// shape. A join is neither: against a build side holding `f` rows per key, one 16,384-row probe
/// morsel yields `16,384 x f` output rows, and emitting them as a single `RecordBatch` made peak
/// memory scale with the *product* of the inputs. Measured before the fix on a cartesian join
/// over two 20,000-row tables: 13.1 GB RSS, from ~500 KB of input.
///
/// This is not a cartesian-only concern, which is why the fixture below is an ordinary equi-join
/// on a **skewed key**: every probe row matches all `fanout` build rows. That is what a real
/// query hits when one key dominates a dimension table.
///
/// The assertion is on the morsel *size*, not on total memory: total memory still contains the
/// join index buffers, which are the next thing to bound. What must never come back is a single
/// output batch proportional to `probe_rows x fanout`.
#[test]
fn a_high_fanout_join_emits_bounded_morsels() {
    let probe_rows = 4_096i64;
    let fanout = 500i64;

    // Every row on both sides carries the same key, so the join is `probe_rows x fanout`.
    let one_key = |n: i64, name: &str| -> Vec<RecordBatch> {
        let k: ArrayRef = Arc::new(Int64Array::from(vec![7i64; n as usize]));
        let v: ArrayRef = Arc::new(Int64Array::from((0..n).collect::<Vec<_>>()));
        vec![RecordBatch::try_from_iter(vec![("k", k), (name, v)]).unwrap()]
    };
    let sources = vec![one_key(probe_rows, "v"), one_key(fanout, "w")];

    let json = r#"{"op":"hash_join",
        "left":{"op":"scan","source_id":0},
        "right":{"op":"scan","source_id":1},
        "left_keys":["k"],"right_keys":["k"],"join_type":"inner",
        "output":[{"side":"left","name":"v","alias":"v"},
                  {"side":"right","name":"w","alias":"w"}],
        "strategy":"hash"}"#;
    let plan: RelOp = serde_json::from_str(json).unwrap();

    let streamed = execute_streaming(&plan, &sources, 0).expect("streaming join");
    let total: usize = streamed.iter().map(|b| b.num_rows()).sum();
    let expected = (probe_rows * fanout) as usize;
    assert_eq!(total, expected, "the join must still produce every row");

    let largest = streamed.iter().map(|b| b.num_rows()).max().unwrap_or(0);
    assert!(
        largest <= bc_arrow::DEFAULT_MORSEL_ROWS,
        "a {expected}-row join result came back in a {largest}-row morsel; the output must be \
         morselized, or peak memory scales with probe_rows x fanout"
    );
    assert!(
        streamed.len() > 1,
        "a {expected}-row result must span more than one morsel"
    );

    // And it must still equal the materializing oracle, row for row.
    let oracle = execute(&plan, &sources).expect("materializing join");
    let rows = |bs: &[RecordBatch]| -> Vec<(i64, i64)> {
        let mut out = Vec::new();
        for b in bs {
            let v = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let w = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..b.num_rows() {
                out.push((v.value(i), w.value(i)));
            }
        }
        out.sort_unstable();
        out
    };
    assert_eq!(rows(&streamed), rows(&oracle));
}
