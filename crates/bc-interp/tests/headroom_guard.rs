//! The machine running short of memory moves a budgeted query onto a path that spills, and
//! never changes its answer.
//!
//! The executors' memory checks compare estimates with a budget; `bc_resource::headroom` adds
//! the one figure the kernel acts on, what is still available. This binary forces that reading
//! low (`force_available`), which is process-global -- so it lives in its own test binary and
//! its tests take one lock, rather than leaking a "low memory" reading into unrelated tests.
use std::sync::{Arc, Mutex};

use arrow::array::{Array, ArrayRef, Int64Array};
use arrow::record_batch::RecordBatch;
use bc_interp::{
    execute, execute_parallel_with_metrics, execute_streaming_parallel_or_hand_off, ExecOptions,
    InterpError,
};
use bc_ir::RelOp;
use bc_resource::headroom;

static FORCING: Mutex<()> = Mutex::new(());

/// Hold the lock and force the reading; the guard restores the real reading when dropped.
struct Low(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);
impl Low {
    fn force(available: u64) -> Self {
        let g = FORCING.lock().unwrap_or_else(|e| e.into_inner());
        headroom::force_available(Some(available));
        Low(g)
    }
}
impl Drop for Low {
    fn drop(&mut self) {
        headroom::force_available(None);
    }
}

const PLAN: &str = r#"{"op":"aggregate","input":{"op":"scan","source_id":0},
    "group_keys":[{"expr":{"e":"col","name":"k"},"alias":"k"}],
    "aggregates":[{"func":"sum","input":{"e":"col","name":"v"},"alias":"s"}]}"#;

fn facts() -> Vec<RecordBatch> {
    (0..8)
        .map(|b| {
            let ks: Vec<i64> = (0..20_000).map(|i| (b * 20_000 + i) % 5_000).collect();
            let vs: Vec<i64> = (0..20_000).map(|i| b * 20_000 + i).collect();
            RecordBatch::try_from_iter(vec![
                ("k", Arc::new(Int64Array::from(ks)) as ArrayRef),
                ("v", Arc::new(Int64Array::from(vs)) as ArrayRef),
            ])
            .unwrap()
        })
        .collect()
}

fn sorted(batches: &[RecordBatch]) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    for b in batches {
        let k = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let s = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        out.extend((0..b.num_rows()).map(|i| (k.value(i), s.value(i))));
    }
    out.sort_unstable();
    out
}

fn budgeted() -> ExecOptions {
    let cfg = bc_ir::EngineConfig {
        memory_budget_bytes: 1 << 40, // far above anything this test holds
        spill_dir: Some(std::env::temp_dir().to_string_lossy().into_owned()),
        ..bc_ir::EngineConfig::default()
    };
    ExecOptions::default().with_engine_config(&cfg)
}

#[test]
fn the_streaming_executor_hands_off_when_memory_runs_short() {
    let plan = RelOp::from_json(PLAN).unwrap();
    let src = [facts()];
    // Positive control: with memory to spare, a budget this size never hands off.
    execute_streaming_parallel_or_hand_off(&plan, &src, 4, 1 << 40, false, None, None)
        .expect("ample memory: runs streaming");
    let _low = Low::force(0);
    let err = execute_streaming_parallel_or_hand_off(&plan, &src, 4, 1 << 40, false, None, None)
        .expect_err("no memory left: must hand off");
    assert!(
        matches!(err, InterpError::MemoryBudgetExceeded { reason, .. } if reason.contains("available memory")),
        "{err:?}"
    );
    // An unbudgeted query is not guarded: no budget, nothing to hand off to.
    execute_streaming_parallel_or_hand_off(&plan, &src, 4, 0, false, None, None)
        .expect("no budget: unaffected by the guard");
}

#[test]
fn the_materializing_executor_spills_when_memory_runs_short_and_agrees() {
    let plan = RelOp::from_json(PLAN).unwrap();
    let src = [facts()];
    let want = sorted(&execute(&plan, &src).expect("oracle"));
    let opts = budgeted();
    let (roomy, m) = execute_parallel_with_metrics(&plan, &src, &opts).expect("ample memory");
    assert!(
        !m.ops.iter().any(|o| o.spilled),
        "positive control: no spill with memory to spare"
    );
    assert_eq!(sorted(&roomy), want);
    let _low = Low::force(0);
    let (short, m) = execute_parallel_with_metrics(&plan, &src, &opts).expect("short of memory");
    assert!(
        m.ops.iter().any(|o| o.spilled),
        "the aggregate must spill: {:?}",
        m.ops
    );
    assert_eq!(sorted(&short), want, "spilling changed the answer");
}

#[test]
fn the_pool_refuses_reservations_when_memory_runs_short() {
    let pool = bc_resource::MemoryPool::new(1 << 40);
    pool.try_reserve_bytes(1 << 20)
        .expect("ample memory admits");
    pool.release_bytes(1 << 20);
    let _low = Low::force(0);
    assert!(pool.try_reserve_bytes(1 << 20).is_err());
    assert!(
        pool.try_reserve_bytes(0).is_ok(),
        "a zero-byte reservation holds nothing"
    );
}
