//! Spill scratch abandoned by a killed process must be reclaimed, and nothing else may be.
//!
//! A store removes its own directory on drop, which covers success, error, and panic. It
//! does not cover `SIGKILL` — and the process most likely to be `SIGKILL`ed is the one
//! spilling, because that is the process the kernel OOM killer picks. Nothing runs on that
//! path, so the scratch survives *on the spill filesystem*: the next query has less room,
//! spills harder, and is likelier to be killed in turn. Left alone this ratchets a node into
//! a state where every large query fails for space while the data that filled the disk
//! belongs to no process at all.
//!
//! The sweep is therefore as much about what it does *not* delete. A concurrently spilling
//! sibling shares the root — that is why the pid is in the name — and deleting its scratch
//! would corrupt a running query rather than merely fail one.

#![cfg(target_os = "linux")]

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use bc_runtime::agg::spill::{DiskSpillStore, SpillStore};

/// A pid that is not running. Above `/proc/sys/kernel/pid_max` on any normal Linux, so it
/// cannot be allocated to a live process while the test runs.
const DEAD_PID: u32 = 4_194_303;

/// Create `root/name` holding one file, standing in for scratch a store left behind.
fn plant(root: &std::path::Path, name: &str) -> std::path::PathBuf {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).expect("plant dir");
    std::fs::write(dir.join("part-0.arrow"), b"spilled rows").expect("plant file");
    dir
}

#[test]
fn scratch_from_a_dead_process_is_reclaimed_and_nothing_else_is() {
    // A root unique to this test, so it is this process's first sweep of it (the sweep runs
    // once per root per process).
    let root =
        std::env::temp_dir().join(format!("bc-spill-sweep-{}-{}", std::process::id(), line!()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("root");

    let orphan = plant(&root, &format!("bc-spill-{DEAD_PID}-0"));
    let orphan2 = plant(&root, &format!("bc-spill-{DEAD_PID}-17"));
    // A live sibling: this very process. Its scratch is in use and must survive.
    let live = plant(&root, &format!("bc-spill-{}-999999", std::process::id()));
    // Names the sweep did not create. It must not touch them, whatever they look like.
    let foreign = plant(&root, "someone-elses-data");
    let near_miss = plant(&root, "bc-spill-not-a-pid-0");
    let near_miss2 = plant(&root, &format!("bc-spill-{DEAD_PID}-not-a-seq"));

    // Creating a store on this root triggers the sweep.
    let mut store = DiskSpillStore::new(root.clone(), 1).expect("spill store");
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1_i64]))]).expect("batch");
    store.append(0, &batch).expect("append");

    assert!(
        !orphan.exists(),
        "scratch from a dead pid was not reclaimed"
    );
    assert!(
        !orphan2.exists(),
        "only one orphan of a dead pid was reclaimed"
    );
    assert!(
        live.exists(),
        "the sweep deleted scratch belonging to a LIVE process — that corrupts a running \
         query rather than merely failing one"
    );
    assert!(foreign.exists(), "the sweep deleted an unrelated directory");
    assert!(
        near_miss.exists() && near_miss2.exists(),
        "the sweep deleted a directory whose name only resembles its own"
    );

    // And the store it swept for still works.
    assert_eq!(store.read(0).expect("read back").len(), 1);

    drop(store);
    let _ = std::fs::remove_dir_all(&root);
}
