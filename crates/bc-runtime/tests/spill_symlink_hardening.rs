//! A planted symlink in the shared spill root must not redirect the query's rows.
//!
//! Spill scratch is a *shared* path — `/tmp` by default, or one spill directory serving
//! several worker processes on a node. Any local user who can write there can pre-create the
//! name a spill store is about to claim, pointing it at a directory they can read. With
//! `create_dir_all` that succeeds silently: the store proceeds to write every spilled row
//! through the attacker's symlink, and the owner-only mode it applies afterwards lands on a
//! path the attacker still controls. The query reports nothing, because from its side
//! everything worked.
//!
//! Claiming the leaf with `create_dir` — which fails on an existing name, symlink included —
//! is what closes it. The retry that follows a clash is why this stays a hardening step and
//! not a new failure mode: a stale directory from a reused pid must not fail the query.
//!
//! This file has exactly one test on purpose. It reasons about the *next* scratch names the
//! process-wide counter will hand out, so a second test creating stores in parallel would
//! make it flaky.

#![cfg(unix)]

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use bc_runtime::agg::spill::{DiskSpillStore, SpillStore};

/// The single scratch directory a store created under `root`.
fn sole_scratch_dir(root: &std::path::Path) -> std::path::PathBuf {
    let mut entries: Vec<_> = std::fs::read_dir(root)
        .expect("read spill root")
        .flatten()
        .map(|e| e.path())
        .collect();
    assert_eq!(entries.len(), 1, "expected one scratch dir in {root:?}");
    entries.pop().expect("one entry")
}

/// The counter suffix of a `bc-spill-{pid}-{seq}` directory name.
fn seq_of(dir: &std::path::Path) -> u64 {
    dir.file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.rsplit('-').next())
        .and_then(|n| n.parse().ok())
        .expect("scratch dir name ends in its counter")
}

#[test]
fn a_planted_symlink_cannot_capture_spilled_rows() {
    let base = std::env::temp_dir().join(format!("bc-spill-symlink-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let root = base.join("root");
    let attacker = base.join("attacker");
    std::fs::create_dir_all(&root).expect("root");
    std::fs::create_dir_all(&attacker).expect("attacker dir");

    // Learn which counter value the next store will use, by spending one.
    let probe = DiskSpillStore::new(root.clone(), 1).expect("probe store");
    let next_seq = seq_of(&sole_scratch_dir(&root)) + 1;
    drop(probe);

    // Plant symlinks over the next few names the counter will hand out, all pointing into a
    // directory the "attacker" can read.
    let pid = std::process::id();
    for seq in next_seq..next_seq + 3 {
        std::os::unix::fs::symlink(&attacker, root.join(format!("bc-spill-{pid}-{seq}")))
            .expect("plant symlink");
    }

    // The store must still succeed — a name clash is a stale directory as often as an attack,
    // and failing the query over it would be its own reliability bug.
    let mut store = DiskSpillStore::new(root.clone(), 1).expect("store must skip planted names");
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![42_i64]))])
        .expect("batch");
    store.append(0, &batch).expect("append");
    store.close_partition(0).expect("close");

    // ... and not one spilled byte may have gone through the planted path.
    let leaked: Vec<_> = std::fs::read_dir(&attacker)
        .expect("read attacker dir")
        .flatten()
        .map(|e| e.file_name())
        .collect();
    assert!(
        leaked.is_empty(),
        "spilled rows were written through a planted symlink into a directory the query \
         does not own: {leaked:?}"
    );
    assert_eq!(store.read(0).expect("read back").len(), 1);

    let _ = std::fs::remove_dir_all(&base);
}
