//! Spilled data must not be readable by other users on the node.
//!
//! A spill file holds the query's *actual rows* — the same bytes a caller may have taken
//! care to encrypt in flight — written to a shared scratch path such as `/tmp`. Created
//! with the default mode it is world-readable.
//!
//! This was measured, not assumed. Before the fix the directory came out `0700` (from
//! `restrict_to_owner`) and every `part-*.arrow` inside it came out **`0644`**. The
//! directory alone did protect them in practice, but `restrict_to_owner` is best-effort
//! and **ignores its own failure by design** — so on any filesystem where that `chmod`
//! fails, the whole protection evaporated silently and the rows were world-readable.
//!
//! Hence two independent assertions below. Either one alone is a single point of failure.

#![cfg(unix)]

use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use bc_runtime::agg::spill::{DiskSpillStore, SpillStore};

/// Owner-only: no permission bits set for group or other.
fn is_private(mode: u32) -> bool {
    mode & 0o077 == 0
}

#[test]
fn spilled_rows_are_not_readable_by_other_users() {
    let root = std::env::temp_dir().join(format!("bc-spill-perms-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("scratch root");

    let mut store = DiskSpillStore::new(root.clone(), 3).expect("spill store");
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1_i64, 2, 3]))])
        .expect("batch");
    for partition in 0..3 {
        store.append(partition, &batch).expect("append");
    }

    let mut checked_dirs = 0;
    let mut checked_files = 0;
    for entry in std::fs::read_dir(&root).expect("read root").flatten() {
        let dir_mode = entry.metadata().expect("dir metadata").permissions().mode();
        assert!(
            is_private(dir_mode),
            "spill directory {:?} is {:o}, readable beyond the owner",
            entry.file_name(),
            dir_mode & 0o777
        );
        checked_dirs += 1;
        for file in std::fs::read_dir(entry.path())
            .expect("read spill dir")
            .flatten()
        {
            let mode = file.metadata().expect("file metadata").permissions().mode();
            assert!(
                is_private(mode),
                "spill file {:?} is {:o}, readable beyond the owner — the query's rows are \
                 exposed to any local user the moment the directory mode is widened",
                file.file_name(),
                mode & 0o777
            );
            checked_files += 1;
        }
    }
    assert_eq!(checked_dirs, 1, "expected exactly one spill directory");
    assert_eq!(checked_files, 3, "expected one file per partition");

    let _ = std::fs::remove_dir_all(&root);
}
