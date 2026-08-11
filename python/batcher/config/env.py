"""Every `BATCHER_*` environment variable the engine reads, declared in one place.

`config.py` is the documented configuration contract — typed, validated, profile-aware,
serializable, and rendered into the docs. Beside it a second configuration surface had grown:
**38 `BATCHER_*` variables read inline** with `os.environ.get(...)`, each with its own literal
default, spread across `io`, `dist`, `core` and `_internal`. They are deliberately env-only —
last-resort tuning knobs an operator reaches for on a running cluster, not things a user sets
in a `Config` — and that is a reasonable thing to want.

What is not reasonable is that they were undiscoverable. A knob read at its point of use is
invisible to `Config`, absent from the docs, unvalidated, and impossible to enumerate: the only
way to learn one existed was to find the line that read it. Two knobs could disagree about a
default for the same concept and nothing would say so.

This module does not change how any of them are read. It **declares** them, and
`tests/unit/test_env_knobs.py` fails when the code reads a `BATCHER_*` variable that is not
declared here, or declares one nothing reads. That makes the surface enumerable and keeps it
honest, without moving 38 call sites and their defaults — which would be a behavioral change
dressed up as tidying.

Adding a knob means adding a line here. If a setting deserves validation, a profile, or a
place in the docs, it does not belong in this file at all — it belongs in `Config`.
"""

from __future__ import annotations

from typing import Final

__all__ = ["ENV_KNOBS"]

#: `BATCHER_*` variable -> what it controls. Grouped by the subsystem that reads it.
ENV_KNOBS: Final[dict[str, str]] = {
    # --- process / bootstrap -------------------------------------------------------
    "BATCHER_CONFIG_FILE": "path to a TOML/JSON config loaded at import",
    "BATCHER_HOME": "root for engine-owned state (event logs, scratch); defaults under XDG",
    "BATCHER_DEADLINE_EPOCH_S": "wall-clock deadline the query budget counts down to",
    # --- IO: reads, footers, retries -----------------------------------------------
    "BATCHER_IO_THREADS": "filesystem thread-pool width",
    "BATCHER_FOOTER_CONCURRENCY": "parallel Parquet footer reads during planning",
    "BATCHER_MAX_FOOTER_PLAN_FILES": "cap on footers read to plan one scan",
    "BATCHER_READAHEAD_BYTES": "per-source read-ahead window",
    "BATCHER_REMOTE_READ_CONCURRENCY": "in-flight object-store range requests",
    "BATCHER_READ_RETRY_ATTEMPTS": "retries for a failed source read",
    "BATCHER_READ_RETRY_BACKOFF_S": "base backoff between source read retries",
    "BATCHER_REMOTE_WRITE_CONCURRENCY": "in-flight object-store PUTs during a write",
    "BATCHER_PARTITION_WARN_THRESHOLD": "partition directories in one shard before a warning",
    "BATCHER_MAX_WEIGHED_SPLITS": "split count past which an unknown weight is taken as 1",
    "BATCHER_WRITE_RETRY_ATTEMPTS": "retries for a failed sink write",
    "BATCHER_WRITE_RETRY_BACKOFF_S": "base backoff between sink write retries",
    "BATCHER_JSON_CHUNK_BYTES": "JSON reader chunk size",
    "BATCHER_NATIVE_STREAM_MAX_DEPTH": "native Parquet stream prefetch depth",
    "BATCHER_NATIVE_WINDOW_BYTES": "native Parquet decode window",
    "BATCHER_FOOTER_CACHE_ROW_GROUPS": "row groups held in the split planner's footer cache",
    "BATCHER_ORC_STRIPE_BYTES": "target bytes per ORC stripe read",
    # --- streaming file readers, bounded by payload bytes rather than rows ----------
    # A row in these formats is a whole image or array, so a row count bounds nothing that
    # matters; these are the byte budgets that actually cap a batch.
    "BATCHER_NUMPY_CHUNK_BYTES": "bytes per chunk streamed off a memory-mapped .npy",
    "BATCHER_WEBDATASET_BATCH_BYTES": "payload bytes per WebDataset batch",
    # --- distributed scan / scheduling ---------------------------------------------
    "BATCHER_SPLIT_TARGET_BYTES": "target bytes per scan split",
    "BATCHER_SCAN_PREFETCH": "scan-task prefetch depth",
    "BATCHER_SCAN_CACHE_BYTES": "per-worker scan cache size",
    "BATCHER_SCAN_CACHE_FRACTION": "scan cache as a fraction of worker memory",
    "BATCHER_BATCH_READAHEAD": "batches read ahead per scan task",
    "BATCHER_FRAGMENT_READAHEAD": "fragments read ahead per scan task",
    "BATCHER_NATIVE_READER": "force on/off the native Parquet reader",
    "BATCHER_NATIVE_RG_WINDOW": "row groups read per native-reader window",
    "BATCHER_FOLD_CHUNK_BYTES": "bytes per chunk in the distributed fold",
    "BATCHER_MIN_TASK_CPU": "floor on the CPU a map task reserves",
    "BATCHER_MAP_COMPUTE_WEIGHT": "compute weight used to size map tasks",
    "BATCHER_INFERENCE_CPU_WORKERS": "CPU-side workers feeding an inference stage",
    # --- shuffle transport ----------------------------------------------------------
    "BATCHER_ADVERTISE_HOST": "host a worker advertises for Flight connections",
    "BATCHER_SHUFFLE_PORT_RANGE": "port range the Flight server may bind",
    "BATCHER_SHUFFLE_TOKEN": "shared secret authenticating Flight peers",
    # --- UDF / streaming execution ---------------------------------------------------
    "BATCHER_CPU_STREAM_BATCH_BYTES": "target bytes per CPU streaming batch",
    "BATCHER_GPU_STREAM_BATCH_BYTES": "target bytes per GPU streaming batch",
    "BATCHER_GPU_STREAM_BATCH_ROWS": "target rows per GPU streaming batch",
    "BATCHER_GPU_STREAM_BATCH_MIN": "floor on GPU streaming batch size",
    "BATCHER_STREAM_PREFETCH_DEPTH": "batches prefetched ahead of a streaming UDF",
    "BATCHER_STREAM_MAX_PREFETCH_DEPTH": "cap on the streaming prefetch depth",
    "BATCHER_GPU_PIPELINE_DEPTH": "in-flight batches per GPU pipeline stage",
    "BATCHER_GPU_SOLO_PIPELINE_DEPTH": "pipeline depth when a stage owns the device alone",
    # --- optimizer diagnostics --------------------------------------------------------
    "BATCHER_VERIFY_EXPR_MATCHES": "cross-check the expression dispatch index (debug only)",
}
