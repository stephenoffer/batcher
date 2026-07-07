"""Conductor adaptive-tuning: activate the learned decisions and close the feedback loops.

The façade over `decisions.py`; see it for the contract. Every export tunes performance and
scheduling only — a first run over a cold hub is byte-for-byte the pre-tuning behavior.
"""

from __future__ import annotations

from batcher.api.tuning.decisions import (
    auto_num_partitions,
    distributed_grant,
    learned_num_workers,
    learned_output_rows,
    learned_partition_seed,
    record_distributed,
    record_join_outcomes,
    record_run_feedback,
    record_shuffle_outcome,
    spill_compression_scope,
    total_source_rows,
)

__all__ = [
    "auto_num_partitions",
    "distributed_grant",
    "learned_num_workers",
    "learned_output_rows",
    "learned_partition_seed",
    "record_distributed",
    "record_join_outcomes",
    "record_run_feedback",
    "record_shuffle_outcome",
    "spill_compression_scope",
    "total_source_rows",
]
