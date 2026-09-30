"""The memory ledger: reserved, resident, and unaccounted bytes as separate figures.

Carbonite (layer 3). The pressure monitor folds every reading into one number — the
**maximum** of both pools' utilization and the process footprint — because a level has to
be one number. That is the right input for a decision and the wrong one for a diagnosis:
the maximum says *that* the box is full, not *who* filled it. A query whose pools are
nearly empty while the process is large is holding memory the pools never heard of
(pyarrow buffers allocated on the Python side, a UDF's model tensors, allocator slack), and
the fix for that is nothing like the fix for an envelope that is simply too small.

This module reports the figures the maximum collapses, side by side and unsummed. It only
reads; nothing here reserves, and nothing here changes a decision.
"""

from __future__ import annotations

import pyarrow as pa

from batcher.carbonite.memory import probe
from batcher.carbonite.memory.pool import current_process_pool, engine_pool_stats

__all__ = ["memory_ledger"]


def memory_ledger() -> dict[str, int | None]:
    """The reserved, resident, and unaccounted bytes of this process, right now.

    The two reservations describe the *same* running work from two sides — Carbonite's
    coarse per-query estimate and the engine's actual operator reservations — so the
    accounted figure is their **maximum**, never their sum (the rule
    `policies.admission` applies for the same reason). `unaccounted_bytes` is the process's
    resident set above that: memory no pool was asked for. It is a lower bound on foreign
    allocations rather than a measurement of them, because a reservation is taken *before*
    its state is allocated, so reserved-but-not-yet-resident bytes offset foreign ones.

    One foreign category *is* measured exactly: `pyarrow_allocated_bytes` is what pyarrow's
    default memory pool holds, the Arrow buffers built on the Python side (a `from_pydict`
    source, a UDF's output batch). Batches the engine produced were allocated by Rust and
    cross the FFI zero-copy, so they are not in that figure and are not double-counted.

    Examples:
        .. doctest::

            >>> from batcher.carbonite.memory.ledger import memory_ledger
            >>> sorted(memory_ledger())  # doctest: +NORMALIZE_WHITESPACE
            ['accounted_bytes', 'cgroup_unreclaimable_bytes', 'control_plane_reserved_bytes',
             'engine_reserved_bytes', 'pyarrow_allocated_bytes', 'resident_bytes',
             'unaccounted_bytes']

    Returns:
        `control_plane_reserved_bytes` and `engine_reserved_bytes` (`None` when that pool
        does not exist yet), `accounted_bytes` (their maximum), `pyarrow_allocated_bytes`
        (Python-side Arrow buffers, never reserved in either pool), `resident_bytes` (this
        process's RSS, `None` without a reader), `cgroup_unreclaimable_bytes` (the whole
        container's anonymous memory, `None` outside a cgroup), and `unaccounted_bytes`
        (`resident - accounted`, floored at 0; `None` when RSS is unreadable).
    """
    pool = current_process_pool()
    control = pool.used if pool is not None else None
    engine = engine_pool_stats()
    engine_used = int(engine["used_bytes"]) if engine is not None else None
    accounted = max(control or 0, engine_used or 0)
    resident = probe.process_rss_bytes()
    return {
        "control_plane_reserved_bytes": control,
        "engine_reserved_bytes": engine_used,
        "accounted_bytes": accounted,
        "pyarrow_allocated_bytes": pa.total_allocated_bytes(),
        "resident_bytes": resident,
        "cgroup_unreclaimable_bytes": probe.cgroup_current_bytes(),
        "unaccounted_bytes": None if resident is None else max(0, resident - accounted),
    }
