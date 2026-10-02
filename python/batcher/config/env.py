"""Every `BATCHER_*` environment variable the engine reads, declared in one place.

`config.py` is the documented configuration contract: typed, validated, profile-aware,
serializable, and rendered into the docs. Beside it sit the env-only knobs, read with
`os.environ.get(...)` at their point of use across `io`, `dist`, `core` and `_internal`. They
are deliberately env-only — last-resort tuning knobs an operator reaches for on a running
cluster, not things a user sets in a `Config`.

A knob read at its point of use is invisible to `Config`, absent from the docs, unvalidated,
and impossible to enumerate, and two knobs can disagree about a default for the same concept
with nothing to say so. So this module **declares** them, and `tests/unit/test_env_knobs.py`
fails when the code reads a `BATCHER_*` variable that is not declared here, or declares one
nothing reads. The declaration leaves each call site's default where it is.

Adding a knob means adding a line here. If a setting deserves validation, a profile, or a
place in the docs, it does not belong in this file at all — it belongs in `Config`.

It also holds the one *reading* of each kind of knob. `truthy` / `env_flag` is the only
spelling of "is this string yes", so a diagnostic flag set to ``true`` cannot be silently off
because one reader accepted only ``"1"``. `env_int` / `env_float` parse numbers with a warning
and a fallback, so a typo in a knob read at import cannot break `import batcher`.
"""

from __future__ import annotations

import logging
import os
from typing import Final, TypeVar

__all__ = [
    "ENV_KNOBS",
    "FALSE_TOKENS",
    "TRUE_TOKENS",
    "env_flag",
    "env_float",
    "env_int",
    "falsy",
    "truthy",
]

_log = logging.getLogger("batcher.config.env")
_N = TypeVar("_N", int, float)

#: `BATCHER_*` variable -> what it controls. Grouped by the subsystem that reads it.
ENV_KNOBS: Final[dict[str, str]] = {
    # --- process / bootstrap -------------------------------------------------------
    "BATCHER_CONFIG_FILE": "path to a TOML/JSON config loaded at import",
    "BATCHER_HOME": "root for engine-owned state (event logs, scratch); defaults under XDG",
    "BATCHER_DEADLINE_EPOCH_S": "wall-clock deadline the query budget counts down to",
    "BATCHER_DEADLINE_SECONDS": "lease length in seconds, counted from process start",
    # --- site detection -------------------------------------------------------------
    "BATCHER_SCHEDULER": "force the detected batch scheduler (slurm, pbs, lsf, ...)",
    "BATCHER_SCRATCH_DIR": "force the node-local scratch directory spill uses",
    "BATCHER_PROVIDER": "force the detected cloud provider",
    "BATCHER_NODE_NAME": "name this node reports, ahead of the orchestrator's",
    "BATCHER_SPOT": "declare this node preemptible, selecting the spot profile",
    "BATCHER_AUTOSCALE": "declare the cluster autoscaling (truthy) or fixed (falsy)",
    "BATCHER_RAY_CLUSTER": "mark the process as running on a managed Ray cluster",
    "BATCHER_METADATA_URI": "durable location the spot profile moves learned metadata to",
    "BATCHER_MPS_CLIENTS": "CUDA MPS clients sharing one device, for feeder-CPU sizing",
    # --- security -------------------------------------------------------------------
    "BATCHER_SECRET_COMMAND": "helper command that resolves a secret reference",
    "BATCHER_REQUIRE_KEY_REFS": "refuse literal encryption keys; accept only key references",
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
    "BATCHER_SECRET_TIMEOUT_SECONDS": "per-request timeout for an HTTP-answered key store",
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
    "BATCHER_SCAN_PREFETCH": "concurrent reads a scan task keeps in flight",
    "BATCHER_SCAN_CACHE_BYTES": "per-worker scan cache size",
    "BATCHER_SCAN_CACHE_FRACTION": "scan cache as a fraction of worker memory",
    "BATCHER_BATCH_READAHEAD": "batches read ahead per scan task",
    "BATCHER_FRAGMENT_READAHEAD": "fragments read ahead per scan task",
    "BATCHER_NATIVE_READER": "force on/off the native Parquet reader",
    "BATCHER_NATIVE_RG_WINDOW": "row groups read per native-reader window",
    "BATCHER_NATIVE_READ_BUDGET_BYTES": "projected bytes one native scan keeps in flight",
    "BATCHER_ALIGNED_UNIT_BYTES": "target input bytes per aligned (co-partitioned) unit",
    "BATCHER_ALIGNED_UNIT_CPUS": "cores one aligned unit runs on",
    "BATCHER_ALIGNED_BROADCAST_BYTES": "largest broadcast an aligned cut holds on every node",
    "BATCHER_ALIGNED_NESTED_BYTES": "largest broadcast nested inside another aligned broadcast",
    "BATCHER_ALIGNED_DRIVER_READ_BYTES": "largest broadcast input the driver reads itself",
    "BATCHER_ALIGNED_RESULT_BYTES": "unit results gathered on the driver before declining",
    "BATCHER_ALIGNED_LOCAL_BYTES": "broadcast size past which each node evaluates it itself",
    "BATCHER_ALIGNED_LOCAL_FILTERED_BYTES": "the same bound, for a filtered broadcast",
    "BATCHER_FOLD_CHUNK_BYTES": "bytes per chunk in the distributed fold",
    "BATCHER_MIN_TASK_CPU": "floor on the CPU a map task reserves",
    "BATCHER_MAP_COMPUTE_WEIGHT": "compute weight used to size map tasks",
    "BATCHER_TARGET_TASK_CPUS": "cores a map task should be, capping the row-based fan-out",
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


#: Strings that mean "yes" to a boolean env var or connection option, and their negations.
#:
#: The single set every boolean reader uses. A reader with its own spelling drifts: one that
#: accepts only ``"1"`` turns `BATCHER_VERIFY_EXPR_MATCHES=true` into a silent no-op, the
#: worst failure for a *diagnostic* flag, since the operator believes verification is on.
TRUE_TOKENS: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
FALSE_TOKENS: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


def truthy(raw: str | None) -> bool:
    """Whether a string spells "yes" — the one reading of a boolean knob or option.

    Case- and whitespace-insensitive. Anything unrecognized is `False`, including `None`, so
    an unset variable and a variable set to nonsense agree: a knob nobody deliberately turned
    on stays off.

    Args:
        raw: The value as the environment or the option map supplied it.

    Returns:
        `True` when `raw` names one of `TRUE_TOKENS`.

    Examples:
        .. doctest::

            >>> from batcher.config.env import truthy
            >>> truthy("YES"), truthy(" on "), truthy("0"), truthy(None)
            (True, True, False, False)
    """
    return raw is not None and raw.strip().lower() in TRUE_TOKENS


def falsy(raw: str | None) -> bool:
    """Whether a string spells "no" *explicitly*, as opposed to being unset or unrecognized.

    Not the negation of `truthy`: the three-way distinction is what a `bool | str` config field
    needs. ``runtime_bloom_join = "auto"`` is neither true nor false and must stay the string
    it is, so a caller asks both questions and keeps the value when both say no.

    Args:
        raw: The value as the environment or the option map supplied it.

    Returns:
        `True` when `raw` names one of `FALSE_TOKENS`.

    Examples:
        .. doctest::

            >>> from batcher.config.env import falsy, truthy
            >>> falsy("off"), falsy("auto"), truthy("auto")
            (True, False, False)
    """
    return raw is not None and raw.strip().lower() in FALSE_TOKENS


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable, one way across the whole engine.

    Args:
        name: The variable's name. It must be declared in `ENV_KNOBS`, which
            `tests/unit/test_env_knobs.py` checks.
        default: What an *unset* variable means. A variable that is set but unrecognized is
            `False` regardless, per `truthy`.

    Returns:
        The flag's value.

    Examples:
        .. doctest::

            >>> import os
            >>> from batcher.config.env import env_flag
            >>> os.environ["BATCHER_VERIFY_EXPR_MATCHES"] = "yes"
            >>> env_flag("BATCHER_VERIFY_EXPR_MATCHES")
            True
            >>> del os.environ["BATCHER_VERIFY_EXPR_MATCHES"]
            >>> env_flag("BATCHER_VERIFY_EXPR_MATCHES", default=True)
            True
    """
    raw = os.environ.get(name)
    return default if raw is None else truthy(raw)


def env_int(name: str, default: int, *, floor: int | None = None) -> int:
    """Read an integer environment variable, falling back on `default` when it is malformed.

    Most knobs are read into module constants at import, so a strict `int(...)` there turned
    one typo (`BATCHER_REMOTE_READ_CONCURRENCY=32x`) into an `import batcher` that fails with
    a bare `ValueError` naming neither the variable nor the option. A tuning knob is not worth
    refusing to start over: a malformed value logs a warning naming the variable and the
    default takes its place.

    Args:
        name: The variable's name. It must be declared in `ENV_KNOBS`.
        default: The value for an unset, empty, or malformed variable.
        floor: A lower bound applied to the result, or None for none.

    Returns:
        The parsed value, raised to `floor`.

    Examples:
        .. doctest::

            >>> import os
            >>> from batcher.config.env import env_int
            >>> os.environ["BATCHER_IO_THREADS"] = "32x"
            >>> env_int("BATCHER_IO_THREADS", 64)
            64
            >>> os.environ["BATCHER_IO_THREADS"] = "4"
            >>> env_int("BATCHER_IO_THREADS", 64, floor=8)
            8
            >>> del os.environ["BATCHER_IO_THREADS"]
    """
    value = _env_number(name, default, int)
    return value if floor is None else max(floor, value)


def env_float(name: str, default: float, *, floor: float | None = None) -> float:
    """Read a float environment variable, falling back on `default` when it is malformed.

    The float twin of `env_int`, with the same warn-and-fall-back contract.

    Args:
        name: The variable's name. It must be declared in `ENV_KNOBS`.
        default: The value for an unset, empty, or malformed variable.
        floor: A lower bound applied to the result, or None for none.

    Returns:
        The parsed value, raised to `floor`.

    Examples:
        .. doctest::

            >>> import os
            >>> from batcher.config.env import env_float
            >>> os.environ["BATCHER_READ_RETRY_BACKOFF_S"] = "-1"
            >>> env_float("BATCHER_READ_RETRY_BACKOFF_S", 0.5, floor=0.0)
            0.0
            >>> del os.environ["BATCHER_READ_RETRY_BACKOFF_S"]
    """
    value = _env_number(name, default, float)
    return value if floor is None else max(floor, value)


def _env_number(name: str, default: _N, parse: type[_N]) -> _N:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return parse(raw)
    except ValueError:
        _log.warning("%s=%r is not a valid %s; using %r", name, raw, parse.__name__, default)
        return default
