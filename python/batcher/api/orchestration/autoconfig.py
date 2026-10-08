"""Zero-config resolution: sense the machine once, and pin it for the query's scope."""

from __future__ import annotations

import dataclasses
import functools
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeVar

import pyarrow as pa

from batcher.config import Config, active_config, config_context

if TYPE_CHECKING:
    from collections.abc import Callable


_R = TypeVar("_R")


# The last config `resolve_auto_config` derived, as `(base config, sensed bytes, resolved)`.
# See `resolve_auto_config` for why the *same object* is handed back when the envelope has
# not meaningfully moved.
_RESOLVED: tuple[Config, int, Config] | None = None
# The engine's resident bytes when `_RESOLVED`'s envelope was last confirmed, and when the
# last terminal finished: what `_retained_credit` measures a drop against.
_HELD_AT_SENSE: int | None = None
_LAST_END: float | None = None
# How long after a query ends a drop in free RAM that the engine's own resident set accounts
# for is read as retained arena rather than pressure. The allocator holds freed regions for
# `bc-py`'s `PURGE_DELAY_MS` (10 s) before returning them, so three of those is past the
# point where anything still held is retention: after it, the drop is real and the envelope
# follows it.
_RETENTION_WINDOW_S = 30.0
# Re-derive the resolved config only when the sensed envelope moves by more than this
# fraction. Free RAM jitters by a few pages between two back-to-back queries; that jitter
# cannot change a spill decision, and honoring it would rebuild (and re-validate) the whole
# config every query for a cap that differs in its seventh digit.
_ENVELOPE_TOLERANCE = 1 / 32


def resolve_auto_config(config: Config | None = None) -> Config:
    """Return `config` with auto-sensed tunables filled in (a no-op `config` if none).

    When `memory.max_memory_bytes` is unset and `memory.unbounded_memory` is off, a
    concrete cap is sensed from the live envelope (host RAM / cgroup, via Carbonite's
    `PressureMonitor`) and frozen in — driving both the data plane's spill budget and
    the control plane's admission envelope, so a large query spills instead of OOMing
    with zero config. An explicit cap or `unbounded_memory=True` is returned untouched
    (the same object, so a caller can detect the no-op with ``is``).

    ## The derived config is reused while the envelope holds still

    The sensed value is *live free RAM*, so a naive implementation builds a brand-new
    `Config` on every query — and every one of them is a cache miss for `validate_config`
    and re-runs its ~60 range checks, to certify a config that differs from the last one
    only in how many pages the page cache happened to hold. Two `dataclasses.replace`s plus
    a full validation, ~32 µs, on every `collect()`.

    So the previous result is reused while the newly sensed envelope stays within
    `_ENVELOPE_TOLERANCE` of the one it was built from. This is not a staleness compromise:
    a 3% drift in free RAM cannot change an admission or spill decision (those compare
    against fractions of the envelope), and a real change — a large allocation, a container
    limit, a neighbouring process — moves it far past the tolerance and rebuilds
    immediately. What it buys is *object identity*: the conductor hands the same config back
    each query, so validation, and everything else keyed on the config, hits.

    Args:
        config: The config to resolve. Defaults to the active config.

    Returns:
        The config with auto-sensed tunables filled in — the same object as the last call
        when nothing has meaningfully changed.
    """
    global _RESOLVED
    cfg = config if config is not None else active_config()
    mem = cfg.memory
    if mem.max_memory_bytes is not None or mem.unbounded_memory:
        return cfg
    # `api` may consult Carbonite (it is the conductor); `config` may not.
    from batcher.carbonite.memory.pressure import PressureMonitor

    global _HELD_AT_SENSE
    sensed = PressureMonitor(cfg).envelope_bytes()
    if sensed <= 0:
        return cfg  # could not sense — keep the safe unbounded fallback
    held = _engine_held_bytes()
    cached = _RESOLVED
    credit = 0
    if cached is not None and cached[0] is cfg:
        credit = _retained_credit(cached[1] - sensed, held)
        sensed += credit
        if _within_tolerance(sensed, cached[1]):
            if not credit:
                _HELD_AT_SENSE = held
            return cached[2]
    resolved = dataclasses.replace(
        cfg,
        memory=dataclasses.replace(mem, max_memory_bytes=sensed, max_memory_bytes_sensed=True),
    )
    _RESOLVED = (cfg, sensed, resolved)
    if not credit:
        # Growth is measured from an envelope sensed without retention in it, so a credited
        # one does not move the baseline.
        _HELD_AT_SENSE = held
    return resolved


def _retained_credit(drop: int, held: int | None) -> int:
    """How much of a `drop` in free RAM is the engine's own retained arena, not pressure.

    mimalloc keeps the pages a query freed for its purge delay, so the query after a large
    one senses less free RAM by roughly what that one allocated -- memory this process holds
    and will reuse on its next allocation, not memory anything else took. Read as pressure,
    it changed the plan: on TPC-DS sf10 at 16 cores (64 GB, tables preloaded) a staged run
    left 5.4 GB resident, the next query's envelope fell from 14.7 to 8.7 GB, its common
    -subplan budget halved from 512 to 256 MB, the cached reuse verdict keyed on that budget
    missed, and q4 ran 2.1 s without its shared CTE against 0.6 s with it. The run after
    that sensed 10.5 GB, found the old verdict again, and was back to 0.7 s.

    So the drop is credited back, as far as the engine's resident set grew since the envelope
    was last sensed without a credit, and only within `_RETENTION_WINDOW_S` of the last query's
    end. Real pressure is not hidden: a drop another process caused does not grow this
    resident set, so it is never credited, and anything this process still holds once the
    allocator would have purged it is live, so the first query past the window sees the
    envelope fall. The cost of the bound is that a large result the caller still holds from
    the query before is credited as if retained for up to that window: a planning figure
    that high spills later rather than sooner, and Carbonite's live pressure reading, which
    this does not touch, still sees the process's real resident set.

    Args:
        drop: Bytes the sensed envelope fell below the cached one.
        held: The engine's resident bytes now, `None` when unreadable.

    Returns:
        Bytes to add back to the sensed envelope; `0` when none of the drop is retention.
    """
    if drop <= 0 or held is None or _HELD_AT_SENSE is None or _LAST_END is None:
        return 0
    if time.monotonic() - _LAST_END > _RETENTION_WINDOW_S:
        return 0
    return max(0, min(drop, held - _HELD_AT_SENSE))


def _engine_held_bytes() -> int | None:
    """The process's resident set less pyarrow's pool: the engine's arena, live or retained."""
    from batcher.carbonite.memory.probe import process_rss_bytes

    rss = process_rss_bytes()
    if rss is None:
        return None
    try:
        return rss - int(pa.total_allocated_bytes())
    except Exception:  # a reading must never fail a query
        return rss


def _within_tolerance(sensed: int, previous: int) -> bool:
    """True when `sensed` is close enough to `previous` to reuse the config built from it."""
    return abs(sensed - previous) <= previous * _ENVELOPE_TOLERANCE


def with_auto_config(fn: Callable[..., _R]) -> Callable[..., _R]:
    """Decorate a terminal entry point to run under the auto-resolved config.

    Fixes a query's sensed memory envelope once, at the materializing-terminal
    boundary (collect / write / stats and what delegates to them) — not per stage,
    where adaptive re-planning and the growing working set would drift it. A no-op
    when the user pinned the memory config or sensing is unavailable.

    Also where `execution.query_timeout_s` starts counting: `timed_terminal` opens the
    query's cancellable scope here, around the whole operation, when a limit is set.
    """
    from batcher.core.runtime import timed_terminal

    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> _R:
        global _LAST_END
        resolved = resolve_auto_config()
        try:
            if resolved is active_config():
                with timed_terminal():
                    return fn(*args, **kwargs)
            with config_context(resolved), timed_terminal():
                return fn(*args, **kwargs)
        finally:
            _LAST_END = time.monotonic()  # when the allocator's retention window opens

    return wrapper


def approx_quantile(batches: Iterable[pa.RecordBatch], column: str, q: float) -> float | None:
    """Approximate quantile `q` of `column` from a streamed, merged TDigest.

    Opt-in and explicitly approximate: tail-accurate (p99/p999) and far cheaper than
    an exact sort. Consumes `batches` one at a time — building a per-batch TDigest and
    merging the (tiny) sketches — so the column is never held whole on the driver; the
    caller projects to just `column` and streams it (single-node or distributed).
    Returns None if the column is non-numeric or empty.
    """
    from batcher import core

    sketches = [sk for b in batches if (sk := core.tdigest_partial([b], column)) is not None]
    return core.tdigest_quantile(sketches, q)
