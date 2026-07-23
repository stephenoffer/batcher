"""Zero-config resolution: sense the machine once, and pin it for the query's scope."""

from __future__ import annotations

import dataclasses
import functools
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

    sensed = PressureMonitor(cfg).envelope_bytes()
    if sensed <= 0:
        return cfg  # could not sense — keep the safe unbounded fallback
    cached = _RESOLVED
    if cached is not None and cached[0] is cfg and _within_tolerance(sensed, cached[1]):
        return cached[2]
    resolved = dataclasses.replace(cfg, memory=dataclasses.replace(mem, max_memory_bytes=sensed))
    _RESOLVED = (cfg, sensed, resolved)
    return resolved


def _within_tolerance(sensed: int, previous: int) -> bool:
    """True when `sensed` is close enough to `previous` to reuse the config built from it."""
    return abs(sensed - previous) <= previous * _ENVELOPE_TOLERANCE


def with_auto_config(fn: Callable[..., _R]) -> Callable[..., _R]:
    """Decorate a terminal entry point to run under the auto-resolved config.

    Fixes a query's sensed memory envelope once, at the materializing-terminal
    boundary (collect / write / stats and what delegates to them) — not per stage,
    where adaptive re-planning and the growing working set would drift it. A no-op
    when the user pinned the memory config or sensing is unavailable.
    """

    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> _R:
        resolved = resolve_auto_config()
        if resolved is active_config():
            return fn(*args, **kwargs)
        with config_context(resolved):
            return fn(*args, **kwargs)

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
