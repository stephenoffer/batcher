"""Resource gauges — what the engine's envelopes, disks, and channels are holding.

The counters in `observe.metrics` say what a process *did*: queries run, rows produced,
operators executed. They cannot say what it is *holding*, and that is the other half of
every capacity question. A job that ran 400 queries with the buffer pool at 30% and one
that ran 400 with it at 99% and 60 GB spilled are the same job by every counter and
completely different operationally.

Carbonite already measures all of it — the buffer pool's envelope and high-water mark, the
spill store's per-tier bytes and free disk, the result cache's hit rate, the admission
limiter's queue depth, the shuffle session's locality and credit window. Each is a `stats()`
method returning a plain dict of numbers, and each was readable only by holding the object
that owned it. This module is the other end of the `RESOURCE` event that carries those
readings onto the bus: it keeps the latest reading per group and renders it as Prometheus
gauges.

**Gauges, not counters.** A reading replaces its group's previous one rather than adding to
it, because these describe a level. A consumer that differences successive readings of
`used_bytes` gets noise, not a rate.

The flattening is deliberately generic — every numeric leaf becomes a gauge named for its
path — so a resource that grows a new field starts being exported without a change here.
That is the property the hand-written exposition in `metrics.py` does not have, and the
reason those readings sat unexported for as long as they did.
"""

from __future__ import annotations

import threading
from typing import Any

from batcher.observe.counters._series import escape_label

__all__ = ["ResourceGauges"]

#: Groups may not exceed this many, so a caller publishing a per-query group name cannot
#: grow the map without bound over a long run. Far above the handful Carbonite publishes.
_MAX_GROUPS = 32

#: A single group's reading may not contribute more than this many series. Bounds a
#: `stats()` that returns a per-partition or per-channel map from becoming unbounded
#: cardinality in the exposition.
_MAX_SERIES_PER_GROUP = 128


class ResourceGauges:
    """The latest reading of each resource group, foldable from the bus and renderable.

    One instance per collector. Guarded by its own lock so a reading published from a
    shuffle worker thread and a scrape from an HTTP thread cannot interleave.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._groups: dict[str, dict[str, Any]] = {}

    def reset(self) -> None:
        """Forget every reading, so the next snapshot reports only what has since arrived."""
        with self._lock:
            self._groups.clear()

    def record(self, group: str, stats: object) -> None:
        """Replace `group`'s reading with `stats`.

        Ignores anything that is not a dict, because the field is carried on an event and a
        publisher that got it wrong must not be able to poison the exposition.

        Args:
            group: The resource group's name (``memory``, ``spill``, ``shuffle``, ...).
            stats: That group's `stats()` reading.
        """
        if not isinstance(stats, dict):
            return
        with self._lock:
            if group not in self._groups and len(self._groups) >= _MAX_GROUPS:
                return
            self._groups[group] = _plain(stats)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Every group's latest reading, nested and deep-copied.

        Returns:
            A dict keyed by group name; empty when nothing has published a reading.
        """
        with self._lock:
            return {name: _plain(stats) for name, stats in sorted(self._groups.items())}

    def render(self) -> list[str]:
        """The readings as Prometheus exposition lines.

        Numeric leaves become ``batcher_<group>_<path>`` gauges; booleans become 0/1; a
        string leaf becomes a state-set series ``batcher_<group>_<path>{state="..."} 1``,
        the conventional way to expose an enumerated level (``pressure_level``,
        ``disk_pressure``) to a system that only stores floats.

        Returns:
            A list of lines, empty when no reading has arrived.
        """
        out: list[str] = []
        for group, stats in self.snapshot().items():
            for path, value in _leaves(stats)[:_MAX_SERIES_PER_GROUP]:
                name = f"batcher_{_identifier(group)}_{path}"
                out.append(f"# TYPE {name} gauge")
                if isinstance(value, str):
                    out.append(f'{name}{{state="{escape_label(value)}"}} 1')
                else:
                    out.append(f"{name} {value}")
        return out


def _plain(value: Any) -> Any:
    """A JSON-safe deep copy: dicts and sequences recursed, everything else left alone."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _leaves(stats: dict[str, Any], prefix: str = "") -> list[tuple[str, float | int | str]]:
    """Every exportable leaf of `stats` as ``(path, value)``, depth-first and sorted.

    Booleans are normalized to 0/1 here rather than at render time, so a reading holding
    `True` and one holding `1` produce the same series. Values that are neither a number
    nor a string (a nested list, `None`) are dropped: they have no gauge representation,
    and inventing one would be worse than omitting them.
    """
    out: list[tuple[str, float | int | str]] = []
    for key, value in sorted(stats.items()):
        path = f"{prefix}{_identifier(key)}"
        if isinstance(value, dict):
            out.extend(_leaves(value, f"{path}_"))
        elif isinstance(value, bool):
            out.append((path, int(value)))
        elif isinstance(value, (int, float, str)):
            out.append((path, value))
    return out


def _identifier(name: str) -> str:
    """`name` reduced to the characters a Prometheus metric name may contain."""
    cleaned = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    return cleaned.lower() or "unknown"
