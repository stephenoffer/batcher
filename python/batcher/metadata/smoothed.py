"""Best-effort read/write of a single learned scalar, exponentially smoothed across runs.

Several subsystems learn one number per key and want the same three properties: a cold
store yields `None` rather than an error, a write blends the new observation into the
prior instead of overwriting it, and neither direction can ever raise into the query path.
Carbonite learns a converged shuffle credit window and a memory-pressure flap rate; `io`
learns a per-source read throughput; `dist` learns a partition skew factor.

Those had grown three byte-identical copies of the same twelve lines, one of them across
the `carbonite`/`metadata` boundary. The subsystems cannot import each other, so a shared
helper has to live in a neutral layer, and `metadata` is where the Hub already is.

The smoothing weight is `max(floor, 1/(n+1))` for `floor =
optimizer.learned_scalar_alpha_floor`: a **running mean** while evidence is thin, decaying
into an exponential average with a `~1/floor`-observation memory once enough runs have
accrued. A fixed weight is the wrong step at both ends — at a static 0.5 the *first*
observation keeps a quarter of the estimate after three runs and an eighth after four,
forever, so one anomalous cold run anchors the value it was supposed to be smoothing; and a
pure running mean never forgets a regime the workload has left.

The floor is `learned_scalar_alpha_floor` and deliberately not `learning_smoothing_alpha`.
The latter is a *static blend weight* used elsewhere, and at its value of 0.5 it would
dominate `1/(n+1)` from the second observation onward — which is to say the floor would
never bind and the running-mean phase would not exist. `kyber.learning._smooth` makes the
same distinction for the same reason; this is the neutral-layer twin of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.logging import note_suppressed
from batcher.config import active_config

if TYPE_CHECKING:
    from batcher.config import Config
    from batcher.metadata.hub import MetadataHub

__all__ = ["load_scalar", "record_smoothed_scalar"]


def load_scalar(hub: MetadataHub | None, namespace: str, key: str) -> float | None:
    """Read one learned scalar, or `None` when it was never recorded.

    Reads both the current shape (a `{"value", "n"}` record carrying the observation count)
    and the bare float an older store may hold, so a hub written by a previous build keeps
    answering.

    Args:
        hub: The metadata hub, or `None` when learning is off.
        namespace: The learned-parameter namespace, e.g. `"carbonite.shuffle_window"`.
        key: The identity the value is learned per, e.g. a shuffle signature.

    Returns:
        The stored value, or `None` for a cold store, an absent key, or any read failure.
    """
    if hub is None:
        return None
    try:
        stored = hub.get_keyed_param(namespace, key)
    except Exception:  # pragma: no cover - a learned read must never break a query
        return None
    return _value_of(stored)


def _value_of(stored: object) -> float | None:
    """The scalar held by a stored record, in either shape."""
    if stored is None:
        return None
    if isinstance(stored, dict):
        value = stored.get("value")
        return None if value is None else float(value)
    try:
        return float(stored)  # type: ignore[arg-type]
    except (TypeError, ValueError):  # pragma: no cover - a foreign blob
        return None


def record_smoothed_scalar(
    hub: MetadataHub | None,
    namespace: str,
    key: str,
    value: float,
    config: Config | None = None,
) -> None:
    """Blend one observation into the learned scalar at `namespace`/`key`.

    The first observation is stored as-is; every later one moves the stored value a step
    `max(floor, 1/(n+1))` of the way toward it, where `n` counts the observations already
    folded in. That step is a plain running mean while `n` is small and settles into a fixed
    exponential average once `n` passes `1/floor`, which is what makes an early anomalous run
    wash out instead of anchoring the estimate.

    Best-effort in both directions, so a metadata backend that is unreachable, read-only, or
    mid-migration degrades to "learned nothing this run".

    Args:
        hub: The metadata hub, or `None` when learning is off.
        namespace: The learned-parameter namespace.
        key: The identity the value is learned per.
        value: The new observation.
        config: Config to read the smoothing weight from; defaults to the active one.
    """
    if hub is None:
        return
    try:
        floor = (config or active_config()).optimizer.learned_scalar_alpha_floor
        stored = hub.get_keyed_param(namespace, key)
        prior = _value_of(stored)
        count = float(stored.get("n", 1.0)) if isinstance(stored, dict) else 1.0
        if prior is None:
            smoothed, count = value, 1.0
        else:
            step = max(floor, 1.0 / (count + 1.0))
            smoothed = prior + step * (value - prior)
            count += 1.0
        hub.put_keyed_param(namespace, key, {"value": smoothed, "n": count})
    except Exception as exc:  # pragma: no cover - a learned write must never break a query
        note_suppressed("metadata", f"record learned scalar {namespace}/{key}", exc)
