"""Process-wide runtime services for Core (the default MetadataHub)."""

from __future__ import annotations

from batcher._internal.logging import get_logger
from batcher.config import active_config
from batcher.metadata import MetadataHub
from batcher.metadata.backends import InProcessBackend, make_backend

__all__ = ["default_hub", "reset_default_hub"]

_log = get_logger("metadata")

_hub: MetadataHub | None = None
_hub_backend_key: tuple[str, str | None] | None = None


def reset_default_hub() -> None:
    """Drop the cached process-wide hub so the next `default_hub()` rebuilds fresh.

    For test isolation: learned stats accumulate in the process-wide hub, so a test
    that asserts on cardinality/cost-driven plan shape can otherwise be perturbed by
    stats an earlier test recorded. Resetting between tests makes those assertions
    deterministic without changing production behavior.
    """
    global _hub, _hub_backend_key
    _hub = None
    _hub_backend_key = None


def default_hub() -> MetadataHub:
    """Return a process-wide MetadataHub built from the active config.

    Rebuilt if the configured backend changes, so `config_context` switching the
    metadata backend takes effect.
    """
    global _hub, _hub_backend_key
    meta = active_config().metadata
    key = (meta.backend, meta.uri)
    if _hub is None or key != _hub_backend_key:
        _hub = MetadataHub(_build_backend(meta.backend, meta.uri))
        _hub_backend_key = key
    return _hub


def _build_backend(backend: str, uri: str | None):
    """Construct the configured backend, degrading to in-process on failure.

    A durable backend (object storage / SQLite / Redis) can fail to construct — a
    missing optional dependency, an unreachable or misconfigured URI. Learned stats are
    an optimization, never a correctness input, so a broken store must not fail every
    query: fall back to the in-process store (this session still learns; only cross-run
    persistence is lost) and log once instead of raising into the hot path.
    """
    if backend == "in_process":
        return InProcessBackend()
    try:
        return make_backend(backend, uri)
    except Exception:
        _log.warning(
            "metadata backend %r (uri=%r) unavailable; using an in-process store "
            "(cross-run learning disabled this session)",
            backend,
            uri,
            exc_info=True,
        )
        return InProcessBackend()
