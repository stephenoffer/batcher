"""The `map_batches` stage `Dataset.lookup_join` runs on each worker.

A thin, *picklable* shell around `LookupEnricher`. It exists because what crosses to a
worker has to survive being serialized, and neither a live connection nor a closure over
one does. Everything here is plain data — a URI, a `{name: dtype}` mapping, a few flags —
and the store is opened in `__init__`, which `map_batches` calls once per worker rather
than once per batch.

That "once per worker" is the whole reason this is a class. A function would reconnect and
throw its cache away on every batch, which turns a lookup join into a round trip per row
against a store that is usually shared with production traffic.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

__all__ = ["LookupStage"]


class LookupStage:
    """Opens a lookup store once per worker and enriches each batch through it."""

    def __init__(
        self,
        *,
        source: str,
        on: str,
        schema: dict[str, str],
        how: str = "left",
        prefix: str = "",
        cache_size: int = 100_000,
        cache_ttl: str | None = None,
        hash_values: bool = False,
    ) -> None:
        """Open the store and build this worker's enricher.

        Args:
            source: The store URI.
            on: The column to look up by.
            schema: The contributed columns, as `{name: dtype_name}`.
            how: ``"left"`` or ``"inner"``.
            prefix: Prepended to each contributed column name.
            cache_size: Entries this worker's cache holds.
            cache_ttl: How long an entry stays usable, as a duration string.
            hash_values: Read a Redis key as a hash rather than a JSON string.
        """
        from batcher.io.lookup.join import LookupEnricher
        from batcher.io.lookup.spec import build_lookup, lookup_schema

        resolved = lookup_schema(schema)
        options: dict[str, Any] = {"prefix": "", "hash_values": hash_values}
        self._enricher = LookupEnricher(
            lambda: build_lookup(source, resolved, options),
            on,
            how=how,
            prefix=prefix,
            cache_size=cache_size,
            cache_ttl_seconds=_ttl_seconds(cache_ttl),
        )

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        """Enrich one batch.

        Args:
            batch: The input rows.

        Returns:
            The batch with the looked-up columns appended.
        """
        return self._enricher(batch)

    def stats(self) -> dict[str, int | float]:
        """This worker's lookup-cache effectiveness.

        Returns:
            The hit, negative-hit and miss counts and the entry count.
        """
        return self._enricher.stats()


def _ttl_seconds(cache_ttl: str | None) -> float | None:
    """A duration string as seconds, or `None`.

    `_duration_micros` is the engine's one duration parser — the same one `with_watermark`
    and the windowing functions use — so ``"5m"`` means the same thing everywhere and a
    typo produces the same error message wherever it is typed.
    """
    if cache_ttl is None:
        return None
    from batcher.plan.functions.temporal import _duration_micros

    return _duration_micros(cache_ttl, arg="lookup cache_ttl") / 1_000_000.0
