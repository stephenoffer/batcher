"""The per-batch enrichment step: keys in, columns out, as few round trips as possible.

`LookupEnricher` is what `Dataset.lookup_join` hands to `map_batches`. It is a *class*
rather than a function for the same reason an inference model is: `map_batches`
constructs one per worker and reuses it, so the connection is opened once and the cache
survives across batches. A plain function would reconnect and forget on every batch, which
is the difference between a lookup join and a denial-of-service attack on the store.

The work per batch is four steps, and only one of them touches the store:

1. Read the key column, render it as strings, and reduce it to its distinct values — all
   in Arrow's kernels.
2. Ask the cache to resolve those, fetching only what it does not already know
   (`io.lookup.cache`).
3. Build the columns and the matched mask (`io.lookup.base.lookup_arrays`).
4. Concatenate onto the input batch, dropping unmatched rows for an inner join.

**Every step is `O(distinct keys)` in Python and `O(rows)` only inside Arrow.** That is not
a nicety, it is the feature: a lookup join claims its cost scales with the distinct keys the
data contains, and a per-row Python loop anywhere in this list makes that claim false. See
`lookup_arrays` for what the row-wise version measured.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from batcher._internal.errors import PlanError
from batcher.io.lookup.base import lookup_arrays
from batcher.io.lookup.cache import LookupCache

__all__ = ["LookupEnricher"]


def _distinct_keys(keys: pa.Array) -> list[str]:
    """The batch's distinct non-null keys, as a Python list.

    `unique` and `drop_null` run in Arrow's kernels, so the only list this materializes is
    the *distinct* one — which is the list the store is about to be asked for anyway. The
    obvious `[k for k in keys.to_pylist() if k is not None]` builds a Python list the size
    of the batch first, which on a two-million-row batch touching five thousand keys is
    1,995,000 objects created to be thrown away.
    """
    import pyarrow.compute as pc

    return pc.unique(pc.drop_null(keys)).to_pylist()


def _as_strings(column: pa.ChunkedArray | pa.Array) -> pa.Array:
    """`column` as a string array, so any key type joins against a string keyspace.

    A key-value store keys on bytes. An integer customer id has to become ``"41"`` to be
    looked up, and it has to become the *same* ``"41"`` on every batch and every worker —
    so the conversion is Arrow's `cast`, not Python's `str`, which would render a float or
    a decimal differently depending on how it arrived.
    """
    array = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if pa.types.is_string(array.type) or pa.types.is_large_string(array.type):
        return array
    return array.cast(pa.string())


class LookupEnricher:
    """Enriches each batch from a key-value store, caching what it learns.

    One instance per worker. Constructed by `Dataset.lookup_join` through
    `map_batches`'s stateful-class path, which is also what makes it work unchanged
    single-node, distributed, and over a stream: the enrichment is per batch, so nothing
    ever holds more than one batch's keys.
    """

    def __init__(
        self,
        make_lookup: Any,
        on: str,
        *,
        how: str = "left",
        prefix: str = "",
        cache_size: int = 100_000,
        cache_ttl_seconds: float | None = None,
    ) -> None:
        """Build the enricher and open its store.

        Args:
            make_lookup: A zero-argument callable returning a `KeyValueLookup`. A factory
                rather than an instance because this object is constructed *on the
                worker*: a live connection cannot be pickled to get there, and a
                `rocksdict` handle cannot be shared between processes at all.
            on: The column to look up by.
            how: ``"left"`` keeps every input row, filling nulls where the store has no
                match; ``"inner"`` drops those rows.
            prefix: Prepended to every looked-up column name, so a dimension whose columns
                collide with the input's can still be joined.
            cache_size: Entries the per-worker cache holds; `0` disables it.
            cache_ttl_seconds: How long an entry stays usable. This is the join's freshness
                bound.

        Raises:
            PlanError: If `how` is not ``"left"`` or ``"inner"``.
        """
        if how not in ("left", "inner"):
            raise PlanError(
                f"lookup_join(how={how!r}) is not supported",
                hint=(
                    "A point-lookup store can answer 'left' (keep every row, null-fill "
                    "the misses) or 'inner' (drop the misses). A right or outer join "
                    "would have to enumerate the store, which is the scan a lookup join "
                    "exists to avoid — read the dimension as a dataset and join it."
                ),
            )
        self._lookup = make_lookup()
        self._on = on
        self._how = how
        self._prefix = prefix
        self._cache = LookupCache(cache_size, cache_ttl_seconds)

    def __call__(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        """Enrich one batch.

        Args:
            batch: The input rows.

        Returns:
            The batch with the looked-up columns appended, minus any unmatched rows under
            an inner join.

        Raises:
            PlanError: If the batch has no `on` column. Raised per batch rather than
                checked once, because a `map_batches` stage sees whatever schema reaches
                it and a missing column is otherwise a `KeyError` from inside Arrow.
        """
        if self._on not in batch.schema.names:
            raise PlanError(
                f"lookup_join(): column {self._on!r} is not in the batch "
                f"({', '.join(batch.schema.names)})"
            )
        keys = _as_strings(batch.column(batch.schema.get_field_index(self._on)))
        resolved = self._cache.resolve(_distinct_keys(keys), self._lookup.multi_get)
        schema = self._lookup.value_schema()
        added, matched = lookup_arrays(keys, resolved, schema)
        names = [self._prefix + field.name for field in schema]
        out = pa.RecordBatch.from_arrays(
            list(batch.columns) + added, names=list(batch.schema.names) + names
        )
        if self._how == "left":
            return out
        import pyarrow.compute as pc

        return out.filter(pc.fill_null(matched, False))

    def stats(self) -> dict[str, int | float]:
        """This worker's cache effectiveness — see `LookupCache.stats`.

        Returns:
            The hit, negative-hit and miss counts and the entry count.
        """
        return self._cache.stats()
