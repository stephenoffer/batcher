"""The `Dataset.scd` namespace — dimension maintenance from snapshots and change feeds.

Breadth on `Dataset` lives on accessors. SCD maintenance composes existing ops
(merge / join / union / with_columns) — no new IR:

- ``type1`` — overwrite-in-place (no history): a keyed upsert.
- ``type2`` — full history via effective-dating columns (`valid_from`/`valid_to`/
  `is_current`): expire the current row of a changed key and append a new version.
- ``type3`` — keep the previous value in a ``<attr>_prev`` column.
- ``apply_changes`` — apply a **change feed** (CDC) rather than a snapshot: deletes,
  redeliveries, and out-of-order rows, reconciled idempotently.

The first three take a clean snapshot of the dimension as it is *now*. `apply_changes`
takes the stream of what *happened*, which is what a database's change-data-capture
connector actually produces, and is the harder and more common ETL shape.
"""

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir import Col, Expr, lit, nullif


def _typed_null(of: Expr) -> Expr:
    """A NULL of the same type as `of` (``nullif(x, x)`` is always NULL) — the engine
    has no typed null literal, so this is how a null column is built."""
    return nullif(of, of)


def _require_keys(method: str, keys: str | list[str]) -> list[str]:
    """Normalize `keys` to a non-empty list, or raise a `PlanError` naming the method."""
    key_list = [keys] if isinstance(keys, str) else list(keys)
    if not key_list:
        raise PlanError(
            f"{method}(): needs at least one natural key column — "
            "pass keys='id' (or keys=['a', 'b'] for a composite key)"
        )
    return key_list


def _require_track(method: str, track: list[str]) -> list[str]:
    """Require a non-empty `track` list, or raise a `PlanError` naming the method."""
    if not track:
        raise PlanError(
            f"{method}(): needs at least one attribute column to track in `track` — "
            "these are the columns whose change makes a new version (e.g. track=['city'])"
        )
    return list(track)


if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.io.manifest import WriteManifest

__all__ = ["DatasetSCD"]


class DatasetSCD:
    """Accessor for slowly-changing-dimension upserts over a `Dataset` (``ds.scd``).

    The dataset is the *incoming* dimension snapshot (natural keys + attributes).

    Examples:
        .. doctest::

            >>> import os
            >>> import tempfile

            >>> import batcher as bt
            >>> target = os.path.join(tempfile.mkdtemp(), "dim.parquet")
            >>> _ = bt.from_pydict({"id": [1], "city": ["NYC"]}).scd.type1(target, keys="id")
            >>> bt.read.parquet(target).to_pydict()
            {'id': [1], 'city': ['NYC']}
    """

    __slots__ = ("_ds",)

    def __init__(self, ds: Dataset) -> None:
        """Bind the SCD accessor to its `Dataset`; reached as `ds.scd`, not constructed directly."""
        self._ds = ds

    def type1(
        self, target: str, *, keys: str | list[str], format: str | None = None, **opts
    ) -> WriteManifest:
        """SCD type 1 — overwrite changed attributes in place (no history).

        A keyed upsert into `target` (delegates to `ds.write.merge`). An absent target is the
        dimension's **first load**: every incoming row is new, so the snapshot is written as
        the table. `type2`, `type3` and `apply_changes` all handle that case; this did not, so
        the first run of a type-1 load into a Delta table raised
        ``IOError: path '...' does not exist`` instead of creating it. A file target (Parquet,
        CSV) happened to work, because writing a MERGE to a missing file just writes the file,
        so only the transactional formats broke.

        Args:
            target: Path of the dimension table to upsert into.
            keys: The natural key column(s) identifying a row.
            format: Target format; inferred from the path when omitted. Accepted here for
                parity with `type2`/`type3`, which have always taken it.
            **opts: Forwarded to `ds.write.merge`.

        Returns:
            The `WriteManifest` of the rewritten target.

        Examples:
            .. doctest::

                >>> import os
                >>> import tempfile

                >>> import batcher as bt
                >>> target = os.path.join(tempfile.mkdtemp(), "dim.parquet")
                >>> base = bt.from_pydict({"id": [1, 2], "city": ["NYC", "LA"]})
                >>> _ = base.write.parquet(target)
                >>> _ = bt.from_pydict({"id": [2], "city": ["SF"]}).scd.type1(target, keys="id")
                >>> bt.read.parquet(target).sort("id").to_pydict()
                {'id': [1, 2], 'city': ['NYC', 'SF']}

                The first load needs no existing table:

                >>> fresh = os.path.join(tempfile.mkdtemp(), "new.parquet")
                >>> _ = bt.from_pydict({"id": [1], "city": ["NYC"]}).scd.type1(fresh, keys="id")
                >>> bt.read.parquet(fresh).to_pydict()
                {'id': [1], 'city': ['NYC']}
        """
        from batcher.io.detect import detect_format
        from batcher.io.filesystem import resolve_filesystem

        key_list = _require_keys("type1", keys)
        fmt = detect_format(target, format)
        if not resolve_filesystem(target).exists(target):
            return self._ds.write(target, fmt, mode="overwrite", **opts)
        return self._ds.write.merge(target, on=key_list, when_matched="update", format=fmt, **opts)

    def type2(
        self,
        target: str,
        *,
        keys: str | list[str],
        track: list[str],
        as_of: str,
        valid_from: str = "valid_from",
        valid_to: str = "valid_to",
        is_current: str = "is_current",
        format: str | None = None,
        **opts,
    ) -> WriteManifest:
        """SCD type 2 — keep full history with effective-dating columns.

        For each natural `keys` whose `track` attributes changed, the current version
        is expired (``valid_to = as_of``, ``is_current = False``) and a new version is
        appended (``valid_from = as_of``, ``valid_to = NULL``, ``is_current = True``).
        Brand-new keys are inserted as a first version; unchanged keys are untouched.
        `as_of` is the effective timestamp (e.g. the batch date), stored as a string.

        Args:
            target: Path of the dimension table to maintain.
            keys: The natural key column(s) identifying a row.
            track: The attribute columns whose change triggers a new version.
            as_of: The effective timestamp for this batch, stored as a string.
            valid_from: Name of the version-start column.
            valid_to: Name of the version-end column (NULL while current).
            is_current: Name of the boolean current-version flag column.
            format: Target format; inferred from the path when omitted.
            **opts: Forwarded to the writer.

        Returns:
            The `WriteManifest` of the rewritten target.

        Examples:
            .. doctest::

                >>> import os
                >>> import tempfile

                >>> import batcher as bt
                >>> target = os.path.join(tempfile.mkdtemp(), "dim.parquet")
                >>> v1 = bt.from_pydict({"id": [1], "city": ["NYC"]})
                >>> _ = v1.scd.type2(target, keys="id", track=["city"], as_of="2024-01-01")
                >>> v2 = bt.from_pydict({"id": [1], "city": ["LA"]})
                >>> _ = v2.scd.type2(target, keys="id", track=["city"], as_of="2024-06-01")
                >>> hist = bt.read.parquet(target).sort("valid_from")
                >>> hist.select("valid_from", "is_current").to_pydict()
                {'valid_from': ['2024-01-01', '2024-06-01'], 'is_current': [False, True]}
        """
        from batcher.api.session import read as _read
        from batcher.io.detect import detect_format
        from batcher.io.filesystem import resolve_filesystem

        key_list = _require_keys("type2", keys)
        track = _require_track("type2", track)
        fmt = detect_format(target, format)
        incoming = self._ds.select(*key_list, *track)

        def _versioned(ds: Dataset, current: bool) -> Dataset:
            return ds.with_columns(
                **{
                    valid_from: lit(as_of),
                    valid_to: _typed_null(lit(as_of)) if current else lit(as_of),
                    is_current: lit(current),
                }
            )

        # First load: every incoming row becomes an open (current) first version.
        if not resolve_filesystem(target).exists(target):
            return _versioned(incoming, current=True).write(target, fmt, mode="overwrite", **opts)

        existing = _read(target, format=fmt)
        current = existing.filter(Col(is_current) == lit(True))
        history = existing.filter(Col(is_current) == lit(False))

        # An incoming row is new-or-changed iff no *current* version matches on
        # keys AND all tracked attributes (anti-join on keys+track avoids comparing
        # suffixed join columns).
        current_kt = current.select(*key_list, *track)
        changed_or_new = incoming.join(current_kt, on=[*key_list, *track], how="anti")
        changed_keys = changed_or_new.select(*key_list).distinct()

        # Expire the superseded current rows: keep their original valid_from, just
        # close valid_to and clear is_current.
        expired = current.join(changed_keys, on=key_list, how="semi").with_columns(
            **{valid_to: lit(as_of), is_current: lit(False)}
        )
        kept_current = current.join(changed_keys, on=key_list, how="anti")
        new_versions = _versioned(changed_or_new, current=True)

        result = reduce(lambda a, b: a.union(b), [history, kept_current, expired, new_versions])
        return result.write(target, fmt, mode="overwrite", **opts)

    def type3(
        self,
        target: str,
        *,
        keys: str | list[str],
        track: list[str],
        format: str | None = None,
        **opts,
    ) -> WriteManifest:
        """SCD type 3 — keep each `track` attribute's previous value in a ``<attr>_prev`` column.

        Limited history. For a matched key the existing current value moves to
        ``<attr>_prev`` and the incoming value becomes current; new keys get NULL
        previous values; untouched target keys are preserved.

        Args:
            target: Path of the dimension table to maintain.
            keys: The natural key column(s) identifying a row.
            track: The attribute columns whose previous value is kept.
            format: Target format; inferred from the path when omitted.
            **opts: Forwarded to the writer.

        Returns:
            The `WriteManifest` of the rewritten target.

        Examples:
            .. doctest::

                >>> import os
                >>> import tempfile

                >>> import batcher as bt
                >>> target = os.path.join(tempfile.mkdtemp(), "dim.parquet")
                >>> _ = bt.from_pydict({"id": [1], "city": ["NYC"]}).scd.type3(
                ...     target, keys="id", track=["city"]
                ... )
                >>> _ = bt.from_pydict({"id": [1], "city": ["LA"]}).scd.type3(
                ...     target, keys="id", track=["city"]
                ... )
                >>> bt.read.parquet(target).to_pydict()
                {'id': [1], 'city': ['LA'], 'city_prev': ['NYC']}
        """
        from batcher.api.session import read as _read
        from batcher.io.detect import detect_format
        from batcher.io.filesystem import resolve_filesystem

        key_list = _require_keys("type3", keys)
        track = _require_track("type3", track)
        fmt = detect_format(target, format)
        incoming = self._ds.select(*key_list, *track)
        if not resolve_filesystem(target).exists(target):
            first = incoming.with_columns(**{f"{a}_prev": _typed_null(Col(a)) for a in track})
            return first.write(target, fmt, mode="overwrite", **opts)

        existing = _read(target, format=fmt)
        # Target rows whose key is not in the incoming snapshot survive unchanged.
        survivors = existing.join(incoming.select(*key_list).distinct(), on=key_list, how="anti")
        # Left-join incoming to the current target values; the colliding `track`
        # columns from the right side are suffixed → exactly the ``<attr>_prev`` names
        # (NULL for brand-new keys). Result columns match the target schema.
        old = existing.select(*key_list, *track)
        updated = incoming.join(old, on=key_list, how="left", suffix="_prev")
        return survivors.union(updated).write(target, fmt, mode="overwrite", **opts)

    def apply_changes(
        self,
        target: str,
        *,
        keys: str | list[str],
        sequence_by: str,
        deletes: Expr | None = None,
        columns: list[str] | None = None,
        format: str | None = None,
        **opts,
    ) -> WriteManifest:
        """Apply a change feed (CDC) to `target`: a sequenced upsert that honors deletes.

        The dataset is a **change feed**, not a snapshot: it may carry deletes, redeliver
        rows it has already sent, and present them out of order. That is what a database
        CDC connector (Debezium, a Delta change feed, a Snowflake stream) emits, and it
        is why `type1` — which assumes a clean current-state snapshot — cannot consume it.

        Reconciliation follows Delta Live Tables' ``APPLY CHANGES INTO ... STORED AS SCD
        TYPE 1``:

        * Within the batch, only the greatest-`sequence_by` change per key survives.
        * A change applies only if its key is new, or its sequence is **at least** the
          sequence already stored for that key. So a redelivered change is a no-op and a
          late change that lost a race is discarded, rather than resurrecting old data.
        * A row matching `deletes` removes the target row and inserts nothing. A delete
          for an absent key is a tombstone and changes nothing.

        `sequence_by` is stored in the target — that is what lets a *later* run recognize
        a stale change. It must be unique per key; ties are broken arbitrarily.

        Re-applying a batch is therefore a no-op, and applying batches in non-decreasing
        sequence order converges on the source's state. Like `type1`, it is a copy-on-write
        overwrite of `target`, so it is single-writer only.

        .. warning::
            The apply is idempotent but not commutative. A delete is physical, not a
            tombstone, so a deleted key stores no sequence to compare against — replaying
            an *old insert* for a key that was since deleted will resurrect it. Feed
            batches in sequence order (which a CDC reader does), and treat a full replay
            from the beginning of a feed containing deletes as a rebuild, not a resume.

        Examples:
            .. doctest::

                >>> import os
                >>> import tempfile

                >>> import batcher as bt
                >>> target = os.path.join(tempfile.mkdtemp(), "customers.parquet")
                >>> feed = bt.from_pydict(
                ...     {
                ...         "id": [1, 2, 1],
                ...         "city": ["NYC", "LA", "SF"],
                ...         "op": ["INSERT", "INSERT", "UPDATE"],
                ...         "seq": [1, 2, 3],
                ...     }
                ... )
                >>> _ = feed.scd.apply_changes(
                ...     target,
                ...     keys="id",
                ...     sequence_by="seq",
                ...     deletes=bt.col("op") == "DELETE",
                ...     columns=["id", "city"],
                ... )
                >>> bt.read.parquet(target).sort("id").select("id", "city").to_pydict()
                {'id': [1, 2], 'city': ['SF', 'LA']}

                A later batch deletes a row; a stale change for it is ignored.

                >>> later = bt.from_pydict(
                ...     {
                ...         "id": [2, 1],
                ...         "city": ["LA", "OLD"],
                ...         "op": ["DELETE", "UPDATE"],
                ...         "seq": [4, 0],
                ...     }
                ... )
                >>> _ = later.scd.apply_changes(
                ...     target,
                ...     keys="id",
                ...     sequence_by="seq",
                ...     deletes=bt.col("op") == "DELETE",
                ...     columns=["id", "city"],
                ... )
                >>> bt.read.parquet(target).sort("id").select("id", "city").to_pydict()
                {'id': [1], 'city': ['SF']}

        Args:
            target: Path of the table to maintain. Created if it does not exist.
            keys: The natural key column(s) identifying a row across changes.
            sequence_by: Column ordering the changes for a key (a log offset, a commit
                timestamp, a version). Persisted in `target`.
            deletes: Predicate that is TRUE for a change representing a deletion. NULL
                counts as "not a delete". Omit if the feed carries no deletes.
            columns: The columns to store, excluding CDC control columns such as the
                operation. Defaults to every column of the feed. `keys` and
                `sequence_by` are always stored.
            format: Target format; inferred from the path when omitted.
            **opts: Forwarded to the writer.

        Returns:
            The `WriteManifest` of the rewritten target.

        Raises:
            PlanError: If a named column is absent from the feed, or the existing
                target's columns do not match the ones being applied.
        """
        from batcher.api.merge import cdc_stored_columns, compose_cdc_apply
        from batcher.api.session import read as _read
        from batcher.io.detect import detect_format
        from batcher.io.filesystem import resolve_filesystem

        key_list = _require_keys("apply_changes", keys)
        if not sequence_by:
            raise PlanError(
                "apply_changes(): needs a `sequence_by` column that orders changes for a "
                "key (a log offset, commit timestamp, or version)"
            )
        fmt = detect_format(target, format)
        stored = cdc_stored_columns(self._ds.columns, key_list, sequence_by, columns)

        exists = resolve_filesystem(target).exists(target)
        existing = _read(target, format=fmt) if exists else None
        result = compose_cdc_apply(self._ds, existing, key_list, sequence_by, stored, deletes)
        return result.write(target, fmt, mode="overwrite", **opts)
