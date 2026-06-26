"""The `Dataset.scd` namespace — slowly-changing-dimension upserts.

Breadth on `Dataset` lives on accessors. SCD maintenance composes existing ops
(merge / join / union / with_columns) — no new IR:

- ``type1`` — overwrite-in-place (no history): a keyed upsert.
- ``type2`` — full history via effective-dating columns (`valid_from`/`valid_to`/
  `is_current`): expire the current row of a changed key and append a new version.
- ``type3`` — keep the previous value in a ``<attr>_prev`` column.
"""

from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

from batcher.plan.expr_ir import Col, Expr, lit, nullif


def _typed_null(of: Expr) -> Expr:
    """A NULL of the same type as `of` (``nullif(x, x)`` is always NULL) — the engine
    has no typed null literal, so this is how a null column is built."""
    return nullif(of, of)


if TYPE_CHECKING:
    from batcher.api.dataset import Dataset
    from batcher.io.manifest import WriteManifest

__all__ = ["DatasetSCD"]


class DatasetSCD:
    """Accessor for slowly-changing-dimension upserts over a `Dataset` (``ds.scd``).

    The dataset is the *incoming* dimension snapshot (natural keys + attributes).
    """

    __slots__ = ("_ds",)

    def __init__(self, ds: Dataset) -> None:
        """Bind the SCD accessor to its `Dataset`; reached as `ds.scd`, not constructed directly."""
        self._ds = ds

    def type1(self, target: str, *, keys: str | list[str], **opts) -> WriteManifest:
        """SCD type 1 — overwrite changed attributes in place (no history). A keyed
        upsert into `target` (delegates to `ds.write.merge`).

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
        """
        return self._ds.write.merge(target, on=keys, when_matched="update", **opts)

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

        key_list = [keys] if isinstance(keys, str) else list(keys)
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
        """SCD type 3 — keep the immediately previous value of each `track` attribute
        in a ``<attr>_prev`` column (limited history). For a matched key the existing
        current value moves to ``<attr>_prev`` and the incoming value becomes current;
        new keys get NULL previous values; untouched target keys are preserved.

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

        key_list = [keys] if isinstance(keys, str) else list(keys)
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
