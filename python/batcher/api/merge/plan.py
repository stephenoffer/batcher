"""Deciding what a MERGE has to rewrite — the pruning half, with no side effects.

This is where the 10x is decided, and it is a *pure* decision: it reads file statistics and
returns a `MergePlan` saying which of the target's data files could contain one of the
source's keys. Nothing is written and nothing is deleted here, which is what lets a test (or
the benchmark) assert that pruning actually happened instead of inferring it from a stopwatch.

The premise: **a data file only has to be rewritten if it can contain one of the source's
keys**, and its Parquet footer proves whether it can (`io.stats.key_pruning`). Merging a
1,000-row change set into a 100M-row table touches the handful of files holding those 1,000
keys, not all of them — so the merge costs the change set, not the table.

## When pruning is unavailable

Pruning is sound only while a target row that no source key can reach is guaranteed to come
out unchanged. Three things break that, and each falls back to rewriting the whole target —
correct, just not fast:

* a ``WHEN NOT MATCHED BY SOURCE`` clause is *about* the rows the source never mentions, so
  it acts on precisely the rows pruning would have skipped (Delta and Snowflake pay the same
  price for the same reason);
* a single-*file* target has nothing to skip — the one file is the table;
* a format with no footer statistics can prove nothing.

`execute` performs whatever this module decides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.api.merge.clauses import NOT_MATCHED_BY_SOURCE, MergeClause
from batcher.api.merge.format import target_format

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset
    from batcher.io.stats.key_pruning import KeyDigest

__all__ = ["MergePlan", "plan_merge"]

# Only a format whose files carry footer statistics can be pruned; only these can be rewritten
# file-by-file under a tokenized name that will not collide with a survivor.
_PRUNABLE_FORMATS = frozenset({"parquet"})


class MergePlan:
    """What a merge decided to do — which target files it will rewrite, and which it will skip.

    Exposed so a caller (and the tests, and the benchmark) can assert that pruning actually
    happened, rather than inferring it from a wall-clock number.

    `rows_per_file` is the layout the merge writes its output back at. Without it, rewriting
    N files would emit **one** file, and a table would collapse to a single part after a few
    merges — taking its zone maps, and therefore all future pruning, with it. Preserving the
    incoming file size is what keeps a repeatedly-merged table fast.
    """

    __slots__ = ("rewritten", "rows_per_file", "single_file", "skipped", "total")

    def __init__(
        self,
        rewritten: list[str],
        skipped: list[str],
        *,
        single_file: bool = False,
        rows_per_file: int | None = None,
    ) -> None:
        self.rewritten = rewritten
        self.skipped = skipped
        self.total = len(rewritten) + len(skipped)
        self.single_file = single_file
        self.rows_per_file = rows_per_file

    @property
    def pruned(self) -> bool:
        """True when at least one target file was proven irrelevant and left untouched."""
        return bool(self.skipped)


def plan_merge(
    source: Dataset,
    target: str,
    keys: Sequence[str],
    clauses: Sequence[MergeClause],
    *,
    prune: bool = True,
    format: str | None = None,
) -> MergePlan | None:
    """Decide which of `target`'s files the merge must rewrite. None if the target is new.

    Pure decision, no write — which is what lets a test assert the pruning without timing
    it. See the module docstring for when pruning is unavailable.
    """
    from batcher.io.filesystem import resolve_filesystem

    keys = list(keys)
    fmt = target_format(target, format)

    # The cardinality check is a correctness requirement of MERGE, not of pruning, so it runs
    # first and on every shape — including the ones below that skip pruning entirely. Putting
    # it inside the pruning branch meant a single-*file* target silently accepted a source
    # with duplicate keys and fanned out the target row it matched.
    key_count = _check_source_cardinality(source, keys)

    fs = resolve_filesystem(target)
    if not fs.exists(target):
        return None

    files = _target_files(fs, target, fmt)
    # `files == [target]` iff the target names a file rather than a directory. A single-file
    # target has nothing to skip and nothing to swap: the one file *is* the table.
    if files == [target]:
        return MergePlan(rewritten=files, skipped=[], single_file=True)
    if fmt not in _PRUNABLE_FORMATS:
        return MergePlan(rewritten=files, skipped=[])

    from batcher.io.stats.columnar_footer import parquet_file_manifest

    # The footer manifest is read even when pruning is off, because the merge still needs the
    # target's file *layout* to write its output back at the same granularity.
    manifest = parquet_file_manifest(fs, files, keys)
    rows_per_file = _rows_per_file(manifest)

    if not _prunable(clauses, prune=prune):
        return MergePlan(rewritten=files, skipped=[], rows_per_file=rows_per_file)

    from batcher.io.stats.key_pruning import surviving_files

    digest = _source_digest(source, keys, key_count)
    touched = surviving_files(digest, manifest) if digest is not None else None
    if touched is None:  # nothing could be proven → rewrite everything
        return MergePlan(rewritten=files, skipped=[], rows_per_file=rows_per_file)
    kept = set(touched)
    return MergePlan(
        rewritten=touched,
        skipped=[f for f in files if f not in kept],
        rows_per_file=rows_per_file,
    )


def _rows_per_file(manifest: Any) -> int | None:
    """The target's current rows-per-file, so the rewrite reproduces its layout.

    The **max** rather than the mean: it is a cap, and sizing to the mean would split every
    already-average file in two, doubling the file count on each merge.
    """
    if manifest is None or "num_records" not in manifest.column_names:
        return None
    import pyarrow.compute as pc

    largest = pc.max(manifest.column("num_records")).as_py()
    return int(largest) if largest else None


def _prunable(clauses: Sequence[MergeClause], *, prune: bool) -> bool:
    """Whether skipping a target file could be **sound** for this merge.

    Pruning rests on one guarantee: a target row no source key can reach comes out
    unchanged. A ``WHEN NOT MATCHED BY SOURCE`` clause is *about* exactly those rows, so it
    voids the guarantee — every row of every file pruning would have skipped is a row that
    clause acts on. Delta and Snowflake fall back to a full scan for the same reason.
    """
    if not prune:
        return False
    return not any(c.kind == NOT_MATCHED_BY_SOURCE for c in clauses)


def _target_files(fs: Any, target: str, fmt: str) -> list[str]:
    """The target's current data files (the directory listing *is* the manifest here)."""
    from batcher.io.formats.base import SOURCES

    suffix = getattr(SOURCES.get(fmt), "suffix", "")
    try:
        return list(fs.expand(target, suffix=suffix))
    except (OSError, ValueError):
        return []


def _check_source_cardinality(source: Dataset, keys: list[str]) -> int:
    """Reject a source with two rows for one key, and return its distinct-key count.

    Two source rows claiming the same target row have no defined winner, and SQL requires a
    MERGE to reject that rather than pick one arbitrarily (the "cardinality violation").
    Because a *valid* source has one row per key, its row count and its distinct-key count
    are the same number — so comparing them is the whole check, and it hands back the key
    count the pruning digest needs anyway.

    Two aggregations, each reduced to a single number in the engine. Nothing but those
    numbers crosses to the driver.
    """
    key_count = source.select(*keys).distinct().count()
    if source.count() != key_count:
        raise PlanError(
            "merge(): the source has more than one row for a key, so a matched target row "
            "has no single row to merge from (SQL calls this a cardinality violation). "
            "Deduplicate the source first — e.g. "
            f".distinct(subset={keys!r}, keep='last', order_by=<sequence column>)."
        )
    return key_count


def _source_digest(source: Dataset, keys: list[str], key_count: int) -> KeyDigest | None:
    """The source's key digest — or None when its key set is too wide to be worth one.

    The sharp occupancy test needs the distinct keys *on the driver*, and a change set with
    tens of millions of them would be gigabytes there. Past `MAX_EXACT_KEYS` this returns
    None (meaning: rewrite everything) rather than collect them — which costs nothing real,
    because a key set that wide reaches into nearly every file of any table anyway. Pruning
    is for *selective* change sets; that is the whole premise, and this is where it is
    enforced rather than hoped for.
    """
    from batcher.io.stats.key_pruning import MAX_EXACT_KEYS, key_digest

    if key_count > MAX_EXACT_KEYS:
        return None
    return key_digest(source.select(*keys).distinct().collect())
