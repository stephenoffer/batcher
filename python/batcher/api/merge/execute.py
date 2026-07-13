"""Executing a MERGE: rewrite only the files that can match, and swap them atomically.

The composition in `compose` is correct against *any* target. The cost is decided here,
and it is the whole difference between an upsert that takes a second and one that takes
an hour: **a data file only has to be rewritten if it can contain one of the source's
keys**, and its Parquet footer proves whether it can (`io.stats.key_pruning`). Merging a
1,000-row change set into a 100M-row table touches the handful of files that hold those
1,000 keys, not all of them.

So a merge runs in three phases:

1. **Digest** — the source's distinct keys, plus the cardinality check SQL requires. This
   is a *control-plane* object: keys and counts, never a row of the merge, and it is
   capped, so a change set with a huge key space skips pruning rather than filling the
   driver with keys (`_source_digest`).
2. **Prune** — the target's per-file key bounds are tested against the digest. Files that
   cannot match are never opened, never read, never rewritten. They are simply left where
   they are, which is what makes the merge sublinear in the table's size.
3. **Rewrite and swap** — the surviving files are merged with the source (through the
   ordinary distributed executor — this is a plain relational plan) into new files
   carrying a unique write token, then the replaced files are deleted.

## When pruning is off

Pruning is *sound* only when a target row that no source key can reach is guaranteed to
come out unchanged. A ``WHEN NOT MATCHED BY SOURCE`` clause breaks exactly that
guarantee — it is *about* the rows the source never mentions — so its presence forces a
full rewrite. Delta and Snowflake pay the same price for the same reason. Likewise a
single-*file* target has nothing to skip (there is one file and the merge must rewrite
it), and a format with no footer statistics cannot prove anything.

Every one of those cases falls back to rewriting the whole target, which is what the
merge did before it could prune — correct, just not fast.

## Atomicity

A plain directory has no transaction log; the listing *is* the manifest. The swap writes
the new files first and deletes the replaced ones second, so a crash between the two
leaves the old and new copies of a key both present. That window is why this path is
documented single-writer, and why a table with concurrent writers belongs on a
transactional sink (Delta), where the commit is one atomic log append.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from batcher.api.merge.builder import simple_clauses
from batcher.api.merge.clauses import NOT_MATCHED, MergeClause
from batcher.api.merge.compose import compose_merge
from batcher.api.merge.format import target_format
from batcher.api.merge.native import NATIVE_MERGE_SINKS, native_merge
from batcher.api.merge.plan import MergePlan, plan_merge

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset
    from batcher.io.manifest import WriteManifest

__all__ = [
    "MergePlan",
    "execute_merge",
    "plan_merge",
    "run_merge",
    "target_format",
]


def execute_merge(
    writer: Any,
    target: str,
    *,
    on: str | list[str],
    when_matched: str,
    when_not_matched: str,
    format: str | None,
    opts: dict,
) -> WriteManifest:
    """Dispatch `Writer.merge`'s keyword form — the two-clause upsert — onto `run_merge`.

    The shorthand is just two clauses, so it compiles to them and takes the same path as
    the builder. It used to short-circuit a Delta target into a hard-coded
    ``update_all`` + ``insert_all``, which meant ``when_matched="delete"`` and
    ``when_not_matched="ignore"`` were accepted and then silently ignored — the merge did
    something other than what it was asked. Compiling to real clauses is what fixes that.
    """
    keys = [on] if isinstance(on, str) else list(on)
    return run_merge(
        writer._ds,
        target,
        keys,
        simple_clauses(when_matched, when_not_matched),
        format=target_format(target, format),
        opts=opts,
    )


def run_merge(
    source: Dataset,
    target: str,
    keys: Sequence[str],
    clauses: Sequence[MergeClause],
    *,
    prune: bool = True,
    format: str | None = None,
    opts: dict | None = None,
) -> WriteManifest:
    """Execute a merge of `source` into the table at `target` and commit it.

    Args:
        source: The change set.
        target: Path/URI of the table being merged into.
        keys: The columns matching a source row to a target row.
        clauses: The ordered ``WHEN …`` clauses.
        prune: Skip target files the source's keys provably cannot reach. Correctness does
            not depend on it; turning it off only makes the merge rewrite everything.
        format: Sink format override; inferred from `target` when omitted.
        opts: Extra write options forwarded to the sink.

    Returns:
        A `WriteManifest` of the data files written.
    """
    keys = list(keys)
    opts = dict(opts or {})
    fmt = target_format(target, format)

    # A transactional target runs the clauses through its own MERGE; the copy-on-write path
    # below cannot serve one at all (see `native`).
    if fmt in NATIVE_MERGE_SINKS:
        return native_merge(source, target, keys, clauses, fmt, opts)

    plan = plan_merge(source, target, keys, clauses, prune=prune, format=fmt)
    return _run(source, target, keys, clauses, fmt, plan, opts)


def _run(
    source: Dataset,
    target: str,
    keys: list[str],
    clauses: Sequence[MergeClause],
    fmt: str,
    plan: MergePlan | None,
    opts: dict,
) -> WriteManifest:
    """Compose the merge over the files `plan` selected, write it, and swap them in."""
    from batcher.api.session import read as _read
    from batcher.io.filesystem import resolve_filesystem

    if plan is None:
        return _write_new_table(source, target, keys, clauses, fmt, opts)

    # Read *only* the files that survived pruning. This is the whole win: the skipped files
    # are never opened, and they stay on storage exactly as they are.
    if plan.rewritten:
        current = _read_files(target, fmt, plan.rewritten)
    else:
        # Nothing in the target can match, so the merge is pure insert. Take the schema from
        # the target without reading a row of it.
        current = _read(target, format=fmt).limit(0)

    merged = compose_merge(source, current, keys, clauses)

    if plan.single_file:
        # The target *is* one file: there is nothing to skip and nothing to swap. Overwriting
        # it in place is correct and already atomic (temp file + rename).
        return merged.write(target, fmt, mode="overwrite", **opts)

    fs = resolve_filesystem(target)
    return _write_and_swap(merged, target, fmt, plan, fs, opts)


def _read_files(target: str, fmt: str, files: list[str]) -> Dataset:
    """A `Dataset` over an explicit subset of `target`'s data files.

    Pinning the source to a file *list* rather than the directory is also what makes the
    rewrite safe to run against the directory it is writing into: the reader cannot pick up
    the new files the write is landing beside it.
    """
    from batcher.api.session import _scan
    from batcher.io.formats.base import SOURCES

    return _scan(SOURCES.get(fmt)(target, files=files))


def _write_new_table(
    source: Dataset,
    target: str,
    keys: list[str],
    clauses: Sequence[MergeClause],
    fmt: str,
    opts: dict,
) -> WriteManifest:
    """The target does not exist: only the insert clauses can fire, against an empty table.

    Composing against ``source.limit(0)`` rather than special-casing the insert path keeps
    one code path — the clause chain, its conditions, and its column defaults all behave
    exactly as they would against a real (empty) table.
    """
    from batcher.io.manifest import WriteManifest

    if not any(c.kind == NOT_MATCHED for c in clauses):
        return WriteManifest()  # no insert clause ⇒ nothing can land in a table with no rows
    merged = compose_merge(source, source.limit(0), keys, clauses)
    return merged.write(target, fmt, mode="overwrite", **opts)


def _write_and_swap(
    merged: Dataset,
    target: str,
    fmt: str,
    plan: MergePlan,
    fs: Any,
    opts: dict,
) -> WriteManifest:
    """Write the merged rows as new, uniquely-named files, then delete the ones they replace.

    The new files carry a per-merge token, so they cannot collide with the files pruning
    proved irrelevant — those are still sitting in the directory and are still live data.
    Writing first and deleting second means a crash leaves *more* rows than it should, never
    fewer; the other order can lose data outright.

    The token rides in `sink_kwargs`, not on a sink object, because a distributed write
    reconstructs its sink on every worker — see `ParquetSink.suffix`.
    """
    from batcher.api.terminal.core import _write

    token = uuid.uuid4().hex[:12]
    options = _write_options(opts)
    options.setdefault("max_rows_per_file", plan.rows_per_file)
    manifest = _write(
        merged._plan,
        merged._sources,
        merged.columns,
        target,
        fmt,
        directory=True,
        sink_kwargs={"file_token": token},
        **options,
    )
    for path in plan.rewritten:
        _remove(fs, path)
    return manifest


def _write_options(opts: dict) -> dict:
    """The subset of write options a merge's rewrite honors.

    A merge writes *into* an existing table, so the options that would redefine the table's
    layout (`mode`, `sort_by`, `replace_where`, a streaming trigger) have no meaning here and
    are dropped rather than silently half-applied.
    """
    allowed = ("partition_by", "distributed", "num_workers", "max_rows_per_file")
    return {k: v for k, v in opts.items() if k in allowed}


def _remove(fs: Any, path: str) -> None:
    import contextlib

    with contextlib.suppress(OSError, ValueError, NotImplementedError):
        fs.remove(path)
