"""The `ds.write` namespace — typed, per-format dataset sinks.

``ds.write(path)`` infers the sink format from the path; ``ds.write.<format>(...)``
is the explicit spelling. Methods are thin wrappers over `terminal._write` and the
merge helpers; sink implementations live in `io/formats/` and register into the
`SINKS` registry.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from batcher.api.io_namespace._discovery import (
    PathLike,
    namespace_dir,
    namespace_repr,
    unknown_attribute,
)
from batcher.api.io_namespace._write_opts import (
    MODE_AWARE_SINKS as _MODE_AWARE_SINKS,
)
from batcher.api.io_namespace._write_opts import (
    derive_partition_columns,
    normalize_partition_by,
    normalize_save_mode,
    one_or_many,
    reject_row_index,
)
from batcher.api.session import read as _read
from batcher.io.sink import check_write_options

if TYPE_CHECKING:
    import pyarrow as pa

    from batcher.api.dataset import Dataset
    from batcher.api.merge import MergeBuilder
    from batcher.api.streaming import StreamingQuery
    from batcher.io.manifest import WriteManifest
    from batcher.plan.streaming import Trigger

__all__ = ["Writer"]


def _prune_stale_after_overwrite(
    path: str, fmt: str, manifest: WriteManifest, *, only_written_partitions: bool = False
) -> None:
    """Delete files a prior write left under `path` that this overwrite did not rewrite.

    A file sink writes ``part-NNNNN`` (and Hive ``col=v/…``) files whose *names* depend
    on the shard/chunk count and partition values of the current data — not the previous
    write's. So overwriting a 5-file output with a 2-file one, or a ``{a, b}``-partitioned
    table with an ``{a}``-only one, leaves the extra ``part-`` files / ``col=b`` directory
    in place, and the next read unions the stale rows back in (silent data corruption). A
    plain overwrite must *replace* the output, so any surviving file this write did not
    produce is deleted after the (atomic, hence complete) write commits.

    `only_written_partitions` narrows that to the partition directories this write
    actually produced files in — the dynamic partition overwrite (``mode=
    "overwrite_partitions"``). A partition the incoming data says nothing about is then
    left exactly as it was, which is what makes a daily reload of one day's data safe
    against a table holding five years of them.

    Runs only for the plain file sinks (`FileSink`): the lakehouse/warehouse sinks manage
    their own overwrite through a transaction log or a target table. Fails safe — if the
    manifest's own files are not all found in the listing (a path-normalization mismatch),
    nothing is deleted rather than risk removing a live file.
    """
    import contextlib
    import os.path

    from batcher._internal.errors import IOError as _IOError
    from batcher.io.base.sink import FileSink
    from batcher.io.filesystem import resolve_filesystem
    from batcher.io.sink import SINKS

    sink_cls = SINKS.get(fmt)
    if not (isinstance(sink_cls, type) and issubclass(sink_cls, FileSink)):
        return
    keep = {f.path for f in manifest.files}
    # A single-file write (`<path>` is the file itself) is replaced atomically in place —
    # there is no sibling directory of parts to prune.
    if not keep or keep == {path}:
        return
    # The format extension to list by, read from the files this write actually produced
    # (a sink's `suffix` can be an instance property — e.g. a per-write token — so it is
    # not reliably readable off the class). All shards of one write share the extension.
    suffixes = tuple(
        {os.path.splitext(f.path)[1] for f in manifest.files if os.path.splitext(f.path)[1]}
    )
    if not suffixes:
        return
    fs = resolve_filesystem(path)

    def _walk(directory: str) -> list[str]:
        found: list[str] = []
        # A directory with no matching data files (e.g. a partition root) raises.
        with contextlib.suppress(OSError, ValueError, _IOError):
            found.extend(fs.expand(directory, suffix=suffixes))
        for sub in fs.list_dirs(directory):
            found.extend(_walk(sub))
        return found

    # Dynamic partition overwrite: only look inside the directories this write wrote to,
    # so a partition the new data never mentioned is neither listed nor touched.
    roots = (
        sorted({f.path.rsplit("/", 1)[0] for f in manifest.files})
        if only_written_partitions
        else [path]
    )
    existing = {f for root in roots for f in _walk(root)}
    if not keep.issubset(existing):
        return  # listing did not surface our own files — do not risk a wrong deletion
    for stale in existing - keep:
        fs.remove(stale)
    # Then the directories those files were the last occupants of. Overwriting a table
    # partitioned by ``dt=`` with one partitioned by ``g=`` otherwise leaves the ``dt=``
    # directories standing, empty, and the tree advertises two partition schemes at once.
    from batcher.io.filesystem import prune_empty_dirs

    prune_empty_dirs(fs, path)


def _check_overwrite_partitions(fmt: str, partition_by: list[str] | None) -> None:
    """Refuse a dynamic partition overwrite the target cannot give the caller.

    Both refusals name the operation that *does* work, because the mode is reached by
    someone who already knows exactly which rows they mean to replace.

    Args:
        fmt: The sink format this write resolved to.
        partition_by: The partition columns the write names, if any.

    Raises:
        PlanError: On a transactional target, or with no partitioning to scope to.
    """
    from batcher._internal.errors import PlanError

    if fmt in _MODE_AWARE_SINKS:
        raise PlanError(
            "write(mode='overwrite_partitions') is for a Hive-partitioned file "
            f"directory; {fmt!r} is a transactional table, where the same intent "
            "is a scoped commit rather than a file rewrite. Use "
            "write(replace_where=<predicate over the partition columns>), which "
            "retires exactly the matching partitions inside one transaction."
        )
    if not partition_by:
        raise PlanError(
            "write(mode='overwrite_partitions') replaces only the partitions the "
            "incoming data covers, so it needs partition_by=[...] to know what a "
            "partition is. Without partitioning, that is a plain mode='overwrite'."
        )


def _undistributable_stream_reason(plan: Any) -> str | None:
    """Why the cluster cannot fold this streaming plan with single-node semantics, or None.

    A top-level `Aggregate` over a breaker-free input is exactly the shape the mergeable
    algebra covers: each worker runs `partial` on its share of the epoch and the driver
    `combine`s the partials into the running state. Anything else is **refused rather than run
    with different semantics than the single-node path** — which is invariant #7
    (single-node == distributed), and is not a slogan: the one shape that slipped through this
    gate silently returned a different answer on a cluster than on one box.
    """
    from batcher.plan.logical import Aggregate, TransformWithState, is_streamable

    if isinstance(plan, TransformWithState):
        # Named rather than folded into "another pipeline breaker": this one *has* a
        # mergeable form (a shuffle by the group keys, so each key's state lives on exactly
        # one worker and the partitions' key sets are disjoint) and the runner simply does
        # not implement it yet. A reader who is told "sort / join / window" will restructure
        # a plan that did not need restructuring.
        return (
            "transform_with_state has no distributed implementation yet: its mergeable "
            "form is a shuffle by the group keys, which the distributed runner does not "
            "do. Run it with distributed=False rather than have the cluster compute "
            "something else."
        )
    if not isinstance(plan, Aggregate) or not is_streamable(plan.input):
        return (
            "distributed streaming supports a stateless pipeline or a top-level "
            "streaming aggregation; this plan has another pipeline breaker "
            "(sort / join / window) — restructure it, or omit distributed."
        )
    if plan.watermark is not None:
        # The shape that slipped through. A watermarked aggregate is an `Aggregate` over a
        # streamable input, so the old gate waved it past — and `dist/` implements no
        # watermark at all (no window eviction, no late-row drop, no append mode). It
        # degraded to an unbounded complete-mode aggregate that re-emits the whole running
        # result every epoch and grows state forever: the same query, single-node vs
        # distributed, produced different results with no error and no warning.
        return (
            "distributed streaming does not implement event-time watermarks: the distributed "
            "runner has no window eviction, no late-row drop, and no append output mode, so a "
            "watermarked aggregation would silently degrade to an unbounded complete-mode "
            "aggregate and return a different result than the same query run single-node. Run "
            "it with distributed=False, or drop the watermark to accept complete-mode "
            "semantics."
        )
    return None


def _writes_into_a_partition_directory(path: str, single_file: bool) -> bool:
    """Whether `path` names a Hive partition directory rather than a file to create.

    A last segment of the form ``col=value`` is a partition directory everywhere in this
    ecosystem, and writing one partition of a table at a time is how a daily job appends:
    Spark's ``df.write.parquet("table/day=2024-01-01")`` puts ``part-*`` files inside it.
    Batcher wrote a *file* at that exact path instead, because the path carries no
    extension and a single-shard write goes straight to its destination. The result was a
    tree Batcher could not read back: ``table/day=2024-01-01`` is an extensionless file, so
    ``read.parquet("table")`` found no ``.parquet`` files at all and raised — a round trip
    broken through Batcher's own writer, on the layout a ported Spark job produces first.

    `single_file=True` is the caller saying they meant one file, and wins.

    Args:
        path: The write destination.
        single_file: Whether the caller explicitly asked for one file.

    Returns:
        True when the destination should hold ``part-*`` files.
    """
    from batcher.io.base._paths import hive_segment

    return not single_file and hive_segment(path) is not None


class Writer:
    """The `ds.write` namespace: callable for autodetect, typed methods per format.

    ``ds.write(path)`` infers the sink format from the path; ``ds.write.<format>(...)``
    is the explicit spelling. All methods accept `partition_by=` (Hive directory) and
    `distributed=`/`num_workers=` (parallel shard write + atomic commit), and return
    a `WriteManifest`.

    Examples:
        .. doctest::

            >>> import batcher as bt, tempfile, os
            >>> out = os.path.join(tempfile.mkdtemp(), "t.parquet")
            >>> _ = bt.from_pydict({"x": [1, 2, 3]}).write(out)
            >>> bt.read(out).count()
            3
    """

    __slots__ = ("_ds",)

    def __init__(self, ds: Dataset) -> None:
        """Bind the write namespace to the `Dataset` whose result it writes."""
        self._ds = ds

    def __repr__(self) -> str:
        """List the formats this namespace writes, grouped by family."""
        return namespace_repr(self, "ds.write")

    def __dir__(self) -> list[str]:
        """Every format method, so tab-completion shows the writable formats."""
        return namespace_dir(self)

    def __getattr__(self, name: str) -> Any:
        """Answer a misspelled format with a suggestion instead of a bare `AttributeError`.

        Only ever reached on a miss, so it cannot shadow a real method. A `_`-prefixed
        name still raises `AttributeError` — `copy`, `pickle`, and IPython probe for those
        and require a miss to look like a miss.
        """
        raise unknown_attribute(self, "ds.write", name)

    def __call__(
        self,
        path: PathLike,
        format: str | None = None,
        *,
        mode: str = "overwrite",
        partition_by: list[str | Any] | None = None,
        single_file: bool = False,
        distributed: bool | str = "auto",
        num_workers: int | None = None,
        resume: bool = False,
        max_rows_per_file: int | None = None,
        sort_by: str | list[str] | None = None,
        replace_where: Any = None,
        trigger: Trigger | None = None,
        output_mode: str = "append",
        checkpoint: str | None = None,
        query_name: str | None = None,
        auto_compact: bool = False,
        **opts: Any,
    ) -> WriteManifest | StreamingQuery:
        """Execute and write the result, inferring `format` from the path when omitted.

        **Batch vs streaming is one API.** With a bounded source and no `trigger`,
        this is a one-shot write returning a `WriteManifest` (the behavior below).
        If `trigger` is set, or any source is unbounded (a stream), the write runs as
        a streaming query — micro-batches are appended to `path` per the `trigger`
        cadence and `output_mode` — and it returns a `StreamingQuery` handle instead.

        `mode` is the save mode (Spark ``SaveMode`` parity):

        * ``"overwrite"`` (default) — write, replacing any existing output.
        * ``"overwrite_partitions"`` — replace only the partitions the incoming data
          covers and leave every other one untouched (Spark's
          ``partitionOverwriteMode="dynamic"``, Hive's ``INSERT OVERWRITE``). Needs
          `partition_by`, and is the safe way to reload one day into a table holding
          years of them: a plain ``"overwrite"`` would delete the rest.
        * ``"error"`` — raise `PlanError` if `path` already exists.
        * ``"ignore"`` — skip the write (return an empty manifest) if `path` exists.
        * ``"append"`` — add to an existing table; only the sinks that can add to one
          (`delta`/`iceberg`/`hudi`/`snowflake`) support it. A file sink raises, because
          it has nothing to append to.

        Spark's own ``"errorIfExists"`` and Python's file modes (``"w"``, ``"a"``,
        ``"x"``) are accepted as spellings of those, so a ported job does not fail
        on its last line. `partition_by` likewise answers to Spark's ``partitionBy=``
        and pandas' ``partition_cols=``, and pandas' ``index=False`` is accepted and
        dropped — Batcher has no row index, so there is nothing to suppress.

        A `partition_by` entry may be an **expression** rather than a column name, which
        is how a *partition transform* is spelled (Iceberg's ``days(ts)`` /
        ``bucket(16, id)``, Spark's generated partition column). The expression is
        evaluated once and partitioned on by its alias, so
        ``partition_by=[bt.col("ts").dt.year().alias("year")]`` writes ``year=2024/``
        directories without adding a column to the data or to the source table. An
        expression key must carry an ``.alias(...)``, since the alias is the directory
        name.

        ``single_file=True`` guarantees the output is the one file at `path` rather than
        a directory of shards: it refuses the arguments that shard (`partition_by`,
        `max_rows_per_file`) instead of silently ignoring them, and keeps the write on
        one worker.

        ``replace_where=<predicate>`` is a dynamic partition/range overwrite (Delta
        ``replaceWhere`` / the backfill pattern): atomically replace only the rows
        matching the predicate and keep the rest. This is predicate-scoped, not
        key-matched — for a key-matched upsert (update/insert by join key) use
        `merge` instead.

        ``sort_by="col"`` or ``sort_by=[cols]`` clusters the output: rows are sorted
        (ascending) before writing, so each file / row-group's min/max bounds are tight
        and downstream queries skip far more data via zonemaps and bloom filters — the engine-side
        slice of liquid clustering (you choose the keys; there is no managed table
        service). Bounded batch writes only.

        ``resume=True`` makes the write idempotent: output files already present
        (necessarily fully committed, since writes are atomic) are skipped, so a job
        re-run after a crash or spot preemption finishes only the unwritten shards —
        the resumability Ray Data lacks without external bookkeeping.

        **Correctness precondition (important):** resume identifies done work by file
        *position* (``part-NNNNN``), so it is exactly-once **only when the plan is
        deterministic** — the same input produces the same rows in the same order, so
        a given part file holds the same rows on every run. This holds for the
        read → ``map_batches``/``filter``/``select`` → write (ETL / batch-inference)
        path. It does **not** hold for a plan whose row→file assignment can vary
        between runs — a ``group_by``/``join``/``sort`` or distributed shuffle, where
        ordering is hash- and worker-count-dependent. Resuming such a plan could skip
        a file that now holds *different* rows, dropping or duplicating data. For
        those, write to a fresh path (no resume) or materialize a stable, keyed
        intermediate first.

        Examples:
            .. doctest::

                >>> import batcher as bt, os, tempfile
                >>> out = os.path.join(tempfile.mkdtemp(), "t.parquet")
                >>> _ = bt.from_pydict({"x": [1, 2, 3]}).write(out)
                >>> bt.read(out).count()
                3
        """
        from batcher._internal.errors import PlanError
        from batcher.api.terminal import _write
        from batcher.io.base._paths import normalize_path
        from batcher.io.detect import detect_format, hive_partition_keys
        from batcher.io.manifest import WriteManifest
        from batcher.io.source import is_bounded

        # Normalize the whole call vocabulary once, here: every typed method
        # (`.parquet`, `.delta`, …) funnels through this call, so a spelling accepted
        # here is accepted everywhere, and the sink below only ever sees canonical names.
        path = normalize_path(path, what="write path")
        mode = normalize_save_mode(mode)
        partition_by = normalize_partition_by(opts, partition_by)
        reject_row_index(opts)
        if partition_by and any(not isinstance(k, str) for k in partition_by):
            # A partition transform (`col('ts').dt.year().alias('year')`) becomes an
            # ordinary derived column, then re-enters this call with a plain name list —
            # so every layer below sees the column-name partitioning it already handles.
            derived, names = derive_partition_columns(self._ds, partition_by)
            return derived.write(
                path,
                format,
                mode=mode,
                partition_by=names,
                single_file=single_file,
                distributed=distributed,
                num_workers=num_workers,
                resume=resume,
                max_rows_per_file=max_rows_per_file,
                sort_by=sort_by,
                replace_where=replace_where,
                trigger=trigger,
                output_mode=output_mode,
                checkpoint=checkpoint,
                query_name=query_name,
                auto_compact=auto_compact,
                **opts,
            )
        fmt = detect_format(path, format)
        # Before anything is provisioned, sorted or written: a write keyword the sink
        # cannot take is a typo, and saying so here costs nothing where letting it reach a
        # Ray worker's constructor costs a provisioned cluster to say the same thing worse.
        check_write_options(fmt, opts)
        if single_file:
            self._check_single_file(partition_by, max_rows_per_file, path, fmt)
            distributed = False

        # `sort_by` clusters the output: sort rows (ascending) before writing so each
        # file / row-group's min/max bounds are tight, maximizing downstream zonemap +
        # bloom skipping (the engine-side slice of liquid clustering — no managed table
        # service). Delegate to a sorted dataset so all the write logic below runs on it.
        # A global sort is a bounded-batch notion; refuse it on an unbounded stream.
        if sort_by is not None:
            if trigger is not None or any(not is_bounded(s) for s in self._ds._sources):
                raise PlanError(
                    "write(sort_by=...) clusters a bounded batch write; it cannot sort an "
                    "unbounded stream"
                )
            return self._ds.sort(*one_or_many(sort_by)).write(
                path,
                format,
                mode=mode,
                partition_by=partition_by,
                single_file=single_file,
                distributed=distributed,
                num_workers=num_workers,
                resume=resume,
                max_rows_per_file=max_rows_per_file,
                replace_where=replace_where,
                output_mode=output_mode,
                checkpoint=checkpoint,
                query_name=query_name,
                auto_compact=auto_compact,
                **opts,
            )

        # Unified surface: a trigger or an unbounded source means this is a streaming
        # write — append micro-batches to `path` and return a StreamingQuery.
        #
        # The distributed *stream* drain stays EXPLICIT opt-in (`distributed=True`), not
        # `"auto"`: it supports only a stateless `available_now()` backfill and raises for
        # a checkpointed / continuous / stateful stream. Letting `"auto"` resolve it True
        # on a cluster would turn those into errors for users who never asked to
        # distribute. The batch path below resolves `"auto"` normally.
        if trigger is not None or any(not is_bounded(s) for s in self._ds._sources):
            # A path (file/Delta) sink can only *append* micro-batches. In "complete"/"update"
            # mode a streaming aggregate re-emits its full running result every micro-batch (the
            # sink is meant to replace/upsert it — `MemoryStreamSink` does), so an append-only path
            # sink writes each running snapshot as another part file and readback silently
            # **duplicates** the result across files (`streaming != batch`). Reject it — Spark's
            # rule: file sinks support append only — rather than produce wrong data.
            if output_mode in ("complete", "update"):
                from batcher._internal.errors import PlanError

                raise PlanError(
                    f"streaming write to a path sink ({path!r}) supports output_mode='append' "
                    f"only, not {output_mode!r}: a file/Delta sink appends each micro-batch, so a "
                    f"running {output_mode!r} aggregate would be duplicated across part files. Use "
                    "output_mode='append', a memory sink (.write.memory(name, "
                    "output_mode='complete')), or .write.for_each_batch(fn) for a custom upsert."
                )
            if distributed is True:
                if max_rows_per_file is not None:
                    raise PlanError(
                        "write(max_rows_per_file=..., distributed=True) is not supported for "
                        "a streaming write: the distributed drain names each file after its "
                        "epoch and shard, and does not subdivide one further. Run the stream "
                        "single-node to cap the file size, or compact the output afterwards "
                        "with bt.compact(path, target_size_mb=...)."
                    )
                drain = self._maybe_distributed_stream(
                    path, fmt, opts, trigger, checkpoint, num_workers, query_name, output_mode
                )
                if drain is not None:
                    return drain
            sink = self._stream_sink_for(path, fmt, opts, query_name, max_rows_per_file)
            return self._start_stream(sink, trigger, output_mode, query_name, checkpoint)

        # Resume is exactly-once only on a deterministic plan: the same input must
        # produce the same rows in the same part file on every run. A pipeline breaker
        # (group_by, join, sort, distinct, window, union, limit) or its shuffle makes the
        # row-to-file assignment vary between runs, so resuming could skip a file that now
        # holds different rows, silently dropping or duplicating data. Refuse it up front.
        # The ETL and batch-inference path (read, map_batches, filter, select, write) stays
        # streamable, so it still resumes.
        if resume:
            from batcher.plan.logical import is_streamable

            if not is_streamable(self._ds._plan):
                raise PlanError(
                    "write(resume=True) is exactly-once only on a deterministic plan "
                    "(read → map_batches / filter / select → write). This plan contains a "
                    "group_by / join / sort / distinct / window / union / limit, whose "
                    "row→file assignment can vary between runs, so resuming risks dropping "
                    "or duplicating rows. Write to a fresh path without resume, or "
                    "materialize a stable keyed intermediate first."
                )

        # `replace_where` = dynamic partition/range overwrite (Delta `replaceWhere` / the
        # backfill pattern): atomically replace only the rows matching the predicate and
        # keep the rest.
        #
        # On a **Delta** table this is a scoped commit: the workers write the new
        # partition's files and the driver retires exactly the matching partitions from the
        # log. Nothing else is read and nothing else is rewritten, so backfilling one day of
        # a 100 TB table costs one day. The copy-on-write path below cannot do that — it
        # reads the *whole* table, filters out the replaced range, unions, and overwrites
        # everything, which turns a one-day backfill into a full-table rewrite. That is what
        # every lakehouse target used to do.
        if replace_where is not None:
            from batcher.io.filesystem import resolve_filesystem

            if fmt in ("delta", "iceberg"):
                # A lakehouse target scopes the overwrite to the predicate, inside its own
                # transaction. Iceberg used to fall through the `exists(path)` check below —
                # an Iceberg "path" is a catalog identifier, not a file, so the check was
                # always False, `replace_where` was silently dropped, and the write ran as a
                # plain overwrite that DELETED THE REST OF THE TABLE.
                opts = dict(opts)
                opts["replace_where"] = replace_where.to_ir()
                mode = "overwrite"
            elif resolve_filesystem(path).exists(path):
                kept = _read(path, format=fmt).filter(~replace_where)
                # `union` is positional, and a partitioned read hands its partition columns
                # back *last* (they come from the directory names, not the files), so the
                # kept rows and the incoming ones agree on names and disagree on order.
                # Align to the incoming dataset — it is the one whose column order the
                # rewritten table should keep.
                if set(kept.columns) == set(self._ds.columns) and kept.columns != self._ds.columns:
                    kept = kept.select(*self._ds.columns)
                combined = kept.union(self._ds)
                # A backfill replaces *rows*, never the table's organization. Carry the
                # existing Hive layout forward when the call did not name one, or the
                # rewrite lands the whole table flat and the partitioning the caller was
                # backfilling *into* is gone — with every row still present, so nothing
                # fails until the next partition-pruned query reads the lot.
                if partition_by is None:
                    partition_by = hive_partition_keys(path) or None
                # Forward the execution options too, not just the layout ones. A
                # copy-on-write `replace_where` rewrites the *whole* table, so it is the
                # write most in need of the cluster — and dropping `distributed` here ran
                # exactly that rewrite on the driver alone, on the largest input of any
                # write shape. (`sort_by` is already None here: its branch above re-enters
                # this call on the sorted dataset.)
                return combined.write(
                    path,
                    fmt,
                    mode="overwrite",
                    partition_by=partition_by,
                    single_file=single_file,
                    distributed=distributed,
                    num_workers=num_workers,
                    max_rows_per_file=max_rows_per_file,
                    auto_compact=auto_compact,
                    **opts,
                )
        if mode == "overwrite_partitions":
            _check_overwrite_partitions(fmt, partition_by)
        if mode == "append" and fmt not in _MODE_AWARE_SINKS:
            raise PlanError(
                f"write(): mode='append' is only supported for {sorted(_MODE_AWARE_SINKS)}, "
                f"not {fmt!r}: a plain file sink has no table to add to, so appending would "
                "mean rewriting the whole output. Either write each batch to its own path "
                "under a directory and read the directory back as one relation "
                f"(write(f'{{root}}/batch-{{n}}.{fmt}'), then read(root)), or use a "
                "transactional sink — write.delta / write.iceberg — where append is a "
                "real commit. (Not write.hudi: Batcher reads Hudi but cannot write it.)"
            )
        # error/ignore are a pre-write existence gate (resume has its own per-file
        # idempotence, so it is exempt).
        if mode in ("error", "ignore") and not resume:
            from batcher.io.filesystem import resolve_filesystem

            if resolve_filesystem(path).exists(path):
                if mode == "error":
                    raise PlanError(f"write(): path {path!r} already exists and mode='error'")
                return WriteManifest()  # ignore: leave the existing output untouched

        # The lakehouse sinks consume append/overwrite as a constructor option; the
        # file sinks always overwrite, so `mode` only drives the gate above for them.
        sink_kwargs = dict(opts)
        if fmt in _MODE_AWARE_SINKS:
            sink_kwargs["mode"] = mode if mode in ("append", "overwrite") else "overwrite"

        # A `repartition(...)` layout (set via ds.repartition) supplies write defaults:
        # `by` Hive-partitions, `num_files`/`target_size_mb` set the per-file row cap
        # (resolved post-materialization in `_write`).
        num_files: int | None = None
        target_bytes: int | None = None
        spec = self._ds._repartition
        # `single_file=True` is the caller overriding the layout at the write, so a
        # standing `repartition(num_files=...)` must not silently shard it back apart.
        if spec is not None and not single_file:
            if spec.by and partition_by is None:
                partition_by = list(spec.by)
            num_files = spec.num_files
            if spec.target_size_mb is not None:
                target_bytes = int(spec.target_size_mb * 1024 * 1024)

        manifest = _write(
            self._ds._plan,
            self._ds._sources,
            self._ds.columns,
            path,
            fmt,
            auto_compact=auto_compact,
            partition_by=partition_by,
            distributed=distributed,
            num_workers=num_workers,
            resume=resume,
            max_rows_per_file=max_rows_per_file,
            num_files=num_files,
            target_bytes_per_file=target_bytes,
            directory=_writes_into_a_partition_directory(path, single_file),
            sink_kwargs=sink_kwargs,
        )
        # Overwrite must REPLACE the output: drop any stale files a prior, differently
        # shaped write left under `path` that this write did not rewrite (else the next
        # read unions them back in). `resume` is exempt — it intentionally keeps and skips
        # already-present files. Only the plain file sinks need this; the mode-aware sinks
        # overwrite through their own log/table.
        overwriting = mode in ("overwrite", "overwrite_partitions")
        if overwriting and not resume and fmt not in _MODE_AWARE_SINKS:
            _prune_stale_after_overwrite(
                path, fmt, manifest, only_written_partitions=mode == "overwrite_partitions"
            )
        return manifest

    @staticmethod
    def _check_single_file(
        partition_by: list[str] | None, max_rows_per_file: int | None, path: str, fmt: str
    ) -> None:
        """Refuse anything that would stop a `single_file=True` write being one file.

        `partition_by` and `max_rows_per_file` both produce a *directory* of ``part-*``
        files, which is the one layout `single_file` exists to rule out. Dropping them
        silently would hand back the sharded output the caller just said they did not want
        — and the caller usually wants one file because something downstream (a
        spreadsheet, a script, a tool that takes a filename) cannot open a directory.

        A destination that *already is* a directory is the same conflict arriving from the
        other side: a write there is a directory rewrite, so the promise cannot be kept.
        Said plainly here rather than left to fail as an `IsADirectoryError` from inside a
        rename, three layers down.
        """
        from batcher._internal.errors import PlanError

        conflict = None
        if partition_by:
            conflict = "partition_by"
        elif max_rows_per_file:
            conflict = "max_rows_per_file"
        if conflict is not None:
            raise PlanError(
                f"write(single_file=True, {conflict}=...): {conflict} writes a directory of "
                "part files, which is what single_file rules out. Pass one or the other."
            )
        # Only a file sink has a *location* to be occupied. A warehouse or table sink's
        # "path" is an identifier, so stat-ing it would ask a question about the local
        # working directory that has nothing to do with the write.
        from batcher.io.base.sink import FileSink
        from batcher.io.filesystem import resolve_filesystem
        from batcher.io.sink import SINKS

        sink_cls = SINKS.get(fmt)
        if not (isinstance(sink_cls, type) and issubclass(sink_cls, FileSink)):
            return
        try:
            occupied = resolve_filesystem(path).is_dir(path)
        except Exception:
            occupied = False
        if occupied:
            raise PlanError(
                f"write(single_file=True): {path!r} is already a directory, so one file "
                "cannot be written at that exact path. Write to a new path, or remove the "
                "directory first."
            )

    # --- streaming sink targets -------------------------------------------
    def _start_stream(
        self,
        sink: Any,
        trigger: Trigger | None,
        output_mode: str,
        query_name: str | None,
        checkpoint: str | None = None,
    ) -> StreamingQuery:
        """Launch a streaming query writing this dataset's stream to `sink`."""
        from batcher.api.streaming import start_streaming_query

        return start_streaming_query(
            self._ds._plan,
            self._ds._sources,
            sink,
            trigger=trigger,
            output_mode=output_mode,
            name=query_name,
            checkpoint=checkpoint,
        )

    def _maybe_distributed_stream(
        self,
        path: str,
        fmt: str,
        opts: dict[str, Any],
        trigger: Trigger | None,
        checkpoint: str | None,
        num_workers: int | None,
        query_name: str | None,
        output_mode: str = "append",
    ) -> StreamingQuery | None:
        """Run a `distributed=True` streaming write across the cluster, if eligible.

        Two distributed shapes, one for each kind of trigger:

        * a **drain** (`available_now`/`once`) over a bounded, splittable source is a
          parallel backfill — every worker drains its own partition once (Spark's
          `Trigger.AvailableNow`);
        * a **continuous / processing-time** stream runs each micro-batch as a cluster-wide
          epoch: the workers stage the epoch's data files and the driver publishes them as
          a single transaction, so the log records one transaction per micro-batch and a
          replayed epoch adds neither a row nor a commit.

        A benign mismatch (a source with nothing to fan out) returns ``None`` so the caller
        falls back to the single-node engine. A request we cannot honor *correctly* raises
        `PlanError` rather than silently ignoring the flag — or, worse, quietly delivering a
        weaker guarantee than the API implies.
        """
        from batcher._internal.errors import PlanError
        from batcher.api.streaming import _is_stateless
        from batcher.dist.executor import _is_splittable_source
        from batcher.io.source import is_bounded

        srcs = self._ds._sources
        if len(srcs) != 1:
            return None  # not worth (or not able to) fan out — fall back to single-node
        source = srcs[0]

        # Probing an *unbounded* source with `splits()` is not free: on an incremental file
        # source a listing IS the discovery pass, so asking "could this be split?" would
        # consume the very files the epoch was about to read. Unbounded sources therefore
        # declare `partitionable` instead of being interrogated.
        splittable = (
            _is_splittable_source(source)
            if is_bounded(source)
            else getattr(source, "partitionable", False)
        )
        if not splittable:
            return None

        drain = trigger is not None and trigger.is_drain
        if drain and checkpoint is None and is_bounded(source):
            from batcher.api.streaming import start_distributed_stream_drain

            return start_distributed_stream_drain(
                self._ds._plan,
                srcs,
                path,
                fmt,
                opts,
                self._ds.columns,
                num_workers=num_workers,
                name=query_name,
            )

        if fmt in _MODE_AWARE_SINKS and fmt != "delta":
            raise PlanError(
                f"distributed streaming to {fmt!r} is not supported: its writer has no "
                "transaction-id check, so a replayed micro-batch would duplicate rows "
                "instead of being recognized as already-committed. Write to Delta for an "
                "exactly-once distributed stream, or run single-node (distributed=False)."
            )
        if not _is_stateless(self._ds._plan) and trigger is not None:
            if trigger.kind == "continuous":
                raise PlanError(
                    "a continuous trigger supports only stateless pipelines (filter / "
                    "select / map_batches); a streaming aggregation needs a micro-batch "
                    "boundary to fold — use Trigger.processing_time(...)"
                )
            reason = _undistributable_stream_reason(self._ds._plan)
            if reason is not None:
                raise PlanError(reason)
        from batcher.api.streaming import start_distributed_stream
        from batcher.plan.streaming import Trigger as _Trigger

        return start_distributed_stream(
            self._ds._plan,
            srcs,
            path,
            fmt,
            opts,
            trigger=trigger or _Trigger.processing_time(0),
            output_mode=output_mode,
            name=query_name,
            checkpoint=checkpoint,
            num_workers=num_workers,
        )

    def _stream_sink_for(
        self,
        path: str,
        fmt: str,
        opts: dict[str, Any],
        query_name: str | None = None,
        max_rows_per_file: int | None = None,
    ) -> Any:
        """Build the per-micro-batch `StreamSink` for a path/format streaming write.

        `query_name` becomes the transactional sink's ``txn`` application id, which is
        what makes a restarted query's replayed micro-batch idempotent — so it has to
        reach the sink, not just the query engine.

        `fmt` reaches the sink too, and used not to: every mode-aware format was handed to
        a Delta-pinned sink, so a streaming ``format="iceberg"`` write produced a Delta
        table at the Iceberg path, with the right rows and no error anywhere.
        """
        from batcher._internal.errors import PlanError
        from batcher.io.formats.streaming.sinks import FileStreamSink, TransactionalStreamSink

        if fmt in _MODE_AWARE_SINKS:
            if max_rows_per_file is not None:
                raise PlanError(
                    f"write(max_rows_per_file=...) has no meaning for a streaming {fmt!r} "
                    "write: each micro-batch is one transaction, and the file layout inside "
                    "it belongs to the table. Compact the table instead — "
                    "bt.compact(path, target_size_mb=...) — or write with auto_compact=True."
                )
            return TransactionalStreamSink(path, fmt, query_name=query_name, **opts)
        return FileStreamSink(path, fmt, max_rows_per_file=max_rows_per_file, **opts)

    def console(
        self,
        *,
        trigger: Trigger | None = None,
        output_mode: str = "append",
        num_rows: int = 20,
        truncate: bool | int = True,
        query_name: str | None = None,
        checkpoint: str | None = None,
    ) -> StreamingQuery:
        """Stream each micro-batch to stdout (development sink).

        Args:
            trigger: Micro-batch cadence; a one-shot batch when omitted.
            output_mode: Streaming output mode (``"append"``/``"complete"``/``"update"``).
            num_rows: Rows to print per micro-batch (default 20).
            truncate: Shorten string cells for display (Spark's ``truncate``). ``True``
                keeps 20 characters, an int keeps that many, ``False`` prints them whole.
                Display only; nothing downstream sees the shortened value.
            query_name: Optional name for the streaming query.
            checkpoint: Optional checkpoint location for offset tracking.

        Returns:
            A `StreamingQuery` handle for the running console stream.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> stream = bt.read.json("events/", stream=True)  # doctest: +SKIP
                >>> query = stream.write.console(  # doctest: +SKIP
                ...     num_rows=10, trigger=bt.Trigger.processing_time("5 seconds")
                ... )
                >>> query.await_termination()  # doctest: +SKIP
        """
        from batcher.io.formats.streaming.sinks import ConsoleStreamSink

        return self._start_stream(
            ConsoleStreamSink(num_rows=num_rows, truncate=truncate),
            trigger,
            output_mode,
            query_name,
            checkpoint,
        )

    def memory(
        self,
        name: str,
        *,
        trigger: Trigger | None = None,
        output_mode: str = "append",
        query_name: str | None = None,
        checkpoint: str | None = None,
    ) -> StreamingQuery:
        """Stream into an in-memory table queryable via ``bt.read_memory(name)``.

        Args:
            name: Name of the in-memory table to write into.
            trigger: Micro-batch cadence; a one-shot batch when omitted.
            output_mode: Streaming output mode (``"append"``/``"complete"``/``"update"``).
            query_name: Optional name for the streaming query.
            checkpoint: Optional checkpoint location for offset tracking.

        Returns:
            A `StreamingQuery` handle for the running in-memory stream.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> query = ds.write.memory("scratch")
                >>> _ = query.await_termination()
                >>> bt.read_memory("scratch").count()
                3
        """
        from batcher._internal.errors import PlanError
        from batcher.io.formats.streaming.sinks import MemoryStreamSink

        # The plan's output schema goes to the sink, so a stream that matches nothing still
        # reads back as this query's relation rather than as a table with no columns.
        #
        # But deriving that schema can require *executing* a `limit(0)` — a `map_batches`
        # output schema is not knowable statically — and materializing is exactly what an
        # unbounded source refuses. That refusal made `write.memory()` raise on every
        # streaming query whose schema was not statically derivable, which is the case this
        # sink exists for. `MemoryStreamSink` treats the schema as optional and learns it from
        # the first batch it writes, so hand it `None` rather than failing the query. Nothing
        # is swallowed by this: a plan that is invalid for any other reason still raises when
        # the query runs it.
        try:
            schema = self._ds.schema
        except PlanError:
            schema = None

        return self._start_stream(
            MemoryStreamSink(name, output_mode=output_mode, schema=schema),
            trigger,
            output_mode,
            query_name,
            checkpoint,
        )

    def for_each_batch(
        self,
        fn: Callable[[pa.Table, int], Any],
        *,
        trigger: Trigger | None = None,
        output_mode: str = "append",
        query_name: str | None = None,
        checkpoint: str | None = None,
    ) -> StreamingQuery:
        """Stream each micro-batch into a user callback ``fn(table, batch_id)``.

        The callback gets the whole Arrow table for the micro-batch — the sanctioned
        hook for custom upserts (``MERGE``/SCD), multi-sink fan-out, or any per-batch
        commit logic (the sink-side twin of `map_batches`).

        Args:
            fn: Callback ``fn(table, batch_id)`` invoked once per micro-batch.
            trigger: Micro-batch cadence; a one-shot batch when omitted.
            output_mode: Streaming output mode (``"append"``/``"complete"``/``"update"``).
            query_name: Optional name for the streaming query.
            checkpoint: Optional checkpoint location for offset tracking.

        Returns:
            A `StreamingQuery` handle for the running stream.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> def upsert(table, batch_id):
                ...     print(f"batch {batch_id}: {table.num_rows} rows")
                >>> stream = bt.read.json("events/", stream=True)  # doctest: +SKIP
                >>> query = stream.write.for_each_batch(upsert)  # doctest: +SKIP
        """
        from batcher.io.formats.streaming.sinks import ForeachBatchStreamSink

        return self._start_stream(
            ForeachBatchStreamSink(fn), trigger, output_mode, query_name, checkpoint
        )

    def noop(
        self,
        *,
        trigger: Trigger | None = None,
        output_mode: str = "append",
        query_name: str | None = None,
        checkpoint: str | None = None,
    ) -> StreamingQuery:
        """Run the pipeline and discard its output (Spark ``format("noop")``).

        The benchmark sink. Measuring a pipeline through a real sink measures the sink too
        — a Parquet write is compression and fsyncs, the console sink is a terminal, the
        memory sink grows until the box dies. Discarding the rows leaves the read, the
        transform and the engine, which is the thing under test.

        The rows are still counted, so `recent_progress` reports what the query processed.
        A benchmark sink that swallowed the count would make "this is fast" and "this
        produced nothing" look identical.

        Args:
            trigger: Micro-batch cadence; a one-shot batch when omitted.
            output_mode: Streaming output mode (``"append"``/``"complete"``/``"update"``).
            query_name: Optional name for the streaming query.
            checkpoint: Optional checkpoint location for offset tracking.

        Returns:
            A `StreamingQuery` handle for the running (output-discarding) stream.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> demo = bt.read.rate_micro_batch(100, num_rows=1000)
                >>> query = demo.write.noop(trigger=bt.Trigger.available_now())
                >>> query.await_termination()
                True
                >>> sum(p.num_input_rows for p in query.recent_progress)
                1000
        """
        from batcher.io.formats.streaming.sinks import NoopStreamSink

        return self._start_stream(NoopStreamSink(), trigger, output_mode, query_name, checkpoint)

    def kafka(
        self,
        topic: str | None = None,
        *,
        bootstrap_servers: str = "localhost:9092",
        trigger: Trigger | None = None,
        output_mode: str = "append",
        query_name: str | None = None,
        checkpoint: str | None = None,
        **options: Any,
    ) -> StreamingQuery:
        """Publish each row of each micro-batch to Kafka (Spark ``format("kafka")``).

        The write side of `bt.read.kafka`, and the column contract is Spark's: a ``value``
        column is required, ``key`` / ``topic`` / ``partition`` / ``headers`` are optional,
        and `topic` here is the destination for rows that carry no ``topic`` column.
        Payload columns may be binary or string.

        Delivery is at-least-once. Each micro-batch is flushed and acknowledged before it
        is reported written, so a failure replays rather than losing rows — but a replayed
        epoch republishes, so downstream consumers must be idempotent or dedup on the key.

        Args:
            topic: Destination topic for rows with no ``topic`` column.
            bootstrap_servers: The Kafka bootstrap servers.
            trigger: Micro-batch cadence; a one-shot batch when omitted.
            output_mode: Streaming output mode (``"append"``/``"complete"``/``"update"``).
            query_name: Optional name for the streaming query.
            checkpoint: Optional checkpoint location for offset tracking.
            options: Further ``confluent-kafka`` producer configuration; underscores
                become dots, so ``compression_type="zstd"`` sets ``compression.type``.

        Returns:
            A `StreamingQuery` handle for the running Kafka write.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> events = bt.read.kafka("raw", stream=True)  # doctest: +SKIP
                >>> query = events.select(  # doctest: +SKIP
                ...     value=bt.col("value")
                ... ).write.kafka("clean", bootstrap_servers="broker:9092")
        """
        from batcher.io.formats.streaming.kafka_sink import KafkaStreamSink

        return self._start_stream(
            KafkaStreamSink(topic=topic, bootstrap_servers=bootstrap_servers, **options),
            trigger,
            output_mode,
            query_name,
            checkpoint,
        )

    def for_each(
        self,
        fn: Callable[[dict[str, Any]], Any],
        *,
        trigger: Trigger | None = None,
        output_mode: str = "append",
        query_name: str | None = None,
        checkpoint: str | None = None,
    ) -> StreamingQuery:
        """Stream each row of each micro-batch into a user callback ``fn(row)``.

        `fn` may also be a `ForeachWriter` — an object with `open(partition_id, epoch_id)`,
        `process(row)`, and `close(error)` (Spark's shape). That is what a destination
        needing a connection wants: `open` acquires it, `close` releases it even when the
        epoch failed. A bare function has nowhere to put one.

        Args:
            fn: Callback ``fn(row)`` invoked once per row (``row`` a dict), or a
                `ForeachWriter` whose `process` receives each row.
            trigger: Micro-batch cadence; a one-shot batch when omitted.
            output_mode: Streaming output mode (``"append"``/``"complete"``/``"update"``).
            query_name: Optional name for the streaming query.
            checkpoint: Optional checkpoint location for offset tracking.

        Returns:
            A `StreamingQuery` handle for the running stream.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> def send(row):
                ...     print(row)
                >>> stream = bt.read.json("events/", stream=True)  # doctest: +SKIP
                >>> query = stream.write.for_each(send)  # doctest: +SKIP
        """
        from batcher.io.formats.streaming.sinks import ForeachStreamSink

        return self._start_stream(
            ForeachStreamSink(fn), trigger, output_mode, query_name, checkpoint
        )

    # --- File / object-store formats (path-addressed) ----------------------
    def parquet(self, path: PathLike, *, compression: str = "zstd", **opts: Any) -> WriteManifest:
        """Write as Parquet (see `__call__` for `partition_by`/`distributed`).

        Args:
            path: Output path/URI (file or directory) to write to.
            compression: Parquet compression codec (default ``"zstd"``).
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the files written.

        Examples:
            .. doctest::

                >>> import batcher as bt, os, tempfile
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> out = os.path.join(tempfile.mkdtemp(), "t")
                >>> _ = ds.write.parquet(out)
                >>> bt.read.parquet(out).count()
                3
        """
        return self(path, "parquet", compression=compression, **opts)

    def csv(self, path: PathLike, **opts: Any) -> WriteManifest:
        """Write as CSV.

        Args:
            path: Output path/URI (file or directory) to write to.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the files written.

        Examples:
            .. doctest::

                >>> import batcher as bt, os, tempfile
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> out = os.path.join(tempfile.mkdtemp(), "t")
                >>> _ = ds.write.csv(out)
                >>> bt.read.csv(out).count()
                3
        """
        return self(path, "csv", **opts)

    def json(self, path: PathLike, **opts: Any) -> WriteManifest:
        """Write as newline-delimited JSON.

        Args:
            path: Output path/URI (file or directory) to write to.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the files written.

        Examples:
            .. doctest::

                >>> import batcher as bt, os, tempfile
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> out = os.path.join(tempfile.mkdtemp(), "t")
                >>> _ = ds.write.json(out)
                >>> bt.read.json(out).count()
                3
        """
        return self(path, "json", **opts)

    def orc(self, path: PathLike, **opts: Any) -> WriteManifest:
        """Write as ORC.

        Args:
            path: Output path/URI (file or directory) to write to.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the files written.

        Examples:
            .. doctest::

                >>> import batcher as bt, os, tempfile
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> out = os.path.join(tempfile.mkdtemp(), "t")
                >>> _ = ds.write.orc(out)
                >>> bt.read.orc(out).count()
                3
        """
        return self(path, "orc", **opts)

    def arrow(self, path: PathLike, **opts: Any) -> WriteManifest:
        """Write as Arrow/Feather IPC.

        Args:
            path: Output path/URI (file or directory) to write to.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the files written.

        Examples:
            .. doctest::

                >>> import batcher as bt, os, tempfile
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> out = os.path.join(tempfile.mkdtemp(), "t")
                >>> _ = ds.write.arrow(out)
                >>> bt.read.arrow(out).count()
                3
        """
        return self(path, "arrow", **opts)

    def avro(self, path: PathLike, **opts: Any) -> WriteManifest:
        """Write as Avro (needs ``batcher-engine[avro]``).

        Args:
            path: Output path/URI to write to.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the files written.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.write.avro("data.avro")  # doctest: +SKIP
        """
        return self(path, "avro", **opts)

    def fasta(self, path: PathLike, **opts: Any) -> WriteManifest:
        """Write as FASTA, one record per row.

        The dataset must carry `id` and `sequence` columns; a `description` column is used
        for the rest of the header line when present. Sequences are wrapped at 60 characters,
        the width the NCBI and UniProt reference files use.

        Args:
            path: Output path/URI to write to.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the files written.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": ["chr1"], "sequence": ["ATGGCC"]})
                >>> ds.write.fasta("genome.fasta")  # doctest: +SKIP
        """
        return self(path, "fasta", **opts)

    def fastq(self, path: PathLike, **opts: Any) -> WriteManifest:
        """Write as four-line FASTQ, one read per row.

        The dataset must carry `id`, `sequence`, and `quality` columns, and the sequence and
        quality strings must be the same length in every row — a quality string is one
        character per base, so a mismatch is refused rather than written out for the next
        reader to misinterpret.

        Args:
            path: Output path/URI to write to.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the files written.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict(
                ...     {"id": ["r1"], "sequence": ["ACGT"], "quality": ["IIII"]}
                ... )
                >>> ds.write.fastq("reads.fastq")  # doctest: +SKIP
        """
        return self(path, "fastq", **opts)

    def bed(self, path: PathLike, **opts: Any) -> WriteManifest:
        """Write interval rows as BED, in the specification's column order.

        The dataset must carry `chrom`, `start`, and `end`. The leading run of standard
        columns also present is written after them, so a table with `name` writes BED4. BED
        is positional, so a gap cannot be expressed and the run stops at the first absent
        column.

        Args:
            path: Output path/URI to write to.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the files written.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"chrom": ["chr1"], "start": [0], "end": [100]})
                >>> ds.write.bed("regions.bed")  # doctest: +SKIP
        """
        return self(path, "bed", **opts)

    def gff(self, path: PathLike, **opts: Any) -> WriteManifest:
        """Write annotation rows as GFF3, with the version directive on the first line.

        All nine columns are required, because GFF is positional. Nulls are written as ``.``,
        the format's own missing marker.

        Args:
            path: Output path/URI to write to.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the files written.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> genes.write.gff("out.gff3")  # doctest: +SKIP
        """
        return self(path, "gff", **opts)

    def lance(self, path: PathLike, **opts: Any) -> WriteManifest:
        """Write a Lance dataset (needs ``batcher-engine[lance]``).

        Args:
            path: Output directory path for the Lance dataset.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the files written.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.write.lance("data.lance")  # doctest: +SKIP
        """
        return self(path, "lance", **opts)

    def msgpack(self, path: PathLike, **opts: Any) -> WriteManifest:
        """Write as MessagePack (needs ``batcher-engine[msgpack]``).

        Args:
            path: Output path/URI to write to.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the files written.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"x": [1, 2, 3]})
                >>> ds.write.msgpack("data.msgpack")  # doctest: +SKIP
        """
        return self(path, "msgpack", **opts)

    # --- Upserts / MERGE INTO ----------------------------------------------
    def merge(
        self,
        target: str,
        *,
        on: str | list[str],
        when_matched: str = "update",
        when_not_matched: str = "insert",
        format: str | None = None,
        **opts: Any,
    ) -> WriteManifest:
        """Upsert (``MERGE INTO``) this dataset into an existing `target`, keyed on `on`.

        For a transactional sink (Delta) this delegates to the native ``MERGE``. For a
        plain file target it is a copy-on-write merge: read the current target,
        reconcile with this source (``when_matched`` ∈ ``update``/``delete``,
        ``when_not_matched`` ∈ ``insert``/``ignore``), and atomically overwrite. If the
        target does not exist yet, the source is written as-is (all inserts).

        File merge is read-modify-write over the whole path — single-writer only; use a
        Delta target for concurrent writers.

        Use `merge` for **key-matched upserts** (reconcile rows by `on`). For replacing
        a known slice of a partitioned table wholesale — a backfill or idempotent
        partition reload — use ``write(path, replace_where=<predicate>)`` instead, which
        overwrites every row matching the predicate regardless of keys.

        Args:
            target: Path/URI of the table to merge into.
            on: Join key column name(s) matching source rows to target rows.
            when_matched: Action for matched rows (``"update"`` or ``"delete"``).
            when_not_matched: Action for new rows (``"insert"`` or ``"ignore"``).
            format: Sink format override; inferred from `target` when omitted.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the merged output.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> updates = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                >>> updates.write.merge(  # doctest: +SKIP
                ...     "warehouse/orders",
                ...     on="id",
                ...     when_matched="update",
                ...     when_not_matched="insert",
                ... )
        """
        from batcher.api.merge import execute_merge

        return execute_merge(
            self,
            target,
            on=on,
            when_matched=when_matched,
            when_not_matched=when_not_matched,
            format=format,
            opts=opts,
        )

    def merge_into(
        self,
        target: str,
        *,
        on: str | list[str],
        prune: bool = True,
        format: str | None = None,
        **opts: Any,
    ) -> MergeBuilder:
        """Open a full ``MERGE INTO`` against `target`, keyed on `on` — the general form.

        `merge` is the two-clause shorthand (update the matched, insert the rest). This is
        the whole statement: any number of ordered ``WHEN`` clauses, each with its own
        condition, each writing whichever columns it likes — including ``WHEN NOT MATCHED
        BY SOURCE``, which acts on the target rows the change set never mentioned (how a
        snapshot load expires departed rows, and how SCD-2 closes a version). Clauses are
        tried in the order added; the first whose condition holds wins.

        The merge rewrites only the data files whose key statistics prove they could hold
        one of the source's keys — so an upsert costs the change set, not the table. (A
        ``when_not_matched_by_source`` clause is *about* the untouched rows, so it forces a
        full rewrite; see `when_not_matched_by_source`.)

        Args:
            target: Path/URI of the table to merge into.
            on: Key column(s) matching a source row to a target row.
            prune: Skip target files the source's keys provably cannot reach. Correctness
                does not depend on it — turning it off only rewrites everything.
            format: Sink format override; inferred from `target` when omitted.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `MergeBuilder`; add clauses, then call `execute`.

        Examples:
            .. doctest::

                >>> import tempfile, os
                >>> import batcher as bt
                >>> from batcher import lit, source_col
                >>> path = os.path.join(tempfile.mkdtemp(), "orders.parquet")
                >>> _ = bt.from_pydict({"id": [1, 2], "amount": [10, 20]}).write.parquet(path)
                >>> changes = bt.from_pydict({"id": [2, 3], "amount": [99, 30]})
                >>> _ = (
                ...     changes.write.merge_into(path, on="id")
                ...     .when_matched(source_col("amount") > lit(50))
                ...     .delete()
                ...     .when_matched()
                ...     .update_all()
                ...     .when_not_matched()
                ...     .insert_all()
                ...     .execute()
                ... )
                >>> sorted(bt.read.parquet(path).collect().to_pydict()["id"])
                [1, 3]
        """
        from batcher.api.merge import MergeBuilder

        keys = [on] if isinstance(on, str) else list(on)
        return MergeBuilder(self._ds, target, keys, prune=prune, format=format, opts=opts)

    # --- Lakehouse / catalog ----------------------------------------------
    def delta(
        self,
        uri: str,
        *,
        mode: str = "append",
        merge_on: str | list[str] | None = None,
        auto_compact: bool = False,
        merge_schema: bool = False,
        table_properties: dict[str, str] | None = None,
        **opts: Any,
    ) -> WriteManifest:
        """Write to a Delta Lake table (one transactional commit).

        With `merge_on`, performs a ``MERGE INTO`` upsert keyed on those columns —
        matched rows are updated and new rows inserted (Spark/Delta ``MERGE``). The
        keys build the match predicate; pass `merge_predicate=` instead for a custom
        one. Otherwise `mode` is ``"append"`` (default) or ``"overwrite"``.

        ``auto_compact=True`` bin-packs the table after the commit if it has accumulated
        enough small files. An incremental writer leaves one small file per commit and the
        next write cannot fix that — it only adds another — so a table nobody compacts
        eventually costs more to *plan* than to read. The check counts the table's standing
        small files (from the log, not by opening anything), and the compaction is the same
        transaction `bt.compact` runs, so it never deletes a file an older version still
        references. It runs *after* the commit, so a failed compaction cannot fail the write.

        Args:
            uri: Path/URI of the Delta table root.
            mode: ``"append"`` (default) or ``"overwrite"`` when not merging.
            merge_on: Key column(s) to upsert on; triggers a ``MERGE INTO``.
            auto_compact: Bin-pack the table after the commit if small files have piled up.
            merge_schema: Evolve the table to accept columns this write has and the table
                does not. Off by default — an unexpected column is refused rather than
                silently written into the files where the table cannot see it.
            table_properties: Delta table properties, Spark's ``TBLPROPERTIES`` — set when
                this write creates the table, altered when it already exists. This is how
                protocol features are turned on: ``delta.enableChangeDataFeed`` (required
                before `bt.read.read_change_feed` can read the table),
                ``delta.appendOnly``, the retention durations.
            opts: Additional write options (e.g. ``merge_predicate=``) forwarded to the sink.

        Returns:
            A `WriteManifest` describing the committed Delta files.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                >>> ds.write.delta("warehouse/orders", merge_on="id")  # doctest: +SKIP
                >>> ds.write.delta(  # doctest: +SKIP
                ...     "warehouse/orders",
                ...     table_properties={"delta.enableChangeDataFeed": "true"},
                ... )
        """
        if merge_on is not None:
            from batcher.api.merge import merge_predicate_for

            opts["merge_predicate"] = merge_predicate_for(merge_on)
        opts["merge_schema"] = merge_schema
        if table_properties is not None:
            opts["table_properties"] = table_properties
        return self(uri, "delta", mode=mode, auto_compact=auto_compact, **opts)

    def iceberg(self, identifier: str, *, mode: str = "append", **opts: Any) -> WriteManifest:
        """Write to an Iceberg table (``mode="append"|"overwrite"``).

        Args:
            identifier: Table identifier within the catalog (``db.table``).
            mode: ``"append"`` (default) or ``"overwrite"``.
            opts: Additional write options forwarded to the sink.

        Returns:
            A `WriteManifest` describing the committed Iceberg files.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                >>> ds.write.iceberg("db.orders", mode="append")  # doctest: +SKIP
        """
        return self(identifier, "iceberg", mode=mode, **opts)

    def hudi(self, table_uri: str, *, mode: str = "append", **opts: Any) -> WriteManifest:
        """Raise: Batcher reads Hudi tables but does not write them.

        A Hudi write needs that project's Spark/Flink write stack — timeline, file groups
        and the index — which Batcher does not implement. The method exists so the refusal
        names the reason instead of arriving as a missing attribute, and so `bt.read.hudi`
        has a visible counterpart. Use `write.delta` or `write.iceberg`, both of which are
        real transactional writes here.

        Args:
            table_uri: Path/URI of the Hudi table root.
            mode: Accepted for signature parity; no mode is writable.
            opts: Accepted for signature parity.

        Returns:
            Never returns.

        Raises:
            BackendError: Always — Hudi writes are not supported.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                >>> ds.write.hudi("s3://lake/orders")  # doctest: +SKIP
                Traceback (most recent call last):
                BackendError: Hudi writes require Spark/Flink; Batcher supports Hudi reads only
        """
        return self(table_uri, "hudi", mode=mode, **opts)

    # --- SQL / warehouses --------------------------------------------------
    def sql(self, table: str, **opts: Any) -> WriteManifest:
        """Bulk-ingest into a database table via ADBC/FlightSQL.

        Takes the same standard connection URI as `bt.read.sql`, so a read and the write
        that follows it are spelled the same way. Rows are ingested as Arrow, in bulk,
        never row by row.

        Pass ``password="env:PGPASSWORD"`` rather than embedding a credential in the URI.

        Args:
            table: Destination table name.
            opts: Connection options — ``uri=``, ``password=``, or an explicit
                ``driver=``/``db_kwargs=`` pair — plus ``mode=`` (``"create"``,
                ``"append"``, ``"replace"``, ``"create_append"``).

        Returns:
            A `WriteManifest` describing the written rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                >>> ds.write.sql(  # doctest: +SKIP
                ...     "orders", uri="postgresql://localhost/warehouse"
                ... )
        """
        return self(table, "adbc", **opts)

    def snowflake(self, table: str, **opts: Any) -> WriteManifest:
        """Write to a Snowflake table.

        Args:
            table: Destination Snowflake table name.
            opts: ``connection_kwargs=`` — a dict passed to the Snowflake connector
                (``account``, ``user``, ``warehouse``, ``database``, …) — plus
                ``mode=`` (``"append"`` or ``"overwrite"``).

        Returns:
            A `WriteManifest` describing the written rows.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                >>> ds.write.snowflake(  # doctest: +SKIP
                ...     "ORDERS",
                ...     connection_kwargs={
                ...         "account": "acct",
                ...         "warehouse": "WH",
                ...         "database": "DB",
                ...     },
                ... )
        """
        return self(table, "snowflake", **opts)

    def mongo(self, collection: str, **opts: Any) -> WriteManifest:
        """Write to a MongoDB collection.

        Args:
            collection: Destination collection name.
            opts: Connection (``uri=``), ``database=``, and write options as keywords.

        Returns:
            A `WriteManifest` describing the written documents.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> ds = bt.from_pydict({"id": [1, 2], "amount": [10, 20]})
                >>> ds.write.mongo(  # doctest: +SKIP
                ...     "orders", uri="mongodb://localhost:27017", database="shop"
                ... )
        """
        return self(collection, "mongo", **opts)
