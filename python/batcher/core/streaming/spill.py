"""Cold windows of a streaming aggregate's state, held on disk instead of in memory.

A watermarked windowed aggregate bounds its state by *evicting* closed windows, which is the
right bound and not always a small one: the open set is `allowed_lateness / hop` windows wide,
and each window holds one row per group key. A high-cardinality key over a generous lateness
allowance therefore reaches `memory.streaming_state_max_bytes` while behaving exactly as
designed — and what happened at that point was a `ResourceError`. The query was correct right
up to the moment it stopped.

This is the disk tier under that bound. It exploits the one thing a windowed aggregate has
that a general state store does not: **the watermark only moves forward**, so windows are
evicted in increasing order, and a window written to disk is read back exactly once and never
sought into. That turns a state backend — the hard, keyed, random-access problem — into an
ordered run of Arrow IPC files, which is a problem the engine already knows how to solve.

Correctness rests on the same property the rest of the streaming state does: the runs hold
*partial* aggregate state, and `combine` is associative and commutative (invariant #7), so a
window's rows may be split across any number of runs and the in-memory state and still
combine to the one answer. A late row landing in a window that was already spilled is
therefore not a special case — it folds into memory and meets its spilled half at eviction.

Layer: `core`, the executor. It decides *how* to keep running, never *whether* the state is
too big — that threshold is `memory.streaming_state_max_bytes`, read from config, which is
also what Carbonite's policies are written against.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
from pyarrow import ipc

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["SpilledWindows"]


@dataclass(frozen=True, slots=True)
class _Run:
    """One spilled file and the closed range of window starts it covers.

    The range is what makes reading a run back *once* the common case. Eviction asks for
    everything at or below a threshold that only rises, so a run is needed exactly when its
    `low` is at or below that threshold. Ranges are contiguous and disjoint while the fold
    only ever spills its oldest windows; they can overlap after a straddling run's open
    remainder returns to memory and is re-spilled, which costs an early read-back and
    nothing else.
    """

    path: str
    low: int
    high: int
    rows: int
    nbytes: int


class SpilledWindows:
    """The on-disk tail of a windowed aggregate's open-window state.

    Runs are written oldest-window-first and read back in the same order, so this is an
    append-and-drain structure rather than a keyed store. Nothing here interprets the
    partial-state columns; it moves whole Arrow batches and reports what it is holding.
    """

    __slots__ = ("_root", "_runs", "_seq", "_window_column")

    def __init__(self, window_column: str, *, root: str | None = None) -> None:
        """Create a spill area for windows keyed by `window_column`.

        Args:
            window_column: The state column holding each row's window start.
            root: Where to place the scratch directory. Defaults to the engine's resolved
                spill location, so a streaming spill lands on the same disk every other
                spill does and the operator's `memory.spill_dir` is honoured.
        """
        from batcher._internal.site.scratch import spill_scratch_dir

        self._window_column = window_column
        self._root = tempfile.mkdtemp(
            prefix="batcher_stream_state_", dir=root or spill_scratch_dir()
        )
        self._runs: list[_Run] = []
        self._seq = 0

    @property
    def window_column(self) -> str:
        """The state column holding each row's window start."""
        return self._window_column

    def __len__(self) -> int:
        """How many runs are currently on disk."""
        return len(self._runs)

    def nbytes(self) -> int:
        """Logical bytes currently spilled — what resident memory was traded for."""
        return sum(run.nbytes for run in self._runs)

    def rows(self) -> int:
        """Rows of partial state currently on disk."""
        return sum(run.rows for run in self._runs)

    def spill(self, batch: pa.RecordBatch) -> None:
        """Write `batch` as the next run, recording the window range it covers.

        Args:
            batch: Partial state to write out — in practice the oldest windows the caller
                still holds, though nothing depends on that. Runs may overlap once an
                eviction has returned a straddling run's open remainder to memory and that
                remainder is later re-spilled; `drain_through` selects on range membership
                rather than on write order, so an overlap costs a run being read back one
                sweep early and never a wrong answer.
        """
        if batch.num_rows == 0:
            return
        low, high = self._window_span(batch)
        path = os.path.join(self._root, f"run-{self._seq:06d}.arrow")
        self._seq += 1
        with open(path, "wb") as handle, ipc.new_file(handle, batch.schema) as writer:
            writer.write_batch(batch)
        self._runs.append(
            _Run(path=path, low=low, high=high, rows=batch.num_rows, nbytes=batch.nbytes)
        )

    def drain_through(self, threshold: int) -> Iterator[pa.RecordBatch]:
        """Read back and delete every run that can hold a window at or below `threshold`.

        The batches come back whole, including any rows *above* the threshold that shared a
        straddling run. Separating them is the caller's job because only the caller knows the
        state's schema well enough to split it, and it already performs exactly that split on
        the in-memory side — so handing back whole runs keeps one splitting rule rather than
        two that must agree.

        Args:
            threshold: The highest window start the watermark has closed.

        Yields:
            One batch per run read back, oldest window range first.
        """
        keep: list[_Run] = []
        for run in self._runs:
            if run.low > threshold:
                keep.append(run)
                continue
            with open(run.path, "rb") as handle:
                table = ipc.open_file(handle).read_all()
            for batch in table.to_batches():
                if batch.num_rows:
                    yield batch
            self._remove(run)
        self._runs = keep

    def iter_runs(self) -> Iterator[pa.RecordBatch]:
        """Read every run back **without** consuming it, one run resident at a time.

        The checkpoint path. A snapshot has to persist the spilled half of the state or
        recovery silently resumes with only what happened to be in memory — but reading it
        all back to build one batch would undo the bound the spill exists to hold, on the
        exact query that is large enough to have spilled. Yielding run by run lets the
        snapshot writer stream them into one file with a single run resident at the peak.

        Yields:
            One batch per run, in the order the runs were written.
        """
        for run in self._runs:
            with open(run.path, "rb") as handle:
                table = ipc.open_file(handle).read_all()
            for batch in table.to_batches():
                if batch.num_rows:
                    yield batch

    def drain_all(self) -> Iterator[pa.RecordBatch]:
        """Read back and delete every run, whatever window range it covers.

        Deliberately **not** `drain_through` at the highest window seen. That is what this
        did, taking the threshold from the last run's `high` on the assumption that runs are
        written in increasing window order — an assumption `spill` explicitly does not make.
        A row arriving out of order for an already-spilled window puts that window back in
        memory, and the next spill writes it as the newest run with the *lowest* range: the
        threshold then collapsed to that low value and the drain silently skipped every run
        above it. The flush emitted two windows out of four, with nothing raised.

        Yields:
            One batch per run, in write order.
        """
        runs, self._runs = self._runs, []
        for run in runs:
            with open(run.path, "rb") as handle:
                table = ipc.open_file(handle).read_all()
            for batch in table.to_batches():
                if batch.num_rows:
                    yield batch
            self._remove(run)

    def close(self) -> None:
        """Delete every run and the scratch directory. Idempotent.

        Spilled state is scratch by construction: it is rebuilt from the checkpoint on
        recovery, so leaving it behind costs disk and buys nothing.
        """
        self._runs = []
        shutil.rmtree(self._root, ignore_errors=True)

    def _window_span(self, batch: pa.RecordBatch) -> tuple[int, int]:
        """The lowest and highest window start in `batch`, as int64 microseconds."""
        import pyarrow.compute as pc

        from batcher.plan.streaming import event_micros

        column = event_micros(batch.column(self._window_column))
        low, high = pc.min_max(column).as_py().values()
        # A null window start is the keyless/global row of an aggregate that has one; it
        # never closes on a threshold, so it must not drag a run's range down to it.
        return (0 if low is None else int(low), 0 if high is None else int(high))

    @staticmethod
    def _remove(run: _Run) -> None:
        """Delete one run's file, tolerating a scratch dir already swept by the OS."""
        import contextlib

        with contextlib.suppress(OSError):
            os.remove(run.path)
