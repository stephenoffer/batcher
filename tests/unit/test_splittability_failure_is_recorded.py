"""The two distributed paths that lose a capability silently must leave a record.

`dist.executor._is_splittable_source` decides whether real distributed data exists, by
*calling* `source.splits()` rather than trusting a declared flag. When that call raises it
returns `False`, which is the right answer -- a source that cannot enumerate its splits must
not be handed to workers.

But `False` routes the query to `_single_node` **without** passing `_unsupported`, which is
the function that exists to refuse loudly rather than let a whole job run on one node. So a
broken splitter and a genuinely unsplittable source were indistinguishable from outside: the
query returns the right rows, on one node, indefinitely, and nothing anywhere records why.
That is the exact difference `note_suppressed` was written to keep observable -- its own
docstring calls it the difference between "this optimization did not apply" and "this
optimization has been broken since March".

`partition_io._sources.source_pushdown` is the same shape with a different casualty. When
its analysis raises it returns `(None, None)` -- read every column, push no predicate --
which is safe and silently reproduces the exact defect the function was written to fix. Its
own docstring measures that defect: "the same query read two columns spilled and thirteen
distributed".

Both behaviours are deliberately unchanged. The safe answers are the right answers; only
their silence was not. Both paths are quiet in practice, which is what makes a record cheap:
`source_pushdown` does not raise for project/filter/aggregate/sort/window/union or a
`map_batches` pipeline, measured before the log was added.
"""

from __future__ import annotations

import logging

import pyarrow as pa
import pytest

from batcher.dist.executor import _is_splittable_source

pytestmark = pytest.mark.unit


def _step(record) -> str:
    """The `step` field of a `note_suppressed` record.

    `log_kv` packs its key/values into a `batcher_fields` dict on the record rather than
    setting them as attributes, so `record.step` does not exist and `getattr(record, "step",
    "")` silently returns `""` -- which makes a test looking for it pass by finding nothing.
    """
    return str((getattr(record, "batcher_fields", None) or {}).get("step", ""))


class _BrokenSplitter:
    """A source whose split enumeration fails the way a real one would."""

    def splits(self):
        raise RuntimeError("footer unreadable")


class _WholeOnly:
    """A source that enumerates exactly one whole-source split, i.e. nothing to distribute."""

    def splits(self):
        from batcher.io.splits import WholeSourceSplit

        # `WholeSourceSplit` holds the source itself; constructing it with no argument
        # raises, and the raise would be logged -- turning this control into a copy of the
        # test above rather than its opposite.
        return [WholeSourceSplit(self)]


def test_a_broken_splitter_is_not_treated_as_splittable():
    """The behaviour that must not change: still `False`, still no workers."""
    assert _is_splittable_source(_BrokenSplitter()) is False


def test_a_broken_splitter_leaves_a_record(caplog):
    """...and now says so, at DEBUG, with the exception attached."""
    with caplog.at_level(logging.DEBUG, logger="batcher.dist"):
        _is_splittable_source(_BrokenSplitter())

    matching = [r for r in caplog.records if "enumerate splits" in _step(r)]
    assert matching, (
        "a source whose `splits()` raised produced no record; a broken splitter and an "
        "unsplittable source are indistinguishable again"
    )
    assert matching[0].exc_info is not None, (
        "the record carries no traceback, which is the half that says *why* it broke"
    )
    assert "_BrokenSplitter" in _step(matching[0])


def test_an_unsplittable_source_stays_quiet(caplog):
    """The control, and the reason the record is worth having.

    A source that legitimately has nothing to distribute must NOT log -- otherwise the
    record fires on every in-memory query and stops meaning anything. Without this, the
    assertion above would pass against a version that logged unconditionally.
    """
    with caplog.at_level(logging.DEBUG, logger="batcher.dist"):
        assert _is_splittable_source(_WholeOnly()) is False

    assert not [r for r in caplog.records if "enumerate splits" in _step(r)]


def test_an_in_memory_source_is_still_not_splittable():
    """The ordinary case, so the fixtures above are not the only shapes exercised."""
    import batcher as bt

    ds = bt.from_arrow(pa.table({"a": [1, 2, 3]}))
    assert _is_splittable_source(ds._sources[0]) is False


def test_a_broken_pushdown_analysis_leaves_a_record(caplog):
    """`source_pushdown` falling back to "read everything" must say so."""
    import batcher as bt
    from batcher.dist.executors.partition_io import _sources

    ds = bt.from_arrow(pa.table({"a": [1, 2, 3]})).select("a")

    def boom(_plan):
        raise RuntimeError("plan shape not walkable")

    import batcher.kyber.rules.projections as projections

    original = projections.required_columns_per_source
    projections.required_columns_per_source = boom
    try:
        with caplog.at_level(logging.DEBUG, logger="batcher.dist"):
            assert _sources.source_pushdown(ds._plan, 0) == (None, None)
    finally:
        projections.required_columns_per_source = original

    assert [r for r in caplog.records if "source pushdown" in _step(r)], (
        "the pushdown analysis failed and nothing recorded it; the read silently widens to "
        "every column with no way to tell that from a plan that needs every column"
    )


def test_an_ordinary_plan_computes_a_pushdown_without_logging(caplog):
    """The control: the quiet path must stay quiet, or the record means nothing."""
    import batcher as bt
    from batcher.dist.executors.partition_io import _sources

    ds = bt.from_arrow(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})).select("a")
    with caplog.at_level(logging.DEBUG, logger="batcher.dist"):
        projection, _predicate = _sources.source_pushdown(ds._plan, 0)

    assert projection == ["a"], f"expected a narrowed read, got {projection}"
    assert not [r for r in caplog.records if "source pushdown" in _step(r)]
