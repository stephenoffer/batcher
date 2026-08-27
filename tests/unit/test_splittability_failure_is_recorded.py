"""A source whose `splits()` raises must be recorded, not silently un-distributed.

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

The behaviour is deliberately unchanged. The refusal to distribute is correct; only its
silence was not.
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
