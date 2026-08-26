"""A `map_batches` UDF that stops emitting a column gets a warning, not silence.

`reconcile_batches` unions drifting output schemas on purpose, so a UDF whose later batches
carry *extra* fields (LLM structured outputs) concatenates instead of failing at the
merge -- the schema-inference footgun Ray Data hits. That is a feature and stays one.

The reverse drift is not the same thing. When a column an earlier batch produced is absent
from a later one, the union keeps the column and null-fills the later rows, so a UDF that
renames or drops a column returns a full-height table that is mostly null and says nothing.
Measured on 40,000 rows across three morsels, renaming the output column after the first
batch produced a 40,000x2 result with each column ~50% null.

It is a warning rather than an error because the two drifts are indistinguishable at the
schema level and the additive one is supported. Readers are deliberately not wired to it: a
file missing a column is ordinary schema evolution.
"""

from __future__ import annotations

import logging

import pyarrow as pa
import pytest

from batcher.io.schema.evolution import note_dropped_columns

pytestmark = pytest.mark.unit


def _batch(**cols):
    return pa.record_batch({k: pa.array(v) for k, v in cols.items()})


def _warnings(caplog):
    """The structured fields of each warning -- what tooling reads, not the rendered text.

    `log_kv` attaches detail as a record attribute rather than formatting it into the
    message, so asserting on `caplog.text` would silently pass on a warning that named no
    columns at all.
    """
    return [getattr(r, "batcher_fields", {}) for r in caplog.records if r.levelname == "WARNING"]


def test_warns_when_a_column_disappears(caplog):
    batches = [_batch(a=[1, 2]), _batch(b=[3, 4])]
    with caplog.at_level(logging.WARNING):
        note_dropped_columns(batches, context="map_batches")
    assert [f["dropped"] for f in _warnings(caplog)] == [["a"]]
    assert _warnings(caplog)[0]["context"] == "map_batches"


def test_silent_when_later_batches_only_add_columns(caplog):
    """The supported drift: extra fields appear. Earlier rows genuinely have no value."""
    batches = [_batch(a=[1]), _batch(a=[2], extra=[9])]
    with caplog.at_level(logging.WARNING):
        note_dropped_columns(batches, context="map_batches")
    assert _warnings(caplog) == []


def test_silent_on_a_stable_schema(caplog):
    with caplog.at_level(logging.WARNING):
        note_dropped_columns([_batch(a=[1]), _batch(a=[2])], context="map_batches")
    assert _warnings(caplog) == []


def test_silent_on_a_single_batch(caplog):
    with caplog.at_level(logging.WARNING):
        note_dropped_columns([_batch(a=[1])], context="map_batches")
    assert _warnings(caplog) == []


def test_names_every_dropped_column(caplog):
    batches = [_batch(a=[1], b=[2], c=[3]), _batch(c=[4])]
    with caplog.at_level(logging.WARNING):
        note_dropped_columns(batches, context="map_batches")
    assert [f["dropped"] for f in _warnings(caplog)] == [["a", "b"]]


def test_a_column_that_comes_back_still_counts_as_dropped(caplog):
    """It vanished for one batch, so those rows were null-filled -- that is the defect.

    Both columns are reported here, and both genuinely were dropped: the middle batch has
    no `a`, and the last has no `b`. Each gap is a run of null-filled rows.
    """
    batches = [_batch(a=[1]), _batch(b=[2]), _batch(a=[3])]
    with caplog.at_level(logging.WARNING):
        note_dropped_columns(batches, context="map_batches")
    assert [f["dropped"] for f in _warnings(caplog)] == [["a", "b"]]
