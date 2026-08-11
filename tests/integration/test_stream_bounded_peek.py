"""Ten rows off an unfamiliar topic, which is the first thing anyone types.

`bt.read.kafka(...).head(10).to_pydict()` refused: every materializing terminal guarded on
"any source is unbounded", which is right for the general case and wrong for this one. A
`LIMIT n` over a breaker-free pipeline is finite in both memory and time -- the router
already stops reading the moment it has n rows -- so the only thing the guard was
protecting against was a query that could not hang.

`show()` was the same refusal wearing a different name, and it is what a person reaches for
before `to_pydict()`.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher import col
from batcher._internal.errors import PlanError

_SCHEMA = pa.schema([("v", pa.int64())])


def _endless():
    """A source that does not stop, so a materializing terminal cannot be waiting it out."""

    def feed():
        for i in range(1_000_000):
            yield pa.record_batch({"v": [i]}, schema=_SCHEMA)

    return bt.from_batches(feed, _SCHEMA, bounded=False)


@pytest.mark.integration
def test_head_materializes_a_bounded_peek():
    assert _endless().head(3).to_pydict() == {"v": [0, 1, 2]}


@pytest.mark.integration
def test_the_peek_can_be_counted_too():
    """Without this, `head(10)` could be materialized but not counted -- `count()` wraps the
    plan in an aggregate, and an aggregate over an endless source is exactly what the guard
    refuses. An inconsistency nobody could have explained."""
    assert _endless().head(3).count() == 3


@pytest.mark.integration
def test_an_offset_is_respected():
    assert _endless().slice(2, 3).to_pydict() == {"v": [2, 3, 4]}


@pytest.mark.integration
def test_a_filter_beneath_the_peek_still_streams():
    assert _endless().filter(col("v") > 1).head(2).to_pydict() == {"v": [2, 3]}


@pytest.mark.integration
def test_show_prints_the_first_rows_of_a_stream(capsys):
    _endless().show(3)
    assert "0" in capsys.readouterr().out


@pytest.mark.integration
def test_show_on_an_existing_peek_does_not_wrap_a_limit_in_a_limit(capsys):
    """A limit over a limit is correct and unstreamable: the router recognizes a limit over
    a *breaker-free* pipeline, and a limit is not one. Folding the two keeps it one node."""
    _endless().head(2).show()
    printed = capsys.readouterr().out
    assert "0" in printed and "1" in printed


@pytest.mark.integration
def test_to_pylist_and_the_other_materializing_terminals_follow():
    assert _endless().head(2).to_pylist() == [{"v": 0}, {"v": 1}]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("label", "build"),
    [
        ("top_n", lambda: _endless().sort("v").head(3)),
        ("whole_stream", _endless),
        ("aggregate", lambda: _endless().agg(total=col("v").sum())),
    ],
)
def test_what_is_still_refused(label, build):
    """Top-N is finite too, and unreachable: it is not known until the last row arrives.
    The other two are unbounded by construction."""
    with pytest.raises(PlanError, match="materializes the full result"):
        build().to_pydict()


@pytest.mark.integration
def test_the_refusal_now_points_at_the_peek():
    with pytest.raises(PlanError, match="bounded peek with head"):
        _endless().to_pydict()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda ds: ds.head(3), [0, 1, 2]),
        (lambda ds: ds.slice(2, 5), [2, 3, 4, 5, 6]),
        (lambda ds: ds.head(3).head(2), [0, 1]),
    ],
)
def test_a_bounded_source_is_unchanged(build, expected):
    """The fold happens for bounded plans too, so it has to mean the same thing there."""
    bounded = bt.from_pydict({"v": list(range(10))})
    assert build(bounded).to_pydict() == {"v": expected}
