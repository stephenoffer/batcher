"""An offset-window split must be a *cover*: exhaustive and disjoint. It was neither.

Couchbase and Neo4j have no native shard/token/segment primitive, so a parallel read of them
is split with ``LIMIT``/``OFFSET`` (or ``SKIP``/``LIMIT``) windows. Both connectors' docstrings
promised "a disjoint and exhaustive cover". Both produced this:

    return [(i * _WINDOW_ROWS, _WINDOW_ROWS) for i in range(segments)]

with ``_WINDOW_ROWS = 100_000``. `segments` windows of a *fixed* size cover only
``segments * 100_000`` rows — everything past that was **silently dropped, with no error**.
Eight segments over a billion-row collection returned 800,000 rows and reported success.
Turning on parallelism, the thing you do *because* the data is large, was what truncated it.

This is the worst failure class there is: a wrong answer that looks like a right one, on the
only code path that runs at scale, in a connector with no server available in CI to catch it.
And it needed no server to catch — the windows are pure arithmetic, which is what these tests
hold. The two properties are all that matter:

* **exhaustive** — every row of the result is inside some window;
* **disjoint** — no row is inside two.
"""

from __future__ import annotations

import pytest

from batcher.io.formats.nosql.base import offset_windows

_UNBOUNDED = 0  # a `limit` of 0 means "read to the end"


def _covered(windows: list[tuple[int, int]], total: int) -> list[int]:
    """The row indices `windows` actually read, in order, from a result of `total` rows."""
    seen: list[int] = []
    for offset, limit in windows:
        end = total if limit == _UNBOUNDED else min(offset + limit, total)
        seen.extend(range(min(offset, total), end))
    return seen


@pytest.mark.parametrize("total", [1, 2, 7, 100, 999, 1_000, 100_001, 1_000_000])
@pytest.mark.parametrize("segments", [1, 2, 3, 8, 64])
def test_the_windows_cover_every_row_exactly_once(total: int, segments: int) -> None:
    """The regression, stated as arithmetic: exhaustive and disjoint, for every shape."""
    windows = offset_windows(total, segments)
    covered = _covered(windows, total)

    assert sorted(covered) == list(range(total)), (
        f"not a cover: {total} rows / {segments} segments read {len(covered)} rows"
    )
    assert len(covered) == len(set(covered)), "windows overlap — a row is read twice"


def test_the_old_fixed_window_scheme_would_have_truncated() -> None:
    """Pin the bug itself, so nobody reintroduces it thinking it looks reasonable."""
    total, segments, window = 1_000_000, 8, 100_000
    old = [(i * window, window) for i in range(segments)]
    assert len(_covered(old, total)) == 800_000, "the old scheme's shape has changed"

    new = offset_windows(total, segments)
    assert len(_covered(new, total)) == total


def test_the_last_window_is_always_unbounded() -> None:
    """The tail is what makes the cover exhaustive — including for rows written after the
    count was taken, which is the case a `total`-sized cover would otherwise miss."""
    for segments in (1, 2, 8, 64):
        windows = offset_windows(1_000, segments)
        assert windows[-1][1] == _UNBOUNDED, f"{segments} segments: the tail is bounded"


def test_rows_written_after_the_count_are_still_read() -> None:
    """The store is live. A cover sized from a stale count must not lose the new rows."""
    windows = offset_windows(1_000, 4)  # counted 1,000...
    covered = _covered(windows, 1_500)  # ...but 1,500 are there by the time we read
    assert sorted(covered) == list(range(1_500))


def test_an_unknown_total_reads_serially_rather_than_guessing() -> None:
    """No count ⇒ no way to size a window. One unbounded reader: slow and right."""
    assert offset_windows(None, 8) == [(0, _UNBOUNDED)]
    assert offset_windows(0, 8) == [(0, _UNBOUNDED)]


def test_a_single_segment_is_one_unbounded_window() -> None:
    assert offset_windows(1_000_000, 1) == [(0, _UNBOUNDED)]


def test_more_segments_than_rows_still_covers_exactly() -> None:
    windows = offset_windows(3, 64)
    assert sorted(_covered(windows, 3)) == [0, 1, 2]


def test_neo4j_refuses_to_split_an_unordered_query(monkeypatch) -> None:
    """SKIP/LIMIT over an undefined order is not a cover — the windows overlap and miss.

    Neo4j emitted `SKIP`/`LIMIT` with no `ORDER BY` when `order_by` was unset, so even the
    truncated prefix was nondeterministic. Refusing to split is the only sound answer.
    """
    from batcher.io.formats.nosql.base import PartitionSpec
    from batcher.io.formats.nosql.neo4j import Neo4jSource

    source = Neo4jSource.__new__(Neo4jSource)
    object.__setattr__(source, "_conn_kwargs", {"cypher": "MATCH (n) RETURN n", "order_by": None})
    object.__setattr__(source, "_partition_spec", PartitionSpec(segments=8))

    assert source._enumerate_partitions() == [(0, _UNBOUNDED)], (
        "an unordered Cypher query must not be split by SKIP/LIMIT"
    )
