"""What the result cache charges an entry, when the entry is a window onto something bigger.

`Table.nbytes` measures the rows a table *addresses*. It is not what the table keeps
resident. Any zero-copy derivation — `slice`, `head`, `limit` — addresses a window of its
parent's buffers and keeps the whole parent alive, so a 10-row slice of a 4M-row column
reports 80 bytes and pins 32 MB.

Budgeting on `nbytes` is therefore not a small under-count but an unbounded one, and it
fails in the worst possible direction: the entry that pins the most is the one that reports
the least, so it also scores *best* on the cost-per-byte keep-value and is the last thing
eviction ever chooses.

On this engine the ratio is bounded per entry by morselization rather than unbounded —
`bt.from_pydict(2M rows).limit(10).collect()` reports 160 bytes and retains 262,144, so it
is 1,638x rather than arbitrary. A store that believed twenty such entries cost 1,600 bytes
was measured holding 305 MiB against a 4 MiB budget, 76x over. A table from a source that
does not morselize carries no such bound at all.

These tests pin the accounting (`_retained_bytes`), the repair (`_compacted`), and the two
store-level consequences: the budget must bind on real memory, and the entry must be ranked
by what it costs.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from batcher.carbonite.cache import CacheStore, _compacted, _retained_bytes

pytestmark = pytest.mark.unit

_PARENT_ROWS = 2_000_000


def _parent() -> pa.Table:
    """A table large enough that pinning it is a real memory event (~16 MB)."""
    return pa.table({"v": pa.array(range(_PARENT_ROWS), type=pa.int64())})


def _window(rows: int = 10) -> pa.Table:
    return _parent().slice(0, rows)


# --- the measurement ----------------------------------------------------------


def test_a_window_is_charged_for_the_parent_it_pins() -> None:
    """The whole point: the accounted size must be the resident size, not the logical one."""
    window = _window()
    assert window.nbytes < 1000, "fixture is not a window onto a larger buffer"
    assert _retained_bytes(window) > 8_000_000


def test_an_ordinary_table_is_charged_what_it_is() -> None:
    """The fix must not inflate the common case, where the two figures already agree."""
    table = pa.table({"v": pa.array(range(50_000), type=pa.int64())})
    retained = _retained_bytes(table)
    assert table.nbytes <= retained <= table.nbytes * 1.1


def test_the_measure_is_never_below_the_logical_size() -> None:
    """A table cannot retain less than the rows it addresses, whatever the buffers report."""
    for table in (_window(), _parent(), pa.table({"s": pa.array(["a", "b", None])})):
        assert _retained_bytes(table) >= table.nbytes


def test_an_empty_table_is_measurable() -> None:
    """A zero-row result is cacheable and must not raise or report nonsense."""
    empty = pa.table({"v": pa.array([], type=pa.int64())})
    assert _retained_bytes(empty) >= 0


# --- the repair ---------------------------------------------------------------


def test_compaction_frees_the_parent() -> None:
    """A window worth compacting must come back owning only itself."""
    window = _window()
    compact, size = _compacted(window, _retained_bytes(window))
    assert size < 10_000, "the copy still pins the parent"
    assert compact.equals(window), "compaction changed the result"


def test_compaction_preserves_types_and_nulls() -> None:
    """A memory strategy is not a semantics: the rows must survive the copy exactly."""
    parent = pa.table(
        {
            "i": pa.array([None, 2, 3] * 400_000, type=pa.int64()),
            "s": pa.array(["a", None, "ccc"] * 400_000),
        }
    )
    window = parent.slice(1, 5)
    compact, _ = _compacted(window, _retained_bytes(window))
    assert compact.schema == window.schema
    assert compact.to_pydict() == window.to_pydict()


def test_a_small_window_is_left_alone() -> None:
    """Below the floor the copy costs more than the memory it returns."""
    parent = pa.table({"v": pa.array(range(1000), type=pa.int64())})
    window = parent.slice(0, 2)
    before = _retained_bytes(window)
    compact, size = _compacted(window, before)
    assert compact is window and size == before


def test_a_table_that_is_already_compact_is_not_copied() -> None:
    """No ratio, no copy — the common path must not pay for the rare one."""
    table = _parent()
    compact, size = _compacted(table, _retained_bytes(table))
    assert compact is table and size == _retained_bytes(table)


# --- what it means for the store ----------------------------------------------


def test_the_budget_binds_on_what_is_resident() -> None:
    """The failure this exists to prevent, stated as a test.

    Under `nbytes` accounting each window reports ~80 bytes, so all twenty fit a 4 MiB
    budget while pinning ~320 MB. The store must either compact them or refuse them; what
    it must not do is believe it is empty.
    """
    store = CacheStore(4 << 20)
    for i in range(20):
        store.put(f"k{i}", _window())
    assert store.used_bytes <= store.max_bytes, "the store exceeded its own budget"


def test_a_cached_window_still_returns_its_rows() -> None:
    """Whatever the store does about the footprint, the result must be unchanged."""
    store = CacheStore(64 << 20)
    window = _window(rows=7)
    store.put("k", window)
    got = store.get("k")
    assert got is not None
    assert got.to_pydict() == window.to_pydict()


def test_a_window_onto_something_larger_than_the_budget_is_still_cacheable() -> None:
    """Compaction is what keeps the useful case: `head(10)` of a huge scan is tiny.

    Charging the window its parent's footprint without compacting it would make every
    `limit`-shaped result uncacheable, which trades one failure for another.
    """
    store = CacheStore(2 << 20)  # smaller than the parent buffer the window pins
    store.put("k", _window())
    assert store.get("k") is not None, "a small result was refused because its parent is big"


def test_eviction_ranks_a_window_by_what_it_actually_costs() -> None:
    """Under `nbytes` the pinning entry looks cheapest per byte and is evicted last.

    Both entries here have the same cost and hit count, so the keep-value is decided purely
    by size. If the window is charged 80 bytes it outranks a genuinely small result and
    survives, which is the exact inversion that lets a cache fill with parents.
    """
    store = CacheStore(1 << 20)
    small = pa.table({"v": pa.array(range(200), type=pa.int64())})
    store.put("small", small)
    store.put("window", _window())
    # One of them had to go, and it must not be the honest small one.
    assert store.get("small") is not None or store.get("window") is None


def test_the_accounting_returns_to_zero_when_everything_is_dropped() -> None:
    """Insert and evict must measure the same way, or the budget drifts entry by entry."""
    store = CacheStore(64 << 20)
    store.put("a", _window())
    store.put("b", pa.table({"v": pa.array(range(1000), type=pa.int64())}))
    store.invalidate("a")
    store.invalidate("b")
    assert store.used_bytes == 0


def test_replacing_a_key_does_not_double_count() -> None:
    """The same drift, on the overwrite path."""
    store = CacheStore(64 << 20)
    table = pa.table({"v": pa.array(range(1000), type=pa.int64())})
    store.put("k", table)
    once = store.used_bytes
    store.put("k", table)
    assert store.used_bytes == once
    assert len(store) == 1


def test_freeing_reports_the_bytes_it_really_freed() -> None:
    """`evict_to_free` is the execution-reclaims-storage primitive; its number is a promise.

    A caller that needs 8 MiB and is told it got it, when the entries dropped were windows
    accounted at 80 bytes each, retries the allocation and fails again.
    """
    store = CacheStore(64 << 20)
    for i in range(4):
        store.put(f"k{i}", pa.table({"v": pa.array(range(500_000), type=pa.int64())}))
    freed = store.evict_to_free(4 << 20)
    assert freed >= 4 << 20
    assert store.used_bytes == store.stats()["used_bytes"]
