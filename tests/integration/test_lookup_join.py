"""`lookup_join` end to end: batching, caching across batches, streaming, and the edges.

`tests/differential/test_diff_lookup_join.py` proves the *rows* are what a SQL join would
return. This proves the things a differential test cannot see: that the store is opened
once per worker rather than once per batch, that the cache actually spans batches, that a
key-value store is asked for each key once rather than once per occurrence, and that the
whole thing works over a stream where nothing is ever materialized.

The store counts what it is asked for, because those counts are the entire value of the
feature — a lookup join that returns the right rows while issuing a round trip per row is
correct and useless.
"""

from __future__ import annotations

from typing import ClassVar

import pyarrow as pa
import pytest

import batcher as bt

pytest.importorskip("batcher._native", reason="native engine not built")

_DIM = pa.table(
    {
        "k": ["c1", "c2", "c3"],
        "name": ["Ann", "Bob", None],
        "tier": [1, 2, 3],
    }
)
_SCHEMA = {"name": "string", "tier": "int64"}


class _CountingLookup:
    """An `InMemoryLookup` that records every fetch, and how many times it was built."""

    builds = 0
    fetches: ClassVar[list[list[str]]] = []

    def __init__(self) -> None:
        from batcher.io.lookup.backends import InMemoryLookup

        type(self).builds += 1
        self._inner = InMemoryLookup(_DIM, "k")

    def multi_get(self, keys):
        type(self).fetches.append(list(keys))
        return self._inner.multi_get(keys)

    def value_schema(self):
        return self._inner.value_schema()

    def close(self):
        self._inner.close()


@pytest.fixture
def counting(monkeypatch):
    """Point `lookup_join` at a store that counts what it is asked for."""
    from batcher.io.lookup import spec

    _CountingLookup.builds = 0
    _CountingLookup.fetches = []
    monkeypatch.setattr(spec, "build_lookup", lambda uri, schema, options: _CountingLookup())
    return _CountingLookup


def _fetched(store) -> list[str]:
    return [key for call in store.fetches for key in call]


def test_a_repeated_key_costs_one_fetch_across_the_whole_input(counting):
    rows = {"k": ["c1"] * 50 + ["c2"] * 50, "v": list(range(100))}
    out = (
        bt.from_pydict(rows)
        .lookup_join("memory://", on="k", schema=_SCHEMA, batch_size=10, num_workers=1)
        .collect()
    )
    assert out.num_rows == 100
    # 100 rows over 10 batches, two distinct keys. Anything above two means the cache is
    # not spanning batches, which is the whole reason the stage is a class.
    assert sorted(_fetched(counting)) == ["c1", "c2"]


def test_an_absent_key_is_fetched_once_not_once_per_batch(counting):
    rows = {"k": ["zz"] * 40, "v": list(range(40))}
    out = (
        bt.from_pydict(rows)
        .lookup_join("memory://", on="k", schema=_SCHEMA, batch_size=4, num_workers=1)
        .collect()
    )
    assert out.num_rows == 40
    assert out.column("name").null_count == 40
    assert _fetched(counting) == ["zz"], "the absence must be remembered, not re-asked"


def test_the_store_is_opened_once_per_worker_not_once_per_batch(counting):
    rows = {"k": ["c1"] * 100, "v": list(range(100))}
    bt.from_pydict(rows).lookup_join(
        "memory://", on="k", schema=_SCHEMA, batch_size=5, num_workers=1
    ).collect()
    assert counting.builds == 1


def test_disabling_the_cache_fetches_every_batch(counting):
    rows = {"k": ["c1"] * 20, "v": list(range(20))}
    bt.from_pydict(rows).lookup_join(
        "memory://", on="k", schema=_SCHEMA, batch_size=5, num_workers=1, cache_size=0
    ).collect()
    # The setting exists to measure what the cache is buying, so it has to actually
    # disable it: four batches, four fetches.
    assert len(counting.fetches) == 4


def test_a_null_key_is_never_looked_up(counting):
    out = (
        bt.from_pydict({"k": [None, None, "c1"], "v": [1, 2, 3]})
        .lookup_join("memory://", on="k", schema=_SCHEMA, num_workers=1)
        .collect()
    )
    assert _fetched(counting) == ["c1"]
    assert out.column("name").to_pylist() == [None, None, "Ann"]


def test_an_integer_key_joins_against_a_string_keyspace(monkeypatch):
    from batcher.io.lookup import backends, spec

    dim = pa.table({"k": ["1", "2"], "label": ["one", "two"]})
    monkeypatch.setattr(
        spec, "build_lookup", lambda uri, schema, options: backends.InMemoryLookup(dim, "k")
    )
    out = (
        bt.from_pydict({"k": [1, 2, 3], "v": [10, 20, 30]})
        .lookup_join("memory://", on="k", schema={"label": "string"}, num_workers=1)
        .collect()
    )
    assert out.column("label").to_pylist() == ["one", "two", None]


def test_it_streams_without_materializing(counting):
    rows = {"k": ["c1", "c2"] * 500, "v": list(range(1000))}
    ds = bt.from_pydict(rows).lookup_join(
        "memory://", on="k", schema=_SCHEMA, batch_size=100, num_workers=1
    )
    seen = 0
    for batch in ds.iter_batches():
        assert "name" in batch.schema.names
        seen += batch.num_rows
    assert seen == 1000


def test_the_schema_is_known_before_anything_runs(counting):
    ds = bt.from_pydict({"k": ["c1"], "v": [1]}).lookup_join(
        "memory://", on="k", schema=_SCHEMA, prefix="d_"
    )
    # The columns have to be answerable from the plan: a downstream `select` or a schema
    # read must not have to execute the join, and under a distributed run two workers must
    # agree about the shape before either has seen a row.
    assert ds.columns == ["k", "v", "d_name", "d_tier"]
    assert counting.fetches == []


def test_an_empty_input_produces_an_empty_result_with_the_right_columns(counting):
    out = (
        bt.from_pydict({"k": [], "v": []})
        .lookup_join("memory://", on="k", schema=_SCHEMA, num_workers=1)
        .collect()
    )
    assert out.num_rows == 0
    assert set(out.column_names) == {"k", "v", "name", "tier"}
    assert counting.fetches == []


def test_a_cache_ttl_is_accepted_as_a_duration_string(counting):
    out = (
        bt.from_pydict({"k": ["c1"], "v": [1]})
        .lookup_join("memory://", on="k", schema=_SCHEMA, cache_ttl="5m", num_workers=1)
        .collect()
    )
    assert out.column("name").to_pylist() == ["Ann"]


def test_a_malformed_cache_ttl_is_refused(counting):
    from batcher._internal.errors import PlanError

    with pytest.raises(PlanError):
        bt.from_pydict({"k": ["c1"], "v": [1]}).lookup_join(
            "memory://", on="k", schema=_SCHEMA, cache_ttl="soon", num_workers=1
        ).collect()


def test_a_parametrized_dtype_survives_the_trip_to_the_worker(monkeypatch):
    """A `timestamp(us)` or `decimal(p,s)` column must resolve on the worker too.

    The schema is resolved on the driver to validate it and to name the output columns,
    and the *unresolved* mapping is what travels. Sending the resolved one instead looked
    equivalent and was not: `resolve_dtype("timestamp(us)")` renders as ``timestamp[us]``,
    which `resolve_dtype` does not parse back, so a temporal lookup column planned cleanly
    and then failed inside the first batch.
    """
    import decimal

    from batcher.io.lookup import backends, spec

    dim = pa.table(
        {
            "k": ["a", "b"],
            "ts": pa.array([1_000_000, 2_000_000], pa.timestamp("us")),
            "d": pa.array([decimal.Decimal("1.5"), decimal.Decimal("2.5")], pa.decimal128(12, 4)),
        }
    )
    monkeypatch.setattr(
        spec, "build_lookup", lambda uri, schema, options: backends.InMemoryLookup(dim, "k")
    )
    out = (
        bt.from_pydict({"k": ["a", "z", "b"], "v": [1, 2, 3]})
        .lookup_join(
            "memory://",
            on="k",
            schema={"ts": "timestamp(us)", "d": "decimal(12,4)"},
            num_workers=1,
        )
        .collect()
    )
    assert out.schema.field("ts").type == pa.timestamp("us")
    assert out.schema.field("d").type == pa.decimal128(12, 4)
    assert out.column("d").to_pylist()[1] is None
