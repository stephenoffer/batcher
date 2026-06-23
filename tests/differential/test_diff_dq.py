"""Data-quality framework — drop/quarantine/validate/fail and uniqueness.

Drop/quarantine lower to FILTER, so they are checked against the equivalent DuckDB
``WHERE`` / ``WHERE NOT``; the valid/invalid split is asserted to be a total partition.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import DataQualityError


def _people():
    return pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5], pa.int64()),
            "age": pa.array([25, -3, 40, 200, None], pa.int64()),
            "email": ["a@x.com", "bad", "c@y.com", "d@z.com", "e@w.com"],
        }
    )


def test_drop_keeps_only_valid(duck):
    from conftest import assert_same

    t = _people()
    duck.register("p", t)
    out = bt.from_arrow(t).dq.not_null("age").in_range("age", 0, 120).drop().collect()
    # NULL passes in_range but fails not_null; -3 and 200 fail in_range.
    assert_same(out, duck.sql("SELECT * FROM p WHERE age IS NOT NULL AND age BETWEEN 0 AND 120"))


def test_quarantine_is_total_partition(duck):
    t = _people()
    clean, bad = bt.from_arrow(t).dq.not_null("age").in_range("age", 0, 120).quarantine()
    clean_t, bad_t = clean.collect(), bad.collect()
    # valid ⊎ invalid == input (no row lost or duplicated).
    assert clean_t.num_rows + bad_t.num_rows == t.num_rows
    clean_ids = set(clean_t.to_pydict()["id"])
    bad_ids = set(bad_t.to_pydict()["id"])
    assert clean_ids == {1, 3}
    assert bad_ids == {2, 4, 5}
    assert clean_ids.isdisjoint(bad_ids)


def test_matches_and_accepted_values(duck):
    from conftest import assert_same

    t = _people()
    duck.register("p", t)
    out = bt.from_arrow(t).dq.matches("email", r"^[^@]+@[^@]+$").drop().collect()
    assert_same(out, duck.sql(r"SELECT * FROM p WHERE regexp_matches(email, '^[^@]+@[^@]+$')"))


def test_validate_reports_counts():
    t = _people()
    report = bt.from_arrow(t).dq.not_null("age").in_range("age", 0, 120).validate()
    assert not report.ok
    assert report.violations["not_null(age)"] == 1  # the NULL
    assert report.violations["in_range(age, 0, 120)"] == 2  # -3 and 200


def test_fail_raises_on_violation():
    t = _people()
    with pytest.raises(DataQualityError):
        bt.from_arrow(t).dq.in_range("age", 0, 120).fail()


def test_fail_passes_clean_data():
    t = pa.table({"id": [1, 2, 3], "age": [10, 20, 30]})
    ds = bt.from_arrow(t).dq.not_null("id", "age").in_range("age", 0, 120).fail()
    assert ds.collect().num_rows == 3


def test_unique_quarantine_routes_duplicate_rows():
    t = pa.table({"id": [1, 1, 2, 3, 3], "v": ["a", "b", "c", "d", "e"]})
    clean, bad = bt.from_arrow(t).dq.unique(["id"]).quarantine()
    # ids 1 and 3 are duplicated → all their rows are rejected; id 2 is unique.
    assert set(clean.collect().to_pydict()["id"]) == {2}
    assert sorted(bad.collect().to_pydict()["id"]) == [1, 1, 3, 3]


def test_unique_validate_counts_duplicate_keys():
    t = pa.table({"id": [1, 1, 2, 3, 3]})
    report = bt.from_arrow(t).dq.unique(["id"]).validate()
    assert report.violations["unique(id)"] == 2  # keys 1 and 3


def test_foreign_key_finds_orphans(duck):
    from conftest import assert_same

    facts = pa.table({"cid": [1, 2, 3, 9], "amt": [10, 20, 30, 40]})
    dim = pa.table({"id": [1, 2, 3]})
    duck.register("f", facts)
    duck.register("d", dim)
    orphans = (
        bt.from_arrow(facts)
        .dq.foreign_key("cid", references=bt.from_arrow(dim), ref_columns="id")
        .collect()
    )
    assert_same(orphans, duck.sql("SELECT * FROM f WHERE cid NOT IN (SELECT id FROM d)"))
