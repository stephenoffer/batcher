"""Data-quality framework — drop/quarantine/validate/fail and uniqueness.

Drop/quarantine lower to FILTER, so they are checked against the equivalent DuckDB
``WHERE`` / ``WHERE NOT``; the valid/invalid split is asserted to be a total partition.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

import batcher as bt
from _harness import assert_same
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


def test_unique_validate_counts_the_rows_it_rejects_not_the_keys():
    """This assertion used to read ``== 2  # keys 1 and 3``. That was the wrong contract.

    `ValidationReport.total_violations` is documented as "the total number of violating rows
    summed across every constraint", and every row-wise constraint reports rows. The summing
    is what settles it: a total that adds a *key* count from `unique` to a *row* count from
    `not_null` is dimensionally meaningless, so `unique` owes rows like the rest.

    The test directly above already showed `quarantine` rejecting ``[1, 1, 3, 3]`` — four rows
    — for this very input. Both facts lived in this file and nothing compared them.
    """
    t = pa.table({"id": [1, 1, 2, 3, 3]})
    dq = bt.from_arrow(t).dq.unique(["id"])
    report = dq.validate()
    assert report.violations["unique(id)"] == 4  # rows 1, 1, 3, 3 — not the two keys
    _clean, bad = bt.from_arrow(t).dq.unique(["id"]).quarantine()
    assert report.total_violations == bad.count()


def test_foreign_key_finds_orphans(duck):
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


# --- the report and the split must agree on how many rows failed -----------------------


@pytest.mark.parametrize(
    ("ids", "rejected"),
    [
        ([1, 1, 2], 2),
        ([1, 1, 1, 2], 3),
        ([1, 1, 2, 2, 3], 4),
        ([7, 7, 7, 7], 4),
        ([1, None, None, 2], 2),
        ([1, 2, 3], 0),
    ],
    ids=["pair", "triple", "two-pairs", "all-same", "null-key", "clean"],
)
def test_validate_counts_the_rows_a_uniqueness_check_rejects(ids, rejected):
    """`total_violations` is documented as violating *rows*, and `unique` owes that too.

    It counted the duplicated *groups* instead — one `count()` over the grouped keys — so it
    under-reported by the size of each group, and by an unbounded factor: over
    ``[1, 1, 1, 2]`` `drop` removes three rows and `quarantine` rejects three, while the
    report said `1`. A key repeated a thousand times still said `1`.

    `validate()` and `quarantine()` are the two non-raising paths, and someone reading a
    monitoring number beside a dead-letter sink is entitled to have them agree.
    """
    ds = bt.from_arrow(pa.table({"id": pa.array(ids, pa.int64())}))

    report = ds.dq.unique("id").validate()
    _clean, bad = ds.dq.unique("id").quarantine()
    kept = ds.dq.unique("id").drop().count()

    assert bad.count() == rejected, "quarantine() rejected an unexpected number of rows"
    assert report.total_violations == rejected, (
        f"validate() reported {report.total_violations} for {rejected} rejected rows"
    )
    assert report.violations["unique(id)"] == rejected
    assert kept + bad.count() == len(ids), "drop and quarantine must partition the input"
    assert report.ok is (rejected == 0)


def test_validate_agrees_with_quarantine_across_every_constraint_kind():
    """The row-wise constraints already agreed; this pins all of them together.

    A report whose count means "rows" for one constraint and "keys" for another is worse than
    either convention, because nothing tells the reader which they are looking at.
    """
    table = pa.table(
        {
            "id": pa.array([1, 2, 2, 4, None], pa.int64()),
            "score": pa.array([10, -5, 50, 200, 7], pa.int64()),
            "name": pa.array(["ana", "bob", None, "dan", "eve"]),
        }
    )
    ds = bt.from_arrow(table)
    for label, apply_to in (
        ("not_null", lambda dq: dq.not_null("id")),
        ("in_range", lambda dq: dq.in_range("score", 0, 100)),
        ("accepted_values", lambda dq: dq.accepted_values("name", ["ana", "bob"])),
        ("unique", lambda dq: dq.unique("id")),
        ("check", lambda dq: dq.check(bt.col("score") > 0, name="positive")),
    ):
        _clean, bad = apply_to(ds.dq).quarantine()
        report = apply_to(ds.dq).validate()
        assert report.total_violations == bad.count(), (
            f"{label}: validate() says {report.total_violations}, quarantine() rejects "
            f"{bad.count()}"
        )
