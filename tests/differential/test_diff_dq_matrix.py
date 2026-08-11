"""Every `ds.dq` terminal x every execution path, on one edge-case-loaded input.

`collect()`, `collect(spill=True)` and `iter_batches()` are three *schedulings* of the same
semantics, so a contract must decide the same rows on all three — on nulls, on an empty
relation, on a single row, on `-0.0`/NaN, on duplicate keys, and on an input long enough to
cross a morsel boundary. The per-constraint tests each cover their own constraint on its own
happy path; the *combinations* are nobody's job, which is the gap `test_diff_operator_matrix`
was written for and this is the data-quality half of.

Two invariants carry most of the weight here, because both are the kind that hold on a
single batch and break under partitioning:

* **The split is total.** ``clean.count() + rejected.count() == input.count()``, on every
  path and every input. Validity is forced to a non-null boolean precisely so a row cannot
  fall into neither side, and a NULL-valued `check()` predicate is what would otherwise do it.
* **The report agrees with the split.** `validate()` counts rows a different way than
  `quarantine()` removes them — a keyless aggregate against a filter, and for uniqueness a
  `group_by` against a window — so the two numbers agreeing is a real cross-check rather
  than a tautology. They disagreed once already, by the size of every duplicate group.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from _harness import assert_tables_equal

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")

#: Nulls in every column that a constraint reads, both zeros and a NaN in the float, a key
#: that repeats, an empty string, and a value out of every plausible range.
BASE = pa.table(
    {
        "id": pa.array([1, 2, None, 4, 5, 2, 7, None, 9, 10], pa.int64()),
        "amount": pa.array(
            [1.5, -0.0, 0.0, None, float("nan"), -2.5, float("inf"), 3.0, 7.5, -1.0],
            pa.float64(),
        ),
        "code": pa.array(["US", "", None, "CA", "US", "  ", "ZZ", "CA", None, "US"]),
        "ref": pa.array([1, 1, None, 2, 99, 1, None, 2, 3, 99], pa.int64()),
    }
)
#: `BASE` repeated past two 16,384-row morsels, so the paths compared here are actually
#: different rather than three names for one batch.
MULTIBATCH = pa.concat_tables([BASE] * 3600)  # 36,000 rows

INPUTS = {
    "base": BASE,
    "empty": BASE.slice(0, 0),
    "single": BASE.slice(0, 1),
    "multibatch": MULTIBATCH,
}

REFERENCE = pa.table({"ref": pa.array([1, 2, 3], pa.int64())})


def _ref() -> bt.Dataset:
    return bt.from_arrow(REFERENCE)


#: name -> a chain builder. One per constraint *kind*, because the kinds lower differently:
#: a row expression, a window count, a join, and a user predicate that may evaluate to NULL.
CHAINS = {
    "values": lambda d: d.dq.not_null("id").is_finite("amount"),
    "text": lambda d: d.dq.not_empty("code").accepted_values("code", ["US", "CA"]),
    "unique": lambda d: d.dq.unique("id"),
    "unique_composite": lambda d: d.dq.unique(["id", "code"]),
    "reference": lambda d: d.dq.references("ref", to=_ref()),
    "custom_nullable": lambda d: d.dq.check(bt.col("amount") > 0, name="amount_positive"),
    "scoped": lambda d: d.dq.where(bt.col("code") == "US").not_null("amount"),
    "tolerated": lambda d: d.dq.not_null("id", mostly=0.5),
    "warned": lambda d: d.dq.not_null("id", severity="warn").is_finite("amount"),
    "mixed": lambda d: d.dq.not_null("id").unique("id").references("ref", to=_ref()),
}


def _stream(ds) -> pa.Table:
    """Materialize through the streaming path, falling back to `collect()` when empty.

    The fallback is deliberate rather than incidental: it streams a handle and then collects
    *the same one*, which is the sequence that found the plan-corruption bug this matrix
    surfaced (`test_diff_plan_reuse` is its regression test). Building the empty table from
    `ds.schema` instead would be equally correct and would quietly stop exercising it.
    """
    batches = list(ds.iter_batches())
    if not batches:
        return ds.collect().slice(0, 0)
    return pa.Table.from_batches(batches)


@pytest.mark.parametrize("chain", sorted(CHAINS))
@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_drop_agrees_across_paths(chain, shape):
    """Spill and streaming are schedulings of `collect()`, so they must equal it exactly."""
    table = INPUTS[shape]
    build = CHAINS[chain]
    oracle = build(bt.from_arrow(table)).drop().collect()
    assert_tables_equal(build(bt.from_arrow(table)).drop().collect(spill=True), oracle)
    assert_tables_equal(_stream(build(bt.from_arrow(table)).drop()), oracle)


@pytest.mark.parametrize("chain", sorted(CHAINS))
@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_quarantine_is_a_total_partition_on_every_path(chain, shape):
    table = INPUTS[shape]
    build = CHAINS[chain]
    for materialize in (
        lambda d: d.collect(),
        lambda d: d.collect(spill=True),
        _stream,
    ):
        clean, bad = build(bt.from_arrow(table)).quarantine()
        kept, rejected = materialize(clean), materialize(bad)
        assert kept.num_rows + rejected.num_rows == table.num_rows
        assert kept.schema.names == rejected.schema.names == table.schema.names


@pytest.mark.parametrize("chain", sorted(CHAINS))
@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_drop_is_the_clean_side_of_quarantine(chain, shape):
    table = INPUTS[shape]
    build = CHAINS[chain]
    clean, _bad = build(bt.from_arrow(table)).quarantine()
    assert_tables_equal(build(bt.from_arrow(table)).drop().collect(), clean.collect())


@pytest.mark.parametrize("chain", sorted(CHAINS))
@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_the_report_counts_what_the_split_rejects(chain, shape):
    """Counted by aggregate, removed by filter — two ways to the same number."""
    table = INPUTS[shape]
    build = CHAINS[chain]
    report = build(bt.from_arrow(table)).validate()
    _clean, bad = build(bt.from_arrow(table)).quarantine()
    enforced = sum(r.violations for r in report.results if r.severity == "error")
    if len(report.results) == 1:
        # One constraint: the rejected side is exactly its violating rows.
        assert enforced == bad.count()
    else:
        # Several: a row can violate more than one, so the sum bounds the rejected side.
        assert bad.count() <= enforced
        assert (bad.count() > 0) == (enforced > 0)


@pytest.mark.parametrize("chain", sorted(CHAINS))
@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_annotate_keeps_every_row_on_every_path(chain, shape):
    table = INPUTS[shape]
    build = CHAINS[chain]
    labelled = build(bt.from_arrow(table)).annotate()
    oracle = labelled.collect()
    assert oracle.num_rows == table.num_rows
    assert_tables_equal(build(bt.from_arrow(table)).annotate().collect(spill=True), oracle)
    assert_tables_equal(_stream(build(bt.from_arrow(table)).annotate()), oracle)


@pytest.mark.parametrize("chain", sorted(set(CHAINS) - {"warned"}))
@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_annotate_labels_exactly_the_rows_quarantine_rejects(chain, shape):
    """A row carries a label iff it lands on the rejected side — the two must not drift.

    `warned` is excluded and gets its own test below: a warning is *named* per row and
    *enforced* nowhere, which is exactly the case where the two counts should differ.
    """
    table = INPUTS[shape]
    build = CHAINS[chain]
    labelled = build(bt.from_arrow(table)).annotate().collect().to_pydict()
    _clean, bad = build(bt.from_arrow(table)).quarantine()
    flagged = sum(1 for label in labelled["dq_failed"] if label)
    assert flagged == bad.count()


@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_a_warning_is_named_per_row_and_enforced_nowhere(shape):
    table = INPUTS[shape]
    build = CHAINS["warned"]
    labelled = build(bt.from_arrow(table)).annotate().collect().to_pydict()
    _clean, bad = build(bt.from_arrow(table)).quarantine()
    named_by_the_warning = sum(1 for label in labelled["dq_failed"] if "not_null(id)" in label)
    nulls = sum(1 for value in table.column("id").to_pylist() if value is None)
    # The warning names every row it applies to...
    assert named_by_the_warning == nulls
    # ...and the rejected side is decided by the enforced constraint alone.
    non_finite = sum(
        1
        for value in table.column("amount").to_pylist()
        if value is not None and (value != value or value in (float("inf"), float("-inf")))
    )
    assert bad.count() == non_finite


@pytest.mark.parametrize("shape", sorted(INPUTS))
def test_relation_level_checks_agree_across_paths(shape):
    """An aggregate merges, so the bound holds identically however the work was scheduled."""
    table = INPUTS[shape]
    ds = bt.from_arrow(table)
    gate = ds.dq.row_count_between(0).null_rate_below("id", 1.0).distinct_count_between("code", 0)
    report = gate.validate()
    assert report.ok, report.violations
    assert report.result("row_count_between(0, None)").value == table.num_rows
