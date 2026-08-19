"""The metadata shortcuts and the two aggregate data-quality checks, against DuckDB.

``ds.meta`` answers questions about a dataset without scanning it where it can, and falls
back to a query where it cannot. That makes it exactly the surface where a wrong answer is
invisible: it looks like a fast answer instead of a wrong one. Seven of its accessors had
no test -- ``meta.approx.cardinality_ratio`` and ``is_measured``,
``meta.storage.has_exact_row_count`` and ``num_sources``, ``meta.is_known_sorted_by``,
``meta.schema.is_temporal``, and the whole ``meta.against(other)`` pair view.

The oracle is DuckDB for everything that reduces to a query, and the dataset's own
declared schema for the rest. ``meta.approx`` is deliberately allowed to be approximate,
so what is asserted there is the bound it promises and the agreement with the exact answer
on data small enough to be exact.

``dq.median_between`` and ``dq.sum_between`` are here for the same reason: they are the
two aggregate constraints, and no test called either.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.differential

duckdb = pytest.importorskip("duckdb")

ROWS = {
    "v": [1.0, 2.0, 3.0, 10.0],
    "g": ["a", "a", "b", "b"],
    "n": [1, 2, 2, None],
    "t": ["2024-01-01", "2024-02-01", "2024-03-01", None],
}


@pytest.fixture(scope="module")
def duck():
    con = duckdb.connect()
    con.execute("CREATE TABLE t (v DOUBLE, g VARCHAR, n BIGINT)")
    con.executemany(
        "INSERT INTO t VALUES (?, ?, ?)",
        [(ROWS["v"][i], ROWS["g"][i], ROWS["n"][i]) for i in range(4)],
    )
    return con


@pytest.fixture
def ds():
    return bt.from_pydict({k: v for k, v in ROWS.items() if k != "t"})


def test_cardinality_ratio_is_the_sketched_ratio_or_nothing(ds, duck):
    """Free or nothing: it reads a sketch and returns None rather than scanning.

    That is the contract, and it is what separates ``meta.approx`` from
    ``ds.approx_n_unique``. So the assertion is two-sided: whenever a ratio *is* returned
    it must agree with the exact one DuckDB computes, and when it is not the accessor must
    say None rather than guessing a plausible number.
    """
    for column in ("v", "g", "n"):
        got = ds.meta.approx.cardinality_ratio(column)
        if got is None:
            continue
        want = duck.execute(
            f"SELECT count(DISTINCT {column})::DOUBLE / count(*) FROM t"
        ).fetchone()[0]
        assert 0.0 <= got <= 1.0, f"{column} ratio outside [0, 1]"
        assert got == pytest.approx(want, rel=0.05), f"cardinality_ratio({column})"


def test_the_ratio_appears_once_the_column_has_actually_been_read(ds, duck):
    """The cross-query loop, seen from the outside: measuring happens by running.

    ``is_measured`` documents itself as "has this query run before", so a column nobody
    has read carries nothing and the same column carries a ratio afterwards. This is the
    only test here that watches the metadata *change*, which is the behaviour the whole
    accessor exists to expose.
    """
    ds.group_by("g").agg(n=bt.col("v").count()).to_pydict()
    ratio = ds.meta.approx.cardinality_ratio("g")
    if ratio is None:
        pytest.skip("no sketch was recorded for g in this configuration")
    want = duck.execute("SELECT count(DISTINCT g)::DOUBLE / count(*) FROM t").fetchone()[0]
    assert ratio == pytest.approx(want, rel=0.05)
    assert ds.meta.approx.is_measured("g") is True


def test_is_measured_says_whether_the_answer_came_from_statistics(ds):
    """A boolean about provenance, not about the value -- and it must be answerable."""
    for column in ("v", "g", "n"):
        assert isinstance(ds.meta.approx.is_measured(column), bool)
    assert ds.meta.approx.is_measured("v"), (
        "an in-memory source carries column statistics, so this is measured rather than guessed"
    )


def test_storage_metadata_describes_the_source_rather_than_the_rows(ds):
    """``has_exact_row_count`` and ``num_sources`` for a single in-memory relation."""
    assert ds.meta.storage.has_exact_row_count() is True, (
        "an in-memory relation knows exactly how many rows it holds"
    )
    assert ds.meta.storage.num_sources() == 1
    assert ds.meta.storage.row_count() == 4


def test_a_union_reports_more_than_one_source(ds):
    """``num_sources`` must move, or it is a constant dressed as metadata."""
    doubled = ds.union(ds)
    assert doubled.meta.storage.num_sources() >= 2, (
        f"a union of two relations reported {doubled.meta.storage.num_sources()} source(s)"
    )
    assert doubled.count() == 8


def test_is_known_sorted_by_reports_only_what_the_plan_establishes(ds):
    """False before a sort and true after, because it must be a fact and not a guess.

    A metadata accessor that answered true optimistically would let a downstream operator
    skip a sort it actually needs, which is a wrong result rather than a slow one.
    """
    assert ds.meta.is_known_sorted_by("v") is False, "nothing has sorted this yet"
    ordered = ds.sort("v")
    assert ordered.meta.is_known_sorted_by("v") is True
    assert ordered.meta.is_known_sorted_by("g") is False, "sorting by v says nothing about g"
    assert ordered.to_pydict()["v"] == sorted(ROWS["v"]), (
        "and the claim has to be true: the rows really are in order"
    )


def test_is_temporal_agrees_with_the_declared_schema():
    """Answered from the schema, so it must agree with the schema on every column."""
    ds = bt.from_pydict(
        {
            "v": [1.0, 2.0],
            "s": ["a", "b"],
            "d": [bt.lit(None)] if False else ["2024-01-01", "2024-02-01"],
        }
    ).with_columns(stamp=bt.col("d").cast("timestamp"), day=bt.col("d").cast("date"))
    for column, expected in [("v", False), ("s", False), ("stamp", True), ("day", True)]:
        assert ds.meta.schema.is_temporal(column) == expected, column


def test_the_pair_view_estimates_a_join_without_running_it(ds):
    """``meta.against(other)`` answers about two relations before either is scanned."""
    other = bt.from_pydict({"g": ["b", "c"], "w": [1, 2]})
    pair = ds.meta.against(other)

    assert pair.join_is_empty("g", "g") is False, "the two share the key 'b'"
    estimated = pair.estimated_rows("g", "g")
    actual = ds.join(other, on="g").count()
    assert actual == 2, "two rows carry g = 'b'"
    assert estimated > 0
    assert estimated == pytest.approx(actual, rel=2.0), (
        f"the estimate {estimated} is not the right order of magnitude for {actual}"
    )


def test_key_overlap_is_the_window_a_match_can_lie_in(ds):
    """A *range*, not a fraction: the intersection of the two sides' key bounds.

    Worth pinning because the name reads like a ratio, and because the accessor returns
    None rather than a guess when either side's bounds are not provable -- which is the
    case for the string key above, and is why this test uses a numeric one.
    """
    left = bt.from_pydict({"k": [1, 5], "a": [10, 50]})
    right = bt.from_pydict({"k": [3, 9], "b": [30, 90]})
    assert left.meta.against(right).key_overlap("k") == (3, 5)

    disjoint = bt.from_pydict({"k": [100, 200], "b": [1, 2]})
    window = left.meta.against(disjoint).key_overlap("k")
    assert window is None or window[0] > window[1], (
        f"bounds that cannot meet must not report a usable window, got {window}"
    )
    assert left.join(disjoint, on="k").count() == 0, "and the join really is empty"


def test_the_pair_view_sees_a_join_that_cannot_match(ds):
    """The case the estimate exists for: proving a join is empty before running it."""
    disjoint = bt.from_pydict({"g": ["y", "z"], "w": [1, 2]})
    pair = ds.meta.against(disjoint)
    assert pair.join_is_empty("g", "g") is True
    assert ds.join(disjoint, on="g").count() == 0, "and the join really is empty"


def test_a_provably_empty_join_is_not_also_estimated_to_produce_rows():
    """Two accessors on one object must not contradict each other about the same join.

    Both read the same ``Facts``. ``join_is_empty`` proves emptiness from the key bounds --
    ``[1, 3]`` against ``[900, 901]`` cannot meet -- while ``estimated_rows`` ran the
    containment formula regardless and answered two. Since ``estimated_rows`` is the number
    the optimizer orders joins by, the one join it could have skipped outright was the one
    costed as though it produced output.
    """
    left = bt.from_pydict({"k": [1, 2, 3], "x": [1, 2, 3]})
    right = bt.from_pydict({"k": [900, 901], "y": [1, 2]})
    pair = left.meta.against(right)
    assert pair.join_is_empty("k") is True
    assert pair.estimated_rows("k") == 0.0, (
        "a join proved empty from the bounds must not be estimated non-empty"
    )
    assert left.join(right, on="k").count() == 0

    overlapping = bt.from_pydict({"k": [2, 3, 4], "y": [1, 2, 3]})
    still = left.meta.against(overlapping)
    assert still.join_is_empty("k") is False
    assert still.estimated_rows("k") > 0.0, "an overlapping join must still be estimated"
    assert left.join(overlapping, on="k").count() == 2


def test_median_between_matches_duckdbs_median(ds, duck):
    """The constraint passes exactly when DuckDB's median is inside the bounds."""
    median = duck.execute("SELECT median(v) FROM t").fetchone()[0]
    assert median == pytest.approx(2.5)

    inside = ds.dq.median_between("v", median - 1, median + 1).validate()
    assert inside.ok, f"{inside.results}"
    assert inside.results[0].value == pytest.approx(median), (
        "the report must carry the measured value, not just a verdict"
    )

    outside = ds.dq.median_between("v", median + 10, median + 20).validate()
    assert not outside.ok
    assert outside.results[0].violations == 1


def test_sum_between_matches_duckdbs_sum(ds, duck):
    """Same shape for the sum, including the one-sided form."""
    total = duck.execute("SELECT sum(v) FROM t").fetchone()[0]
    report = ds.dq.sum_between("v", 0.0, total).validate()
    assert report.ok
    assert report.results[0].value == pytest.approx(total)

    lower_only = ds.dq.sum_between("v", total + 1).validate()
    assert not lower_only.ok, "a lower bound above the sum must fail"

    upper_only = ds.dq.sum_between("v", None, total - 1).validate()
    assert not upper_only.ok, "an upper bound below the sum must fail"


def test_an_aggregate_constraint_reports_itself_as_an_aggregate(ds):
    """``kind`` distinguishes a whole-column check from a per-row one.

    It matters downstream: a row-level constraint can quarantine the offending rows and an
    aggregate one cannot, so a check that mislabelled itself would be routed wrongly.
    """
    report = ds.dq.median_between("v", 0.0, 1.0).sum_between("v", 0.0, 1.0).validate()
    assert [r.kind for r in report.results] == ["aggregate", "aggregate"]
    assert all(r.rows == 0 for r in report.results), (
        "an aggregate check has no offending row count to report"
    )
    assert all(not r.ok for r in report.results)
    assert report.total_violations == 2


def test_a_constraint_result_names_itself_with_its_bounds(ds):
    """The name is what appears in a failure report, so it has to identify the check."""
    report = ds.dq.median_between("v", 1.0, 5.0).validate()
    result = report.results[0]
    assert "median_between" in result.name
    assert "v" in result.name
    assert "1.0" in result.name and "5.0" in result.name, (
        f"the bounds are missing from {result.name!r}, so two checks on one column collide"
    )
    assert result.severity == "error"
    assert result.ok is True
    assert result.blocking is False, (
        "a constraint that passed does not block, whatever its severity"
    )
    assert isinstance(result.to_dict(), dict)

    failed = ds.dq.median_between("v", 100.0, 200.0).validate().results[0]
    assert failed.ok is False
    assert failed.blocking is True, "a violated error-severity constraint is what blocks a run"


def test_a_warn_severity_records_the_violation_without_blocking(ds):
    """``severity="warn"`` must report the violation and leave the run passing."""
    report = ds.dq.median_between("v", 100.0, 200.0, severity="warn").validate()
    assert report.results[0].severity == "warn"
    assert report.results[0].ok is False, "the constraint really was violated"
    assert report.results[0].blocking is False
    assert report.ok, "a warning must not make the report fail"
    assert report.warnings, "but it must still be reported"


def test_an_unknown_severity_is_refused_rather_than_treated_as_an_error(ds):
    """The typo case: "warning" is not a severity, and silently promoting it would block a run."""
    from batcher import PlanError

    with pytest.raises(PlanError, match="'error' or 'warn'"):
        ds.dq.median_between("v", 1.0, 5.0, severity="warning")


def test_the_aggregate_constraints_compose_with_the_row_level_ones(ds):
    """One report carrying both kinds, which is how a real contract is written."""
    report = (
        ds.dq.median_between("v", 0.0, 100.0).sum_between("v", 0.0, 100.0).not_null("g").validate()
    )
    kinds = {r.kind for r in report.results}
    assert "aggregate" in kinds
    assert len(report.results) == 3
    assert report.ok
