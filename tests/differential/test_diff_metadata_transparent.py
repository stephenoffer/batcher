"""The metadata layer must speed the *ordinary* API up without anyone asking it to.

`ds.meta` is an introspection surface, not the way you get the speed. The speed belongs to
`ds.join(...)`, `ds.dq.…fail()`, `ds.null_count()` — the calls people already write. This file
pins that: for each shortcut the optimizer now takes on the ordinary path, the result must be
**identical** to the same query with the metadata layer switched off, and identical to DuckDB.

That is the whole risk of this class of optimization, and it is not theoretical. Every rewrite
here can *delete rows*: pruning a join to empty, discharging a data-quality contract without
looking, folding a projection to a constant. A rewrite that is wrong does not run slowly — it
returns the wrong answer, quickly. So each one is tested twice: once for the shape that should
fire (and be right), and once for the neighbouring shape that must **not** fire.

`map_batches` is the switch: the IR cannot describe a Python callback, so Kyber declines to
reason about the plan at all and everything falls through to the engine. The callback is the
identity, so the relation is unchanged — it is the same query, computed the long way round.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import assert_tables_equal

pytestmark = pytest.mark.differential

bt = pytest.importorskip("batcher")

ROWS = 2_000


def _forced(ds):
    """The same relation with the metadata layer switched off (see the module docstring)."""
    return ds.map_batches(lambda batch: batch)


@pytest.fixture(scope="module")
def facts_path(tmp_path_factory) -> str:
    """A Parquet file whose footer records everything these rewrites reason from."""
    path = str(tmp_path_factory.mktemp("transparent") / "facts.parquet")
    table = pa.table(
        {
            "id": pa.array(range(ROWS), pa.int64()),  # unique, no nulls → a key
            "amount": pa.array([(i % 100) + 1 for i in range(ROWS)], pa.int64()),  # 1..100
            "note": pa.array([None if i % 5 == 0 else f"n{i}" for i in range(ROWS)]),
        }
    )
    pq.write_table(table, path, row_group_size=ROWS // 4)
    return path


@pytest.fixture
def ds(facts_path):
    return bt.read.parquet(facts_path)


# --- joins: a disjoint key range proves the join empty, before either side is read ---------


@pytest.mark.parametrize(
    ("right_keys", "how"),
    [
        ([-3, -2, -1], "inner"),  # entirely below the left range → prunes
        ([10**9, 10**9 + 1], "inner"),  # entirely above → prunes
        ([-3, -2, -1], "semi"),
        ([-3, -2, -1], "left"),  # preserved side survives whole — must NOT become empty
        ([-3, -2, -1], "anti"),  # every left row survives — must NOT become empty
        ([1, 2, 3], "inner"),  # overlapping → must NOT prune
        ([1, 2, 3], "left"),
    ],
)
def test_join_pruning_never_changes_the_answer(ds, right_keys, how):
    """A join the optimizer prunes must return exactly what running it returns.

    Both directions matter. An inner/semi join over disjoint keys is provably empty and is
    skipped — but a `left`/`anti` join over the *same* disjoint keys keeps every left row, so a
    rule that emptied it would delete the entire answer. And an overlapping range proves
    nothing at all, so it must simply run.
    """
    other = bt.from_pydict({"id": right_keys})
    optimized = ds.join(other, on="id", how=how).collect()
    executed = _forced(ds).join(other, on="id", how=how).collect()
    assert_tables_equal(optimized, executed)


@pytest.mark.parametrize(
    ("left_keys", "right_keys"),
    [
        pytest.param([-0.0, 1.5, 2.0], [0.0, 1.5], id="minus-zero-left"),
        pytest.param([0.0, 1.5, 2.0], [-0.0, 1.5], id="minus-zero-right"),
        pytest.param([float("nan"), 1.5], [float("nan"), 1.5], id="nan-key"),
    ],
)
def test_a_float_join_key_is_never_pruned_by_a_range(left_keys, right_keys):
    """A bound-derived filter must never delete a row the join's *canonicalized* key matches.

    This is the sharp edge of every join-key bound optimization, and it drew blood. An equi-join
    canonicalizes its float key (`bc_runtime::keys` folds `-0.0` into `0.0` and every NaN into
    one value), so `-0.0` on one side **matches** `0.0` on the other. But `runtime_join_filter`
    pushes `key BETWEEN other_min AND other_max`, and `BETWEEN` does not canonicalize — on the
    engine's total order `-0.0 < 0.0`, so the filter deletes exactly the row the join would have
    matched. Joining `[-0.0, 1.5, 2.0]` against `[0.0, 1.5]` returned **one** row where the join
    returns two.

    The bug sat dormant for as long as float join-key bounds were never fetched from the footer,
    and woke the moment they were. Every rule that reasons from a join key's bounds now refuses
    an ambiguous float one (`plan.stats.ambiguous_float_bound`); this pins that they all do.
    """
    left = bt.from_arrow(
        pa.table({"k": pa.array(left_keys, pa.float64()), "l": list(range(len(left_keys)))})
    )
    right = bt.from_arrow(
        pa.table({"k": pa.array(right_keys, pa.float64()), "r": list(range(len(right_keys)))})
    )
    optimized = left.join(right, on="k", how="inner").collect()
    executed = _forced(left).join(right, on="k", how="inner").collect()
    assert_tables_equal(optimized, executed)
    assert optimized.num_rows == 2  # the zero/NaN pair, and 1.5


def test_disjoint_join_matches_duckdb(duck, ds, facts_path):
    """...and the pruned-to-empty join agrees with the oracle, not just with our executor."""
    other = pa.table({"id": pa.array([-3, -2, -1], pa.int64())})
    duck.register("l", pq.read_table(facts_path))
    duck.register("r", other)
    got = ds.join(bt.from_arrow(other), on="id", how="inner").select("id", "amount").collect()
    want = duck.sql("SELECT l.id, l.amount FROM l JOIN r USING (id)").to_arrow_table()
    assert_tables_equal(got, want)


# --- data quality: a contract that holds is discharged from the footer ----------------------


def test_a_contract_that_holds_is_discharged_without_a_scan(ds):
    """`fail()`/`drop()`/`validate()` must agree with executing them, when the contract holds."""
    contract = lambda d: d.dq.not_null("id").in_range("amount", 0, 10_000)  # noqa: E731

    assert contract(ds).validate().violations == contract(_forced(ds)).validate().violations
    assert_tables_equal(contract(ds).fail().collect(), contract(_forced(ds)).fail().collect())
    assert_tables_equal(contract(ds).drop().collect(), contract(_forced(ds)).drop().collect())

    clean, rejected = contract(ds).quarantine()
    ref_clean, ref_rejected = contract(_forced(ds)).quarantine()
    assert_tables_equal(clean.collect(), ref_clean.collect())
    assert_tables_equal(rejected.collect(), ref_rejected.collect())


@pytest.mark.parametrize(
    "contract",
    [
        pytest.param(lambda d: d.dq.in_range("amount", 0, 50), id="range-violated"),
        pytest.param(lambda d: d.dq.not_null("note"), id="not_null-violated"),
        pytest.param(lambda d: d.dq.in_range("amount", 0, 10_000).not_null("note"), id="mixed"),
        pytest.param(lambda d: d.dq.check(bt.col("amount") > 50, name="c"), id="custom-check"),
    ],
)
def test_a_contract_that_fails_still_fails(ds, contract):
    """The dangerous direction: a violated contract must never be discharged from metadata.

    A shortcut that wrongly proved a contract clean would wave bad data through the gate whose
    entire job is to stop it — the most consequential wrong answer this layer could give.
    """
    assert contract(ds).validate().violations == contract(_forced(ds)).validate().violations
    assert not contract(ds).validate().ok
    with pytest.raises(Exception, match="data-quality"):
        contract(ds).fail()
    assert_tables_equal(contract(ds).drop().collect(), contract(_forced(ds)).drop().collect())


def test_a_custom_check_is_never_discharged_from_metadata(ds):
    """A `check()` predicate may be NULL for a row, and a NULL validity is a *violation*.

    The metadata probe asks whether `filter(NOT valid)` keeps a row — and `NOT NULL` is NULL,
    which that filter drops. So a custom check must not take the shortcut. Here the predicate
    is null exactly where `note` is, and those rows must be counted as violations.
    """
    contract = ds.dq.check(bt.col("note").str.len() > 0, name="note_nonempty")
    assert (
        contract.validate().violations
        == _forced(ds)
        .dq.check(bt.col("note").str.len() > 0, name="note_nonempty")
        .validate()
        .violations
    )
    assert contract.validate().violations["note_nonempty"] == ROWS // 5  # every null note


# --- constant folding through an aggregate: `null_count()` and friends ----------------------


def test_null_count_is_answered_from_metadata_and_agrees(ds, duck, facts_path):
    """`null_count()` lowers to `count(*) - count(col)` — constants the footer already holds."""
    got = ds.null_count().collect()
    executed = _forced(ds).null_count().collect()
    assert_tables_equal(got, executed)

    duck.register("t", pq.read_table(facts_path))
    want = duck.sql(
        "SELECT count(*) - count(id) AS id, count(*) - count(amount) AS amount, "
        "count(*) - count(note) AS note FROM t"
    ).to_arrow_table()
    assert_tables_equal(got, want)


def test_a_projection_over_non_constant_columns_is_not_folded(ds):
    """The neighbouring shape that must not fire: `amount - id` is not a constant."""
    got = ds.select(diff=bt.col("amount") - bt.col("id")).collect()
    executed = _forced(ds).select(diff=bt.col("amount") - bt.col("id")).collect()
    assert_tables_equal(got, executed)
    assert got.num_rows == ROWS


# --- the IS NOT NULL filter keeps its column exact -------------------------------------------


def test_count_through_a_not_null_filter_is_exact_and_correct(ds, duck, facts_path):
    """`filter(col IS NOT NULL)` drops exactly the rows the null count counts — so the surviving
    count is exact, and the column's bounds survive (nulls are not in a min or a max anyway)."""
    got = ds.filter(bt.col("note").is_not_null()).count()
    executed = _forced(ds).filter(bt.col("note").is_not_null()).count()
    duck.register("t", pq.read_table(facts_path))
    want = duck.sql("SELECT count(*) FROM t WHERE note IS NOT NULL").fetchone()[0]
    assert got == executed == want

    # ...and the min/max carried through that filter must still be the real ones.
    assert ds.filter(bt.col("amount").is_not_null()).min("amount") == _forced(ds).filter(
        bt.col("amount").is_not_null()
    ).min("amount")


# --- string columns keep their exact null count despite untrustworthy bounds ----------------


def test_string_column_null_answers_come_from_metadata(ds, duck, facts_path):
    """A string column's footer bounds may be truncated, but its null count is exact.

    Tying the two together threw the exact null count away with the inexact bounds, so every
    null question about a string column — the columns most tables are made of — scanned. The
    null count rides on its own provenance now; each of these must agree with executing and with
    the oracle, over the string `note` column.
    """
    duck.register("t", pq.read_table(facts_path))
    nulls = duck.sql("SELECT count(*) - count(note) FROM t").fetchone()[0]

    assert ds.n_null("note") == _forced(ds).n_null("note") == nulls
    assert ds.has_nulls("note") == _forced(ds).has_nulls("note") == (nulls > 0)
    assert ds.null_count().to_pydict()["note"] == [nulls]
    # `count(note)` derives from the null count, not the bounds — so it is exact too.
    got = ds.agg(c=bt.col("note").count()).collect()
    assert_tables_equal(got, _forced(ds).agg(c=bt.col("note").count()).collect())


def test_a_partial_scan_never_teaches_a_source_level_distinct_count(tmp_path):
    """A distinct count learned from a *subset* of the rows must not answer for the whole source.

    The learning loop records a scanned column's ndv under the *source's* identity, so a query
    that scanned only part of the source (a pushed filter, a limit) would file a partial distinct
    count as the source's — and `approx_n_unique`, which reads exactly that record, would return
    it for the whole table. Reproduced directly: `filter(id < 100).collect()` scanned 100 of
    2,000 rows, and `approx_n_unique("id")` then answered ~100.
    """
    path = str(tmp_path / "u.parquet")
    n = 2_000
    pq.write_table(pa.table({"id": pa.array(range(n), pa.int64())}), path, row_group_size=n // 4)

    poisoned = bt.read.parquet(path)
    poisoned.filter(bt.col("id") < 100).collect()  # a partial scan — must teach nothing
    after_partial = poisoned.approx_n_unique("id")
    # Either nothing was learned (fell back to a real sketch) or it is near the truth — never 100.
    assert after_partial is None or after_partial > n * 0.5

    whole = bt.read.parquet(path)
    whole.collect()  # a whole scan — may learn
    learned = whole.approx_n_unique("id")
    assert learned is None or abs(learned - n) / n < 0.1  # within HLL tolerance of the truth
