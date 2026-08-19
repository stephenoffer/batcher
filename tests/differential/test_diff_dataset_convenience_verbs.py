"""The ``Dataset`` convenience verbs, against DuckDB and pandas.

Eleven methods on the public ``Dataset`` had no test: the null and constant droppers, the
class-balance pair, one-hot encoding, the two text-budget filters, the three-way split,
the CSV writer and ``glimpse``. Each is a shorthand for a query a user could write by
hand, which is exactly why they are easy to leave untested and easy to get subtly wrong --
``dropna`` over the wrong subset, ``get_dummies`` missing a category, a split whose parts
overlap.

The oracle is whichever system already defines the verb: DuckDB for the relational ones,
pandas for ``dropna`` and ``get_dummies``, which is where those names come from. The split
and the balancer are checked against their invariants instead, because both are
randomized and neither has a reference implementation to agree with -- what matters is
that the parts partition the input, that the split is stable under a seed, and that
balancing actually equalizes the classes.
"""

from __future__ import annotations

import pytest

import batcher as bt

pytestmark = pytest.mark.differential

duckdb = pytest.importorskip("duckdb")

#: A frame with a constant column, nulls in two columns, an empty string, unbalanced
#: classes and text of very different lengths -- one row for each verb to bite on.
ROWS = {
    "g": ["a", "a", "a", "b", "x", "x"],
    "n": [1, 2, 3, 4, None, 6],
    "const": [7, 7, 7, 7, 7, 7],
    "txt": ["hello world", "hi", "", "a much longer piece of text here", None, "mid"],
}


@pytest.fixture
def ds():
    """The fixture as a Batcher dataset."""
    return bt.from_pydict(ROWS)


@pytest.fixture(scope="module")
def duck():
    """The same fixture in DuckDB."""
    con = duckdb.connect()
    con.execute("CREATE TABLE t (g VARCHAR, n BIGINT, const BIGINT, txt VARCHAR)")
    con.executemany(
        "INSERT INTO t VALUES (?, ?, ?, ?)",
        [(ROWS["g"][i], ROWS["n"][i], ROWS["const"][i], ROWS["txt"][i]) for i in range(6)],
    )
    return con


def test_dropna_matches_duckdb_and_pandas(ds, duck):
    """The whole-row form and the subset form, which differ on this fixture."""
    pandas = pytest.importorskip("pandas")
    whole = ds.dropna().to_pydict()
    want = duck.execute(
        "SELECT g, n, const, txt FROM t WHERE g IS NOT NULL AND n IS NOT NULL"
        " AND const IS NOT NULL AND txt IS NOT NULL ORDER BY g, n"
    ).fetchall()
    got = sorted(zip(whole["g"], whole["n"], whole["const"], whole["txt"], strict=True))
    assert got == sorted(want)

    subset = ds.dropna(subset=["txt"]).to_pydict()
    assert None not in subset["txt"]
    assert len(subset["txt"]) == 5, "only the null text row goes; the null n row stays"

    frame = pandas.DataFrame(ROWS)
    assert len(whole["g"]) == len(frame.dropna())
    assert len(subset["g"]) == len(frame.dropna(subset=["txt"]))


def test_drop_empty_removes_the_empty_string_as_well_as_the_null(ds):
    """The difference from ``dropna``, and the reason the verb exists separately."""
    kept = ds.drop_empty("txt").to_pydict()["txt"]
    assert kept == ["hello world", "hi", "a much longer piece of text here", "mid"]
    assert "" not in kept and None not in kept
    assert None not in ds.dropna(subset=["txt"]).to_pydict()["txt"]
    assert "" in ds.dropna(subset=["txt"]).to_pydict()["txt"], (
        "dropna keeps the empty string, which is what makes drop_empty a different verb"
    )


def test_drop_constant_columns_removes_exactly_the_single_valued_columns(ds, duck):
    """A column with one distinct value goes; a column with a null and a value stays."""
    got = ds.drop_constant_columns().to_pydict()
    assert set(got) == {"g", "n", "txt"}
    distinct = {
        name: duck.execute(f"SELECT count(DISTINCT {name}) FROM t").fetchone()[0]
        for name in ("g", "n", "const", "txt")
    }
    for name, count in distinct.items():
        assert (name in got) == (count > 1), f"{name} has {count} distinct values"


def test_class_balance_matches_a_grouped_fraction(ds, duck):
    """The fraction each label holds, which must sum to one."""
    got = ds.class_balance("g").to_pydict()
    want = {
        row[0]: row[1]
        for row in duck.execute(
            "SELECT g, count(*)::DOUBLE / (SELECT count(*) FROM t) FROM t GROUP BY g"
        ).fetchall()
    }
    assert dict(zip(got["g"], got["fraction"], strict=True)) == pytest.approx(want)
    assert sum(got["fraction"]) == pytest.approx(1.0)


def test_balance_classes_equalizes_the_class_counts(ds):
    """Every class must come out at the size of the smallest, and rows must be real rows."""
    balanced = ds.balance_classes("g", order_by="n").to_pydict()
    counts: dict[str, int] = {}
    for label in balanced["g"]:
        counts[label] = counts.get(label, 0) + 1
    assert len(set(counts.values())) == 1, f"classes are not equal: {counts}"
    assert set(counts) == set(ROWS["g"]), "no class may be dropped entirely"
    assert min(counts.values()) == 1, "the smallest class in the fixture has one row"
    for label, value in zip(balanced["g"], balanced["const"], strict=True):
        assert value == 7, f"balancing invented a row for {label}"


def test_get_dummies_matches_pandas(ds):
    """One column per category, zero or one per row, and the prefix option."""
    pandas = pytest.importorskip("pandas")
    got = ds.get_dummies("g").to_pydict()
    want = pandas.get_dummies(pandas.DataFrame(ROWS)["g"], prefix="g")
    for category in ("a", "b", "x"):
        name = f"g_{category}"
        assert name in got, f"{name} missing from {sorted(got)}"
        assert [bool(v) for v in got[name]] == want[name].tolist()
    per_row = [sum(bool(got[f"g_{c}"][i]) for c in ("a", "b", "x")) for i in range(len(ROWS["g"]))]
    assert per_row == [1] * len(ROWS["g"]), "exactly one indicator is set per row"

    prefixed = ds.get_dummies("g", prefix="cls").to_pydict()
    assert {"cls_a", "cls_b", "cls_x"} <= set(prefixed)


def test_filter_by_length_keeps_the_rows_inside_the_character_bounds(ds, duck):
    """Both bounds, against an explicit ``length()`` predicate."""
    got = ds.filter_by_length("txt", min_chars=3, max_chars=20).to_pydict()["txt"]
    want = [
        row[0]
        for row in duck.execute(
            "SELECT txt FROM t WHERE txt IS NOT NULL AND length(txt) >= 3"
            " AND length(txt) <= 20 ORDER BY txt"
        ).fetchall()
    ]
    assert sorted(got) == sorted(want)
    assert "hi" not in got, "two characters is below the minimum"
    assert None not in got


def test_filter_by_token_budget_agrees_with_the_estimate_it_thresholds(ds):
    """The row filter must keep exactly the rows ``fits_token_budget`` marks."""
    budget = 4
    kept = ds.filter_by_token_budget("txt", budget).to_pydict()["txt"]
    marked = ds.select(
        txt=bt.col("txt"), fits=bt.col("txt").str.fits_token_budget(budget)
    ).to_pydict()
    expected = [t for t, fits in zip(marked["txt"], marked["fits"], strict=True) if fits]
    assert sorted(kept) == sorted(expected)


def test_train_val_test_split_partitions_the_rows_without_overlap(ds):
    """Three disjoint parts whose union is the input, and stable under the seed."""
    train, val, test = ds.train_val_test_split("g", val_size=0.34, test_size=0.34, seed=7)
    parts = [part.to_pydict()["const"] for part in (train, val, test)]
    assert sum(len(p) for p in parts) == len(ROWS["g"]), "rows were lost or duplicated"

    keyed = [
        set(zip(part.to_pydict()["g"], part.to_pydict()["n"], strict=True))
        for part in (train, val, test)
    ]
    assert keyed[0] & keyed[1] == set()
    assert keyed[0] & keyed[2] == set()
    assert keyed[1] & keyed[2] == set()

    again = ds.train_val_test_split("g", val_size=0.34, test_size=0.34, seed=7)
    assert [p.to_pydict() for p in again] == [p.to_pydict() for p in (train, val, test)], (
        "the same seed must produce the same split"
    )


def test_the_split_is_stratified_rather_than_grouped():
    """``by`` names the label whose proportions are preserved, not a key to keep together.

    Worth pinning because the parameter reads like the ``groups`` argument of a
    leakage-free group split, and it is the opposite: a stratified split puts *some* rows
    of every class in every part on purpose. A pipeline that needs all of one user's rows
    in one part must not reach for this method.
    """
    labelled = bt.from_pydict({"y": ["a"] * 60 + ["b"] * 40, "v": list(range(100))})
    parts = labelled.train_val_test_split("y", val_size=0.2, test_size=0.2, seed=3)
    total = 0
    for part in parts:
        labels = part.to_pydict()["y"]
        total += len(labels)
        share = labels.count("a") / len(labels)
        assert share == pytest.approx(0.6, abs=0.12), (
            f"a part holds {share:.2f} of class a, not the 0.60 of the whole"
        )
        assert set(labels) == {"a", "b"}, "a stratified part carries every class"
    assert total == 100, "rows were lost or duplicated"

    rows = [set(zip(p.to_pydict()["y"], p.to_pydict()["v"], strict=True)) for p in parts]
    assert rows[0] & rows[1] == set()
    assert rows[0] & rows[2] == set()
    assert rows[1] & rows[2] == set()


def test_the_split_refuses_fractions_that_leave_no_training_set():
    """``val_size + test_size`` must stay below one, and says so rather than emptying train."""
    from batcher import PlanError

    labelled = bt.from_pydict({"y": ["a", "b"] * 10, "v": list(range(20))})
    with pytest.raises(PlanError):
        labelled.train_val_test_split("y", val_size=0.6, test_size=0.6)


def test_to_csv_round_trips_through_the_reader(ds, tmp_path):
    """The CSV shorthand, checked by reading the file back rather than by eyeballing it."""
    target = tmp_path / "out.csv"
    ds.to_csv(str(target))
    written = sorted(tmp_path.glob("**/*.csv"))
    assert written, f"nothing was written under {tmp_path}"
    back = bt.read.csv(str(target)).to_pydict()
    assert sorted(back) == sorted(ROWS)
    assert len(back["g"]) == len(ROWS["g"])
    assert sorted(v for v in back["n"] if v is not None) == [1, 2, 3, 4, 6]


def test_glimpse_prints_every_column_and_returns_nothing(ds, capsys):
    """A display helper, so what is checked is that it names each column and does not raise."""
    assert ds.glimpse() is None
    printed = capsys.readouterr().out
    for name in ROWS:
        assert name in printed, f"glimpse did not mention {name}"


def test_glimpse_bounds_what_it_prints(capsys):
    """``max_items_per_column`` must actually bound the output, or it is decoration."""
    wide = bt.from_pydict({"v": [f"value-{i}" for i in range(500)]})
    wide.glimpse(max_items_per_column=3)
    short = capsys.readouterr().out
    wide.glimpse(max_items_per_column=50)
    long = capsys.readouterr().out
    assert len(short) < len(long), "the bound had no effect on the output"
    assert "value-499" not in short


def test_the_verbs_survive_an_empty_frame():
    """Each verb over zero rows must answer, not raise."""
    empty = bt.from_pydict({"g": [], "n": [], "txt": []})
    assert empty.dropna().to_pydict() == {"g": [], "n": [], "txt": []}
    assert empty.drop_empty("txt").to_pydict()["txt"] == []
    assert empty.filter_by_length("txt", min_chars=1).to_pydict()["txt"] == []
    assert empty.class_balance("g").to_pydict()["g"] == []
