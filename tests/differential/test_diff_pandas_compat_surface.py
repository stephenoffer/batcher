"""The pandas-compatibility aliases and the last few expression methods, against pandas.

Batcher exports a pandas-shaped spelling for several verbs so a ported script keeps
reading the way it did: ``ds.groupby`` beside ``ds.group_by``, ``ds.merge`` beside
``ds.join``, ``ds.sort_values``, ``ds.assign``, ``ds.abs``, ``ds.round``, ``ds.notna``,
``ds.size``, and ``.str.startswith`` / ``.endswith`` / ``.match``. The execution-coverage
sweep found none of them called by a test, which is the worst place for that: an alias
whose semantics have quietly drifted from the name it borrows is a silent wrong answer in
exactly the script that trusted the name.

So pandas is the oracle, not Batcher's own primary spelling. Each alias is checked against
what pandas does *and* against the primary spelling it delegates to, because those are two
different failures: the first catches a borrowed name that means something else, the second
catches an alias that has stopped delegating.

The remaining ``Expr`` methods (``arccos``, ``arcsinh``, ``chr``, ``clip_min``,
``softmax``, ``to_base``, ``to_ir``) and the two selector renamers (``name.keep``,
``name.to_lowercase``) are here for the same reason -- last of the public expression
surface with no test -- and are checked against Python's ``math`` and ``format`` rather
than pandas, which has no equivalent.
"""

from __future__ import annotations

import math

import pytest

import batcher as bt

pytestmark = pytest.mark.differential

pandas = pytest.importorskip("pandas")

ROWS = {
    "g": ["a", "a", "b"],
    "x": [-1.5, 2.25, 3.0],
    "s": ["Hello", "world", None],
    "n": [1, None, 3],
}


@pytest.fixture
def ds():
    return bt.from_pydict(ROWS)


@pytest.fixture
def frame():
    return pandas.DataFrame(ROWS)


def test_size_is_rows_times_columns_as_in_pandas(ds, frame):
    """``ds.size`` is the cell count, not the row count -- the one that surprises people."""
    assert ds.size == frame.size
    assert ds.size == len(ROWS["g"]) * len(ROWS)
    assert ds.size != ds.count(), "size is not a row count, and conflating them is the bug"


def test_abs_matches_pandas(ds, frame):
    """Frame-wide absolute value, over the numeric column only."""
    got = ds.select(bt.col("x")).abs().to_pydict()["x"]
    assert got == pytest.approx(frame[["x"]].abs()["x"].tolist())


def test_round_borrows_the_pandas_name_and_keeps_sql_tie_breaking(ds, frame):
    """The alias rounds ties **away from zero**; pandas rounds them to even.

    ``round(0.5)`` is 1.0 here and 0.0 in pandas, because ``Expr.round`` follows SQL and the
    DataFrame spelling delegates to it rather than to NumPy. That is a deliberate choice --
    one query must not disagree with itself depending on whether it was written in SQL or in
    the DataFrame API -- and it is pinned here because the borrowed name invites the other
    expectation. Away from the ties the two agree exactly, which is also asserted.
    """
    ties = [0.5, 1.5, 2.5, 2.25, -2.25]
    ours = bt.from_pydict({"x": ties}).select(bt.col("x")).round().to_pydict()["x"]
    theirs = pandas.DataFrame({"x": ties}).round()["x"].tolist()
    assert ours == pytest.approx([1.0, 2.0, 3.0, 2.0, -2.0]), "away from zero on every tie"
    assert theirs == pytest.approx([0.0, 2.0, 2.0, 2.0, -2.0]), (
        "the departure is only real while pandas still rounds half to even"
    )

    ours_1dp = bt.from_pydict({"x": ties}).select(bt.col("x")).round(1).to_pydict()["x"]
    assert ours_1dp == pytest.approx([0.5, 1.5, 2.5, 2.3, -2.3])

    unambiguous = [1.234, -9.876, 0.05001]
    assert bt.from_pydict({"x": unambiguous}).select(bt.col("x")).round(1).to_pydict()[
        "x"
    ] == pytest.approx(pandas.DataFrame({"x": unambiguous}).round(1)["x"].tolist()), (
        "off the ties, the alias must agree with pandas exactly"
    )


def test_round_leaves_a_non_numeric_column_alone(ds):
    """The numeric selector picks the columns, so a string column must survive untouched."""
    got = ds.round(1).to_pydict()
    assert got["s"] == ROWS["s"]
    assert got["g"] == ROWS["g"]
    assert got["x"] == pytest.approx([-1.5, 2.3, 3.0])


def test_notna_matches_pandas_and_is_the_complement_of_isna(ds, frame):
    """A frame of booleans, and the identity that makes the pair coherent."""
    got = ds.select(bt.col("n")).notna().to_pydict()["n"]
    assert got == frame[["n"]].notna()["n"].tolist()
    missing = ds.select(bt.col("n")).isna().to_pydict()["n"]
    assert [not v for v in got] == missing


def test_sort_values_matches_pandas_and_really_orders_the_rows(ds, frame):
    """Compared in order, element by element, because a sort is the one thing
    an order-independent comparison cannot see.
    """
    got = ds.sort_values("x").to_pydict()["x"]
    assert got == frame.sort_values("x")["x"].tolist()
    assert got == sorted(ROWS["x"]), "and it is genuinely ascending"

    # `ascending=` is the pandas keyword, not Batcher's own `descending=`; the alias takes
    # the borrowed spelling, which is the point of having it.
    descending = ds.sort_values("x", ascending=False).to_pydict()["x"]
    assert descending == frame.sort_values("x", ascending=False)["x"].tolist()
    assert descending == sorted(ROWS["x"], reverse=True)
    assert descending == list(reversed(got))


def test_sort_values_delegates_to_sort(ds):
    """The alias and the primary spelling must produce the same rows in the same order."""
    assert ds.sort_values("x").to_pydict() == ds.sort("x").to_pydict()
    assert (
        ds.sort_values("x", ascending=False).to_pydict()
        == ds.sort("x", descending=True).to_pydict()
    )


def test_assign_matches_pandas_and_delegates_to_with_columns(ds, frame):
    """Adds a column, keeps the rest, and is exactly ``with_columns``."""
    got = ds.assign(y=bt.col("x") * 2).to_pydict()
    want = frame.assign(y=frame["x"] * 2)
    assert got["y"] == pytest.approx(want["y"].tolist())
    assert sorted(got) == sorted(want.columns), "assign must not drop a column"
    assert got == ds.with_columns(y=bt.col("x") * 2).to_pydict()


def test_groupby_matches_pandas_and_delegates_to_group_by(ds, frame):
    """The pandas spelling of the aggregate, keyed the same way."""
    got = ds.groupby("g").agg(total=bt.col("x").sum()).to_pydict()
    want = frame.groupby("g")["x"].sum()
    assert dict(zip(got["g"], got["total"], strict=True)) == pytest.approx(want.to_dict())
    assert got == ds.group_by("g").agg(total=bt.col("x").sum()).to_pydict()


def test_merge_matches_pandas_and_delegates_to_join(ds, frame):
    """An inner join on a shared key, and the ``how`` and ``suffix`` arguments."""
    right_rows = {"g": ["a", "b", "z"], "w": [1, 2, 9]}
    right = bt.from_pydict(right_rows)
    right_frame = pandas.DataFrame(right_rows)

    got = ds.merge(right, on="g").to_pydict()
    want = frame.merge(right_frame, on="g")
    assert sorted(got["w"]) == sorted(want["w"].tolist())
    assert got == ds.join(right, on="g").to_pydict()

    left_outer = ds.merge(right, on="g", how="left").to_pydict()
    assert len(left_outer["g"]) == len(frame.merge(right_frame, on="g", how="left"))


def test_merge_suffixes_a_clashing_column_the_way_it_says_it_does(ds):
    """The ``suffix`` argument, which is where pandas and Batcher spell things differently.

    pandas takes a ``suffixes`` pair; Batcher takes one ``suffix`` for the right side. That
    is a deliberate narrowing rather than a missing feature, and pinning it here means the
    difference is visible to anyone porting a ``merge`` that relied on the pair.
    """
    right = bt.from_pydict({"g": ["a", "b"], "x": [10.0, 20.0]})
    got = ds.merge(right, on="g", suffix="_r").to_pydict()
    assert "x" in got and "x_r" in got, f"columns were {sorted(got)}"
    assert got["x_r"] == pytest.approx([10.0, 10.0, 20.0])


def test_groupby_std_and_var_match_pandas(ds, frame):
    """Sample statistics, and the null a single-row group must produce.

    pandas' ``std``/``var`` default to one degree of freedom, so a group of one has no
    sample deviation. Group ``b`` here holds one row, which is what makes this test able to
    tell a sample statistic from a population one.
    """
    got = ds.group_by("g").agg(deviation=bt.col("x").std(), variance=bt.col("x").var()).to_pydict()
    want_std = frame.groupby("g")["x"].std().to_dict()
    want_var = frame.groupby("g")["x"].var().to_dict()
    by_group = dict(zip(got["g"], got["deviation"], strict=True))
    var_by_group = dict(zip(got["g"], got["variance"], strict=True))
    assert by_group["a"] == pytest.approx(want_std["a"])
    assert var_by_group["a"] == pytest.approx(want_var["a"])
    assert by_group["b"] is None, "one row has no sample deviation"
    assert math.isnan(want_std["b"]), "and pandas agrees, spelling it NaN"


#: ``(label, batcher expression, python reference over ROWS["x"])`` for the numeric methods.
NUMERIC = [
    ("arccos", lambda: (bt.col("x") / 4).arccos(), lambda v: math.acos(v / 4)),
    ("arcsinh", lambda: bt.col("x").arcsinh(), math.asinh),
    ("clip_min", lambda: bt.col("x").clip_min(0.0), lambda v: max(v, 0.0)),
]


@pytest.mark.parametrize(("label", "expr", "reference"), NUMERIC)
def test_numeric_expression_matches_the_python_function_it_names(ds, label, expr, reference):
    """Each against the ``math`` function of the same name, to full precision."""
    got = ds.select(v=expr()).to_pydict()["v"]
    want = [reference(v) for v in ROWS["x"]]
    assert got == pytest.approx(want, rel=1e-15), f"{label}: {got} vs {want}"


def test_clip_min_is_the_lower_half_of_clip(ds):
    """``clip_min`` and ``clip_max`` composed must equal ``clip``, or one of them is wrong."""
    got = ds.select(
        both=bt.col("x").clip(0.0, 2.0),
        chained=bt.col("x").clip_min(0.0).clip_max(2.0),
    ).to_pydict()
    assert got["both"] == pytest.approx(got["chained"])


def test_softmax_is_a_distribution_over_the_column(ds):
    """Sums to one, preserves order, and matches the exponential form.

    A window function rather than a row function, which is the thing to get wrong: an
    implementation that normalized per row would return 1.0 everywhere and pass any test
    that only checked the range.
    """
    got = ds.select(v=bt.col("x").softmax()).to_pydict()["v"]
    assert sum(got) == pytest.approx(1.0, abs=1e-12)
    assert all(0.0 < v < 1.0 for v in got), f"{got}"
    shifted = max(ROWS["x"])
    weights = [math.exp(v - shifted) for v in ROWS["x"]]
    total = sum(weights)
    assert got == pytest.approx([w / total for w in weights], rel=1e-12)
    assert got == sorted(got), "the fixture is ascending, so the distribution must be too"


def test_chr_and_to_base_match_pythons_own_conversions(ds):
    """``chr`` of a code point and ``to_base`` of an integer, nulls preserved."""
    got = ds.select(
        letter=(bt.col("n") + 64).chr(),
        binary=(bt.col("n") * 10).to_base(2),
        hexed=(bt.col("n") * 10).to_base(16),
    ).to_pydict()
    for i, value in enumerate(ROWS["n"]):
        if value is None:
            assert got["letter"][i] is None
            assert got["binary"][i] is None
            assert got["hexed"][i] is None
            continue
        assert got["letter"][i] == chr(value + 64)
        assert got["binary"][i] == format(value * 10, "b")
        # Uppercase digits, as SQL's `hex` renders them; compared case-insensitively so
        # this asserts the *value* and not the convention.
        assert got["hexed"][i].lower() == format(value * 10, "x")
        assert got["hexed"][i] == got["hexed"][i].upper(), "the engine renders hex uppercase"


def test_to_ir_is_the_wire_shape_the_engine_receives(ds):
    """``Expr.to_ir`` is the boundary contract, so its tag and fields are asserted.

    Not a display helper: this dict is what crosses to Rust, and its ``e`` tag has to match
    a `bc_expr::Expr` serde variant. A round trip through the engine is the strongest cheap
    check that it still does.
    """
    ir = bt.col("x").to_ir()
    assert isinstance(ir, dict)
    assert ir == {"e": "col", "name": "x"}

    arithmetic = (bt.col("x") + 1).to_ir()
    assert arithmetic["e"] != "col", "an expression tree is not a bare column reference"
    assert "x" in str(arithmetic), "and it still names the column it reads"

    # The round trip: the same expression, executed, proves the IR the engine got was valid.
    assert ds.select(v=bt.col("x") + 1).to_pydict()["v"] == pytest.approx(
        [v + 1 for v in ROWS["x"]]
    )


#: ``(method, argument, pandas method)`` for the three ``.str`` aliases.
STRING_ALIASES = [("startswith", "Hel", "startswith"), ("endswith", "rld", "endswith")]


@pytest.mark.parametrize(("ours", "argument", "theirs"), STRING_ALIASES)
def test_string_alias_matches_pandas(ds, frame, ours, argument, theirs):
    """The pandas spelling of the prefix and suffix tests, nulls included."""
    got = ds.select(v=getattr(bt.col("s").str, ours)(argument)).to_pydict()["v"]
    want = getattr(frame["s"].str, theirs)(argument).tolist()
    assert got == [None if isinstance(v, float) and math.isnan(v) else v for v in want], got


def test_str_match_is_anchored_like_pandas(ds, frame):
    """``match`` anchors at the start, which is what separates it from an unanchored search.

    It did not: it was a second spelling of ``regexp_matches`` while its docstring claimed
    pandas parity, so ``match("ello")`` was true for ``"Hello"`` where pandas says false --
    a ported filter quietly keeping rows pandas drops.
    """
    got = ds.select(v=bt.col("s").str.match("^H")).to_pydict()["v"]
    want = frame["s"].str.match("^H").tolist()
    assert got == [None if isinstance(v, float) and math.isnan(v) else v for v in want]

    anchored = ds.select(
        matched=bt.col("s").str.match("ello"), searched=bt.col("s").str.regexp_matches("ello")
    ).to_pydict()
    assert anchored["matched"][0] is False, "match is anchored at the start"
    assert anchored["searched"][0] is True, "the unanchored search is not, which is the point"
    assert anchored["matched"] == [
        None if isinstance(v, float) and math.isnan(v) else v
        for v in frame["s"].str.match("ello").tolist()
    ]


def test_str_match_anchors_every_arm_of_an_alternation():
    """Wrapping rather than prefixing, so ``match("a|b")`` does not become ``^a|b``."""
    docs = ["abc", "xbc", "bca"]
    got = bt.from_pydict({"s": docs}).select(v=bt.col("s").str.match("a|x")).to_pydict()["v"]
    assert got == pandas.Series(docs).str.match("a|x").tolist()
    assert got == [True, True, False], f"{got}"


def test_the_string_aliases_delegate_to_the_primary_spellings(ds):
    """Each alias must equal the method it borrows the pandas name for."""
    got = ds.select(
        alias_start=bt.col("s").str.startswith("Hel"),
        primary_start=bt.col("s").str.starts_with("Hel"),
        alias_end=bt.col("s").str.endswith("rld"),
        primary_end=bt.col("s").str.ends_with("rld"),
    ).to_pydict()
    assert got["alias_start"] == got["primary_start"]
    assert got["alias_end"] == got["primary_end"]


def test_selector_name_keep_leaves_the_column_names_alone(ds):
    """``name.keep`` cancels an earlier rename, which is the only reason it exists."""
    kept = list(ds.select(bt.numeric().name.keep()).to_pydict())
    assert kept == ["x", "n"], f"{kept}"
    suffixed = list(ds.select(bt.numeric().name.suffix("_z")).to_pydict())
    assert suffixed == ["x_z", "n_z"], "so the rename really was doing something"
    assert list(ds.select(bt.numeric().name.suffix("_z").name.keep()).to_pydict()) == kept


def test_selector_name_to_lowercase_folds_every_matched_name():
    """A whole-frame rename, which is the usual first step after reading a foreign header."""
    ds = bt.from_pydict({"AB": [1], "Cd": [2], "ee": [3]})
    assert list(ds.select(bt.all().name.to_lowercase()).to_pydict()) == ["ab", "cd", "ee"]
    assert ds.select(bt.all().name.to_lowercase()).to_pydict()["ab"] == [1], (
        "renaming must not shuffle the values between columns"
    )
