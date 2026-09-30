"""The Ray Data parameters on Batcher's preprocessors, checked without Ray.

`output_columns=` (and `LabelEncoder`'s `output_column=`), `OrdinalEncoder(encode_lists=)`,
`Tokenizer`'s default split, and `KBinsDiscretizer`'s per-column `n_bins`, `right=` and
`duplicates=`. The live comparison against Ray Data itself is
`tests/differential/test_diff_ml_ray_preprocessors.py`; this file pins what holds whether or
not Ray is installed: the defaults are unchanged, the inputs survive beside the outputs,
nulls/NaN/empty inputs behave, and a saved preprocessor keeps the new parameters.
"""

from __future__ import annotations

import math

import pyarrow as pa
import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.preprocessors import (
    KBinsDiscretizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OrdinalEncoder,
    Preprocessor,
    SimpleImputer,
    StandardScaler,
    Tokenizer,
)

pytestmark = pytest.mark.unit

NUM = {"a": [1.0, None, 3.0, 4.0], "b": [2.0, 0.0, -2.0, 6.0]}


def _num() -> bt.Dataset:
    return bt.from_pydict(NUM)


SCALERS = [
    lambda outs: StandardScaler(["a", "b"], output_columns=outs),
    lambda outs: MinMaxScaler(["a", "b"], output_columns=outs),
    lambda outs: MaxAbsScaler(["a", "b"], output_columns=outs),
    lambda outs: Normalizer(["a", "b"], output_columns=outs),
    lambda outs: SimpleImputer(["a", "b"], output_columns=outs),
]


@pytest.mark.parametrize("make", SCALERS)
def test_output_columns_writes_beside_the_inputs_and_matches_in_place(make) -> None:
    in_place = make(None).fit_transform(_num()).to_pydict()
    beside = make(["a_out", "b_out"]).fit_transform(_num()).to_pydict()

    assert list(beside) == ["a", "b", "a_out", "b_out"]
    assert beside["a"] == NUM["a"]
    assert beside["b"] == NUM["b"]
    assert beside["a_out"] == in_place["a"]
    assert beside["b_out"] == in_place["b"]


@pytest.mark.parametrize("make", SCALERS)
def test_output_columns_none_is_the_default_in_place_rewrite(make) -> None:
    out = make(None).fit_transform(_num()).to_pydict()
    assert list(out) == ["a", "b"]
    assert out != NUM  # the inputs were rewritten, not copied


@pytest.mark.parametrize(
    "outs, message",
    [
        (["only_one"], "one name per column"),
        (["x", "x"], "repeats a name"),
        ([1, "y"], "must be strings"),
    ],
)
def test_output_columns_is_validated_at_construction(outs, message) -> None:
    with pytest.raises(PlanError, match=message):
        StandardScaler(["a", "b"], output_columns=outs)


def test_output_column_nulls_and_nan_pass_through_the_scaler() -> None:
    ds = bt.from_arrow(pa.table({"x": pa.array([0.0, None, float("nan"), 10.0])}))
    out = MinMaxScaler("x", output_columns="y").fit_transform(ds).to_pydict()
    assert out["x"][1] is None and math.isnan(out["x"][2])
    assert out["y"][1] is None


def test_output_columns_on_an_empty_input_keeps_the_schema() -> None:
    ds = bt.from_arrow(pa.table({"x": pa.array([], pa.float64())}))
    table = StandardScaler("x", output_columns="z").fit_transform(ds).collect()
    assert table.column_names == ["x", "z"]
    assert table.num_rows == 0


def test_label_encoder_output_column_keeps_the_labels() -> None:
    ds = bt.from_pydict({"y": ["dog", None, "cat", "dog"]})
    out = LabelEncoder("y", output_column="code").fit_transform(ds).to_pydict()
    assert out == {"y": ["dog", None, "cat", "dog"], "code": [1, -1, 0, 1]}
    with pytest.raises(PlanError, match="single column name"):
        LabelEncoder("y", output_column=["a"])  # type: ignore[arg-type]


def test_ordinal_encoder_encodes_list_elements() -> None:
    ds = bt.from_arrow(
        pa.table({"t": pa.array([["b", "a"], ["c"], [], None, ["a", None]], pa.list_(pa.string()))})
    )
    encoder = OrdinalEncoder("t").fit(ds)
    assert encoder.categories_ == {"t": ["a", "b", "c"]}
    out = encoder.transform(ds).to_pydict()["t"]
    # A null list stays null; a null element is unknown, like a null scalar.
    assert out == [[1, 0], [2], [], None, [0, -1]]


def test_ordinal_encoder_list_unknown_value_and_output_columns() -> None:
    train = bt.from_pydict({"t": [["x", "y"]], "s": ["p"]})
    test = bt.from_pydict({"t": [["y", "z"]], "s": ["q"]})
    enc = OrdinalEncoder(["t", "s"], unknown_value=9, output_columns=["tc", "sc"]).fit(train)
    assert enc.transform(test).to_pydict() == {
        "t": [["y", "z"]],
        "s": ["q"],
        "tc": [[1, 9]],
        "sc": [9],
    }


def test_ordinal_encoder_encode_lists_false_refuses_a_list_column() -> None:
    ds = bt.from_pydict({"t": [["a"], ["b"]], "s": ["a", "b"]})
    with pytest.raises(PlanError, match="list literals"):
        OrdinalEncoder("t", encode_lists=False).fit(ds)
    # A scalar column ignores the flag.
    assert OrdinalEncoder("s", encode_lists=False).fit_transform(ds).to_pydict()["s"] == [0, 1]


def test_tokenizer_default_splits_on_single_spaces_like_str_split() -> None:
    texts = ["a  b", "", " x", "one", None]
    out = Tokenizer("t").fit_transform(bt.from_pydict({"t": texts})).to_pydict()["t"]
    assert out == [None if s is None else s.split(" ") for s in texts]


def test_tokenizer_default_refuses_batched_only_options() -> None:
    with pytest.raises(PlanError, match="batched tokenizer"):
        Tokenizer("t", max_length=4)


def test_tokenizer_default_saves_but_a_callable_still_refuses(tmp_path) -> None:
    ds = bt.from_pydict({"t": ["a b", "c"]})
    target = str(tmp_path / "tok.json")
    Tokenizer("t", output_column="toks").fit(ds).save(target)
    loaded = Preprocessor.load(target)
    assert loaded.transform(ds).to_pydict()["toks"] == [["a", "b"], ["c"]]
    # A callable has no JSON form. Now that `tokenizer` is optional, an unrecorded one would
    # reload as the default split; recording it makes the save refuse instead.
    with pytest.raises(PlanError, match="Tokenizer cannot be saved"):
        Tokenizer("t", str.upper).fit(ds).save(target)


def test_kbins_default_is_unchanged_left_closed_and_clamped() -> None:
    ds = bt.from_pydict({"v": [0.0, 5.0, 10.0]})
    kb = KBinsDiscretizer("v", n_bins=2, strategy="uniform").fit(ds)
    probe = bt.from_pydict({"v": [-1.0, 5.0, 11.0, None]})
    assert kb.transform(probe).to_pydict()["v"] == [0, 1, 1, None]


def test_kbins_right_puts_an_edge_value_in_the_lower_bin() -> None:
    ds = bt.from_pydict({"v": [0.0, 2.5, 5.0, 7.5, 10.0]})
    left = KBinsDiscretizer("v", n_bins=4, strategy="uniform").fit_transform(ds)
    right = KBinsDiscretizer("v", n_bins=4, strategy="uniform", right=True).fit_transform(ds)
    assert left.to_pydict()["v"] == [0, 1, 2, 3, 3]
    assert right.to_pydict()["v"] == [0, 0, 1, 2, 3]


def test_kbins_per_column_bin_counts() -> None:
    ds = bt.from_pydict({"a": [0.0, 10.0], "b": [0.0, 10.0]})
    kb = KBinsDiscretizer(["a", "b"], n_bins={"a": 2, "b": 5}, strategy="uniform").fit(ds)
    assert kb.edges_ == {"a": [5.0], "b": [2.0, 4.0, 6.0, 8.0]}
    with pytest.raises(PlanError, match="no entry"):
        KBinsDiscretizer(["a", "b"], n_bins={"a": 2})
    with pytest.raises(PlanError, match=">= 2"):
        KBinsDiscretizer(["a"], n_bins={"a": 1})


def test_kbins_duplicates_policy_on_a_constant_column() -> None:
    ds = bt.from_pydict({"v": [3.0, 3.0, 3.0]})
    keep = KBinsDiscretizer("v", n_bins=3, strategy="uniform").fit(ds)
    assert keep.edges_ == {"v": [3.0, 3.0]}
    drop = KBinsDiscretizer("v", n_bins=3, strategy="uniform", duplicates="drop").fit(ds)
    assert drop.edges_ == {"v": [3.0]}
    assert drop.transform(ds).to_pydict()["v"] == [1, 1, 1]
    with pytest.raises(PlanError, match="repeated bin edges"):
        KBinsDiscretizer("v", n_bins=3, strategy="uniform", duplicates="raise").fit(ds)
    with pytest.raises(PlanError, match="duplicates must be"):
        KBinsDiscretizer("v", duplicates="nope")


def test_kbins_output_columns() -> None:
    ds = bt.from_pydict({"v": [0.0, 10.0]})
    out = KBinsDiscretizer("v", n_bins=2, strategy="uniform", output_columns="bin")
    assert out.fit_transform(ds).to_pydict() == {"v": [0.0, 10.0], "bin": [0, 1]}


@pytest.mark.parametrize(
    "pre",
    [
        StandardScaler(["a", "b"], output_columns=["x", "y"]),
        SimpleImputer("a", output_columns="a2"),
        KBinsDiscretizer("b", n_bins={"b": 3}, strategy="uniform", right=True, output_columns="c"),
        OrdinalEncoder("b", encode_lists=False, output_columns="c"),
    ],
    ids=["scaler", "imputer", "kbins", "ordinal"],
)
def test_the_new_parameters_survive_save_and_load(pre, tmp_path) -> None:
    fitted = pre.fit(_num())
    target = str(tmp_path / "pre.json")
    fitted.save(target)
    loaded = Preprocessor.load(target)
    assert loaded.get_params() == fitted.get_params()
    assert loaded.transform(_num()).to_pydict() == fitted.transform(_num()).to_pydict()
