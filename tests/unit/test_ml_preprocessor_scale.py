"""Scale and safety regression tests for the `Preprocessor` family.

Each test here pins a defect that made a preprocessor unusable on high-cardinality,
wide, or large feature data: an unbounded category set pulled to the driver, an
unbounded schema expansion, a chain that rescanned its source once per step, a
tokenizer that ran per row in Python, and an unbounded polynomial term count.
"""

from __future__ import annotations

from typing import Any

import pytest

import batcher as bt
from batcher._internal.errors import PlanError
from batcher.ml.preprocessors import (
    Chain,
    KBinsDiscretizer,
    LabelEncoder,
    MinMaxScaler,
    MultiHotEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    SimpleImputer,
    StandardScaler,
    TargetEncoder,
    Tokenizer,
)

pytestmark = pytest.mark.unit


def _wide_categoricals(n: int) -> bt.Dataset:
    return bt.from_pydict({"c": [f"cat{i}" for i in range(n)]})


# --- 1 + 2: cardinality guards on the CASE-chain / indicator encoders -------------


def test_ordinal_encoder_rejects_high_cardinality() -> None:
    """A 5,000-category fit must fail fast, not build a 5,000-deep CASE chain."""
    with pytest.raises(PlanError, match="max_categories"):
        OrdinalEncoder(["c"]).fit(_wide_categoricals(5000))


def test_ordinal_encoder_max_categories_is_configurable() -> None:
    """The guard is tunable in both directions, and the error names the limit."""
    with pytest.raises(PlanError, match="max_categories=10"):
        OrdinalEncoder(["c"], max_categories=10).fit(_wide_categoricals(50))
    enc = OrdinalEncoder(["c"], max_categories=100).fit(_wide_categoricals(50))
    assert len(enc.categories_["c"]) == 50


def test_label_encoder_rejects_high_cardinality() -> None:
    with pytest.raises(PlanError, match="max_categories"):
        LabelEncoder("c", max_categories=10).fit(_wide_categoricals(50))


def test_one_hot_encoder_rejects_schema_explosion() -> None:
    """One column per category is a schema explosion; it must be bounded."""
    with pytest.raises(PlanError, match="max_categories"):
        OneHotEncoder(["c"]).fit(_wide_categoricals(5000))


def test_multi_hot_encoder_rejects_high_cardinality() -> None:
    ds = bt.from_pydict({"tags": [[f"t{i}"] for i in range(50)]})
    with pytest.raises(PlanError, match="max_categories"):
        MultiHotEncoder("tags", max_categories=10).fit(ds)


def test_target_encoder_rejects_high_cardinality() -> None:
    n = 3000
    ds = bt.from_pydict({"c": [f"cat{i}" for i in range(n)], "y": [1.0] * n})
    with pytest.raises(PlanError, match="max_categories"):
        TargetEncoder(["c"], "y").fit(ds)


def test_default_guard_admits_a_reasonable_pipeline() -> None:
    """The default limit is generous enough for an ordinary categorical column."""
    ds = _wide_categoricals(200)
    out = OrdinalEncoder(["c"]).fit_transform(ds).to_pydict()
    assert len(out["c"]) == 200


def test_encoders_keep_their_documented_results() -> None:
    """The guards are additive: the existing encodings are unchanged."""
    ds = bt.from_pydict({"c": ["b", "a", "c", "a"]})
    assert OrdinalEncoder(["c"]).fit_transform(ds).to_pydict() == {"c": [1, 0, 2, 0]}
    assert LabelEncoder("c").fit_transform(ds).to_pydict() == {"c": [1, 0, 2, 0]}
    hot = OneHotEncoder(["c"]).fit_transform(ds).to_pydict()
    assert hot == {"c_a": [0, 1, 0, 1], "c_b": [1, 0, 0, 0], "c_c": [0, 0, 1, 0]}


# --- 3: a chain must not rescan its source once per step -------------------------


def _counting_source(counter: list[int]) -> bt.Dataset:
    base = bt.from_pydict({"a": [1.0, 2.0, None, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})

    def _udf(batch: Any) -> Any:
        counter.append(1)
        return batch

    return base.map_batches(_udf, output_columns=["a", "b"])


def test_chain_fit_scans_the_source_once() -> None:
    """N steps must not mean N full scans of the upstream plan."""
    counter: list[int] = []
    ds = _counting_source(counter)
    Chain(SimpleImputer(["a"]), StandardScaler(["a"]), MinMaxScaler(["a"])).fit(ds)
    assert len(counter) == 1, f"source scanned {len(counter)} times"


def test_chain_fit_transform_scans_the_source_once() -> None:
    counter: list[int] = []
    ds = _counting_source(counter)
    chain = Chain(SimpleImputer(["a"]), StandardScaler(["a"]), MinMaxScaler(["a"]))
    out = chain.fit_transform(ds).to_pydict()
    assert len(counter) == 1, f"source scanned {len(counter)} times"
    assert len(out["a"]) == 4


def test_chain_without_caching_still_works() -> None:
    """`cache=False` keeps the old lazy behavior for a too-large training split."""
    train = bt.from_pydict({"age": [10.0, 20.0, None, 40.0]})
    chain = Chain(SimpleImputer(["age"]), StandardScaler(["age"]), cache=False)
    got = [round(v, 3) for v in chain.fit_transform(train).to_pydict()["age"]]
    assert got == [-1.234, -0.309, 0.0, 1.543]


def test_chain_results_are_identical_with_and_without_cache() -> None:
    train = bt.from_pydict({"age": [10.0, 20.0, None, 40.0]})
    test = bt.from_pydict({"age": [None, 30.0]})
    cached = Chain(SimpleImputer(["age"]), StandardScaler(["age"])).fit(train)
    lazy = Chain(SimpleImputer(["age"]), StandardScaler(["age"]), cache=False).fit(train)
    assert cached.transform(test).to_pydict() == lazy.transform(test).to_pydict()


# --- 4: the tokenizer must use the batched fast path -----------------------------


class _FakeHFTokenizer:
    """A HuggingFace-shaped tokenizer: callable on a *list*, plus a per-string `encode`."""

    def __init__(self) -> None:
        self.batch_calls = 0
        self.row_calls = 0

    def __call__(
        self,
        texts: list[str],
        *,
        truncation: bool = False,
        max_length: int | None = None,
        padding: bool | str = False,
    ) -> dict[str, list[list[int]]]:
        self.batch_calls += 1
        ids = [[len(w) for w in t.split()] for t in texts]
        if truncation and max_length is not None:
            ids = [row[:max_length] for row in ids]
        mask = [[1] * len(row) for row in ids]
        if padding and max_length is not None:
            width = max_length
            mask = [row + [0] * (width - len(row)) for row in mask]
            ids = [row + [0] * (width - len(row)) for row in ids]
        return {"input_ids": ids, "attention_mask": mask}

    def encode(self, text: str) -> list[int]:
        self.row_calls += 1
        return [len(w) for w in text.split()]


def test_tokenizer_uses_one_batched_call_not_one_call_per_row() -> None:
    """Per-row Python `encode` in the hot path is an architecture violation."""
    tok = _FakeHFTokenizer()
    ds = bt.from_pydict({"t": ["a bb ccc", "dddd e", "ff", "g hh"]})
    out = Tokenizer("t", tok, output_column="ids").fit_transform(ds).to_pydict()
    assert tok.row_calls == 0, "tokenizer was called per row"
    assert tok.batch_calls == 1
    assert out["ids"] == [[1, 2, 3], [4, 1], [2], [1, 2]]


def test_tokenizer_truncates_and_emits_an_attention_mask() -> None:
    tok = _FakeHFTokenizer()
    ds = bt.from_pydict({"t": ["a bb ccc dddd", "e"]})
    out = (
        Tokenizer(
            "t",
            tok,
            output_column="ids",
            max_length=2,
            truncation=True,
            padding="max_length",
            attention_mask_column="mask",
        )
        .fit_transform(ds)
        .to_pydict()
    )
    assert out["ids"] == [[1, 2], [1, 0]]
    assert out["mask"] == [[1, 1], [1, 0]]


def test_tokenizer_batched_path_preserves_nulls() -> None:
    tok = _FakeHFTokenizer()
    ds = bt.from_pydict({"t": ["a bb", None, "ccc"]})
    out = Tokenizer("t", tok, output_column="ids").fit_transform(ds).to_pydict()
    assert out["ids"] == [[1, 2], None, [3]]
    assert tok.row_calls == 0


def test_tokenizer_still_supports_a_plain_callable() -> None:
    ds = bt.from_pydict({"t": ["a b", "c"]})
    assert Tokenizer("t", str.split).fit_transform(ds).to_pydict() == {"t": [["a", "b"], ["c"]]}


# --- 6: cross-fitted target encoding ---------------------------------------------


def test_target_encoder_cross_fitting_breaks_the_leak() -> None:
    """Plain target encoding reproduces the target exactly; `cv` must not."""
    cats = [f"c{i}" for i in range(12)]
    y = [float(i % 2) for i in range(12)]
    ds = bt.from_pydict({"c": cats, "y": y})
    leaky = TargetEncoder(["c"], "y", smoothing=0.0).fit_transform(ds).to_pydict()["c"]
    assert leaky == y, "sanity: the plain encoder leaks the target"
    fitted = TargetEncoder(["c"], "y", smoothing=0.0, cv=3).fit_transform(ds).to_pydict()["c"]
    assert fitted != y, "cross-fitted encoding still reproduced the target"


def test_target_encoder_cv_transform_uses_the_full_mapping() -> None:
    """`transform` on unseen data keeps the plain, full-data encoding."""
    ds = bt.from_pydict({"c": ["a", "a", "b", "b"], "y": [1.0, 1.0, 0.0, 0.0]})
    enc = TargetEncoder(["c"], "y", smoothing=0.0, cv=2).fit(ds)
    got = enc.transform(bt.from_pydict({"c": ["a", "z"]})).to_pydict()["c"]
    assert got == [1.0, 0.5]


def test_target_encoder_without_cv_is_unchanged() -> None:
    ds = bt.from_pydict({"c": ["a", "a", "b", "b"], "y": [1.0, 1.0, 0.0, 0.0]})
    got = TargetEncoder(["c"], "y", smoothing=0.0).fit_transform(ds).to_pydict()["c"]
    assert got == [1.0, 1.0, 0.0, 0.0]


def test_target_encoder_rejects_a_degenerate_cv() -> None:
    with pytest.raises(PlanError, match="cv"):
        TargetEncoder(["c"], "y", cv=1)


# --- 7: polynomial term-count guard ----------------------------------------------


def test_polynomial_features_rejects_a_term_explosion() -> None:
    """degree=3 over 20 columns is 1,540 silent new expressions."""
    cols = [f"x{i}" for i in range(20)]
    ds = bt.from_pydict({c: [1.0] for c in cols})
    pre = PolynomialFeatures(cols, degree=3)
    with pytest.raises(PlanError, match="max_terms"):
        pre.fit_transform(ds)


def test_polynomial_features_max_terms_is_configurable() -> None:
    cols = [f"x{i}" for i in range(20)]
    ds = bt.from_pydict({c: [1.0] for c in cols})
    out = PolynomialFeatures(cols, degree=3, max_terms=2000).fit_transform(ds)
    assert len(out.columns) == 20 + 210 + 1540  # degree 2 and degree 3 monomials


def test_polynomial_features_small_case_is_unchanged() -> None:
    ds = bt.from_pydict({"a": [2.0], "b": [3.0]})
    got = PolynomialFeatures(["a", "b"], degree=2).fit_transform(ds).to_pydict()
    assert got == {"a": [2.0], "b": [3.0], "a^2": [4.0], "a*b": [6.0], "b^2": [9.0]}


# --- the binning guard shares the CASE-chain shape --------------------------------


def test_kbins_rejects_an_unbounded_bin_count() -> None:
    ds = bt.from_pydict({"v": [float(i) for i in range(100)]})
    with pytest.raises(PlanError, match="n_bins"):
        KBinsDiscretizer(["v"], n_bins=5000, strategy="uniform").fit(ds)
