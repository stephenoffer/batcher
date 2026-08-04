"""`HashingVectorizer` — a bag of words with no vocabulary and therefore no fit.

The vocabulary is what makes `CountVectorizer` expensive at the edges: it has to be learned
in a pass over the corpus, held on the driver, broadcast to every worker, and kept in sync
between training and serving. The hashing trick removes all four problems by deciding a
term's feature index arithmetically — ``hash(term) % n_features`` — which is stateless, so
there is nothing to learn, nothing to ship, and no train/serve skew possible.

What it costs is collisions and interpretability: two terms can land on one feature, and no
feature can be named. In practice a wide enough feature space makes collisions rare enough
not to matter for a linear model, which is why this is the standard choice for a streaming
text classifier — and streaming is exactly where a vocabulary pass is impossible anyway.

The hash runs in the engine, not in Python: `str.hash64` over the term list produces the
codes as an ordinary `Expr`, so the per-token work stays in Rust.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.base import Preprocessor, column_arg
from batcher.ml.preprocessors.vectorizers.assemble import bag_of_words, set_columns
from batcher.ml.preprocessors.vectorizers.tokens import (
    DEFAULT_TOKEN_PATTERN,
    resolve_stop_words,
    term_expr,
    validate_ngram_range,
)
from batcher.plan.expr_ir.constructors import lit
from batcher.plan.functions.collection import element

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["HashingVectorizer"]

_NORMS = ("l1", "l2", None)


class HashingVectorizer(Preprocessor):
    """Vectorize text into a fixed-width feature space by hashing, with no fitted state.

    `fit` does nothing and exists only so this composes into a `Chain` like every other
    preprocessor. `transform` writes ``<output_column>_indices`` and
    ``<output_column>_values``, or one fixed-width ``List<Float64>`` column named
    `output_column` under ``dense=True``.

    Because there is no vocabulary, `n_features` is the whole feature space and should be
    generous — a few hundred thousand is ordinary. Collisions degrade a model gracefully;
    too narrow a space does not.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import HashingVectorizer
            >>> ds = bt.from_pydict({"t": ["red car", "red bike"]})
            >>> out = HashingVectorizer("t", n_features=8, norm=None).fit_transform(ds)
            >>> [len(v) for v in out.to_pydict()["features_values"]]
            [2, 2]

    Args:
        column: The text column to vectorize.
        n_features: The width of the hashed feature space.
        output_column: The base name of the emitted columns.
        lowercase: Lowercase each document before tokenizing.
        token_pattern: The regex whose matches are the tokens.
        stop_words: ``None``, ``"english"``, or the words to drop.
        ngram_range: The inclusive ``(min_n, max_n)`` bounds on n-gram length.
        binary: Record presence as ``1.0`` rather than the count.
        norm: ``"l2"`` (the default), ``"l1"``, or ``None`` to leave rows unscaled.
        dense: Emit one fixed-width list column instead of the index/value pair.
    """

    __slots__ = (
        "binary",
        "column",
        "dense",
        "lowercase",
        "n_features",
        "ngram_range",
        "norm",
        "output_column",
        "stop_words",
        "token_pattern",
    )

    def __init__(
        self,
        column: str,
        *,
        n_features: int = 2**18,
        output_column: str = "features",
        lowercase: bool = True,
        token_pattern: str = DEFAULT_TOKEN_PATTERN,
        stop_words: str | Sequence[str] | None = None,
        ngram_range: tuple[int, int] = (1, 1),
        binary: bool = False,
        norm: str | None = "l2",
        dense: bool = False,
    ) -> None:
        what = type(self).__name__
        self.column = column_arg(column, what=what)
        if n_features < 1:
            raise PlanError(f"{what}: n_features must be at least 1, got {n_features}")
        self.n_features = n_features
        self.output_column = output_column
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = resolve_stop_words(stop_words, what=what)
        self.ngram_range = validate_ngram_range(ngram_range, what=what)
        self.binary = binary
        if norm not in _NORMS:
            raise PlanError(f"{what}: norm must be 'l1', 'l2', or None, got {norm!r}")
        self.norm = norm
        self.dense = dense

    @property
    def indices_column(self) -> str:
        """The name of the emitted feature-index column.

        Examples:
            .. doctest::

                >>> from batcher.ml.preprocessors import HashingVectorizer
                >>> HashingVectorizer("t", output_column="bow").indices_column
                'bow_indices'

        Returns:
            The column name; unused under ``dense=True``, which emits no index column.
        """
        return f"{self.output_column}_indices"

    @property
    def values_column(self) -> str:
        """The name of the emitted values column.

        Examples:
            .. doctest::

                >>> from batcher.ml.preprocessors import HashingVectorizer
                >>> HashingVectorizer("t").values_column
                'features_values'
                >>> HashingVectorizer("t", dense=True).values_column
                'features'

        Returns:
            The column name, which is `output_column` itself in dense mode.
        """
        return self.output_column if self.dense else f"{self.output_column}_values"

    def _codes(self) -> Any:
        """The `List<Int64>` expression of hashed feature indices for each document.

        The hash is signed, so it is folded into range with `abs` before the modulo rather
        than with a bare ``%``, which would map half of the vocabulary onto negative
        indices and silently drop it.
        """
        terms = term_expr(
            self.column,
            lowercase=self.lowercase,
            token_pattern=self.token_pattern,
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
        )
        return terms.list.transform(element().str.hash64().abs() % lit(self.n_features))

    def transform(self, ds: Dataset) -> Dataset:
        """Append each document's hashed term counts, lazily.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import HashingVectorizer
                >>> ds = bt.from_pydict({"t": ["car car"]})
                >>> pre = HashingVectorizer("t", n_features=16, norm=None).fit(ds)
                >>> pre.transform(ds).to_pydict()["features_values"]
                [[2.0]]

        Args:
            ds: The dataset whose text column to vectorize.

        Returns:
            A new lazy `Dataset` with the vectorized columns appended.
        """
        self._require_fitted()
        width, binary, dense, norm = self.n_features, self.binary, self.dense, self.norm
        indices_column, values_column = self.indices_column, self.values_column
        code_column = "__bt_codes"
        keep = list(ds.columns)
        for extra in [values_column] if dense else [indices_column, values_column]:
            if extra not in keep:
                keep.append(extra)

        def _udf(batch: Any) -> Any:
            built = bag_of_words(
                batch.column(code_column),
                vocabulary=None,
                n_features=width,
                binary=binary,
                norm=norm,
                dense=dense,
            )
            written = {values_column: built["values"]}
            if not dense:
                written[indices_column] = built["indices"]
            return set_columns(batch, written).select(keep)

        staged = ds.with_columns(**{code_column: self._codes()})
        return staged.map_batches(_udf, output_columns=keep)
