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
from batcher.ml.preprocessors.vectorizers.assemble import (
    bag_of_words,
    null_document_policy,
    set_columns,
)
from batcher.ml.preprocessors.vectorizers.tokens import (
    DEFAULT_TOKEN_PATTERN,
    resolve_stop_words,
    term_expr,
    validate_ngram_range,
)
from batcher.plan.expr_ir.constructors import hash_rows, lit, when
from batcher.plan.functions.collection import element

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["HashingVectorizer"]

_NORMS = ("l1", "l2", None)
#: The term hashes on offer; ``"murmur3"`` is `hash_rows(algorithm="iceberg")`, which is the
#: standard seed-0 MurmurHash3_x86_32 of the UTF-8 bytes that scikit-learn computes.
_HASH_FUNCTIONS = frozenset({"fnv1a", "murmur3"})


class HashingVectorizer(Preprocessor):
    """Vectorize text into a fixed-width feature space by hashing, with no fitted state.

    `fit` does nothing and exists only so this composes into a `Chain` like every other
    preprocessor. `transform` writes ``<output_column>_indices`` and
    ``<output_column>_values``, or one fixed-width ``List<Float64>`` column named
    `output_column` under ``dense=True``.

    Because there is no vocabulary, `n_features` is the whole feature space and should be
    generous — a few hundred thousand is ordinary. Collisions degrade a model gracefully;
    too narrow a space does not.

    By default the feature index is ``abs(fnv1a_64(term)) % n_features``, from the engine's
    ``str.hash64``. scikit-learn's ``HashingVectorizer`` uses signed 32-bit MurmurHash3 and, by
    default, ``alternate_sign=True``, so under the default the two assign different indices
    and signs to the same term, and a model trained on one cannot score features from the
    other. ``hash_function="murmur3"`` with ``alternate_sign=True`` reproduces scikit-learn's
    indices and signs exactly: the index is ``abs(murmur3_32(term)) % n_features`` and a term
    whose hash is negative contributes -1, so a model fitted on scikit-learn's hashed
    features scores Batcher's. Tokenization must match too, and this class's defaults
    (lowercasing, words of two or more characters) are scikit-learn's.

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
        hash_function: ``"fnv1a"`` (the engine's ``str.hash64``) or ``"murmur3"``
            (scikit-learn's signed 32-bit MurmurHash3, seed 0).
        alternate_sign: Give a term the sign of its hash, so colliding terms tend to cancel
            rather than add, as scikit-learn does by default.
        null_documents: ``"empty"`` reads a null document as one with no terms; ``"null"``
            makes every output column null for it, so missing text stays distinguishable.
    """

    __slots__ = (
        "alternate_sign",
        "binary",
        "column",
        "dense",
        "hash_function",
        "lowercase",
        "n_features",
        "ngram_range",
        "norm",
        "null_documents",
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
        hash_function: str = "fnv1a",
        alternate_sign: bool = False,
        null_documents: str = "empty",
    ) -> None:
        what = type(self).__name__
        self.null_documents = null_document_policy(null_documents, what=what)
        if hash_function not in _HASH_FUNCTIONS:
            raise PlanError(
                f"{what}: hash_function must be one of {sorted(_HASH_FUNCTIONS)}, "
                f"got {hash_function!r}"
            )
        self.hash_function = hash_function
        self.alternate_sign = alternate_sign
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
        term = element()
        digest = (
            hash_rows(term, algorithm="iceberg")
            if self.hash_function == "murmur3"
            else term.str.hash64()
        )
        index = digest.abs() % lit(self.n_features)
        if not self.alternate_sign:
            return terms.list.transform(index)
        # Signed codes, ±(index + 1): `bag_of_words` reads the sign as the token's
        # contribution and the magnitude as its feature. The +1 keeps index 0 signable.
        signed = when(digest >= lit(0)).then(index + lit(1)).otherwise(lit(-1) - index)
        return terms.list.transform(signed)

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
        signed = self.alternate_sign
        indices_column, values_column = self.indices_column, self.values_column
        code_column = "__bt_codes"
        outputs = [values_column] if dense else [indices_column, values_column]
        final = list(ds.columns)
        for extra in outputs:
            if extra not in final:
                final.append(extra)
        keep_nulls = self.null_documents == "null"
        text_column = self.column

        def _udf(batch: Any) -> Any:
            null_rows = (
                batch.column(text_column).is_null().to_numpy(zero_copy_only=False)
                if keep_nulls
                else None
            )
            built = bag_of_words(
                batch.column(code_column),
                vocabulary=None,
                n_features=width,
                binary=binary,
                norm=norm,
                dense=dense,
                signed_codes=signed,
                null_rows=null_rows,
            )
            written = {values_column: built["values"]}
            if not dense:
                written[indices_column] = built["indices"]
            return set_columns(batch, written).select(final)

        staged = ds.with_columns(**{code_column: self._codes()})
        return staged.map_batches(_udf, output_columns=final)
