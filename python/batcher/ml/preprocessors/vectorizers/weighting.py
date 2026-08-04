"""`TfidfVectorizer` — counts reweighted by how rare each term is across the corpus.

A raw count says "this document says *the* forty times", which is true and useless. TF-IDF
divides that out: a term's weight rises with its frequency in the document and falls with
the number of documents that contain it, so what survives is what distinguishes this
document from the rest. It is still the default feature representation for text
classification, retrieval, and near-duplicate detection whenever an embedding model is more
machinery than the problem needs.

The vectorizer subclasses `CountVectorizer` rather than post-processing it, because the
document frequency it needs is already what that class learns — `fit` is the same single
relational aggregate, and only the value written per term changes. The IDF variant matches
scikit-learn's smoothed default, so a model trained against either library sees the same
numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batcher._internal.errors import PlanError
from batcher.ml.preprocessors.vectorizers.counts import MAX_VOCABULARY, CountVectorizer
from batcher.ml.preprocessors.vectorizers.tokens import DEFAULT_TOKEN_PATTERN

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["TfidfVectorizer"]

_NORMS = ("l1", "l2", None)


class TfidfVectorizer(CountVectorizer):
    """Learn a vocabulary and write each document as TF-IDF weights.

    Everything `CountVectorizer` does, with each count multiplied by the term's inverse
    document frequency and the row optionally scaled to unit norm. The output columns are
    the same: ``<output_column>_indices`` and ``<output_column>_values``, or one
    fixed-width ``List<Float64>`` column named `output_column` under ``dense=True``.

    The IDF is ``ln((1 + n) / (1 + df)) + 1`` when `smooth_idf` is set, which is
    scikit-learn's default and is the form that cannot divide by zero on a term the
    transform set has but the fit set did not.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import TfidfVectorizer
            >>> ds = bt.from_pydict({"t": ["red car", "red bike"]})
            >>> out = TfidfVectorizer("t").fit_transform(ds)
            >>> [round(v, 3) for v in out.to_pydict()["features_values"][0]]
            [0.815, 0.58]

    Args:
        column: The text column to vectorize.
        output_column: The base name of the emitted columns.
        lowercase: Lowercase each document before tokenizing.
        token_pattern: The regex whose matches are the tokens.
        stop_words: ``None``, ``"english"``, or the words to drop.
        ngram_range: The inclusive ``(min_n, max_n)`` bounds on n-gram length.
        min_df: Keep a term appearing in at least this many documents; a float is read as
            a fraction of the corpus.
        max_df: Drop a term appearing in more than this many documents; a float is read as
            a fraction of the corpus.
        max_features: Keep only this many terms, the most frequent first.
        binary: Use presence rather than the count as the term frequency.
        dense: Emit one fixed-width list column instead of the index/value pair.
        max_vocabulary: The ceiling on the learned vocabulary size.
        norm: ``"l2"`` (the default), ``"l1"``, or ``None`` to leave rows unscaled.
        use_idf: Apply the inverse-document-frequency weighting at all.
        smooth_idf: Add one to every document frequency, as if a document containing every
            term existed.
        sublinear_tf: Replace a term frequency ``n`` with ``1 + log(n)``.
    """

    __slots__ = ("norm", "smooth_idf", "sublinear_tf", "use_idf")

    def __init__(
        self,
        column: str,
        *,
        output_column: str = "features",
        lowercase: bool = True,
        token_pattern: str = DEFAULT_TOKEN_PATTERN,
        stop_words: str | Sequence[str] | None = None,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        max_features: int | None = None,
        binary: bool = False,
        dense: bool = False,
        max_vocabulary: int = MAX_VOCABULARY,
        norm: str | None = "l2",
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
    ) -> None:
        super().__init__(
            column,
            output_column=output_column,
            lowercase=lowercase,
            token_pattern=token_pattern,
            stop_words=stop_words,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            binary=binary,
            dense=dense,
            max_vocabulary=max_vocabulary,
        )
        if norm not in _NORMS:
            raise PlanError(
                f"{type(self).__name__}: norm must be 'l1', 'l2', or None, got {norm!r}"
            )
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    @property
    def idf_(self) -> list[float]:
        """The inverse document frequency of each vocabulary term, in vocabulary order.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import TfidfVectorizer
                >>> ds = bt.from_pydict({"t": ["red car", "red bike"]})
                >>> pre = TfidfVectorizer("t").fit(ds)
                >>> [round(v, 3) for v in pre.idf_]
                [1.405, 1.405, 1.0]

        Returns:
            One weight per vocabulary term; all ones when `use_idf` is off.
        """
        self._require_fitted()
        if not self.use_idf:
            return [1.0] * len(self.vocabulary_)
        import math

        offset = 1 if self.smooth_idf else 0
        total = self.document_count_ + offset
        return [math.log(total / (df + offset)) + 1.0 for df in self.document_frequencies_]

    def _weights(self) -> Any:
        """The IDF vector as a NumPy array, or ``None`` when the weighting is off."""
        if not self.use_idf:
            return None
        import numpy as np

        return np.asarray(self.idf_, dtype=np.float64)

    def _row_norm(self) -> str | None:
        """The per-row normalization to apply after weighting."""
        return self.norm

    def _sublinear(self) -> bool:
        """Whether a term frequency ``n`` is replaced by ``1 + log(n)``."""
        return self.sublinear_tf
