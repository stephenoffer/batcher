"""`CountVectorizer` — a learned vocabulary and the term counts of each document.

The bag-of-words step every classical text model starts from, and the one piece of
scikit-learn's text stack that has no equivalent elsewhere in this package: `Tokenizer`
produces tokens, `HashingEncoder` hashes a single categorical value, and neither of them
learns a vocabulary or counts terms within a document.

`fit` is a relational aggregate, not a driver-side `Counter`. The term column is an
expression (`tokens.term_expr`), so the vocabulary comes from an `explode` into a
`group_by`, which is mergeable — the same fit runs on one core, on every core, or across a
cluster, and spills rather than dying on a corpus that does not fit in memory. That is what
makes a vocabulary over a corpus larger than the driver possible at all.
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
from batcher.plan.expr_ir.constructors import col, lit
from batcher.plan.functions.aggregate import count_if

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.api.dataset import Dataset

__all__ = ["MAX_VOCABULARY", "CountVectorizer"]

#: The default ceiling on a learned vocabulary. The vocabulary is materialized on the
#: driver and broadcast to every worker, so it is bounded for the same reason a categorical
#: encoder's category set is — but a text vocabulary is legitimately far larger than a
#: category set, so the ceiling is correspondingly higher. Raise it, or set `max_features`,
#: rather than treating the error as a reason to sample the corpus.
MAX_VOCABULARY = 1_000_000


def _document_stats(ds: Dataset, terms: Any) -> Dataset:
    """One aggregate returning each term's document frequency and total occurrences.

    Both numbers are needed and they come from different explodes — document frequency
    counts a term once per document, total occurrences counts every appearance — so the two
    term streams are unioned with a discriminator and reduced in a single `group_by`. That
    is one shuffle instead of two, and the counts cannot disagree about which documents
    were in scope.
    """
    occurrences = ds.select(__bt_term=terms, __bt_doc=lit(0)).explode("__bt_term")
    presences = ds.select(__bt_term=terms.list.unique(), __bt_doc=lit(1)).explode("__bt_term")
    return (
        occurrences.union(presences)
        .group_by("__bt_term")
        .agg(
            __bt_df=count_if(col("__bt_doc") == lit(1)),
            __bt_tf=count_if(col("__bt_doc") == lit(0)),
        )
    )


class CountVectorizer(Preprocessor):
    """Learn a vocabulary from a text column and count each document's terms.

    `fit` learns which terms make the vocabulary; `transform` writes each document as the
    aligned pair of list columns a sparse row is: ``<output_column>_indices`` holds the
    vocabulary positions present in the document and ``<output_column>_values`` holds the
    matching counts. Pass ``dense=True`` for a small vocabulary to get one fixed-width
    ``List<Float64>`` column named `output_column` instead, which is what a trainer reading
    through `iter_torch_batches` wants.

    A term unseen at fit time is ignored rather than bucketed, matching scikit-learn: the
    fitted vocabulary is the feature space, and a serving-time document that uses new words
    simply has fewer features set.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors import CountVectorizer
            >>> ds = bt.from_pydict({"t": ["red car", "red red bike"]})
            >>> out = CountVectorizer("t", dense=True).fit_transform(ds)
            >>> out.to_pydict()["features"]
            [[0.0, 1.0, 1.0], [1.0, 0.0, 2.0]]

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
        max_features: Keep only this many terms, the most frequent first. Terms tied on
            frequency at the cut are broken alphabetically, so the fitted vocabulary is
            reproducible; scikit-learn leaves that tie to an unstable sort, so the two can
            legitimately keep different terms from a tied group.
        binary: Record presence as ``1.0`` rather than the count.
        dense: Emit one fixed-width list column instead of the index/value pair.
        max_vocabulary: The ceiling on the learned vocabulary size.
    """

    __slots__ = (
        "binary",
        "column",
        "dense",
        "document_count_",
        "document_frequencies_",
        "lowercase",
        "max_df",
        "max_features",
        "max_vocabulary",
        "min_df",
        "ngram_range",
        "output_column",
        "stop_words",
        "token_pattern",
        "vocabulary_",
    )

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
    ) -> None:
        what = type(self).__name__
        self.column = column_arg(column, what=what)
        self.output_column = output_column
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = resolve_stop_words(stop_words, what=what)
        self.ngram_range = validate_ngram_range(ngram_range, what=what)
        self.min_df = min_df
        self.max_df = max_df
        if max_features is not None and max_features < 1:
            raise PlanError(f"{what}: max_features must be at least 1, got {max_features}")
        self.max_features = max_features
        self.binary = binary
        self.dense = dense
        self.max_vocabulary = max_vocabulary
        self.vocabulary_: list[str] = []
        self.document_frequencies_: list[int] = []
        self.document_count_ = 0

    @property
    def indices_column(self) -> str:
        """The name of the emitted vocabulary-index column.

        Examples:
            .. doctest::

                >>> from batcher.ml.preprocessors import CountVectorizer
                >>> CountVectorizer("t", output_column="bow").indices_column
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

                >>> from batcher.ml.preprocessors import CountVectorizer
                >>> CountVectorizer("t").values_column
                'features_values'
                >>> CountVectorizer("t", dense=True).values_column
                'features'

        Returns:
            The column name, which is `output_column` itself in dense mode.
        """
        return self.output_column if self.dense else f"{self.output_column}_values"

    def _terms(self) -> Any:
        """The `List<Utf8>` term expression this vectorizer's settings describe."""
        return term_expr(
            self.column,
            lowercase=self.lowercase,
            token_pattern=self.token_pattern,
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
        )

    def _bounds(self, n_documents: int) -> tuple[int, int]:
        """The absolute document-frequency window `min_df`/`max_df` describe."""
        low = self.min_df
        high = self.max_df
        # A float is a fraction of the corpus and an int is an absolute count — the same
        # split scikit-learn draws, and the reason the type is checked rather than the value.
        lower = round(low * n_documents) if isinstance(low, float) else low
        upper = round(high * n_documents) if isinstance(high, float) else high
        if lower > upper:
            raise PlanError(
                f"{type(self).__name__}: min_df={self.min_df!r} resolves to {lower} documents, "
                f"above max_df={self.max_df!r} at {upper}. No term can satisfy both."
            )
        return max(lower, 1), upper

    def fit(self, ds: Dataset) -> CountVectorizer:
        """Learn the vocabulary and each term's document frequency.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import CountVectorizer
                >>> ds = bt.from_pydict({"t": ["red car", "red bike"]})
                >>> CountVectorizer("t").fit(ds).vocabulary_
                ['bike', 'car', 'red']

        Args:
            ds: The corpus to learn the vocabulary from.

        Returns:
            ``self``, fitted.

        Raises:
            PlanError: If the vocabulary exceeds `max_vocabulary`, or the document-frequency
                bounds admit nothing.
        """
        n_documents = ds.count()
        lower, upper = self._bounds(n_documents)
        stats = _document_stats(ds, self._terms()).filter(
            (col("__bt_df") >= lit(lower)) & (col("__bt_df") <= lit(upper))
        )
        # Ranked before the ceiling is applied, so `max_features` keeps the *most frequent*
        # terms rather than whichever ones the shuffle happened to emit first. The tie-break
        # on the term itself makes the fitted vocabulary reproducible run to run.
        ranked = stats.sort("__bt_tf", "__bt_term", descending=[True, False])
        keep = self.max_features if self.max_features is not None else self.max_vocabulary + 1
        table = ranked.limit(keep).collect()
        terms = table.column("__bt_term").to_pylist()
        if self.max_features is None and len(terms) > self.max_vocabulary:
            raise PlanError(
                f"{type(self).__name__}: {self.column!r} yields more than "
                f"{self.max_vocabulary} distinct terms. The vocabulary is held on the driver "
                "and broadcast to every worker, so it is bounded on purpose. Set max_features "
                "to keep the most frequent terms, raise min_df to drop the long tail, or raise "
                "max_vocabulary to accept the cost."
            )
        frequencies = dict(zip(terms, table.column("__bt_df").to_pylist(), strict=True))
        # Sorted so the feature indices are stable and readable; the frequency ranking above
        # only decided *which* terms survive, not their order in the feature space.
        self.vocabulary_ = sorted(terms)
        self.document_frequencies_ = [int(frequencies[t]) for t in self.vocabulary_]
        self.document_count_ = int(n_documents)
        self._fitted = True
        return self

    def _output_columns(self, ds: Dataset) -> list[str]:
        """The transformed dataset's column names, in order."""
        names = list(ds.columns)
        added = [self.values_column] if self.dense else [self.indices_column, self.values_column]
        for extra in added:
            if extra not in names:
                names.append(extra)
        return names

    def _weights(self) -> Any:
        """The per-feature multiplier applied to each count; none for plain counts."""
        return None

    def _row_norm(self) -> str | None:
        """The per-row normalization applied after weighting; none for plain counts."""
        return None

    def _sublinear(self) -> bool:
        """Whether a count ``n`` is replaced by ``1 + log(n)``."""
        return False

    def transform(self, ds: Dataset) -> Dataset:
        """Append each document's term counts, lazily.

        Examples:
            .. doctest::

                >>> import batcher as bt
                >>> from batcher.ml.preprocessors import CountVectorizer
                >>> pre = CountVectorizer("t").fit(bt.from_pydict({"t": ["red car"]}))
                >>> pre.transform(bt.from_pydict({"t": ["car car"]})).to_pydict()["features_values"]
                [[2.0]]

        Args:
            ds: The dataset whose text column to vectorize.

        Returns:
            A new lazy `Dataset` with the vectorized columns appended.
        """
        self._require_fitted()
        import pyarrow as pa

        vocabulary = pa.array(self.vocabulary_, type=pa.string())
        width = len(self.vocabulary_)
        weights, norm, sublinear = self._weights(), self._row_norm(), self._sublinear()
        binary, dense = self.binary, self.dense
        indices_column, values_column = self.indices_column, self.values_column
        term_column = "__bt_terms"
        keep = self._output_columns(ds)

        def _udf(batch: Any) -> Any:
            built = bag_of_words(
                batch.column(term_column),
                vocabulary=vocabulary,
                n_features=width,
                binary=binary,
                sublinear_tf=sublinear,
                weights=weights,
                norm=norm,
                dense=dense,
            )
            written = {values_column: built["values"]}
            if not dense:
                written[indices_column] = built["indices"]
            return set_columns(batch, written).select(keep)

        staged = ds.with_columns(**{term_column: self._terms()})
        return staged.map_batches(_udf, output_columns=keep)
