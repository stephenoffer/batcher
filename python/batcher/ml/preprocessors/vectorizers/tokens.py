"""The tokenization step every text vectorizer shares, as one native expression.

Turning a document into terms is the part of a bag-of-words pipeline that has to run over
every row, so it runs in the engine rather than in Python: lowercasing, regex token
extraction, stop-word removal and n-gram assembly are all existing `Expr` operations
(`str.to_lowercase`, `str.extract_all`, `list.filter`, `str.token_ngrams`, `list.concat`).
The vectorizers above this module therefore never see a string — they see a `List<Utf8>`
column of terms, which the relational aggregates can count and the Arrow kernels can look
up.

That matters for more than tidiness. Because the term column is an ordinary expression, a
vectorizer's `fit` is an ordinary `group_by` and inherits the mergeable, spillable,
distributed path for free, and its `transform` stays lazy until a terminal op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.constructors import col, lit
from batcher.plan.functions.collection import element

if TYPE_CHECKING:
    from collections.abc import Sequence

    from batcher.plan.expr_ir import Expr

__all__ = ["ENGLISH_STOP_WORDS", "resolve_stop_words", "term_expr", "validate_ngram_range"]

#: The default token pattern, matching scikit-learn's: runs of two or more word characters.
#: Single characters are dropped, which is what makes ``a`` and ``I`` disappear from a
#: bag of words without needing a stop-word list to name them.
DEFAULT_TOKEN_PATTERN = r"\b\w\w+\b"

# Written as prose and split at import rather than as a list literal: the point of a
# stop-word list is that a reader can scan it and judge whether it suits their corpus,
# and one word per line would run to several screens.
_STOP_WORD_TEXT = """
    a about above after again against all am an and any are aren't as at be because been
    before being below between both but by can cannot could couldn't did didn't do does
    doesn't doing don't down during each few for from further had hadn't has hasn't have
    haven't having he her here hers herself him himself his how i if in into is isn't it
    its itself let's me more most mustn't my myself no nor not of off on once only or
    other ought our ours ourselves out over own same shan't she should shouldn't so some
    such than that the their theirs them themselves then there these they this those
    through to too under until up very was wasn't we were weren't what when where which
    while who whom why with won't would wouldn't you your yours yourself yourselves
"""

#: A general-purpose English stop-word list. It is offered for convenience and is
#: deliberately unopinionated about a specific corpus: a list that suits news text will
#: discard signal in clinical notes, so prefer `min_df`/`max_df` for a corpus you can
#: measure, and pass an explicit list when the domain has its own filler words.
ENGLISH_STOP_WORDS = frozenset(_STOP_WORD_TEXT.split())


def validate_ngram_range(ngram_range: tuple[int, int], *, what: str) -> tuple[int, int]:
    """Check an ``(min_n, max_n)`` pair and return it normalized to a tuple.

    Args:
        ngram_range: The inclusive bounds on n-gram length.
        what: The caller's class name, used in the error message.

    Returns:
        The pair as a two-element tuple of ints.

    Raises:
        PlanError: If the pair is malformed, non-positive, or inverted.

    Examples:
        .. doctest::

            >>> from batcher.ml.preprocessors.vectorizers.tokens import validate_ngram_range
            >>> validate_ngram_range((1, 2), what="CountVectorizer")
            (1, 2)
    """
    try:
        low, high = (int(n) for n in ngram_range)
    except (TypeError, ValueError) as exc:
        raise PlanError(
            f"{what}: ngram_range must be a (min_n, max_n) pair of ints, got {ngram_range!r}"
        ) from exc
    if low < 1 or high < low:
        raise PlanError(
            f"{what}: ngram_range must satisfy 1 <= min_n <= max_n, got ({low}, {high})"
        )
    return (low, high)


def resolve_stop_words(stop_words: str | Sequence[str] | None, *, what: str) -> list[str]:
    """Normalize the `stop_words` argument into an explicit, sorted word list.

    Accepts ``None`` (keep everything), the string ``"english"`` for the built-in list, or
    any sequence of words. The result is sorted so a fitted vectorizer serializes
    identically across runs.

    Args:
        stop_words: ``None``, ``"english"``, or the words to drop.
        what: The caller's class name, used in the error message.

    Returns:
        The words to drop, sorted; empty when `stop_words` is ``None``.

    Raises:
        PlanError: If a string other than ``"english"`` is passed, or a non-string
            sneaks into the sequence.

    Examples:
        .. doctest::

            >>> from batcher.ml.preprocessors.vectorizers.tokens import resolve_stop_words
            >>> resolve_stop_words(["b", "a"], what="CountVectorizer")
            ['a', 'b']
    """
    if stop_words is None:
        return []
    if isinstance(stop_words, str):
        if stop_words != "english":
            raise PlanError(
                f"{what}: stop_words must be None, 'english', or a list of words, got "
                f"{stop_words!r}. A bare string is read as a named list, not as one word."
            )
        return sorted(ENGLISH_STOP_WORDS)
    words = list(stop_words)
    bad = [w for w in words if not isinstance(w, str)]
    if bad:
        raise PlanError(f"{what}: stop_words must all be strings, got {bad[0]!r}")
    return sorted(set(words))


def term_expr(
    column: str,
    *,
    lowercase: bool = True,
    token_pattern: str = DEFAULT_TOKEN_PATTERN,
    stop_words: Sequence[str] = (),
    ngram_range: tuple[int, int] = (1, 1),
) -> Expr:
    """Build the ``List<Utf8>`` expression of terms for one text column.

    The pipeline is lowercase, extract tokens by regex, drop stop words, then assemble
    n-grams. A null document is read as an empty one, so it yields an empty term list
    rather than a null — a document with no terms is still a document, and keeping it null
    would silently drop the row from the count aggregates.

    Args:
        column: The text column to read.
        lowercase: Lowercase before tokenizing.
        token_pattern: The regex whose matches are the tokens.
        stop_words: Tokens to drop after extraction, before n-gram assembly.
        ngram_range: The inclusive ``(min_n, max_n)`` bounds on n-gram length.

    Returns:
        An expression evaluating to the list of terms for each row.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> from batcher.ml.preprocessors.vectorizers.tokens import term_expr
            >>> ds = bt.from_pydict({"t": ["The cat sat"]})
            >>> ds.select(terms=term_expr("t")).to_pydict()
            {'terms': [['the', 'cat', 'sat']]}
    """
    text = col(column).fill_null(lit(""))
    if lowercase:
        text = text.str.to_lowercase()
    tokens = text.str.extract_all(token_pattern)
    if stop_words:
        tokens = tokens.list.filter(~element().is_in(list(stop_words)))
    low, high = ngram_range
    if (low, high) == (1, 1):
        return tokens
    # An n-gram is n adjacent tokens joined by a space, which is exactly what
    # `str.token_ngrams` produces from a whitespace-joined string. The tokens are already
    # normalized at this point, so joining and re-splitting on whitespace is lossless.
    joined = tokens.list.join(" ")
    # Unigrams come from the token list directly; `token_ngrams(1)` would work too but
    # would pay for a join and a re-split to produce what is already in hand.
    grams = tokens if low == 1 else _exact_ngrams(joined, low)
    for n in range(low + 1, high + 1):
        grams = grams.list.concat(_exact_ngrams(joined, n))
    return grams


def _exact_ngrams(joined: Expr, n: int) -> Expr:
    """The n-grams of a whitespace-joined token string, with the short-document case removed.

    `str.token_ngrams` yields the whole text as a single gram when the text has fewer than
    `n` tokens, which is the right answer for the generation metrics it was written for and
    the wrong one here: a two-word document has no trigram, and counting one made
    ``ngram_range=(2, 3)`` double-count every short document's bigram.
    """
    return joined.str.token_ngrams(n).list.filter(element().str.word_count() == lit(n))
