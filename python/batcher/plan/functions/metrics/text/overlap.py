"""Lexical-overlap metrics — scoring generated text against a reference, in one pass.

Evaluating a language model means comparing a generated string to a reference, and the honest,
model-free signals for that are overlap measures. Two granularities live here. Word-level metrics
compare the *sets* of SQuAD-normalized tokens, which is the right stable choice for "did it recover
the key terms"; character n-gram metrics compare overlapping character chunks instead, which needs
no tokenizer and so works on languages without word boundaries. Both are the precision/recall/F1
family plus Jaccard, and both aggregate to a corpus score, so an eval over a million generations is
one scan through the engine rather than a Python loop over examples.

These are surface-overlap proxies, not semantic judges. They are the cheap first-pass filter you
run before paying for a model-based grader.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import lit, when
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if
from batcher.plan.functions.metrics.text._text import char_ngrams, mean_ratio, normalize, tokens

__all__ = [
    "char_ngram_f1",
    "char_ngram_jaccard",
    "char_ngram_precision",
    "char_ngram_recall",
    "exact_match",
    "length_ratio",
    "normalized_exact_match",
    "token_set_f1",
    "token_set_jaccard",
    "token_set_precision",
    "token_set_recall",
]


def exact_match(prediction: IntoExpr, reference: IntoExpr) -> Expr:
    """The fraction of predictions that equal the reference exactly, character for character.

    The strictest generation metric: the model's output must match the reference string byte for
    byte. It is the right score for a task with one canonical answer (a label, a normalized entity,
    a formatted number) and the wrong one for free-form text, where `normalized_exact_match` or the
    token metrics are fairer. Returned as the corpus rate, so it is directly the accuracy.

    Args:
        prediction: The generated-text column.
        reference: The gold-reference column.

    Returns:
        The exact-match rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"p": ["Paris", "London", "rome"], "r": ["Paris", "Lisbon", "Rome"]}
            ... )
            >>> round(ds.agg(em=bt.exact_match("p", "r")).to_pydict()["em"][0], 4)
            0.3333
    """
    return count_if(_as_column(prediction) == _as_column(reference)) / count_if(lit(True))


def normalized_exact_match(prediction: IntoExpr, reference: IntoExpr) -> Expr:
    """The exact-match rate after SQuAD normalization — the fair "same answer" score.

    Like `exact_match`, but both sides are first lowercased and stripped of articles, punctuation,
    and extra whitespace, so ``"The Paris."`` matches ``"paris"``. This is the standard
    quasi-exact-match used across question-answering and instruction-following benchmarks, and the
    right default when the answer is a short phrase whose surface form is not what you are grading.

    Args:
        prediction: The generated-text column.
        reference: The gold-reference column.

    Returns:
        The normalized exact-match rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["The Paris.", "london"], "r": ["paris", "Lisbon"]})
            >>> ds.agg(em=bt.normalized_exact_match("p", "r")).to_pydict()["em"][0]
            0.5
    """
    predicted = normalize(_as_column(prediction))
    gold = normalize(_as_column(reference))
    return count_if(predicted == gold) / count_if(lit(True))


def token_set_precision(prediction: IntoExpr, reference: IntoExpr) -> Expr:
    """The mean per-example token-set precision — the predicted words that are in the reference.

    Over the *set* of tokens (repeats counted once): the intersection over the prediction's token
    count. High precision means the model added few spurious words. It is the "did it avoid saying
    wrong things" half of `token_set_f1`, averaged per example over the corpus.

    Args:
        prediction: The generated-text column.
        reference: The gold-reference column.

    Returns:
        The mean token-set precision, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["the cat sat"], "r": ["the cat sat on the mat"]})
            >>> ds.agg(p=bt.token_set_precision("p", "r")).to_pydict()["p"][0]
            1.0
    """
    predicted, gold = tokens(_as_column(prediction)), tokens(_as_column(reference))
    intersection = predicted.list.set_intersection(gold).list.len()
    return mean_ratio(intersection, predicted.list.n_unique())


def token_set_recall(prediction: IntoExpr, reference: IntoExpr) -> Expr:
    """The mean per-example token-set recall — the reference words the model produced.

    Over the *set* of tokens: the intersection over the reference's token count. High recall means
    the model covered the reference's content; it is the "did it say the right things" half of
    `token_set_f1`, averaged per example over the corpus.

    Args:
        prediction: The generated-text column.
        reference: The gold-reference column.

    Returns:
        The mean token-set recall, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["the cat sat on the mat"], "r": ["the cat sat"]})
            >>> ds.agg(r=bt.token_set_recall("p", "r")).to_pydict()["r"][0]
            1.0
    """
    predicted, gold = tokens(_as_column(prediction)), tokens(_as_column(reference))
    intersection = predicted.list.set_intersection(gold).list.len()
    return mean_ratio(intersection, gold.list.n_unique())


def token_set_f1(prediction: IntoExpr, reference: IntoExpr) -> Expr:
    """The mean per-example token-set F1 — the balanced word-overlap score for free-form answers.

    The harmonic mean of `token_set_precision` and `token_set_recall`, which reduces to
    ``2 * |intersection| / (|prediction| + |reference|)`` over the token sets, averaged per example.
    It is the workhorse metric for grading a generated answer against a reference when neither exact
    match nor a single substring is right — an open-ended QA answer, a summary, a rewrite.

    Args:
        prediction: The generated-text column.
        reference: The gold-reference column.

    Returns:
        The mean token-set F1, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"p": ["the quick brown fox"], "r": ["a slow brown fox"]}
            ... )
            >>> round(ds.agg(f=bt.token_set_f1("p", "r")).to_pydict()["f"][0], 4)
            0.6667
    """
    predicted, gold = tokens(_as_column(prediction)), tokens(_as_column(reference))
    intersection = predicted.list.set_intersection(gold).list.len()
    total = predicted.list.n_unique() + gold.list.n_unique()
    ratio = when(total > lit(0)).then(lit(2.0) * intersection / total).otherwise(lit(0.0))
    return ratio.mean()


def token_set_jaccard(prediction: IntoExpr, reference: IntoExpr) -> Expr:
    """The mean per-example token-set Jaccard — the intersection over the union of the word sets.

    The overlap coefficient sibling of `token_set_f1`: ``|intersection| / |union|`` of the token
    sets, averaged per example. It penalizes both missing and spurious words in one symmetric
    number, and unlike F1 it is a true set-similarity in ``[0, 1]`` that reads as "what fraction of
    all the words the two share".

    Args:
        prediction: The generated-text column.
        reference: The gold-reference column.

    Returns:
        The mean token-set Jaccard similarity, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict(
            ...     {"p": ["the quick brown fox"], "r": ["a slow brown fox"]}
            ... )
            >>> round(ds.agg(j=bt.token_set_jaccard("p", "r")).to_pydict()["j"][0], 4)
            0.5
    """
    predicted, gold = tokens(_as_column(prediction)), tokens(_as_column(reference))
    intersection = predicted.list.set_intersection(gold).list.len()
    union = predicted.list.set_union(gold).list.len()
    ratio = when(union > lit(0)).then(intersection / union).otherwise(lit(0.0))
    return ratio.mean()


def length_ratio(prediction: IntoExpr, reference: IntoExpr) -> Expr:
    """The mean ratio of prediction length to reference length, in words — a verbosity check.

    How long the model's output is relative to the reference: 1 is length-matched, above 1 is
    padding or rambling, below 1 is terseness or truncation. It is not a quality score on its own,
    but it is the fastest way to catch a model that is systematically over- or under-generating,
    which a length-insensitive metric like `token_set_f1` hides.

    Args:
        prediction: The generated-text column.
        reference: The gold-reference column.

    Returns:
        The mean prediction-to-reference word-count ratio.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["a b c d"], "r": ["a b"]})
            >>> ds.agg(lr=bt.length_ratio("p", "r")).to_pydict()["lr"][0]
            2.0
    """
    predicted_words = _as_column(prediction).str.word_count()
    gold_words = _as_column(reference).str.word_count()
    ratio = when(gold_words > lit(0)).then(predicted_words / gold_words).otherwise(lit(0.0))
    return ratio.mean()


def _validate_n(n: int) -> None:
    from batcher._internal.errors import PlanError

    if n < 1:
        raise PlanError(f"char n-gram size must be >= 1, got {n}")


def char_ngram_precision(prediction: IntoExpr, reference: IntoExpr, n: int = 3) -> Expr:
    """The mean per-example character n-gram precision — predicted n-grams found in the reference.

    Over the *set* of character n-grams (each counted once): the intersection over the prediction's
    n-gram count, averaged per example. It is the tokenizer-free, language-agnostic counterpart of
    `token_set_precision`, and the right precision half of chrF for text without word boundaries.

    Args:
        prediction: The generated-text column (name or expression).
        reference: The gold-reference column.
        n: The character n-gram size.

    Returns:
        The mean character n-gram precision, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["abcd"], "r": ["abce"]})
            >>> round(ds.agg(p=bt.char_ngram_precision("p", "r", n=2)).to_pydict()["p"][0], 4)
            0.6667
    """
    _validate_n(n)
    pred, gold = char_ngrams(_as_column(prediction), n), char_ngrams(_as_column(reference), n)
    intersection = pred.list.set_intersection(gold).list.len()
    size = pred.list.n_unique()
    return when(size > lit(0)).then(intersection / size).otherwise(lit(0.0)).mean()


def char_ngram_recall(prediction: IntoExpr, reference: IntoExpr, n: int = 3) -> Expr:
    """The mean per-example character n-gram recall — the reference n-grams the prediction produced.

    Over the *set* of character n-grams: the intersection over the reference's n-gram count,
    averaged per example. The tokenizer-free counterpart of `token_set_recall`, and the recall half
    of chrF. chrF weights recall more heavily than precision; use `char_ngram_f1` for the balanced
    combination or read this alongside `char_ngram_precision` to weight them yourself.

    Args:
        prediction: The generated-text column.
        reference: The gold-reference column.
        n: The character n-gram size.

    Returns:
        The mean character n-gram recall, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["abce"], "r": ["abcd"]})
            >>> round(ds.agg(r=bt.char_ngram_recall("p", "r", n=2)).to_pydict()["r"][0], 4)
            0.6667
    """
    _validate_n(n)
    pred, gold = char_ngrams(_as_column(prediction), n), char_ngrams(_as_column(reference), n)
    intersection = pred.list.set_intersection(gold).list.len()
    size = gold.list.n_unique()
    return when(size > lit(0)).then(intersection / size).otherwise(lit(0.0)).mean()


def char_ngram_f1(prediction: IntoExpr, reference: IntoExpr, n: int = 3) -> Expr:
    """The mean per-example character n-gram F1 — the chrF-style overlap score.

    The harmonic mean of character n-gram precision and recall per example, averaged over the
    corpus. It is the language-agnostic overlap metric for generation: unlike `token_set_f1` it
    needs no word boundaries, so it scores Chinese, Japanese, or inflected output fairly. This is a
    set-based F1 (β = 1); true chrF uses β = 2 to favor recall, so treat this as a stable chrF-style
    number rather than the exact statistic.

    Args:
        prediction: The generated-text column.
        reference: The gold-reference column.
        n: The character n-gram size.

    Returns:
        The mean character n-gram F1, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["abcd"], "r": ["abcd"]})
            >>> ds.agg(f=bt.char_ngram_f1("p", "r", n=2)).to_pydict()["f"][0]
            1.0
    """
    _validate_n(n)
    pred, gold = char_ngrams(_as_column(prediction), n), char_ngrams(_as_column(reference), n)
    intersection = pred.list.set_intersection(gold).list.len()
    precision = (
        when(pred.list.n_unique() > lit(0))
        .then(intersection / pred.list.n_unique())
        .otherwise(lit(0.0))
    )
    recall = (
        when(gold.list.n_unique() > lit(0))
        .then(intersection / gold.list.n_unique())
        .otherwise(lit(0.0))
    )
    denominator = precision + recall
    f1 = (
        when(denominator > lit(0))
        .then(lit(2.0) * precision * recall / denominator)
        .otherwise(lit(0.0))
    )
    return f1.mean()


def char_ngram_jaccard(prediction: IntoExpr, reference: IntoExpr, n: int = 3) -> Expr:
    """The mean per-example character n-gram Jaccard — intersection over union of the n-gram sets.

    The set-similarity view of character n-gram overlap: the shared n-grams over the total distinct
    n-grams across both strings, averaged per example. It is symmetric in the two columns (unlike
    precision and recall) and the tokenizer-free counterpart of `token_set_jaccard`.

    Args:
        prediction: The generated-text column.
        reference: The gold-reference column.
        n: The character n-gram size.

    Returns:
        The mean character n-gram Jaccard index, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["abcd"], "r": ["abce"]})
            >>> ds.agg(j=bt.char_ngram_jaccard("p", "r", n=2)).to_pydict()["j"][0]
            0.5
    """
    _validate_n(n)
    pred, gold = char_ngrams(_as_column(prediction), n), char_ngrams(_as_column(reference), n)
    intersection = pred.list.set_intersection(gold).list.len()
    union = pred.list.set_union(gold).list.len()
    return when(union > lit(0)).then(intersection / union).otherwise(lit(0.0)).mean()
