"""Generation-quality metrics — scoring an LLM's text output against a reference, in one pass.

Evaluating a language model means comparing a generated string to a reference, and the honest
metrics for that are lexical-overlap measures: did the model produce the right answer (exact
match), the right words (token overlap), or at least contain the answer somewhere. Each one here
is a per-row expression over two text columns that aggregates to a corpus score, so an eval over a
million generations is one scan through the engine rather than a Python loop over examples.

The token metrics are *set*-based — they compare the sets of tokens, so repeated tokens count
once. That is the right, stable choice for the "did it recover the key terms" question these
answer, and it is stated in each metric so it is never confused with a multiset BLEU/ROUGE score.
Normalization (lowercasing, stripping punctuation and articles) follows the SQuAD convention, so a
prediction that differs from the reference only in casing or a trailing period still counts.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import lit, when
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if
from batcher.plan.functions.metrics._text import mean_ratio, normalize, tokens

__all__ = [
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
