"""Word n-gram overlap — the clipped-count metrics BLEU and ROUGE-N are defined on.

The set metrics in `overlap.py` ask *which* tokens two texts share. These ask *how many
times*, and the difference decides whether a degenerate generation scores well. A model
stuck emitting ``the the the the`` shares the token ``the`` with almost any reference, so a
set precision reads 1.0; clipping each n-gram at the number of times the reference actually
contains it reads 0.25. Clipping is the entire reason BLEU is defined the way it is, and
`list.multiset_overlap` is the primitive that does it in the engine.

Every function here reduces to a corpus number, so an eval over a million generations is one
scan rather than a Python loop over examples. All of them tokenize through the package's
SQuAD normalization (lowercase, articles and punctuation dropped), which is what keeps every
word-level metric here comparable — and which is *not* what a reference BLEU implementation
does, so treat these as stable in-engine scores for ranking runs against each other rather
than as numbers to publish against a paper's table.
"""

from __future__ import annotations

from batcher._internal.errors import PlanError
from batcher.plan.expr_ir.constructors import lit, when
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column
from batcher.plan.functions.metrics.text._text import token_ngrams, tokens

__all__ = [
    "bleu",
    "brevity_penalty",
    "distinct_ngram_ratio",
    "ngram_f1",
    "ngram_novelty",
    "ngram_precision",
    "ngram_recall",
    "rouge_l_f1",
    "rouge_l_precision",
    "rouge_l_recall",
]


def _validate_n(n: int, func: str) -> None:
    """Reject an n-gram size that cannot form a gram."""
    if n < 1:
        raise PlanError(f"{func}: n must be at least 1, got {n}")


def _clipped(prediction: IntoExpr, reference: IntoExpr, n: int) -> tuple[Expr, Expr, Expr]:
    """The clipped overlap and the two n-gram counts every metric here divides by."""
    pred = token_ngrams(_as_column(prediction), n)
    gold = token_ngrams(_as_column(reference), n)
    return pred.list.multiset_overlap(gold), pred.list.len(), gold.list.len()


def _mean_over(numerator: Expr, denominator: Expr) -> Expr:
    """The corpus mean of a per-row ratio, scoring an empty denominator zero, not null."""
    return when(denominator > lit(0)).then(numerator / denominator).otherwise(lit(0.0)).mean()


def ngram_precision(prediction: IntoExpr, reference: IntoExpr, n: int = 1) -> Expr:
    """The mean modified n-gram precision — BLEU's per-order term, averaged over the corpus.

    Of the n-grams the model emitted, what fraction the reference can account for, with each
    n-gram **clipped** at the number of times the reference contains it. The clip is what
    makes this the honest precision: repeating one correct n-gram cannot raise the score past
    the one occurrence the reference justifies. Read alongside `ngram_recall`, or combined by
    `ngram_f1`; `bleu` is the geometric mean of this across several orders.

    Args:
        prediction: The generated-text column (name or expression).
        reference: The gold-reference column.
        n: The n-gram size. ``1`` is unigram precision.

    Returns:
        The mean modified n-gram precision over the corpus, in ``[0, 1]``.

    Raises:
        PlanError: If `n` is less than 1.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["the cat sat"], "r": ["the cat sat down"]})
            >>> ds.agg(p=bt.ngram_precision("p", "r")).to_pydict()["p"][0]
            1.0

            >>> # Clipping refuses to reward a repeated token.
            >>> deg = bt.from_pydict({"p": ["cat cat cat cat"], "r": ["cat sat down"]})
            >>> deg.agg(p=bt.ngram_precision("p", "r")).to_pydict()["p"][0]
            0.25
    """
    _validate_n(n, "ngram_precision")
    overlap, pred_len, _ = _clipped(prediction, reference, n)
    return _mean_over(overlap, pred_len)


def ngram_recall(prediction: IntoExpr, reference: IntoExpr, n: int = 1) -> Expr:
    """The mean n-gram recall — ROUGE-N, averaged over the corpus.

    Of the n-grams the reference contains, what fraction the model reproduced, clipped the
    same way as `ngram_precision`. This is the recall-oriented score ROUGE-N reports, the
    right primary metric for summarization, where covering the reference's content matters
    more than saying nothing extra.

    Args:
        prediction: The generated-text column.
        reference: The gold-reference column.
        n: The n-gram size. ``1`` is ROUGE-1, ``2`` is ROUGE-2.

    Returns:
        The mean n-gram recall over the corpus, in ``[0, 1]``.

    Raises:
        PlanError: If `n` is less than 1.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["the cat sat"], "r": ["the cat sat down"]})
            >>> round(ds.agg(r=bt.ngram_recall("p", "r")).to_pydict()["r"][0], 4)
            0.6667
    """
    _validate_n(n, "ngram_recall")
    overlap, _, gold_len = _clipped(prediction, reference, n)
    return _mean_over(overlap, gold_len)


def ngram_f1(prediction: IntoExpr, reference: IntoExpr, n: int = 1) -> Expr:
    """The mean n-gram F1 — the balanced ROUGE-N score, averaged over the corpus.

    The harmonic mean of the clipped precision and recall, computed per example and then
    averaged. Use it when a summary is penalized both for missing the reference's content and
    for padding beyond it; take `ngram_recall` alone when only coverage matters.

    Args:
        prediction: The generated-text column.
        reference: The gold-reference column.
        n: The n-gram size.

    Returns:
        The mean n-gram F1 over the corpus, in ``[0, 1]``.

    Raises:
        PlanError: If `n` is less than 1.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["cat sat"], "r": ["cat sat"]})
            >>> ds.agg(f=bt.ngram_f1("p", "r")).to_pydict()["f"][0]
            1.0
    """
    _validate_n(n, "ngram_f1")
    overlap, pred_len, gold_len = _clipped(prediction, reference, n)
    precision = when(pred_len > lit(0)).then(overlap / pred_len).otherwise(lit(0.0))
    recall = when(gold_len > lit(0)).then(overlap / gold_len).otherwise(lit(0.0))
    total = precision + recall
    f1 = when(total > lit(0)).then(lit(2.0) * precision * recall / total).otherwise(lit(0.0))
    return f1.mean()


def brevity_penalty(prediction: IntoExpr, reference: IntoExpr) -> Expr:
    """The mean BLEU brevity penalty — how much a too-short generation is discounted.

    ``min(1, exp(1 - reference_length / prediction_length))`` per example, in tokens. Precision
    alone rewards saying less, because a single well-chosen word can be perfectly precise; the
    brevity penalty is what removes that incentive. It is 1.0 whenever the generation is at
    least as long as its reference, and falls toward 0 as it gets shorter. `bleu` already
    applies it; report it separately to see *why* a BLEU score is low.

    Args:
        prediction: The generated-text column.
        reference: The gold-reference column.

    Returns:
        The mean brevity penalty over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["cat sat down"], "r": ["cat sat"]})
            >>> ds.agg(bp=bt.brevity_penalty("p", "r")).to_pydict()["bp"][0]
            1.0
    """
    return _per_row_brevity(prediction, reference).mean()


def _per_row_brevity(prediction: IntoExpr, reference: IntoExpr) -> Expr:
    """The per-row brevity penalty, shared by `brevity_penalty` and `bleu`."""
    pred_len = token_ngrams(_as_column(prediction), 1).list.len()
    gold_len = token_ngrams(_as_column(reference), 1).list.len()
    # An empty generation is scored 0 rather than left null: it is a real failure, and a null
    # would drop the row out of the corpus mean entirely.
    ratio = when(pred_len > lit(0)).then(gold_len / pred_len).otherwise(lit(0.0))
    penalty = (lit(1.0) - ratio).exp()
    return (
        when(pred_len <= lit(0))
        .then(lit(0.0))
        .otherwise(when(pred_len >= gold_len).then(lit(1.0)).otherwise(penalty))
    )


def bleu(prediction: IntoExpr, reference: IntoExpr, max_n: int = 4) -> Expr:
    """The mean sentence-level BLEU — clipped n-gram precisions with a brevity penalty.

    Per example: the geometric mean of the modified n-gram precisions for orders ``1..max_n``,
    multiplied by the brevity penalty; then averaged over the corpus. This is *sentence* BLEU
    averaged over examples, not corpus BLEU (which pools the counts before dividing) — the
    averaged form is the one that composes with `group_by`, so you can read BLEU per prompt
    category in the same scan.

    There is no smoothing: an example that produces no n-gram of some order scores 0, which is
    the unsmoothed definition and is harsh on short outputs at ``max_n=4``. Lower `max_n` for
    short-answer tasks rather than reading a corpus of zeros.

    Args:
        prediction: The generated-text column.
        reference: The gold-reference column.
        max_n: The highest n-gram order to include. Must be at least 1.

    Returns:
        The mean sentence BLEU over the corpus, in ``[0, 1]``.

    Raises:
        PlanError: If `max_n` is less than 1.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["the cat sat down"], "r": ["the cat sat down"]})
            >>> ds.agg(b=bt.bleu("p", "r")).to_pydict()["b"][0]
            1.0

            >>> # A generation sharing no bigram with its reference scores zero.
            >>> miss = bt.from_pydict({"p": ["dog ran"], "r": ["cat sat"]})
            >>> miss.agg(b=bt.bleu("p", "r", max_n=2)).to_pydict()["b"][0]
            0.0
    """
    _validate_n(max_n, "bleu")
    product: Expr = lit(1.0)
    for n in range(1, max_n + 1):
        overlap, pred_len, _ = _clipped(prediction, reference, n)
        precision = when(pred_len > lit(0)).then(overlap / pred_len).otherwise(lit(0.0))
        product = product * precision
    # The geometric mean as an explicit root of the product rather than exp(mean(log p)):
    # a zero precision must give exactly 0, and `log(0)` is -inf, which propagates as a NaN
    # through the mean instead of the 0 the definition calls for.
    geometric = product.pow(lit(1.0 / max_n))
    return (_per_row_brevity(prediction, reference) * geometric).mean()


def distinct_ngram_ratio(text: IntoExpr, n: int = 2) -> Expr:
    """The mean distinct-n — what fraction of a generation's n-grams are unique.

    The standard word-level degeneration signal: a model that loops on a phrase repeats the
    same n-grams, so this falls even while length and fluency look fine. Read it per run and
    watch for a drop; the absolute value depends on `n` and on how long the outputs are.

    At ``n=1`` this is `distinct_token_ratio` — use that spelling for unigrams. The value here
    is orders ``n >= 2``, where phrase-level looping shows up long before token-level does.

    Args:
        text: The generated-text column (name or expression).
        n: The n-gram size.

    Returns:
        The mean distinct-n ratio over the corpus, in ``[0, 1]``.

    Raises:
        PlanError: If `n` is less than 1.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"t": ["go on go on go on"]})
            >>> round(ds.agg(d=bt.distinct_ngram_ratio("t")).to_pydict()["d"][0], 4)
            0.4
    """
    _validate_n(n, "distinct_ngram_ratio")
    grams = token_ngrams(_as_column(text), n)
    return _mean_over(grams.list.n_unique(), grams.list.len())


def ngram_novelty(prediction: IntoExpr, reference: IntoExpr, n: int = 4) -> Expr:
    """The mean fraction of a generation's distinct n-grams that its reference never contained.

    The copying check. A retrieval-augmented or fine-tuned model that has memorized its
    context reproduces long spans of it verbatim, and at ``n=4`` or higher that shows up here
    as a novelty near 0. It is *not* one minus `ngram_precision`: this counts each distinct
    n-gram once, so a phrase copied ten times is one copied n-gram rather than ten, which is
    what you want when asking "how much of this is new" rather than "how accurate is it".

    A generation shorter than `n` tokens has no n-gram of that order to compare, and scores a
    misleading 1.0. Filter short rows out before reading this on a corpus of one-liners.

    Args:
        prediction: The generated-text column.
        reference: The source or reference column to check against.
        n: The n-gram size. Use 4 or more to detect verbatim spans.

    Returns:
        The mean novel-n-gram fraction over the corpus, in ``[0, 1]``.

    Raises:
        PlanError: If `n` is less than 1.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> copied = bt.from_pydict(
            ...     {
            ...         "p": ["the quick brown fox jumps"],
            ...         "r": ["the quick brown fox jumps over"],
            ...     }
            ... )
            >>> copied.agg(nv=bt.ngram_novelty("p", "r")).to_pydict()["nv"][0]
            0.0

            >>> fresh = bt.from_pydict(
            ...     {"p": ["alpha beta gamma delta epsilon"], "r": ["zeta eta theta iota"]}
            ... )
            >>> fresh.agg(nv=bt.ngram_novelty("p", "r")).to_pydict()["nv"][0]
            1.0
    """
    _validate_n(n, "ngram_novelty")
    pred = token_ngrams(_as_column(prediction), n)
    gold = token_ngrams(_as_column(reference), n)
    distinct = pred.list.n_unique()
    shared = pred.list.set_intersection(gold).list.len()
    return _mean_over(distinct - shared, distinct)


def _lcs_parts(prediction: IntoExpr, reference: IntoExpr) -> tuple[Expr, Expr, Expr]:
    """The LCS length and the two token counts the ROUGE-L ratios divide by."""
    pred = tokens(_as_column(prediction))
    gold = tokens(_as_column(reference))
    return pred.list.lcs_length(gold), pred.list.len(), gold.list.len()


def rouge_l_precision(prediction: IntoExpr, reference: IntoExpr) -> Expr:
    """The mean ROUGE-L precision — the longest in-order match, over the generation's length.

    Where `ngram_precision` asks which n-grams the reference can account for, this asks how much
    of the generation lies on a single thread running through the reference in order. A summary
    that uses the reference's vocabulary in its own arrangement scores well on the first and
    badly here, which is the distinction ROUGE-L exists to make.

    Args:
        prediction: The generated-text column (name or expression).
        reference: The gold-reference column.

    Returns:
        The mean ROUGE-L precision over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["cat sat down"], "r": ["cat sat down today"]})
            >>> ds.agg(p=bt.rouge_l_precision("p", "r")).to_pydict()["p"][0]
            1.0
    """
    lcs, pred_len, _ = _lcs_parts(prediction, reference)
    return _mean_over(lcs, pred_len)


def rouge_l_recall(prediction: IntoExpr, reference: IntoExpr) -> Expr:
    """The mean ROUGE-L recall — the longest in-order match, over the reference's length.

    How much of the reference the generation covered *in order*. This is the recall-oriented
    half ROUGE-L reports, and the primary number for summarization, where covering the source's
    content in its sequence matters more than saying nothing extra.

    Args:
        prediction: The generated-text column.
        reference: The gold-reference column.

    Returns:
        The mean ROUGE-L recall over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["cat sat"], "r": ["cat sat down today"]})
            >>> round(ds.agg(r=bt.rouge_l_recall("p", "r")).to_pydict()["r"][0], 4)
            0.5
    """
    lcs, _, gold_len = _lcs_parts(prediction, reference)
    return _mean_over(lcs, gold_len)


def rouge_l_f1(prediction: IntoExpr, reference: IntoExpr) -> Expr:
    """The mean ROUGE-L F1 — the balanced longest-in-order-match score.

    The harmonic mean of `rouge_l_precision` and `rouge_l_recall` per example, averaged over the
    corpus. This is the number usually meant by "ROUGE-L", and the one to report alongside
    `ngram_f1`: a gap between them says the generation has the right content in the wrong order.

    It is `O(n·m)` per row in the two token counts, unlike every other metric here — see
    :meth:`~batcher.Expr.list.lcs_length`. Truncate long documents, or score per sentence.

    Args:
        prediction: The generated-text column.
        reference: The gold-reference column.

    Returns:
        The mean ROUGE-L F1 over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"p": ["cat sat down"], "r": ["cat sat down"]})
            >>> ds.agg(f=bt.rouge_l_f1("p", "r")).to_pydict()["f"][0]
            1.0
    """
    lcs, pred_len, gold_len = _lcs_parts(prediction, reference)
    precision = when(pred_len > lit(0)).then(lcs / pred_len).otherwise(lit(0.0))
    recall = when(gold_len > lit(0)).then(lcs / gold_len).otherwise(lit(0.0))
    total = precision + recall
    f1 = when(total > lit(0)).then(lit(2.0) * precision * recall / total).otherwise(lit(0.0))
    return f1.mean()
