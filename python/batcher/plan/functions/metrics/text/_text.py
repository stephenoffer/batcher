"""The tokenization and ratio-shape helpers every text metric in this package builds on.

Text metrics divide into two jobs: turn a string column into comparable units, and reduce a
per-row ratio of those units to a corpus number. Both were being written out per module, which is
how `_char_ngrams` came to exist twice and five overlap metrics came to share one body. They live
here once so a change to the tokenization changes every metric that depends on it.

Nothing here touches a row. Each helper returns an `Expr` the engine evaluates column-wise in
Rust, which is what lets a million-row eval be one scan rather than a Python loop.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import lit, when
from batcher.plan.expr_ir.core import Expr
from batcher.plan.functions.collection import element

__all__ = ["char_ngrams", "mean_ratio", "normalize", "token_ngrams", "tokens"]


def normalize(text: Expr) -> Expr:
    """The SQuAD answer normalization: lowercase, drop articles and punctuation, collapse spaces."""
    lowered = text.str.lower()
    without_articles = lowered.str.regexp_replace_all(r"\b(a|an|the)\b", " ")
    without_punct = without_articles.str.remove_punctuation()
    return without_punct.str.normalize_whitespace().str.strip()


def tokens(text: Expr) -> Expr:
    """The normalized whitespace-delimited tokens of a text column, as a list.

    Splitting an *empty* normalized string yields `[""]` rather than `[]`, which would give a
    row with no tokens a phantom single token: `n_unique / len` reads 1/1 instead of scoring
    the row zero. That is not hypothetical — normalization drops articles, so a generation
    stuck on "the the the" normalizes to empty and scored a perfect diversity of 1.0, the exact
    degeneration `distinct_token_ratio` exists to catch. Dropping empty tokens makes the list
    genuinely empty so `mean_ratio`'s zero-denominator guard fires.
    """
    return normalize(text).str.split(" ").list.filter(element() != lit(""))


def token_ngrams(text: Expr, n: int) -> Expr:
    """The word n-grams of a SQuAD-normalized text column, as a list of joined strings.

    The word-level sibling of `char_ngrams`, and it normalizes the same way `tokens` does —
    lowercased, articles and punctuation dropped — so every word-level metric in this package
    agrees on what a token is. That is a deliberate departure from the reference BLEU
    implementations, which n-gram the text exactly as tokenized; the functions built on this
    say so in their own documentation.

    A row with fewer than `n` tokens still yields one n-gram of everything it has, so a short
    reference is scored rather than silently skipped.
    """
    return normalize(text).str.token_ngrams(n)


def char_ngrams(text: Expr, n: int) -> Expr:
    """The character n-grams of a case-folded, space-collapsed text column, as a list."""
    normalized = text.str.lower().str.normalize_whitespace().str.strip()
    return normalized.str.chunk(n, overlap=n - 1)


def mean_ratio(numerator: Expr, denominator: Expr) -> Expr:
    """The corpus mean of a per-row ratio, scoring an empty denominator as zero.

    Every overlap and diversity metric here has the same shape: count something per row, divide by
    a per-row total that can be zero on an empty string, then average over the corpus. Guarding the
    division rather than letting it produce a null is what keeps an empty row a real zero in the
    mean instead of silently dropping it from the denominator.

    Args:
        numerator: The per-row count being scored.
        denominator: The per-row total to divide by; a row where this is zero contributes zero.

    Returns:
        The mean of the per-row ratio over the corpus.
    """
    ratio = when(denominator > lit(0)).then(numerator / denominator).otherwise(lit(0.0))
    return ratio.mean()
