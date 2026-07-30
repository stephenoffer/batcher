"""Length, size, and reading-level metrics — how much text there is and how hard it is.

Length is the first thing to measure about a text corpus and the last thing to be measured
carefully. These cover the three questions that matter: how long the rows are (characters, words,
and their quantiles), how many tokens they will cost a model (estimated from characters, with the
budget-overflow rate that decides whether a batch will fit a context window), and how demanding
they are to read (the ARI grade and the sentence- and word-length drivers behind it).

They share a module because they are all derived from the same counting pass and because a length
distribution is what you look at first when a token budget or a readability target is missed.
"""

from __future__ import annotations

from batcher.plan.expr_ir.constructors import lit, when
from batcher.plan.expr_ir.core import AggExpr, Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if
from batcher.plan.functions.quantiles import approx_quantile

__all__ = [
    "automated_readability_index",
    "char_length_quantile",
    "char_length_range",
    "long_word_rate",
    "max_char_length",
    "mean_chars_per_word",
    "mean_paragraph_count",
    "mean_words_per_sentence",
    "min_char_length",
    "token_budget_exceed_rate",
    "token_estimate_quantile",
    "token_spend",
    "total_token_estimate",
    "word_count_quantile",
]


def _safe_div(num: Expr, den: Expr) -> Expr:
    """``num / den`` where ``den > 0``, else ``0.0`` — a per-row divide-by-zero guard."""
    return when(den > lit(0)).then(num / den).otherwise(lit(0.0))


def automated_readability_index(text: IntoExpr) -> Expr:
    """Mean automated readability index (ARI) over the corpus — a US-grade readability score.

    ARI scores a row as ``4.71 * chars/words + 0.5 * words/sentences - 21.43`` and approximates
    the US school grade needed to read it. A higher value means denser, harder text. Rows with no
    words or no sentences contribute a guarded 0 for that term, so an empty output cannot divide by
    zero. The corpus score is the mean of the per-row scores, mergeable across partitions.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The mean per-row ARI over the corpus.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["The cat sat on the mat. It was warm."]})
            >>> round(ds.agg(m=bt.automated_readability_index("o")).to_pydict()["m"][0], 4)
            -0.34
    """
    col = _as_column(text)
    chars = col.str.len_chars()
    words = col.str.word_count()
    sentences = col.str.sentence_count()
    score = (
        lit(4.71) * _safe_div(chars, words) + lit(0.5) * _safe_div(words, sentences) - lit(21.43)
    )
    return score.mean()


def mean_words_per_sentence(text: IntoExpr) -> Expr:
    """Mean words per sentence over the corpus — the core readability driver in one number.

    For each row this divides the word count by the sentence count, guarding an unpunctuated row
    (zero sentences) with a 0, then averages across rows. Long sentences read as denser prose, so
    a rising value across runs flags a drift toward harder output.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The mean of the per-row words-per-sentence ratio over the corpus.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["The cat sat on the mat. It was warm."]})
            >>> round(ds.agg(m=bt.mean_words_per_sentence("o")).to_pydict()["m"][0], 4)
            4.5
    """
    col = _as_column(text)
    ratio = _safe_div(col.str.word_count(), col.str.sentence_count())
    return ratio.mean()


def mean_chars_per_word(text: IntoExpr) -> Expr:
    """Mean characters per word over the corpus — a tokenizer-free lexical-complexity signal.

    Each row contributes its average letters-per-word, and the corpus score is the mean of those.
    Longer words track denser vocabulary, so this is a cheap complementary signal to the sentence
    length driver.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The mean of the per-row average word length over the corpus.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["The cat sat on the mat. It was warm."]})
            >>> round(ds.agg(m=bt.mean_chars_per_word("o")).to_pydict()["m"][0], 4)
            2.8889
    """
    return _as_column(text).str.avg_word_length().mean()


def long_word_rate(text: IntoExpr, min_length: int = 7) -> Expr:
    """Mean fraction of long words per row over the corpus — a lexical-complexity proxy.

    For each row this divides the count of words at least `min_length` characters by the total word
    count, guarding an empty row with a 0, then averages across rows. A high long-word fraction
    marks technical or ornate prose, so this tracks how advanced the vocabulary of a slice is.

    Args:
        text: The generated-text column (name or expression).
        min_length: The minimum character length for a word to count as long.

    Returns:
        The mean of the per-row long-word fraction over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["extraordinary complicated situations", "the cat sat"]})
            >>> round(ds.agg(m=bt.long_word_rate("o")).to_pydict()["m"][0], 4)
            0.5
    """
    col = _as_column(text)
    ratio = _safe_div(col.str.long_word_count(min_length), col.str.word_count())
    return ratio.mean()


def mean_paragraph_count(text: IntoExpr) -> Expr:
    """Mean number of paragraphs per generation over the corpus — a structural-density signal.

    Each row contributes its paragraph count (blank-line-separated blocks), and the corpus score is
    the mean of those. More paragraphs mark longer, more structured output, so a shift here flags a
    change in generation shape between runs.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The mean per-row paragraph count over the corpus.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["First para.\\n\\nSecond para.\\n\\nThird."]})
            >>> round(ds.agg(m=bt.mean_paragraph_count("o")).to_pydict()["m"][0], 4)
            3.0
    """
    return _as_column(text).str.paragraph_count().mean()


def char_length_quantile(text: IntoExpr, q: float) -> AggExpr:
    """Approximate ``q``-quantile of per-row character length across the corpus.

    Sizes a point in the length tail: ``q=0.95`` is the character length that 95% of generations
    stay under, the number you provision a context window or a truncation limit against. Character
    length comes from `str.len_chars`, and the quantile rides a mergeable sketch, so the estimate
    is identical single-node and distributed but approximate rather than exact.

    Args:
        text: The generated-text column (name or expression).
        q: The quantile to estimate, between 0 and 1.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["ab", "abcd", "abcdef"]})
            >>> round(ds.agg(m=bt.char_length_quantile("o", 0.5)).to_pydict()["m"][0])
            4
    """
    return approx_quantile(_as_column(text).str.len_chars(), q)


def word_count_quantile(text: IntoExpr, q: float) -> AggExpr:
    """Approximate ``q``-quantile of per-row word count across the corpus.

    Sizes a point in the verbosity tail: ``q=0.95`` is the word count that 95% of generations stay
    under, useful when a token budget tracks words more closely than characters. Word count comes
    from `str.word_count`, and the quantile rides a mergeable sketch, so the estimate is identical
    single-node and distributed but approximate rather than exact.

    Args:
        text: The generated-text column (name or expression).
        q: The quantile to estimate, between 0 and 1.

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["a b", "a b c", "a b c d e"]})
            >>> round(ds.agg(m=bt.word_count_quantile("o", 0.5)).to_pydict()["m"][0])
            3
    """
    return approx_quantile(_as_column(text).str.word_count(), q)


def max_char_length(text: IntoExpr) -> AggExpr:
    """Exact character length of the longest generation in the corpus.

    The worst-case output, the one that overflows a fixed buffer or blows a truncation limit. This
    is the exact maximum over per-row `str.len_chars`, so it is a single mergeable aggregate that
    breaks down per model or per day with `group_by`.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["ab", "abcd", "abcdef"]})
            >>> ds.agg(m=bt.max_char_length("o")).to_pydict()["m"][0]
            6
    """
    return _as_column(text).str.len_chars().max()


def min_char_length(text: IntoExpr) -> AggExpr:
    """Exact character length of the shortest generation in the corpus.

    The floor of the distribution, where a near-empty or truncated output shows up as a suspiciously
    small value. This is the exact minimum over per-row `str.len_chars`, a single mergeable
    aggregate that breaks down per model or per day with `group_by`.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        An aggregate expression; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["ab", "abcd", "abcdef"]})
            >>> ds.agg(m=bt.min_char_length("o")).to_pydict()["m"][0]
            2
    """
    return _as_column(text).str.len_chars().min()


def char_length_range(text: IntoExpr) -> Expr:
    """Exact spread of character length across the corpus — longest minus shortest.

    A single number for how wide the length distribution is: a large range means the corpus mixes
    terse and sprawling generations, a signal worth splitting by model. Character length is read
    once from `str.len_chars`, then its exact maximum and minimum are subtracted, so the result is
    mergeable and breaks down per group.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        An expression for the character-length range; pass it to ``agg(...)``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["ab", "abcd", "abcdef"]})
            >>> ds.agg(m=bt.char_length_range("o")).to_pydict()["m"][0]
            4
    """
    length = _as_column(text).str.len_chars()
    return length.max() - length.min()


def total_token_estimate(text: IntoExpr, chars_per_token: float = 4.0) -> Expr:
    """The total estimated token count over a text column — the corpus cost number.

    The sum of the per-row token estimate, which is what a per-token bill is charged on. Run it on
    the prompt column to size an input cost and on the output column to size a generation cost, and
    put it inside `group_by` to attribute spend per model, per tenant, or per day. It is an estimate
    (characters over ``chars_per_token``), so use it to plan and compare, not to reconcile a bill.

    Args:
        text: The text column (name or expression) to size.
        chars_per_token: The average characters per token used for the estimate.

    Returns:
        The total estimated token count over the corpus.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["12345678", "1234"]})
            >>> ds.agg(t=bt.total_token_estimate("o", chars_per_token=4.0)).to_pydict()["t"][0]
            3
    """
    return _as_column(text).str.estimate_tokens(chars_per_token).sum()


def token_budget_exceed_rate(text: IntoExpr, budget: int, chars_per_token: float = 4.0) -> Expr:
    """The fraction of rows whose estimated tokens exceed ``budget`` — the context-overflow rate.

    A row longer than the context window is silently truncated by most engines, losing its tail.
    This is the corpus rate of those rows, the number that tells you whether to raise the window,
    chunk the inputs, or accept the loss before a run rather than discovering it in the outputs.

    Args:
        text: The text column to size.
        budget: The token budget a row must exceed to count as over.
        chars_per_token: The average characters per token used for the estimate.

    Returns:
        The over-budget rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["abcd", "abcdefghijkl"]})
            >>> ds.agg(r=bt.token_budget_exceed_rate("o", budget=2)).to_pydict()["r"][0]
            0.5
    """
    fits = _as_column(text).str.fits_token_budget(budget, chars_per_token)
    return count_if(~fits) / count_if(lit(True))


def token_estimate_quantile(text: IntoExpr, q: float, chars_per_token: float = 4.0) -> AggExpr:
    """The ``q``-quantile of the per-row estimated token count — the tail that sizes the window.

    The mean token length hides the tail, and it is the tail that overflows a context window. This
    is the approximate ``q``-quantile of the per-row estimate over a mergeable sketch, so at
    ``q = 0.95`` it is the window that covers 95% of inputs without truncation. Use it to pick a
    context size or a per-request cap from the data rather than a guess, sketch-approximate.

    Args:
        text: The text column to size.
        q: The quantile in ``[0, 1]`` (``0.95`` for the 95th percentile).
        chars_per_token: The average characters per token used for the estimate.

    Returns:
        The approximate ``q``-quantile of the per-row token estimate.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["1234", "12345678", "123456789012"]})
            >>> round(ds.agg(p=bt.token_estimate_quantile("o", q=0.5)).to_pydict()["p"][0])
            2
    """
    return approx_quantile(_as_column(text).str.estimate_tokens(chars_per_token), q)


def token_spend(
    input_tokens: IntoExpr,
    output_tokens: IntoExpr,
    *,
    input_price: float,
    output_price: float,
) -> Expr:
    """The total spend implied by two token-count columns, at the given per-million prices.

    Prices are **per million tokens**, the unit every current provider quotes, and input and
    output are separate because they are priced separately — output is typically several times
    input, so a run's bill is driven by generation length far more than by prompt length.

    Feed it the *measured* usage columns `ds.ml.generate(usage=True)` appends, not an estimate,
    and it reconciles against an invoice. Inside `group_by` it attributes spend per model, per
    tenant, or per prompt template in one scan, which is the breakdown a provider's own billing
    page does not give you.

    Args:
        input_tokens: The prompt-token count column.
        output_tokens: The completion-token count column.
        input_price: Price per million input tokens.
        output_price: Price per million output tokens.

    Returns:
        The total spend over the corpus, in the currency the prices were given in.

    Raises:
        PlanError: If either price is negative.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> usage = bt.from_pydict({"pt": [1_000_000], "ct": [500_000]})
            >>> usage.agg(
            ...     cost=bt.token_spend("pt", "ct", input_price=3.0, output_price=15.0)
            ... ).to_pydict()["cost"][0]
            10.5
    """
    from batcher._internal.errors import PlanError

    for name, price in (("input_price", input_price), ("output_price", output_price)):
        if price < 0:
            raise PlanError(f"token_spend: {name} must not be negative, got {price}")
    per_row = _as_column(input_tokens) * lit(input_price / 1_000_000.0) + _as_column(
        output_tokens
    ) * lit(output_price / 1_000_000.0)
    return per_row.sum()
