"""Surface-quality metrics — the hygiene of raw text, before anyone judges its meaning.

Text arriving from a scrape, an OCR pass, or a model's output carries defects that have nothing to
do with whether the content is right: stray whitespace, tabs where spaces belong, shouted
all-caps, mojibake, runaway punctuation, stray URLs, outputs that are empty or absurdly long.
These are the corpus rates for those defects, and they are what a data-quality gate thresholds on
before a training set is accepted or a generation is shown to a user.

Each is a per-row predicate averaged over the corpus, so the number reads directly as "what
fraction of my rows have this problem".
"""

from __future__ import annotations

from batcher._internal.errors import require_int
from batcher.plan.expr_ir.constructors import lit
from batcher.plan.expr_ir.core import Expr, IntoExpr
from batcher.plan.functions.aggregate import _as_column, count_if

__all__ = [
    "all_caps_rate",
    "blank_line_rate",
    "code_block_rate",
    "double_space_rate",
    "empty_or_whitespace_rate",
    "has_tab_rate",
    "leading_whitespace_rate",
    "long_output_rate",
    "mean_sentence_count",
    "mean_word_length",
    "non_ascii_rate",
    "repeated_punctuation_rate",
    "short_output_rate",
    "trailing_whitespace_rate",
    "url_rate",
]


def _rate(condition: Expr) -> Expr:
    """The fraction of rows where ``condition`` holds, as a corpus rate in ``[0, 1]``."""
    return count_if(condition) / count_if(lit(True))


def all_caps_rate(text: IntoExpr) -> Expr:
    """The fraction of generations written entirely in capitals — a shouting-degeneration monitor.

    An output with letters and no lowercase is almost always wrong: a model stuck in caps, a copied
    banner, or a degenerate loop. This is the corpus rate of those, a cheap red flag that something
    is off with a slice before you read a single row.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The all-caps rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["STOP DOING THAT", "a normal reply", "OK"]})
            >>> round(ds.agg(a=bt.all_caps_rate("o")).to_pydict()["a"][0], 4)
            0.6667
    """
    return _rate(_as_column(text).str.is_all_caps())


def repeated_punctuation_rate(text: IntoExpr) -> Expr:
    """The fraction of generations with a run of repeated punctuation — a degeneration monitor.

    A run such as ``!!!`` or ``???`` is a marker of low-quality or degenerate output, and a spike in
    this rate between runs is a fast signal that decoding has drifted. It is the corpus rate of
    outputs containing at least one such run.

    Args:
        text: The generated-text column.

    Returns:
        The repeated-punctuation rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["really???", "calm.", "wow!!!"]})
            >>> round(ds.agg(r=bt.repeated_punctuation_rate("o")).to_pydict()["r"][0], 4)
            0.6667
    """
    return _rate(_as_column(text).str.has_repeated_punctuation())


def non_ascii_rate(text: IntoExpr) -> Expr:
    """The fraction of generations containing a non-ASCII character — a language-drift monitor.

    A jump in this rate on an English task flags mojibake, an unexpected language, or hallucinated
    glyphs. It is the corpus rate of outputs with at least one non-ASCII character, so on a
    genuinely multilingual corpus it will be high by design; read it against its own baseline.

    Args:
        text: The generated-text column.

    Returns:
        The non-ASCII rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["café", "plain", "naive"]})
            >>> round(ds.agg(n=bt.non_ascii_rate("o")).to_pydict()["n"][0], 4)
            0.3333
    """
    return _rate(_as_column(text).str.has_non_ascii())


def url_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that contain a URL — a hallucinated-link and injection monitor.

    A model that invents citations or is steered by an injected instruction often emits a URL. This
    is the corpus rate of outputs containing one, so a rise flags a batch worth auditing for
    fabricated links before they reach a user.

    Args:
        text: The generated-text column.

    Returns:
        The URL rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["see https://example.com", "no link here"]})
            >>> ds.agg(u=bt.url_rate("o")).to_pydict()["u"][0]
            0.5
    """
    return _rate(_as_column(text).str.has_url())


def code_block_rate(text: IntoExpr) -> Expr:
    """The fraction of generations containing a fenced code block — a code-leakage/format monitor.

    On a task that should return prose, a fenced code block is a formatting failure; on a code task,
    its absence is. This is the corpus rate of outputs with at least one triple-backtick fence,
    either the alarm or the health check depending on what the task expects.

    Args:
        text: The generated-text column.

    Returns:
        The code-block rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["use ```print(1)```", "just prose"]})
            >>> ds.agg(c=bt.code_block_rate("o")).to_pydict()["c"][0]
            0.5
    """
    return _rate(_as_column(text).str.code_fence_count() > lit(0))


def long_output_rate(text: IntoExpr, min_chars: int) -> Expr:
    """The fraction of generations longer than ``min_chars`` characters — a verbosity/cost monitor.

    The rate of outputs above a length threshold, for catching a model that has started to ramble
    or for sizing the fraction of a batch at risk of hitting a token limit. Length is in characters,
    so pick a threshold near the character budget you care about.

    Args:
        text: The generated-text column.
        min_chars: The exclusive character count an output must exceed to count as long.

    Returns:
        The long-output rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["short", "a much longer answer here"]})
            >>> ds.agg(l=bt.long_output_rate("o", min_chars=10)).to_pydict()["l"][0]
            0.5
    """
    min_chars = require_int(min_chars, func="long_output_rate", arg="min_chars")
    return _rate(_as_column(text).str.len_chars() > lit(min_chars))


def short_output_rate(text: IntoExpr, max_chars: int) -> Expr:
    """The fraction of generations shorter than ``max_chars`` characters — a low-effort monitor.

    The rate of outputs below a length threshold, for catching terse or empty-ish answers a model
    gives when it has nothing useful to say. Pair it with `long_output_rate` to bound the length
    distribution from both ends. Length is in characters.

    Args:
        text: The generated-text column.
        max_chars: The exclusive character count an output must fall under to count as short.

    Returns:
        The short-output rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["ok", "a much longer answer here"]})
            >>> ds.agg(s=bt.short_output_rate("o", max_chars=10)).to_pydict()["s"][0]
            0.5
    """
    max_chars = require_int(max_chars, func="short_output_rate", arg="max_chars")
    return _rate(_as_column(text).str.len_chars() < lit(max_chars))


def mean_sentence_count(text: IntoExpr) -> Expr:
    """The mean number of sentences per generation — a structural-length monitor.

    Sentence count captures a different axis of length than token count: a model can grow verbose by
    writing longer sentences or by writing more of them, and this tracks the second. Useful for
    watching structure drift between prompt versions.

    Args:
        text: The generated-text column.

    Returns:
        The mean sentence count over the corpus.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["One. Two. Three.", "Just one."]})
            >>> ds.agg(s=bt.mean_sentence_count("o")).to_pydict()["s"][0]
            2.0
    """
    return _as_column(text).str.sentence_count().mean()


def mean_word_length(text: IntoExpr) -> Expr:
    """The mean average word length across generations — a lexical-style monitor.

    The corpus average of each output's mean word length, a cheap proxy for register: technical or
    formal text uses longer words, and a shift signals a change in style or a different model. Read
    it as a trend against its own baseline.

    Args:
        text: The generated-text column.

    Returns:
        The mean average word length over the corpus, in characters.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["aa bb", "cccc dddd"]})
            >>> ds.agg(w=bt.mean_word_length("o")).to_pydict()["w"][0]
            3.0
    """
    return _as_column(text).str.avg_word_length().mean()


def trailing_whitespace_rate(text: IntoExpr) -> Expr:
    """The fraction of generations whose final character is whitespace — a trailing-whitespace rate.

    Matches ``\\s$``, which anchors at the absolute end of the string (the Rust regex ``$`` is not
    multiline here), so a generation counts when its last character is a space, a tab, or a
    newline. A trailing newline is therefore itself flagged as trailing whitespace. A rise between
    runs points at a prompt change that started leaving stray whitespace at the end of answers.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The trailing-whitespace rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["clean text", "trailing space ", "\\ttabbed"]})
            >>> round(ds.agg(t=bt.trailing_whitespace_rate("o")).to_pydict()["t"][0], 4)
            0.3333
    """
    return count_if(_as_column(text).str.regexp_matches(r"\s$")) / count_if(lit(True))


def leading_whitespace_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that begin with a whitespace char — a leading-whitespace rate.

    Matches ``^\\s``, one space, tab, or newline at the very start of the string. A model that
    indents or blank-pads the top of its answer produces output that misaligns when concatenated or
    re-wrapped, and this catches it as a corpus number rather than one row at a time.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The leading-whitespace rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["clean text", "trailing space ", "\\ttabbed"]})
            >>> round(ds.agg(le=bt.leading_whitespace_rate("o")).to_pydict()["le"][0], 4)
            0.3333
    """
    return count_if(_as_column(text).str.regexp_matches(r"^\s")) / count_if(lit(True))


def has_tab_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that contain at least one tab character — a tab-presence rate.

    Counts a generation when its tab count is positive, using the ``tab_count`` string primitive
    rather than a regex. Tabs render inconsistently across viewers and break column-aligned or
    tab-separated downstream parsing, so a nonzero rate is worth watching even when the text looks
    fine.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The tab-present rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["clean text", "trailing space ", "\\ttabbed"]})
            >>> round(ds.agg(h=bt.has_tab_rate("o")).to_pydict()["h"][0], 4)
            0.3333
    """
    return count_if(_as_column(text).str.tab_count() > lit(0)) / count_if(lit(True))


def double_space_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that contain a run of two or more spaces — a doubled-space rate.

    Matches ``  +`` (a space followed by one or more spaces) anywhere in the string. Collapsed or
    doubled spacing survives most rendering but corrupts fixed-width layout and shows up as noise in
    a diff, so this flags the outputs where the model doubled a space.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The doubled-space rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["a  b", "a b", "a   b"]})
            >>> round(ds.agg(d=bt.double_space_rate("o")).to_pydict()["d"][0], 4)
            0.6667
    """
    return count_if(_as_column(text).str.regexp_matches(r"  +")) / count_if(lit(True))


def blank_line_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that contain a blank line — a blank-line-gap rate.

    Matches ``\\n[ \\t]*\\n``, two newlines separated only by optional spaces or tabs, so a fully
    empty line or a line of only whitespace between two content lines both count. Blank-line gaps
    inflate output and can split a record that a downstream reader expected to be contiguous.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The blank-line rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["para one\\n\\npara two", "single line", "no gap"]})
            >>> round(ds.agg(b=bt.blank_line_rate("o")).to_pydict()["b"][0], 4)
            0.3333
    """
    return count_if(_as_column(text).str.regexp_matches(r"\n[ \t]*\n")) / count_if(lit(True))


def empty_or_whitespace_rate(text: IntoExpr) -> Expr:
    """The fraction of generations that are empty or contain only whitespace — a blank-output rate.

    Matches ``^\\s*$``, which requires the whole string to be zero or more whitespace characters
    from start to end, so both an empty string and one made only of spaces, tabs, or newlines
    count. This is the hard failure of output cleanliness: the model returned nothing usable at all,
    and it is distinct from an empty-generation check on a separate column.

    Args:
        text: The generated-text column (name or expression).

    Returns:
        The empty-or-whitespace-only rate over the corpus, in ``[0, 1]``.

    Examples:
        .. doctest::

            >>> import batcher as bt
            >>> ds = bt.from_pydict({"o": ["real answer", "", "   "]})
            >>> round(ds.agg(e=bt.empty_or_whitespace_rate("o")).to_pydict()["e"][0], 4)
            0.6667
    """
    return count_if(_as_column(text).str.regexp_matches(r"^\s*$")) / count_if(lit(True))
